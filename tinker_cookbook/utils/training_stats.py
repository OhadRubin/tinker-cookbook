"""
Training pipeline statistics for diagnosing sampling vs training throughput.

Tracks:
- Sampling: groups sampled, in-flight sampling, latencies
- Training: fwd_bwd calls, in-flight training, latencies
- Queue depth: backlog indicator (groups waiting for training)
- Rate comparison: sampling vs training throughput

Stats are emitted to Loki for Grafana dashboards.

Usage:
    from tinker_cookbook.utils.training_stats import training_stats

    # When sampling starts/ends for a group
    async with training_stats.track_sampling():
        await do_sampling()

    # When fwd_bwd starts/ends
    async with training_stats.track_training():
        await training_client.forward_backward_async(...)

    # Or manual tracking:
    training_stats.sampling_started()
    training_stats.sampling_completed(latency_ms)
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from evalbox.call_stats import CallStatsTracker, RollingWindow
from observability import log

DEFAULT_LATENCY_WINDOW_SEC = 30 * 60  # 30 minutes


class TrainingPipelineStats:
    """Tracks training pipeline statistics for Grafana dashboards.

    Uses two CallStatsTracker instances for sampling and training,
    plus additional tracking for queue depth and rate comparison.
    """

    def __init__(
        self,
        log_interval: float = 5.0,
        latency_window_sec: float = DEFAULT_LATENCY_WINDOW_SEC,
    ):
        self._log_interval = log_interval
        self._start_time: Optional[float] = None
        self._last_log_time: Optional[float] = None

        # Sampling stats (groups being sampled)
        self._sampling = CallStatsTracker(
            name="sampling",
            log_interval=float("inf"),  # disable auto-logging, we log combined stats
            latency_window_sec=latency_window_sec,
        )

        # Training stats (fwd_bwd calls)
        self._training = CallStatsTracker(
            name="training",
            log_interval=float("inf"),
            latency_window_sec=latency_window_sec,
        )

        # Queue depth tracking
        self._backlog: int = 0  # groups sampled but not yet training-done
        self._max_backlog: int = 0
        self._backlog_window = RollingWindow(window_sec=600)  # 10 min window

        # E2E latency (group start to training done)
        self._e2e_latency_window = RollingWindow(window_sec=latency_window_sec)
        self._e2e_total_latency_ms: float = 0.0
        self._e2e_count: int = 0
        self._e2e_min_ms: Optional[float] = None
        self._e2e_max_ms: Optional[float] = None

        # Optim step counter
        self._total_optim_steps: int = 0

        # Locks
        self._lock: Optional[asyncio.Lock] = None
        self._sync_lock = threading.Lock()

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _maybe_init(self, now: float) -> None:
        if self._start_time is None:
            self._start_time = now
            self._last_log_time = now

    # =========================================================================
    # Sync API (for use from trajectory_progress.py)
    # =========================================================================

    def sampling_started_sync(self) -> None:
        """Sync: record that a group started sampling."""
        now = time.time()
        self._maybe_init(now)
        with self._sync_lock:
            self._sampling._call_started_sync(now)

    def sampling_completed_sync(self, latency_ms: float) -> None:
        """Sync: record that a group finished sampling."""
        now = time.time()
        self._maybe_init(now)
        with self._sync_lock:
            self._sampling._call_completed_sync(now, latency_ms)
            # Update backlog
            self._backlog += 1
            if self._backlog > self._max_backlog:
                self._max_backlog = self._backlog
            self._backlog_window.add(float(self._backlog), now)
        self._maybe_emit_stats_sync()

    def training_started_sync(self) -> None:
        """Sync: record that training (fwd_bwd) started for a group."""
        now = time.time()
        self._maybe_init(now)
        with self._sync_lock:
            self._training._call_started_sync(now)

    def training_completed_sync(self, latency_ms: float, num_groups: int = 1) -> None:
        """Sync: record that training (fwd_bwd) completed for groups."""
        now = time.time()
        self._maybe_init(now)
        with self._sync_lock:
            self._training._call_completed_sync(now, latency_ms)
            # Update backlog
            self._backlog = max(0, self._backlog - num_groups)
            self._backlog_window.add(float(self._backlog), now)
        self._maybe_emit_stats_sync()

    def record_e2e_latency_sync(self, e2e_latency_ms: float) -> None:
        """Sync: record end-to-end latency (group start to training done)."""
        now = time.time()
        self._e2e_latency_window.add(e2e_latency_ms, now)
        with self._sync_lock:
            self._e2e_total_latency_ms += e2e_latency_ms
            self._e2e_count += 1
            if self._e2e_min_ms is None or e2e_latency_ms < self._e2e_min_ms:
                self._e2e_min_ms = e2e_latency_ms
            if self._e2e_max_ms is None or e2e_latency_ms > self._e2e_max_ms:
                self._e2e_max_ms = e2e_latency_ms

    def record_optim_step_sync(self) -> None:
        """Sync: record an optim_step completion."""
        with self._sync_lock:
            self._total_optim_steps += 1
        self._maybe_emit_stats_sync()

    def _maybe_emit_stats_sync(self) -> None:
        """Sync: emit stats if log interval has elapsed."""
        now = time.time()
        with self._sync_lock:
            if self._last_log_time is None:
                return
            if now - self._last_log_time < self._log_interval:
                return
            self._last_log_time = now
            self._emit_stats(now)

    # =========================================================================
    # Async API (for direct use from async code)
    # =========================================================================

    @asynccontextmanager
    async def track_sampling(self):
        """Context manager for tracking sampling duration."""
        await self._sampling._call_started()
        t0 = time.time()
        self._maybe_init(t0)
        try:
            yield
        finally:
            latency_ms = (time.time() - t0) * 1000
            await self._sampling._call_completed(latency_ms)
            async with self._get_lock():
                self._backlog += 1
                if self._backlog > self._max_backlog:
                    self._max_backlog = self._backlog
                self._backlog_window.add(float(self._backlog))
            await self._maybe_emit_stats()

    @asynccontextmanager
    async def track_training(self, num_groups: int = 1):
        """Context manager for tracking training (fwd_bwd) duration."""
        await self._training._call_started()
        t0 = time.time()
        self._maybe_init(t0)
        try:
            yield
        finally:
            latency_ms = (time.time() - t0) * 1000
            await self._training._call_completed(latency_ms)
            async with self._get_lock():
                self._backlog = max(0, self._backlog - num_groups)
                self._backlog_window.add(float(self._backlog))
            await self._maybe_emit_stats()

    async def record_e2e_latency(self, e2e_latency_ms: float) -> None:
        """Record end-to-end latency (group start to training done)."""
        now = time.time()
        self._e2e_latency_window.add(e2e_latency_ms, now)
        async with self._get_lock():
            self._e2e_total_latency_ms += e2e_latency_ms
            self._e2e_count += 1
            if self._e2e_min_ms is None or e2e_latency_ms < self._e2e_min_ms:
                self._e2e_min_ms = e2e_latency_ms
            if self._e2e_max_ms is None or e2e_latency_ms > self._e2e_max_ms:
                self._e2e_max_ms = e2e_latency_ms

    async def record_optim_step(self) -> None:
        """Record an optim_step completion."""
        async with self._get_lock():
            self._total_optim_steps += 1
        await self._maybe_emit_stats()

    async def _maybe_emit_stats(self) -> None:
        """Emit stats if log interval has elapsed."""
        now = time.time()
        async with self._get_lock():
            if self._last_log_time is None:
                return
            if now - self._last_log_time < self._log_interval:
                return
            self._last_log_time = now
            self._emit_stats(now)

    def _emit_stats(self, now: float) -> None:
        """Emit combined stats to Loki."""
        elapsed = now - self._start_time if self._start_time else 0

        sampling_stats = self._sampling.get_stats()
        training_stats = self._training.get_stats()

        # Backlog windowed stats
        backlog_1m = self._backlog_window.stats_for_window(60)
        backlog_5m = self._backlog_window.stats_for_window(300)

        # E2E latency stats
        e2e_window = self._e2e_latency_window.stats()
        e2e_avg = self._e2e_total_latency_ms / self._e2e_count if self._e2e_count > 0 else 0

        # Rate delta: positive = training keeping up, negative = falling behind
        rate_delta_1m = training_stats["cps_1m"] - sampling_stats["cps_1m"]
        rate_delta_5m = training_stats["cps_5m"] - sampling_stats["cps_5m"]

        log.info(
            "training stats",
            # Sampling metrics
            sampling_total=sampling_stats["total_calls"],
            sampling_in_flight=sampling_stats["in_flight"],
            sampling_max_in_flight=sampling_stats["max_in_flight"],
            sampling_max_in_flight_1m=sampling_stats["max_in_flight_1m"],
            sampling_avg_in_flight_1m=round(sampling_stats["avg_in_flight_1m"], 2) if sampling_stats["avg_in_flight_1m"] else None,
            sampling_cps_1m=round(sampling_stats["cps_1m"], 3),
            sampling_cps_5m=round(sampling_stats["cps_5m"], 3),
            sampling_latency_avg_ms=round(sampling_stats["avg_latency_ms"], 1),
            sampling_latency_min_ms=round(sampling_stats["min_latency_ms"], 1),
            sampling_latency_max_ms=round(sampling_stats["max_latency_ms"], 1),
            sampling_latency_p50_ms=round(sampling_stats["latency_p50_ms"], 1) if sampling_stats["latency_p50_ms"] else None,
            sampling_latency_p95_ms=round(sampling_stats["latency_p95_ms"], 1) if sampling_stats["latency_p95_ms"] else None,
            sampling_latency_p99_ms=round(sampling_stats["latency_p99_ms"], 1) if sampling_stats["latency_p99_ms"] else None,
            # Training metrics
            training_total=training_stats["total_calls"],
            training_in_flight=training_stats["in_flight"],
            training_max_in_flight=training_stats["max_in_flight"],
            training_max_in_flight_1m=training_stats["max_in_flight_1m"],
            training_avg_in_flight_1m=round(training_stats["avg_in_flight_1m"], 2) if training_stats["avg_in_flight_1m"] else None,
            training_cps_1m=round(training_stats["cps_1m"], 3),
            training_cps_5m=round(training_stats["cps_5m"], 3),
            training_latency_avg_ms=round(training_stats["avg_latency_ms"], 1),
            training_latency_min_ms=round(training_stats["min_latency_ms"], 1),
            training_latency_max_ms=round(training_stats["max_latency_ms"], 1),
            training_latency_p50_ms=round(training_stats["latency_p50_ms"], 1) if training_stats["latency_p50_ms"] else None,
            training_latency_p95_ms=round(training_stats["latency_p95_ms"], 1) if training_stats["latency_p95_ms"] else None,
            training_latency_p99_ms=round(training_stats["latency_p99_ms"], 1) if training_stats["latency_p99_ms"] else None,
            # Backlog (queue depth)
            backlog=self._backlog,
            max_backlog=self._max_backlog,
            max_backlog_1m=backlog_1m["max"],
            avg_backlog_1m=round(backlog_1m["avg"], 2) if backlog_1m["avg"] else None,
            max_backlog_5m=backlog_5m["max"],
            avg_backlog_5m=round(backlog_5m["avg"], 2) if backlog_5m["avg"] else None,
            # Rate comparison (negative = training falling behind)
            rate_delta_1m=round(rate_delta_1m, 3),
            rate_delta_5m=round(rate_delta_5m, 3),
            # E2E latency
            e2e_latency_avg_ms=round(e2e_avg, 1),
            e2e_latency_min_ms=round(self._e2e_min_ms, 1) if self._e2e_min_ms else None,
            e2e_latency_max_ms=round(self._e2e_max_ms, 1) if self._e2e_max_ms else None,
            e2e_latency_p50_ms=round(e2e_window["p50"], 1) if e2e_window.get("p50") else None,
            e2e_latency_p95_ms=round(e2e_window["p95"], 1) if e2e_window.get("p95") else None,
            e2e_latency_p99_ms=round(e2e_window["p99"], 1) if e2e_window.get("p99") else None,
            # Optim steps
            total_optim_steps=self._total_optim_steps,
            # Meta
            elapsed_sec=round(elapsed, 1),
            component="training_stats",
        )

    async def force_emit_stats(self) -> None:
        """Force emit stats regardless of interval."""
        now = time.time()
        async with self._get_lock():
            self._last_log_time = now
            self._emit_stats(now)

    def get_stats(self) -> Dict[str, Any]:
        """Get current stats as a dict (for programmatic access)."""
        sampling_stats = self._sampling.get_stats()
        training_stats = self._training.get_stats()
        backlog_1m = self._backlog_window.stats_for_window(60)
        e2e_window = self._e2e_latency_window.stats()

        return {
            "sampling": sampling_stats,
            "training": training_stats,
            "backlog": self._backlog,
            "max_backlog": self._max_backlog,
            "max_backlog_1m": backlog_1m["max"],
            "avg_backlog_1m": backlog_1m["avg"],
            "rate_delta_1m": training_stats["cps_1m"] - sampling_stats["cps_1m"],
            "e2e_latency_p50_ms": e2e_window.get("p50"),
            "e2e_latency_p95_ms": e2e_window.get("p95"),
            "total_optim_steps": self._total_optim_steps,
        }


# Module-level singleton
training_stats = TrainingPipelineStats()
