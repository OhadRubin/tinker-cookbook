"""
Progress tracking for RL trajectory collection.

Writes state to a JSON file that can be watched by a separate display process.
Run `uv run python -m tinker_cookbook.utils.trajectory_progress` in another terminal to watch.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

PROGRESS_FILE = Path("/tmp/trajectory_progress.json")

trajectory_group_id: ContextVar[int | None] = ContextVar("traj_group_id", default=None)
trajectory_index: ContextVar[int | None] = ContextVar("traj_index", default=None)


# TODO: move this into a parameter of TrajectoryProgressTracker, for now, we will hardcode this so this will work with the run i currently have going
NUM_TRAINING_GROUPS = 32

class TrajectoryStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SAMPLED = "sampled"  # done sampling, awaiting scoring
    COMPLETED = "completed"


@dataclass
class TrajectoryState:
    group_id: int
    trajectory_id: int
    status: TrajectoryStatus = TrajectoryStatus.PENDING
    tokens_generated: int = 0
    max_tokens: int = 65536
    reward: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    last_touched_time: float | None = None
    num_llm_calls: int = 0
    training_status: str = "pending"  # "pending" | "enqueued" | "done"
    enqueued_time: float | None = None  # when training_status became "enqueued"
    fwd_bwd_done_time: float | None = None  # when training_status became "done"

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "trajectory_id": self.trajectory_id,
            "status": self.status.value,
            "tokens_generated": self.tokens_generated,
            "max_tokens": self.max_tokens,
            "reward": self.reward,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "last_touched_time": self.last_touched_time,
            "num_llm_calls": self.num_llm_calls,
            "training_status": self.training_status,
            "enqueued_time": self.enqueued_time,
            "fwd_bwd_done_time": self.fwd_bwd_done_time,
        }


@dataclass
class GroupState:
    group_id: int
    trajectories: dict[int, TrajectoryState] = field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "trajectories": {k: v.to_dict() for k, v in self.trajectories.items()},
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class TrajectoryProgressTracker:
    """
    Thread-safe progress tracker for RL trajectory collection.
    Writes state to JSON file for external display.
    """

    _instance: TrajectoryProgressTracker | None = None
    _lock = threading.Lock()

    def __init__(self):
        self._groups: dict[int, GroupState] = {}
        self._update_lock = threading.Lock()
        self._enabled: bool = False
        self._max_tokens: int = 65536
        self._group_size: int = 8
        self._call_counters: dict[int, int] = {}
        self._batch_start_time: float | None = None
        self._num_workers: int | None = None
        self._next_group_id: int = 0

    @classmethod
    def get_instance(cls) -> TrajectoryProgressTracker:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def configure(
        self,
        max_tokens: int,
        group_size: int,
        enabled: bool = True,
        refresh_rate: float = 4.0,
    ) -> None:
        self._max_tokens = max_tokens
        self._group_size = group_size
        self._enabled = enabled

    @contextmanager
    def start_continuous(self, num_workers: int) -> Iterator[None]:
        """Context manager for continuous streaming mode.
        num_workers indicates expected capacity (max groups in flight).
        Groups are created lazily on first track_llm_call and removed via remove_group."""
        if not self._enabled:
            yield
            return

        self._num_workers = num_workers
        self._groups.clear()
        self._call_counters.clear()
        self._next_group_id = 0
        self._batch_start_time = time.time()
        self._write_state()
        try:
            yield
        finally:
            self._num_workers = None
            self._write_state()

    @contextmanager
    def track_batch(self, num_groups: int) -> Iterator[None]:
        """Context manager for tracking a batch of groups."""
        if not self._enabled:
            yield
            return

        self._groups.clear()
        self._call_counters.clear()
        self._batch_start_time = time.time()
        self._write_state()
        try:
            yield
        finally:
            self._write_state()

    def _write_state(self) -> None:
        """Write current state to JSON file."""
        state = {
            "timestamp": time.time(),
            "batch_start_time": self._batch_start_time,
            "max_tokens": self._max_tokens,
            "group_size": self._group_size,
            "num_workers": self._num_workers,
            "groups": {k: v.to_dict() for k, v in self._groups.items()},
        }
        try:
            PROGRESS_FILE.write_text(json.dumps(state))
        except Exception:
            pass

    def _create_group(self, group_id: int, start_time: float | None) -> None:
        """Create a group with all trajectories. Must be called with _update_lock held."""
        self._groups[group_id] = GroupState(group_id=group_id, start_time=start_time)
        for t in range(self._group_size):
            self._groups[group_id].trajectories[t] = TrajectoryState(
                group_id=group_id,
                trajectory_id=t,
                max_tokens=self._max_tokens,
            )

    @contextmanager
    def _maybe_create_group(self, group_id: int) -> Iterator[GroupState]:
        """Acquire lock, create group if it doesn't exist, then yield it."""
        with self._update_lock:
            if group_id not in self._groups:
                self._create_group(group_id, start_time=time.time())
                self._call_counters[group_id] = 0
            yield self._groups[group_id]

    def allocate_group_id(self) -> int:
        """Allocate and return a new group ID. Group is created lazily on first track_llm_call."""
        with self._update_lock:
            group_id = self._next_group_id
            self._next_group_id += 1
        return group_id

    def track_llm_call(self, group_id: int, tokens: int, traj_idx: int) -> None:
        with self._maybe_create_group(group_id) as group:
            assert traj_idx in group.trajectories, f"traj_idx={traj_idx} not in group_id={group_id}"

            traj = group.trajectories[traj_idx]
            now = time.time()
            if traj.status == TrajectoryStatus.PENDING:
                traj.status = TrajectoryStatus.IN_PROGRESS
                traj.start_time = now
            traj.tokens_generated = tokens
            traj.last_touched_time = now
            traj.num_llm_calls += 1

            self._call_counters[group_id] += 1

        self._write_state()

    def mark_trajectory_sampled(self, group_id: int, trajectory_id: int, total_tokens: int) -> None:
        """Called when a trajectory finishes sampling but hasn't been scored yet."""
        with self._maybe_create_group(group_id) as group:
            assert trajectory_id in group.trajectories, f"trajectory_id={trajectory_id} not in group_id={group_id}"
            traj = group.trajectories[trajectory_id]
            traj.status = TrajectoryStatus.SAMPLED
            traj.tokens_generated = total_tokens
            traj.end_time = time.time()
        self._write_state()

    def complete_group(self, group_id: int, rewards: list[float]) -> None:
        with self._maybe_create_group(group_id) as group:
            group.end_time = time.time()

            for i, reward in enumerate(rewards):
                assert i in group.trajectories, f"trajectory_id={i} not in group_id={group_id}"
                traj = group.trajectories[i]
                traj.status = TrajectoryStatus.COMPLETED
                traj.reward = reward
                traj.end_time = time.time()

        self._write_state()

    def remove_group(self, group_id: int) -> None:
        """Remove a completed group from tracking."""
        with self._maybe_create_group(group_id) as group:
            del self._groups[group_id]
        self._write_state()

    def mark_trajectory_training_enqueued(self, group_id: int, trajectory_id: int) -> None:
        """Called when forward_backward_async is invoked for a trajectory."""
        with self._maybe_create_group(group_id) as group:
            assert trajectory_id in group.trajectories, f"trajectory_id={trajectory_id} not in group_id={group_id}"
            traj = group.trajectories[trajectory_id]
            traj.training_status = "enqueued"
            traj.enqueued_time = time.time()
        self._write_state()

    def mark_trajectory_fwd_bwd_done(self, group_id: int, trajectory_id: int) -> None:
        """Called when forward_backward result is consumed for a trajectory."""
        with self._maybe_create_group(group_id) as group:
            assert trajectory_id in group.trajectories, f"trajectory_id={trajectory_id} not in group_id={group_id}"
            traj = group.trajectories[trajectory_id]
            traj.training_status = "done"
            traj.fwd_bwd_done_time = time.time()
        self._write_state()


def set_trajectory_context(group_id: int, traj_id: int) -> None:
    trajectory_group_id.set(group_id)
    trajectory_index.set(traj_id)


def get_trajectory_context() -> tuple[int | None, int | None]:
    return trajectory_group_id.get(), trajectory_index.get()


def clear_trajectory_context() -> None:
    trajectory_group_id.set(None)
    trajectory_index.set(None)


# ============================================================================
# WATCHER - Run in separate terminal: uv run python -m tinker_cookbook.utils.trajectory_progress
# ============================================================================


@dataclass
class RollingStats:
    """Track rolling statistics over time for the watcher."""

    window_sec: float = 60.0
    _history: list[tuple[float, dict]] = field(default_factory=list)

    def update(self, stats: dict) -> None:
        """Add a new observation."""
        now = time.time()
        self._history.append((now, stats))
        # Prune old entries
        cutoff = now - self.window_sec
        self._history = [(t, s) for t, s in self._history if t > cutoff]

    def get_rate(self, key: str) -> float | None:
        """Compute rate of change for a cumulative metric (e.g., total_tokens)."""
        if len(self._history) < 2:
            return None
        t0, s0 = self._history[0]
        t1, s1 = self._history[-1]
        dt = t1 - t0
        if dt < 1.0:
            return None
        v0, v1 = s0.get(key, 0), s1.get(key, 0)
        return (v1 - v0) / dt

    def get_eta_for_target(self, current_key: str, target_key: str) -> float | None:
        """Estimate time until current reaches target based on rolling rate."""
        if len(self._history) < 2:
            return None
        latest = self._history[-1][1]
        current = latest.get(current_key, 0)
        target = latest.get(target_key, 0)
        if current >= target:
            return 0.0
        rate = self.get_rate(current_key)
        if rate is None or rate <= 0:
            return None
        return (target - current) / rate

    def get_trend(self, key: str) -> str:
        """Get trend indicator: ↑ ↓ or →"""
        if len(self._history) < 4:
            return "→"
        # Compare first half rate to second half rate
        mid = len(self._history) // 2
        first_half = self._history[:mid]
        second_half = self._history[mid:]

        def half_rate(half: list[tuple[float, dict]]) -> float | None:
            if len(half) < 2:
                return None
            t0, s0 = half[0]
            t1, s1 = half[-1]
            dt = t1 - t0
            if dt < 0.5:
                return None
            return (s1.get(key, 0) - s0.get(key, 0)) / dt

        r1, r2 = half_rate(first_half), half_rate(second_half)
        if r1 is None or r2 is None:
            return "→"
        if r2 > r1 * 1.1:
            return "↑"
        elif r2 < r1 * 0.9:
            return "↓"
        return "→"


@dataclass
class RollingDurationStats:
    """Track rolling window of duration measurements for latency metrics."""

    window_sec: float = 300.0  # 5 minute window
    _durations: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, duration)
    _seen_ids: set[str] = field(default_factory=set)  # track already-seen trajectory IDs

    def add(self, traj_id: str, duration: float) -> None:
        """Add a duration measurement if not already seen."""
        if traj_id in self._seen_ids:
            return
        self._seen_ids.add(traj_id)
        now = time.time()
        self._durations.append((now, duration))
        cutoff = now - self.window_sec
        self._durations = [(t, d) for t, d in self._durations if t > cutoff]

    def get_mean(self) -> float | None:
        """Get mean duration over the window."""
        if not self._durations:
            return None
        return sum(d for _, d in self._durations) / len(self._durations)

    def get_count(self) -> int:
        """Get number of samples in window."""
        return len(self._durations)


def watch():
    """Watch the progress file and display with rich."""
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    def compute_stats(state: dict) -> dict:
        """Compute summary statistics from current state."""
        groups = state.get("groups", {})
        batch_start = state.get("batch_start_time")
        group_size = state.get("group_size", 8)
        now = time.time()

        total_groups = len(groups)
        total_trajectories = 0
        completed_trajectories = 0
        training_enqueued = 0
        training_done = 0
        total_tokens = 0
        all_rewards: list[float] = []

        # Track completion times for rolling average
        group_completion_times: list[float] = []
        traj_completion_times: list[float] = []

        # New metrics
        enqueued_groups = 0  # groups with ≥1 enqueued trajectory (not all done)
        done_groups = 0  # groups where all trajectories have training_status="done"
        group_reward_variances: list[float] = []

        for gid, group in groups.items():
            trajectories = group.get("trajectories", {})
            total_trajectories += len(trajectories)

            if group.get("end_time"):
                group_completion_times.append(group["end_time"])

            # Track per-group training status and rewards
            group_enqueued_count = 0
            group_done_count = 0
            group_rewards: list[float] = []

            for tid, traj in trajectories.items():
                tokens = traj.get("tokens_generated", 0)
                total_tokens += tokens

                if traj.get("status") == "completed":
                    completed_trajectories += 1
                    if traj.get("end_time"):
                        traj_completion_times.append(traj["end_time"])
                    reward = traj.get("reward")
                    if reward is not None:
                        all_rewards.append(reward)
                        group_rewards.append(reward)

                ts = traj.get("training_status", "pending")
                if ts == "enqueued":
                    training_enqueued += 1
                    group_enqueued_count += 1
                elif ts == "done":
                    training_done += 1
                    group_done_count += 1

            # Classify groups
            num_trajs = len(trajectories) if trajectories else group_size
            if group_done_count == num_trajs and num_trajs > 0:
                done_groups += 1
            elif group_enqueued_count > 0:
                enqueued_groups += 1

            # Compute within-group reward variance
            if len(group_rewards) >= 2:
                mean_r = sum(group_rewards) / len(group_rewards)
                variance = sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)
                group_reward_variances.append(variance)

        completed_groups = len(group_completion_times)
        pending_groups = total_groups - completed_groups
        pending_trajectories = total_trajectories - completed_trajectories

        # Calculate elapsed time
        elapsed = (now - batch_start) if batch_start else 0

        # Overall average rates (since batch start, per second)
        overall_traj_rate = completed_trajectories / elapsed if elapsed > 0 else 0
        overall_group_rate = completed_groups / elapsed if elapsed > 0 else 0
        overall_token_rate = total_tokens / elapsed if elapsed > 0 else 0

        # Rolling average (last 60 seconds)
        rolling_window = 60.0
        cutoff = now - rolling_window

        recent_trajs = sum(1 for t in traj_completion_times if t > cutoff)
        recent_groups = sum(1 for t in group_completion_times if t > cutoff)

        rolling_traj_rate = recent_trajs / rolling_window if recent_trajs > 0 else overall_traj_rate
        rolling_group_rate = recent_groups / rolling_window if recent_groups > 0 else overall_group_rate

        # ETAs
        eta_batch_overall = pending_groups / overall_group_rate if overall_group_rate > 0 else float('inf')
        eta_batch_rolling = pending_groups / rolling_group_rate if rolling_group_rate > 0 else float('inf')
        eta_traj_overall = pending_trajectories / overall_traj_rate if overall_traj_rate > 0 else float('inf')
        eta_traj_rolling = pending_trajectories / rolling_traj_rate if rolling_traj_rate > 0 else float('inf')

        # Reward stats
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else None
        min_reward = min(all_rewards) if all_rewards else None
        max_reward = max(all_rewards) if all_rewards else None
        positive_count = sum(1 for r in all_rewards if r > 0)
        positive_pct = (positive_count / len(all_rewards) * 100) if all_rewards else None

        # Within-group reward variance (mean across groups)
        mean_within_group_variance = (
            sum(group_reward_variances) / len(group_reward_variances)
            if group_reward_variances else None
        )

        return {
            "elapsed": elapsed,
            "total_groups": total_groups,
            "completed_groups": completed_groups,
            "total_trajectories": total_trajectories,
            "completed_trajectories": completed_trajectories,
            "training_enqueued": training_enqueued,
            "training_done": training_done,
            "total_tokens": total_tokens,
            "overall_traj_rate": overall_traj_rate,
            "overall_group_rate": overall_group_rate,
            "overall_token_rate": overall_token_rate,
            "rolling_traj_rate": rolling_traj_rate,
            "rolling_group_rate": rolling_group_rate,
            "eta_batch_overall": eta_batch_overall,
            "eta_batch_rolling": eta_batch_rolling,
            "eta_traj_overall": eta_traj_overall,
            "eta_traj_rolling": eta_traj_rolling,
            "num_workers": state.get("num_workers"),
            "mean_reward": mean_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "positive_pct": positive_pct,
            "enqueued_groups": enqueued_groups,
            "done_groups": done_groups,
            "mean_within_group_variance": mean_within_group_variance,
        }

    def format_time(seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        if seconds == float('inf') or seconds < 0:
            return "--:--"
        if seconds >= 3600:
            return f"{int(seconds // 3600)}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def build_dashboard(
        stats: dict,
        rolling: RollingStats,
        rolling_enq_to_done: RollingDurationStats,
        rolling_pend_to_sampled: RollingDurationStats,
        rolling_e2e: RollingDurationStats,
    ) -> Panel:
        """Build the summary dashboard panel."""
        text = Text()

        # Row 1: Progress counts
        text.append("Progress: ", style="bold")
        if stats.get('num_workers'):
            text.append(f"Groups {stats['total_groups']}/{stats['num_workers']}", style="cyan")
        else:
            text.append(f"Groups {stats['total_groups']}", style="cyan")
        text.append(" │ ", style="dim")
        text.append(f"Sampled {stats['completed_trajectories']}/{stats['total_trajectories']}", style="green")
        text.append(" │ ", style="dim")
        text.append(f"Training R:{stats['training_enqueued']} D:{stats['training_done']}", style="yellow")
        text.append("\n")

        # Row 2: Throughput with trends (converted to /min)
        text.append("Throughput: ", style="bold")
        traj_trend = rolling.get_trend("completed_trajectories")
        tok_trend = rolling.get_trend("total_tokens")
        text.append(f"{stats['rolling_traj_rate'] * 60:.1f} traj/min {traj_trend}", style="magenta")
        text.append(" │ ", style="dim")
        text.append(f"{stats['rolling_group_rate'] * 60:.1f} grp/min", style="magenta")
        text.append(" │ ", style="dim")
        rolling_tok_rate = rolling.get_rate("total_tokens")
        if rolling_tok_rate is not None:
            text.append(f"{rolling_tok_rate / 1000:.1f}k tok/s {tok_trend}", style="magenta")
        else:
            text.append(f"{stats['overall_token_rate'] / 1000:.1f}k tok/s", style="magenta")
        text.append("\n")

        # Row 3: Latencies (rolling windows)
        text.append("Latencies: ", style="bold")
        enq_to_done_mean = rolling_enq_to_done.get_mean()
        pend_to_sampled_mean = rolling_pend_to_sampled.get_mean()
        e2e_mean = rolling_e2e.get_mean()
        if enq_to_done_mean is not None:
            text.append(f"Enq→Done μ={enq_to_done_mean:.1f}s", style="cyan")
        else:
            text.append("Enq→Done --", style="dim")
        text.append(" │ ", style="dim")
        if pend_to_sampled_mean is not None:
            text.append(f"Pend→Sampled μ={pend_to_sampled_mean:.1f}s", style="cyan")
        else:
            text.append("Pend→Sampled --", style="dim")
        text.append(" │ ", style="dim")
        if e2e_mean is not None:
            text.append(f"E2E μ={e2e_mean:.1f}s (*)", style="bright_cyan bold")
        else:
            text.append("E2E --", style="dim")
        text.append("\n")

        # Row 4: State counts and variance
        text.append("State: ", style="bold")
        text.append(f"{stats['enqueued_groups']} enqueued grps", style="yellow")
        text.append(" │ ", style="dim")
        text.append(f"{stats['done_groups']} done grps", style="green")
        text.append(" │ ", style="dim")
        if stats['mean_within_group_variance'] is not None:
            text.append(f"Grp Var={stats['mean_within_group_variance']:.2f}", style="cyan")
        else:
            text.append("Grp Var=--", style="dim")
        text.append("\n")

        # Row 5: ETAs (rolling-based) + Step ETA using E2E
        text.append("ETA: ", style="bold")
        rolling_traj_eta = rolling.get_eta_for_target("completed_trajectories", "total_trajectories")
        if rolling_traj_eta is not None:
            text.append(f"Trajs {format_time(rolling_traj_eta)}", style="blue")
        else:
            text.append(f"Trajs {format_time(stats['eta_traj_rolling'])}", style="blue")
        text.append(" │ ", style="dim")
        # Step ETA: remaining trajs in current step × E2E mean
        remaining_trajs = stats['total_trajectories'] - stats['training_done']
        if e2e_mean is not None and remaining_trajs > 0:
            step_eta = remaining_trajs * e2e_mean
            text.append(f"Step {format_time(step_eta)}", style="blue")
        else:
            text.append("Step --:--", style="dim")
        text.append(" │ ", style="dim")
        text.append(f"Elapsed {format_time(stats['elapsed'])}", style="dim")
        text.append("\n")

        # Row 6: Reward stats
        text.append("Rewards: ", style="bold")
        if stats['mean_reward'] is not None:
            text.append(f"μ={stats['mean_reward']:+.2f}", style="green")
            text.append(" │ ", style="dim")
            text.append(f"min={stats['min_reward']:+.2f} max={stats['max_reward']:+.2f}", style="cyan")
            text.append(" │ ", style="dim")
            text.append(f"{stats['positive_pct']:.0f}% positive", style="yellow")
        else:
            text.append("--", style="dim")

        return Panel(text, title="Dashboard", border_style="bright_black", padding=(0, 1))

    def build_table(state: dict) -> Table:
        import numpy as np

        groups = state.get("groups", {})
        group_size = state.get("group_size", 8)
        num_workers = state.get("num_workers")
        now = time.time()

        # Limit to num_workers + 10 groups, keeping lowest group IDs (actively being processed)
        max_display = num_workers + 10 if num_workers else None
        if max_display and len(groups) > max_display:
            sorted_gids = sorted(groups.keys(), key=int)[:max_display]
            groups = {gid: groups[gid] for gid in sorted_gids}

        table = Table(title="Trajectory Collection", expand=False, box=None)
        table.add_column("Grp", style="cyan", width=3, no_wrap=True)

        # Add 4 columns per trajectory: ctx, rwd, age, status + delimiter
        for tid in range(group_size):
            table.add_column(f"ctx", width=3, justify="right")
            table.add_column(f"rwd", width=4, justify="right")
            table.add_column(f"age", width=3, justify="right")
            table.add_column(f"st", width=1, justify="center")
            if tid < group_size - 1:
                table.add_column("", width=1)  # delimiter column

        table.add_column("Done", width=5, justify="right")
        table.add_column("μRwd", width=5, justify="right")
        table.add_column("Time", width=5, justify="right")
        table.add_column("Min", width=3, justify="right")
        table.add_column("Max", width=3, justify="right")

        # First pass: compute mean rewards for all groups
        group_mean_rewards: dict[str, float | None] = {}
        group_data: dict[str, dict] = {}
        for gid in groups.keys():
            group = groups[gid]
            trajectories = group.get("trajectories", {})
            all_rewards = []
            positive_rewards = []
            for tid in range(group_size):
                traj = trajectories.get(str(tid), trajectories.get(tid, {}))
                r = traj.get("reward")
                if r is not None:
                    all_rewards.append(r)
                    if r > 0:
                        positive_rewards.append(r)
            positive_rewards.sort(reverse=True)
            top3_threshold = positive_rewards[2] if len(positive_rewards) >= 3 else (positive_rewards[-1] if positive_rewards else float('inf'))
            mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else None
            group_mean_rewards[gid] = mean_reward
            group_data[gid] = {"top3_threshold": top3_threshold}

        # Compute histogram bins for mean reward coloring (4 bins: red, yellow, bright_yellow, bright_green)
        valid_means = [m for m in group_mean_rewards.values() if m is not None]
        if len(valid_means) >= 2:
            _, bin_edges = np.histogram(valid_means, bins=4)
        else:
            bin_edges = None

        def get_mean_reward_style(mean_val: float | None) -> str:
            if mean_val is None or bin_edges is None:
                return "dim"
            # bin_edges has 5 edges for 4 bins: [e0, e1, e2, e3, e4]
            # bin 0: [e0, e1) -> red (worst)
            # bin 1: [e1, e2) -> yellow
            # bin 2: [e2, e3) -> bright_yellow
            # bin 3: [e3, e4] -> bright_green (best)
            styles = ["red", "yellow", "bright_yellow", "bright_green"]
            for i in range(3):
                if mean_val < bin_edges[i + 1]:
                    return styles[i]
            return styles[3]

        for gid in sorted(groups.keys(), key=int):
            group = groups[gid]
            trajectories = group.get("trajectories", {})
            top3_threshold = group_data[gid]["top3_threshold"]
            mean_reward = group_mean_rewards[gid]

            # Check if any trajectory is enqueued
            has_enqueued = any(
                trajectories.get(str(t), trajectories.get(t, {})).get("training_status") == "enqueued"
                for t in range(group_size)
            )

            row: list[str | Text] = [f"G{int(gid):02d}"]
            completed = 0

            for tid in range(group_size):
                traj = trajectories.get(str(tid), trajectories.get(tid, {}))
                status = traj.get("status", "pending")
                tokens = traj.get("tokens_generated", 0)
                reward = traj.get("reward")
                training_status = traj.get("training_status", "pending")
                end_time = traj.get("end_time")
                last_touched_time = traj.get("last_touched_time")

                # Context length in k
                k = tokens // 1000

                # Reward and context styling based on status
                if status == "completed":
                    completed += 1
                    ctx_text = Text(f"{k:2d}k" if k > 0 else "  ·", style="white bold")
                    if reward is not None:
                        is_top3 = reward > 0 and reward >= top3_threshold
                        rwd_style = "bright_green" if is_top3 else "yellow"
                        rwd_text = Text(f"{reward:+.1f}" if reward != 0 else " 0.0", style=rwd_style)
                    else:
                        rwd_text = Text("   ?", style="yellow")
                elif status == "sampled":
                    completed += 1  # count as completed for Done column
                    ctx_text = Text(f"{k:2d}k" if k > 0 else "  ·", style="white bold")
                    rwd_text = Text("   ·", style="dim")
                elif status == "in_progress":
                    ctx_text = Text(f"{k:2d}k" if k > 0 else "  ·", style="white bold")
                    rwd_text = Text("   ·", style="dim")
                else:
                    ctx_text = Text("  ·", style="dim")
                    rwd_text = Text("   ·", style="dim")

                # Time since last activity (age in seconds)
                # Hide individual ages when enqueued (all same, shown in Min/Max instead)
                if has_enqueued or training_status == "done" or status == "sampled":
                    age_text = Text("  ·", style="dim")
                elif end_time:
                    age = int(now - end_time)
                    age_text = Text(f"{age:3d}" if age < 1000 else "999", style="dim")
                elif last_touched_time:
                    age = int(now - last_touched_time)
                    age_text = Text(f"{age:3d}" if age < 1000 else "999", style="dim")
                else:
                    age_text = Text("  ·", style="dim")

                # Training status
                if training_status == "done":
                    st_text = Text("D", style="bright_cyan bold")
                elif training_status == "enqueued":
                    st_text = Text("R", style="red bold")
                elif status == "sampled":
                    st_text = Text("W", style="bright_magenta bold")
                else:
                    st_text = Text("S", style="green")

                row.extend([ctx_text, rwd_text, age_text, st_text])
                if tid < group_size - 1:
                    row.append(Text("│", style="dim"))

            # Global columns
            total = len(trajectories) if trajectories else group_size
            grp_start = group.get("start_time")
            grp_end = group.get("end_time")

            if grp_start:
                elapsed = (grp_end or now) - grp_start
                time_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
            else:
                time_str = "--:--"

            row.append(f"{completed}/{total}")
            if mean_reward is not None:
                mean_style = get_mean_reward_style(mean_reward)
                row.append(Text(f"{mean_reward:+.2f}", style=mean_style))
            else:
                row.append(Text("--", style="dim"))
            row.append(time_str)

            # Min/Max age across all trajectories (including W/sampled)
            ages = []
            for t in range(group_size):
                traj = trajectories.get(str(t), trajectories.get(t, {}))
                traj_end = traj.get("end_time")
                traj_last = traj.get("last_touched_time")
                ref_time = traj_end or traj_last
                if ref_time:
                    ages.append(int(now - ref_time))

            if ages:
                row.append(Text(f"{min(ages):3d}", style="white"))
                row.append(Text(f"{max(ages):3d}", style="white"))
            else:
                row.append(Text("  ·", style="dim"))
                row.append(Text("  ·", style="dim"))

            table.add_row(*row)

        return table

    def feed_rolling_durations(
        state: dict,
        rolling_enq_to_done: RollingDurationStats,
        rolling_pend_to_sampled: RollingDurationStats,
        rolling_e2e: RollingDurationStats,
    ) -> None:
        """Feed rolling duration trackers from current state."""
        groups = state.get("groups", {})
        for gid, group in groups.items():
            trajectories = group.get("trajectories", {})
            for tid, traj in trajectories.items():
                traj_id = f"{gid}_{tid}"

                # Enqueued → Done (training latency)
                enq_time = traj.get("enqueued_time")
                done_time = traj.get("fwd_bwd_done_time")
                if enq_time is not None and done_time is not None:
                    rolling_enq_to_done.add(traj_id, done_time - enq_time)

                # Pending → Sampled (sampling latency)
                start_time = traj.get("start_time")
                end_time = traj.get("end_time")
                if start_time is not None and end_time is not None:
                    rolling_pend_to_sampled.add(traj_id, end_time - start_time)

                # End-to-end: start_time → fwd_bwd_done_time
                if start_time is not None and done_time is not None:
                    rolling_e2e.add(traj_id, done_time - start_time)

    def build_display(
        state: dict,
        rolling: RollingStats,
        rolling_enq_to_done: RollingDurationStats,
        rolling_pend_to_sampled: RollingDurationStats,
        rolling_e2e: RollingDurationStats,
    ) -> Group:
        """Build complete display with dashboard and table."""
        stats = compute_stats(state)
        rolling.update(stats)
        feed_rolling_durations(state, rolling_enq_to_done, rolling_pend_to_sampled, rolling_e2e)
        dashboard = build_dashboard(stats, rolling, rolling_enq_to_done, rolling_pend_to_sampled, rolling_e2e)
        table = build_table(state)
        return Group(dashboard, table)

    console.print("[yellow]Watching /tmp/trajectory_progress.json...[/yellow]")
    console.print("[dim]Start training in another terminal[/dim]\n")

    last_mtime = 0.0
    state = {}
    rolling = RollingStats(window_sec=60.0)
    rolling_enq_to_done = RollingDurationStats(window_sec=300.0)
    rolling_pend_to_sampled = RollingDurationStats(window_sec=300.0)
    rolling_e2e = RollingDurationStats(window_sec=300.0)

    with Live(Table(), console=console, refresh_per_second=4) as live:
        while True:
            try:
                if PROGRESS_FILE.exists():
                    mtime = PROGRESS_FILE.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        state = json.loads(PROGRESS_FILE.read_text())
                    live.update(build_display(
                        state, rolling, rolling_enq_to_done, rolling_pend_to_sampled, rolling_e2e
                    ))
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
            except Exception:
                time.sleep(0.5)


if __name__ == "__main__":
    watch()
