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
            group.trajectories[trajectory_id].training_status = "enqueued"
        self._write_state()

    def mark_trajectory_fwd_bwd_done(self, group_id: int, trajectory_id: int) -> None:
        """Called when forward_backward result is consumed for a trajectory."""
        with self._maybe_create_group(group_id) as group:
            assert trajectory_id in group.trajectories, f"trajectory_id={trajectory_id} not in group_id={group_id}"
            group.trajectories[trajectory_id].training_status = "done"
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
        now = time.time()

        total_groups = len(groups)
        total_trajectories = 0
        completed_trajectories = 0
        training_enqueued = 0
        training_done = 0
        total_tokens = 0

        # Track completion times for rolling average
        group_completion_times: list[float] = []
        traj_completion_times: list[float] = []

        for gid, group in groups.items():
            trajectories = group.get("trajectories", {})
            total_trajectories += len(trajectories)

            if group.get("end_time"):
                group_completion_times.append(group["end_time"])

            for tid, traj in trajectories.items():
                tokens = traj.get("tokens_generated", 0)
                total_tokens += tokens

                if traj.get("status") == "completed":
                    completed_trajectories += 1
                    if traj.get("end_time"):
                        traj_completion_times.append(traj["end_time"])

                ts = traj.get("training_status", "pending")
                if ts == "enqueued":
                    training_enqueued += 1
                elif ts == "done":
                    training_done += 1

        completed_groups = len(group_completion_times)
        pending_groups = total_groups - completed_groups
        pending_trajectories = total_trajectories - completed_trajectories

        # Calculate elapsed time
        elapsed = (now - batch_start) if batch_start else 0

        # Overall average rates (since batch start)
        overall_traj_rate = completed_trajectories / elapsed if elapsed > 0 else 0
        overall_group_rate = completed_groups / elapsed if elapsed > 0 else 0
        overall_token_rate = total_tokens / elapsed if elapsed > 0 else 0

        # Rolling average (last 60 seconds)
        rolling_window = 60.0
        cutoff = now - rolling_window

        recent_trajs = sum(1 for t in traj_completion_times if t > cutoff)
        recent_groups = sum(1 for t in group_completion_times if t > cutoff)

        # Calculate actual window duration (min of rolling_window or time since first completion in window)
        recent_traj_times = [t for t in traj_completion_times if t > cutoff]
        recent_group_times = [t for t in group_completion_times if t > cutoff]

        rolling_traj_rate = recent_trajs / rolling_window if recent_trajs > 0 else overall_traj_rate
        rolling_group_rate = recent_groups / rolling_window if recent_groups > 0 else overall_group_rate

        # ETAs
        # Per-batch ETA: time until all groups finish sampling
        eta_batch_overall = pending_groups / overall_group_rate if overall_group_rate > 0 else float('inf')
        eta_batch_rolling = pending_groups / rolling_group_rate if rolling_group_rate > 0 else float('inf')

        # Per-trajectory ETA: time until all trajectories sampled
        eta_traj_overall = pending_trajectories / overall_traj_rate if overall_traj_rate > 0 else float('inf')
        eta_traj_rolling = pending_trajectories / rolling_traj_rate if rolling_traj_rate > 0 else float('inf')

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
        }

    def format_time(seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        if seconds == float('inf') or seconds < 0:
            return "--:--"
        if seconds >= 3600:
            return f"{int(seconds // 3600)}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def build_dashboard(stats: dict) -> Panel:
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

        # Row 2: Throughput
        text.append("Throughput: ", style="bold")
        text.append(f"{stats['rolling_traj_rate']:.2f} traj/s", style="magenta")
        text.append(" │ ", style="dim")
        text.append(f"{stats['rolling_group_rate']:.2f} grp/s", style="magenta")
        text.append(" │ ", style="dim")
        text.append(f"{stats['overall_token_rate'] / 1000:.1f}k tok/s", style="magenta")
        text.append("\n")

        # Row 3: ETAs
        text.append("ETA (rolling/overall): ", style="bold")
        text.append(f"Batch {format_time(stats['eta_batch_rolling'])}/{format_time(stats['eta_batch_overall'])}", style="blue")
        text.append(" │ ", style="dim")
        text.append(f"Trajs {format_time(stats['eta_traj_rolling'])}/{format_time(stats['eta_traj_overall'])}", style="blue")
        text.append(" │ ", style="dim")
        text.append(f"Elapsed {format_time(stats['elapsed'])}", style="dim")

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
                    rwd_text = Text("   ?", style="bright_magenta bold")
                elif status == "in_progress":
                    ctx_text = Text(f"{k:2d}k" if k > 0 else "  ·", style="white bold")
                    rwd_text = Text("   ?", style="bright_cyan bold")
                else:
                    ctx_text = Text("  ·", style="dim")
                    rwd_text = Text("   ·", style="dim")

                # Time since last activity (age in seconds)
                if training_status == "done":
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
                    st_text = Text("·", style="dim")

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

            table.add_row(*row)

        return table

    def build_display(state: dict) -> Group:
        """Build complete display with dashboard and table."""
        stats = compute_stats(state)
        dashboard = build_dashboard(stats)
        table = build_table(state)
        return Group(dashboard, table)

    console.print("[yellow]Watching /tmp/trajectory_progress.json...[/yellow]")
    console.print("[dim]Start training in another terminal[/dim]\n")

    last_mtime = 0.0
    state = {}

    with Live(Table(), console=console, refresh_per_second=4) as live:
        while True:
            try:
                if PROGRESS_FILE.exists():
                    mtime = PROGRESS_FILE.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        state = json.loads(PROGRESS_FILE.read_text())
                    live.update(build_display(state))
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
            except Exception:
                time.sleep(0.5)


if __name__ == "__main__":
    watch()
