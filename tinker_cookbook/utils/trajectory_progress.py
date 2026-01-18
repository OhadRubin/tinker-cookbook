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
    num_llm_calls: int = 0

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
            "num_llm_calls": self.num_llm_calls,
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
    def track_batch(self, num_groups: int) -> Iterator[None]:
        """Context manager for tracking a batch of groups."""
        if not self._enabled:
            yield
            return

        self._groups.clear()
        self._call_counters.clear()
        self._batch_start_time = time.time()

        for g in range(num_groups):
            self._groups[g] = GroupState(group_id=g)
            for t in range(self._group_size):
                self._groups[g].trajectories[t] = TrajectoryState(
                    group_id=g,
                    trajectory_id=t,
                    max_tokens=self._max_tokens,
                )

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
            "groups": {k: v.to_dict() for k, v in self._groups.items()},
        }
        try:
            PROGRESS_FILE.write_text(json.dumps(state))
        except Exception:
            pass

    def start_group(self, group_id: int) -> None:
        with self._update_lock:
            if group_id in self._groups:
                self._groups[group_id].start_time = time.time()
                self._call_counters[group_id] = 0
        self._write_state()

    def track_llm_call(self, group_id: int, tokens: int, traj_idx: int) -> None:
        with self._update_lock:
            if group_id not in self._groups:
                raise ValueError(f"Unknown group_id: {group_id}")

            group = self._groups[group_id]

            if traj_idx not in group.trajectories:
                raise ValueError(f"Unknown traj_idx: {traj_idx} for group_id: {group_id}")

            if group_id not in self._call_counters:
                self._call_counters[group_id] = 0

            traj = group.trajectories[traj_idx]
            if traj.status == TrajectoryStatus.PENDING:
                traj.status = TrajectoryStatus.IN_PROGRESS
                traj.start_time = time.time()
            traj.tokens_generated = tokens
            traj.num_llm_calls += 1

            self._call_counters[group_id] += 1

        self._write_state()

    def complete_trajectory(
        self, group_id: int, trajectory_id: int, reward: float, total_tokens: int
    ) -> None:
        with self._update_lock:
            if group_id in self._groups and trajectory_id in self._groups[group_id].trajectories:
                traj = self._groups[group_id].trajectories[trajectory_id]
                traj.status = TrajectoryStatus.COMPLETED
                traj.reward = reward
                traj.tokens_generated = total_tokens
                traj.end_time = time.time()
        self._write_state()

    def complete_group(self, group_id: int, rewards: list[float], token_counts: list[int]) -> None:
        with self._update_lock:
            if group_id not in self._groups:
                return

            group = self._groups[group_id]
            group.end_time = time.time()

            for i, (reward, tokens) in enumerate(zip(rewards, token_counts)):
                if i in group.trajectories:
                    traj = group.trajectories[i]
                    traj.status = TrajectoryStatus.COMPLETED
                    traj.reward = reward
                    traj.tokens_generated = tokens
                    traj.end_time = time.time()

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
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text

    console = Console()

    def build_table(state: dict) -> Table:
        table = Table(title="Trajectory Collection", expand=False, box=None)
        table.add_column("Group", style="cyan", width=6, no_wrap=True)
        table.add_column("Trajectories", min_width=60)
        table.add_column("Done", width=8, justify="right")
        table.add_column("Time", width=6, justify="right")

        groups = state.get("groups", {})
        max_tokens = state.get("max_tokens", 65536)

        for gid in sorted(groups.keys(), key=int):
            group = groups[gid]
            trajectories = group.get("trajectories", {})

            text = Text()
            completed = 0
            for tid in sorted(trajectories.keys(), key=int):
                traj = trajectories[tid]
                status = traj.get("status", "pending")
                tokens = traj.get("tokens_generated", 0)
                reward = traj.get("reward")

                if status == "completed":
                    completed += 1
                    if reward is not None:
                        r = f"{reward:+.2f}" if reward != 0 else "0.00"
                    else:
                        r = "done"
                    text.append(f"[{r}]", style="green bold")
                elif status == "in_progress":
                    k = tokens // 1000
                    text.append(f"[{k:3d}k]", style="blue")
                else:
                    text.append("[  · ]", style="dim")
                text.append(" ")

            total = len(trajectories)
            start_time = group.get("start_time")
            end_time = group.get("end_time")

            if start_time:
                elapsed = (end_time or time.time()) - start_time
                time_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
            else:
                time_str = "--:--"

            table.add_row(f"G{int(gid):02d}", text, f"{completed}/{total}", time_str)

        return table

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
                        live.update(build_table(state))
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
            except Exception:
                time.sleep(0.5)


if __name__ == "__main__":
    watch()
