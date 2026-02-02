"""
Progress tracking for RL trajectory collection.

Provides TrajectoryProgress and GroupProgress dataclasses with RAII-style
context managers for lifecycle management. Module-level functions operate
on the current trajectory from context.

The JSON state file is watched by viewer.py for real-time display.
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


class TrajectoryStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SAMPLED = "sampled"
    COMPLETED = "completed"


# =============================================================================
# Module-level State
# =============================================================================

_current_trajectory: ContextVar[TrajectoryProgress | None] = ContextVar(
    "traj", default=None
)
_active_groups: dict[int, GroupProgress] = {}
_write_lock: threading.Lock = threading.Lock()
_next_group_id: int = 0
_enabled: bool = False
_group_size: int = 8
_num_workers: int | None = None
_batch_start_time: float | None = None


# =============================================================================
# Configuration
# =============================================================================


def enable_tracking(group_size: int, num_workers: int | None) -> None:
    """Enable progress tracking with the given group size."""
    global _enabled, _group_size, _num_workers, _batch_start_time
    _enabled = True
    _group_size = group_size
    _num_workers = num_workers
    _batch_start_time = time.time()
    _write_progress()


def disable_tracking() -> None:
    """Disable progress tracking."""
    global _enabled
    _enabled = False


def is_tracking_enabled() -> bool:
    """Check if progress tracking is enabled."""
    return _enabled


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class TrajectoryProgress:
    """Progress state for a single trajectory."""

    group_id: int
    trajectory_id: int
    status: TrajectoryStatus = TrajectoryStatus.PENDING
    training_status: str = "pending"  # "pending" | "enqueued" | "done"
    tokens_generated: int = 0
    reward: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    last_touched_time: float | None = None
    num_llm_calls: int = 0
    enqueued_time: float | None = None
    fwd_bwd_done_time: float | None = None

    @contextmanager
    def context(self) -> Iterator[TrajectoryProgress]:
        """Sets this as current trajectory on enter, clears on exit."""
        token = _current_trajectory.set(self)
        try:
            yield self
        finally:
            _current_trajectory.reset(token)

    def to_dict(self) -> dict:
        """JSON serialization for watch() display."""
        return {
            "group_id": self.group_id,
            "trajectory_id": self.trajectory_id,
            "status": self.status.value,
            "training_status": self.training_status,
            "tokens_generated": self.tokens_generated,
            "reward": self.reward,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "last_touched_time": self.last_touched_time,
            "num_llm_calls": self.num_llm_calls,
            "enqueued_time": self.enqueued_time,
            "fwd_bwd_done_time": self.fwd_bwd_done_time,
        }


@dataclass
class GroupProgress:
    """Progress state for a group of trajectories."""

    group_id: int
    trajectories: list[TrajectoryProgress] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None

    @classmethod
    def create(cls, group_size: int) -> GroupProgress:
        """Factory that allocates group_id and creates trajectories."""
        group_id = _allocate_group_id()
        trajectories = [
            TrajectoryProgress(group_id=group_id, trajectory_id=i)
            for i in range(group_size)
        ]
        return cls(group_id=group_id, trajectories=trajectories)

    @contextmanager
    def context(self) -> Iterator[GroupProgress]:
        """Registers group on enter, unregisters on exit."""
        self.start_time = time.time()
        _register_group(self)
        try:
            yield self
        finally:
            self.end_time = time.time()
            _write_progress()

    def to_dict(self) -> dict:
        """JSON serialization for watch() display."""
        return {
            "group_id": self.group_id,
            "trajectories": {
                str(t.trajectory_id): t.to_dict() for t in self.trajectories
            },
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


# =============================================================================
# Internal Functions
# =============================================================================


def _allocate_group_id() -> int:
    """Returns next group ID."""
    global _next_group_id
    with _write_lock:
        group_id = _next_group_id
        _next_group_id += 1
        return group_id


def _register_group(progress: GroupProgress) -> None:
    """Adds group to _active_groups, writes JSON."""
    with _write_lock:
        _active_groups[progress.group_id] = progress
    _write_progress()


def _unregister_group(group_id: int) -> None:
    """Removes group from _active_groups, writes JSON."""
    with _write_lock:
        _active_groups.pop(group_id, None)
    _write_progress()


def _write_progress() -> None:
    """Write current progress state to JSON file."""
    if not _enabled:
        return

    with _write_lock:
        state = {
            "timestamp": time.time(),
            "batch_start_time": _batch_start_time,
            "group_size": _group_size,
            "num_workers": _num_workers,
            "groups": {
                str(gid): group.to_dict() for gid, group in _active_groups.items()
            },
        }

    PROGRESS_FILE.write_text(json.dumps(state, indent=2))


def _get_current_trajectory() -> TrajectoryProgress | None:
    """Get current trajectory from context, or None if not in context."""
    return _current_trajectory.get()


# =============================================================================
# Trajectory State Updates
# =============================================================================


def set_trajectory_in_progress() -> None:
    """
    Called: At start of rollout for a trajectory
    Sets: status=IN_PROGRESS, start_time=now
    Lifecycle: PENDING -> IN_PROGRESS
    """
    traj = _get_current_trajectory()
    if traj is None:
        return
    traj.status = TrajectoryStatus.IN_PROGRESS
    traj.start_time = time.time()
    _write_progress()


def set_trajectory_context(prompt_tokens: int) -> None:
    """
    Called: In tinker_openai.py after building prompt, before sampling
    Sets: tokens_generated=prompt_tokens, last_touched_time=now, num_llm_calls++
    Purpose: Track token usage and LLM call count during generation
    Note: Called potentially multiple times per trajectory (multi-turn)
    """
    traj = _get_current_trajectory()
    if traj is None:
        return
    traj.tokens_generated = prompt_tokens
    traj.last_touched_time = time.time()
    traj.num_llm_calls += 1
    _write_progress()


def set_trajectory_sampled(total_tokens: int) -> None:
    """
    Called: In train.py after run_rollout completes successfully
    Sets: status=SAMPLED, tokens_generated=total_tokens, end_time=now
    Lifecycle: IN_PROGRESS -> SAMPLED
    Meaning: Generation done, awaiting scoring/reward
    """
    traj = _get_current_trajectory()
    if traj is None:
        return
    traj.status = TrajectoryStatus.SAMPLED
    traj.tokens_generated = total_tokens
    traj.end_time = time.time()
    _write_progress()


def set_trajectory_completed(reward: float) -> None:
    """
    Called: After rubric.score_group() assigns rewards
    Sets: status=COMPLETED, reward=reward
    Lifecycle: SAMPLED -> COMPLETED
    Meaning: Trajectory fully processed, has final reward
    """
    traj = _get_current_trajectory()
    if traj is None:
        return
    traj.status = TrajectoryStatus.COMPLETED
    traj.reward = reward
    _write_progress()


def set_trajectory_enqueued() -> None:
    """
    Called: When forward_backward_async is invoked for this trajectory
    Sets: training_status="enqueued", enqueued_time=now
    Lifecycle: training_status: "pending" -> "enqueued"
    Meaning: Training request submitted to TPU, waiting for result
    """
    traj = _get_current_trajectory()
    if traj is None:
        return
    traj.training_status = "enqueued"
    traj.enqueued_time = time.time()
    _write_progress()


def set_trajectory_fwd_bwd_done() -> None:
    """
    Called: When forward_backward result is consumed
    Sets: training_status="done", fwd_bwd_done_time=now
    Lifecycle: training_status: "enqueued" -> "done"
    Meaning: Training complete for this trajectory
    """
    traj = _get_current_trajectory()
    if traj is None:
        return
    traj.training_status = "done"
    traj.fwd_bwd_done_time = time.time()
    _write_progress()


# =============================================================================
# Batch Cleanup
# =============================================================================


def set_new_optim_step() -> None:
    """
    Called: After optim_step completes (weights updated)
    Action: Removes all groups where ALL trajectories have training_status="done"
    Purpose: Clean up finished groups from JSON state in one batch operation
    Note: Groups still mid-training (some trajectories not "done") are kept
    """
    if not _enabled:
        return

    with _write_lock:
        groups_to_remove = []
        for gid, group in _active_groups.items():
            all_done = all(
                t.training_status == "done" for t in group.trajectories
            )
            if all_done:
                groups_to_remove.append(gid)

        for gid in groups_to_remove:
            del _active_groups[gid]

    _write_progress()
