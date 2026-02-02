# Plan: Add progress tracking

## Goal

1. Progress state lives directly on `EnvGroupBuilder.progress`
2. Context managers on dataclasses handle lifecycle (`group.context()`, `traj.context()`)
3. Module-level `set_trajectory_*()` functions operate on current trajectory from context

---

## Suggested API

### Dataclasses

```python
@dataclass
class TrajectoryProgress:
    """Progress state for a single trajectory."""
    group_id: int
    trajectory_id: int
    status: TrajectoryStatus = TrajectoryStatus.PENDING
    training_status: str = "pending"  # "pending" | "enqueued" | "done"
    tokens_generated: int = 0
    max_tokens: int = 65536
    reward: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    last_touched_time: float | None = None
    num_llm_calls: int = 0
    enqueued_time: float | None = None
    fwd_bwd_done_time: float | None = None

    def context(self) -> ContextManager[TrajectoryProgress]
        """Sets this as current trajectory on enter, clears on exit."""

    def to_dict(self) -> dict


@dataclass
class GroupProgress:
    """Progress state for a group of trajectories."""
    group_id: int
    trajectories: list[TrajectoryProgress]
    start_time: float | None = None
    end_time: float | None = None

    @classmethod
    def create(cls, group_size: int) -> GroupProgress
        """Factory that allocates group_id and creates trajectories."""

    def context(self) -> ContextManager[GroupProgress]
        """Registers group on enter, unregisters on exit."""

    def to_dict(self) -> dict
```

### Module-level Functions

```python

# === Trajectory State Updates (operate on current trajectory from context) ===
#
# All these functions:
# 1. Get current trajectory from _current_trajectory context var
# 2. Update fields on that trajectory
# 3. Call _write_progress() to persist to JSON
# 4. No-op if no trajectory in context (tracking disabled)

set_trajectory_in_progress() -> None
    # Called: At start of rollout for a trajectory
    # Sets: status=IN_PROGRESS, start_time=now
    # Lifecycle: PENDING -> IN_PROGRESS

set_trajectory_context(prompt_tokens: int) -> None
    # Called: In tinker_openai.py after building prompt, before sampling
    # Sets: tokens_generated=prompt_tokens, last_touched_time=now, num_llm_calls++
    # Purpose: Track token usage and LLM call count during generation
    # Note: Called potentially multiple times per trajectory (multi-turn)

set_trajectory_sampled(total_tokens: int) -> None
    # Called: In train.py after run_rollout completes successfully
    # Sets: status=SAMPLED, tokens_generated=total_tokens, end_time=now
    # Lifecycle: IN_PROGRESS -> SAMPLED
    # Meaning: Generation done, awaiting scoring/reward

set_trajectory_completed(reward: float) -> None
    # Called: After rubric.score_group() assigns rewards
    # Sets: status=COMPLETED, reward=reward
    # Lifecycle: SAMPLED -> COMPLETED
    # Meaning: Trajectory fully processed, has final reward

set_trajectory_enqueued() -> None
    # Called: When forward_backward_async is invoked for this trajectory
    # Sets: training_status="enqueued", enqueued_time=now
    # Lifecycle: training_status: "pending" -> "enqueued"
    # Meaning: Training request submitted to TPU, waiting for result

set_trajectory_fwd_bwd_done() -> None
    # Called: When forward_backward result is consumed
    # Sets: training_status="done", fwd_bwd_done_time=now
    # Lifecycle: training_status: "enqueued" -> "done"
    # Meaning: Training complete for this trajectory

# === Batch Cleanup ===
set_new_optim_step() -> None
    # Called: After optim_step completes (weights updated)
    # Action: Removes all groups where ALL trajectories have training_status="done"
    # Purpose: Clean up finished groups from JSON state in one batch operation
    # Note: Groups still mid-training (some trajectories not "done") are kept

# === Internal (called by dataclass methods) ===
_allocate_group_id() -> int
_register_group(progress: GroupProgress) -> None
_unregister_group(group_id: int) -> None
_write_progress() -> None
```

### Trajectory Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SAMPLING PHASE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PENDING ──────────────> IN_PROGRESS ──────────────> SAMPLED               │
│            set_trajectory    │                        set_trajectory        │
│            _in_progress()    │                        _sampled()            │
│                              │                                              │
│                              ▼                                              │
│                    set_trajectory_context()                                 │
│                    (called N times during                                   │
│                     multi-turn generation)                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           SCORING PHASE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SAMPLED ──────────────────────────────────────────> COMPLETED             │
│                          set_trajectory_completed()                         │
│                          (after rubric.score_group)                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                          TRAINING PHASE                                      │
│                    (parallel to status lifecycle)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   training_status:                                                          │
│                                                                             │
│   "pending" ──────────────> "enqueued" ──────────────> "done"               │
│              set_trajectory   │                        set_trajectory       │
│              _enqueued()      │                        _fwd_bwd_done()      │
│                               │                                             │
│              (forward_backward_async called)    (result consumed)           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                          CLEANUP PHASE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   After optim_step completes:                                               │
│                                                                             │
│   set_new_optim_step()                                                      │
│       │                                                                     │
│       └──> Removes all groups where ALL trajectories have                   │
│            training_status="done" from _active_groups                       │
│                                                                             │
│   Note: This is a batch cleanup. Individual groups can also be              │
│         removed via GroupProgress.context() exit (calls _unregister_group)  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module-level State

```python
_current_trajectory: ContextVar[TrajectoryProgress | None] = ContextVar("traj", default=None)
_active_groups: dict[int, GroupProgress] = {}
_write_lock: threading.Lock = threading.Lock()
_next_group_id: int = 0
_enabled: bool = False
_group_size: int = 8
_batch_start_time: float | None = None
```

---



###  Dataclass Methods
| Method | Description |
|--------|-------------|
| `TrajectoryProgress.context()` | Context manager that sets `_current_trajectory` on enter, clears on exit |
| `TrajectoryProgress.to_dict()` | JSON serialization for `watch()` display |
| `GroupProgress.create(group_size)` | Factory that allocates ID and creates trajectories |
| `GroupProgress.context()` | Context manager that registers on enter, unregisters on exit |
| `GroupProgress.to_dict()` | JSON serialization for `watch()` display |

###  Module Functions
| Function | When Called | Fields Updated |
|----------|-------------|----------------|
| `set_trajectory_in_progress()` | Start of rollout | `status=IN_PROGRESS`, `start_time` |
| `set_trajectory_context(prompt_tokens)` | In tinker_openai.py before sampling | `tokens_generated`, `last_touched_time`, `num_llm_calls++` |
| `set_trajectory_sampled(total_tokens)` | After run_rollout completes | `status=SAMPLED`, `tokens_generated`, `end_time` |
| `set_trajectory_completed(reward)` | After rubric.score_group | `status=COMPLETED`, `reward` |
| `set_trajectory_enqueued()` | When forward_backward_async called | `training_status="enqueued"`, `enqueued_time` |
| `set_trajectory_fwd_bwd_done()` | When forward_backward result consumed | `training_status="done"`, `fwd_bwd_done_time` |
| `set_new_optim_step()` | After optim_step completes | Removes all done groups from `_active_groups` |

### Internal Functions (called by dataclass methods)
| Function | Description |
|----------|-------------|
| `_allocate_group_id()` | Returns next group ID |
| `_register_group(progress)` | Adds group to `_active_groups`, writes JSON |
| `_unregister_group(group_id)` | Removes group from `_active_groups`, writes JSON |

---

## Existing Code to Preserve

| Item | Notes |
|------|-------|
| `TrajectoryStatus` enum | Status values |
| `PROGRESS_FILE` constant | File path |
| `RollingStats` class | Display code |
| `RollingDurationStats` class | Display code |
| `watch()` function | Display code |
| `compute_stats()` | Display code |
| `build_dashboard()` | Display code |
| `build_table()` | Display code |
| `__main__` block | Entry point |

---




## JSON Format (must match for watch() compatibility)

```python
# _write_progress() output format
{
    "timestamp": float,
    "batch_start_time": float | None,
    "group_size": int,
    "groups": {
        "<group_id>": {
            "group_id": int,
            "trajectories": {
                "<traj_id>": {
                    "group_id": int,
                    "trajectory_id": int,
                    "status": str,  # "pending" | "in_progress" | "sampled" | "completed"
                    "tokens_generated": int,
                    "reward": float | None,
                    "start_time": float | None,
                    "end_time": float | None,
                    "last_touched_time": float | None,
                    "num_llm_calls": int,
                    "training_status": str,  # "pending" | "enqueued" | "done"
                    "enqueued_time": float | None,
                    "fwd_bwd_done_time": float | None,
                }
            },
            "start_time": float | None,
            "end_time": float | None,
        }
    }
}
```

---

## Usage Examples

```python
# train.py
with GroupProgress.create(cfg.group_size, cfg.max_tokens).context() as progress:
    builder.progress = progress

    # Per-trajectory rollout
    with progress.trajectories[traj_idx].context():
        set_trajectory_in_progress()
        result = await run_rollout(...)
        set_trajectory_sampled(token_counts["total_tokens"])

    # Mark completed with rewards
    for i, reward in enumerate(rewards):
        with progress.trajectories[i].context():
            set_trajectory_completed(reward)

# Auto-unregistered on context exit
builder.progress = None

# tinker_openai.py
set_trajectory_context(len(prompt_token_ids))  # just works if in traj context
```

---

## Files to Modify

### 1. `tinker_cookbook/utils/trajectory_progress.py`
- Add: `TrajectoryProgress`, `GroupProgress`, module-level functions
- Keep: `TrajectoryStatus`, display code (`watch()`, `RollingStats`, etc.)

### 2. `tinker_cookbook/rl/types.py`
- Add: `progress: GroupProgress | None` attribute to `EnvGroupBuilder`

### 3. `tinker_cookbook/rl/train.py`
- Import progress tracking functions
- Use `GroupProgress.create(...).context()` to manage group lifecycle
- Use `with traj.context():` pattern for per-trajectory tracking

### 4. `tinker_cookbook/recipes/verifiers_rl/train.py`
- Import progress tracking functions
- Use `builder.progress` and `with traj.context():` pattern in `custom_do_group_rollout`

### 5. `tinker_cookbook/recipes/verifiers_rl/tinker_openai.py`
- Import: `from ... import set_trajectory_context`
- Call: `set_trajectory_context(len(prompt_token_ids))`

---

## Implementation Order

1. Add `TrajectoryProgress`, `GroupProgress`, and module functions to `trajectory_progress.py`
2. Add `progress` attribute to `EnvGroupBuilder` in `rl/types.py`
3. Integrate into `recipes/verifiers_rl/train.py`
4. Integrate into `recipes/verifiers_rl/tinker_openai.py`
5. Integrate into `rl/train.py`
6. Test with `watch()` to verify JSON compatibility

---

## How This Relates to `verifiers_env.py`

### Direct Relationship

1. **Inheritance chain**: `VerifiersEnvGroupBuilder` (in `recipes/verifiers_rl/verifiers_env.py:140`) extends `EnvGroupBuilder`. The `progress: GroupProgress | None` attribute is added to `EnvGroupBuilder`.

2. **No direct modification needed**: `verifiers_env.py` itself doesn't need changes - it inherits the `progress` attribute from its parent class.

### Usage in Consumers

| File | Integration |
|------|-------------|
| `rl/types.py` | Add `progress: GroupProgress \| None` to `EnvGroupBuilder` |
| `recipes/verifiers_rl/train.py` | Set `builder.progress`, use `with traj.context():` pattern |
| `recipes/verifiers_rl/tinker_openai.py` | Call `set_trajectory_context(len(prompt_token_ids))` |

`VerifiersEnvGroupBuilder` instances get progress tracking through inheritance. The calling code in `train.py` manages lifecycle using context managers.
