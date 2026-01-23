"""
Implements RL on general MDPs
"""

import asyncio
import io
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Sequence

import chz
import numpy as np
import tinker
import torch
from tinker.types import LossFnType
from tinker_cookbook.utils.trajectory_progress import TrajectoryProgressTracker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import (
    compute_kl_sample_train,
    compute_post_kl,
    compute_sampling_client_metrics,
    incorporate_kl_penalty,
)
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import logtree, ml_log
from tinker_cookbook.utils.misc_utils import safezip, split_list, timed, all_same
from tinker_cookbook.utils.trace import scope, update_scope_context, trace_init

logger = logging.getLogger(__name__)


def _get_evaluator_name(evaluator: SamplingClientEvaluator) -> str:
    return (
        evaluator.name
        if isinstance(evaluator, RLTestSetEvaluator) and evaluator.name is not None
        else ""
    )


@contextmanager
def _get_logtree_scope(
    log_path: str | None, num_groups_to_log: int, f_name: str, scope_name: str
) -> Iterator[None]:
    """
    Creates a context manager; all log inside this context will be logged under the section `scope_name`.
    It will create a file with the path of log_path/f_name.html
    If num_groups_to_log is 0, it will disable logging (but note that this function does not actually implement the logic for logging itself!)
    """
    if log_path is not None and num_groups_to_log > 0:
        logtree_path = os.path.join(log_path, f"{f_name}.html")
        with logtree.init_trace(scope_name, path=logtree_path):
            yield
    else:
        yield


@scope
def _select_representative_inds(scores: list[float], num_inds: int) -> list[int]:
    assert num_inds <= len(scores)
    sorted_inds = np.argsort(scores)
    uniform_inds = np.linspace(0, len(sorted_inds) - 1, num_inds).astype(int)
    return [int(sorted_inds[i]) for i in uniform_inds]


@scope
def print_group(traj_group: TrajectoryGroup, tokenizer: Tokenizer):
    """
    Print a subset of the trajectory group to the console.
    """
    # Cut down the number of trajectories to print
    max_trajs_to_print = 4
    if len(traj_group.trajectories_G) > max_trajs_to_print:
        inds = _select_representative_inds(traj_group.get_total_rewards(), max_trajs_to_print)
        traj_group = TrajectoryGroup(
            trajectories_G=[traj_group.trajectories_G[i] for i in inds],
            final_rewards_G=[traj_group.final_rewards_G[i] for i in inds],
            metrics_G=[traj_group.metrics_G[i] for i in inds],
        )

    rewards = traj_group.get_total_rewards()
    advantages_G = compute_advantages([traj_group])
    data_D, metadata_D = assemble_training_data([traj_group], advantages_G)

    buf = io.StringIO()

    @scope
    def bprint(s: str):
        print(s, file=buf)

    bprint("\n====== Trajectory Group ======")
    last_metadata = None
    for datum, metadata in safezip(data_D, metadata_D):
        idx = metadata["traj_idx"]
        if metadata != last_metadata:
            bprint(f"****** trajectory idx={idx}, reward={rewards[idx]:.3g} ******")
            # Print trajectory-level metrics
            if traj_group.metrics_G[idx]:
                bprint("Trajectory metrics:")
                for key, value in traj_group.metrics_G[idx].items():
                    bprint(f"  {key}: {value}")
            # Print per-transition metrics
            transition_metrics = [
                transition.metrics
                for transition in traj_group.trajectories_G[idx].transitions
                if transition.metrics
            ]
            if transition_metrics:
                bprint("Per-step metrics:")
                for i, metrics in enumerate(transition_metrics):
                    bprint(f"  Step {i}:")
                    for key, value in metrics.items():
                        bprint(f"    {key}: {value}")
        bprint("---- datum ----")
        bprint(colorize_example(datum, tokenizer, key="advantages"))
        last_metadata = metadata
    bprint("====== End Trajectory Group ======")
    logger.info(buf.getvalue().rstrip())


def _remove_mask(datum: tinker.Datum) -> tinker.Datum:
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


def _training_logprobs_from_fwd_bwd(
    fwd_bwd_result: tinker.ForwardBackwardOutput,
) -> list[torch.Tensor]:
    return [output["logprobs"].to_torch() for output in fwd_bwd_result.loss_fn_outputs]


@scope
async def train_step(
    data_D: List[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: LossFnType,
    loss_fn_config: dict[str, float] | None,  # TODO: must fix this slop - should be typed config
) -> List[torch.Tensor]:
    """Train the model on collected trajectories.

    Pipelines forward_backward and optim_step so they land on the same clock cycle.
    """
    batches = split_list(data_D, min(num_substeps, len(data_D)))
    if not batches:
        return []

    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    training_logprobs_D: list[torch.Tensor] = []

    # Enqueue first batch
    fwd_bwd_future = await training_client.forward_backward_async(
        [_remove_mask(d) for d in batches[0]], loss_fn=loss_fn, loss_fn_config=loss_fn_config
    )
    optim_future = await training_client.optim_step_async(adam_params)

    for i in range(len(batches)):
        # Enqueue next batch before consuming current results (to stay on same clock cycle)
        if i + 1 < len(batches):
            next_fwd_bwd_future = await training_client.forward_backward_async(
                [_remove_mask(d) for d in batches[i + 1]], loss_fn=loss_fn, loss_fn_config=loss_fn_config
            )
            next_optim_future = await training_client.optim_step_async(adam_params)
        else:
            next_fwd_bwd_future = None
            next_optim_future = None
        # Consume current results
        fwd_bwd_result = await fwd_bwd_future.result_async()
        training_logprobs_D.extend(_training_logprobs_from_fwd_bwd(fwd_bwd_result))
        await optim_future.result_async()
        # Move to next iteration
        if next_fwd_bwd_future is not None and next_optim_future is not None:
            fwd_bwd_future = next_fwd_bwd_future
            optim_future = next_optim_future

    return training_logprobs_D


@chz.chz
class StreamMinibatchConfig:
    """
    Configuration for training with minibatch streaming.
    Once we have accumulated enough trajectories for a minibatch, we will
    immediately train on them, instead of waiting for the full batch of
    trajectories to be ready.
    """

    # Total number of trajectory groups across all minibatches and substeps
    groups_per_batch: int
    # For each substep, we will divide up the number of trajectory groups
    # into this many minibatches.
    # We will do num_minibatches forward_backward() passes and one optim_step()
    # per substep.
    num_minibatches: int


@chz.chz
class AsyncConfig:
    """Configuration for async RL training"""

    # If samples are generated from a sample more than this many steps ago,
    # we will skip training on them.
    max_steps_off_policy: int
    # We will ensure all batches have at least this many groups, even
    # as we discard stale samples
    groups_per_batch: int
    # Multiplier for how many groups can be in flight relative to groups_per_batch.
    # E.g., in_flight_ratio=2.0 with groups_per_batch=32 → 64 concurrent sampling workers.
    in_flight_ratio: float = 1.0


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: RLDatasetBuilder  # also determines batch size
    model_name: str
    max_tokens: int
    temperature: float = 1.0  # Changing sampling temperature is not generally recommended; does not currently play well with KL penalty
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0

    # Loss function to use for training: "importance_sampling" or "ppo"
    loss_fn: LossFnType = "importance_sampling"
    # TODO: must fix this slop - dict[str, float] | None should be a typed config, None default is bad
    loss_fn_config: dict[str, float] | None = None  # e.g. {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None
    enable_trace: bool = False

    remove_constant_reward_groups: bool = False
    eval_every: int = 20  # 0 = disabled
    save_every: int = 20  # 0 = disabled
    load_checkpoint_path: str | None = None

    async_config: AsyncConfig | None = None
    stream_minibatch_config: StreamMinibatchConfig | None = None

    # Logtree configuration
    num_groups_to_log: int = 4  # Number of groups to log per iteration (0 = disable logging)


@scope
async def run_single_evaluation(evaluator, cfg, i_batch, sampling_client):
    ev_name = _get_evaluator_name(evaluator)
    with _get_logtree_scope(
        log_path=cfg.log_path,
        num_groups_to_log=cfg.num_groups_to_log,
        f_name=f"eval_{ev_name}_iteration_{i_batch:06d}",
        scope_name=f"Running evaluation {ev_name} {i_batch}",
    ):
        eval_metrics = await evaluator(sampling_client)
        return eval_metrics


@scope
async def run_evaluations_parallel(
    evaluators: list[SamplingClientEvaluator],
    sampling_client: tinker.SamplingClient,
    cfg: Config,
    i_batch: int,
) -> dict[str, Any]:
    """Run all evaluators in parallel and return aggregated metrics."""

    # Create tasks for all evaluators with names for better traceability
    tasks = []
    for i, evaluator in enumerate(evaluators):
        ev_name = _get_evaluator_name(evaluator)
        task = asyncio.create_task(
            run_single_evaluation(evaluator, cfg, i_batch, sampling_client),
            name=f"eval_{ev_name or i}_iteration_{i_batch:06d}",
        )
        tasks.append(task)

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Merge all metrics
    metrics = {}
    for result in results:
        metrics.update(result)

    return metrics


@scope
async def do_sync_training_with_stream_minibatch(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """
    Implements fully synchronous on-policy training with minibatch streaming.
    Once we have accumulated enough trajectories for a minibatch, we will
    immediately train on them, instead of waiting for the full batch of
    trajectories to be ready. This allows us to overlap sampling and training.
    """
    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every, start_batch
    )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if (cfg.eval_every > 0 and i_batch % cfg.eval_every == 0) or i_batch == end_batch - 1:
            with timed("run_evals", metrics):
                eval_metrics = await run_evaluations_parallel(
                    evaluators, sampling_client, cfg, i_batch
                )
                metrics.update(eval_metrics)

        with _get_logtree_scope(
            cfg.log_path,
            cfg.num_groups_to_log,
            f"train_iteration_{i_batch:06d}",
            f"RL Iteration {i_batch}",
        ):
            # Samplers will produce trajectory groups asynchronously,
            # and the trainer will consume them as soon as they are ready
            trajectory_groups_queue = asyncio.Queue[WrappedTrajectoryGroup | None]()
            # Dummy queue for builder recycling (sync mode creates fresh builders each batch)
            env_group_builders_queue = asyncio.Queue[EnvGroupBuilder | None]()
            env_group_builders_P = dataset.get_batch(i_batch)
            tracker = TrajectoryProgressTracker.get_instance()

            @scope
            async def trajectory_group_worker_task(
                builder: EnvGroupBuilder, enable_logging: bool
            ) -> None:
                metrics = {}
                t_start = time.time()
                trajectory_group = await do_group_rollout_and_filter_constant_reward(
                    sampling_client,
                    builder,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                    enable_logging=enable_logging,
                )
                metrics["time/trajectory_group_worker_loop/total"] = time.time() - t_start
                if trajectory_group is not None:
                    trajectory_groups_queue.put_nowait(
                        WrappedTrajectoryGroup(
                            trajectory_group=trajectory_group,
                            env_group_builder=builder,
                            sampling_client_step=i_batch,
                            metrics=metrics,
                        )
                    )
                else:
                    trajectory_groups_queue.put_nowait(None)

            # Sample all trajectories asynchronously. If we have multiple minibatches,
            # then sampling can overlap with training.
            with tracker.track_batch(len(env_group_builders_P)):
                for i, builder in enumerate(env_group_builders_P):
                    builder._progress_group_id = i
                    asyncio.create_task(
                        trajectory_group_worker_task(builder, enable_logging=i < cfg.num_groups_to_log),
                        name=f"trajectory_group_worker_task_{i}",
                    )

                # Run multiple optimizer substeps per training iteration
                (
                    sampling_client,
                    full_batch_metrics,
                ) = await do_train_step_streaming_and_get_sampling_client(
                    cfg,
                    i_batch,
                    trajectory_groups_queue,
                    env_group_builders_queue,
                    training_client,
                    service_client,
                    tokenizer,
                    lambda _: True,  # No stale filtering in sync mode (on-policy)
                )

        # Log metrics
        metrics.update(full_batch_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)

        # Clean up temporary attributes
        for builder in env_group_builders_P:
            if hasattr(builder, "_progress_group_id"):
                delattr(builder, "_progress_group_id")


@chz.chz
class WrappedTrajectoryGroup:
    """
    A wrapper around a trajectory group that includes metadata about how it was generated.
    Used when we need to overlap sampling and training.
    """

    trajectory_group: TrajectoryGroup
    # The env group builder that produced the trajectory group.
    # Pass this along in case the sampler is too stale, and we need to
    # requeue this group.
    env_group_builder: EnvGroupBuilder
    # The step that produced this trajectory group.
    sampling_client_step: int
    metrics: dict[str, Any] = chz.field(default_factory=dict)


@scope
async def do_async_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements async off-policy training, capped at K steps off policy."""
    assert cfg.async_config is not None
    num_workers = int(cfg.async_config.groups_per_batch * cfg.async_config.in_flight_ratio)

    shutdown_event = asyncio.Event()
    # Unbounded queue - maxsize was causing put() to block during recycling.
    # With fire-and-forget recycling, unbounded is safe and prevents starvation.
    env_group_builders_queue = asyncio.Queue[EnvGroupBuilder | None]()
    trajectory_groups_queue = asyncio.Queue[WrappedTrajectoryGroup | None]()
    tracker = TrajectoryProgressTracker.get_instance()

    # Initial sampling client to use
    path_dict = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name=f"{start_batch:06d}",
        log_path=cfg.log_path,
        loop_state={"batch": start_batch},
        kind="both",
    )

    # This will be updated by the training loop
    sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
    sampling_client_step = start_batch
    sampling_client_updated_event = asyncio.Event()
    sampling_client_updated_event.set()

    @scope
    def shutdown_loops():
        """Trigger all loops to shutdown"""
        logger.info(f"[shutdown_loops] Triggering shutdown. Setting shutdown_event and sending {num_workers} None values to env_group_builders_queue")
        shutdown_event.set()
        for _ in range(num_workers):
            env_group_builders_queue.put_nowait(None)
        sampling_client_updated_event.set()
        logger.info("[shutdown_loops] Shutdown signals sent")

    @scope
    async def dataloader_loop():
        """Gets the next set of env builders to run"""
        logger.info(f"[dataloader] Starting. start_batch={start_batch} end_batch={end_batch}")
        i_batch = start_batch
        total_builders_added = 0
        while not shutdown_event.is_set() and i_batch < end_batch:
            env_group_builders_P = dataset.get_batch(i_batch)
            logger.info(f"[dataloader] Batch {i_batch}: Got {len(env_group_builders_P)} builders from dataset. env_group_builders_queue qsize={env_group_builders_queue.qsize()}")
            for env_group_builder in env_group_builders_P:
                gid = tracker.allocate_group_id()
                env_group_builder._progress_group_id = gid
                await env_group_builders_queue.put(env_group_builder)
                total_builders_added += 1
                logger.debug(f"[dataloader] Batch {i_batch}: Added builder gid={gid} to queue. qsize={env_group_builders_queue.qsize()}")
            logger.info(f"[dataloader] Batch {i_batch}: Done adding builders. Total added so far: {total_builders_added}")
            i_batch += 1
        logger.info(f"[dataloader] Finished. Added {total_builders_added} builders total across {i_batch - start_batch} batches")

    @scope
    async def trajectory_group_worker_loop():
        """Generates trajectories for a single env builder"""
        worker_id = id(asyncio.current_task())
        logger.info(f"[worker:{worker_id}] Worker started")
        rollout_count = 0
        while not shutdown_event.is_set():
            logger.debug(f"[worker:{worker_id}] Waiting for builder from env_group_builders_queue (qsize={env_group_builders_queue.qsize()})")
            env_group_builder = await env_group_builders_queue.get()
            if env_group_builder is None:
                logger.info(f"[worker:{worker_id}] Received None, shutting down after {rollout_count} rollouts")
                break

            # Allocate new group ID for requeued builders (stale ones have ID deleted)
            is_requeued = not hasattr(env_group_builder, "_progress_group_id")
            if is_requeued:
                env_group_builder._progress_group_id = tracker.allocate_group_id()
            gid = env_group_builder._progress_group_id
            logger.info(f"[worker:{worker_id}] Got builder gid={gid} requeued={is_requeued} sampling_client_step={sampling_client_step}")

            metrics = {}
            t_start = time.time()
            # Save a reference to the sampling client step in case it changes
            # while we're running the rollout
            sampling_client_step_copy = sampling_client_step
            logger.info(f"[worker:{worker_id}] gid={gid} Starting rollout with sampling_client_step_copy={sampling_client_step_copy}")
            try:
                trajectory_group = await do_group_rollout_and_filter_constant_reward(
                    sampling_client,
                    env_group_builder,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                )
            except Exception:
                logger.exception(f"[worker:{worker_id}] gid={gid} Exception during rollout after {time.time() - t_start:.1f}s, recycling builder")
                # Clean up tracker if group was allocated
                if hasattr(env_group_builder, "_progress_group_id"):
                    tracker.remove_group(env_group_builder._progress_group_id)
                    delattr(env_group_builder, "_progress_group_id")
                # Recycle the builder (fire-and-forget to avoid blocking)
                asyncio.create_task(
                    env_group_builders_queue.put(env_group_builder),
                    name=f"recycle_failed_builder_{worker_id}",
                )
                continue
            rollout_duration = time.time() - t_start
            rollout_count += 1
            if trajectory_group is None:
                logger.info(f"[worker:{worker_id}] gid={gid} Rollout returned None (constant reward filtered) after {rollout_duration:.1f}s, putting None in trajectory_groups_queue")
                trajectory_groups_queue.put_nowait(None)
            else:
                logger.info(f"[worker:{worker_id}] gid={gid} Rollout completed in {rollout_duration:.1f}s, putting in trajectory_groups_queue (qsize={trajectory_groups_queue.qsize()})")
                metrics["time/trajectory_group_worker_loop/total"] = rollout_duration
                trajectory_groups_queue.put_nowait(
                    WrappedTrajectoryGroup(
                        trajectory_group=trajectory_group,
                        env_group_builder=env_group_builder,
                        sampling_client_step=sampling_client_step_copy,
                        metrics=metrics,
                    )
                )
                logger.debug(f"[worker:{worker_id}] gid={gid} Successfully put in trajectory_groups_queue, new qsize={trajectory_groups_queue.qsize()}")

    @scope
    async def training_loop():
        """
        Waits for a sufficient number of valid trajectories to be accumulated and trains on them.
        Will discard trajectories that are too stale.
        """
        assert cfg.async_config is not None
        logger.info(f"[training_loop] Started. start_batch={start_batch} end_batch={end_batch} max_steps_off_policy={cfg.async_config.max_steps_off_policy}")

        i_batch = start_batch
        wrapped_trajectory_groups = []
        stale_count_this_step = 0
        valid_count_this_step = 0
        while i_batch < end_batch:
            logger.debug(f"[training_loop] Step {i_batch}: Waiting for group from trajectory_groups_queue (qsize={trajectory_groups_queue.qsize()}) env_builders_qsize={env_group_builders_queue.qsize()}")
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if wrapped_trajectory_group is None:
                logger.debug(f"[training_loop] Step {i_batch}: Got None from queue, continuing")
                continue

            wg_gid = getattr(wrapped_trajectory_group.env_group_builder, "_progress_group_id", "NO_GID")
            logger.debug(f"[training_loop] Step {i_batch}: Got group gid={wg_gid} sampling_client_step={wrapped_trajectory_group.sampling_client_step}")

            @scope
            def filter_stale_trajectory_group(
                wrapped_trajectory_group: WrappedTrajectoryGroup | None,
            ) -> bool:
                """Returns False if the trajectory group is too stale or not valid"""
                nonlocal stale_count_this_step
                nonlocal valid_count_this_step
                if wrapped_trajectory_group is None:
                    logger.debug(f"[filter_stale] Step {i_batch}: Got None, returning False")
                    return False

                gid = getattr(wrapped_trajectory_group.env_group_builder, "_progress_group_id", "NO_GID")
                staleness = i_batch - wrapped_trajectory_group.sampling_client_step
                # If the samples are too stale, requeue the data so that it will be used eventually.
                # Requeue on a separate coroutine to avoid blocking the training loop
                assert cfg.async_config is not None
                if staleness > cfg.async_config.max_steps_off_policy:
                    stale_count_this_step += 1
                    logger.info(f"[filter_stale] Step {i_batch}: gid={gid} STALE (staleness={staleness} > max={cfg.async_config.max_steps_off_policy}) stale_count={stale_count_this_step} valid_count={valid_count_this_step}")
                    # Remove stale group from tracker and clear ID so it gets reallocated
                    if hasattr(wrapped_trajectory_group.env_group_builder, "_progress_group_id"):
                        tracker.remove_group(gid)
                        delattr(wrapped_trajectory_group.env_group_builder, "_progress_group_id")
                        logger.debug(f"[filter_stale] Step {i_batch}: gid={gid} Removed from tracker, cleared _progress_group_id")
                    asyncio.create_task(
                        env_group_builders_queue.put(wrapped_trajectory_group.env_group_builder),
                        name="requeue_stale_sample_task",
                    )
                    logger.debug(f"[filter_stale] Step {i_batch}: gid={gid} Requeued builder to env_group_builders_queue")
                    return False
                valid_count_this_step += 1
                logger.info(f"[filter_stale] Step {i_batch}: gid={gid} VALID (staleness={staleness} <= max={cfg.async_config.max_steps_off_policy}) stale_count={stale_count_this_step} valid_count={valid_count_this_step}")
                return True

            metrics = {
                "training_client/step": i_batch,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
            }
            t_start = time.time()

            nonlocal sampling_client
            nonlocal sampling_client_step
            if cfg.stream_minibatch_config is not None:
                logger.info(f"[training_loop] Step {i_batch}: Putting first group back in queue and calling do_train_step_streaming_and_get_sampling_client")
                logger.info(f"[training_loop] Step {i_batch}: Queue sizes BEFORE streaming: trajectory_groups={trajectory_groups_queue.qsize()} env_builders={env_group_builders_queue.qsize()}")
                stale_count_this_step = 0
                valid_count_this_step = 0
                await trajectory_groups_queue.put(wrapped_trajectory_group)
                t_streaming_start = time.time()
                (
                    sampling_client,
                    train_step_metrics,
                ) = await do_train_step_streaming_and_get_sampling_client(
                    cfg,
                    i_batch,
                    trajectory_groups_queue,
                    env_group_builders_queue,
                    training_client,
                    service_client,
                    tokenizer,
                    filter_stale_trajectory_group,
                )
                streaming_duration = time.time() - t_streaming_start
                logger.info(f"[training_loop] Step {i_batch}: do_train_step_streaming_and_get_sampling_client DONE in {streaming_duration:.2f}s. stale_count={stale_count_this_step} valid_count={valid_count_this_step}")
                logger.info(f"[training_loop] Step {i_batch}: Queue sizes AFTER streaming: trajectory_groups={trajectory_groups_queue.qsize()} env_builders={env_group_builders_queue.qsize()}")
            else:
                if not filter_stale_trajectory_group(wrapped_trajectory_group):
                    continue

                # Dynamic sampling: Wait for enough trajectories to accumulate to
                # ensure all batch sizes are the same size. This avoids needing to adjust
                # the learning rate for different batch sizes.
                wrapped_trajectory_groups.append(wrapped_trajectory_group)
                if len(wrapped_trajectory_groups) < cfg.async_config.groups_per_batch:
                    continue
                logger.info(
                    f"[training_loop] Step {i_batch}: Will train on batch, num groups: {len(wrapped_trajectory_groups)}"
                )

                # Compute sampling client metrics, as samples may have been generated with
                # different sampler versions
                metrics.update(compute_sampling_client_metrics(wrapped_trajectory_groups))

                # TODO: For proper checkpointing, we also need to save dataloader state and
                # all queued trajectory groups that haven't been trained on yet
                for wg in wrapped_trajectory_groups:
                    gid = wg.env_group_builder._progress_group_id
                    for tid in range(len(wg.trajectory_group.trajectories_G)):
                        tracker.mark_trajectory_training_enqueued(gid, tid)

                sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
                    cfg,
                    i_batch,
                    training_client,
                    service_client,
                    tokenizer,
                    [g.env_group_builder for g in wrapped_trajectory_groups],
                    [g.trajectory_group for g in wrapped_trajectory_groups],
                )
                # Mark training complete and cleanup groups
                for wg in wrapped_trajectory_groups:
                    gid = wg.env_group_builder._progress_group_id
                    for tid in range(len(wg.trajectory_group.trajectories_G)):
                        tracker.mark_trajectory_fwd_bwd_done(gid, tid)
                    tracker.remove_group(gid)
                    if hasattr(wg.env_group_builder, "_progress_group_id"):
                        delattr(wg.env_group_builder, "_progress_group_id")
            sampling_client_step = i_batch + 1
            sampling_client_updated_event.set()
            logger.info(f"[training_loop] Step {i_batch}: Updated sampling_client_step to {sampling_client_step}")

            # Log metrics
            metrics.update(train_step_metrics)
            metrics["time/training_loop/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=i_batch)
            logger.info(f"[training_loop] Step {i_batch}: COMPLETED in {metrics['time/training_loop/total']:.2f}s. Advancing to step {i_batch + 1}")
            logger.info(f"[training_loop] Step {i_batch}: Final queue sizes: trajectory_groups={trajectory_groups_queue.qsize()} env_builders={env_group_builders_queue.qsize()}")
            i_batch += 1
            wrapped_trajectory_groups = []

        logger.info(f"[training_loop] Reached end_batch={end_batch}, calling shutdown_loops()")
        shutdown_loops()

    @scope
    async def evaluation_loop():
        """Runs evals periodically"""
        if len(evaluators) == 0 or cfg.eval_every == 0:
            return

        while not shutdown_event.is_set():
            await sampling_client_updated_event.wait()
            sampling_client_updated_event.clear()

            metrics = {}
            t_start = time.time()
            # Save a reference to the original values in case it changes
            # while we're running the evals
            sampling_client_eval_step = sampling_client_step
            sampling_client_eval = sampling_client
            if cfg.eval_every > 0 and sampling_client_eval_step % cfg.eval_every == 0:
                with timed("run_evals", metrics):
                    for evaluator in evaluators:
                        eval_metrics = await evaluator(sampling_client_eval)
                        metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})
                metrics["time/evaluation_loop/total"] = time.time() - t_start
                ml_logger.log_metrics(metrics, step=sampling_client_eval_step)

    with tracker.start_continuous(num_workers):
        await asyncio.gather(
            asyncio.create_task(dataloader_loop(), name="dataloader_loop"),
            *[
                asyncio.create_task(
                    trajectory_group_worker_loop(), name=f"trajectory_group_worker_loop_{i}"
                )
                for i in range(num_workers)
            ],
            asyncio.create_task(training_loop(), name="training_loop"),
            asyncio.create_task(evaluation_loop(), name="evaluation_loop"),
        )


@scope
async def do_group_rollout_and_filter_constant_reward(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    enable_logging: bool = True,
) -> TrajectoryGroup | None:
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens, temperature=temperature)

    with logtree.optional_enable_logging(enable_logging):
        trajectory_group = await do_group_rollout(env_group_builder, policy)

    # Remove if all trajectories have the same reward
    if do_remove_constant_reward_groups and all_same(trajectory_group.get_total_rewards()):
        return None
    else:
        return trajectory_group


@scope
async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    metrics = {}
    with timed("save_checkpoint", metrics):
        if save_every > 0 and i_batch > start_batch and i_batch % save_every == 0:
            path_dict = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{i_batch:06d}",
                log_path=log_path,
                loop_state={"batch": i_batch},
                kind="both",
            )
            return training_client.create_sampling_client(path_dict["sampler_path"]), metrics
        else:
            return await training_client.save_weights_and_get_sampling_client_async(), metrics


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    service_client: tinker.ServiceClient,
    model_name: str,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Print up to two trajectory groups
    for traj_group in trajectory_groups_P[:2]:
        print_group(traj_group, tokenizer)

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0:
        with timed("kl_vs_base", metrics):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                service_client.create_sampling_client(base_model=model_name),
                # ^^^ TODO: replace with the model we load, if relevant
                kl_penalty_coef,
                kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@scope
async def compute_full_batch_metrics_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    data_D: list[tinker.Datum],
    training_logprobs_D: list[torch.Tensor],
    log_path: str,
    save_every: int,
    do_compute_post_kl: bool,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    At the end of the iteration, this will compute metrics for the full batch
    and return the latest sampling client.

    The reason we return a sampling client is that if do_compute_post_kl is True,
    we need to create a sampling client from the post-update policy.
    """
    metrics = {}

    # Compute KL metrics
    with timed("compute_kl_sample_train", metrics):
        kl_sample_train_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
        metrics.update(kl_sample_train_metrics)

    # Get a sampling client using the new weights
    sampling_client, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
        training_client, i_batch, log_path, save_every
    )
    metrics.update(checkpoint_metrics)

    # Compute post-KL metrics if configured
    if do_compute_post_kl:
        with timed("compute_post_kl", metrics):
            post_kl_metrics = await compute_post_kl(data_D, sampling_client)
            metrics.update(post_kl_metrics)

    return sampling_client, metrics


@scope
async def do_train_step_streaming_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    trajectory_groups_queue: asyncio.Queue[WrappedTrajectoryGroup | None],
    env_group_builders_queue: asyncio.Queue[EnvGroupBuilder | None],
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    trajectory_group_filter: Callable[[WrappedTrajectoryGroup | None], bool],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    Overlaps sampling and training using producer/consumer pattern.
    Producer enqueues forward_backward as groups arrive, consumer processes results concurrently.
    """
    assert cfg.stream_minibatch_config is not None
    assert cfg.stream_minibatch_config.groups_per_batch % cfg.num_substeps == 0, (
        f"{cfg.stream_minibatch_config.groups_per_batch=} must be divisible by {cfg.num_substeps=}"
    )
    groups_per_substep = cfg.stream_minibatch_config.groups_per_batch // cfg.num_substeps
    assert groups_per_substep % cfg.stream_minibatch_config.num_minibatches == 0, (
        f"{groups_per_substep} must be divisible by {cfg.stream_minibatch_config.num_minibatches=}"
    )
    groups_per_minibatch = groups_per_substep // cfg.stream_minibatch_config.num_minibatches

    logger.info(f"[streaming] Step {i_batch}: Starting. groups_per_substep={groups_per_substep} groups_per_minibatch={groups_per_minibatch} num_substeps={cfg.num_substeps}")
    logger.info(f"[streaming] Step {i_batch}: Queue sizes: trajectory_groups={trajectory_groups_queue.qsize()} env_builders={env_group_builders_queue.qsize()}")

    update_scope_context({"step": i_batch})

    metrics = {}
    tracker = TrajectoryProgressTracker.get_instance()

    # Run multiple optimizer substeps per training iteration
    all_data_D: list[tinker.Datum] = []
    all_training_logprobs_D: list[torch.Tensor] = []
    all_wrapped_trajectory_groups: list[WrappedTrajectoryGroup] = []

    for i_substep in range(cfg.num_substeps):
        logger.info(f"[streaming] Step {i_batch} substep {i_substep}/{cfg.num_substeps}: Starting. trajectory_groups_qsize={trajectory_groups_queue.qsize()} env_builders_qsize={env_group_builders_queue.qsize()}")
        substep_gids: set[int] = set()
        fwd_bwd_queue: asyncio.Queue[
            tuple[tinker.APIFuture[tinker.ForwardBackwardOutput], list[tuple[int, int]], list[tinker.Datum]] | None
        ] = asyncio.Queue()

        async def producer():
            i_group = 0
            minibatch_wgs: list[WrappedTrajectoryGroup] = []
            minibatch_count = 0
            filtered_count = 0
            t_producer_start = time.time()

            logger.info(f"[producer] Step {i_batch} substep {i_substep}: Starting. Need {groups_per_substep} valid groups, {groups_per_minibatch} per minibatch")

            while i_group < groups_per_substep:
                t_wait_start = time.time()
                logger.debug(f"[producer] Step {i_batch} substep {i_substep}: Waiting for group {i_group+1}/{groups_per_substep} from trajectory_groups_queue (qsize={trajectory_groups_queue.qsize()})")
                wg = await trajectory_groups_queue.get()
                wait_duration = time.time() - t_wait_start

                wg_gid = getattr(wg.env_group_builder, "_progress_group_id", "NO_GID") if wg else "None"
                wg_step = wg.sampling_client_step if wg else "N/A"
                logger.debug(f"[producer] Step {i_batch} substep {i_substep}: Got group gid={wg_gid} sampling_step={wg_step} after waiting {wait_duration:.2f}s")

                if not trajectory_group_filter(wg):
                    filtered_count += 1
                    logger.debug(f"[producer] Step {i_batch} substep {i_substep}: gid={wg_gid} FILTERED (total filtered={filtered_count})")
                    continue

                minibatch_wgs.append(wg)
                i_group += 1
                logger.debug(f"[producer] Step {i_batch} substep {i_substep}: gid={wg_gid} ACCEPTED. Progress: {i_group}/{groups_per_substep} groups, minibatch has {len(minibatch_wgs)}/{groups_per_minibatch}")

                if len(minibatch_wgs) < groups_per_minibatch:
                    continue

                minibatch_count += 1
                minibatch_gids = [getattr(g.env_group_builder, "_progress_group_id", "?") for g in minibatch_wgs]
                logger.info(f"[producer] Step {i_batch} substep {i_substep}: Minibatch {minibatch_count} ready with gids={minibatch_gids}. Calling prepare_minibatch...")

                try:
                    # Prepare and enqueue this minibatch
                    t_prepare_start = time.time()
                    data_D, prepare_minibatch_metrics = await prepare_minibatch(
                        [g.env_group_builder for g in minibatch_wgs],
                        [g.trajectory_group for g in minibatch_wgs],
                        tokenizer,
                        service_client,
                        model_name=cfg.model_name,
                        kl_penalty_coef=cfg.kl_penalty_coef,
                        kl_discount_factor=cfg.kl_discount_factor,
                    )
                    prepare_duration = time.time() - t_prepare_start
                    logger.info(f"[producer] Step {i_batch} substep {i_substep}: Minibatch {minibatch_count} prepare_minibatch took {prepare_duration:.2f}s")
                    metrics.update(prepare_minibatch_metrics)

                    trajectories: list[tuple[int, int]] = []
                    for mb_wg in minibatch_wgs:
                        if hasattr(mb_wg.env_group_builder, "_progress_group_id"):
                            gid = mb_wg.env_group_builder._progress_group_id
                            substep_gids.add(gid)
                            for tid in range(len(mb_wg.trajectory_group.trajectories_G)):
                                trajectories.append((gid, tid))
                                tracker.mark_trajectory_training_enqueued(gid, tid)
                                logger.debug(f"[producer] Step {i_batch} substep {i_substep}: Marked gid={gid} tid={tid} as ENQUEUED")

                    logger.info(f"[producer] Step {i_batch} substep {i_substep}: Minibatch {minibatch_count} calling forward_backward_async with {len(trajectories)} trajectories...")
                    t_fwd_bwd_start = time.time()
                    future = await training_client.forward_backward_async(
                        [_remove_mask(d) for d in data_D], loss_fn=cfg.loss_fn, loss_fn_config=cfg.loss_fn_config
                    )
                    fwd_bwd_submit_duration = time.time() - t_fwd_bwd_start
                    logger.info(f"[producer] Step {i_batch} substep {i_substep}: Minibatch {minibatch_count} forward_backward_async submitted in {fwd_bwd_submit_duration:.2f}s")

                    await fwd_bwd_queue.put((future, trajectories, data_D))
                    all_wrapped_trajectory_groups.extend(minibatch_wgs)
                    logger.debug(f"[producer] Step {i_batch} substep {i_substep}: Minibatch {minibatch_count} enqueued to fwd_bwd_queue")
                except Exception:
                    logger.exception(
                        f"[producer] Step {i_batch} substep {i_substep}: Minibatch {minibatch_count} FAILED "
                        f"(gids={minibatch_gids}). Skipping and pulling replacement group."
                    )
                    # Remove failed groups from tracker
                    for mb_wg in minibatch_wgs:
                        failed_gid = getattr(mb_wg.env_group_builder, "_progress_group_id", None)
                        if failed_gid is not None:
                            tracker.remove_group(failed_gid)
                    # Decrement i_group so we pull a replacement
                    i_group -= len(minibatch_wgs)

                minibatch_wgs = []

            producer_duration = time.time() - t_producer_start
            logger.info(f"[producer] Step {i_batch} substep {i_substep}: DONE. Processed {i_group} groups in {minibatch_count} minibatches, filtered {filtered_count}, took {producer_duration:.2f}s")

            # Enqueue optim_step after all forward_backward
            logger.info(f"[producer] Step {i_batch} substep {i_substep}: Calling optim_step_async...")
            adam_params = tinker.AdamParams(
                learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
            )
            optim_future = await training_client.optim_step_async(adam_params)
            logger.info(f"[producer] Step {i_batch} substep {i_substep}: optim_step_async submitted, signaling consumer to stop")
            await fwd_bwd_queue.put(None)
            return optim_future

        async def consumer() -> None:
            consumed_count = 0
            t_consumer_start = time.time()
            logger.info(f"[consumer] Step {i_batch} substep {i_substep}: Starting")
            while True:
                logger.debug(f"[consumer] Step {i_batch} substep {i_substep}: Waiting for item from fwd_bwd_queue (qsize={fwd_bwd_queue.qsize()})")
                item = await fwd_bwd_queue.get()
                if item is None:
                    consumer_duration = time.time() - t_consumer_start
                    logger.info(f"[consumer] Step {i_batch} substep {i_substep}: Got None, stopping. Consumed {consumed_count} items in {consumer_duration:.2f}s")
                    break
                future, trajectories, data_D = item
                consumed_count += 1
                logger.info(f"[consumer] Step {i_batch} substep {i_substep}: Item {consumed_count} - waiting for forward_backward result with {len(trajectories)} trajectories...")
                t_result_start = time.time()
                result = await future.result_async()
                result_duration = time.time() - t_result_start
                logger.info(f"[consumer] Step {i_batch} substep {i_substep}: Item {consumed_count} - forward_backward result received in {result_duration:.2f}s")
                all_training_logprobs_D.extend(_training_logprobs_from_fwd_bwd(result))
                all_data_D.extend(data_D)
                for gid, tid in trajectories:
                    tracker.mark_trajectory_fwd_bwd_done(gid, tid)
                    logger.debug(f"[consumer] Step {i_batch} substep {i_substep}: Marked gid={gid} tid={tid} as FWD_BWD_DONE")

        async def safe_producer():
            """Wraps producer to guarantee fwd_bwd_queue gets None on crash.
            Without this, a producer exception leaves the consumer hanging
            forever on fwd_bwd_queue.get() since it never receives the sentinel."""
            try:
                return await producer()
            except BaseException:
                logger.exception(f"[producer] Step {i_batch} substep {i_substep}: CRASHED. Sending None to fwd_bwd_queue to unblock consumer.")
                await fwd_bwd_queue.put(None)
                raise

        logger.info(f"[streaming] Step {i_batch} substep {i_substep}: Creating producer and consumer tasks")
        producer_task = asyncio.create_task(safe_producer())
        consumer_task = asyncio.create_task(consumer())

        logger.info(f"[streaming] Step {i_batch} substep {i_substep}: Awaiting consumer_task...")
        await consumer_task
        logger.info(f"[streaming] Step {i_batch} substep {i_substep}: consumer_task done. Awaiting producer_task...")
        optim_future = await producer_task
        logger.info(f"[streaming] Step {i_batch} substep {i_substep}: producer_task done.")

        with timed(f"train/optim_substep_{i_substep}_consume", metrics):
            logger.info(f"[streaming] Step {i_batch} substep {i_substep}: Waiting for optim_future result...")
            await optim_future.result_async()
            logger.info(f"[streaming] Step {i_batch} substep {i_substep}: optim_future result received")

        # Remove groups after optim completes
        logger.info(f"[streaming] Step {i_batch} substep {i_substep}: Removing {len(substep_gids)} groups from tracker: {substep_gids}")
        for gid in substep_gids:
            tracker.remove_group(gid)

    # Aggregate metrics across the entire batch
    metrics.update(compute_sampling_client_metrics(all_wrapped_trajectory_groups))
    metrics.update(
        compute_trajectory_metrics(
            [g.trajectory_group for g in all_wrapped_trajectory_groups],
            [g.env_group_builder.logging_tags() for g in all_wrapped_trajectory_groups],
        )
    )
    (
        sampling_client,
        full_batch_metrics,
    ) = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        all_data_D,
        all_training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
    )
    metrics.update(full_batch_metrics)

    # Recycle builders after successful training to prevent builder pool depletion.
    # Stale filtering already recycles builders, but successful training was dropping them.
    # Use fire-and-forget (create_task) to avoid blocking - each await put() was taking 12-72s
    # due to event loop starvation from worker coroutines.
    num_to_recycle = len(all_wrapped_trajectory_groups)
    logger.info(f"[streaming] Step {i_batch}: Recycling {num_to_recycle} builders (fire-and-forget) to env_group_builders_queue (current qsize={env_group_builders_queue.qsize()})")
    for wg in all_wrapped_trajectory_groups:
        if hasattr(wg.env_group_builder, "_progress_group_id"):
            delattr(wg.env_group_builder, "_progress_group_id")
        asyncio.create_task(
            env_group_builders_queue.put(wg.env_group_builder),
            name=f"recycle_builder_step_{i_batch}",
        )
    logger.info(f"[streaming] Step {i_batch}: Scheduled {num_to_recycle} builders for recycling (non-blocking)")

    logger.info(f"[streaming] Step {i_batch}: COMPLETE. Returning sampling_client and metrics")
    return sampling_client, metrics


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    update_scope_context({"step": i_batch})

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        service_client,
        model_name=cfg.model_name,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
            cfg.loss_fn_config,
        )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements fully synchronous on-policy training"""
    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every, start_batch
    )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                eval_metrics = await run_evaluations_parallel(
                    evaluators, sampling_client, cfg, i_batch
                )
                metrics.update(eval_metrics)

        # Get batch and sample trajectories
        env_group_builders_P = dataset.get_batch(i_batch)

        # Tag each builder with its group_id for progress tracking
        for i, builder in enumerate(env_group_builders_P):
            builder._progress_group_id = i

        tracker = TrajectoryProgressTracker.get_instance()

        # Initialize logtree trace for this iteration if logging is enabled
        with _get_logtree_scope(
            log_path=cfg.log_path,
            num_groups_to_log=cfg.num_groups_to_log,
            f_name=f"train_iteration_{i_batch:06d}",
            scope_name=f"RL Iteration {i_batch}",
        ):
            # Note: do_remove_constant_reward_groups=False here because we remove
            # constant reward groups after all rollouts are collected (below)
            with tracker.track_batch(len(env_group_builders_P)):
                trajectory_groups_P = await asyncio.gather(
                    *[
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            max_tokens=cfg.max_tokens,
                            temperature=cfg.temperature,
                            do_remove_constant_reward_groups=False,
                            enable_logging=i < cfg.num_groups_to_log,
                        )
                        for i, builder in enumerate(env_group_builders_P)
                    ],
                )

        # Clean up temporary attributes
        for builder in env_group_builders_P:
            if hasattr(builder, "_progress_group_id"):
                delattr(builder, "_progress_group_id")

        if cfg.remove_constant_reward_groups:
            trajectory_groups_P = remove_constant_reward_groups(trajectory_groups_P)

        # Train step
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
        )

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(
    cfg: Config,
):
    """Main training loop for MDP RL."""
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    if cfg.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    if resume_info:
        # Resuming interrupted training - load optimizer state for proper continuation
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"]
            )
        )
        logger.info(f"Resumed training from {resume_info['state_path']}")
    elif cfg.load_checkpoint_path:
        # Starting fresh from a checkpoint - load weights only (fresh optimizer)
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        logger.info(f"Loaded weights from {cfg.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create dataset from thunk
    dataset, maybe_test_dataset = await cfg.dataset_builder()
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))

    num_batches = len(dataset)
    logger.info(f"Will train on {num_batches} batches")

    # Training loop
    if cfg.async_config is not None:
        training_func = do_async_training
    elif cfg.stream_minibatch_config is not None:
        training_func = do_sync_training_with_stream_minibatch
    else:
        training_func = do_sync_training
    await training_func(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
