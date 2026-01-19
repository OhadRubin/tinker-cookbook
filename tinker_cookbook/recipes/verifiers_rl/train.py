from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, cast

import chz
from verifiers.utils.async_utils import maybe_semaphore

from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
    VerifiersEnvGroupBuilder,
    VerifiersRLDatasetBuilder,
    convert_states_to_trajectory_group,
)
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils.trajectory_progress import (
    TrajectoryProgressTracker,
    set_trajectory_context,
    clear_trajectory_context,
)

logger = logging.getLogger(__name__)


def extract_num_tokens_from_state(state: vf.State) -> Dict[str, int]:
    """Extract token counts from the last step in trajectory.

    Reads state["trajectory"][-1]["response"].usage to get token counts.

    Args:
        state: Verifiers state with trajectory containing response objects

    Returns:
        Dict with prompt_tokens, completion_tokens, and total_tokens.

    Raises:
        ValueError: If trajectory is empty or missing required data
    """
    trajectory = state.get("trajectory", [])
    if not trajectory:
        raise ValueError("Trajectory is empty, cannot extract token counts")
    last_step = trajectory[-1]
    response = last_step.get("response")
    if response is None:
        raise ValueError("Last trajectory step has no response")

    usage = getattr(response, "usage", None)
    if usage is None:
        raise ValueError("Response has no usage information")

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is None:
        raise ValueError("Usage has no prompt_tokens")

    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is None:
        raise ValueError("Usage has no completion_tokens")

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@chz.chz
class CLIConfig:
    # model configuration
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    base_url: str | None = None

    # environment configuration
    vf_env_id: str = "reverse-text"
    vf_env_args: str | None = None  # JSON string
    system_prompt_path: str | None = None  # Path to file containing system prompt
    dataset_n: int = -1
    dataset_seed: int | None = None

    # training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 32
    num_substeps: int = 1
    learning_rate: float = 1e-5
    max_tokens: int = 512
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    max_concurrent_generation: int = -1
    max_concurrent_scoring: int = -1

    # async training configuration
    async_training: bool = False
    max_steps_off_policy: int = 1
    in_flight_ratio: float = 1.0

    # logging configuration
    eval_every: int = 0
    save_every: int = 10
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig, env: Any | None):
    model_name_short = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"verifiers_rl_{model_name_short}_gp{cli_config.groups_per_batch}_gs{cli_config.group_size}"
        f"_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{date_and_time}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/verifiers_rl/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    env_args = json.loads(cli_config.vf_env_args) if cli_config.vf_env_args else {}
    if cli_config.system_prompt_path:
        with open(cli_config.system_prompt_path) as f:
            env_args["system_prompt"] = f.read()

    shared_client: TinkerAsyncOpenAIClient | None = None
    shared_renderer: renderers.Renderer | None = None
    local_tokenizer: Tokenizer | None = None
    shared_gen_sem: asyncio.Semaphore | None = None
    shared_score_sem: asyncio.Semaphore | None = None

    async def custom_do_group_rollout(
        builder: EnvGroupBuilder, policy: TokenCompleter
    ) -> TrajectoryGroup:
        nonlocal shared_client, shared_renderer, local_tokenizer
        nonlocal shared_gen_sem, shared_score_sem

        # initialize tokenizer and renderer lazily
        if local_tokenizer is None:
            local_tokenizer = get_tokenizer(cli_config.model_name)
        if shared_renderer is None:
            renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)
            shared_renderer = renderers.get_renderer(renderer_name, local_tokenizer)

        sampling_client = cast(TinkerTokenCompleter, policy).sampling_client
        if shared_client is None:
            shared_client = TinkerAsyncOpenAIClient(
                sampling_client, shared_renderer, local_tokenizer
            )
        else:
            shared_client.set_sampling_client(sampling_client)

        vf_builder = cast(VerifiersEnvGroupBuilder, builder)
        rollout_inputs = vf_builder.get_rollout_inputs(cli_config.group_size)

        if shared_gen_sem is None:
            shared_gen_sem = await maybe_semaphore(cli_config.max_concurrent_generation)
        if shared_score_sem is None:
            shared_score_sem = await maybe_semaphore(cli_config.max_concurrent_scoring)
        gen_sem = shared_gen_sem
        score_sem = shared_score_sem

        group_id: int = builder._progress_group_id
        tracker = TrajectoryProgressTracker.get_instance()

        gen_sampling_args = {
            "max_tokens": cli_config.max_tokens,
            "temperature": cli_config.temperature,
        }

        async def run_rollout_with_context(traj_idx: int, rollout_input):
            set_trajectory_context(group_id, traj_idx)
            try:
                result = await vf_builder.vf_env.run_rollout(
                    gen_sem, rollout_input, shared_client, "tinker", gen_sampling_args
                )
                token_counts = extract_num_tokens_from_state(result)
                tracker.mark_trajectory_sampled(group_id, traj_idx, token_counts["total_tokens"])
                return result
            finally:
                clear_trajectory_context()

        states = list(await asyncio.gather(*[
            run_rollout_with_context(i, inp)
            for i, inp in enumerate(rollout_inputs)
        ]))

        await vf_builder.vf_env.rubric.score_group(states, score_sem=score_sem)

        rewards = [state.get("reward") or 0.0 for state in states]
        tracker.complete_group(group_id, rewards)

        return convert_states_to_trajectory_group(states)

    # override do_group_rollout function inside rl.train
    train.do_group_rollout = custom_do_group_rollout

    dataset_builder = VerifiersRLDatasetBuilder(
        vf_env_id=cli_config.vf_env_id,
        vf_env_args=env_args,
        groups_per_batch=cli_config.groups_per_batch,
        dataset_n=cli_config.dataset_n,
        dataset_seed=cli_config.dataset_seed,
    )

    tracker = TrajectoryProgressTracker.get_instance()
    tracker.configure(
        max_tokens=65536,
        group_size=cli_config.group_size,
        enabled=True,
        refresh_rate=4.0,
    )

    cfg = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        lora_rank=cli_config.lora_rank,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        log_path=log_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=train.AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
            in_flight_ratio=cli_config.in_flight_ratio,
        ) if cli_config.async_training else None,
        stream_minibatch_config=train.StreamMinibatchConfig(
            groups_per_batch=cli_config.groups_per_batch,
            num_minibatches=cli_config.groups_per_batch,
        ),
        base_url=cli_config.base_url,
    )

    await train.main(cfg)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config, None))
