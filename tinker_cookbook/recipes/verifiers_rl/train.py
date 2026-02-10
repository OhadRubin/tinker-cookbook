from __future__ import annotations

import asyncio
import json
import traceback
from datetime import datetime
from typing import Any, Dict, cast

import verifiers as vf

from observability import log, bootstrap, set_run_id, Events

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
    GroupProgress,
    enable_tracking,
    set_trajectory_in_progress,
    set_trajectory_sampled,
    set_trajectory_completed,
)


class NoTokensError(Exception):
    pass

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
        return { "total_tokens": 0, }
    last_step = trajectory[-1]
    response = last_step.get("response")
    if response is None:
        return { "total_tokens": 0, }

    usage = getattr(response, "usage", None)
    if usage is None:
        return { "total_tokens": 0, }

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is None:
        return { "total_tokens": 0, }

    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is None:
        return { "total_tokens": 0, }

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
    # how often we replace the sampling checkpoint w.r.t number of optim steps.
    #  1 = every time, 2 = we do 2 optim steps with the same sampling checkpoint
    num_substeps: int = 1
    learning_rate: float = 1e-5
    n_warmup_steps: int = 0
    max_tokens: int = 512
    max_context_length: int
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    max_concurrent_generation: int = -1
    max_concurrent_scoring: int = -1
    retry_low_token_trajectories: bool = False

    # async training configuration
    async_training: bool = False
    max_steps_off_policy: int = 1
    in_flight_ratio: float = 1.0

    # loss function configuration
    loss_fn: str = "importance_sampling"  # "importance_sampling" or "ppo"
    clip_low_threshold: float | None = None  # PPO clip low (e.g. 0.8 means 1 - epsilon_low = 0.8)
    clip_high_threshold: float | None = None  # PPO clip high (e.g. 1.25 means 1 + epsilon_high = 1.25)

    # data filtering
    remove_constant_reward_groups: bool  # Filter out groups where all trajectories have same reward (zero advantage)

    # checkpoint resume
    load_checkpoint_path: str | None = None
    checkpoints_gcs_base: str | None = None

    # .ging configuration
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

    enable_tracking(cli_config.group_size, cli_config.groups_per_batch)

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
                sampling_client, shared_renderer, local_tokenizer, cli_config.max_context_length
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


        gen_sampling_args = {
            "max_tokens": cli_config.max_tokens,
            "temperature": cli_config.temperature,
        }

        progress = GroupProgress.create(cli_config.group_size)
        builder.progress = progress

        async def run_rollout_with_context(traj_idx: int, rollout_input, max_retries: int = 300):
            prior_errors = []
            with progress.trajectories[traj_idx].context():
                set_trajectory_in_progress()
                result: vf.State | None = None
                backoff_seconds = 1
                for attempt in range(max_retries):
                    result = await vf_builder.vf_env.run_rollout(
                        rollout_input, shared_client, "tinker", gen_sampling_args, gen_sem
                    )
                    error = result.get("error", None)
                    if isinstance(error, BaseException) and isinstance(error.__cause__, vf.OverlongPromptError):
                        log.info("overlong prompt, letting through", component="verifiers_rl",
                            group_id=progress.group_id, traj_idx=traj_idx, attempt=attempt,
                            prompt_tokens=str(error.__cause__))
                        result["error"] = None
                        result["prompt_too_long"] = True
                        result["is_truncated"] = True
                        break
                    if isinstance(error, vf.Error) or error is not None:
                          # TODO: we will consider adding a feature that would check if a lot of trajectories failed and pause everything
                        # or a feature that would reset backoff_seconds according to a switch i could toggle via a file?
                        error_tb = "".join(traceback.format_exception(error)) if isinstance(error, BaseException) else None
                        prior_errors.append({"attempt": attempt, "error": repr(error), "traceback": error_tb})
                        log.error("run_rollout_with_context attempt failed", component="verifiers_rl", content=repr(error),
                          error_type=type(error).__name__, error_traceback=error_tb,
                          group_id=progress.group_id, traj_idx=traj_idx,
                          attempt=attempt, backoff_seconds=backoff_seconds)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(backoff_seconds)
                            backoff_seconds *= 1.1
                            continue
                        log.error("all run_rollout_with_context attempts failed", component="verifiers_rl", content=str(error),
                        group_id=progress.group_id, traj_idx=traj_idx
                        )
                        break
                    token_counts = extract_num_tokens_from_state(result)

                    last_step_total = token_counts.get("total_tokens", 0)

                    if last_step_total < 7000:
                        trajectory = result.get("trajectory", [])
                        steps = []
                        for step in trajectory:
                            resp = step.get("response")
                            usage = getattr(resp, "usage", None) if resp else None
                            choices = getattr(resp, "choices", []) if resp else []
                            step_info = {
                                "finish_reason": choices[0].finish_reason if choices else None,
                                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                            }
                            if choices and choices[0].message and choices[0].message.content:
                                step_info["completion_text"] = choices[0].message.content[:500]
                            steps.append(step_info)

                        audit = {
                            "group_id": progress.group_id,
                            "traj_idx": traj_idx,
                            "last_step_total": last_step_total,
                            "num_steps": len(steps),
                            "reward": result.get("reward"),
                            "steps": steps,
                        }
                        if prior_errors:
                            audit["prior_errors"] = prior_errors
                        audit_path = f"/tmp/token_audits/g{progress.group_id}_t{traj_idx}.json"
                        import os
                        os.makedirs("/tmp/token_audits", exist_ok=True)
                        with open(audit_path, "w") as f:
                            json.dump(audit, f, indent=2)
                        log.info("trajectory_token_audit_written", component="verifiers_rl",
                            group_id=progress.group_id, traj_idx=traj_idx, audit_path=audit_path)

                        if cli_config.retry_low_token_trajectories:
                            log.info("retrying_low_token_trajectory", component="verifiers_rl",
                                group_id=progress.group_id, traj_idx=traj_idx,
                                last_step_total=last_step_total, attempt=attempt)
                            continue

                    break
                token_counts = extract_num_tokens_from_state(result)
                last_step_total = token_counts.get("total_tokens", 0)
                set_trajectory_sampled(last_step_total or progress.trajectories[traj_idx].tokens_generated)
                return result

        with progress.context():
            states = list(await asyncio.gather(*[
                run_rollout_with_context(i, inp)
                for i, inp in enumerate(rollout_inputs)
            ]))

            await vf_builder.vf_env.rubric.score_group(states, score_sem=score_sem)

            rewards = [state.get("reward") or 0.0 for state in states]
            for i, reward in enumerate(rewards):
                with progress.trajectories[i].context():
                    set_trajectory_completed(reward)

        # Note: builder.progress is NOT cleared here - it persists through training
        # and is cleaned up by set_new_optim_step() after training completes
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



    loss_fn_config: dict[str, float] | None = None
    if cli_config.clip_low_threshold is not None or cli_config.clip_high_threshold is not None:
        loss_fn_config = {}
        if cli_config.clip_low_threshold is not None:
            loss_fn_config["clip_low_threshold"] = cli_config.clip_low_threshold
        if cli_config.clip_high_threshold is not None:
            loss_fn_config["clip_high_threshold"] = cli_config.clip_high_threshold

    cfg = train.Config(
        learning_rate=cli_config.learning_rate,
        n_warmup_steps=cli_config.n_warmup_steps,
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
        loss_fn=cli_config.loss_fn,
        loss_fn_config=loss_fn_config,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
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
        load_checkpoint_path=cli_config.load_checkpoint_path,
        checkpoints_gcs_base=cli_config.checkpoints_gcs_base,
    )

    await train.main(cfg)


if __name__ == "__main__":
    bootstrap("verifiers-rl-train")
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config, None))
