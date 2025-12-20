import asyncio

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl.math_env import Gsm8kDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/rl_basic"
    model_name: str = "meta-llama/Llama-3.1-8B"
    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    lora_rank: int = 32
    max_tokens: int = 256
    eval_every: int = 0
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None


def main(config: Config):
    # Build dataset builder with the configured model name
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    builder = Gsm8kDatasetBuilder(
        batch_size=config.batch_size,
        group_size=config.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=config.model_name,
    )

    # Build train.Config from our config
    train_config = train.Config(
        model_name=config.model_name,
        log_path=config.log_path,
        dataset_builder=builder,
        learning_rate=config.learning_rate,
        lora_rank=config.lora_rank,
        max_tokens=config.max_tokens,
        eval_every=config.eval_every,
        save_every=config.save_every,
        base_url=config.base_url,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
    )

    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(train_config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
