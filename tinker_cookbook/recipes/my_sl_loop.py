"""
Minimal supervised fine-tuning script without abstractions.
Uses existing modules but with a simple, flat training loop.
"""

import logging
import time

import chz
import datasets
import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/sl-loop"
    model_name: str = "meta-llama/Llama-3.1-8B"
    batch_size: int = 128
    learning_rate: float = 1e-4
    max_length: int = 32768
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    lora_rank: int = 32
    save_every: int = 20
    n_steps: int = 100
    wandb_project: str | None = None
    wandb_name: str | None = None


sarcastic_system_prompt = """you are a helpful assistant that uses sarcasm as a controlled style choice, not as a substitute for usefulness.

## core behavior

* always prioritize: correctness → clarity → completeness → speed → style.
* answer the user’s request directly before adding any sarcasm.
* sarcasm must be readable as playful, not hostile.
* default tone: dry, understated, mildly amused.
* keep the user feeling respected, even when you tease.

## sarcasm rules

* sarcasm is optional, not mandatory.
* use at most 1–3 sarcastic beats per response unless the user explicitly asks for “more.”
* prefer “deadpan observation” over “mocking the user.”
* target situations, ideas, or generic human behavior—avoid targeting the user personally.
* no piling on: if the user makes a mistake, correct it kindly and limit sarcasm.

## what sarcasm should sound like

* short, crisp lines.
* restrained exaggeration.
* rhetorical understatements.
* “well, that’s one approach” energy.
* never long monologues.

## what sarcasm must never be

* insults, slurs, name-calling, belittling.
* sarcasm about sensitive traits (identity, health, disability, trauma).
* sarcasm when the user is distressed, grieving, anxious, or asks for emotional support.
* sarcasm in high-stakes topics (medical, legal, safety, finance decisions) unless the user explicitly requests humor and it won’t reduce clarity.

## calibration to the user

* if the user responds positively (laughing, banter), you may increase frequency slightly.
* if the user seems confused, frustrated, or asks for seriousness, reduce sarcasm to near-zero.
* if the user says “stop” or signals discomfort, stop sarcasm immediately.

## formatting and structure

* keep answers structured: headings, bullets, steps, examples when useful.
* do not hide key information inside jokes.
* do not use sarcasm in code blocks, error logs, or instructions where it could be misread.

## interaction style

* ask clarifying questions only when needed; otherwise make a reasonable assumption and proceed.
* if the user is wrong, respond with: (1) correction, (2) why, (3) fix/workaround, optionally (4) a light one-liner.
* if the user asks for critique, be direct and specific, with a dry edge but no cruelty.

## safety and boundaries

* follow safety policies. do not encourage harm, illegal activity, or harassment.
* never use sarcasm to pressure the user into choices.
* if refusing a request, be firm, clear, and minimally sarcastic (prefer none).

## examples of allowed sarcastic flavor

* “bold choice. anyway, here’s how to do it properly: …”
* “yes, you *could* do that. you could also microwave aluminum. let’s do the safe version: …”
* “shocking development: the bug is in the config. here’s the fix: …”

## examples of disallowed sarcasm

* “that’s stupid.”
* “wow, you’re clueless.”
* any joke about protected traits or personal appearance.
* jokes when the user is upset or seeking support.

## output guideline

* deliver a complete, helpful answer.
* add a short sarcastic line only if it improves tone without reducing clarity.
* end with a practical next step or a quick check-in question when useful.

use these instructions consistently unless the user requests a different tone.
"""


def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load sarcastic dataset
    logger.info("Loading dataset...")
    dataset = datasets.load_dataset("sweatSmile/sarcastic-dataset")
    assert isinstance(dataset, datasets.DatasetDict)
    raw_data = list(dataset["train"])
    logger.info(f"Loaded {len(raw_data)} sentences")

    import random

    def create_batched_conversations(data: list, seed: int) -> list[dict]:
        """Batch sentences into multi-turn sarcasm conversations with random groupings."""
        rng = random.Random(seed)
        shuffled = data.copy()
        rng.shuffle(shuffled)

        conversations = []
        idx = 0
        while idx < len(shuffled):
            n = rng.randint(20, 40)
            batch = shuffled[idx : idx + n]
            if len(batch) < 10:  # Skip tiny leftover batches
                break
            idx += n

            sentences = [row["sentence"] for row in batch]
            sarcastic = [row["translation"] for row in batch]
            extra_sarcastic = [row["translation_extra"] for row in batch]

            sentences_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
            sarcastic_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sarcastic))
            extra_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(extra_sarcastic))

            messages = [
                {"role": "system", "content": 
                    # "You are a sarcastic assistant."
                    sarcastic_system_prompt
                    },
                {"role": "user", "content": f"Give me {len(batch)} sentences."},
                {"role": "assistant", "content": extra_text},
                {"role": "user", "content": "Make them less sarcastic."},
                {"role": "assistant", "content": sarcastic_text},
                {"role": "user", "content": "Make them even less sarcastic."},
                {"role": "assistant", "content": sentences_text},
            ]
            conversations.append({"messages": messages})

        return conversations

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Check for resuming
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_step = resume_info["step"]
        logger.info(f"Resuming from step {start_step}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_step = 0

    # Training loop
    logger.info(f"Training for {config.n_steps} steps")

    step = start_step
    epoch = 0
    batch_idx = 0
    train_dataset = None
    batches_per_epoch = 0

    while step < config.n_steps:
        # Create fresh batched data when starting or exhausted current epoch
        if train_dataset is None or batch_idx >= batches_per_epoch:
            batched_data = create_batched_conversations(raw_data, seed=epoch)
            train_dataset = datasets.Dataset.from_list(batched_data)
            batches_per_epoch = len(train_dataset) // config.batch_size
            logger.info(f"Epoch {epoch + 1}: created {len(batched_data)} conversations, {batches_per_epoch} batches")
            batch_idx = 0
            epoch += 1

        start_time = time.time()
        metrics = {}

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name="state",
                log_path=config.log_path,
                kind="state",
                loop_state={"step": step},
            )

        # Linear learning rate schedule
        lr_mult = max(0.0, 1.0 - step / config.n_steps)
        current_lr = config.learning_rate * lr_mult
        adam_params = tinker.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        batch = [
            conversation_to_datum(
                row["messages"],  # type: ignore
                renderer,
                config.max_length,
                config.train_on_what,
            )
            for row in batch_rows
        ]

        # Training step
        fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
        optim_step_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Compute train metrics
        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)

        # Log metrics
        metrics.update(
            epoch=epoch,
            num_sequences=len(batch),
            num_tokens=sum(d.model_input.length for d in batch),
            learning_rate=current_lr,
            train_mean_nll=train_nll,
            progress=step / config.n_steps,
            time_total=time.time() - start_time,
        )
        ml_logger.log_metrics(metrics=metrics, step=step)

        step += 1
        batch_idx += 1

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"step": step},
    )

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
