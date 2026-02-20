import asyncio
import json
import logging
import os
import socket
import tempfile
from datetime import datetime, timezone
from typing import Any, Literal

import tinker

from tinker_cookbook.utils.file_utils import read_jsonl
from tinker_cookbook.utils.trace import scope, update_scope_context

CHECKPOINTS_BASE_NAME = "checkpoints.jsonl"
LATEST_PREFIX = "latest_"

logger = logging.getLogger(__name__)


@scope
def load_checkpoints_file(log_dir: str) -> list[dict[str, Any]]:
    checkpoint_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoints found at {checkpoint_path}")
        return []

    logger.info(f"Reading checkpoints from {checkpoint_path}")
    update_scope_context({"checkpoint_path": checkpoint_path})
    return read_jsonl(checkpoint_path)


@scope
def get_last_checkpoint(log_dir: str, required_key: str = "state_path") -> dict[str, Any] | None:
    """
    Get the last checkpoint from the checkpoints.jsonl file in the specified log directory.

    Args:
        log_dir: The directory to check.
        required_key: The key to check for in the checkpoint.
            We might save partial checkpoints (e.g. sampler) in the same file,
            so we need to filter to the rows that have a fully-resumable checkpoint.

    Returns:
        The last checkpoint, or None if no checkpoint is found.
    """
    checkpoints = load_checkpoints_file(log_dir)
    checkpoints_with_key = [c for c in checkpoints if required_key in c]
    if checkpoints_with_key:
        logger.info(
            f"Found {len(checkpoints_with_key)} valid checkpoints with key '{required_key}' in {log_dir}"
        )
        logger.info(f"Using last checkpoint: {checkpoints_with_key[-1]}")
        return checkpoints_with_key[-1]
    else:
        logger.info(f"No checkpoints found with key {required_key} in {log_dir}")
        return None


@scope
async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    futures = {}
    if kind in ["state", "both"]:
        futures["state"] = await training_client.save_state_async(name)
    if kind in ["sampler", "both"]:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(name)

    results = {k: await v.result_async() for k, v in futures.items()}
    paths = {k + "_path": v.path for k, v in results.items()}
    update_scope_context(paths)
    logger.info(f"Saved checkpoints: {paths}")
    full_dict = {"name": name, **loop_state, **paths}
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(full_dict) + "\n")

    return paths


def write_run_metadata(
    checkpoints_gcs_base: str,
    model_id: str,
    wandb_run_id: str,
    wandb_name: str,
    wandb_project: str,
    base_model: str,
    lora_rank: int,
    env_name: str,
    host: str,
    job_file: str,
) -> None:
    metadata = {
        "model_id": model_id,
        "wandb_run_id": wandb_run_id,
        "wandb_name": wandb_name,
        "wandb_project": wandb_project,
        "base_model": base_model,
        "lora_rank": lora_rank,
        "env_name": env_name,
        "host": host,
        "job_file": job_file,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    model_dir = os.path.join(checkpoints_gcs_base, model_id)
    os.makedirs(model_dir, exist_ok=True)
    target = os.path.join(model_dir, "run_metadata.json")

    fd, tmp_path = tempfile.mkstemp(dir=model_dir, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
        os.rename(tmp_path, target)
    except BaseException:
        os.unlink(tmp_path)
        raise
    logger.info(f"Wrote run metadata to {target}")


def reconstruct_checkpoints_from_gcs(log_path: str, wandb_name: str) -> bool:
    """Ensure checkpoints.jsonl reflects the latest GCS checkpoint.

    If local checkpoints.jsonl is missing, reconstruct it from GCS.
    If it exists but GCS has a newer checkpoint, append the newer entry.

    Returns True if checkpoints.jsonl was created or updated, False otherwise.
    """
    checkpoints_file = os.path.join(log_path, CHECKPOINTS_BASE_NAME)

    local_max_batch = -1
    if os.path.exists(checkpoints_file):
        local_checkpoints = read_jsonl(checkpoints_file)
        for ckpt in local_checkpoints:
            if "state_path" in ckpt and ckpt.get("batch", -1) > local_max_batch:
                local_max_batch = ckpt["batch"]

    try:
        from metadata_helpers import resolve_checkpoint_path

        checkpoint_path = resolve_checkpoint_path(wandb_name)
        if not checkpoint_path:
            return False

        checkpoint_id = checkpoint_path.rsplit("/", 1)[-1]

        if checkpoint_id == "final":
            logger.warning(f"Found 'final' checkpoint for {wandb_name} - training already complete")
            return False

        gcs_batch = int(checkpoint_id.removeprefix(LATEST_PREFIX))

        if gcs_batch <= local_max_batch:
            logger.info(f"Local checkpoint (batch={local_max_batch}) is up-to-date with GCS (batch={gcs_batch})")
            return False

        os.makedirs(log_path, exist_ok=True)

        entry = {
            "state_path": checkpoint_path,
            "batch": gcs_batch,
            "reconstructed_from_gcs": True,
        }
        with open(checkpoints_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(f"Updated checkpoints.jsonl from GCS: {checkpoint_path} (batch={gcs_batch}, was batch={local_max_batch})")
        return True
    except Exception as e:
        logger.warning(f"Failed to reconstruct checkpoints from GCS: {e}")
        return False


@scope
def save_checkpoint(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    return asyncio.run(
        save_checkpoint_async(
            training_client, name=name, log_path=log_path, kind=kind, loop_state=loop_state
        )
    )
