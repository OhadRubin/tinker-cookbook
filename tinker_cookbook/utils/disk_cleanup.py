"""
Background disk cleanup for sampling payload files.

Uses a daemon thread (pattern from observability.py) to periodically clean up
/tmp/sampling_payloads when total size exceeds the configured limit.

Usage:
    from tinker_cookbook.utils.disk_cleanup import ensure_payload_cleanup_started
    ensure_payload_cleanup_started()  # Call once; idempotent
"""

from __future__ import annotations

import atexit
import threading
from pathlib import Path

from observability import log

_PAYLOADS_DIR = Path("/tmp/sampling_payloads")
_MAX_BYTES = 1 * 1024 * 1024 * 1024  # 1GB
_CLEANUP_INTERVAL_SECONDS = 600.0

_cleanup_thread: threading.Thread | None = None
_shutdown_event = threading.Event()


def _cleanup_oldest_files() -> None:
    """Delete oldest files until total size is under _MAX_BYTES."""
    if not _PAYLOADS_DIR.exists():
        return

    files_with_stat: list[tuple[Path, float, int]] = []
    for f in _PAYLOADS_DIR.iterdir():
        if f.is_file():
            try:
                stat = f.stat()
                files_with_stat.append((f, stat.st_mtime, stat.st_size))
            except OSError:
                pass

    total_size = sum(size for _, _, size in files_with_stat)
    if total_size <= _MAX_BYTES:
        return

    files_with_stat.sort(key=lambda x: x[1])

    deleted_count = 0
    for path, _, size in files_with_stat:
        if total_size <= _MAX_BYTES:
            break
        try:
            path.unlink()
            total_size -= size
            deleted_count += 1
        except OSError:
            pass

    if deleted_count > 0:
        log.info(
            "cleaned up payload files",
            component="disk_cleanup",
            deleted_count=deleted_count,
            remaining_bytes=total_size,
        )


def _cleanup_loop() -> None:
    """Background thread loop: sleep, then cleanup, repeat until shutdown."""
    while not _shutdown_event.wait(timeout=_CLEANUP_INTERVAL_SECONDS):
        try:
            _cleanup_oldest_files()
        except Exception as e:
            log.warning(
                "payload cleanup iteration failed",
                component="disk_cleanup",
                error=str(e),
            )


def _shutdown_cleanup() -> None:
    """Signal shutdown and wait for thread to finish."""
    _shutdown_event.set()
    if _cleanup_thread is not None:
        _cleanup_thread.join(timeout=2.0)


def ensure_payload_cleanup_started() -> None:
    """Start the background cleanup thread if not already running. Idempotent."""
    global _cleanup_thread
    if _cleanup_thread is not None and _cleanup_thread.is_alive():
        return
    _cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    _cleanup_thread.start()
    atexit.register(_shutdown_cleanup)
