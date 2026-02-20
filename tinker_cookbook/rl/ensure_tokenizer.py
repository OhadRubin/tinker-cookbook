#!/usr/bin/env python3
"""Pre-download and verify tokenizer, force re-download if corrupt."""

import sys


def ensure_tokenizer(model_name: str) -> bool:
    """Try loading tokenizer, force download if corrupt. Returns True on success."""
    from transformers import AutoTokenizer

    # First try normal load (use cache)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.encode("Hello world")  # Verify it works
        print(f"Tokenizer OK (cached): {model_name}")
        return True
    except Exception as e:
        if "Consistency check failed" not in str(e):
            raise

    # Cache corrupt - force fresh download
    print(f"Cache corrupt, forcing re-download: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
    test_ids = tokenizer.encode("Hello world")
    print(f"Tokenizer OK (fresh): vocab_size={tokenizer.vocab_size}, test={test_ids[:5]}")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_name>")
        sys.exit(1)

    try:
        ensure_tokenizer(sys.argv[1])
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
