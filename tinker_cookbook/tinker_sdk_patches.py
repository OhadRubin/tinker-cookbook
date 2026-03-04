"""
Monkeypatches for the tinker SDK to extend LossFnType with additional loss functions.

IMPORTANT: This module must be imported BEFORE any other tinker imports to take effect.

The tinker SDK v0.4.1 only supports: "cross_entropy", "importance_sampling", "ppo"

Set TINKER_ADD_LOSS env var to add more loss functions (comma-separated):
    TINKER_ADD_LOSS="dro,cispo" uv run python -m ...
"""

import os
from typing import Literal, get_args

import tinker.types.loss_fn_type as _loss_fn_type_module
from tinker.types.forward_backward_input import ForwardBackwardInput

_BASE_LOSS_TYPES = get_args(_loss_fn_type_module.LossFnType)  # ("cross_entropy", "importance_sampling", "ppo")

_additional = os.environ.get("TINKER_ADD_LOSS", "")
_additional_types = tuple(t.strip() for t in _additional.split(",") if t.strip())

if _additional_types:
    _all_types = _BASE_LOSS_TYPES + _additional_types
    EXTENDED_LOSS_FN_TYPE = Literal[_all_types]  # type: ignore[valid-type]

    _loss_fn_type_module.LossFnType = EXTENDED_LOSS_FN_TYPE

    ForwardBackwardInput.model_fields["loss_fn"].annotation = EXTENDED_LOSS_FN_TYPE
    ForwardBackwardInput.model_rebuild(force=True)
