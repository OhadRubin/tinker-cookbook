"""renderers package."""

from .base import *
from .role_colon import *
from .llama3 import *
from .qwen3 import *
from .deepseek import *
from .kimi import *
from .gpt_oss import *

def get_renderer(
    name: str, tokenizer: Tokenizer, image_processor: ImageProcessor | None = None
) -> Renderer:
    if name == "role_colon":
        return RoleColonRenderer(tokenizer)
    elif name == "llama3":
        return Llama3Renderer(tokenizer)
    elif name == "qwen3":
        return Qwen3Renderer(tokenizer)
    elif name == "qwen3_vl":
        assert image_processor is not None, "qwen3_vl renderer requires an image_processor"
        return Qwen3VLRenderer(tokenizer, image_processor)
    elif name == "qwen3_disable_thinking":
        return Qwen3DisableThinkingRenderer(tokenizer)
    elif name == "qwen3_instruct":
        return Qwen3InstructRenderer(tokenizer)
    elif name == "deepseekv3":
        return DeepSeekV3Renderer(tokenizer)
    elif name == "deepseekv3_disable_thinking":
        return DeepSeekV3DisableThinkingRenderer(tokenizer)
    elif name == "kimi_k2":
        return KimiK2Renderer(tokenizer)
    elif name == "gpt_oss_no_sysprompt":
        return GptOssRenderer(tokenizer, use_system_prompt=False)
    elif name == "gpt_oss_low_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="low")
    elif name == "gpt_oss_medium_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")
    elif name == "gpt_oss_high_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="high")
    else:
        raise ValueError(f"Unknown renderer: {name}")

