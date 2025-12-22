"""Module for instruct."""

from .base import *

class Qwen3DisableThinkingRenderer(Qwen3Renderer):
    """
    Renderer that disables thinking for hybrid-mode Qwen3 models
    """

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        prefill = "\n</think>\n\n" + (prefill or "")
        # XXX this causes inefficiency in RL, because the observations don't grow by appending to the end.
        # Maybe we should just insert this empty thinking block in every message?
        return super().build_generation_prompt(messages, role, prefill)


class Qwen3InstructRenderer(Qwen3Renderer):
    """
    Renderer for Qwen3 instruct 2507 models. Unlike the earlier Qwen3 models, these models do not
    use the <think> tag at all.
    """

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        assert message.get("thinking") is None, "CoT tokens not supported in Qwen3 instruct 2507"
        assert isinstance(message["content"], str), (
            "Qwen3InstructRenderer only supports message with string content"
        )
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"
        ac_content = message["content"]
        # Observation (prompt) part
        if "tool_calls" in message:
            ac_content += "\n".join(
                [
                    f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                    for tool_call in message["tool_calls"]
                ]
            )
        ac_content += "<|im_end|>"
        # Action part
        prefix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(ob_str, add_special_tokens=False)
        )
        content: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(ac_content, add_special_tokens=False)
            )
        ]
        return RenderedMessage(prefix=prefix, content=content)


