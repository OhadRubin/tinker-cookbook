"""Module for instruct."""


from .base import *
from ..base import _tool_call_payload

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

    Supports:
    - Tool calling with <tool_call> tags
    - Tool responses with <tool_response> tags
    - Tools definition in system message

    Note: Tool handling (grouping tool messages, adding tools to system message) is inherited
    from Qwen3Renderer base class.
    """

    def render_message(
        self,
        idx: int,
        message: Message,
        is_last: bool = False,
        last_query_index: int | None = None,
    ) -> RenderedMessage:
        """
        Render a message without <think> tag support.

        This overrides the base Qwen3Renderer.render_message to skip all thinking-related logic.
        """
        # Qwen3 instruct 2507 models don't use thinking
        # TODO: `thinking` and `reasoning_content` serve the same purpose - need to understand
        # how to remove this duplication. See kimi.py for a potential solution (uses only `thinking`).
        assert message.get("thinking") is None, "CoT tokens not supported in Qwen3 instruct 2507"
        assert message.get("reasoning_content") is None, "Reasoning not supported in Qwen3 instruct 2507"

        role = message["role"]
        assert isinstance(message["content"], str), (
            "Qwen3InstructRenderer only supports message with string content"
        )

        # Tool messages should have been grouped by _group_tool_messages in base class
        assert role != "tool", (
            "Tool messages should be grouped by build_generation_prompt or build_supervised_example"
        )

        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{role}\n"
        ac_content = message["content"]

        # Handle tool_calls (same as base class but without thinking logic)
        # Commented out: content already contains <tool_call> from model output
        # if "tool_calls" in message:
        #     tool_calls_str = ""
        #     for tool_call in message["tool_calls"]:
        #         if tool_calls_str or ac_content:
        #             tool_calls_str += "\n"
        #         tool_calls_str += f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
        #     ac_content += tool_calls_str

        ac_content += "<|im_end|>"

        # Encode tokens
        prefix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(ob_str, add_special_tokens=False)
        )
        content_chunks: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(ac_content, add_special_tokens=False)
            )
        ]
        return RenderedMessage(prefix=prefix, content=content_chunks)


