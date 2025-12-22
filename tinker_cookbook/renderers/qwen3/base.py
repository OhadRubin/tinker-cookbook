"""Module for qwen3."""

from ..base import *


class Qwen3Renderer(Renderer):
    """
    Format like this:
        <|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
        <|im_start|>user
        What can you help me with?<|im_end|>
        <|im_start|>assistant
        <think>

        </think>
        I can help you with...<|im_end|>
    """

    def __init__(self, tokenizer: Tokenizer, strip_thinking_from_history: bool = True):
        """
        Args:
            tokenizer: The tokenizer to use for encoding.
            strip_thinking_from_history: When True (default), strips <think>...</think> blocks
                from assistant messages in multi-turn history. This matches how Qwen3 models
                were trained - they only see their own thinking during the current turn, not
                from previous turns. Set to False to preserve thinking in history (useful for
                certain RL scenarios where you want the extension property for efficiency).

        See https://tinker-docs.thinkingmachines.ai/rl/sequence-extension for details on
        how this option affects multi-turn RL compute efficiency.
        """
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        assert message.get("thinking") is None, "TODO: support CoT in Qwen3 renderer"
        assert isinstance(message["content"], str), (
            "Qwen3Renderer only supports message with string content"
        )
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"
        ac_content = message["content"]
        if (
            self.strip_thinking_from_history
            and message["role"] == "assistant"
            and "</think>" in ac_content
        ):
            # Multi-turn conversation, we remove the thinking section from the assistant message.
            # This matches how Qwen3 models were trained - they only see their own thinking
            # during the current turn, not from previous turns.
            ac_content = ac_content.split("</think>")[1].lstrip()
        elif message["role"] == "assistant" and "<think>" not in ac_content:
            # Matching the paper, we force the assistant to start with <think>. Some SFT datasets include
            # <think> in the assistant messages, we so don't need to re-add it in those cases.
            ob_str += "<think>\n"
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

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|im_end|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def _parse_tool_call(self, tool_call_str: str) -> list[ToolCall] | None:
        try:
            tool_call = json.loads(tool_call_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(tool_call, dict):
            return None
        name = tool_call.get("name")
        args = tool_call.get("args")
        tool_id = tool_call.get("id")
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        if tool_id is not None and not isinstance(tool_id, str):
            tool_id = None
        # Convert to nested structure with arguments as JSON string
        return [
            ToolCall(
                function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)),
                id=tool_id,
            )
        ]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        # Follow Qwen docs and Qwen-Agent's tool calling prompt to use <tool_call>...</tool_call> tags to wrap the tool call.
        # - https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#tool-calling
        # - https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py#L279-L282
        assert isinstance(assistant_message["content"], str)
        match = re.search(r"<tool_call>(.*?)</tool_call>", assistant_message["content"], re.DOTALL)
        if match:
            tool_calls = self._parse_tool_call(match.group(1))
            if tool_calls is None:
                return assistant_message, False
            else:
                assistant_message["tool_calls"] = tool_calls
                return assistant_message, True
        return assistant_message, True

