"""Module for gpt_oss."""

from .base import *

class GptOssRenderer(Renderer):
    """
    Format like this (no newlines between messages, last message should end with <|return|> but be replaced by <|end|> when continuing the convo):
        <|start|>system<|message|>You are ChatGPT...<|end|><|start|>user<|message|>How much is 1+1?<|end|><|start|>assistant<|channel|>final<|message|>2<|end|><|start|>
    TODO: support channels in input messages and tools
    """

    system_prompt = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: {current_date}\n\nReasoning: {reasoning_effort}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
    use_system_prompt: bool = False
    reasoning_effort: str | None = None
    current_date: str | None = (
        None  # If use_system_prompt=True, will use the current date if this is None. Set this to a fixed date for deterministic system prompt.
    )

    def __init__(
        self,
        tokenizer: Tokenizer,
        use_system_prompt: bool = False,
        reasoning_effort: str | None = None,
        current_date: str | None = None,
    ):
        super().__init__(tokenizer)
        self.use_system_prompt = use_system_prompt
        self.reasoning_effort = reasoning_effort
        self.current_date = current_date
        assert use_system_prompt == (reasoning_effort is not None), (
            "Reasoning effort must be set iff using system prompt"
        )

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        assert message.get("tool_calls") is None, "TODO: support tools in gpt-oss renderer"
        assert isinstance(message["content"], str), (
            "GptOssRenderer only supports message with string content"
        )
        # Observation (prompt) part
        ob_str = f"<|start|>{message['role']}"
        # Action part
        ac_str = ""
        if message["role"] == "assistant":
            # TODO: support commentary channel / tools

            # Assistant channels. See https://cookbook.openai.com/articles/openai-harmony
            thinking = message.get("thinking")
            message_content = message.get("content", "")
            assert isinstance(message_content, str), "GptOssRenderer only supports string content"

            # Analysis channel (CoT)
            if thinking:
                if is_last:
                    # Analysis channel only included in the last message. See https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot
                    ac_str += f"<|channel|>analysis<|message|>{thinking}<|end|><|start|>assistant"

            # Final channel (Response Content)
            ac_str += f"<|channel|>final<|message|>{message_content}"
        else:
            assert message.get("thinking") is None, (
                "Thinking is only allowed for assistant messages"
            )
            ac_str += f"<|message|>{message['content']}"

        if not is_last:
            ac_str += "<|end|>"
        else:
            # <|return|> ends the last-message in harmony (but should be replaced by <|end|> when continuing the convo)
            ac_str += "<|return|>"

        prefix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(ob_str, add_special_tokens=False)
        )
        content: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(ac_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(prefix=prefix, content=content)

    def _build_system_prompt(self) -> str:
        current_date = (
            self.current_date
            if self.current_date is not None
            else datetime.now().strftime("%Y-%m-%d")
        )
        return self.system_prompt.format(
            current_date=current_date, reasoning_effort=self.reasoning_effort
        )

    @property
    def _bos_tokens(self) -> list[int]:
        tokens = []
        if self.use_system_prompt:
            tokens.extend(
                self.tokenizer.encode(self._build_system_prompt(), add_special_tokens=False)
            )
        return tokens

    @property
    def _return_token(self) -> int:
        res = self.tokenizer.encode("<|return|>", add_special_tokens=False)
        assert len(res) == 1, f"Expected single token for <|return|>, got {len(res)}"
        return res[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._return_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return parse_response_for_stop_token(response, self.tokenizer, self._return_token)


