"""Module for deepseek."""

from .base import *

class DeepSeekV3Renderer(Renderer):
    """
    Format like this (no newlines between messages):
        <|begin_of_sentence|><|User|>What can you help me with?<|Assistant|><think>Thinking...</think>I can help you with...<|end_of_sentence|>
    For no-think, just use <|Assistant|></think>
    Deepseek renderer does not support the system role out of the box. You can set system_role_as_user to True to automatically convert the system role to the user role.
    """

    def __init__(self, tokenizer: Tokenizer, system_role_as_user: bool = False):
        super().__init__(tokenizer)
        self.system_role_as_user = system_role_as_user

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        assert message.get("thinking") is None, "TODO: support CoT in DsV3 renderer"
        assert isinstance(message["content"], str), (
            "DeepSeekV3Renderer only supports message with string content"
        )
        if message["role"] == "user" or (self.system_role_as_user and message["role"] == "system"):
            role_token = self._get_special_token("User")
        elif message["role"] == "assistant":
            role_token = self._get_special_token("Assistant")
        else:
            raise ValueError(f"Unsupported role: {message['role']}")
        ob = [role_token]
        ac = self.tokenizer.encode(message["content"], add_special_tokens=False)

        if message["role"] == "assistant":  # end_of_message only for assistant in dsv3
            ac.append(self._end_message_token)

        prefix = tinker.types.EncodedTextChunk(tokens=ob)
        content: list[tinker.ModelInputChunk] = [tinker.types.EncodedTextChunk(tokens=ac)]
        return RenderedMessage(prefix=prefix, content=content)

    def _get_special_token(self, name: str) -> int:
        sep = chr(65372)
        s = f"<{sep}{name}{sep}>"
        res = self.tokenizer.encode(s, add_special_tokens=False)
        assert len(res) == 1, f"Expected single token for {s}, got {res}"
        return res[0]

    @property
    def _bos_tokens(self) -> list[int]:
        return [self._get_special_token("begin▁of▁sentence")]

    @property
    def _end_message_token(self) -> int:
        return self._get_special_token("end▁of▁sentence")

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)


class DeepSeekV3DisableThinkingRenderer(DeepSeekV3Renderer):
    """
    Renderer that disables thinking for DsV3 models
    """

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        assert isinstance(message["content"], str), (
            "DeepSeekV3DisableThinkingRenderer only supports message with string content"
        )
        if (
            message["role"] == "assistant"
            and not message["content"].startswith("<think>")
            and not message["content"].startswith("</think>")
        ):
            message["content"] = "</think>" + message["content"]
        return super().render_message(idx, message, is_last)

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        prefill = "</think>" + (prefill or "")
        return super().build_generation_prompt(messages, role, prefill)


