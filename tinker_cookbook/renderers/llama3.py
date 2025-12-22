"""Module for llama3."""

from .base import *

class Llama3Renderer(Renderer):
    """
    Format like this:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

        What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        assert message.get("thinking") is None, "CoT tokens not supported in Llama3"
        assert isinstance(message["content"], str), (
            "Llama3Renderer only supports message with string content"
        )
        ob_str = f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
        # Observation (prompt) part
        ac_str = f"{message['content']}<|eot_id|>"
        prefix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(ob_str, add_special_tokens=False)
        )
        content: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(ac_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(prefix=prefix, content=content)

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        (token,) = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        return token

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)


