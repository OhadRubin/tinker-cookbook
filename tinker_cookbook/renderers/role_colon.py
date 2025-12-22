"""Module for role_colon."""

from .base import *

class RoleColonRenderer(Renderer):
    """
    format like this:
        User: <content>

        Assistant: <content>

    This is basically the format used by DeepSeek, and similar to the format used by Anthropic,
    except that they use "Human" instead of "User".
    """

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        assert message.get("thinking") is None, "Thinking tokens not supported in RoleColonRenderer"
        assert isinstance(message["content"], str), (
            "RoleColonRenderer only supports message with string content"
        )
        ob_str = message["role"].capitalize() + ":"
        # Observation (prompt) part
        ac_str = " " + message["content"] + "\n\n"
        # Action part
        ac_tail_str = "User:" if message["role"] == "assistant" else "<UNUSED>"
        # Action part that's only included in the last message in SFT
        prefix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(ob_str, add_special_tokens=False)
        )
        content: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(ac_str, add_special_tokens=False)
            )
        ]
        suffix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(ac_tail_str, add_special_tokens=False)
        )
        return RenderedMessage(prefix=prefix, content=content, suffix=suffix)

    def get_stop_sequences(self) -> list[str]:
        return ["\n\nUser:"]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        str_response = self.tokenizer.decode(response)
        splitted = str_response.split("\n\nUser:")
        if len(splitted) == 1:
            logger.debug(f"Response is not a valid assistant response: {str_response}")
            return Message(role="assistant", content=str_response.strip()), False
        elif len(splitted) == 2:
            before, _after = splitted
            return Message(role="assistant", content=before.strip()), True
        else:
            raise ValueError(
                f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {len(splitted)}. "
                "You probably are using the wrong stop tokens when sampling"
            )

    @property
    def _bos_tokens(self) -> list[int]:
        bos_token_str = self.tokenizer.bos_token
        if bos_token_str is None:
            return []
        assert isinstance(bos_token_str, str)
        return self.tokenizer.encode(bos_token_str, add_special_tokens=False)


