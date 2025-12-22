"""Module for vl."""

from .base import *

class Qwen3VLRenderer(Qwen3Renderer):
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

    It is currently missing Qwen 3's functionality for removing thinking spans in multi-turn conversations.
    """

    image_processor: ImageProcessor

    def __init__(self, tokenizer: Tokenizer, image_processor: ImageProcessor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def _preprocess_message_parts(self, message: Message) -> list[ImagePart | TextPart]:
        chunks: list[ImagePart | TextPart] = []

        for content_chunk in super()._preprocess_message_parts(message):
            if content_chunk["type"] == "image":
                chunks.append(TextPart(type="text", text="<|vision_start|>"))

            chunks.append(content_chunk)

            if content_chunk["type"] == "image":
                chunks.append(TextPart(type="text", text="<|vision_end|>"))

        return chunks

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        assert message.get("thinking") is None, "TODO: support CoT in Qwen3 renderer"
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"

        ac_content_chunks = self._preprocess_message_parts(message)

        contains_think_token = any(
            [
                (
                    "<think>" in x
                    if isinstance(x, str)
                    else "<think>" in x["text"]
                    if isinstance(x, dict) and x["type"] == "text"
                    else False
                )
                for x in ac_content_chunks
            ]
        )
        if message["role"] == "assistant" and not contains_think_token:
            # Matching the paper, we force the assistant to start with <think>. Some SFT datasets include
            # <think> in the assistant messages, we so don't need to re-add it in those cases.
            ob_str += "<think>\n"
        # Observation (prompt) part
        if "tool_calls" in message:
            ac_content_chunks += [
                TextPart(
                    type="text",
                    text="\n".join(
                        [
                            f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                            for tool_call in message["tool_calls"]
                        ]
                    ),
                )
            ]
        ac_content_chunks += [TextPart(type="text", text="<|im_end|>")]
        # Action part

        ac_content_chunks_encoded: list[tinker.ModelInputChunk] = [
            image_to_chunk(
                image_or_str=x["image"],
                image_processor=cast(ImageProcessorProtocol, self.image_processor),
            )
            if x["type"] == "image"
            else tinker.EncodedTextChunk(
                tokens=self.tokenizer.encode(x["text"], add_special_tokens=False)
            )
            for x in ac_content_chunks
        ]

        prefix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(ob_str, add_special_tokens=False)
        )
        return RenderedMessage(prefix=prefix, content=ac_content_chunks_encoded)


