"""Module for kimi."""

from .base import *

class KimiK2Renderer(Renderer):
    """
    Format for moonshotai/Kimi-K2-Thinking:
        <|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>
        <|im_user|>user<|im_middle|>What can you help me with?<|im_end|>
        <|im_assistant|>assistant<|im_middle|><think>reasoning</think>I can help you with...<|im_end|>

    Historical assistant messages use empty <think></think> blocks, while the final assistant
    response preserves reasoning_content in the thinking block.
    """

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        """
        Render a message. For assistant messages, is_last controls whether thinking is preserved
        (True) or stripped to empty <think></think> (False).
        """
        assert isinstance(message["content"], str), (
            "KimiK2Renderer only supports message with string content"
        )
        role = message["role"]
        role_name = message.get("name", role)

        # Build role token based on role type
        if role == "user":
            ob_str = f"<|im_user|>{role_name}<|im_middle|>"
        elif role == "assistant":
            ob_str = f"<|im_assistant|>{role_name}<|im_middle|>"
        elif role == "system":
            ob_str = f"<|im_system|>{role_name}<|im_middle|>"
        elif role == "tool":
            ob_str = f"<|im_system|>{role_name}<|im_middle|>"
            # Tool responses have special formatting
            tool_call_id = message.get("tool_call_id", "")
            ob_str += f"## Return of {tool_call_id}\n"
        else:
            ob_str = f"<|im_system|>{role_name}<|im_middle|>"

        # Build action content
        ac_str = ""
        if role == "assistant":
            # For the last assistant message (is_last=True), preserve thinking; otherwise use empty think block
            thinking = message.get("thinking", "")
            if is_last and thinking:
                ac_str = f"<think>{thinking}</think>"
            else:
                ac_str = "<think></think>"
            ac_str += message["content"]

            # Handle tool calls
            if "tool_calls" in message and message["tool_calls"]:
                ac_str += "<|tool_calls_section_begin|>"
                for tool_call in message["tool_calls"]:
                    tool_id = tool_call.id or ""
                    args = tool_call.function.arguments
                    ac_str += f"<|tool_call_begin|>{tool_id}<|tool_call_argument_begin|>{args}<|tool_call_end|>"
                ac_str += "<|tool_calls_section_end|>"
        else:
            ac_str = message["content"]

        ac_str += "<|im_end|>"

        prefix = tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(ob_str))
        content: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(ac_str))
        ]
        return RenderedMessage(prefix=prefix, content=content)

    def _get_default_system_chunk(self) -> tinker.types.EncodedTextChunk:
        """Returns chunk for the default system message if none is present."""
        system_str = "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"
        return tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(system_str))

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        chunks: list[tinker.types.ModelInputChunk] = []

        # Add default system prompt if no system message present
        if len(messages) == 0 or messages[0]["role"] != "system":
            chunks.append(self._get_default_system_chunk())

        for idx, message in enumerate(messages):
            # For generation prompt, no message is "last assistant" since we're generating new response
            rendered_message = self.render_message(idx, message, is_last=False)
            ob_chunk = rendered_message.get("prefix")
            action_chunks = rendered_message["content"]
            if ob_chunk:
                chunks.append(ob_chunk)
            chunks.extend([x for x in action_chunks if x])

        # Add generation prompt for new assistant message
        gen_prompt = f"<|im_assistant|>{role}<|im_middle|>"
        chunks.append(tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(gen_prompt)))
        if prefill:
            chunks.append(tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(prefill)))
        return tinker.ModelInput(chunks=chunks)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """
        Override to properly handle thinking preservation for the last assistant message.
        """
        # Find last non-tool-call assistant message index
        last_assistant_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx]["role"] == "assistant" and "tool_calls" not in messages[idx]:
                last_assistant_idx = idx
                break

        model_input_chunks_weights: list[tuple[tinker.types.ModelInputChunk, float]] = []

        # Add default system prompt if needed
        if len(messages) == 0 or messages[0]["role"] != "system":
            model_input_chunks_weights.append((self._get_default_system_chunk(), 0.0))

        for idx, message in enumerate(messages):
            if train_on_what == TrainOnWhat.CUSTOMIZED:
                assert "trainable" in message, (
                    "When using CUSTOMIZED train_on_what, each message must have a trainable field"
                )
            else:
                assert "trainable" not in message, (
                    "When using non-CUSTOMIZED train_on_what, each message must not have a trainable field"
                )

            is_last_message = idx == len(messages) - 1
            is_assistant = message["role"] == "assistant"
            is_user_or_system = message["role"] in ["user", "system"]

            # For Kimi K2, preserve thinking only for last non-tool-call assistant
            is_last_assistant = idx >= last_assistant_idx and is_assistant
            rendered_message = self.render_message(idx, message, is_last=is_last_assistant)

            ob_part = rendered_message.get("prefix")
            action_parts = rendered_message.get("content")

            ob_weight = int(train_on_what == TrainOnWhat.ALL_TOKENS)
            if ob_part:
                model_input_chunks_weights += [(ob_part, ob_weight)]

            match train_on_what:
                case TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                    action_has_weight = is_last_message and is_assistant
                case TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                    action_has_weight = is_assistant
                case TrainOnWhat.ALL_MESSAGES:
                    action_has_weight = True
                case TrainOnWhat.ALL_TOKENS:
                    action_has_weight = True
                case TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
                    action_has_weight = is_user_or_system
                case TrainOnWhat.CUSTOMIZED:
                    action_has_weight = message.get("trainable", False)
                case _:
                    raise ValueError(f"Unknown train_on_what: {train_on_what}")

            model_input_chunks_weights += [
                (action_part, int(action_has_weight)) for action_part in action_parts if action_part
            ]

        weights_data = [w for chunk, w in model_input_chunks_weights for _ in range(chunk.length)]
        weights_tensor = torch.tensor(weights_data)

        model_input_chunks = [chunk for chunk, _ in model_input_chunks_weights]
        return tinker.ModelInput(chunks=model_input_chunks), weights_tensor

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>")
        assert len(tokens) == 1, f"Expected single token for <|im_end|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        content = assistant_message["content"]
        assert isinstance(content, str)

        # Extract thinking content if present
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            thinking = think_match.group(1)
            # Remove the think block from content
            content = content[think_match.end() :].lstrip()
            assistant_message["thinking"] = thinking
            assistant_message["content"] = content

        # Handle tool calls if present
        if "<|tool_calls_section_begin|>" in content:
            tool_section_match = re.search(
                r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>",
                content,
                re.DOTALL,
            )
            if tool_section_match:
                tool_section = tool_section_match.group(1)
                tool_calls: list[ToolCall] = []

                # Parse individual tool calls
                tool_call_pattern = r"<\|tool_call_begin\|>(.*?)<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>"
                for match in re.finditer(tool_call_pattern, tool_section, re.DOTALL):
                    tool_id = match.group(1)
                    args_str = match.group(2)
                    # Try to parse as JSON to validate, but store as string
                    try:
                        json.loads(args_str)
                        tool_calls.append(
                            ToolCall(
                                function=ToolCall.FunctionBody(name="", arguments=args_str),
                                id=tool_id if tool_id else None,
                            )
                        )
                    except json.JSONDecodeError:
                        return assistant_message, False

                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                    # Remove tool section from content
                    content = content[: content.find("<|tool_calls_section_begin|>")]
                    assistant_message["content"] = content

        return assistant_message, True


