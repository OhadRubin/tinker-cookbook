"""Module for qwen3."""

from ..base import *
from ..base import _tool_call_payload


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

    def __init__(
        self,
        tokenizer: Tokenizer,
        strip_thinking_from_history: bool = True,
        tools: list[dict] | None = None,
    ):
        """
        Args:
            tokenizer: The tokenizer to use for encoding.
            strip_thinking_from_history: When True (default), strips <think>...</think> blocks
                from assistant messages in multi-turn history. This matches how Qwen3 models
                were trained - they only see their own thinking during the current turn, not
                from previous turns. Set to False to preserve thinking in history (useful for
                certain RL scenarios where you want the extension property for efficiency).
            tools: Optional list of tool definitions to inject into the system message.
                Each tool should be a dict with the tool's JSON schema following OpenAI format.

        See https://tinker-docs.thinkingmachines.ai/rl/sequence-extension for details on
        how this option affects multi-turn RL compute efficiency.
        """
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history
        self.tools = tools

    def _find_last_query_index(self, messages: list[Message]) -> int:
        """
        Find the last user query index (excludes tool responses).
        Used for multi-step tool handling to determine which assistant messages
        should have thinking blocks.
        """
        for idx in range(len(messages) - 1, -1, -1):
            message = messages[idx]
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, str):
                    # Check if this is NOT a tool response
                    if not (content.startswith("<tool_response>") and content.endswith("</tool_response>")):
                        return idx
        return len(messages) - 1

    def render_message(
        self,
        idx: int,
        message: Message,
        is_last: bool = False,
        last_query_index: int | None = None,
    ) -> RenderedMessage:
        """
        Render a single message.

        Args:
            idx: Index of the message in the conversation.
            message: The message to render.
            is_last: Whether this is the last message in the conversation.
            last_query_index: Index of the last user query (for multi-step tool handling).
                If None, will not apply multi-step tool logic.
        """
        # TODO: `thinking` and `reasoning_content` serve the same purpose - need to understand
        # how to remove this duplication. See kimi.py for a potential solution (uses only `thinking`).
        assert message.get("thinking") is None, "Use 'reasoning_content' field instead of 'thinking'"

        role = message["role"]

        # Handle tool role messages - they get grouped under user with <tool_response> tags
        if role == "tool":
            # Tool messages should be handled by grouping logic, not individually
            # This is a simplified render - actual grouping happens in build_* methods
            assert isinstance(message["content"], str), "Tool messages must have string content"
            prefix = tinker.types.EncodedTextChunk(tokens=[])  # Empty prefix for grouped tools
            content_str = f"\n<tool_response>\n{message['content']}\n</tool_response>"
            content: list[tinker.ModelInputChunk] = [
                tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode(content_str, add_special_tokens=False)
                )
            ]
            return RenderedMessage(prefix=prefix, content=content)

        # Regular message handling
        assert isinstance(message["content"], str), (
            "Qwen3Renderer only supports message with string content (except for tool role)"
        )

        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{role}\n"
        ac_content = message["content"]

        # Handle assistant messages with reasoning
        if role == "assistant":
            # Extract reasoning_content if provided, or parse from content
            reasoning_content = ""
            if "reasoning_content" in message and isinstance(message.get("reasoning_content"), str):
                reasoning_content = message["reasoning_content"]
            elif "</think>" in ac_content:
                # Parse reasoning from content
                parts = ac_content.split("</think>")
                reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                ac_content = parts[-1].lstrip("\n")

            # Determine if we should show reasoning based on multi-step tool logic
            should_show_reasoning = False
            if last_query_index is not None:
                # Multi-step tool logic: show reasoning only after the last query
                should_show_reasoning = idx > last_query_index and (is_last or reasoning_content)
            else:
                # Default behavior: show reasoning for last message or if stripping is disabled
                should_show_reasoning = is_last or not self.strip_thinking_from_history

            if should_show_reasoning and reasoning_content:
                ob_str += f"<think>\n{reasoning_content.strip()}\n</think>\n\n"
            elif should_show_reasoning:
                # Empty thinking block if we should show reasoning but don't have any
                ob_str += "<think>\n\n</think>\n\n"
            elif self.strip_thinking_from_history and "</think>" in message["content"]:
                # Strip thinking from history (already done above in content parsing)
                pass
            elif "<think>" not in message["content"] and not reasoning_content:
                # Force <think> block for assistant if not present (matching paper)
                ob_str += "<think>\n"

        # # Render tool_calls from structured field (source of truth)
        # if "tool_calls" in message:
        #     # Strip any raw <tool_call> text from content - structured field is authoritative
        #     ac_content = re.sub(r"\s*<tool_call>.*?</tool_call>\s*", "", ac_content, flags=re.DOTALL)
        #     tool_calls_str = ""
        #     for tool_call in message["tool_calls"]:
        #         if tool_calls_str or ac_content:
        #             tool_calls_str += "\n"
        #         tool_calls_str += f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
        #     ac_content += tool_calls_str

        ac_content += "<|im_end|>"

        # Action part
        prefix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(ob_str, add_special_tokens=False)
        )
        content_chunks: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(ac_content, add_special_tokens=False)
            )
        ]
        return RenderedMessage(prefix=prefix, content=content_chunks)

    def _build_tools_system_message(self, system_content: str = "") -> str:
        """
        Build a system message that includes tool definitions.

        Args:
            system_content: Optional existing system message content.

        Returns:
            Combined system message with tools definition.
        """
        if not self.tools:
            return system_content

        tools_section = (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>"
        )
        for tool in self.tools:
            tools_section += f"\n{json.dumps(tool)}"
        tools_section += (
            "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

        if system_content:
            return f"{system_content}\n\n{tools_section}"
        return tools_section

    def _group_tool_messages(self, messages: list[Message]) -> list[Message]:
        """
        Group consecutive tool messages under a single user message.

        Tool messages with role="tool" get grouped together and wrapped in
        <|im_start|>user...<|im_end|> with each tool response in <tool_response> tags.
        """
        grouped: list[Message] = []
        i = 0
        while i < len(messages):
            if messages[i]["role"] == "tool":
                # Found start of tool messages - collect consecutive tool messages
                tool_responses = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    assert isinstance(messages[i]["content"], str)
                    tool_responses.append(messages[i]["content"])
                    i += 1

                # Create a single user message with all tool responses
                combined_content = ""
                for resp in tool_responses:
                    combined_content += f"\n<tool_response>\n{resp}\n</tool_response>"

                grouped.append(Message(role="user", content=combined_content))
            else:
                grouped.append(messages[i])
                i += 1

        return grouped
    # TODO: verify if this shit is legal from a "TrainOnWhat" perspective
    def _render_message_tokens(
        self,
        idx: int,
        message: Message,
        is_last: bool = False,
        last_query_index: int | None = None,
    ) -> list[int]:
        """Render a message and return flattened token list (prefix + content)."""
        rendered = self.render_message(idx, message, is_last=is_last, last_query_index=last_query_index)
        tokens: list[int] = []
        prefix = rendered.get("prefix")
        if prefix:
            tokens.extend(prefix.tokens)
        for chunk in rendered["content"]:
            if chunk:
                tokens.extend(chunk.tokens)
        return tokens

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
        args = tool_call.get("arguments")
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

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
        enable_thinking: bool = True,
    ) -> tinker.ModelInput:
        """
        Generates tokens for sampling from the model.

        Args:
            messages: a list of messages to render.
            role: the role of the partial message to be completed.
            prefill: an optional string to prefill in the model's generation.
            enable_thinking: if False, immediately outputs empty <think></think> block.
                This is useful when you want to skip the thinking phase.
        """
        # TODO: verify if this shit is legal from a "TrainOnWhat" perspective
        # NOTE: usually ModelInputChunk is used for seperating out what get's trained on (the weights) and what doesn't, here we use it to group stuff 
        # to reduce cache misses.
        # Group tool messages
        messages = self._group_tool_messages(messages)

        # Handle tools in system message
        if self.tools and (not messages or messages[0]["role"] != "system"):
            # No system message, create one with tools
            system_msg = Message(role="system", content=self._build_tools_system_message())
            messages = [system_msg] + messages
        elif self.tools and messages and messages[0]["role"] == "system":
            # Update existing system message with tools
            messages = messages.copy()
            system_content = ensure_text(messages[0]["content"])
            messages[0] = Message(role="system", content=self._build_tools_system_message(system_content))

        # Find last query index for multi-step tool handling
        last_query_index = self._find_last_query_index(messages)
        # TODO: verify if this shit is legal from a "TrainOnWhat" perspective
        # Build chunks with turn-based structure for KV cache affinity routing.
        # Structure: [system_chunk, turn1_chunk, turn2_chunk, ..., current_partial_chunk]
        # This reduces chunk count from 2*N to ~N/2, enabling effective hierarchical prefix matching.
        chunks: list[tinker.types.ModelInputChunk] = []

        # Chunk 0: BOS + system message (if present)
        system_tokens: list[int] = list(self._bos_tokens) if self._bos_tokens else []
        msg_start_idx = 0
        if messages and messages[0]["role"] == "system":
            system_tokens.extend(self._render_message_tokens(0, messages[0], last_query_index=last_query_index))
            msg_start_idx = 1
        chunks.append(tinker.types.EncodedTextChunk(tokens=system_tokens))

        # Chunks 1..N-1: Complete turns (user + assistant pairs)
        # A turn ends after an assistant message, unless it's the final message (incomplete turn)
        remaining_messages = messages[msg_start_idx:]
        turn_tokens: list[int] = []

        for idx, message in enumerate(remaining_messages):
            original_idx = idx + msg_start_idx  # Index in original messages list
            turn_tokens.extend(self._render_message_tokens(
                original_idx, message, last_query_index=last_query_index
            ))

            is_assistant = message["role"] == "assistant"
            is_last = idx == len(remaining_messages) - 1

            # End turn after assistant message, but not if it's the last message (incomplete turn)
            if is_assistant and not is_last:
                chunks.append(tinker.types.EncodedTextChunk(tokens=turn_tokens))
                turn_tokens = []

        # Chunk N: Current incomplete turn (remaining tokens + new assistant prefix + prefill)
        new_partial_message = Message(role=role, content="")
        rendered_partial = self.render_message(len(messages), new_partial_message)
        partial_prefix = rendered_partial.get("prefix")
        if partial_prefix:
            turn_tokens.extend(partial_prefix.tokens)

        # Handle enable_thinking option
        if not enable_thinking:
            empty_think = "<think>\n\n</think>\n\n"
            turn_tokens.extend(self.tokenizer.encode(empty_think, add_special_tokens=False))

        if prefill:
            turn_tokens.extend(self.tokenizer.encode(prefill, add_special_tokens=False))

        chunks.append(tinker.types.EncodedTextChunk(tokens=turn_tokens))

        return tinker.ModelInput(chunks=chunks)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """
        Generates tokens and weights (for SFT) with support for tool messages and reasoning.

        Args:
            messages: a list of messages to render.
            train_on_what: an enum that controls how the weights are assigned to the tokens.

        Returns:
            A tuple of two tensors:
                - model_input: the tinker ModelInput for your model
                - weights: a tensor of weights
        """
        # Group tool messages
        messages = self._group_tool_messages(messages)

        # Handle tools in system message
        if self.tools and (not messages or messages[0]["role"] != "system"):
            # No system message, create one with tools
            system_msg = Message(role="system", content=self._build_tools_system_message())
            messages = [system_msg] + messages
        elif self.tools and messages and messages[0]["role"] == "system":
            # Update existing system message with tools
            messages = messages.copy()
            system_content = ensure_text(messages[0]["content"])
            messages[0] = Message(role="system", content=self._build_tools_system_message(system_content))

        model_input_chunks_weights: list[tuple[tinker.types.ModelInputChunk, float]] = []
        if self._bos_tokens:
            model_input_chunks_weights.append(
                (tinker.types.EncodedTextChunk(tokens=self._bos_tokens), 0.0)
            )

        # Find last query index for multi-step tool handling
        last_query_index = self._find_last_query_index(messages)

        for idx, message in enumerate(messages):
            if train_on_what == TrainOnWhat.CUSTOMIZED:
                assert "trainable" in message, (
                    "When using CUSTOMIZED train_on_what, each message must have a trainable field: True if loss is applied on this message, False otherwise"
                )
            else:
                assert "trainable" not in message, (
                    "When using non-CUSTOMIZED train_on_what, each message must not have a trainable field. Either change train_on_what to CUSTOMIZED or remove the trainable field from the message"
                )

            is_last_message = idx == len(messages) - 1
            is_assistant = message["role"] == "assistant"
            is_user_or_system = message["role"] in ["user", "system"]

            # only apply weight to observation part if train_on_what is ALL_TOKENS
            rendered_message = self.render_message(
                idx, message, is_last=is_last_message, last_query_index=last_query_index
            )
            ob_part = rendered_message.get("prefix")
            action_parts = rendered_message.get("content")
            action_tail = rendered_message.get("suffix")

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

            # action tail is effectively the stop_token and the start token for the next turn
            # e.g. \n\nUser:
            if is_last_message and action_tail:
                model_input_chunks_weights += [(action_tail, int(action_has_weight))]

        weights_data = [w for chunk, w in model_input_chunks_weights for _ in range(chunk.length)]
        weights_tensor = torch.tensor(weights_data)

        model_input_chunks = [chunk for chunk, _ in model_input_chunks_weights]
        return tinker.ModelInput(chunks=model_input_chunks), weights_tensor

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]

        # Extract reasoning_content if present in <think>...</think> tags
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            reasoning_content = think_match.group(1).strip("\n")
            # Remove the think block from content and get the part after </think>
            content = content[think_match.end():].lstrip("\n")
            if reasoning_content:  # Only add if non-empty
                assistant_message["reasoning_content"] = reasoning_content
            assistant_message["content"] = content

        # Follow Qwen docs and Qwen-Agent's tool calling prompt to use <tool_call>...</tool_call> tags to wrap the tool call.
        # - https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#tool-calling
        # - https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py#L279-L282
        match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if match:
            tool_calls = self._parse_tool_call(match.group(1))
            if tool_calls is None:
                return assistant_message, False
            else:
                assistant_message["tool_calls"] = tool_calls
                return assistant_message, True
        return assistant_message, True

