"""Module for Qwen3 Coder models with XML-based tool calling."""

import re
from .base import *


class Qwen3CoderRenderer(Renderer):
    """
    Renderer for Qwen3 Coder models that use XML-structured tool calling format.

    Unlike the base Qwen3Renderer which uses JSON for tool calls, Qwen3Coder uses
    XML tags following this format:

    Tool definition in system message:
        <tools>
        <function>
        <name>function_name</name>
        <description>Description here</description>
        <parameters>
        <parameter>
        <name>param_name</name>
        <type>string</type>
        <description>Param description</description>
        </parameter>
        </parameters>
        </function>
        </tools>

    Tool call format:
        <tool_call>
        <function=function_name>
        <parameter=param_name>
        value
        </parameter>
        </function>
        </tool_call>

    Tool response format:
        <|im_start|>user
        <tool_response>
        response content
        </tool_response>
        <|im_end|>

    This renderer does NOT support <think> blocks, as Qwen3 Coder models are
    optimized for coding tasks rather than reasoning with thinking tokens.
    """

    def render_message(self, idx: int, message: Message, is_last: bool = False) -> RenderedMessage:
        assert message.get("thinking") is None, "Thinking blocks not supported in Qwen3CoderRenderer"
        assert isinstance(message["content"], str), (
            "Qwen3CoderRenderer only supports message with string content"
        )

        # Handle tool responses specially - they are wrapped in user messages
        # Following the jinja2 template behavior where role="tool" messages
        # get wrapped in <|im_start|>user\n<tool_response>...<|im_end|>
        if message["role"] == "tool":
            maybe_newline = "\n" if idx > 0 else ""
            ob_str = f"{maybe_newline}<|im_start|>user\n"
            ac_content = f"<tool_response>\n{message['content']}\n</tool_response><|im_end|>"

            prefix = tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(ob_str, add_special_tokens=False)
            )
            content: list[tinker.ModelInputChunk] = [
                tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode(ac_content, add_special_tokens=False)
                )
            ]
            return RenderedMessage(prefix=prefix, content=content)

        # Handle regular messages
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"
        ac_content = message["content"]

        # Handle tool calls - render them in XML format
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                ac_content += self._render_tool_call(tool_call)

        ac_content += "<|im_end|>"

        # Encode the parts
        prefix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(ob_str, add_special_tokens=False)
        )
        content: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(ac_content, add_special_tokens=False)
            )
        ]
        return RenderedMessage(prefix=prefix, content=content)

    def _render_tool_call(self, tool_call: ToolCall) -> str:
        """
        Render a tool call in XML format.

        Format:
            <tool_call>
            <function=function_name>
            <parameter=param_name>
            value
            </parameter>
            </function>
            </tool_call>
        """
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        result = "\n<tool_call>\n"
        result += f"<function={function_name}>\n"

        for param_name, param_value in arguments.items():
            result += f"<parameter={param_name}>\n"
            # Convert to string if needed, preserving multiline values
            if isinstance(param_value, (dict, list)):
                result += json.dumps(param_value)
            else:
                result += str(param_value)
            result += f"\n</parameter>\n"

        result += "</function>\n"
        result += "</tool_call>"

        return result

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|im_end|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def _parse_tool_call(self, tool_call_str: str) -> list[ToolCall] | None:
        """
        Parse a tool call from XML format.

        Expected format:
            <function=function_name>
            <parameter=param1>value1</parameter>
            <parameter=param2>value2</parameter>
            </function>
        """
        try:
            # Extract function name from <function=name> tag
            func_match = re.search(r'<function=([^>]+)>', tool_call_str)
            if not func_match:
                return None

            function_name = func_match.group(1)

            # Extract all parameters
            arguments: dict[str, str] = {}
            param_pattern = r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>'
            for param_match in re.finditer(param_pattern, tool_call_str, re.DOTALL):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()

                # Try to parse as JSON if it looks like JSON
                if param_value.startswith('{') or param_value.startswith('['):
                    try:
                        param_value = json.loads(param_value)
                    except json.JSONDecodeError:
                        # Keep as string if JSON parsing fails
                        pass

                arguments[param_name] = param_value

            # Convert to ToolCall format
            return [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name=function_name,
                        arguments=json.dumps(arguments)
                    ),
                    id=None,  # XML format doesn't include IDs
                )
            ]
        except Exception:
            return None

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """
        Parse the model's response, extracting tool calls if present.

        The model may generate:
        1. Plain text response
        2. Response with tool call in XML format
        3. Malformed response (parse failure)
        """
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        # Look for tool calls in XML format
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
