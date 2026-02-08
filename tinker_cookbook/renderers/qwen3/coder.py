"""Module for Qwen3 Coder models with XML-based tool calling."""

import re
from .base import Qwen3Renderer, RenderedMessage, Message, ToolCall, Tokenizer
from ..base import _tool_call_payload
import tinker
import json


class Qwen3CoderRenderer(Qwen3Renderer):
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

    # Default system message when tools are present but no system message provided
    # Matches qwen3_coder.jinja2 line 28
    DEFAULT_SYSTEM_MESSAGE = "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks."

    def __init__(
        self,
        tokenizer: Tokenizer,
        tools: list[dict] | None = None,
    ):
        super().__init__(tokenizer, strip_thinking_from_history=True, tools=tools)

    def _group_tool_messages(self, messages: list[Message]) -> list[Message]:
        """
        Group consecutive tool messages under a single user message.

        Matches qwen3_coder.jinja2 lines 99-110:
        - No leading newline before first <tool_response>
        - Trailing newline after each </tool_response>
        """
        grouped: list[Message] = []
        i = 0
        while i < len(messages):
            if messages[i]["role"] == "tool":
                tool_responses: list[str] = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    content = messages[i]["content"]
                    assert isinstance(content, str)
                    tool_responses.append(content)
                    i += 1

                # Match jinja2: <tool_response>\ncontent\n</tool_response>\n
                combined_content = ""
                for resp in tool_responses:
                    combined_content += f"<tool_response>\n{resp}\n</tool_response>\n"

                grouped.append(Message(role="user", content=combined_content))
            else:
                grouped.append(messages[i])
                i += 1

        return grouped

    def render_message(
        self,
        idx: int,
        message: Message,
        is_last: bool = False,
        last_query_index: int | None = None,
    ) -> RenderedMessage:
        """
        Render a message without <think> tag support.

        Tool messages should have been grouped by _group_tool_messages in base class
        before this method is called from build_generation_prompt or build_supervised_example.
        """
        assert message.get("thinking") is None, "Thinking blocks not supported in Qwen3CoderRenderer"
        assert message.get("reasoning_content") is None, "Reasoning not supported in Qwen3CoderRenderer"

        role = message["role"]

        # Tool messages should have been grouped by _group_tool_messages in base class
        assert role != "tool", (
            "Tool messages should be grouped by build_generation_prompt or build_supervised_example"
        )

        # Handle assistant messages with tool_calls (jinja2 lines 76-96)
        if role == "assistant" and message.get("tool_calls"):
            ob_str = f"<|im_start|>{role}"
            content = message.get("content", "")

            ac_content = ""
            if isinstance(content, str) and content.strip():
                # jinja2 line 79: '\n' + message.content | trim + '\n'
                ac_content += f"\n{content.strip()}\n"

            for tool_call in message["tool_calls"]:
                ac_content += self._render_tool_call(tool_call)

            ac_content += "<|im_end|>\n"

            prefix = tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(ob_str, add_special_tokens=False)
            )
            content_chunks: list[tinker.ModelInputChunk] = [
                tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode(ac_content, add_special_tokens=False)
                )
            ]
            return RenderedMessage(prefix=prefix, content=content_chunks)

        # Regular message handling (jinja2 line 98)
        assert isinstance(message["content"], str), (
            "Qwen3CoderRenderer only supports message with string content"
        )

        # jinja2 line 98: '<|im_start|>' + role + '\n' + content + '<|im_end|>' + '\n'
        ob_str = f"<|im_start|>{role}\n"
        ac_content = message["content"] + "<|im_end|>\n"

        prefix = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(ob_str, add_special_tokens=False)
        )
        content_chunks: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(ac_content, add_special_tokens=False)
            )
        ]
        return RenderedMessage(prefix=prefix, content=content_chunks)

    def _render_extra_keys(self, json_dict: dict | None, handled_keys: list[str]) -> str:
        """
        Render extra keys from a JSON dict that aren't in the handled_keys list.

        Mirrors the render_extra_keys macro from qwen3_coder.jinja2.
        Handles JSON schema fields like 'enum', 'required', 'items', etc.
        """
        if not isinstance(json_dict, dict):
            return ""

        result = ""
        for key in json_dict:
            if key not in handled_keys:
                value = json_dict[key]
                if isinstance(value, (dict, list)):
                    result += f"\n<{key}>{json.dumps(value)}</{key}>"
                else:
                    result += f"\n<{key}>{value}</{key}>"
        return result

    def _build_tools_system_message(self, system_content: str = "") -> str:
        """
        Build a system message that includes tool definitions in XML format.

        Follows the qwen3_coder.jinja2 template format.
        """
        if not self.tools:
            return system_content

        tools_section = "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>"

        for tool in self.tools:
            # Handle both {"function": {...}} and direct tool format
            tool_def = tool.get("function", tool)
            name = tool_def.get("name", "")
            description = tool_def.get("description", "")
            parameters = tool_def.get("parameters", {})

            tools_section += f"\n<function>\n<name>{name}</name>"
            if description:
                tools_section += f"\n<description>{description.strip()}</description>"
            tools_section += "\n<parameters>"

            # Render parameters
            properties = parameters.get("properties", {})
            for param_name, param_fields in properties.items():
                tools_section += "\n<parameter>"
                tools_section += f"\n<name>{param_name}</name>"
                if "type" in param_fields:
                    tools_section += f"\n<type>{param_fields['type']}</type>"
                if "description" in param_fields:
                    tools_section += f"\n<description>{param_fields['description'].strip()}</description>"
                # Render extra parameter fields (enum, items, default, etc.)
                tools_section += self._render_extra_keys(param_fields, ["name", "type", "description"])
                tools_section += "\n</parameter>"

            # Render extra fields in parameters object (e.g., 'required')
            tools_section += self._render_extra_keys(parameters, ["type", "properties"])
            tools_section += "\n</parameters>"
            # Render extra tool-level fields
            tools_section += self._render_extra_keys(tool_def, ["type", "name", "description", "parameters"])
            tools_section += "\n</function>"

        tools_section += "\n</tools>"

        # Add instruction text per jinja2 template
        tools_section += (
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT>"
        )

        if system_content:
            return f"{system_content}{tools_section}"
        # Use default system message when no system content provided (matches jinja2 line 28)
        return f"{self.DEFAULT_SYSTEM_MESSAGE}{tools_section}"

    def _render_tool_call(self, tool_call: ToolCall | dict) -> str:
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
        payload = _tool_call_payload(tool_call)
        function_name = payload["name"]
        arguments = payload["arguments"]

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

    def _parse_tool_call(self, tool_call_str: str) -> list[ToolCall] | None:
        """
        Parse a tool call from XML format.

        Expected format:
            <function=function_name>
            <parameter=param1>value1</parameter>
            <parameter=param2>value2</parameter>
            </function>
        """
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



    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """
        Parse the model's response, extracting tool calls if present.

        The model may generate:
        1. Plain text response
        2. Response with tool call in XML format
        3. Malformed response (parse failure)
        """
        from ..base import parse_response_for_stop_token

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
