# Qwen3Renderer Updates - Feature Parity with Jinja Template

## Summary

Updated `Qwen3Renderer` to achieve feature parity with `qwen3_chat_template.jinja2`. All missing features from the Jinja template have been successfully integrated into the Python renderer.

## Files Modified

1. **`/home/ohadr/tinker-cookbook/tinker_cookbook/renderers/qwen3/base.py`** - Main renderer implementation
2. **`/home/ohadr/tinker-cookbook/tinker_cookbook/renderers/base.py`** - Added `reasoning_content` field to `Message` type

## New Features Added

### 1. `reasoning_content` Field Support

**What:** Messages can now have a separate `reasoning_content` field distinct from `content`.

**Usage:**
```python
Message(
    role="assistant",
    reasoning_content="Let me think about this problem step by step...",
    content="The answer is 42."
)
```

**Implementation:**
- `reasoning_content` is extracted from messages during rendering
- For backwards compatibility, also parses `<think>...</think>` blocks from content
- `parse_response()` now extracts reasoning into the `reasoning_content` field

### 2. Tool Role Messages with `<tool_response>` Tags

**What:** Proper handling of `role="tool"` messages, grouped under user messages with `<tool_response>` tags.

**Jinja template behavior:**
```jinja
{%- elif message.role == "tool" %}
    {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
        {{- '<|im_start|>user' }}
    {%- endif %}
    {{- '\n<tool_response>\n' }}
    {{- content }}
    {{- '\n</tool_response>' }}
```

**Python implementation:**
- `_group_tool_messages()` method groups consecutive tool messages
- Each tool response wrapped in `<tool_response>` tags
- All grouped under single `<|im_start|>user...<|im_end|>` block

**Usage:**
```python
messages = [
    Message(role="assistant", content="Searching...", tool_calls=[...]),
    Message(role="tool", content="Result 1"),
    Message(role="tool", content="Result 2"),
    Message(role="assistant", content="Based on the results...")
]
```

### 3. Tools Definition in System Message

**What:** Optional `tools` parameter that automatically injects tool definitions into the system message.

**Constructor signature:**
```python
Qwen3Renderer(
    tokenizer: Tokenizer,
    strip_thinking_from_history: bool = True,
    tools: list[dict] | None = None
)
```

**Usage:**
```python
tools = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {...}
    }
}]

renderer = Qwen3Renderer(tokenizer, tools=tools)
```

**Behavior:**
- If tools provided and no system message exists: creates system message with tools
- If tools provided and system message exists: appends tools to existing system content
- Format matches Jinja template exactly: `<tools>...</tools>` with JSON tool definitions

### 4. `enable_thinking` Option

**What:** Option to skip the thinking phase by immediately outputting empty `<think></think>` block.

**Method signature:**
```python
build_generation_prompt(
    messages: list[Message],
    role: Role = "assistant",
    prefill: str | None = None,
    enable_thinking: bool = True
) -> tinker.ModelInput
```

**Usage:**
```python
# Force model to skip thinking and go straight to response
model_input = renderer.build_generation_prompt(
    messages,
    enable_thinking=False
)
```

**When `enable_thinking=False`:**
- Immediately outputs: `<think>\n\n</think>\n\n`
- Model starts generating response content directly

### 5. Multi-Step Tool Handling Logic

**What:** Sophisticated logic to determine when to show reasoning blocks in multi-turn tool-using conversations.

**Jinja template logic:**
```jinja
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- if ns.multi_step_tool and message.role == "user" and ... %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
```

**Python implementation:**
- `_find_last_query_index()` identifies the last real user query (not tool responses)
- Reasoning blocks only shown for assistant messages after the last query
- Matches Jinja template behavior exactly

**Behavior:**
- Before last query: strip thinking from assistant messages (history)
- After last query: preserve thinking for current turn
- Tool response messages (user role with `<tool_response>`) don't count as queries

### 6. Enhanced `parse_response()`

**What:** Now extracts `reasoning_content` from `<think>...</think>` blocks.

**Returns:**
```python
Message(
    role="assistant",
    reasoning_content="<extracted reasoning>",  # New!
    content="<response without think block>",
    tool_calls=[...]  # If present
)
```

## Implementation Details

### Helper Methods Added

1. **`_find_last_query_index(messages)`**
   - Finds last user query (excluding tool responses)
   - Used for multi-step tool handling

2. **`_build_tools_system_message(system_content)`**
   - Builds system message with tools in `<tools>` tags
   - Combines with existing system content if provided

3. **`_group_tool_messages(messages)`**
   - Groups consecutive tool messages under single user message
   - Wraps each in `<tool_response>` tags

### Updated Methods

1. **`render_message()`**
   - New parameter: `last_query_index: int | None = None`
   - Handles `reasoning_content` field
   - Supports tool role messages
   - Multi-step tool logic for thinking display

2. **`build_generation_prompt()`**
   - New parameter: `enable_thinking: bool = True`
   - Groups tool messages
   - Injects tools into system message
   - Applies multi-step tool logic

3. **`build_supervised_example()`**
   - Groups tool messages
   - Injects tools into system message
   - Applies multi-step tool logic

4. **`parse_response()`**
   - Extracts `reasoning_content` from `<think>` blocks
   - Removes think block from content
   - Preserves existing tool call parsing

## Comparison with Jinja Template

### Feature Mapping

| Jinja Feature | Python Implementation | Status |
|--------------|----------------------|---------|
| `enable_thinking` check | `enable_thinking` parameter | ✅ Complete |
| `reasoning_content` field | `reasoning_content` in Message | ✅ Complete |
| `multi_step_tool` logic | `_find_last_query_index()` | ✅ Complete |
| Tool role handling | `_group_tool_messages()` | ✅ Complete |
| Tools in system | `tools` parameter + `_build_tools_system_message()` | ✅ Complete |

### Key Differences

1. **Tool message grouping**: Jinja does inline, Python preprocesses via `_group_tool_messages()`
2. **Last query finding**: Jinja uses reverse iteration, Python uses backward loop
3. **Interface**: Python uses parameters/methods instead of template variables

### Behavioral Parity

All behaviors match the Jinja template:
- ✅ Empty think blocks when `enable_thinking=False`
- ✅ Reasoning extracted/injected in same locations
- ✅ Tool responses grouped identically
- ✅ Tools formatted with exact same XML/JSON structure
- ✅ Multi-step tool logic produces same output

## Testing

Created `/home/ohadr/tinker-cookbook/test_qwen3_features.py` with tests for:
1. `reasoning_content` field
2. Tool messages with `<tool_response>` tags
3. Tools in system message
4. `enable_thinking=False` option
5. `parse_response()` reasoning extraction

All tests pass ✅

## Backwards Compatibility

All changes are backwards compatible:
- Existing code without new features works unchanged
- New parameters have default values matching old behavior
- Old `thinking` field still supported (though deprecated in favor of `reasoning_content`)

## Migration Guide

### Using reasoning_content

**Before:**
```python
Message(
    role="assistant",
    content="<think>reasoning here</think>\n\nResponse here"
)
```

**After:**
```python
Message(
    role="assistant",
    reasoning_content="reasoning here",
    content="Response here"
)
```

### Using tools

**Before:**
```python
# Manually construct system message with tools
system_msg = Message(
    role="system",
    content=f"You are helpful.\n\nTools: {json.dumps(tools)}"
)
```

**After:**
```python
renderer = Qwen3Renderer(tokenizer, tools=tools)
# Tools automatically injected in correct format
```

### Disabling thinking

**Before:**
```python
# Not possible
```

**After:**
```python
model_input = renderer.build_generation_prompt(
    messages,
    enable_thinking=False
)
```

## Future Enhancements

Potential improvements not in Jinja template:
1. Support for multimodal content in tool messages
2. Streaming support for reasoning_content
3. Custom thinking block delimiters
4. Tool call batching/parallelization hints

## Related Files

- Jinja template: `/home/ohadr/tinker-cookbook/qwen3_chat_template.jinja2`
- Renderer: `/home/ohadr/tinker-cookbook/tinker_cookbook/renderers/qwen3/base.py`
- Base types: `/home/ohadr/tinker-cookbook/tinker_cookbook/renderers/base.py`
- Tests: `/home/ohadr/tinker-cookbook/test_qwen3_features.py`
