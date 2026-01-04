# Qwen3Renderer Quick Reference

## Constructor

```python
from tinker_cookbook.renderers.qwen3.base import Qwen3Renderer

renderer = Qwen3Renderer(
    tokenizer=tokenizer,
    strip_thinking_from_history=True,  # Strip <think> from history (default: True)
    tools=None  # Optional tool definitions
)
```

## Message Format

```python
from tinker_cookbook.renderers.base import Message

# Basic message
Message(role="user", content="Hello")

# Assistant with reasoning
Message(
    role="assistant",
    reasoning_content="Let me think...",  # NEW!
    content="The answer is..."
)

# Tool message (will be grouped)
Message(role="tool", content="Search results here")

# Assistant with tool calls
Message(
    role="assistant",
    content="I'll search for that",
    tool_calls=[ToolCall(...)]
)
```

## Methods

### build_generation_prompt()

```python
model_input = renderer.build_generation_prompt(
    messages=[...],
    role="assistant",
    prefill=None,
    enable_thinking=True  # NEW! Set to False to skip thinking
)
```

### build_supervised_example()

```python
model_input, weights = renderer.build_supervised_example(
    messages=[...],
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
)
```

### parse_response()

```python
# Returns Message with reasoning_content extracted
message, success = renderer.parse_response(tokens)

# message = {
#     "role": "assistant",
#     "reasoning_content": "...",  # NEW! Extracted from <think>
#     "content": "...",
#     "tool_calls": [...]  # If present
# }
```

## Tools Setup

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

renderer = Qwen3Renderer(tokenizer, tools=tools)

# Tools automatically injected into system message in <tools>...</tools> format
```

## Complete Example

```python
from tinker_cookbook.renderers.qwen3.base import Qwen3Renderer
from tinker_cookbook.renderers.base import Message, ToolCall
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Setup
tokenizer = get_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
tools = [{"type": "function", "function": {"name": "search", ...}}]
renderer = Qwen3Renderer(tokenizer, tools=tools)

# Multi-turn conversation with tools
messages = [
    Message(role="user", content="Search for Python tutorials"),
    Message(
        role="assistant",
        reasoning_content="I should search for this",
        content="Let me search for that",
        tool_calls=[ToolCall(
            function=ToolCall.FunctionBody(
                name="search",
                arguments='{"query": "Python tutorials"}'
            )
        )]
    ),
    Message(role="tool", content="Found 10 results"),
    Message(
        role="assistant",
        reasoning_content="These results look good",
        content="Here are some great Python tutorials..."
    )
]

# For training
model_input, weights = renderer.build_supervised_example(messages)

# For sampling
model_input = renderer.build_generation_prompt(
    messages[:-1],
    enable_thinking=False  # Skip thinking for faster inference
)

# Parse model output
response_tokens = sampler.sample(model_input)
message, success = renderer.parse_response(response_tokens)
print(f"Reasoning: {message.get('reasoning_content', 'N/A')}")
print(f"Response: {message['content']}")
```

## Feature Comparison

| Feature | Before | After |
|---------|--------|-------|
| Reasoning | Mixed in content with `<think>` tags | Separate `reasoning_content` field |
| Tools | Manual system message construction | Automatic via `tools` parameter |
| Tool messages | Manual handling | Automatic grouping with `<tool_response>` |
| Skip thinking | Not possible | `enable_thinking=False` |
| Multi-step tools | Manual logic | Automatic via `_find_last_query_index()` |

## Output Format

The renderer produces this format:

```
<|im_start|>system
[System message]
[Tools section if tools provided]<|im_end|>
<|im_start|>user
[User message]<|im_end|>
<|im_start|>assistant
<think>
[Reasoning content]
</think>

[Response content]
[Tool calls if any]<|im_end|>
<|im_start|>user
<tool_response>
[Tool result]
</tool_response>
[More tool responses if grouped]<|im_end|>
```

## Tips

1. **Use `reasoning_content`** instead of embedding `<think>` in content
2. **Set `enable_thinking=False`** for faster inference when reasoning isn't needed
3. **Consecutive tool messages** are automatically grouped - no manual handling needed
4. **Tools** are automatically formatted - just pass the OpenAI-style tool definitions
5. **Multi-step tool conversations** work automatically - reasoning is only shown after the last user query
