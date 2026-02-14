"""
OpenAI-compatible client backed by Tinker sampling.

Implements OpenAI client semantics for:
- chat.completions.create(...)
- completions.create(...)

Returns OpenAI types (ChatCompletion / Completion) constructed from sampled tokens.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, overload

import httpx
from observability import log, bootstrap, set_run_id, get_run_id, Events
from tinker_cookbook.utils.trajectory_progress import _get_current_trajectory

import tinker
import verifiers as vf
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_never,
    wait_fixed,
)
from tinker import BadRequestError

# Must match the prefix in tx.tinker.extra.external_inference
TIMEOUT_ERROR_PREFIX = "TINKER_TIMEOUT: "
SAMPLING_TIMEOUT_SECONDS = 300
SAMPLING_TIMEOUT_INCREMENT = 30

from tinker_cookbook.utils.trajectory_progress import set_trajectory_context


def _build_session_id() -> str | None:
    run_id = get_run_id()
    traj = _get_current_trajectory()
    if run_id and traj:
        return f"{run_id}_{traj.group_id}_{traj.trajectory_id}"
    return None


_original_httpx_send = httpx.AsyncClient.send


async def _patched_httpx_send(self, request, **kwargs):
    if "/api/v1/asample" in str(request.url):
        session_id = _build_session_id()
        if session_id:
            request.headers["X-Tx-Param-Session-ID"] = session_id
    return await _original_httpx_send(self, request, **kwargs)


httpx.AsyncClient.send = _patched_httpx_send
from openai._streaming import AsyncStream
from openai.resources.chat import AsyncChat as OpenAIAsyncChat
from openai.resources.chat.completions import AsyncCompletions as OpenAIAsyncChatCompletions
from openai.resources.completions import AsyncCompletions as OpenAIAsyncCompletions
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import Tokenizer


FAILED_CALLS_DIR = Path("/tmp/failed_sampling_calls")


def _dump_failed_call(
    reason: str,
    attempt_number: int,
    sampling_params: tinker.SamplingParams,
    caller_context: Any,
    model_input: tinker.ModelInput,
    error_message: str,
) -> Path:
    FAILED_CALLS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = FAILED_CALLS_DIR / f"{reason}_{ts}_attempt{attempt_number}.json"
    payload = {
        "reason": reason,
        "error_message": error_message,
        "attempt_number": attempt_number,
        "max_tokens": sampling_params.max_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "caller_context": repr(caller_context),
        "model_input": repr(model_input),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _update_payload_file(text: str) -> None:
    session_id = _build_session_id() or "unknown"
    payload_path = Path("/tmp/sampling_payloads") / f"{session_id}.json"
    if not payload_path.exists():
        return
    existing = json.loads(payload_path.read_text())
    existing["completion_text"] = text
    payload_path.write_text(json.dumps(existing, indent=2, default=str))


class SamplingTimeoutError(Exception):
    """Raised when sampling times out (client-side or server-side)."""
    pass


async def sample_with_retries(
    sampling_client: tinker.SamplingClient,
    prompt: tinker.ModelInput,
    sampling_params: tinker.SamplingParams,
    caller_context: Any,
) -> tinker.SampleResponse:
    """Sample with tenacity retries and linearly increasing timeout. Never raises."""
    payloads_dir = Path("/tmp/sampling_payloads")
    payloads_dir.mkdir(parents=True, exist_ok=True)
    session_id = _build_session_id() or "unknown"
    payload_path = payloads_dir / f"{session_id}.json"
    payload_path.write_text(json.dumps({
        "session_id": session_id,
        "max_tokens": sampling_params.max_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "caller_context": caller_context,
        "model_input": prompt.model_dump() if hasattr(prompt, "model_dump") else prompt.__dict__,
    }, indent=2, default=str))

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(SamplingTimeoutError),
        stop=stop_never,
        wait=wait_fixed(0),
    ):
        with attempt:
            attempt_number = attempt.retry_state.attempt_number
            timeout = SAMPLING_TIMEOUT_SECONDS + (attempt_number - 1) * SAMPLING_TIMEOUT_INCREMENT
            try:
                sample = await asyncio.wait_for(
                    sampling_client.sample_async(
                        prompt=prompt,
                        num_samples=1,
                        sampling_params=sampling_params,
                    ),
                    timeout=timeout,
                )
                return sample
            except asyncio.TimeoutError:
                dump_path = _dump_failed_call(
                    reason="client_timeout",
                    attempt_number=attempt_number,
                    sampling_params=sampling_params,
                    caller_context=caller_context,
                    model_input=prompt,
                    error_message=f"timeout after {timeout}s",
                )
                log.error(
                    "client-side timeout, retrying",
                    component="client",
                    timeout_seconds=timeout,
                    attempt_number=attempt_number,
                    dump_path=str(dump_path),
                    caller_context_suffix=repr(caller_context)[-200:],
                )
                raise SamplingTimeoutError()
            except (BadRequestError, ValueError) as e:
                # if TIMEOUT_ERROR_PREFIX not in e.message:
                #     raise
                error_msg = e.message if isinstance(e, BadRequestError) else str(e)
                dump_path = _dump_failed_call(
                    reason="bad_request",
                    attempt_number=attempt_number,
                    sampling_params=sampling_params,
                    caller_context=caller_context,
                    model_input=prompt,
                    error_message=error_msg,
                )
                log.error(
                    "bad request from server, retrying",
                    component="server",
                    attempt_number=attempt_number,
                    error_type=type(e).__name__,
                    error_message=error_msg,
                    dump_path=str(dump_path),
                    caller_context_suffix=repr(caller_context)[-200:],
                )
                raise SamplingTimeoutError()


class TinkerAsyncOpenAIClient(AsyncOpenAI):
    """
    OpenAI-compatible async client that routes calls to a Tinker SamplingClient.
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        tokenizer: Tokenizer,
        max_context_length: int,
    ) -> None:
        super().__init__(api_key="tinker", base_url="http://localhost")
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length

    def set_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        self.sampling_client = sampling_client

    @property
    def chat(self) -> OpenAIAsyncChat:
        return TinkerAsyncChat(self)

    @property
    def completions(self) -> OpenAIAsyncCompletions:
        return TinkerCompletions(self)


class TinkerChatCompletions(OpenAIAsyncChatCompletions):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @overload
    async def create(
        self, *args: Any, stream: Literal[True], **kwargs: Any
    ) -> AsyncStream[Any]: ...

    @overload
    async def create(
        self, *args: Any, stream: Literal[False] = False, **kwargs: Any
    ) -> ChatCompletion: ...

    @overload
    async def create(self, *args: Any, stream: bool, **kwargs: Any) -> ChatCompletion: ...

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion | AsyncStream[Any]:
        model = kwargs.get("model", "tinker")
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools")
        if kwargs.get("stream", False):
            raise ValueError("stream=True not supported by TinkerAsyncOpenAIClient")
        sampling_args = {k: v for k, v in kwargs.items() if k not in ("model", "messages", "tools")}

        stop = sampling_args.get("stop", self._parent.renderer.get_stop_sequences())
        max_completion_tokens = sampling_args.get("max_tokens") or sampling_args.get("max_completion_tokens")

        # Handle tools by temporarily setting them on the renderer
        renderer = self._parent.renderer
        original_tools = getattr(renderer, "tools", None)
        if tools:
            if not hasattr(renderer, "tools"):
                raise NotImplementedError(
                    f"Tool calling is not supported by renderer {type(renderer).__name__}. "
                    "Use a Qwen3Renderer or similar renderer that supports tools."
                )
            renderer.tools = tools

        try:
            model_input = renderer.build_generation_prompt(messages)
            prompt_token_ids: List[int] = model_input.to_ints()

            max_completion_tokens = int(max_completion_tokens or 128)
            total_tokens = len(prompt_token_ids) + max_completion_tokens
            if total_tokens > self._parent.max_context_length:
                max_completion_tokens = self._parent.max_context_length - len(prompt_token_ids) - 3
                if max_completion_tokens <= 0:
                    raise vf.OverlongPromptError(
                        f"Prompt alone exceeds max context length: "
                        f"{len(prompt_token_ids)} prompt tokens > "
                        f"{self._parent.max_context_length} max context length"
                    )

            sample = await sample_with_retries(
                self._parent.sampling_client,
                prompt=model_input,
                sampling_params=tinker.SamplingParams(
                    temperature=float(sampling_args.get("temperature", 1.0)),
                    max_tokens=max_completion_tokens,
                    top_p=float(sampling_args.get("top_p", 1.0)),
                    top_k=int(sampling_args.get("top_k", -1)),
                    stop=stop,
                ),
                caller_context=messages,
            )

            seq = sample.sequences[0]
            completion_token_ids: List[int] = seq.tokens
            logprobs: List[float] = seq.logprobs or [0.0] * len(completion_token_ids)
            _update_payload_file(self._parent.tokenizer.decode(completion_token_ids))

            context_length = len(prompt_token_ids)
            set_trajectory_context(context_length)


            assistant_message, parse_success = renderer.parse_response(
                completion_token_ids
            )

            # Handle tool_calls in OpenAI format if present
            if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
                # Convert internal ToolCall format to OpenAI format
                openai_tool_calls = []
                for i, tc in enumerate(assistant_message["tool_calls"]):
                    openai_tool_calls.append({
                        "id": tc.id or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    })
                assistant_message["tool_calls"] = openai_tool_calls
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop" if parse_success else "length"

            response_dict: Dict[str, Any] = {
                "id": "tinker-chatcmpl",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": assistant_message,
                        "finish_reason": finish_reason,
                        "logprobs": {
                            "content": [
                                {"token": f"token_id:{tid}", "logprob": lp, "top_logprobs": []}
                                for tid, lp in zip(completion_token_ids, logprobs)
                            ]
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": len(completion_token_ids),
                    "total_tokens": len(prompt_token_ids) + len(completion_token_ids),
                },
            }
            response = ChatCompletion.model_validate(response_dict)

            setattr(response, "prompt_token_ids", prompt_token_ids)
            setattr(response.choices[0], "token_ids", completion_token_ids)

            return response
        finally:
            # Restore original tools setting
            if tools:
                renderer.tools = original_tools


class TinkerCompletions(OpenAIAsyncCompletions):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @overload
    async def create(
        self, *args: Any, stream: Literal[True], **kwargs: Any
    ) -> AsyncStream[Completion]: ...

    @overload
    async def create(
        self, *args: Any, stream: Literal[False] = False, **kwargs: Any
    ) -> Completion: ...

    @overload
    async def create(
        self, *args: Any, stream: bool, **kwargs: Any
    ) -> Completion | AsyncStream[Completion]: ...

    async def create(self, *args: Any, **kwargs: Any) -> Completion | AsyncStream[Completion]:
        stream = bool(kwargs.get("stream", False))
        model = kwargs.get("model", "tinker")
        prompt = kwargs.get("prompt", "")
        sampling_args = {k: v for k, v in kwargs.items() if k not in ("model", "prompt")}

        prompt_token_ids: List[int] = self._parent.tokenizer.encode(prompt, add_special_tokens=True)
        model_input = tinker.ModelInput.from_ints(prompt_token_ids)
        max_completion_tokens = sampling_args.get("max_completion_tokens", None) or sampling_args.get("max_tokens", None)
        max_completion_tokens = int(max_completion_tokens or 128)
        total_tokens = len(prompt_token_ids) + max_completion_tokens
        if total_tokens > self._parent.max_context_length:
            max_completion_tokens = self._parent.max_context_length - len(prompt_token_ids) - 3

            # len(prompt_token_ids) + max_completion_tokens < self._parent.max_context_length
            if max_completion_tokens <= 0:
                raise vf.OverlongPromptError(
                    f"Prompt alone exceeds max context length: "
                    f"{len(prompt_token_ids)} prompt tokens > "
                    f"{self._parent.max_context_length} max context length"
                )


        sample = await sample_with_retries(
            self._parent.sampling_client,
            prompt=model_input,
            sampling_params=tinker.SamplingParams(
                temperature=float(sampling_args.get("temperature", 1.0)),
                max_tokens=max_completion_tokens,
                top_p=float(sampling_args.get("top_p", 1.0)),
                top_k=int(sampling_args.get("top_k", -1)),
            ),
            caller_context=prompt,
        )

        seq = sample.sequences[0]
        completion_token_ids: List[int] = seq.tokens
        logprobs: List[float] = seq.logprobs or [0.0] * len(completion_token_ids)

        text = self._parent.tokenizer.decode(completion_token_ids)
        _update_payload_file(text)
        tokens_str = [f"token_id:{tid}" for tid in completion_token_ids]
        response_dict: Dict[str, Any] = {
            "id": "tinker-cmpl",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": tokens_str,
                        "token_logprobs": logprobs,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(completion_token_ids),
                "total_tokens": len(prompt_token_ids) + len(completion_token_ids),
            },
        }
        response = Completion.model_validate(response_dict)

        setattr(response.choices[0], "prompt_token_ids", prompt_token_ids)
        setattr(response.choices[0], "token_ids", completion_token_ids)

        if stream:
            return TinkerAsyncCompletionStream(response)
        return response


class TinkerAsyncChat(OpenAIAsyncChat):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @property
    def completions(self) -> OpenAIAsyncChatCompletions:
        return TinkerChatCompletions(self._parent)


class TinkerAsyncCompletionStream(AsyncStream[Completion]):
    def __init__(self, final: Completion) -> None:
        self._final = final

    def __aiter__(self):
        self._done = True
        return self

    async def __anext__(self) -> Completion:
        raise StopAsyncIteration

    def __await__(self):
        async def _await_final():
            return self._final

        return _await_final().__await__()

    async def get_final_response(self) -> Completion:
        return self._final
