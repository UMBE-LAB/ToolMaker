from collections.abc import Sequence
from typing import Protocol

import litellm
import tenacity
import os
import functools
from litellm.types.utils import ModelResponse
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from pydantic import BaseModel

from toolmaker.utils.llm import pydantic_to_function_schema


class ChatCompletion(Protocol):
    """Basically the signature of `litellm.completion`."""

    def __call__(
        self,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> ModelResponse: ...


# Note: the function must be passed in a second call; `tenacity.retry(fn, **kwargs)`
# silently ignores kwargs and retries every exception forever.
litellm_completion_retry_on_api_error: ChatCompletion = tenacity.retry(
    retry=tenacity.retry_if_exception_type(litellm.APIError),
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(2),
    after=tenacity.after_log(logger, logger.level("WARNING").no),
    reraise=True,
)(litellm.completion)

litellm_completion_retry_on_rate_limit: ChatCompletion = tenacity.retry(
    retry=tenacity.retry_if_exception_type(litellm.RateLimitError),
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(60),
    after=tenacity.after_log(logger, logger.level("WARNING").no),
    reraise=True,
)(litellm.completion)


CLAUDE_MAX_FINISH_NUDGES = 3


def _make_finish_tool(response_format: type[BaseModel]) -> type[BaseModel]:
    class Finish(BaseModel):
        """This tool ends the conversation turn. Use it to respond to the user (in the format requested by the user) once you are ready to do so (i.e. after calling other tools, if applicable)."""

        response: response_format

    return Finish


def claude_completion(
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    tools: Sequence[ChatCompletionToolParam] | None = None,
    response_format: type[BaseModel] | None = None,
) -> ModelResponse:
    FINISH_TOOL_NAME = "__finish__"
    if not tools:
        # return litellm.completion(
        #     model=model, messages=messages, response_format=response_format
        # )
        tools = []
    # Claude doesn't support specifying response_format and tools at the same time,
    # so we need to add a tool that will return the response in the correct format.
    logger.debug(f"Using Claude completion (with an extra {FINISH_TOOL_NAME} tool)")
    finish_tool = _make_finish_tool(
        response_format if response_format is not None else str
    )
    tools = [*tools, pydantic_to_function_schema(finish_tool, name=FINISH_TOOL_NAME)]
    # Newer Claude models reject forced tool use (tool_choice "any"/"tool"), so let the model
    # choose. A plain-text reply is a valid answer for untyped calls; for typed calls, nudge
    # the model to call the finish tool instead.
    messages = list(messages)
    for _ in range(CLAUDE_MAX_FINISH_NUDGES + 1):
        response = litellm_completion_retry_on_rate_limit(
            model=model, messages=messages, tools=tools, tool_choice="auto"
        )
        message = response.choices[0].message
        if message.tool_calls or response_format is None:
            break
        logger.debug(f"Claude replied without a tool call; asking it to call {FINISH_TOOL_NAME}")
        if message.content:
            messages.append({"role": "assistant", "content": message.content})
        messages.append(
            {
                "role": "user",
                "content": f"Please respond by calling a tool. If you are done, call the `{FINISH_TOOL_NAME}` tool with your final response.",
            }
        )
    else:
        raise RuntimeError(
            f"Claude did not call a tool after {CLAUDE_MAX_FINISH_NUDGES} reminders"
        )
    finish_call: ChatCompletionMessageToolCall | None = None
    other_calls: list[ChatCompletionMessageToolCall] = []
    for call in message.tool_calls or ():
        if call.function.name == FINISH_TOOL_NAME:
            finish_call = finish_call or call
        else:
            other_calls.append(call)
    if other_calls:
        # Run the regular tools first; the model can call the finish tool on a later turn.
        message.tool_calls = other_calls
        # With tool_choice="auto", Claude often narrates before a tool call. Drop that text,
        # otherwise it is parsed as the (typed) final response.
        message.content = None
    elif finish_call:
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = "stop"
        parsed = finish_tool.model_validate_json(finish_call.function.arguments)
        response.choices[0].message.content = (
            parsed.response.model_dump_json()
            if isinstance(parsed.response, BaseModel)
            else parsed.response
        )
    return response


def completion_factory(model: str) -> ChatCompletion:
    if model == "o3-mini":
        # Sometimes o3-mini is flaky, so we retry a few times
        return litellm_completion_retry_on_api_error
    elif "claude" in model:
        return claude_completion
    elif "ollama" in model:
        return functools.partial(litellm.completion, api_base=os.getenv("LOCAL_LLM_ENDPOINT"))
    return litellm.completion
