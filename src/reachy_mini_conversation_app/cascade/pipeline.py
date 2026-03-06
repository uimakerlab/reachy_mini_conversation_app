"""LLM response processing and tool execution pipeline."""

from __future__ import annotations
import json
import base64
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List
from dataclasses import dataclass

from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.tts import TTSProvider
from reachy_mini_conversation_app.cascade.config import get_config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, dispatch_tool_call
from reachy_mini_conversation_app.cascade.turn_result import TurnItem, PipelineResult


if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.speech_output import SpeechOutput


logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Bundle of references passed through the LLM/tool pipeline."""

    llm: LLMProvider
    tts: TTSProvider
    speech_output: SpeechOutput | None
    conversation_history: list[dict[str, Any]]
    tool_specs: list[dict[str, Any]]
    deps: ToolDependencies
    result: PipelineResult


def _track_cost(ctx: PipelineContext, provider: Any) -> None:
    """Accumulate provider cost into the pipeline result."""
    if hasattr(provider, "last_cost") and provider.last_cost > 0:
        ctx.result.cost += provider.last_cost
        provider.last_cost = 0.0


async def process_llm_response(ctx: PipelineContext) -> PipelineResult:
    """Process LLM response with retry on failure."""
    max_retries = 2
    for attempt in range(1 + max_retries):
        try:
            await _process_llm_response_once(ctx)
            return ctx.result
        except Exception as e:
            if attempt < max_retries:
                logger.warning("LLM failed (attempt %d), retrying: %s", attempt + 1, e)
                if ctx.speech_output:
                    await ctx.speech_output.speak("Give me a moment.")
                await asyncio.sleep(2)
            else:
                logger.error("LLM failed after %d attempts: %s", max_retries + 1, e)
                if ctx.speech_output:
                    await ctx.speech_output.speak("Sorry, I'm having trouble responding right now.")
    return ctx.result


async def _process_llm_response_once(ctx: PipelineContext) -> None:
    """Single attempt at processing LLM response with streaming, tool calls, and TTS."""
    # Generate streaming response
    text_chunks: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    async for chunk in ctx.llm.generate(
        messages=ctx.conversation_history,
        tools=ctx.tool_specs,
        temperature=get_config().llm_temperature,
    ):
        if chunk.type == "text_delta" and chunk.content:
            text_chunks.append(chunk.content)
            logger.debug(f"LLM text delta: {chunk.content}")

        elif chunk.type == "tool_call" and chunk.tool_call:
            tool_calls.append(chunk.tool_call)
            logger.info(f"LLM tool call: {chunk.tool_call}")

        elif chunk.type == "done":
            logger.debug("LLM generation complete")
            break

    # Aggregate LLM cost after generator completes
    _track_cost(ctx, ctx.llm)

    # Create assistant message with text, tool calls...
    assistant_message: Dict[str, Any] = {"role": "assistant"}
    full_text = ""
    if text_chunks:
        full_text = "".join(text_chunks)
        assistant_message["content"] = full_text
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls

    logger.debug(
        f"process_llm_response: text_chunks={len(text_chunks)}, tool_calls={len(tool_calls)}, full_text_len={len(full_text)}"
    )

    if text_chunks or tool_calls:
        ctx.conversation_history.append(assistant_message)
        logger.debug(f"Added assistant message to history, history_len={len(ctx.conversation_history)}")

    # Handle text-only responses: auto-inject speak tool call
    # This handles cases where LLM returns text without using the speak tool
    # In principle it should not happen thanks to the extra instructions.
    # If it happens, we create a synthetic tool call for speaking
    if full_text and not tool_calls:
        logger.info("❓LLM returned text without speak tool - auto-injecting speak call")

        synthetic_tool_call = {
            "id": "auto_speak",
            "type": "function",
            "function": {"name": "speak", "arguments": json.dumps({"message": full_text})},
        }
        await execute_tool_calls([synthetic_tool_call], ctx)
    elif tool_calls and not any(tc.get("function", {}).get("name") == "speak" for tc in tool_calls):
        # Tool calls but no speak — record assistant text if present
        if full_text:
            ctx.result.turn_items.append(TurnItem(kind="assistant", text=full_text))
    if tool_calls:
        # Process normal tool calls
        await execute_tool_calls(tool_calls, ctx)


async def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    ctx: PipelineContext,
) -> None:
    """Execute tool calls and handle camera/see_image and speak tool specially."""
    camera_image_bytes: bytes | None = None

    # First pass: execute all tools and add ALL tool results to conversation
    # This must be done before adding any other messages (OpenAI requires all tool
    # responses immediately after the assistant message with tool_calls)
    for tool_call in tool_calls:
        call_id = ""
        tool_name = "unknown"
        try:
            call_id, tool_name, arguments = ctx.llm.parse_tool_call(tool_call)

            logger.info(f"Executing tool: {tool_name}({arguments})")

            # Execute tool
            result = await dispatch_tool_call(
                tool_name,
                json.dumps(arguments),
                ctx.deps,
            )

            # Do not log full result if the tool returned base64 (huge)
            if tool_name in ("camera", "see_image") and "b64_im" in result:
                logger.info("Tool result: [image in base64, not shown]")
            else:
                logger.info(f"Tool result: {result}")

            # Add tool result to conversation
            ctx.conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": json.dumps(result),
                }
            )
            logger.debug(
                f"Added tool result to history: name={tool_name}, history_len={len(ctx.conversation_history)}"
            )

            # Special handling for see_image tool - store frame, replace heavy b64
            if tool_name == "see_image":
                if "b64_im" in result:
                    b64_im = result["b64_im"]
                    camera_image_bytes = base64.b64decode(b64_im)
                    frame_index = len(ctx.result.captured_frames)
                    ctx.result.captured_frames.append(camera_image_bytes)
                    # Replace the heavy b64 blob in conversation history with a lightweight marker
                    ctx.conversation_history[-1]["content"] = json.dumps(
                        {"status": "image_captured", "frame_index": frame_index}
                    )
                    ctx.result.turn_items.append(TurnItem(kind="image", image_jpeg=camera_image_bytes))
                    logger.info("see_image: stored frame %d, will add image to conversation", frame_index)
                else:
                    logger.warning(f"see_image returned error: {result}")

            # Special handling for camera tool (backward compat) - store image for later
            elif tool_name == "camera":
                if "b64_im" in result:
                    b64_im = result["b64_im"]
                    logger.info("Camera tool executed - will add image to conversation for LLM analysis")
                    camera_image_bytes = base64.b64decode(b64_im)
                    frame_index = len(ctx.result.captured_frames)
                    ctx.result.captured_frames.append(camera_image_bytes)
                    # Replace the heavy b64 blob in conversation history with a lightweight marker
                    ctx.conversation_history[-1]["content"] = json.dumps(
                        {"status": "image_captured", "frame_index": frame_index}
                    )
                    ctx.result.turn_items.append(TurnItem(kind="image", image_jpeg=camera_image_bytes))
                else:
                    # Camera failed - error already in tool result, LLM will see it
                    logger.warning(f"Camera tool returned error: {result}")

            # Special handling for speak tool
            elif tool_name == "speak" and "message" in result:
                message = result["message"]
                logger.info(f"Speaking: {message}")
                ctx.result.turn_items.append(TurnItem(kind="speak", text=message))

                if ctx.speech_output:
                    await ctx.speech_output.speak(message)
                _track_cost(ctx, ctx.tts)

            # Other tools
            elif tool_name not in ("speak", "see_image", "camera"):
                ctx.result.turn_items.append(
                    TurnItem(kind="tool", tool_name=tool_name, tool_content=json.dumps(result))
                )

        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")

            # Add error to conversation
            ctx.conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": json.dumps({"error": str(e)}),
                }
            )

    # After all tool results are added, add camera image as user message and call LLM
    if camera_image_bytes is not None:
        ctx.conversation_history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": camera_image_bytes,  # Will be converted to provider format in LLM
                    }
                ],
            }
        )
        logger.info("Camera image added to conversation - calling LLM to analyze it")
        await process_llm_response(ctx)
