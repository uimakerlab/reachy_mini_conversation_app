"""LLM response processing and tool execution pipeline."""

from __future__ import annotations
import json
import base64
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Callable

from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.tts import TTSProvider
from reachy_mini_conversation_app.cascade.config import get_config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, dispatch_tool_call
from reachy_mini_conversation_app.cascade.turn_result import TurnItem


if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.speech_output import SpeechOutput


logger = logging.getLogger(__name__)


async def process_llm_response(
    llm: LLMProvider,
    conversation_history: list[dict[str, Any]],
    tool_specs: list[dict[str, Any]],
    speech_output: SpeechOutput | None,
    current_turn_items: list[TurnItem],
    captured_frames: list[bytes],
    deps: ToolDependencies,
    aggregate_cost_fn: Callable,
    tts: TTSProvider,
) -> None:
    """Process LLM response with streaming, tool calls, and TTS."""
    try:
        # Generate streaming response
        text_chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        async for chunk in llm.generate(
            messages=conversation_history,
            tools=tool_specs,
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
        aggregate_cost_fn(llm, "LLM")

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
            conversation_history.append(assistant_message)
            logger.debug(f"Added assistant message to history, history_len={len(conversation_history)}")

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
            await execute_tool_calls(
                [synthetic_tool_call], llm, conversation_history, tool_specs,
                speech_output, current_turn_items, captured_frames, deps,
                aggregate_cost_fn, tts,
            )
        elif tool_calls and not any(tc.get("function", {}).get("name") == "speak" for tc in tool_calls):
            # Tool calls but no speak — record assistant text if present
            if full_text:
                current_turn_items.append(TurnItem(kind="assistant", text=full_text))
        if tool_calls:
            # Process normal tool calls
            await execute_tool_calls(
                tool_calls, llm, conversation_history, tool_specs,
                speech_output, current_turn_items, captured_frames, deps,
                aggregate_cost_fn, tts,
            )

    except Exception as e:
        logger.exception(f"Error processing LLM response: {e}")


async def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    llm: LLMProvider,
    conversation_history: list[dict[str, Any]],
    tool_specs: list[dict[str, Any]],
    speech_output: SpeechOutput | None,
    current_turn_items: list[TurnItem],
    captured_frames: list[bytes],
    deps: ToolDependencies,
    aggregate_cost_fn: Callable,
    tts: TTSProvider,
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
            call_id, tool_name, arguments = llm.parse_tool_call(tool_call)

            logger.info(f"Executing tool: {tool_name}({arguments})")

            # Execute tool
            result = await dispatch_tool_call(
                tool_name,
                json.dumps(arguments),
                deps,
            )

            # Do not log full result if the tool returned base64 (huge)
            if tool_name in ("camera", "see_image") and "b64_im" in result:
                logger.info("Tool result: [image in base64, not shown]")
            else:
                logger.info(f"Tool result: {result}")

            # Add tool result to conversation
            conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": json.dumps(result),
                }
            )
            logger.debug(
                f"Added tool result to history: name={tool_name}, history_len={len(conversation_history)}"
            )

            # Special handling for see_image tool - store frame, replace heavy b64
            if tool_name == "see_image":
                if "b64_im" in result:
                    b64_im = result["b64_im"]
                    camera_image_bytes = base64.b64decode(b64_im)
                    frame_index = len(captured_frames)
                    captured_frames.append(camera_image_bytes)
                    # Replace the heavy b64 blob in conversation history with a lightweight marker
                    conversation_history[-1]["content"] = json.dumps(
                        {"status": "image_captured", "frame_index": frame_index}
                    )
                    current_turn_items.append(TurnItem(kind="image", image_jpeg=camera_image_bytes))
                    logger.info("see_image: stored frame %d, will add image to conversation", frame_index)
                else:
                    logger.warning(f"see_image returned error: {result}")

            # Special handling for camera tool (backward compat) - store image for later
            elif tool_name == "camera":
                if "b64_im" in result:
                    b64_im = result["b64_im"]
                    logger.info("Camera tool executed - will add image to conversation for LLM analysis")
                    camera_image_bytes = base64.b64decode(b64_im)
                    frame_index = len(captured_frames)
                    captured_frames.append(camera_image_bytes)
                    # Replace the heavy b64 blob in conversation history with a lightweight marker
                    conversation_history[-1]["content"] = json.dumps(
                        {"status": "image_captured", "frame_index": frame_index}
                    )
                    current_turn_items.append(TurnItem(kind="image", image_jpeg=camera_image_bytes))
                else:
                    # Camera failed - error already in tool result, LLM will see it
                    logger.warning(f"Camera tool returned error: {result}")

            # Special handling for speak tool
            elif tool_name == "speak" and "message" in result:
                message = result["message"]
                logger.info(f"Speaking: {message}")
                current_turn_items.append(TurnItem(kind="speak", text=message))

                if speech_output:
                    await speech_output.speak(message)
                aggregate_cost_fn(tts, "TTS")

            # Other tools
            elif tool_name not in ("speak", "see_image", "camera"):
                current_turn_items.append(
                    TurnItem(kind="tool", tool_name=tool_name, tool_content=json.dumps(result))
                )

        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")

            # Add error to conversation
            conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": json.dumps({"error": str(e)}),
                }
            )

    # After all tool results are added, add camera image as user message and call LLM
    if camera_image_bytes is not None:
        conversation_history.append(
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
        await process_llm_response(
            llm, conversation_history, tool_specs, speech_output,
            current_turn_items, captured_frames, deps, aggregate_cost_fn, tts,
        )
