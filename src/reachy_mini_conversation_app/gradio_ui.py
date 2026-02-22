"""Gradio UI builder for browser and robot-device modes."""

from __future__ import annotations
import os
import queue
import asyncio
import logging
import threading
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.utils import ensure_openai_api_key
from reachy_mini_conversation_app.config import (
    LOCKED_PROFILE,
    API_KEY_SOURCE_ENV,
    config,
    persist_api_key,
    persist_personality,
)
from reachy_mini_conversation_app.local_stream import LocalStream
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import reload_tools_registry
from reachy_mini_conversation_app.headless_personality import (
    DEFAULT_OPTION,
    list_personalities,
    available_tools_for,
    resolve_profile_dir,
)


logger = logging.getLogger(__name__)

_AUTO_WITH: Dict[str, List[str]] = {
    "dance": ["stop_dance"],
    "play_emotion": ["stop_emotion"],
}

_SUPPORTED_REALTIME_VOICES = ["cedar", "marin", "alloy", "echo", "shimmer"]
_BASE_DIR = Path(__file__).resolve().parent
_AVATAR_IMAGES = (
    str(_BASE_DIR / "images" / "user_avatar.png"),
    str(_BASE_DIR / "images" / "reachymini_avatar.png"),
)


def _update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    chatbot.append(response)
    return chatbot


def _current_api_key() -> str:
    """Return the current OpenAI key from config or process env."""
    key = str(getattr(config, "OPENAI_API_KEY", "") or "").strip()
    if key:
        return key
    env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_key:
        try:
            config.OPENAI_API_KEY = env_key
        except Exception:
            pass
    return env_key


def _has_api_key() -> bool:
    return bool(_current_api_key())


def _api_key_status_components() -> tuple[str, str]:
    if not _has_api_key():
        return "⚠️ **No API key set**", "ℹ️ No API key loaded."

    source = (os.getenv(API_KEY_SOURCE_ENV) or "").strip().lower()
    source_message = "Using API key loaded at startup."
    if source in {"huggingface_setup", "huggingface_dataset"}:
        source_message = "Using API key loaded from Hugging Face setup."
    elif source == "settings_ui":
        source_message = "Using API key assigned in Settings."

    return "✅ **API key configured**", f"ℹ️ {source_message}"


def _normalize_voice(voice: Optional[str]) -> str:
    candidate = (voice or "").strip().lower()
    return candidate if candidate in _SUPPORTED_REALTIME_VOICES else "cedar"


def _persist_profile_voice(profile: str, voice: str) -> None:
    if not profile or profile == DEFAULT_OPTION:
        return
    try:
        voice_path = resolve_profile_dir(profile) / "voice.txt"
        voice_path.write_text(_normalize_voice(voice) + "\n", encoding="utf-8")
    except Exception as e:
        logger.debug("Could not persist voice.txt for profile '%s': %s", profile, e)


def _read_voice_for_profile(name: str) -> str:
    if name == DEFAULT_OPTION:
        return "cedar"
    try:
        voice_file = resolve_profile_dir(name) / "voice.txt"
        if voice_file.exists():
            return _normalize_voice(voice_file.read_text(encoding="utf-8").strip())
    except Exception:
        pass
    return "cedar"


def _parse_enabled_tools(text: str) -> list[str]:
    enabled: list[str] = []
    for line in text.splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        enabled.append(value)
    return enabled


def _read_tools_for_profile(name: str) -> list[str]:
    if name == DEFAULT_OPTION:
        return []
    try:
        tools_path = resolve_profile_dir(name) / "tools.txt"
        if tools_path.exists():
            return _parse_enabled_tools(tools_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _apply_auto_dependencies(selected: list[str]) -> list[str]:
    out = list(selected)
    for main_tool, deps in _AUTO_WITH.items():
        if main_tool not in out:
            continue
        for dep in deps:
            if dep not in out:
                out.append(dep)
    return out


def _persist_tools_for_profile(profile: str, tools: list[str]) -> None:
    if profile == DEFAULT_OPTION:
        return
    tools_path = resolve_profile_dir(profile) / "tools.txt"
    if not tools_path.parent.exists():
        return
    try:
        existing = tools_path.read_text(encoding="utf-8") if tools_path.exists() else ""
        comments = [line for line in existing.splitlines() if line.strip().startswith("#")]
        body = "\n".join(tools)
        new_text = (("\n".join(comments) + "\n" if comments else "") + body).strip() + "\n"
        tools_path.write_text(new_text, encoding="utf-8")
    except Exception as e:
        logger.debug("Could not persist tools.txt: %s", e)


def _add_unique_handler(
    handlers: list[Any],
    seen_ids: set[int],
    candidate: Any,
) -> None:
    """Append a handler once, based on object identity."""
    if candidate is None:
        return
    handler_id = id(candidate)
    if handler_id in seen_ids:
        return
    seen_ids.add(handler_id)
    handlers.append(candidate)


def _collect_live_handlers(browser_stream: Stream) -> list[Any]:
    """Collect active handler instances from browser stream internals."""
    handlers: list[Any] = []
    seen_ids: set[int] = set()
    for owner in (browser_stream, getattr(browser_stream, "webrtc_component", None)):
        if owner is None:
            continue

        handlers_dict = getattr(owner, "handlers", None)
        if isinstance(handlers_dict, dict):
            for candidate in handlers_dict.values():
                _add_unique_handler(handlers, seen_ids, candidate)

        connections_dict = getattr(owner, "connections", None)
        if isinstance(connections_dict, dict):
            for callbacks in connections_dict.values():
                if not isinstance(callbacks, list):
                    continue
                for callback in callbacks:
                    _add_unique_handler(
                        handlers,
                        seen_ids,
                        getattr(callback, "event_handler", None),
                    )
    return handlers


async def _apply_personality_on_live_handler(
    live_handler: Any,
    profile_name: Optional[str],
    voice: str,
) -> Optional[str]:
    """Apply profile+voice on a live handler, including cross-loop fallback."""
    apply_fn = getattr(live_handler, "apply_personality", None)
    if apply_fn is None:
        return None

    try:
        message = await apply_fn(profile_name, voice)
        return message if isinstance(message, str) and message else None
    except RuntimeError:
        handler_loop = getattr(live_handler, "_loop", None)
        if handler_loop is None:
            logger.debug("Live apply_personality failed: missing handler loop")
            return None
        try:
            same_loop = handler_loop is asyncio.get_running_loop()
        except Exception:
            same_loop = False
        if same_loop:
            logger.debug("Live apply_personality failed on current loop")
            return None
        try:
            fut = asyncio.run_coroutine_threadsafe(
                apply_fn(profile_name, voice),
                handler_loop,
            )
            message = await asyncio.wrap_future(fut)
            return message if isinstance(message, str) and message else None
        except Exception as e:
            logger.debug("Live apply_personality failed on handler loop: %s", e)
            return None
    except Exception as e:
        logger.debug("Live apply_personality failed: %s", e)
        return None


def _pick_apply_status(status_messages: list[str], base_status: str) -> str:
    """Pick the status message to show when multiple handler statuses exist."""
    if not status_messages:
        return base_status
    preferred = next(
        (msg for msg in status_messages if "restarted realtime session" in msg.lower()),
        None,
    )
    return preferred or status_messages[0]


def _build_browser_conversation_components(
    handler: OpenaiRealtimeHandler,
) -> Stream:
    chatbot = gr.Chatbot(
        type="messages",
        resizable=False,
        height=500,
        avatar_images=_AVATAR_IMAGES,
    )

    stream = Stream(
        handler=handler,
        mode="send-receive",
        modality="audio",
        track_constraints={
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
            }
        },
        additional_inputs=[chatbot],
        additional_outputs=[chatbot],
        additional_outputs_handler=_update_chatbot,
        ui_args={"hide_title": True, "full_screen": False},
    )
    return stream


def _build_tabbed_ui(
    *,
    handler: OpenaiRealtimeHandler,
    instance_path: Optional[str] = None,
    transcript_queue: Optional[queue.Queue[Dict[str, Any]]] = None,
    browser_stream: Optional[Stream] = None,
) -> gr.Blocks:
    """Build the tabbed Gradio UI for conversation and settings."""
    async def _run_handler_call(
        target_handler: OpenaiRealtimeHandler,
        call_factory: Any,
    ) -> Any:
        try:
            return await call_factory()
        except RuntimeError:
            handler_loop = getattr(target_handler, "_loop", None)
            if handler_loop is None:
                raise
            try:
                same_loop = handler_loop is asyncio.get_running_loop()
            except Exception:
                same_loop = False
            if same_loop:
                raise
            future = asyncio.run_coroutine_threadsafe(call_factory(), handler_loop)
            return await asyncio.wrap_future(future)

    with gr.Blocks(title="Reachy Mini Conversation") as blocks:

        with gr.Tabs():

            with gr.Tab("Conversation"):

                if browser_stream is not None:
                    browser_stream.ui.render()  # type: ignore[no-untyped-call]

                else:
                    transcript_chatbot = gr.Chatbot(
                        type="messages",
                        avatar_images=_AVATAR_IMAGES,
                        height=500,
                    )
                    pause_state = gr.State(False)
                    control_status_md = gr.Markdown(value="", visible=False)
                    with gr.Row():
                        pause_btn = gr.Button("Pause Mic", variant="secondary")
                        reload_btn = gr.Button("Reload Session", variant="secondary")

                    transcript_state = gr.State([])

                    def _poll_transcript(
                        transcript: List[Dict[str, Any]],
                    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
                        """Drain transcript queue and append new messages."""
                        if transcript_queue is None:
                            return transcript, transcript

                        updated = list(transcript)

                        while True:
                            try:
                                message = transcript_queue.get_nowait()
                            except queue.Empty:
                                break

                            if not isinstance(message, dict):
                                continue

                            role = message.get("role")
                            content = message.get("content")

                            if not isinstance(role, str) or role.endswith("_partial"):
                                continue

                            entry: Dict[str, Any] = {
                                "role": role,
                                "content": content,
                            }

                            metadata = message.get("metadata")
                            if isinstance(metadata, dict):
                                entry["metadata"] = metadata

                            updated.append(entry)

                        return updated, updated

                    gr.Timer(value=0.25).tick(
                        fn=_poll_transcript,
                        inputs=[transcript_state],
                        outputs=[transcript_chatbot, transcript_state],
                        queue=False,
                    )

                    async def _on_toggle_pause(current_paused: bool) -> tuple[bool, Any, str]:
                        next_paused = not bool(current_paused)
                        paused = bool(
                            await _run_handler_call(
                                handler,
                                lambda: handler.set_input_paused(next_paused),
                            )
                        )
                        if paused:
                            gr.Info("Microphone paused.", duration=3)
                        else:
                            gr.Info("Microphone resumed.", duration=3)
                        return paused, gr.update(value=("Resume Mic" if paused else "Pause Mic")), ""

                    async def _on_reload_session() -> str:
                        await _run_handler_call(handler, handler.reload_session)
                        gr.Info("Realtime session reloaded.", duration=4)
                        return ""

                    pause_btn.click(
                        fn=_on_toggle_pause,
                        inputs=[pause_state],
                        outputs=[pause_state, pause_btn, control_status_md],
                    )
                    reload_btn.click(
                        fn=_on_reload_session,
                        inputs=[],
                        outputs=[control_status_md],
                    )

            with gr.Tab("Settings"):

                key_status, key_source = _api_key_status_components()

                key_status_md = gr.Markdown(value=key_status)
                key_source_md = gr.Markdown(value=key_source)

                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="sk-...",
                        scale=4,
                    )
                    save_key_btn = gr.Button("Validate & Save", scale=1)

                key_feedback_md = gr.Markdown()

                async def _validate_and_save_key(key: str) -> tuple[str, str, str, Any]:
                    """Validate API key against OpenAI and persist if valid."""
                    k = (key or "").strip()

                    if not k:
                        status, source = _api_key_status_components()
                        return (
                            status,
                            source,
                            "Please enter an API key.",
                            gr.update(value=""),
                        )

                    try:
                        import httpx

                        headers = {
                            "Authorization": f"Bearer {k}",
                            "Content-Type": "application/json",
                        }

                        async with httpx.AsyncClient(timeout=10.0) as client:
                            response = await client.get(
                                "https://api.openai.com/v1/models",
                                headers=headers,
                            )

                        if response.status_code == 401:
                            status, source = _api_key_status_components()
                            return (
                                status,
                                source,
                                "Invalid API key.",
                                gr.update(),
                            )

                        if response.status_code != 200:
                            status, source = _api_key_status_components()
                            return (
                                status,
                                source,
                                f"Validation failed (HTTP {response.status_code}).",
                                gr.update(),
                            )

                    except Exception as exc:
                        status, source = _api_key_status_components()
                        return (
                            status,
                            source,
                            f"Validation error: {exc}",
                            gr.update(),
                        )

                    persist_api_key(k, instance_path, source="settings_ui", custom_logger=logger)
                    status, source = _api_key_status_components()

                    return (
                        status,
                        source,
                        "API key saved.",
                        gr.update(value=""),
                    )

                save_key_btn.click(
                    fn=_validate_and_save_key,
                    inputs=[api_key_input],
                    outputs=[
                        key_status_md,
                        key_source_md,
                        key_feedback_md,
                        api_key_input,
                    ],
                )

                gr.Timer(value=1.5).tick(
                    fn=_api_key_status_components,
                    inputs=[],
                    outputs=[key_status_md, key_source_md],
                    queue=False,
                )

                is_locked = LOCKED_PROFILE is not None
                if is_locked:
                    gr.Markdown(
                        f"ℹ️ Profile switching is locked to `{LOCKED_PROFILE}`.",
                    )

                choices: list[str] = (
                    [LOCKED_PROFILE]  # type: ignore[list-item]
                    if is_locked
                    else [DEFAULT_OPTION, *list_personalities()]
                )

                current_profile: str = (
                    LOCKED_PROFILE  # type: ignore[assignment]
                    if is_locked
                    else (config.REACHY_MINI_CUSTOM_PROFILE or DEFAULT_OPTION)
                )

                if current_profile not in choices:
                    current_profile = choices[0]

                with gr.Row():
                    profile_dropdown = gr.Dropdown(
                        label="Profile",
                        choices=choices,
                        value=current_profile,
                        interactive=not is_locked,
                        scale=3,
                    )

                    voice_dropdown = gr.Dropdown(
                        label="Voice",
                        choices=_SUPPORTED_REALTIME_VOICES,
                        value=_read_voice_for_profile(current_profile),
                        interactive=not is_locked,
                        scale=2,
                    )

                tools_checkbox = gr.CheckboxGroup(
                    label="Active tools",
                    choices=available_tools_for(current_profile),
                    value=_read_tools_for_profile(current_profile),
                    interactive=not is_locked,
                )

                apply_btn = gr.Button(
                    "Apply Changes",
                    interactive=not is_locked,
                )

                profile_status_md = gr.Markdown()

                def _on_profile_change(selected: str) -> tuple[Any, Any, str]:
                    """Update voice and tools when profile changes."""
                    return (
                        gr.update(value=_read_voice_for_profile(selected)),
                        gr.update(
                            choices=available_tools_for(selected),
                            value=_read_tools_for_profile(selected),
                        ),
                        "",
                    )

                profile_dropdown.change(
                    fn=_on_profile_change,
                    inputs=[profile_dropdown],
                    outputs=[voice_dropdown, tools_checkbox, profile_status_md],
                )

                async def _on_apply(
                    selected: str,
                    voice_selected: str,
                    tools_selected: list[str],
                ) -> str:
                    """Apply selected profile, voice, and tools."""
                    try:
                        selected_profile = selected or DEFAULT_OPTION
                        final_voice = _normalize_voice(voice_selected)
                        final_tools = _apply_auto_dependencies(tools_selected or [])
                        profile_name = None if selected_profile == DEFAULT_OPTION else selected_profile

                        _persist_profile_voice(selected_profile, final_voice)
                        _persist_tools_for_profile(selected_profile, final_tools)
                        persist_personality(profile_name, instance_path, custom_logger=logger)
                        reload_tools_registry()

                        status_messages: list[str] = []
                        if browser_stream is not None:
                            for live_handler in _collect_live_handlers(browser_stream):
                                msg = await _apply_personality_on_live_handler(
                                    live_handler,
                                    profile_name,
                                    final_voice,
                                )
                                if msg:
                                    status_messages.append(msg)

                        if status_messages:
                            status = _pick_apply_status(status_messages, status_messages[0])
                        else:
                            status = await handler.apply_personality(
                                profile_name,
                                final_voice,
                            )

                        status_lower = status.lower() if isinstance(status, str) else ""
                        if "restarted realtime session" in status_lower:
                            gr.Info("Applied changes and restarted realtime session.", duration=5)
                        elif "next connection" in status_lower:
                            gr.Warning("Applied changes. They will take effect on next connection.", duration=8)
                        else:
                            gr.Info("Changes applied.", duration=5)

                        return ""  # clear the status markdown

                    except Exception as exc:
                        raise gr.Error(f"Failed to apply changes: {exc}", duration=10)

                apply_btn.click(
                    fn=_on_apply,
                    inputs=[profile_dropdown, voice_dropdown, tools_checkbox],
                    outputs=[profile_status_md],
                )

        return blocks  # type: ignore[no-any-return]


class RobotDeviceGradioManager:
    """Launch robot-device audio stream and Gradio UI together (CLI mode)."""

    def __init__(self, *, ui: gr.Blocks, stream: LocalStream):
        """Store UI and local stream references for coordinated lifecycle management."""
        self._ui = ui
        self._stream = stream
        self._stream_thread: Optional[threading.Thread] = None
        self._closed = False
        self._close_lock = threading.Lock()

    def launch(self) -> None:
        """Start LocalStream in background, then block on Gradio UI."""
        if self._stream_thread is None or not self._stream_thread.is_alive():
            self._stream_thread = threading.Thread(
                target=self._stream.launch,
                name="robot-device-local-stream",
                daemon=True,
            )
            self._stream_thread.start()

        try:
            self._ui.launch(server_name="127.0.0.1", server_port=7860)
        finally:
            self.close()

    def close(self) -> None:
        """Close both UI and local stream."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        try:
            self._stream.close()
        except Exception as e:
            logger.debug("Ignoring LocalStream close error: %s", e)

        if (
            self._stream_thread is not None
            and self._stream_thread.is_alive()
            and threading.current_thread() is not self._stream_thread
        ):
            self._stream_thread.join(timeout=5)

        try:
            self._ui.close()
        except Exception as e:
            logger.debug("Ignoring Gradio UI close error: %s", e)


def build_gradio_ui(
    *,
    handler: OpenaiRealtimeHandler,
    robot: ReachyMini,
    settings_app: Optional[FastAPI],
    instance_path: Optional[str],
    audio_source: Literal["browser", "robot_device"],
) -> gr.Blocks | LocalStream | RobotDeviceGradioManager:
    """Build the Gradio UI and launch manager."""
    ensure_openai_api_key(
        instance_path,
        persist_key=lambda key: persist_api_key(
            key,
            instance_path,
            source="huggingface_setup",
            custom_logger=logger,
        ),
        load_profile=True,
        logger=logger,
    )

    if audio_source == "browser":
        browser_stream = _build_browser_conversation_components(handler)
        tabbed_browser_ui = _build_tabbed_ui(
            handler=handler,
            instance_path=instance_path,
            transcript_queue=None,
            browser_stream=browser_stream,
        )
        if settings_app is not None:
            gr.mount_gradio_app(settings_app, tabbed_browser_ui, path="/")
        return tabbed_browser_ui

    transcript_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

    def _on_transcript_message(message: dict[str, Any]) -> None:
        transcript_queue.put(message)

    robot_ui = _build_tabbed_ui(
        handler=handler,
        instance_path=instance_path,
        transcript_queue=transcript_queue,
    )

    if settings_app is not None:
        local_stream = LocalStream(
            handler,
            robot,
            settings_app=None,
            instance_path=instance_path,
            on_transcript_message=_on_transcript_message,
        )
        gr.mount_gradio_app(settings_app, robot_ui, path="/")
        return local_stream

    local_stream = LocalStream(
        handler,
        robot,
        settings_app=None,
        instance_path=instance_path,
        on_transcript_message=_on_transcript_message,
    )

    return RobotDeviceGradioManager(ui=robot_ui, stream=local_stream)
