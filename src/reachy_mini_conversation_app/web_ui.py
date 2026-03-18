"""Web UI server: FastAPI + WebSocket replacing Gradio.

Serves a React/MUI frontend and streams bidirectional audio
to the OpenAI Realtime handler over a single WebSocket.

Protocol
--------
Client -> Server:
    binary  : raw PCM int16 mono audio chunk (sample rate set by WEB_INPUT_SAMPLE_RATE)
    text    : JSON control message  {"type": "start"} / {"type": "stop"}

Server -> Client:
    binary  : raw PCM int16 mono 24 kHz audio chunk
    text    : JSON  {"type": "transcript"|"tool"|"image"|"interrupt"|"status", ...}
"""

import json
import base64
import asyncio
import logging
from typing import Any, Union, Optional
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastrtc import AdditionalOutputs
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from reachy_mini_conversation_app.config import config, set_custom_profile
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.headless_personality import (
    DEFAULT_OPTION,
    _sanitize_name,
    _write_profile,
    list_personalities,
    available_tools_for,
    resolve_profile_dir,
    read_instructions_for,
)


logger = logging.getLogger(__name__)

WEB_APP_DIR = Path(__file__).parent / "web-app"
SETTINGS_STATIC_DIR = Path(__file__).parent / "static"

WEB_INPUT_SAMPLE_RATE = 24000

_INTERRUPT_SENTINEL = "__interrupt__"
QueueItem = Union[tuple[int, Any], AdditionalOutputs, str]


class WebUI:
    """Lightweight web server that bridges a React frontend to the realtime handler."""

    def __init__(
        self,
        handler: OpenaiRealtimeHandler,
        host: str = "0.0.0.0",
        port: int = 7860,
        instance_path: Optional[str] = None,
    ):
        """Initialize the web UI server with a realtime handler."""
        self.handler = handler
        self.host = host
        self.port = port
        self._instance_path = instance_path
        self.app = FastAPI(title="Reachy Mini Conversation")
        self._active_handler: Optional[OpenaiRealtimeHandler] = None
        self._setup_middleware()
        self._setup_api_routes()
        self._setup_ws_and_static()

    def _setup_middleware(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------------
    # Startup personality persistence (same pattern as console.py)
    # ------------------------------------------------------------------
    def _read_env_lines(self, env_path: Path) -> list[str]:
        """Load env file contents or a template as a list of lines."""
        try:
            if env_path.exists():
                return env_path.read_text(encoding="utf-8").splitlines()
            for candidate in [
                env_path.parent / ".env.example",
                Path.cwd() / ".env.example",
                Path(__file__).parent / ".env.example",
            ]:
                if candidate.exists():
                    return candidate.read_text(encoding="utf-8").splitlines()
        except Exception:
            pass
        return []

    def _persist_personality(self, profile: Optional[str]) -> None:
        """Persist the startup personality to the instance .env and config."""
        selection = (profile or "").strip() or None
        set_custom_profile(selection)
        if not self._instance_path:
            return
        try:
            env_path = Path(self._instance_path) / ".env"
            lines = self._read_env_lines(env_path)
            replaced = False
            for i, ln in enumerate(list(lines)):
                if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                    if selection:
                        lines[i] = f"REACHY_MINI_CUSTOM_PROFILE={selection}"
                    else:
                        lines.pop(i)
                    replaced = True
                    break
            if selection and not replaced:
                lines.append(f"REACHY_MINI_CUSTOM_PROFILE={selection}")
            if selection is None and not env_path.exists():
                return
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Persisted startup personality to %s", env_path)
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist REACHY_MINI_CUSTOM_PROFILE: %s", e)

    def _read_persisted_personality(self) -> Optional[str]:
        """Read persisted startup personality from instance .env (if any)."""
        if not self._instance_path:
            return None
        env_path = Path(self._instance_path) / ".env"
        try:
            if env_path.exists():
                for ln in env_path.read_text(encoding="utf-8").splitlines():
                    if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                        _, _, val = ln.partition("=")
                        v = val.strip()
                        return v or None
        except Exception:
            pass
        return None

    def _startup_choice(self) -> str:
        """Return the persisted startup personality or default."""
        persisted = self._read_persisted_personality()
        if persisted:
            return persisted
        env_val = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        return env_val if env_val else DEFAULT_OPTION

    # ------------------------------------------------------------------
    # REST API for settings (personalities, voices, API key)
    # ------------------------------------------------------------------
    def _setup_api_routes(self) -> None:

        @self.app.get("/api/config")
        def api_config() -> dict[str, Any]:
            """Expose the OpenAI API key (from HF Secret or env) to the TS frontend."""
            import os
            key = os.environ.get("OPENAI_API_KEY", "")
            api_key = key.strip() if key.strip() else None
            if not api_key:
                api_key = str(config.OPENAI_API_KEY).strip() if config.OPENAI_API_KEY else None
            return {"openai_api_key": api_key}

        @self.app.get("/api/status")
        def api_status() -> dict[str, Any]:
            has_key = bool(config.OPENAI_API_KEY and str(config.OPENAI_API_KEY).strip())
            cur = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or DEFAULT_OPTION
            return {"has_key": has_key, "current_profile": cur}

        @self.app.get("/api/personalities")
        def api_personalities() -> dict[str, Any]:
            choices = [DEFAULT_OPTION, *list_personalities()]
            cur = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or DEFAULT_OPTION
            return {"choices": choices, "current": cur, "startup": self._startup_choice()}

        @self.app.get("/api/personalities/load")
        def api_load_personality(name: str) -> dict[str, Any]:
            instr = read_instructions_for(name)
            tools_txt = ""
            voice = "cedar"
            if name != DEFAULT_OPTION:
                pdir = resolve_profile_dir(name)
                tp = pdir / "tools.txt"
                if tp.exists():
                    tools_txt = tp.read_text(encoding="utf-8")
                vf = pdir / "voice.txt"
                if vf.exists():
                    v = vf.read_text(encoding="utf-8").strip()
                    voice = v or "cedar"
            avail = available_tools_for(name)
            enabled = [ln.strip() for ln in tools_txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            return {
                "instructions": instr,
                "tools_text": tools_txt,
                "voice": voice,
                "available_tools": avail,
                "enabled_tools": enabled,
            }

        @self.app.post("/api/personalities/save")
        async def api_save_personality(request: Request) -> JSONResponse:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            name = str(raw.get("name", ""))
            instructions = str(raw.get("instructions", ""))
            tools_text = str(raw.get("tools_text", ""))
            voice = str(raw.get("voice", "cedar")) if raw.get("voice") is not None else "cedar"

            name_s = _sanitize_name(name)
            if not name_s:
                return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)
            try:
                _write_profile(name_s, instructions, tools_text, voice or "cedar")
                value = f"user_personalities/{name_s}"
                choices = [DEFAULT_OPTION, *list_personalities()]
                return JSONResponse({"ok": True, "value": value, "choices": choices})
            except Exception as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

        @self.app.post("/api/personalities/apply")
        async def api_apply_personality(request: Request) -> JSONResponse:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            sel_name = str(raw.get("name", DEFAULT_OPTION))
            persist = bool(raw.get("persist", False))
            handler = self._active_handler or self.handler
            sel = None if sel_name == DEFAULT_OPTION else sel_name
            try:
                status = await handler.apply_personality(sel)
                startup = self._startup_choice()
                if persist:
                    self._persist_personality(sel)
                    startup = self._startup_choice()
                return JSONResponse({"ok": True, "status": status, "startup": startup})
            except Exception as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

        @self.app.get("/api/voices")
        async def api_voices() -> list[str]:
            handler = self._active_handler or self.handler
            try:
                return await handler.get_available_voices()
            except Exception:
                return ["cedar"]

        @self.app.post("/api/validate_api_key")
        async def api_validate_key(request: Request) -> JSONResponse:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            key = str(raw.get("openai_api_key", "")).strip()
            if not key:
                return JSONResponse({"valid": False, "error": "empty_key"}, status_code=400)
            try:
                import httpx
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get("https://api.openai.com/v1/models", headers=headers)
                    if response.status_code == 200:
                        return JSONResponse({"valid": True})
                    elif response.status_code == 401:
                        return JSONResponse({"valid": False, "error": "invalid_api_key"}, status_code=401)
                    else:
                        return JSONResponse({"valid": False, "error": "validation_failed"}, status_code=response.status_code)
            except Exception as e:
                logger.warning("API key validation failed: %s", e)
                return JSONResponse({"valid": False, "error": "validation_error"}, status_code=500)

        @self.app.get("/api/camera/snapshot")
        def api_camera_snapshot() -> Response:
            """Return the latest camera frame as a base64-encoded JPEG."""
            try:
                cam = self.handler.deps.camera_worker
                if cam is None:
                    return JSONResponse({"error": "Camera worker not available"}, status_code=503)
                frame = cam.get_latest_frame()
                if frame is None:
                    return JSONResponse({"error": "No frame available"}, status_code=503)
                ok, buf = cv2.imencode(".jpg", frame)
                if not ok:
                    return JSONResponse({"error": "Failed to encode frame"}, status_code=500)
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                return JSONResponse({"b64": b64})
            except Exception as e:
                logger.error("Camera snapshot error: %s", e, exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/openai_api_key")
        async def api_set_key(request: Request) -> JSONResponse:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            key = str(raw.get("openai_api_key", "")).strip()
            if not key:
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)
            import os
            os.environ["OPENAI_API_KEY"] = key
            try:
                config.OPENAI_API_KEY = key
            except Exception:
                pass
            return JSONResponse({"ok": True})

    # ------------------------------------------------------------------
    # WebSocket + static files
    # ------------------------------------------------------------------
    def _setup_ws_and_static(self) -> None:
        @self.app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket) -> None:
            await ws.accept()
            logger.info("WebSocket client connected")

            session_handler = self.handler.copy()
            self._active_handler = session_handler
            self._setup_interrupt(session_handler, ws)

            handler_task = asyncio.create_task(session_handler.start_up(), name="handler")
            emit_task = asyncio.create_task(self._emit_loop(ws, session_handler), name="emit")

            try:
                await self._receive_loop(ws, session_handler)
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception:
                logger.exception("WebSocket error")
            finally:
                handler_task.cancel()
                emit_task.cancel()
                await session_handler.shutdown()
                if self._active_handler is session_handler:
                    self._active_handler = None
                logger.info("Session cleaned up")

        if SETTINGS_STATIC_DIR.exists():
            self.app.mount(
                "/settings",
                StaticFiles(directory=str(SETTINGS_STATIC_DIR), html=True),
                name="settings-static",
            )

        if WEB_APP_DIR.exists():
            self.app.mount("/", StaticFiles(directory=str(WEB_APP_DIR), html=True), name="web-app")
        else:
            logger.warning("Web app directory %s not found - frontend will not be served", WEB_APP_DIR)

    @staticmethod
    def _setup_interrupt(handler: OpenaiRealtimeHandler, ws: WebSocket) -> None:
        """Wire the handler's queue-clear callback to send an interrupt signal."""
        def clear_queue() -> None:
            while not handler.output_queue.empty():
                try:
                    handler.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            handler.output_queue.put_nowait(_INTERRUPT_SENTINEL)  # type: ignore[arg-type]

        handler._clear_queue = clear_queue

    @staticmethod
    async def _receive_loop(ws: WebSocket, handler: OpenaiRealtimeHandler) -> None:
        """Forward binary audio frames from the client to the handler."""
        while True:
            message = await ws.receive()
            msg_type = message.get("type", "")

            if msg_type == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                pcm = np.frombuffer(message["bytes"], dtype=np.int16).reshape(1, -1)
                await handler.receive((WEB_INPUT_SAMPLE_RATE, pcm))

    @staticmethod
    async def _emit_loop(ws: WebSocket, handler: OpenaiRealtimeHandler) -> None:
        """Forward handler outputs (audio + chat) to the client."""
        while True:
            try:
                output = await handler.emit()
            except Exception:
                logger.debug("emit() interrupted")
                break

            if output is None:
                continue

            if isinstance(output, str) and output == _INTERRUPT_SENTINEL:
                await ws.send_json({"type": "interrupt"})
                continue

            if isinstance(output, tuple):
                _sr, audio = output
                await ws.send_bytes(audio.squeeze().astype(np.int16).tobytes())
                continue

            if isinstance(output, AdditionalOutputs):
                for msg in output.args:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    metadata = msg.get("metadata")

                    if isinstance(content, str) and content.startswith("data:image"):
                        await ws.send_json({"type": "image", "data": content})
                    elif metadata:
                        await ws.send_json({
                            "type": "tool",
                            "title": metadata.get("title", ""),
                            "content": content if isinstance(content, str) else json.dumps(content),
                        })
                    else:
                        await ws.send_json({
                            "type": "transcript",
                            "role": role,
                            "content": content if isinstance(content, str) else str(content),
                        })

    def launch(self) -> None:
        """Start the web server (blocking)."""
        logger.info("Starting web UI on http://%s:%s", self.host, self.port)
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

    def close(self) -> None:
        """No-op - uvicorn handles its own shutdown."""
