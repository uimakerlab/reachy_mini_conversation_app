"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import logging
import asyncio
import argparse
import threading

from typing import Literal, Optional

from fastapi import FastAPI

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
    log_connection_troubleshooting,
)
from reachy_mini_conversation_app.config import LOCKED_PROFILE

logger = logging.getLogger(__name__)

def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    setup_logger(logger, args.debug) 
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler

    
    

    logger.info("Starting Reachy Mini Conversation App")
    if LOCKED_PROFILE is not None:
        logger.info(
            "Profile switching is locked. Using LOCKED_PROFILE='%s'.",
            LOCKED_PROFILE,
        )

    if args.no_camera and args.head_tracker is not None:
        logger.warning(
            "Head tracking disabled: --no-camera flag is set. "
            "Remove --no-camera to enable head tracking."
        )

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(
                "Connection timeout: Failed to connect to Reachy Mini daemon. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(
                "Connection failed: Unable to establish connection to Reachy Mini. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(
                f"Unexpected error during robot initialization: {type(e).__name__}: {e}"
            )
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    # SDK daemon status exposes both MuJoCo (`simulation_enabled`) and mockup (`mockup_sim_enabled`) modes.
    status = robot.client.get_status()
    is_simulation = status.get("simulation_enabled", False) or status.get("mockup_sim_enabled", False)

    if is_simulation and not args.gradio:
        logger.info("Simulation mode detected. Automatically enabling gradio flag.")
        args.gradio = True

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )

    audio_source: Literal["browser", "robot_device"] = "browser" if is_simulation else "robot_device"
    logger.info("Selected audio source: %s", audio_source)

    handler = OpenaiRealtimeHandler(
        deps,
        gradio_mode=(audio_source == "browser"),
        instance_path=instance_path,
    )

    should_launch_ui = bool(args.gradio or settings_app is not None)
    if audio_source == "browser" and not should_launch_ui:
        logger.warning("audio_source=browser requires UI; enabling Gradio launch.")
        should_launch_ui = True

    if should_launch_ui:
        from reachy_mini_conversation_app.gradio_ui import build_gradio_ui

        stream_manager = build_gradio_ui(
            handler=handler,
            robot=robot,
            settings_app=settings_app,
            instance_path=instance_path,
            audio_source=audio_source,
        )
    else:
        logger.info("Using pure console stream")
        from reachy_mini_conversation_app.console import LocalStream

        stream_manager = LocalStream(
            handler,
            robot,
            instance_path=instance_path,
        )

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        # In SDK settings-app mode with browser audio, Gradio is already mounted on the settings FastAPI server.
        if settings_app is not None and audio_source == "browser" and app_stop_event is not None:
            logger.info("Gradio UI mounted on settings app, waiting for stop event.")
            app_stop_event.wait()
        else:
            stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager after KeyboardInterrupt: {e}")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")
        os._exit(0)


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False
    auto_mount_static_ui = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        # is_wireless = reachy_mini.client.get_status()["wireless_version"]
        # args.head_tracker = None if is_wireless else "mediapipe"

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
