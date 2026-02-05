"""Entry point for cascade mode - keeps cascade logic isolated from main.py."""

from __future__ import annotations
import time
import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from argparse import Namespace

    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def run_cascade_mode(
    deps: ToolDependencies,
    robot: ReachyMini,
    args: Namespace,
    logger: logging.Logger,
) -> None:
    """Run the application in cascade mode (ASR→LLM→TTS pipeline).

    This function handles all cascade-specific initialization, UI creation,
    and lifecycle management, keeping main.py clean.

    Args:
        deps: Tool dependencies for robot control
        robot: ReachyMini instance
        args: Parsed command line arguments
        logger: Logger instance

    """
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler

    logger.info("Using cascade pipeline mode (ASR→LLM→TTS)")

    if args.gradio:
        # Gradio UI mode
        from reachy_mini_conversation_app.cascade.ui import CascadeGradioUI

        logger.info("Using Gradio UI for cascade mode")
        handler = CascadeHandler(deps, skip_audio_playback=True)
        cascade_ui = CascadeGradioUI(handler, robot)
        stream_manager = cascade_ui.create_interface()
    else:
        # Console mode with VAD
        from reachy_mini_conversation_app.cascade.console import CascadeLocalStream

        logger.info("Using console mode for cascade (VAD-based speech detection)")
        handler = CascadeHandler(deps, skip_audio_playback=False)
        stream_manager = CascadeLocalStream(handler, robot)

    # Start services
    deps.movement_manager.start()
    deps.head_wobbler.start()

    # Initialize media for video capture (required for camera.get_frame() to work)
    # For Gradio mode: start media here (UI doesn't handle it)
    # For console mode: CascadeLocalStream.launch() handles media start
    if args.gradio and deps.camera_worker:
        logger.info(f"Media backend: {robot.media.backend}")
        logger.info("Starting media recording for video capture...")
        try:
            robot.media.start_recording()
            time.sleep(0.5)  # Give video pipeline time to start
            logger.info("Media recording started")
        except Exception as e:
            logger.warning(f"Could not start media recording: {e}")

    if deps.camera_worker:
        deps.camera_worker.start()
        logger.info("Camera worker started in cascade mode")
    else:
        logger.warning("No camera worker available (deps.camera_worker is None)")

    if deps.vision_manager:
        deps.vision_manager.start()

    # Start cascade handler (only for Gradio mode - console mode runs synchronously)
    if args.gradio:
        handler.start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        # Stop the stream manager
        stream_manager.close()

        # Stop cascade handler (only for Gradio mode)
        if args.gradio:
            handler.stop()

        # Stop other services
        deps.movement_manager.stop()
        deps.head_wobbler.stop()
        if deps.camera_worker:
            deps.camera_worker.stop()
            # Stop media recording (only for Gradio mode - console mode handles its own)
            if args.gradio:
                try:
                    robot.media.stop_recording()
                except Exception as e:
                    logger.debug(f"Error stopping media recording: {e}")
        if deps.vision_manager:
            deps.vision_manager.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # Prevent connection to keep alive some threads
        robot.client.disconnect()

        time.sleep(1)
        logger.info("Cascade mode shutdown complete.")
