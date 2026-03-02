"""Entry point for cascade mode - keeps cascade logic isolated from main.py."""

from __future__ import annotations
import os
import time
import signal
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

    if getattr(args, "test_file", None):
        # Test file mode — feed text utterances through the pipeline
        from reachy_mini_conversation_app.cascade.test_stream import CascadeTestStream

        logger.info(f"Using test file mode: {args.test_file}")
        handler = CascadeHandler(deps)
        stream_manager = CascadeTestStream(handler, robot, args.test_file)
    elif args.gradio:
        # Gradio UI mode
        from reachy_mini_conversation_app.cascade.ui import CascadeGradioUI

        logger.info("Using Gradio UI for cascade mode")
        handler = CascadeHandler(deps)
        cascade_ui = CascadeGradioUI(handler, robot)
        stream_manager = cascade_ui.create_interface()
    else:
        # Console mode with VAD
        from reachy_mini_conversation_app.cascade.console import CascadeLocalStream

        logger.info("Using console mode for cascade (VAD-based speech detection)")
        handler = CascadeHandler(deps)
        stream_manager = CascadeLocalStream(handler, robot)

    # Start services
    deps.movement_manager.start()
    deps.head_wobbler.start()

    # For console mode: CascadeLocalStream.launch() handles media start
    # For Gradio mode: audio recording is handled by ContinuousVADRecorder,
    # and camera frames come directly from CameraWorker (no start_recording needed).
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
        # Ignore further Ctrl+C during cleanup so we always reach os._exit.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

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

        # Hard exit to avoid segfault from PortAudio/sounddevice native
        # cleanup racing with daemon thread teardown during interpreter shutdown.
        os._exit(0)
