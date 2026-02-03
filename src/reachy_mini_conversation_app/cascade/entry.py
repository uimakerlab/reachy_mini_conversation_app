"""Entry point for cascade mode - keeps cascade logic isolated from main.py."""

from __future__ import annotations
import sys
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
    from reachy_mini_conversation_app.cascade.gradio_ui import CascadeGradioUI

    # Cascade mode requires Gradio (console mode with VAD not yet implemented)
    if not args.gradio:
        logger.error("Cascade mode requires --gradio flag. Console mode with VAD is not yet implemented.")
        logger.info("Please run with: --cascade --gradio")
        robot.client.disconnect()
        sys.exit(1)

    logger.info("Using cascade pipeline mode (ASR→LLM→TTS)")
    handler = CascadeHandler(deps)

    logger.info("Using Gradio UI for cascade mode")
    cascade_ui = CascadeGradioUI(handler, robot)
    stream_manager = cascade_ui.create_interface()

    # Start services
    deps.movement_manager.start()
    deps.head_wobbler.start()
    if deps.camera_worker:
        deps.camera_worker.start()
    if deps.vision_manager:
        deps.vision_manager.start()

    # Start cascade handler
    handler.start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        # Stop the stream manager
        stream_manager.close()

        # Stop cascade handler
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

        import time

        time.sleep(1)
        logger.info("Cascade mode shutdown complete.")
