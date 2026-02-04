"""Pre-warmed audio playback system for low-latency TTS output."""

from __future__ import annotations
import time
import base64
import logging
import threading
from queue import Empty, Queue
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import numpy.typing as npt


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.head_wobble import HeadWobbler


logger = logging.getLogger(__name__)


class AudioPlaybackSystem:
    """Pre-warmed audio playback system with persistent threads.

    Manages audio output through either sounddevice (laptop speakers) or
    robot.media (robot speakers), with synchronized head wobbler animation.

    The system pre-initializes audio streams at construction time to eliminate
    startup latency when playback begins.
    """

    def __init__(
        self,
        robot: ReachyMini | None,
        head_wobbler: HeadWobbler | None,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        """Initialize playback system.

        Args:
            robot: Robot instance (if available, enables robot speaker detection)
            head_wobbler: Head wobbler for animation during speech
            shutdown_event: External shutdown event for coordinated shutdown.
                           If None, creates an internal event.

        """
        self.robot = robot
        self.head_wobbler = head_wobbler
        self.shutdown_event = shutdown_event or threading.Event()

        self._audio_queue: Queue[npt.NDArray[np.int16] | None] = Queue(maxsize=100)
        self._wobbler_queue: Queue[bytes | None] = Queue(maxsize=100)

        self._playback_thread: threading.Thread | None = None
        self._wobbler_thread: threading.Thread | None = None
        self._use_robot_media = False

        # Detect playback mode and start threads
        self._init_playback_threads()

    @property
    def audio_queue(self) -> Queue[npt.NDArray[np.int16] | None]:
        """Audio chunk queue (for direct access if needed)."""
        return self._audio_queue

    @property
    def wobbler_queue(self) -> Queue[bytes | None]:
        """Wobbler chunk queue (for direct access if needed)."""
        return self._wobbler_queue

    @property
    def use_robot_media(self) -> bool:
        """Whether using robot.media for playback."""
        return self._use_robot_media

    def put_audio(self, chunk: npt.NDArray[np.int16]) -> None:
        """Queue an audio chunk for playback."""
        self._audio_queue.put(chunk)

    def put_wobbler(self, chunk: bytes) -> None:
        """Queue raw audio bytes for wobbler animation."""
        self._wobbler_queue.put(chunk)

    def signal_end_of_turn(self) -> None:
        """Signal end of current playback session."""
        self._audio_queue.put(None)
        self._wobbler_queue.put(None)

    def close(self) -> None:
        """Shutdown playback threads."""
        logger.info("Shutting down pre-warmed audio system...")
        self.shutdown_event.set()

        if self._playback_thread:
            self._playback_thread.join(timeout=2)
        if self._wobbler_thread:
            self._wobbler_thread.join(timeout=2)

        # Stop robot media playback if using robot.media
        if self._use_robot_media and self.robot is not None and hasattr(self.robot, "media"):
            logger.info("Stopping robot.media playback system...")
            self.robot.media.stop_playing()

        logger.info("Audio system shutdown complete")

    def _init_playback_threads(self) -> None:
        """Initialize persistent audio playback and wobbler threads (pre-warmed)."""
        # Determine playback mode based on system's default audio output device
        # Use robot.media ONLY if:
        # 1. Robot hardware is available (not simulation)
        # 2. Default output device is a robot speaker (reSpeaker, etc.)

        robot_available = (
            self.robot is not None
            and hasattr(self.robot, "media")
            and not self.robot.client.get_status().get("simulation_enabled", False)
        )

        # Check if default output is a robot speaker
        default_is_robot_speaker = False
        if robot_available:
            try:
                default_device = sd.query_devices(kind="output")
                device_name = default_device["name"].lower()
                # Common robot speaker names
                robot_speaker_keywords = ["respeaker", "xvf3800", "reachy"]
                default_is_robot_speaker = any(keyword in device_name for keyword in robot_speaker_keywords)
                logger.info(f"AUDIO PREWARM: Default output device: {default_device['name']}")
                logger.debug(f"AUDIO PREWARM: Is robot speaker? {default_is_robot_speaker}")
            except Exception as e:
                logger.warning(f"Failed to detect default audio device: {e}")

        # Use robot.media only if both robot is available AND default output is robot speaker
        self._use_robot_media = robot_available and default_is_robot_speaker

        if self._use_robot_media:
            logger.info("AUDIO PREWARM: Using robot.media for playback (robot speaker is default)")
            self._init_robot_playback_threads()
        else:
            reason = "laptop/other speaker" if robot_available else "simulation/no robot"
            logger.info(f"AUDIO PREWARM: Using sounddevice for playback ({reason})")
            self._init_sounddevice_playback_threads()

    def _init_sounddevice_playback_threads(self) -> None:
        """Initialize sounddevice playback threads (laptop speakers)."""

        def persistent_playback_thread() -> None:
            """Run persistent audio playback thread (pre-warmed and ready)."""
            from reachy_mini_conversation_app.cascade.timing import tracker

            stream: sd.OutputStream | None = None
            try:
                # Pre-initialize sounddevice stream (happens once at startup)
                tracker.mark("audio_stream_prewarm_start")

                all_devices = sd.query_devices()
                default_output_idx = sd.default.device[1]
                logger.info(f"AUDIO PREWARM: Total {len(all_devices)} devices available")
                for i, dev in enumerate(all_devices):
                    if dev["max_output_channels"] > 0:
                        default_marker = " (DEFAULT OUTPUT)" if i == default_output_idx else ""
                        logger.info(f"  [{i}] {dev['name']} - {dev['max_output_channels']} channels{default_marker}")

                default_device = sd.query_devices(kind="output")
                logger.info(f"AUDIO PREWARM: Using default device: {default_device['name']}")

                # Create stream once (pre-warmed)
                stream = sd.OutputStream(
                    samplerate=24000,
                    channels=1,
                    dtype=np.int16,
                    blocksize=4096,
                    latency="low",
                )
                stream.start()
                actual_latency_ms = stream.latency * 1000
                logger.info(f"AUDIO PREWARM: Stream ready. Latency: {actual_latency_ms:.0f}ms")
                tracker.mark("audio_stream_prewarm_complete", {"stream_latency_ms": round(actual_latency_ms, 1)})

                # Main playback loop - runs forever
                while not self.shutdown_event.is_set():
                    try:
                        # Wait for chunks with timeout to allow shutdown
                        chunk = self._audio_queue.get(timeout=0.1)

                        if chunk is None:  # Sentinel - end of current playback session
                            continue

                        # Write chunk to stream (immediate playback)
                        stream.write(chunk)

                    except Empty:
                        continue

            except Exception as e:
                logger.exception(f"Error in persistent playback thread: {e}")
            finally:
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass
                logger.info("Playback thread shutdown")

        # Start persistent threads
        self._playback_thread = threading.Thread(target=persistent_playback_thread, daemon=True, name="AudioPlayback")
        self._wobbler_thread = threading.Thread(target=self._persistent_wobbler_thread, daemon=True, name="Wobbler")

        self._playback_thread.start()
        self._wobbler_thread.start()

        # Give threads time to initialize
        time.sleep(0.1)

        logger.info("Pre-warmed audio playback system initialized (sounddevice)")

    def _init_robot_playback_threads(self) -> None:
        """Initialize robot.media playback threads (robot speakers)."""
        import librosa

        # Start robot media playback (must be called before pushing audio)
        if self.robot is not None and hasattr(self.robot, "media"):
            logger.info("Starting robot.media playback system...")
            self.robot.media.start_playing()
            time.sleep(0.5)  # Give pipeline time to initialize
            logger.info("Robot.media playback system started")

        def persistent_playback_thread() -> None:
            """Run persistent audio playback thread using robot.media."""
            from reachy_mini_conversation_app.cascade.timing import tracker

            # Type guard: ensure robot and media are available
            if self.robot is None or not hasattr(self.robot, "media"):
                logger.error("Robot media not available for playback")
                return

            try:
                # Pre-initialize robot media
                tracker.mark("audio_stream_prewarm_start")

                # Get robot audio sample rate
                device_sample_rate = self.robot.media.get_audio_samplerate()
                logger.info(f"AUDIO PREWARM: Robot speaker sample rate: {device_sample_rate}Hz")
                tracker.mark(
                    "audio_stream_prewarm_complete", {"device": "robot.media", "sample_rate": device_sample_rate}
                )

                # Main playback loop - runs forever
                while not self.shutdown_event.is_set():
                    try:
                        # Wait for chunks with timeout to allow shutdown
                        chunk = self._audio_queue.get(timeout=0.1)

                        if chunk is None:  # Sentinel - end of current playback session
                            continue

                        # Convert int16 to float32 for robot.media
                        audio_float = chunk.astype(np.float32) / 32768.0

                        # Resample if needed (TTS outputs 24kHz)
                        if device_sample_rate != 24000:
                            audio_float = librosa.resample(
                                audio_float,
                                orig_sr=24000,
                                target_sr=device_sample_rate,
                            )

                        # Push to robot speaker
                        self.robot.media.push_audio_sample(audio_float)

                    except Empty:
                        continue

            except Exception as e:
                logger.exception(f"Error in robot playback thread: {e}")
            finally:
                logger.info("Robot playback thread shutdown")

        # Start persistent threads
        self._playback_thread = threading.Thread(
            target=persistent_playback_thread, daemon=True, name="RobotAudioPlayback"
        )
        self._wobbler_thread = threading.Thread(target=self._persistent_wobbler_thread, daemon=True, name="Wobbler")

        self._playback_thread.start()
        self._wobbler_thread.start()

        # Give threads time to initialize
        time.sleep(0.1)

        logger.info("Pre-warmed audio playback system initialized (robot.media)")

    def _persistent_wobbler_thread(self) -> None:
        """Run persistent wobbler thread (pre-warmed and ready).

        Shared by both sounddevice and robot.media playback modes.
        """
        try:
            logger.info("WOBBLER PREWARM: Thread ready")

            # Main wobbler loop - runs forever
            while not self.shutdown_event.is_set():
                try:
                    # Wait for chunks with timeout to allow shutdown
                    chunk = self._wobbler_queue.get(timeout=0.1)

                    if chunk is None:  # Sentinel - end of current playback session
                        # Reset wobbler between turns
                        if self.head_wobbler:
                            self.head_wobbler.reset()
                        continue

                    # Feed to wobbler
                    if self.head_wobbler:
                        self.head_wobbler.feed(base64.b64encode(chunk).decode("utf-8"))

                    # Rate limit to match playback
                    chunk_duration = len(chunk) / (2 * 24000)
                    time.sleep(chunk_duration)

                except Empty:
                    continue

        except Exception as e:
            logger.exception(f"Error in persistent wobbler thread: {e}")
        finally:
            logger.info("Wobbler thread shutdown")
