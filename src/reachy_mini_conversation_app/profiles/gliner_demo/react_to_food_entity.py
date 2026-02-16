import logging

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)


# Reaction callbacks for entities
async def react_to_food_entity(
    deps: ToolDependencies,
    entity_text: str,
    entity_label: str,
    confidence: float,
) -> None:
    """React when food entity is detected via NER."""
    import os
    import wave
    import asyncio

    import numpy as np

    logger.info(f"🧀 FOOD detected {entity_label}: '{entity_text}' (confidence: {confidence:.2f})")

    # Only react to high-confidence detections
    if confidence > 0.7:
        logger.info(f"  → High confidence, playing reaction for '{entity_text}'")

        try:
            # Get path to audio file (in same directory as this demo)
            demo_dir = os.path.dirname(os.path.abspath(__file__))
            audio_file = os.path.join(demo_dir, "yummy.wav")

            # Read WAV file
            with wave.open(audio_file, "rb") as wf:
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)

            # Determine playback mode based on system's default audio output device
            # Use robot.media ONLY if:
            # 1. Robot hardware is available (not simulation)
            # 2. Default output device is a robot speaker (reSpeaker, etc.)
            import sounddevice as sd

            robot_available = hasattr(deps.reachy_mini, "media") and not deps.reachy_mini.client.get_status().get(
                "simulation_enabled", False
            )

            # Check if default output is a robot speaker
            use_robot_media = False
            if robot_available:
                try:
                    default_device = sd.query_devices(kind="output")
                    device_name = default_device["name"].lower()
                    # Common robot speaker names
                    robot_speaker_keywords = ["respeaker", "xvf3800", "reachy"]
                    use_robot_media = any(keyword in device_name for keyword in robot_speaker_keywords)
                    logger.debug(f"Default output device: {default_device['name']}")
                    logger.debug(f"Is robot speaker? {use_robot_media}")
                except Exception as e:
                    logger.warning(f"Failed to detect default audio device: {e}")

            if use_robot_media:
                logger.info("🔊 Playing through robot.media")

                # Convert int16 to float32 for robot.media
                audio_float = audio_data.astype(np.float32) / 32768.0

                # Check if we need to resample (robot may have different sample rate)
                device_sample_rate = deps.reachy_mini.media.get_audio_samplerate()
                if device_sample_rate != sample_rate:
                    import librosa

                    audio_float = librosa.resample(
                        audio_float,
                        orig_sr=sample_rate,
                        target_sr=device_sample_rate,
                    )

                # Push audio sample to robot speaker
                deps.reachy_mini.media.push_audio_sample(audio_float)

                # Wait for audio to finish (approximate duration)
                duration = len(audio_data) / sample_rate
                await asyncio.sleep(duration)

            else:
                reason = "laptop/other speaker" if robot_available else "simulation/no robot"
                logger.info(f"🔊 Playing through sounddevice ({reason})")

                # Fallback to sounddevice for simulation/laptop
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()

            logger.info("✅ Audio playback complete")

        except FileNotFoundError:
            logger.error(f"❌ Audio file not found: {audio_file}")
        except Exception as e:
            logger.error(f"❌ Error playing audio: {e}")

        return
