"""Shared utilities for TTS providers."""

import logging

import numpy as np
import numpy.typing as npt

from reachy_mini_conversation_app.cascade.config import config


logger = logging.getLogger(__name__)


def trim_leading_silence(
    audio_array: npt.NDArray[np.int16] | npt.NDArray[np.float32],
    sample_rate: int = 24000,
    threshold_int16: int = 327,  # 0.01 * 32767 for int16
    threshold_float: float = 0.01,  # For float32 normalized audio
    min_silence_ms: int = 100,
    provider_name: str = "TTS",
) -> npt.NDArray[np.int16] | npt.NDArray[np.float32]:
    """Trim leading silence from audio if enabled in config.

    Args:
        audio_array: Audio samples (int16 or float32)
        sample_rate: Audio sample rate (default 24kHz)
        threshold_int16: Silence threshold for int16 audio
        threshold_float: Silence threshold for float32 audio
        min_silence_ms: Only trim if silence exceeds this duration
        provider_name: Provider name for logging

    Returns:
        Trimmed audio array (same dtype as input)

    """
    # Select threshold based on dtype
    if audio_array.dtype == np.int16:
        threshold = threshold_int16
    else:
        threshold = threshold_float

    non_silent = np.where(np.abs(audio_array) > threshold)[0]

    if len(non_silent) == 0:
        logger.warning(f"{provider_name}: No non-silent samples found!")
        return audio_array

    first_sound_sample = non_silent[0]
    silence_duration_ms = (first_sound_sample / sample_rate) * 1000

    if silence_duration_ms <= min_silence_ms:
        logger.debug(f"{provider_name}: {silence_duration_ms:.0f}ms leading silence (acceptable)")
        return audio_array

    logger.warning(
        f"{provider_name}: {silence_duration_ms:.0f}ms of leading silence detected (trim_silence={config.tts_trim_silence})"
    )

    if not config.tts_trim_silence:
        logger.info(f"{provider_name}: Keeping silence (tts_trim_silence=false)")
        return audio_array

    logger.info(
        f"{provider_name}: Trimming from {len(audio_array)} to {len(audio_array) - first_sound_sample} samples"
    )
    trimmed = audio_array[first_sound_sample:]
    logger.info(
        f"{provider_name}: After trim - new length: {len(trimmed)} samples ({len(trimmed) / sample_rate * 1000:.0f}ms)"
    )

    return trimmed
