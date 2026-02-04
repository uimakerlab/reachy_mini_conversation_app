# Cascade Module - Code Improvement Assessment

This document provides a detailed technical assessment of the cascade module, identifying opportunities for simplification, deduplication, and cleanup.

**Audit Date:** February 2025
**Scope:** All files in `cascade/` directory

---

## Executive Summary

The cascade module implements a functional ASR → LLM → TTS pipeline with good provider abstractions. However, several areas have accumulated complexity that could be simplified:

- **gradio_ui.py** (1235 lines) handles too many concerns
- **Code duplication** across ASR and TTS providers
- **Over-engineered abstractions** in config and timing
- **Inconsistent patterns** across similar providers

The recommendations below are ordered by impact and effort.

---

## High Priority Improvements

### 1. Extract Shared Audio Utilities for ASR

**Problem:** WAV-to-PCM conversion logic is duplicated across multiple ASR providers.

**Affected Files:**
| File | Function | Lines |
|------|----------|-------|
| `asr/deepgram.py` | `_wav_to_pcm()` | 238-256 |
| `asr/parakeet_mlx_streaming.py` | `_wav_bytes_to_numpy()` | 446-506 |
| `asr/openai_realtime_asr.py` | inline WAV parsing | scattered |

**Current State:**
```python
# deepgram.py
def _wav_to_pcm(self, wav_bytes: bytes) -> bytes:
    with io.BytesIO(wav_bytes) as wav_buffer:
        with wave.open(wav_buffer, "rb") as wav_file:
            return wav_file.readframes(wav_file.getnframes())

# parakeet_mlx_streaming.py (60+ lines of similar logic)
def _wav_bytes_to_numpy(wav_bytes: bytes) -> npt.NDArray[np.float32]:
    # ... similar parsing with additional numpy conversion
```

**Recommendation:** Create `asr/utils.py`:
```python
# asr/utils.py
def wav_to_pcm(wav_bytes: bytes) -> bytes:
    """Extract raw PCM data from WAV bytes."""
    ...

def wav_to_numpy(wav_bytes: bytes, dtype: str = "float32") -> np.ndarray:
    """Convert WAV bytes to numpy array."""
    ...

def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    ...
```

**Impact:** Eliminates ~100 lines of duplicated code, single point of maintenance for audio format handling.

---

### 2. Break Up CascadeGradioUI

**Problem:** `gradio_ui.py` (1235 lines) handles 6+ distinct responsibilities, making it difficult to understand, test, and modify.

**Current Responsibilities:**
1. Gradio interface construction (`create_interface`)
2. Audio recording management (button toggling, frame accumulation)
3. Continuous VAD mode (IDLE/LISTENING/RECORDING/PROCESSING state machine)
4. Audio playback system (pre-warmed threads, queue management)
5. TTS synthesis with parallel sentence generation
6. Chat history formatting and tool result display

**Recommendation:** Extract into focused modules:

```
cascade/
├── ui/
│   ├── __init__.py
│   ├── gradio_interface.py    # Thin UI layer, component wiring
│   ├── audio_recorder.py      # Recording logic, frame accumulation
│   ├── audio_playback.py      # Pre-warmed threads, queue management
│   ├── continuous_mode.py     # VAD state machine
│   └── chat_display.py        # History formatting, tool results
```

**Example extraction - AudioPlaybackSystem:**
```python
# ui/audio_playback.py
class AudioPlaybackSystem:
    """Manages pre-warmed audio playback threads."""

    def __init__(self, use_robot_media: bool, robot: Optional[ReachyMini]):
        self.audio_queue: Queue = Queue(maxsize=100)
        self.wobbler_queue: Queue = Queue(maxsize=100)
        self._init_threads()

    def enqueue_audio(self, chunk: bytes) -> None: ...
    def drain_and_stop(self) -> None: ...
```

**Impact:** Each file becomes ~200-300 lines, single responsibility, easier to test independently.

---

### 3. Simplify TTS Synthesis Orchestration

**Problem:** `_synthesize_for_gradio()` (lines 1009-1135) has complex async task scheduling that's hard to follow.

**Current Complexity:**
```python
async def _synthesize_for_gradio(self, text: str) -> None:
    sentences = self._split_into_sentences(text)
    tasks = []

    async def generate_and_queue_sentence(sentence, index, delay):
        # Nested async function with complex logic
        # Different delays for sentence 1, 2, 3+
        # Returns List[npt.NDArray] but return value unused
        ...

    for i, sentence in enumerate(sentences):
        if i == 0:
            delay = 0
        elif i == 1:
            delay = estimated_duration * 0.5
        else:
            delay = estimated_duration * 0.95

        task = asyncio.create_task(generate_and_queue_sentence(...))
        tasks.append(task)

    await asyncio.gather(*tasks)  # Results discarded
```

**Issues:**
- Nested async function with unused return value
- Complex delay calculation for parallel generation
- `generate_and_queue_sentence` returns `List[npt.NDArray]` but `gather()` result is ignored
- Comments suggest uncertainty ("TODO: verify sentence ordering")

**Recommendation:** Simplify to straightforward sequential generation with pipelining:
```python
async def _synthesize_for_gradio(self, text: str) -> None:
    """Synthesize and play TTS audio for text."""
    sentences = self._split_into_sentences(text)

    for sentence in sentences:
        async for chunk in self.handler.tts.synthesize(sentence):
            self._enqueue_audio_chunk(chunk)
```

Or if parallelism is truly needed, make it explicit:
```python
async def _synthesize_for_gradio(self, text: str) -> None:
    sentences = self._split_into_sentences(text)

    # Generate first sentence immediately
    first_chunks = await self._generate_sentence(sentences[0])
    self._enqueue_chunks(first_chunks)

    # Pipeline: generate next while playing current
    for sentence in sentences[1:]:
        chunks = await self._generate_sentence(sentence)
        self._enqueue_chunks(chunks)
```

**Impact:** Removes ~50 lines of complex scheduling code, makes control flow obvious.

---

### 4. Consolidate Parakeet Warmup Logic

**Problem:** Model warmup code is duplicated between batch and streaming Parakeet providers.

**Affected Files:**
| File | Function | Lines |
|------|----------|-------|
| `asr/parakeet_mlx.py` | `_warmup_model()` | 67-104 |
| `asr/parakeet_mlx_streaming.py` | `_warmup_model()` | 108-143 |

**Current State (both files have nearly identical code):**
```python
def _warmup_model(self) -> None:
    """Pre-load model and run inference on silence."""
    logger.info("Warming up Parakeet MLX model...")

    # Generate 1 second of silence
    silence_duration = 1.0
    silence = np.zeros(int(16000 * silence_duration), dtype=np.float32)

    # Convert to MLX array
    audio_mlx = mx.array(silence)

    # Run inference to warm up
    _ = self.pipeline.transcribe(audio_mlx)

    logger.info("Parakeet MLX model warmed up")
```

**Recommendation:** Create `asr/parakeet_utils.py`:
```python
# asr/parakeet_utils.py
def warmup_parakeet_model(pipeline, logger) -> None:
    """Pre-load Parakeet model and run inference on silence."""
    logger.info("Warming up Parakeet MLX model...")
    silence = np.zeros(16000, dtype=np.float32)  # 1 second
    _ = pipeline.transcribe(mx.array(silence))
    logger.info("Parakeet MLX model warmed up")

def create_parakeet_pipeline(model: str, precision: str):
    """Create configured Parakeet pipeline."""
    ...
```

**Impact:** Eliminates ~40 lines of duplication, ensures consistent warmup behavior.

---

## Medium Priority Improvements

### 5. Consolidate TTS Silence Trimming and Re-chunking

**Problem:** All TTS providers implement identical post-processing logic.

**Affected Files:**
| File | Pattern | Lines |
|------|---------|-------|
| `tts/openai.py` | collect → trim → re-chunk | 76-87 |
| `tts/elevenlabs.py` | collect → trim → re-chunk | 115-131 |
| `tts/kokoro.py` | collect → trim → re-chunk | 119-134 |

**Current State:**
```python
# Pattern repeated in all three providers
all_audio = b"".join(chunks)

if self.trim_silence:
    all_audio = trim_leading_silence(all_audio, sample_rate=24000)

# Re-chunk for streaming (chunk sizes vary: 1024, 4096*2, 4096)
chunk_size = 4096 * 2
for i in range(0, len(all_audio), chunk_size):
    yield all_audio[i : i + chunk_size]
```

**Recommendation:** Extend `tts/utils.py`:
```python
# tts/utils.py
def process_tts_output(
    audio_bytes: bytes,
    trim_silence: bool = True,
    sample_rate: int = 24000,
    chunk_size: int = 4096
) -> Iterator[bytes]:
    """Process TTS output: trim silence and yield chunks."""
    if trim_silence:
        audio_bytes = trim_leading_silence(audio_bytes, sample_rate)

    for i in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[i : i + chunk_size]
```

Then in providers:
```python
# tts/openai.py
all_audio = b"".join(chunks)
for chunk in process_tts_output(all_audio, self.trim_silence):
    yield chunk
```

**Impact:** Eliminates ~30 lines of duplication, standardizes chunk sizes.

---

### 6. Define Constants for Magic Numbers

**Problem:** Magic numbers scattered throughout make code harder to understand and modify.

**Examples Found:**
| File | Value | Usage |
|------|-------|-------|
| `gradio_ui.py:335` | `1024` | Recording blocksize |
| `gradio_ui.py:388` | `512` | VAD chunk samples |
| `gradio_ui.py:114` | `24000` | TTS sample rate |
| `gradio_ui.py:1095` | `0.95` | Rate limiting factor |
| `kokoro.py:128` | `4096` | TTS chunk size |
| `elevenlabs.py:135` | `4096 * 2` | TTS chunk size |
| `parakeet_mlx.py:69` | `16000` | ASR sample rate |

**Recommendation:** Define at module level:
```python
# constants.py or at top of relevant modules
RECORDING_SAMPLE_RATE = 16000
RECORDING_BLOCKSIZE = 1024
TTS_SAMPLE_RATE = 24000
TTS_CHUNK_SIZE = 4096
VAD_CHUNK_SAMPLES = 512
RATE_LIMIT_FACTOR = 0.95
```

**Impact:** Self-documenting code, easier to tune parameters.

---

### 7. Wire Temperature Configuration

**Problem:** LLM temperature is hard-coded despite config support.

**Current State:**
```python
# handler.py:326
async for chunk in self.llm.generate(
    messages=self.conversation_history,
    tools=self.tool_specs,
    temperature=1.0,  # TODO: Move to config
):
```

**cascade.yaml already supports it:**
```yaml
llm:
  openai_gpt:
    temperature: 0.7  # Defined but never used
```

**Recommendation:**
```python
# handler.py
temperature = self.config.get_llm_settings().get("temperature", 1.0)
async for chunk in self.llm.generate(
    messages=self.conversation_history,
    tools=self.tool_specs,
    temperature=temperature,
):
```

**Impact:** Configuration works as expected, removes TODO.

---

### 8. Remove or Use cascade.yaml Metadata

**Problem:** Provider metadata fields are defined but never used at runtime.

**Unused Fields:**
```yaml
asr:
  providers:
    parakeet:
      hardware: "Apple Silicon"  # Never checked
      status: "stable"           # Never displayed
      description: "..."         # Never used
      location: "local"          # Never used
```

**Options:**

**A. Remove from cascade.yaml** (simplest):
```yaml
asr:
  provider: "parakeet"
  parakeet:
    model: "parakeet-large"
    precision: "float16"
```

**B. Use for validation** (more robust):
```python
def _init_provider(self, provider_type: str, provider_name: str):
    info = self.config.get_provider_info(provider_type, provider_name)

    # Validate hardware requirements
    if info.get("hardware") == "Apple Silicon":
        if platform.processor() != "arm":
            raise RuntimeError(f"{provider_name} requires Apple Silicon")

    # Check API key requirements
    for key in info.get("requires", []):
        if not os.environ.get(key):
            raise RuntimeError(f"{provider_name} requires {key}")
```

**Recommendation:** Option A unless hardware validation is desired.

**Impact:** Cleaner config file, or more robust initialization.

---

### 9. Standardize Async Patterns

**Problem:** Inconsistent async/sync wrapping across providers.

**Current Patterns:**
| Provider | Pattern |
|----------|---------|
| `whisper_openai.py` | `asyncio.to_thread()` for sync client |
| `parakeet_mlx.py` | `asyncio.to_thread()` for MLX ops |
| `parakeet_mlx_streaming.py` | Sync MLX ops (thread affinity) |
| `deepgram.py` | Pure async with callbacks |
| `openai_realtime_asr.py` | Pure async with callbacks |
| `elevenlabs.py` | `asyncio.to_thread()` for sync client |
| `kokoro.py` | `asyncio.to_thread()` for local inference |

**Recommendation:** Document the pattern choice in each provider:
```python
class ParakeetMLXASR(ASRProvider):
    """Local ASR using Parakeet MLX.

    Threading: Synchronous MLX operations (no asyncio.to_thread).
    MLX has thread affinity requirements on Apple Silicon.
    """
```

And standardize where possible:
- **Local inference (MLX, Kokoro):** Sync operations, document thread affinity
- **Cloud APIs with sync SDKs:** `asyncio.to_thread()`
- **Cloud APIs with async SDKs:** Native async

**Impact:** Easier to understand and debug threading issues.

---

## Lower Priority Improvements

### 10. Remove Unused Return Value

**File:** `gradio_ui.py:1043-1083`

**Problem:**
```python
async def generate_and_queue_sentence(...) -> List[npt.NDArray[np.int16]]:
    chunks = []
    async for chunk in self.handler.tts.synthesize(sentence):
        chunks.append(chunk)
        # ... queue chunk
    return chunks  # Never used

# Later:
await asyncio.gather(*tasks)  # Results discarded
```

**Recommendation:** Remove return type and return statement:
```python
async def generate_and_queue_sentence(...) -> None:
    async for chunk in self.handler.tts.synthesize(sentence):
        # ... queue chunk
```

---

### 11. Simplify LatencyTracker

**File:** `timing.py`

**Problem:** `get_duration()` (lines 50-80) has complex `use_first` parameter logic and hard-coded VAD flow detection.

**Current:**
```python
def get_duration(self, start_event: str, end_event: str, use_first: bool = False):
    # Complex logic for handling multiple events with same name
    # Special case for use_first parameter
    ...

def print_summary(self):
    # Hard-coded VAD vs button flow detection
    is_vad_flow = "vad_speech_start" in self._events
    if is_vad_flow:
        # ... different output format
```

**Recommendation:** Simplify to generic event tracking:
```python
class LatencyTracker:
    def mark(self, event: str, metadata: dict = None) -> None: ...
    def duration_between(self, start: str, end: str) -> float: ...
    def print_summary(self) -> None:
        # Generic output, no flow-specific logic
```

---

### 12. Consolidate Config Metadata Keys

**File:** `config.py:16-19`

**Problem:**
```python
ASR_METADATA_KEYS = {"hardware", "status", "description", "requires", "location"}
LLM_METADATA_KEYS = {"hardware", "status", "description", "requires", "location"}
TTS_METADATA_KEYS = {"hardware", "status", "description", "requires", "location"}
```

**Recommendation:** Single set:
```python
PROVIDER_METADATA_KEYS = {"hardware", "status", "description", "requires", "location"}
```

---

### 13. Remove Commented Debug Code

**File:** `asr/parakeet_mlx_streaming.py`

**Lines:** 200-201, 288-291

```python
# logger.debug(f"Finalized tokens: {self._num_finalized_tokens}")
# logger.debug(f"Draft tokens: {len(result.draft_tokens)}")
```

**Recommendation:** Either enable with proper debug flag or remove entirely.

---

### 14. Standardize Error Handling

**Problem:** Inconsistent behavior on errors.

| File | Behavior |
|------|----------|
| `whisper_openai.py` | Logs error, returns empty string |
| `handler.py` | Logs error, re-raises |
| `deepgram.py` | Raises immediately |

**Recommendation:** Establish convention:
- **ASR:** Return empty string on transient errors, raise on configuration errors
- **LLM:** Always raise (critical path)
- **TTS:** Return empty audio on transient errors, raise on configuration errors

Document in `base.py` classes.

---

### 15. Standardize Type Hints

**Problem:** Mix of `Optional[str]` and `str | None` syntax.

**Recommendation:** Use `str | None` consistently (Python 3.10+ syntax) throughout, since the project already requires Python 3.10+.

---

## Summary Table

| Priority | Issue | Files | Effort | Impact |
|----------|-------|-------|--------|--------|
| 🔴 High | Extract ASR audio utils | asr/*.py | Medium | High |
| 🔴 High | Break up GradioUI | gradio_ui.py | High | High |
| 🔴 High | Simplify TTS synthesis | gradio_ui.py | Medium | Medium |
| 🔴 High | Consolidate Parakeet warmup | asr/parakeet*.py | Low | Medium |
| 🟡 Medium | TTS post-processing | tts/*.py | Low | Medium |
| 🟡 Medium | Magic number constants | multiple | Low | Low |
| 🟡 Medium | Wire temperature config | handler.py | Low | Low |
| 🟡 Medium | Remove unused metadata | cascade.yaml | Low | Low |
| 🟡 Medium | Document async patterns | multiple | Low | Medium |
| 🟢 Low | Remove unused return | gradio_ui.py | Low | Low |
| 🟢 Low | Simplify LatencyTracker | timing.py | Medium | Low |
| 🟢 Low | Consolidate config keys | config.py | Low | Low |
| 🟢 Low | Remove commented code | parakeet_mlx_streaming.py | Low | Low |
| 🟢 Low | Standardize error handling | multiple | Medium | Medium |
| 🟢 Low | Standardize type hints | multiple | Low | Low |

---

## Recommended Implementation Order

1. **Quick wins first:** Items 6, 7, 10, 12, 13 (low effort, immediate cleanup)
2. **Shared utilities:** Items 1, 4, 5 (reduces duplication before larger refactors)
3. **Major refactor:** Item 2 (break up GradioUI - depends on having clean utilities)
4. **Polish:** Remaining items as time permits

---

## Files by Complexity

| File | Lines | Concerns | Recommendation |
|------|-------|----------|----------------|
| `gradio_ui.py` | 1235 | 6+ | Split into 4-5 focused modules |
| `handler.py` | 558 | 3-4 | Consider splitting tool execution |
| `parakeet_mlx_streaming.py` | 506 | 2-3 | Extract transcript state management |
| `config.py` | ~200 | 1-2 | Simplify metadata handling |
| `timing.py` | 148 | 1 | Remove hard-coded flow detection |
