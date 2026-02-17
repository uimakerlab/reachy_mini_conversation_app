# Cascade Mode - Developer Documentation

This document provides detailed technical documentation for the cascade conversation mode in the Reachy Mini Conversation App.

## Overview

Cascade Mode implements a traditional **ASR → LLM → TTS** conversation pipeline for the Reachy Mini robot. It processes user speech through automatic speech recognition, generates AI responses via configurable LLM providers, and synthesizes speech output with synchronized head animations.

### Key Design Principles

- **Provider abstraction** - Swap ASR/LLM/TTS implementations via configuration
- **Latency optimization** - Pre-warmed audio playback threads, parallel TTS generation
- **Clean separation** - Handler (business logic) vs UI (presentation & I/O)

### Current Scope

- Gradio UI only (console mode with VAD not yet implemented)
- Manual push-to-talk recording
- Optional streaming ASR support
- Head wobbler animation synchronized with audio playback

---

## File Structure

```
cascade/
├── __init__.py                        # Package exports
├── entry.py                           # Entry point from main.py
├── handler.py                         # Core pipeline orchestrator
├── config.py                          # Configuration loader (cascade.yaml)
├── timing.py                          # Latency tracking & profiling
├── vad.py                             # Silero VAD for continuous mode
│
├── ui/                                # Gradio interface components
│   ├── __init__.py                    # Exports CascadeGradioUI
│   ├── audio_playback.py              # Pre-warmed audio output system
│   ├── audio_recording.py             # Push-to-talk and VAD recording
│   └── gradio_app.py                  # Main Gradio interface
│
├── asr/                               # Automatic Speech Recognition
│   ├── __init__.py                    # Provider exports
│   ├── base.py                        # ASRProvider abstract base
│   ├── base_streaming.py              # StreamingASRProvider abstract base
│   ├── whisper_openai.py              # OpenAI Whisper implementation
│   ├── parakeet_mlx.py                # Parakeet MLX batch implementation
│   ├── parakeet_mlx_streaming.py      # Parakeet MLX streaming implementation (RNNT, experimental)
│   ├── parakeet_mlx_progressive.py    # Parakeet MLX progressive (sentence-aware sliding window)
│   ├── deepgram.py                    # Deepgram streaming implementation
│   ├── nemotron.py                    # Nemotron ASR implementation
│   └── openai_realtime_asr.py         # OpenAI Realtime streaming implementation
│
├── llm/                               # Large Language Models
│   ├── __init__.py                    # Provider exports
│   ├── base.py                        # LLMProvider + LLMChunk abstractions
│   ├── openai.py                      # OpenAI GPT implementation
│   └── gemini.py                      # Google Gemini implementation
│
├── tts/                               # Text-to-Speech
│   ├── __init__.py                    # Provider exports
│   ├── base.py                        # TTSProvider abstract base
│   ├── utils.py                       # Shared TTS utilities (silence trimming)
│   ├── openai.py                      # OpenAI TTS implementation
│   ├── kokoro.py                      # Kokoro local TTS
│   └── elevenlabs.py                  # ElevenLabs API implementation
│
└── transcript_analysis/               # Real-time reactions to user speech
    ├── __init__.py                    # Package exports (EntityAnalyzer optional)
    ├── base.py                        # Data classes + TranscriptAnalyzer ABC
    ├── loader.py                      # YAML loader, callback importer
    ├── manager.py                     # Orchestrator, deduplication, dispatch
    ├── keyword_analyzer.py            # Keyword/glob matching
    └── entity_analyzer.py             # NER via GLiNER (optional dependency)
```

---

## Core Components

### CascadeHandler (`handler.py`)

The central orchestrator that manages the conversation pipeline.

**Responsibilities:**
- Initialize ASR/LLM/TTS providers from config
- Manage conversation history (OpenAI message format)
- Orchestrate ASR → LLM → Tool execution pipeline
- Execute tool calls (speak, camera, movements)
- Handle multi-modal inputs (text + images)
- Run async event loop in background thread

**Key Attributes:**
```python
asr: ASRProvider              # Current ASR implementation
llm: LLMProvider              # Current LLM implementation
tts: TTSProvider              # Current TTS implementation
conversation_history: List    # OpenAI format: [{"role": "...", "content": ...}]
processing_lock: asyncio.Lock # Prevent concurrent audio processing
tool_specs: List[Dict]        # Tools in Chat Completions format
```

**Key Methods:**
```python
async process_audio_manual(audio_bytes) -> str
    # Entry point for push-to-talk recording

async process_audio_streaming_start() -> None
    # Initialize streaming ASR session

async process_audio_streaming_chunk(chunk) -> Optional[str]
    # Send chunk to streaming ASR, get partial transcript

async process_audio_streaming_end() -> str
    # Finalize streaming ASR, run LLM pipeline

async _process_llm_response() -> None
    # Stream LLM, collect text/tool calls, execute tools

async _execute_tool_calls(tool_calls) -> None
    # Execute individual tools, handle camera/speak specially
```

### UI Components (`ui/`)

The Gradio interface is split into focused modules:

#### AudioPlaybackSystem (`ui/audio_playback.py`)

Pre-warmed audio output system for low-latency TTS playback.

**Responsibilities:**
- Detect playback backend (sounddevice vs robot.media)
- Manage persistent playback and wobbler threads
- Queue-based communication for zero-startup-cost playback

**Key Interface:**
```python
playback = AudioPlaybackSystem(robot, head_wobbler, shutdown_event)
playback.put_audio(chunk)      # Queue audio for playback
playback.put_wobbler(chunk)    # Queue wobbler data
playback.signal_end_of_turn()  # Signal end of speech
playback.close()               # Shutdown threads
```

#### Recording Classes (`ui/audio_recording.py`)

**ContinuousState:** Enum for VAD state machine (IDLE → LISTENING → RECORDING → PROCESSING)

**StreamingASRCallbacks:** Dataclass for injecting ASR callbacks without coupling to handler.

**PushToTalkRecorder:** Manual recording mode.
```python
recorder = PushToTalkRecorder(sample_rate, streaming_callbacks, event_loop)
recorder.start()   # Start recording
recorder.stop()    # Stop and get audio
wav_bytes = recorder.get_wav_bytes()
```

**ContinuousVADRecorder:** VAD-based continuous recording.
```python
recorder = ContinuousVADRecorder(
    sample_rate, streaming_callbacks, on_speech_captured, event_loop
)
recorder.start()   # Start VAD loop
recorder.stop()    # Stop VAD loop
recorder.state     # Current ContinuousState
```

#### CascadeGradioUI (`ui/gradio_app.py`)

Main orchestrator that ties everything together.

**Responsibilities:**
- Build Gradio interface (chatbot, buttons, status)
- Process audio pipeline (ASR → LLM → TTS)
- Coordinate playback and recording subsystems
- Handle Gradio events and lifecycle

**Key Attributes:**
```python
handler: CascadeHandler        # Reference to cascade handler
robot: ReachyMini | None       # Robot instance
playback: AudioPlaybackSystem  # Audio output
shutdown_event: threading.Event # Coordinated shutdown
```

**Key Methods:**
```python
create_interface() -> gr.Blocks     # Build Gradio UI
launch(**kwargs) -> None            # Start server
close() -> None                     # Shutdown all subsystems
```

---

## Provider Abstractions

### ASR Providers (`asr/`)

**Base class** (`base.py`):
```python
class ASRProvider(ABC):
    async transcribe(audio_bytes, language) -> str
```

**Streaming base** (`base_streaming.py`):
```python
class StreamingASRProvider(ASRProvider):
    async start_stream() -> None
    async send_chunk(chunk) -> Optional[str]  # Partial transcript
    async end_stream() -> str                  # Final transcript
```

**Implementations:**
| Provider | Type | Description |
|----------|------|-------------|
| `WhisperOpenAIASR` | Batch | OpenAI Whisper API |
| `ParakeetMLXASR` | Batch | Local Parakeet via MLX (parakeet-mlx) |
| `ParakeetMLXStreamingASR` | Streaming | Local RNNT streaming via MLX (parakeet-mlx, experimental) |
| `ParakeetMLXProgressiveASR` | Streaming | Local progressive with sentence-aware sliding window (mlx-audio) |
| `DeepgramASR` | Streaming | Deepgram Nova via WebSocket |
| `NemotronASR` | Streaming | NVIDIA Nemotron ASR |
| `OpenAIRealtimeASR` | Streaming | OpenAI Realtime API via WebSocket |

### LLM Providers (`llm/`)

**Base class** (`base.py`):
```python
@dataclass
class LLMChunk:
    type: str  # "text_delta" | "tool_call" | "done"
    content: Optional[str]      # For text_delta
    tool_call: Optional[Dict]   # For tool_call

class LLMProvider(ABC):
    async generate(messages, tools, temperature) -> AsyncIterator[LLMChunk]
    def parse_tool_call(tool_call) -> (call_id, tool_name, args_dict)
        # Default implementation handles OpenAI-style tool call format
        # Subclasses can override if needed
```

**Implementations:**
| Provider | Models |
|----------|--------|
| `OpenAILLM` | GPT-4, GPT-4o, GPT-4-turbo |
| `GeminiLLM` | Gemini 2.0 Flash |

### TTS Providers (`tts/`)

**Base class** (`base.py`):
```python
class TTSProvider(ABC):
    async synthesize(text, voice=None) -> AsyncIterator[bytes]
```

**Implementations:**
| Provider | Description |
|----------|-------------|
| `OpenAITTS` | OpenAI TTS API (24kHz PCM int16) |
| `KokoroTTS` | Local TTS via Kokoro model |
| `ElevenLabsTTS` | ElevenLabs API |

---

## Data Flow

### Push-to-Talk Pipeline

```
User clicks START
  │
  ├─ Start audio recording thread
  │   └─ Capture frames via sounddevice → audio_frames list
  │
User clicks STOP
  │
  ├─ Stop recording, concatenate frames to WAV
  │
  └─> handler.process_audio_manual(wav_bytes)
      │
      ├─ PHASE 1: ASR
      │   └─ asr.transcribe(audio_bytes) → transcript
      │       └─ Add to conversation_history as user message
      │
      ├─ PHASE 2: LLM
      │   └─ llm.generate(history, tools) → response + tool_calls
      │       └─ Add assistant message to history
      │
      ├─ PHASE 3: Tool Execution
      │   └─ For each tool_call:
      │       ├─ Execute tool
      │       ├─ Add result to history
      │       └─ Special handling:
      │           • speak: Extract message for TTS
      │           • camera: Add image, re-call LLM
      │
      └─> Return transcript to UI

UI extracts responses from history:
  │
  ├─ Find speak tool messages → combine text
  ├─ Split into sentences
  │
  └─ PHASE 4: TTS
      └─ For each sentence:
          ├─ tts.synthesize(sentence) → audio chunks
          ├─ Queue to audio_queue → playback thread
          └─ Queue to wobbler_queue → head animation
```

### Streaming ASR Flow

```
User clicks START (streaming provider)
  │
  ├─ handler.process_audio_streaming_start()
  │   └─ Open WebSocket/stream connection
  │
During recording:
  │
  ├─ For each audio chunk:
  │   ├─ handler.process_audio_streaming_chunk(chunk)
  │   │   └─ Get partial transcript
  │   └─ Update UI with partial text
  │
User clicks STOP
  │
  └─ handler.process_audio_streaming_end()
      ├─ Close stream
      ├─ Get final transcript
      └─ Run LLM pipeline (same as batch mode)
```

---

## Configuration

### Config File: `cascade.yaml`

```yaml
asr:
  provider: "parakeet" | "openai_whisper" | "parakeet_streaming" | "deepgram_streaming"
  parakeet:
    model: "parakeet-large"
    precision: "float16" | "bfloat16"
  parakeet_streaming:
    context_size: [256, 256]
    depth: 2
  deepgram_streaming:
    model: "nova-2"

llm:
  provider: "openai_gpt" | "gemini"
  openai_gpt:
    model: "gpt-4o" | "gpt-4" | "gpt-4-turbo"
  gemini:
    model: "gemini-2.0-flash"

tts:
  provider: "openai_tts" | "kokoro" | "elevenlabs"
  trim_silence: true | false
  openai_tts:
    voice: "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer"
  kokoro:
    voice: "af" | "am" | "bf" | "bm"
  elevenlabs:
    voice_id: "..."
    model: "eleven_monolingual_v1" | "eleven_turbo_v2"
```

### Environment Variables

```
OPENAI_API_KEY      # Required for OpenAI ASR/LLM/TTS
GEMINI_API_KEY      # Required for Gemini LLM
DEEPGRAM_API_KEY    # Required for Deepgram streaming ASR
ELEVENLABS_API_KEY  # Required for ElevenLabs TTS
```

### Config Loading (`config.py`)

```python
config = CascadeConfig()  # Singleton, loaded on import
# Access: config.CASCADE_ASR_PROVIDER, config.CASCADE_LLM_MODEL, etc.
```

---

## Audio Playback System

### Architecture

Pre-warmed persistent threads created at UI startup:

```
┌─────────────────────────┐
│ Playback Thread         │  ← Runs forever, blocks on queue
│ (sounddevice or robot)  │
└───────────▲─────────────┘
            │
       audio_queue
       (maxsize=100)
            │
┌───────────┴─────────────┐
│ gradio_ui               │  ← Enqueues TTS chunks
│ _synthesize_for_gradio  │
└───────────┬─────────────┘
            │
       wobbler_queue
       (maxsize=100)
            │
            ▼
┌─────────────────────────┐
│ Wobbler Thread          │  ← Runs forever, blocks on queue
│ (head animation)        │
└─────────────────────────┘
```

### Playback Backends

**sounddevice** (laptop speakers):
- `sd.OutputStream(samplerate=24000, channels=1, dtype=int16)`
- Pre-initialized stream at startup
- Direct chunk writes

**robot.media** (robot speakers):
- `robot.media.start_playing()` called once
- Convert int16 → float32, resample if needed
- `robot.media.push_audio_sample(audio_float)`

### Device Detection

The UI queries the default output device and checks for robot speaker keywords:
- "respeaker", "xvf3800", "reachy"
- Only uses robot.media if default matches a robot speaker

---

## Module Dependencies

```
main.py
  │
  └─> entry.py (run_cascade_mode)
      │
      ├─> handler.py (CascadeHandler)
      │   │
      │   ├─> config.py
      │   ├─> asr/ providers
      │   ├─> llm/ providers
      │   ├─> tts/ providers
      │   └─> transcript_analysis/ (TranscriptAnalysisManager)
      │       ├─> loader.py (reads profiles/<name>/reactions.yaml)
      │       ├─> keyword_analyzer.py
      │       └─> entity_analyzer.py (optional, requires gliner)
      │
      └─> ui/gradio_app.py (CascadeGradioUI)
          │
          ├─> ui/audio_playback.py (AudioPlaybackSystem)
          ├─> ui/audio_recording.py (Recorders)
          └─> handler (reference)
```

**Key relationships:**
- `entry.py` creates both handler and UI, wires them together
- Handler owns conversation state, provider instances, and transcript analysis
- UI owns audio I/O and display, reads from handler
- AudioPlaybackSystem handles pre-warmed playback threads
- Recording classes encapsulate push-to-talk and VAD modes
- Transcript analysis runs in parallel with the main ASR → LLM → TTS pipeline

---

## Entry Point & Integration

### Entry Function (`entry.py`)

```python
def run_cascade_mode(deps, robot, args, logger) -> None:
    """Entry called from main.py when --cascade flag is set."""

    # Validate Gradio requirement
    if not args.gradio:
        logger.error("Cascade mode requires --gradio flag")
        sys.exit(1)

    # Initialize components
    handler = CascadeHandler(deps)
    cascade_ui = CascadeGradioUI(handler, robot)
    stream_manager = cascade_ui.create_interface()

    # Start background services
    deps.movement_manager.start()
    deps.head_wobbler.start()
    deps.camera_worker.start()
    deps.vision_manager.start()
    handler.start()

    # Run Gradio (blocks until user closes)
    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Shutdown...")
    finally:
        # Cleanup all services
        stream_manager.close()
        handler.stop()
        # ... stop other services
```

### Command Line Usage

```bash
reachy-mini-conversation-app --cascade --gradio --head-tracker yolo
```

---

## Latency Tracking (`timing.py`)

### LatencyTracker

Profiles end-to-end latency from user action to first audio playback.

**Key Events:**
- `user_stop_click` - User clicks STOP recording
- `recording_ready` - Audio frames concatenated to WAV
- `asr_complete` - Transcription finished
- `llm_start` / `llm_complete` - LLM generation
- `tts_first_chunk_ready` - First audio chunk from TTS
- `audio_playback_started` - First chunk written to speaker

**Usage:**
```python
from cascade.timing import tracker

tracker.reset("pipeline_start")
tracker.mark("event_name", {"metadata": "value"})
tracker.print_summary()
```

---

## Transcript Analysis (Live Reactions)

The transcript analysis system triggers reactive robot behaviors (sounds, movements, emotions) in real time as the user speaks, **independently of the LLM pipeline**. Reactions fire based on keywords or named entities detected in the ASR transcript.

### Architecture

```
ASR transcript (partial or final)
  │
  └─> TranscriptAnalysisManager
      │
      ├─ KeywordAnalyzer.analyze(text)  ──→ {reaction_name: [matched_words]}
      │
      ├─ EntityAnalyzer.analyze(text)   ──→ [EntityMatch(text, label, confidence)]
      │
      ├─ Deduplication (per-turn)
      │
      └─ Dispatch → reaction.callback(deps, match, **params)
```

The handler calls the manager at three points:
- **`_on_transcript_partial(text)`** — on each streaming ASR partial (debounced at 400ms)
- **`_on_transcript_final(text)`** — when the final transcript is ready (fire-and-forget, parallel with LLM)
- **`_on_turn_complete()`** — resets all deduplication state for the next turn

### Reaction Configuration (`reactions.yaml`)

Each profile can define a `reactions.yaml` file. The loader (`loader.py`) reads it and imports callbacks from the profile's Python modules.

```yaml
# Simple keyword trigger — fires if any word matches
- name: music_excitement
  callback: excited_about_music
  trigger:
    words: [music, guitar, piano, drum, violin]

# Entity trigger with repeatable — fires once per unique entity
- name: food_reaction
  callback: react_to_food_entity
  trigger:
    entities: [food]
  repeatable: true

# Boolean AND trigger — all sub-groups must match
- name: groovy_dance
  callback: do_groovy_dance
  trigger:
    all:
      - words: [danc*]
      - words: [groov*]

# With extra params passed as kwargs to the callback
- name: turn_left
  callback: turn_to_direction
  trigger:
    all:
      - words: [turn*]
      - words: [left]
  params:
    direction: left
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Unique reaction identifier |
| `callback` | yes | Python module name in the profile directory (module must contain a function with the same name) |
| `trigger` | yes | What activates the reaction (see Trigger Types below) |
| `params` | no | Extra kwargs passed to the callback |
| `repeatable` | no | If `true`, can fire multiple times per turn (default: `false`) |

### Trigger Types

#### `words` — Keyword matching

The word list supports three forms, mixable in a single list:

| Form | Example | Matching strategy |
|------|---------|-------------------|
| Plain word | `guitar` | **Substring** match on the full transcript text. Matches "guitar", "guitars", "guitarist". |
| Glob pattern | `music*` | **`fnmatch`** against individual whitespace-split tokens. Matches "music", "musical", "musician". Supports `*` and `?`. |
| Multi-word phrase | `"grand piano"` | **Substring** match (use YAML quotes). Matches any occurrence of "grand piano" in the text. |

Any one match in the list is enough to trigger the reaction (OR logic).

#### `entities` — Named Entity Recognition

Uses GLiNER (optional dependency) to detect entities by semantic label (e.g. `food`, `person`, `location`). The model is configurable in `cascade.yaml`:

```yaml
transcript_analysis:
  gliner_model: "urchade/gliner_small-v2.1"
```

Entity analysis runs in a thread executor since GLiNER inference is CPU-bound.

#### `all` — Boolean AND

A list of sub-triggers that must **all** match for the reaction to fire. Internally, each sub-trigger is registered as a synthetic keyword entry (`reaction_name__all_0`, `reaction_name__all_1`, etc.) and merged back after analysis.

### Deduplication

By default, each reaction fires **at most once per turn** (per conversation exchange). The manager tracks fired reactions in `triggered_reactions: set[str]` and skips duplicates.

**Repeatable reactions** (`repeatable: true`) bypass this gate. For entity-triggered repeatable reactions, deduplication is per unique entity text: "pizza" triggers once, "zucchini" triggers once, but "pizza" again is skipped. This is tracked via `_triggered_entity_keys: set[tuple[str, str]]`.

All deduplication state resets when `_on_turn_complete()` is called.

### Callback Signature

Every callback must be an async function with this signature:

```python
async def my_callback(
    deps: ToolDependencies,
    match: TriggerMatch,
    **kwargs,    # receives params from YAML
) -> None:
```

`TriggerMatch` contains what matched:
- `match.words: list[str]` — matched keywords (for word triggers)
- `match.entities: list[EntityMatch]` — matched entities, each with `.text`, `.label`, `.confidence`

Callbacks are imported from `profiles/<profile_name>/<callback_name>.py` — the module must export a function with the same name as the module.

### Data Classes (`base.py`)

```python
@dataclass
class ReactionConfig:
    name: str
    callback: Callable[..., Awaitable[None]]
    trigger: TriggerConfig
    params: dict[str, Any] = {}
    repeatable: bool = False

@dataclass
class TriggerConfig:
    words: list[str] = []
    entities: list[str] = []
    all: list[TriggerConfig] = []

@dataclass
class TriggerMatch:
    words: list[str] = []
    entities: list[EntityMatch] = []

@dataclass
class EntityMatch:
    text: str          # e.g. "pizza"
    label: str         # e.g. "food"
    confidence: float  # 0.0 – 1.0
```

---

## Design Decisions

**Why Handler doesn't play audio:**
- Gradio UI needs full control for pre-warmed zero-latency playback
- Enables parallel sentence synthesis
- Head wobbler synchronization
- Non-blocking UI updates

**Why TTS is called by UI, not Handler:**
- Handler executes `speak` tool but doesn't synthesize
- UI extracts speak messages from conversation history
- UI can optimize playback strategy independently
- Decouples audio presentation from pipeline logic

**Why conversation history lives in Handler:**
- Handler needs full context for LLM generation
- Tool results must be added for multi-turn reasoning
- Camera tool adds images to conversation for analysis
- UI reads history for display only

---

## Troubleshooting & Known Issues

### MLX Thread Affinity (Apple Silicon)

**Problem:** Local MLX-based providers (Parakeet) may produce empty transcriptions when called from Gradio.

**Root Cause:** MLX has thread affinity requirements. The cascade architecture involves multiple threads:
1. **Main thread** - Gradio UI
2. **Handler event loop thread** - Runs `asyncio` event loop via `threading.Thread`
3. **Thread pool workers** - Created by `asyncio.to_thread()` for "non-blocking" operations

When MLX operations are wrapped in `asyncio.to_thread()`, they execute in different thread pool workers. MLX models loaded in one thread context may not work correctly when inference runs in a different thread.

**Symptoms:**
- Audio chunks are sent successfully (logs show "✓ Sent X samples")
- But `result.text` is always empty
- `draft_tokens` and `finalized_tokens` lists remain empty
- Works fine in standalone tests (single thread), fails in Gradio

**Solution:** Run all MLX operations **synchronously** - no `asyncio.to_thread()`:

```python
# ❌ WRONG - spawns thread pool workers
async def send_audio_chunk(self, chunk):
    def _add_audio():
        with self._lock:
            self.transcriber.add_audio(mx.array(audio))
    await asyncio.to_thread(_add_audio)

# ✅ CORRECT - synchronous, respects thread affinity
async def send_audio_chunk(self, chunk):
    audio_mlx = mx.array(audio)
    self.transcriber.add_audio(audio_mlx)
```

**Why this works:** MLX operations are fast on Apple Silicon (~10-50ms for small chunks). The slight blocking is acceptable and avoids thread context issues.

**Comparison with cloud providers:** Deepgram streaming works with fire-and-forget `asyncio.run_coroutine_threadsafe()` because it only does async network I/O (WebSocket send) - no local compute. The transcription happens server-side.

**Key takeaway:** For local ML inference on Apple Silicon, prefer synchronous execution over threading abstractions.

### OpenAI Realtime ASR - "Streaming" Misconception

**Problem:** Partial transcripts don't appear during speech - they all arrive at once after speech ends.

**Root Cause:** OpenAI's "streaming transcription" means text streams out quickly *after* audio is committed, NOT that you get real-time partials while speaking.

**How it actually works:**
1. Audio chunks sent → buffer on server
2. Server VAD detects silence → commits buffer
3. THEN transcription starts and deltas stream rapidly (~200ms for full text)

**Additional issue - Connection latency:**
- WebSocket connection takes ~800-1000ms to establish after speech starts
- Audio recorded during this time overflows and is lost
- Server VAD may not track speech properly due to this discontinuity

**Configuration options (`cascade.yaml`):**
```yaml
openai_realtime_asr:
  use_server_vad: true   # Real-time partials after each silence detection
  use_server_vad: false  # All transcription at end (manual commit)
```

**With `use_server_vad: true`:**
- Server VAD (500ms silence) and local Silero VAD (700ms) can coexist
- Server streams partials after each detected pause
- Local VAD controls when `end_stream()` is called

**Comparison with Deepgram:**
- Deepgram also streams partials after audio is processed, not during speech
- Both exhibit similar "batch of partials at end" behavior
- True real-time mid-speech partials would require periodic audio commits (causing fragmented transcripts)

**Potential fix (not implemented):** Pre-warm WebSocket connection before speech starts, keeping it in standby mode to eliminate connection latency.

---

## Future Extensions

- Console Mode with VAD (voice activity detection)
- Streaming TTS integration
- Multi-language support with per-provider language codes
- Visual/audio cues for tool execution feedback
- Conversation reset without app restart
- Multi-modal prompts with vision in system prompt
