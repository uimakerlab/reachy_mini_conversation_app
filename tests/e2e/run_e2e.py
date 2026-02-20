#!/usr/bin/env python3
"""Autonomous end-to-end test runner for the Reachy Mini conversation app.

Generates TTS audio, injects it directly into the app's recording pipeline
via PulseAudio stream redirection, captures logs, and verifies outcomes.

Usage:
    python tests/e2e/run_e2e.py              # full run
    python tests/e2e/run_e2e.py --skip-tts   # reuse cached audio
    python tests/e2e/run_e2e.py --delay 15   # longer pause between messages
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_CACHE_DIR = SCRIPT_DIR / "audio_cache"
READINESS_SIGNAL = "Realtime session initialized"
DEFAULT_DELAY = 12  # seconds between utterances
DRAIN_WAIT = 15  # seconds after last utterance before stopping
INJECT_SINK_NAME = "e2e_inject"

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    # --- Memory: save & recall ---
    {
        "name": "introduce_name",
        "utterance": "Hi, my name is Rémi",
        "checks": [
            {"type": "tool_called", "tool": "save_memory", "contains_any": ["Rémi", "Remi", "Remy"]},
            {"type": "no_tool_error"},
        ],
    },
    {
        "name": "share_preference",
        "utterance": "I really enjoy music, especially jazz",
        "checks": [
            {"type": "tool_called", "tool": "save_memory", "contains_any": ["jazz", "music"]},
        ],
    },
    {
        "name": "share_hobby",
        "utterance": "I also like playing basketball on weekends",
        "checks": [
            {"type": "tool_called", "tool": "save_memory", "contains_any": ["basketball", "weekends"]},
        ],
    },
    {
        "name": "recall_name",
        "utterance": "Hey, what's my name?",
        "checks": [
            {"type": "assistant_response_contains", "contains_any": ["Rémi", "Remi", "Remy"]},
        ],
    },
    {
        "name": "recall_preference",
        "utterance": "Do you remember what kind of music I like?",
        "checks": [
            {"type": "assistant_response_contains", "contains_any": ["jazz", "music"]},
        ],
    },
    # --- Tool interaction: dance ---
    {
        "name": "ask_dance",
        "utterance": "Can you dance for me?",
        "checks": [
            {"type": "tool_called", "tool": "dance"},
        ],
    },
    # --- Tool interaction: move head ---
    {
        "name": "move_head",
        "utterance": "Look to the left",
        "checks": [
            {"type": "tool_called", "tool": "move_head", "contains": "left"},
        ],
    },
    # --- General conversation (no tool expected) ---
    {
        "name": "general_question",
        "utterance": "What do you think about the weather today?",
        "checks": [
            {"type": "assistant_response_contains", "contains_any": ["weather", "day", "today", "sun", "rain", "outside"]},
        ],
    },
    # --- Compound recall: remembers multiple facts ---
    {
        "name": "recall_all",
        "utterance": "Can you tell me everything you remember about me?",
        "checks": [
            {"type": "assistant_response_contains", "contains_any": ["Rémi", "Remi", "Remy"]},
            {"type": "assistant_response_contains", "contains_any": ["jazz", "music", "basketball"]},
        ],
    },
]


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def preflight() -> list[str]:
    """Return a list of warnings (empty = all good)."""
    warnings = []
    if not shutil.which("paplay"):
        warnings.append("paplay not found — install pulseaudio-utils")
    if not shutil.which("pactl"):
        warnings.append("pactl not found — install pulseaudio-utils")
    if not os.getenv("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY not set")
    try:
        import reachy_mini_conversation_app  # noqa: F401
    except ImportError:
        warnings.append("reachy_mini_conversation_app not importable — pip install -e .")
    return warnings


# ---------------------------------------------------------------------------
# TTS generation
# ---------------------------------------------------------------------------

def generate_tts(scenarios: list[dict], cache_dir: Path) -> dict[str, Path]:
    """Generate WAV files via OpenAI TTS, caching by scenario name."""
    from openai import OpenAI

    cache_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    wav_map: dict[str, Path] = {}

    for sc in scenarios:
        name = sc["name"]
        wav_path = cache_dir / f"{name}.wav"
        if wav_path.exists():
            print(f"  [cached] {name}: {wav_path}")
            wav_map[name] = wav_path
            continue

        print(f"  [tts]    {name}: generating …", end=" ", flush=True)
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="ash",
            input=sc["utterance"],
            response_format="wav",
        )
        wav_path.write_bytes(response.content)
        print("done")
        wav_map[name] = wav_path

    return wav_map


# ---------------------------------------------------------------------------
# PulseAudio injection: virtual source via null sink + stream redirection
# ---------------------------------------------------------------------------

def pa_create_inject_sink() -> int | None:
    """Create a PulseAudio null sink for audio injection. Returns module ID."""
    result = subprocess.run(
        ["pactl", "load-module", "module-null-sink",
         f"sink_name={INJECT_SINK_NAME}",
         f"sink_properties=device.description=E2E_Audio_Injection"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [!] Failed to create null sink: {result.stderr.strip()}")
        return None
    return int(result.stdout.strip())


def pa_remove_module(module_id: int) -> None:
    """Unload a PulseAudio module by ID."""
    subprocess.run(["pactl", "unload-module", str(module_id)],
                   capture_output=True)


def pa_find_app_source_output(app_pid: int, retries: int = 10) -> str | None:
    """Find the PulseAudio source-output (recording stream) for the app process."""
    for _ in range(retries):
        result = subprocess.run(
            ["pactl", "list", "source-outputs"],
            capture_output=True, text=True,
        )
        # Parse pactl output to find source-output belonging to our app PID
        current_id = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Source Output #"):
                current_id = line.split("#")[1]
            # Match the process ID property
            if current_id and "application.process.id" in line:
                pid_str = line.split("=")[-1].strip().strip('"')
                if pid_str == str(app_pid):
                    return current_id
        time.sleep(1)
    return None


def pa_move_source_output(source_output_id: str, source_name: str) -> bool:
    """Move a recording stream to a different source."""
    result = subprocess.run(
        ["pactl", "move-source-output", source_output_id, source_name],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def play_wav(wav_path: Path, sink: str | None = None) -> None:
    """Play a WAV file via paplay (PulseAudio)."""
    cmd = ["paplay"]
    if sink:
        cmd += ["--device", sink]
    cmd.append(str(wav_path))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# App subprocess management
# ---------------------------------------------------------------------------

def start_app(data_dir: Path, log_file: Path) -> subprocess.Popen:
    """Start the conversation app as a subprocess pointing at *data_dir*."""
    env = os.environ.copy()
    env["REACHY_MINI_DATA_DIRECTORY"] = str(data_dir)
    env["REACHY_MINI_MEMORY_ENABLED"] = "true"

    cmd = [
        sys.executable, "-c",
        "from reachy_mini_conversation_app.main import main; main()",
        "--no-camera",
    ]
    log_fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log_fh,
    )
    # Keep the file handle alive on the process object so it isn't GC'd
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def wait_for_ready(log_file: Path, timeout: float = 90) -> bool:
    """Poll the log file for the readiness signal."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_file.exists():
            text = log_file.read_text(errors="replace")
            if READINESS_SIGNAL in text:
                return True
        time.sleep(1)
    return False


def stop_app(proc: subprocess.Popen, timeout: float = 10) -> None:
    """Send SIGINT, then SIGTERM if the process doesn't exit."""
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=5)
    finally:
        fh = getattr(proc, "_log_fh", None)
        if fh:
            fh.close()


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_jsonl(logs_dir: Path) -> list[dict]:
    """Read all JSONL log files in *logs_dir* and return records in order."""
    records: list[dict] = []
    for f in sorted(logs_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def read_active_memory(data_dir: Path) -> str:
    """Read the active memory file, return empty string if missing."""
    path = data_dir / "memory" / "active_memory.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def read_log_text(log_file: Path) -> str:
    """Read the full app log."""
    if log_file.exists():
        return log_file.read_text(errors="replace")
    return ""


# ---------------------------------------------------------------------------
# Check runners
# ---------------------------------------------------------------------------

def check_tool_called(records: list[dict], check: dict) -> tuple[bool, str]:
    """Verify a tool was called, optionally with content match in args."""
    tool = check.get("tool", "")
    contains = check.get("contains")
    contains_any = check.get("contains_any")

    for r in records:
        if r.get("role") == "tool" and r.get("tool_name") == tool:
            args_str = json.dumps(r.get("args", {})).lower()
            if contains and contains.lower() in args_str:
                return True, f"{tool} called with '{contains}'"
            if contains_any and any(c.lower() in args_str for c in contains_any):
                return True, f"{tool} called with one of {contains_any}"
            if not contains and not contains_any:
                return True, f"{tool} called"

    reason = f"{tool} not called"
    if contains:
        reason += f" with '{contains}'"
    if contains_any:
        reason += f" with any of {contains_any}"
    return False, reason


def check_no_tool_error(records: list[dict], _check: dict) -> tuple[bool, str]:
    """Verify no tool returned an error."""
    for r in records:
        if r.get("role") == "tool":
            result = r.get("result", {})
            if isinstance(result, dict) and "error" in result:
                return False, f"tool error: {result['error']}"
    return True, "no tool errors"


def check_assistant_response_contains(records: list[dict], check: dict) -> tuple[bool, str]:
    """Verify the assistant transcript contains a substring."""
    contains_any = check.get("contains_any", [])
    contains = check.get("contains")
    targets = contains_any if contains_any else ([contains] if contains else [])

    for r in records:
        if r.get("role") == "assistant":
            content = r.get("content", "").lower()
            for t in targets:
                if t.lower() in content:
                    return True, f"assistant said '{t}'"
    return False, f"assistant never mentioned any of {targets}"


CHECK_DISPATCH = {
    "tool_called": check_tool_called,
    "no_tool_error": check_no_tool_error,
    "assistant_response_contains": check_assistant_response_contains,
}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_checks(
    scenarios: list[dict],
    records: list[dict],
    active_memory: str,
    log_text: str,
) -> list[dict]:
    """Run all scenario + global checks and return results."""
    results: list[dict] = []

    for sc in scenarios:
        for check in sc.get("checks", []):
            ctype = check["type"]
            fn = CHECK_DISPATCH.get(ctype)
            if fn:
                passed, detail = fn(records, check)
                results.append({
                    "scenario": sc["name"],
                    "check": ctype,
                    "passed": passed,
                    "detail": detail,
                })

    # Global checks
    if active_memory.strip():
        results.append({"scenario": "(global)", "check": "active_memory_not_empty", "passed": True, "detail": "active memory has content"})
    else:
        results.append({"scenario": "(global)", "check": "active_memory_not_empty", "passed": False, "detail": "active memory is empty"})

    # Conversation log completeness
    user_turns = sum(1 for r in records if r.get("role") == "user")
    assistant_turns = sum(1 for r in records if r.get("role") == "assistant")
    if user_turns > 0 and assistant_turns > 0:
        results.append({"scenario": "(global)", "check": "conversation_logged", "passed": True, "detail": f"{user_turns} user + {assistant_turns} assistant turns"})
    else:
        results.append({"scenario": "(global)", "check": "conversation_logged", "passed": False, "detail": f"user={user_turns}, assistant={assistant_turns} (expected both > 0)"})

    # Split on "Traceback" and check each block; ignore KeyboardInterrupt (expected from SIGINT)
    tb_blocks = log_text.split("Traceback")[1:]  # skip preamble before first Traceback
    real_tracebacks = sum(1 for block in tb_blocks if "KeyboardInterrupt" not in block)
    if real_tracebacks > 0:
        results.append({"scenario": "(global)", "check": "no_tracebacks", "passed": False, "detail": f"{real_tracebacks} traceback(s) in logs"})
    else:
        results.append({"scenario": "(global)", "check": "no_tracebacks", "passed": True, "detail": "no tracebacks"})

    return results


def print_report(results: list[dict], data_dir: Path) -> int:
    """Print pass/fail report. Returns exit code (0 = all passed)."""
    print("\n" + "=" * 60)
    print("E2E TEST REPORT")
    print("=" * 60)

    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])

    for r in results:
        icon = "PASS" if r["passed"] else "FAIL"
        print(f"  [{icon}] {r['scenario']:25s} {r['check']:35s} — {r['detail']}")

    print("-" * 60)
    print(f"  {passed} passed, {failed} failed")
    print(f"  Temp dir for post-mortem: {data_dir}")
    print("=" * 60)
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_names = [sc["name"] for sc in SCENARIOS]
    parser = argparse.ArgumentParser(description="E2E test runner for conversation app")
    parser.add_argument("--skip-tts", action="store_true", help="Reuse cached audio files")
    parser.add_argument("--delay", type=int, default=DEFAULT_DELAY, help=f"Seconds between utterances (default: {DEFAULT_DELAY})")
    parser.add_argument("--scenario", nargs="+", choices=all_names, metavar="NAME",
                        help=f"Run only these scenarios (choices: {', '.join(all_names)})")
    args = parser.parse_args()

    # Filter scenarios if --scenario is specified
    if args.scenario:
        selected = [sc for sc in SCENARIOS if sc["name"] in args.scenario]
    else:
        selected = list(SCENARIOS)

    # Pre-flight
    print("Pre-flight checklist:")
    warnings = preflight()
    if warnings:
        for w in warnings:
            print(f"  [!] {w}")
        print("\nFix the above before running. Aborting.")
        sys.exit(1)
    else:
        print("  [ok] paplay available")
        print("  [ok] pactl available")
        print("  [ok] OPENAI_API_KEY set")
        print("  [ok] app importable")

    if args.scenario:
        print(f"\nRunning {len(selected)}/{len(SCENARIOS)} scenarios: {[s['name'] for s in selected]}")

    # TTS — always generate for all scenarios (so cache stays complete)
    print("\nStep 1: TTS audio generation")
    if args.skip_tts:
        wav_map: dict[str, Path] = {}
        for sc in selected:
            wav_path = AUDIO_CACHE_DIR / f"{sc['name']}.wav"
            if not wav_path.exists():
                print(f"  [!] Missing cached audio: {wav_path}")
                print("      Run without --skip-tts first.")
                sys.exit(1)
            wav_map[sc["name"]] = wav_path
            print(f"  [cached] {sc['name']}: {wav_path}")
    else:
        wav_map = generate_tts(selected, AUDIO_CACHE_DIR)

    # Temp dir for isolated data
    tmp = tempfile.mkdtemp(prefix="e2e_reachy_")
    data_dir = Path(tmp)
    log_file = data_dir / "app.log"
    print(f"\nStep 2: Starting app (data_dir={data_dir})")

    # Create PulseAudio null sink for audio injection
    inject_module_id = pa_create_inject_sink()
    if inject_module_id is None:
        print("  [!] Cannot create PulseAudio null sink. Aborting.")
        sys.exit(1)
    print(f"  Created PulseAudio null sink '{INJECT_SINK_NAME}' (module {inject_module_id})")

    proc = start_app(data_dir, log_file)
    try:
        print(f"  Waiting for readiness ('{READINESS_SIGNAL}') …", end=" ", flush=True)
        if not wait_for_ready(log_file, timeout=90):
            print("TIMEOUT")
            print("  App did not become ready in 90s. Check logs:")
            print(f"    {log_file}")
            stop_app(proc)
            pa_remove_module(inject_module_id)
            sys.exit(1)
        print("ready!")

        # Redirect the app's recording stream to our virtual source
        print("  Redirecting app audio input to virtual source …", end=" ", flush=True)
        so_id = pa_find_app_source_output(proc.pid)
        if so_id is None:
            print("FAILED (source-output not found)")
            print("  The app's recording stream was not found in PulseAudio.")
            print("  This may happen if the audio pipeline hasn't started yet.")
        else:
            monitor = f"{INJECT_SINK_NAME}.monitor"
            ok = pa_move_source_output(so_id, monitor)
            if ok:
                print(f"done (source-output #{so_id} → {monitor})")
            else:
                print(f"FAILED (could not move source-output #{so_id})")

        # Play scenarios — audio goes to the null sink, app reads from its monitor
        print(f"\nStep 3: Playing {len(selected)} scenarios (delay={args.delay}s)")
        for i, sc in enumerate(selected):
            name = sc["name"]
            wav = wav_map[name]
            print(f"  [{i + 1}/{len(SCENARIOS)}] {name}: playing …", end=" ", flush=True)
            play_wav(wav, sink=INJECT_SINK_NAME)
            print("done")
            if i < len(selected) - 1:
                print(f"         waiting {args.delay}s …", end=" ", flush=True)
                time.sleep(args.delay)
                print("ok")

        # Drain
        print(f"\nStep 4: Waiting {DRAIN_WAIT}s for final processing …", end=" ", flush=True)
        time.sleep(DRAIN_WAIT)
        print("done")

    finally:
        print("\nStep 5: Stopping app …", end=" ", flush=True)
        stop_app(proc)
        print("stopped")
        # Clean up PulseAudio module
        pa_remove_module(inject_module_id)
        print(f"  Removed PulseAudio null sink (module {inject_module_id})")

    # Parse and check
    print("\nStep 6: Parsing logs and running checks")
    logs_dir = data_dir / "memory" / "logs"
    records = parse_jsonl(logs_dir)
    active_memory = read_active_memory(data_dir)
    log_text = read_log_text(log_file)

    print(f"  JSONL records: {len(records)}")
    print(f"  Active memory: {len(active_memory)} chars")
    print(f"  App log: {len(log_text)} chars")

    results = run_checks(selected, records, active_memory, log_text)
    exit_code = print_report(results, data_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
