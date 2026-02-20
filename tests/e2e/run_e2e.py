#!/usr/bin/env python3
"""Autonomous end-to-end test runner for the Reachy Mini conversation app.

Generates TTS audio, plays it through laptop speakers so the robot's mic picks
it up, captures logs, and verifies outcomes.

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

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "introduce_name",
        "utterance": "Hi, my name is Rémi",
        "checks": [
            {"type": "tool_called", "tool": "save_memory", "contains": "Rémi"},
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
        "name": "recall_name",
        "utterance": "Hey, what's my name?",
        "checks": [
            {"type": "assistant_response_contains", "contains_any": ["Rémi", "Remi"]},
        ],
    },
    {
        "name": "recall_preference",
        "utterance": "Do you remember what I like?",
        "checks": [
            {"type": "assistant_response_contains", "contains_any": ["jazz", "music"]},
        ],
    },
]

GLOBAL_CHECKS = [
    {"type": "active_memory_not_empty"},
    {"type": "no_tracebacks"},
]


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def preflight() -> list[str]:
    """Return a list of warnings (empty = all good)."""
    warnings = []
    if not shutil.which("paplay"):
        warnings.append("paplay not found — install pulseaudio-utils")
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
        response.stream_to_file(str(wav_path))
        print("done")
        wav_map[name] = wav_path

    return wav_map


# ---------------------------------------------------------------------------
# App subprocess management
# ---------------------------------------------------------------------------

def start_app(data_dir: Path, log_file: Path) -> subprocess.Popen:
    """Start the conversation app as a subprocess pointing at *data_dir*."""
    env = os.environ.copy()
    env["REACHY_MINI_DATA_DIRECTORY"] = str(data_dir)
    env["REACHY_MINI_MEMORY_ENABLED"] = "true"

    cmd = [
        sys.executable, "-m", "reachy_mini_conversation_app",
        "--log-file", str(log_file),
        "--no-camera",
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
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


# ---------------------------------------------------------------------------
# Audio playback
# ---------------------------------------------------------------------------

def play_wav(wav_path: Path) -> None:
    """Play a WAV file via paplay (PulseAudio)."""
    subprocess.run(["paplay", str(wav_path)], check=True)


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

    if "Traceback" in log_text:
        count = log_text.count("Traceback")
        results.append({"scenario": "(global)", "check": "no_tracebacks", "passed": False, "detail": f"{count} traceback(s) in logs"})
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
    parser = argparse.ArgumentParser(description="E2E test runner for conversation app")
    parser.add_argument("--skip-tts", action="store_true", help="Reuse cached audio files")
    parser.add_argument("--delay", type=int, default=DEFAULT_DELAY, help=f"Seconds between utterances (default: {DEFAULT_DELAY})")
    args = parser.parse_args()

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
        print("  [ok] OPENAI_API_KEY set")
        print("  [ok] app importable")

    print("\n** Ensure laptop volume is up and robot is within earshot. **\n")

    # TTS
    print("Step 1: TTS audio generation")
    if args.skip_tts:
        wav_map: dict[str, Path] = {}
        for sc in SCENARIOS:
            wav_path = AUDIO_CACHE_DIR / f"{sc['name']}.wav"
            if not wav_path.exists():
                print(f"  [!] Missing cached audio: {wav_path}")
                print("      Run without --skip-tts first.")
                sys.exit(1)
            wav_map[sc["name"]] = wav_path
            print(f"  [cached] {sc['name']}: {wav_path}")
    else:
        wav_map = generate_tts(SCENARIOS, AUDIO_CACHE_DIR)

    # Temp dir for isolated data
    tmp = tempfile.mkdtemp(prefix="e2e_reachy_")
    data_dir = Path(tmp)
    log_file = data_dir / "app.log"
    print(f"\nStep 2: Starting app (data_dir={data_dir})")

    proc = start_app(data_dir, log_file)
    try:
        print(f"  Waiting for readiness ('{READINESS_SIGNAL}') …", end=" ", flush=True)
        if not wait_for_ready(log_file, timeout=90):
            print("TIMEOUT")
            print("  App did not become ready in 90s. Check logs:")
            print(f"    {log_file}")
            stop_app(proc)
            sys.exit(1)
        print("ready!")

        # Play scenarios
        print(f"\nStep 3: Playing {len(SCENARIOS)} scenarios (delay={args.delay}s)")
        for i, sc in enumerate(SCENARIOS):
            name = sc["name"]
            wav = wav_map[name]
            print(f"  [{i + 1}/{len(SCENARIOS)}] {name}: playing …", end=" ", flush=True)
            play_wav(wav)
            print("done")
            if i < len(SCENARIOS) - 1:
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

    # Parse and check
    print("\nStep 6: Parsing logs and running checks")
    logs_dir = data_dir / "memory" / "logs"
    records = parse_jsonl(logs_dir)
    active_memory = read_active_memory(data_dir)
    log_text = read_log_text(log_file)

    print(f"  JSONL records: {len(records)}")
    print(f"  Active memory: {len(active_memory)} chars")
    print(f"  App log: {len(log_text)} chars")

    results = run_checks(SCENARIOS, records, active_memory, log_text)
    exit_code = print_report(results, data_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
