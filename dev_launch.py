"""Dev launcher - bypasses ReachyMiniApp.wrapped_run() to start the web UI.

Requires a running daemon (reachy-mini-daemon) on localhost.
For USB-connected robots:  reachy-mini-daemon
For simulation:            reachy-mini-daemon --sim
"""
import sys

sys.argv = [sys.argv[0], "--web"]

from reachy_mini_conversation_app.utils import parse_args
from reachy_mini_conversation_app.main import run

args, _ = parse_args()
run(args)
