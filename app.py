#!/usr/bin/env python3
"""
app.py — compatibility shim
============================
The application entry point has moved into the installable package.

Preferred invocation (uses the registered script entry point):
    uv run chatterbox-explorer
    uv run chatterbox-explorer --share
    uv run chatterbox-explorer --port 8080

Module invocation (equivalent):
    uv run python -m chatterbox_explorer

This file is kept for two reasons only:
  1. Backward compatibility — existing `uv run python app.py` invocations
     continue to work without any change.
  2. IDE / editor run-button convenience — editors that look for app.py
     can still launch the app.

Nothing else belongs here.  All application logic lives in:
    src/chatterbox_explorer/
"""

from cli import main

if __name__ == "__main__":
    main()
