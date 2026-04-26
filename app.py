#!/usr/bin/env python3
"""
app.py — compatibility shim
============================
The application entry point lives inside the installed package.

Preferred invocation (uses the registered script entry point):
    uv run chatterbox-explorer
    uv run chatterbox-explorer --share
    uv run chatterbox-explorer --port 8080
    uv run chatterbox-explorer --mcp

Module invocation (equivalent):
    uv run python -m chatterbox_explorer

This file is kept for two reasons only:
  1. Backward compatibility — existing `uv run python app.py` invocations
     continue to work without any change.
  2. IDE / editor run-button convenience — editors that look for app.py
     can still launch the app directly.

Nothing else belongs here.  All application logic lives in:
    src/
    ├── cli.py              ← entry point: argument parsing + startup sequence
    ├── bootstrap.py        ← dependency injection: wires adapters → services
    ├── compat.py           ← migration shims for deprecated third-party APIs
    ├── logging_config.py   ← logging + warning suppression setup
    ├── infrastructure/     ← DI container + app settings
    ├── domain/             ← pure domain models, presets, language data
    ├── ports/              ← ABC port interfaces (input + output)
    ├── services/           ← domain service implementations
    └── adapters/
        ├── inbound/        ← Gradio UI adapter (and future REST / CLI / gRPC)
        └── outbound/       ← infrastructure adapters (models, audio, memory …)
"""

from cli import main

if __name__ == "__main__":
    main()
