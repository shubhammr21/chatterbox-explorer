"""
chatterbox_explorer.__main__
============================
Allows the package to be executed as a module:

    uv run python -m chatterbox_explorer
    uv run python -m chatterbox_explorer --share
    uv run python -m chatterbox_explorer --port 8080
"""

from chatterbox_explorer.cli import main

if __name__ == "__main__":
    main()
