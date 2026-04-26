"""
REST API inbound adapter — Future FastAPI implementation.

To add the REST API:
    uv add --optional rest "fastapi>=0.100.0" "uvicorn[standard]>=0.20.0"

Then implement routes.py and schemas.py in this package.

Entry point would be:
    uv run uvicorn adapters.inbound.rest.app:app

The services accept the same domain models (TTSRequest, etc.) used by
the Gradio adapter — no domain changes required when adding REST.
"""
