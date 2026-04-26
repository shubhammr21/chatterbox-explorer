"""
src/adapters/inbound/rest/concurrency.py
==========================================
Application-level concurrency controls for the REST inbound adapter.

inference_semaphore
    An asyncio.Semaphore that serializes GPU/CPU inference requests.
    A single GPU (or MPS device) executes one kernel at a time; the semaphore
    prevents multiple requests from racing to submit inference work simultaneously.

    While a request waits on the semaphore, the event loop is FREE — health
    checks, model-status polls, and admin requests are served normally.
    Only the inference thread is blocked; the event loop thread is not.

    asyncio.Semaphore() is safe to create at module level in Python 3.11+
    (the deprecation was about asyncio.get_event_loop(), not primitive construction).

MAX_CONCURRENT_INFERENCES
    Change to N if you have N independent GPUs; each model-service instance
    must then be configured to use a specific device via CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import asyncio

MAX_CONCURRENT_INFERENCES: int = 1

inference_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
