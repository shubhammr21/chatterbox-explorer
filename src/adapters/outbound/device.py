"""Device detection and seed management — infra concerns, not domain."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # DeviceType is only referenced in the return annotation of detect_device().
    # With PEP 563 (from __future__ import annotations) all annotations are
    # strings at runtime, so this import is never evaluated outside of a
    # type-checking pass.
    from domain.types import DeviceType

log = logging.getLogger(__name__)


def detect_device() -> DeviceType:
    """Return the best available compute device: 'cuda' > 'mps' > 'cpu'.

    Ordering rationale:
        CUDA   — discrete GPU; highest throughput for large models.
        MPS    — Apple Silicon unified memory; faster than CPU on macOS.
        cpu    — universal fallback; always available.

    torch is imported lazily so this module can be imported without torch
    being present (e.g. during bootstrap, before the compat patches fire).
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int | float) -> None:
    """Set global random seeds for reproducible generation.

    Covers the four RNG states that Chatterbox models touch:
        1. torch CPU generator          — controls all CPU ops
        2. torch CUDA generators        — one per GPU device
        3. Python's built-in ``random`` — used by some samplers
        4. NumPy's global RNG           — used by audio processing utilities

    Args:
        seed: Integer seed value.  ``0`` is treated as "no seed" and is a
              deliberate no-op so callers can pass ``request.seed`` directly
              without an explicit ``if seed != 0`` guard.

    Note:
        MPS does not expose a per-device seed API as of PyTorch 2.x.
        ``torch.manual_seed()`` seeds the CPU generator which MPS kernels
        draw from, so reproducibility on Apple Silicon is still achieved.
    """
    import numpy as np
    import torch

    seed = int(seed)
    if seed == 0:
        return

    torch.manual_seed(seed)

    # Only call cuda.manual_seed_all when CUDA is actually present —
    # the call is a no-op on CPU/MPS but emits a warning on some builds.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    log.debug("Random seed set to %d", seed)
