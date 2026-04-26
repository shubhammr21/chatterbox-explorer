"""
src/infrastructure/config.py
==============================
AppSettings — immutable runtime configuration resolved once at startup.

Responsibility
--------------
Holds the two values that cannot be computed from static configuration alone
and must be resolved at process startup before the DI container is wired:

    device              — best available compute device, detected by
                          adapters.outbound.device.detect_device()
    watermark_available — True when the full PerTh neural watermarker is
                          present; False when the open-source no-op edition
                          is active.  Determined in cli.main() before any
                          chatterbox module is imported.

Relationship to domain.models.AppConfig
----------------------------------------
``AppConfig`` (domain layer) is a domain value-object passed *into* domain
services and the Gradio adapter so they can make runtime decisions
(e.g. whether to show the watermark tab).

``AppSettings`` (infrastructure layer) is the upstream source of those values.
The DI container reads ``AppSettings`` to configure its providers and then
constructs ``AppConfig`` as an injectable domain object.

The distinction keeps the domain layer free of infrastructure concerns:
domain code receives ``AppConfig`` and never knows where those values came from.

Architecture rules
------------------
- Zero runtime imports from torch, gradio, chatterbox, or any adapter package.
- ``DeviceType`` is referenced only under TYPE_CHECKING — no runtime dep on
  ``domain.types`` at import time.
- This module is safe to import at any point in the startup sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.types import DeviceType


@dataclass(frozen=True)
class AppSettings:
    """Immutable snapshot of runtime-resolved application settings.

    Constructed once in ``build_app()`` after:
      1. CLI arguments are parsed.
      2. Compat patches (sdp_kernel, LoRACompatibleLinear) are applied.
      3. The PerTh watermark availability check is performed.
      4. The compute device is detected.

    Passed into ``AppContainer`` so every provider can read settings through
    the container's ``config`` node rather than performing their own detection.

    Attributes:
        device: Best available compute device — one of ``"cuda"``, ``"mps"``,
            or ``"cpu"``.  Detected by
            :func:`adapters.outbound.device.detect_device`.
        watermark_available: ``True`` when
            ``perth.PerthImplicitWatermarker`` is not ``None`` (full PerTh
            edition); ``False`` when the open-source no-op edition is active.
    """

    device: DeviceType
    watermark_available: bool
