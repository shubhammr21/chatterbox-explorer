"""
src/domain/types.py
===================
Closed-set type aliases for the entire project.

Every stringly-typed value that belongs to a *known, finite set* is
defined here as a ``Literal`` so that:

  1. Type checkers (mypy / pyright) catch typos and invalid values
     at static-analysis time, not at runtime.
  2. There is a single canonical source to extend if the set grows
     (e.g. a new model key, a new language, a new device type).
  3. Downstream code can annotate parameters and return types
     precisely instead of using bare ``str``.

Allowed imports: stdlib only (typing).
Forbidden: torch, gradio, chatterbox, numpy, psutil — anything that
           is not in the Python standard library.
"""

from __future__ import annotations

from typing import Literal, NotRequired, Required, TypedDict

# ──────────────────────────────────────────────────────────────────────────────
# Model keys
# ──────────────────────────────────────────────────────────────────────────────

ModelKey = Literal["tts", "turbo", "multilingual", "vc"]
"""Identifier for each Chatterbox model variant.

``"tts"``          — ChatterboxTTS (500M, English, full creative controls)
``"turbo"``        — ChatterboxTurboTTS (350M, English, paralinguistic tags)
``"multilingual"`` — ChatterboxMultilingualTTS (500M, 23 languages)
``"vc"``           — ChatterboxVC (voice conversion, audio-in → audio-out)
"""

ALL_MODEL_KEYS: tuple[ModelKey, ...] = ("tts", "turbo", "multilingual", "vc")
"""Ordered tuple of every valid :data:`ModelKey`.

Prefer iterating over this constant rather than reconstructing the tuple
from the ``Literal`` args — the single definition here keeps both in sync.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Compute device
# ──────────────────────────────────────────────────────────────────────────────

DeviceType = Literal["cuda", "mps", "cpu"]
"""The three compute backends supported by Chatterbox.

``"cuda"`` — NVIDIA GPU via CUDA (highest throughput).
``"mps"``  — Apple Silicon unified memory via Metal Performance Shaders.
``"cpu"``  — Universal CPU fallback; always available but slowest.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Watermark verdict
# ──────────────────────────────────────────────────────────────────────────────

WatermarkVerdict = Literal[
    "detected",
    "not_detected",
    "inconclusive",
    "unavailable",
]
"""Outcome of a PerTh watermark detection pass.

``"detected"``     — confidence score ≥ 0.9; watermark is present.
``"not_detected"`` — confidence score ≤ 0.1; no watermark found.
``"inconclusive"`` — score in (0.1, 0.9); result is ambiguous.
``"unavailable"``  — detection library not installed or initialisation failed.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Language codes
# ──────────────────────────────────────────────────────────────────────────────

LanguageCode = Literal[
    "ar",  # Arabic
    "da",  # Danish
    "de",  # German
    "el",  # Greek
    "en",  # English
    "es",  # Spanish
    "fi",  # Finnish
    "fr",  # French
    "he",  # Hebrew
    "hi",  # Hindi
    "it",  # Italian
    "ja",  # Japanese
    "ko",  # Korean
    "ms",  # Malay
    "nl",  # Dutch
    "no",  # Norwegian
    "pl",  # Polish
    "pt",  # Portuguese
    "ru",  # Russian
    "sv",  # Swedish
    "sw",  # Swahili
    "tr",  # Turkish
    "zh",  # Chinese
]
"""ISO 639-1 two-letter codes for the 23 languages supported by
:class:`~services.tts.MultilingualTTSService`.

The order mirrors the official Chatterbox multilingual demo and the
``LANGUAGE_OPTIONS`` list in :mod:`domain.languages`.
"""

ALL_LANGUAGE_CODES: tuple[LanguageCode, ...] = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr",
    "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
    "pl", "pt", "ru", "sv", "sw", "tr", "zh",
)
"""Ordered tuple of every valid :data:`LanguageCode`.

Use this constant when you need to iterate over all supported languages
without reconstructing the tuple from the ``Literal`` args.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Download mode
# ──────────────────────────────────────────────────────────────────────────────

DlMode = Literal["files", "snapshot"]
"""Strategy used to download model weights from HuggingFace Hub.

``"files"``    — individual ``hf_hub_download`` calls for a named file list.
                 Used when only a small subset of repo files is needed.
``"snapshot"`` — ``snapshot_download`` with ``allow_patterns`` glob filters.
                 Used for repos where multiple files must be consistent.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Model metadata TypedDict
# ──────────────────────────────────────────────────────────────────────────────

class ModelMetadata(TypedDict, total=False):
    """Typed shape of every entry in ``MODEL_REGISTRY``.

    Required fields are present for every model.
    Optional fields depend on :attr:`dl_mode`:

    * ``dl_files``    — populated when ``dl_mode == "files"``
    * ``dl_patterns`` — populated when ``dl_mode == "snapshot"``

    Using ``Required`` / ``NotRequired`` (PEP 655, Python ≥ 3.11) makes this
    explicit so the type checker enforces both presence and type.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    display_name: Required[str]
    """Human-readable label shown in the UI (e.g. ``"Standard TTS"``)."""

    class_name: Required[str]
    """Python class instantiated by ``from_pretrained()``
    (e.g. ``"ChatterboxTTS"``)."""

    description: Required[str]
    """One-line capability summary shown in the Model Manager tab."""

    params: Required[str]
    """Parameter count label, purely informational (e.g. ``"500M"`` or
    ``"—"`` for VC which doesn't have a transformer)."""

    size_gb: Required[float]
    """Approximate total download size in gigabytes."""

    # ── HuggingFace Hub location ───────────────────────────────────────────────
    repo_id: Required[str]
    """HuggingFace Hub repository identifier
    (e.g. ``"ResembleAI/chatterbox"``)."""

    check_file: Required[str]
    """Primary weight filename used as a disk-cache probe.

    ``try_to_load_from_cache(repo_id, check_file)`` is called to determine
    whether the model has been downloaded without touching the network.
    Choose a file that is unique to this model variant (avoids false
    positives when multiple models share the same repo).
    """

    # ── Download strategy ────────────────────────────────────────────────────
    dl_mode: Required[DlMode]
    """Whether to download individual files or an entire snapshot."""

    dl_files: NotRequired[list[str]]
    """Explicit list of filenames to fetch (used when
    ``dl_mode == "files"``).  Must be present and non-empty for that mode."""

    dl_patterns: NotRequired[list[str]]
    """Glob patterns passed to ``snapshot_download(allow_patterns=...)``
    (used when ``dl_mode == "snapshot"``).  Must be present and non-empty
    for that mode."""
