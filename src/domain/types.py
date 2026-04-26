"""
src/domain/types.py
===================
Closed-set type aliases for the entire project.

Every stringly-typed value that belongs to a *known, finite set* is
defined here as a ``Literal`` so that:

  1. Type checkers (mypy / pyright / ty) catch typos and invalid values
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

# Identifier for each Chatterbox model variant:
#   "tts"          — ChatterboxTTS (500M, English, full creative controls)
#   "turbo"        — ChatterboxTurboTTS (350M, English, paralinguistic tags)
#   "multilingual" — ChatterboxMultilingualTTS (500M, 23 languages)
#   "vc"           — ChatterboxVC (voice conversion, audio-in -> audio-out)
ModelKey = Literal["tts", "turbo", "multilingual", "vc"]

# Ordered tuple of every valid ModelKey.
# Prefer iterating over this constant rather than reconstructing the tuple
# from the Literal args — the single definition here keeps both in sync.
ALL_MODEL_KEYS: tuple[ModelKey, ...] = ("tts", "turbo", "multilingual", "vc")

# ──────────────────────────────────────────────────────────────────────────────
# Compute device
# ──────────────────────────────────────────────────────────────────────────────

# The three compute backends supported by Chatterbox:
#   "cuda" — NVIDIA GPU via CUDA (highest throughput).
#   "mps"  — Apple Silicon unified memory via Metal Performance Shaders.
#   "cpu"  — Universal CPU fallback; always available but slowest.
DeviceType = Literal["cuda", "mps", "cpu"]

# ──────────────────────────────────────────────────────────────────────────────
# Watermark verdict
# ──────────────────────────────────────────────────────────────────────────────

# Outcome of a PerTh watermark detection pass:
#   "detected"     — confidence score >= 0.9; watermark is present.
#   "not_detected" — confidence score <= 0.1; no watermark found.
#   "inconclusive" — score in (0.1, 0.9); result is ambiguous.
#   "unavailable"  — detection library not installed or initialisation failed.
WatermarkVerdict = Literal[
    "detected",
    "not_detected",
    "inconclusive",
    "unavailable",
]

# ──────────────────────────────────────────────────────────────────────────────
# Language codes
# ──────────────────────────────────────────────────────────────────────────────

# ISO 639-1 two-letter codes for the 23 languages supported by
# MultilingualTTSService.  The order mirrors the official Chatterbox
# multilingual demo and the LANGUAGE_OPTIONS list in domain.languages.
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

# Ordered tuple of every valid LanguageCode.
# Use this constant when you need to iterate over all supported languages
# without reconstructing the tuple from the Literal args.
ALL_LANGUAGE_CODES: tuple[LanguageCode, ...] = (
    "ar",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sv",
    "sw",
    "tr",
    "zh",
)

# ──────────────────────────────────────────────────────────────────────────────
# Download mode
# ──────────────────────────────────────────────────────────────────────────────

# Strategy used to download model weights from HuggingFace Hub:
#   "files"    — individual hf_hub_download calls for a named file list.
#                Used when only a small subset of repo files is needed.
#   "snapshot" — snapshot_download with allow_patterns glob filters.
#                Used for repos where multiple files must be consistent.
DlMode = Literal["files", "snapshot"]

# ──────────────────────────────────────────────────────────────────────────────
# Model metadata TypedDict
# ──────────────────────────────────────────────────────────────────────────────


class ModelMetadata(TypedDict, total=False):
    """Typed shape of every entry in ``MODEL_REGISTRY``.

    Required fields are present for every model.
    Optional fields depend on ``dl_mode``:

    * ``dl_files``    — populated when ``dl_mode == "files"``
    * ``dl_patterns`` — populated when ``dl_mode == "snapshot"``

    Uses ``Required`` / ``NotRequired`` (PEP 655, Python >= 3.11) for precise
    optionality so the type checker enforces both presence and type.
    """

    # Identity
    display_name: Required[str]
    class_name: Required[str]
    description: Required[str]
    params: Required[str]
    size_gb: Required[float]

    # HuggingFace Hub location
    repo_id: Required[str]
    check_file: Required[str]

    # Download strategy
    dl_mode: Required[DlMode]
    dl_files: NotRequired[list[str]]
    dl_patterns: NotRequired[list[str]]
