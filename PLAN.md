# PLAN — Pydantic Domain Models + Schema Contracts

> **Selected Flow:** System Design (`flow/system-design.md`)
> **Research:** pydantic v2 Generic BaseModel · frozen models · arbitrary_types_allowed ·
> Self type · abstractmethod on BaseModel · InboundSchema/OutboundSchema pattern
> **Status:** Planning Phase — awaiting approval before implementation begins.

---

## 0. Router Output

| Field  | Value                                                                              |
|--------|------------------------------------------------------------------------------------|
| Flow   | `flow/system-design.md`                                                            |
| Trigger| User decision: adopt pydantic over stdlib dataclasses + typed schema contracts    |
| Research| 2 rounds: pydantic vs dataclass (2025) · pydantic v2 Generic BaseModel patterns  |
| Next   | Plan approved → implement phases 1–3, TDD + ty + ruff before each commit          |

---

## 1. Problem Statement

### 1.1 — No contract enforcement on `to_domain` / `from_domain`

Every REST schema has a `to_domain()` or `from_domain()` method, but nothing
enforces their presence or signature. A new schema can omit them silently:

```python
class NewRequestSchema(BaseModel):   # forgot to_domain()
    text: str
    # ... no error until runtime
```

With typed generic base classes, the contract is enforced at class definition time
and self-documents with precise types (`InboundSchema[TTSRequest]` reads as
"this schema translates to a `TTSRequest`").

### 1.2 — Domain models use stdlib dataclasses

`domain/models.py` uses `@dataclass`. User decision: adopt pydantic `BaseModel`
across the codebase for consistency. Pydantic v2 is the de facto standard in 2025
and brings:

- `model_copy(update=...)` — immutable variant creation
- `model_dump()` — structured serialisation (future use)
- `@computed_field` — `duration_s` on `AudioResult` included in `model_dump()`
- Runtime type validation at construction boundaries

---

## 2. Architecture Constraints

1. **Base classes stay in the adapter layer** — `InboundSchema`/`OutboundSchema`
   import pydantic and must NOT live in the domain. The domain itself stays
   framework-free in terms of base-class hierarchy (it uses pydantic as a
   value-object library, not as a framework dependency).

2. **Domain models: no `Field()` constraints** — validation is the adapter's job
   (REST schemas with `min_length`, `ge`, etc.) and the service's job (raising
   `EmptyTextError`, etc.). Domain pydantic models are frozen value objects, not
   boundary validators.

3. **`language: LanguageCode` stays as a Literal** — pydantic will validate it at
   domain-model construction time, which is a useful bug-catcher. The Gradio adapter
   already extracts the bare code (`"fr"`) before constructing the domain object.
   The service test that passes `"fr - French"` must be updated to pass `"fr"`.

4. **`AudioResult.samples: np.ndarray`** — requires `arbitrary_types_allowed=True`.
   `hash()` will fail at runtime (numpy arrays are unhashable) but attribute
   immutability is still enforced by `frozen=True`.

5. **TDD** — failing test before every new module.

6. **ty + ruff + zero suppressions before every commit.**

---

## 3. Design Decisions (all research-backed)

### 3.1 Generic base classes

```python
# adapters/inbound/rest/schemas.py
import abc
from typing import Generic, TypeVar
from typing import Self                # Python 3.11+
from pydantic import BaseModel

DomainT = TypeVar("DomainT")

class InboundSchema(BaseModel, abc.ABC, Generic[DomainT]):
    """HTTP payload → domain object.  Subclasses must implement to_domain()."""
    @abc.abstractmethod
    def to_domain(self) -> DomainT: ...

class OutboundSchema(BaseModel, abc.ABC, Generic[DomainT]):
    """Domain object → HTTP response payload.  Subclasses must implement from_domain()."""
    @classmethod          # classmethod BEFORE abstractmethod (Python 3.11 rule)
    @abc.abstractmethod
    def from_domain(cls, domain: DomainT) -> Self: ...
```

Concrete schemas:
```python
class TTSRequestSchema(InboundSchema[TTSRequest]):
    text: str = Field(..., min_length=1)

    def to_domain(self) -> TTSRequest:
        return TTSRequest(text=self.text, ...)

class ModelStatusResponse(OutboundSchema[ModelStatus]):
    key: str
    ...

    @classmethod
    def from_domain(cls, status: ModelStatus) -> Self:
        return cls(key=status.key, ...)
```

### 3.2 Domain models as pydantic BaseModel

```python
# domain/models.py
from pydantic import BaseModel, ConfigDict, computed_field

class TTSRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    ref_audio_path: str | None = None
    exaggeration: float = 0.5
    # ... no Field() constraints — validated at adapter boundary

class AudioResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    sample_rate: int
    samples: np.ndarray

    @computed_field        # included in model_dump()
    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate
```

### 3.3 `DomainT` TypeVar location

Lives in `adapters/inbound/rest/schemas.py` alongside the base classes — it is an
adapter-layer concern, not a domain concept.

### 3.4 `HealthResponse`, `MessageResponse`, `ErrorDetail`, `ErrorResponse`

These schemas have no domain counterparts. They stay as plain `BaseModel` subclasses
(not `InboundSchema`/`OutboundSchema`) since neither translation direction applies.

---

## 4. File Changes

```
src/domain/models.py               REWRITE  — stdlib @dataclass → pydantic BaseModel
src/adapters/inbound/rest/schemas.py REWRITE — add InboundSchema/OutboundSchema bases;
                                              update concrete schemas to inherit
src/services/tts.py                UPDATE   — AudioResult construction compatible
                                              (pydantic construction vs model_construct)
tests/unit/domain/test_models.py   UPDATE   — dataclasses.is_dataclass → isinstance checks
tests/unit/services/test_tts_service.py UPDATE — "fr - French" → "fr" (pydantic validates)
tests/unit/adapters/rest/test_rest_schemas.py UPDATE — add InboundSchema/OutboundSchema tests
tests/unit/infrastructure/test_container.py UPDATE — AppConfig.is_dataclass removal
```

No changes to:
- `domain/exceptions.py` — pure Python exceptions, unchanged
- `domain/types.py` — pure Literal types, unchanged
- `ports/` — port ABCs, unchanged
- `services/` — service logic, unchanged (except test fix)
- `infrastructure/` — unchanged
- Gradio adapter — unchanged

---

## 5. Implementation Phases

### Phase 1 — `domain/models.py` → pydantic BaseModel

**TDD Steps:**
1. Write additions to `tests/unit/domain/test_models.py` (RED):
   - `isinstance(TTSRequest(...), BaseModel)` is True
   - `TTSRequest` is frozen — mutation raises `ValidationError`
   - `AudioResult.duration_s` is a computed field present in `model_dump()`
   - `TTSRequest.model_copy(update={"text": "new"})` works
   - `WatermarkResult(verdict="invalid")` raises `ValidationError` (pydantic validates Literal)
   - `AppConfig(device="gpu")` raises `ValidationError` (not a valid DeviceType)

2. Rewrite `domain/models.py` (GREEN):
   - All models → `BaseModel` with `ConfigDict(frozen=True)`
   - `AudioResult` → add `arbitrary_types_allowed=True`, `@computed_field`
   - Keep all `TYPE_CHECKING` guards for types only used in annotations
   - `numpy` remains a runtime import (needed for `arbitrary_types_allowed`)

3. Update `tests/unit/domain/test_models.py`:
   - Replace `dataclasses.is_dataclass(X)` checks with `issubclass(X, BaseModel)`
   - Update `AudioResult.duration_s` test (now a `computed_field`, not plain property)

4. Fix `tests/unit/services/test_tts_service.py`:
   - `language=cast("LanguageCode", "fr - French")` → `language="fr"` (or keep as
     `language=cast("LanguageCode", "fr")`)

5. Fix `tests/unit/infrastructure/test_container.py`:
   - `test_app_config_is_dataclass` → `test_app_config_is_pydantic_model`

6. `ty + ruff + pytest`

---

### Phase 2 — `InboundSchema` / `OutboundSchema` base classes

**TDD Steps:**
1. Write `TestInboundSchema` and `TestOutboundSchema` in
   `tests/unit/adapters/rest/test_rest_schemas.py` (RED):
   - `InboundSchema` is abstract — direct instantiation raises `TypeError`
   - `OutboundSchema` is abstract — direct instantiation raises `TypeError`
   - `TTSRequestSchema` is a subclass of `InboundSchema[TTSRequest]`
   - `ModelStatusResponse` is a subclass of `OutboundSchema[ModelStatus]`
   - A subclass that omits `to_domain()` raises `TypeError` on instantiation
   - A subclass that omits `from_domain()` raises `TypeError` on instantiation
   - `TTSRequestSchema.to_domain()` returns a `TTSRequest` instance
   - `ModelStatusResponse.from_domain(status)` returns a `ModelStatusResponse` instance

2. Update `adapters/inbound/rest/schemas.py` (GREEN):
   - Add `DomainT = TypeVar("DomainT")` at module level
   - Add `InboundSchema(BaseModel, abc.ABC, Generic[DomainT])` with abstract `to_domain()`
   - Add `OutboundSchema(BaseModel, abc.ABC, Generic[DomainT])` with abstract `from_domain()`
   - Update `TTSRequestSchema(BaseModel)` → `TTSRequestSchema(InboundSchema[TTSRequest])`
   - Update `TurboRequestSchema` → `TurboRequestSchema(InboundSchema[TurboTTSRequest])`
   - Update `MultilingualRequestSchema` → `MultilingualRequestSchema(InboundSchema[MultilingualTTSRequest])`
   - Update `ModelStatusResponse` → `ModelStatusResponse(OutboundSchema[ModelStatus])`
   - Update `MemoryStatsResponse` → `MemoryStatsResponse(OutboundSchema[MemoryStats])`
   - Update `WatermarkResponse` → `WatermarkResponse(OutboundSchema[WatermarkResult])`
   - Keep `HealthResponse`, `MessageResponse`, `ErrorDetail`, `ErrorResponse` as plain `BaseModel`

3. `ty + ruff + pytest`

---

### Phase 3 — Final Validation

```bash
uv run ty check                         # 0 errors
uv run ruff check src/ tests/          # All checks passed
uv run ruff format --check src/ tests/ # All formatted
uv run pytest tests/unit/ --cov=src -q # ≥ 95% coverage, all green
```

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `dataclasses.asdict()` calls anywhere in codebase | Low | High | `grep -r "asdict\|is_dataclass"` before merge |
| `AudioResult` construction in services uses `model_construct()` or not | Low | Low | Regular constructor fine — pydantic does only isinstance check |
| `MultilingualTTSRequest.language` Literal validation breaks existing tests | **HIGH** | Medium | Phase 1 step 4 — change "fr - French" → "fr" explicitly |
| `frozen=True` pydantic raises `ValidationError` not `FrozenInstanceError` in tests | Medium | Low | Update test assertions from `FrozenInstanceError` → `ValidationError` |
| `ty` flags `computed_field` property return type on `AudioResult` | Low | Low | `@computed_field` return type is inferred by ty correctly in pydantic v2 |
| `hash()` on `AudioResult` fails at runtime (ndarray) | — | — | Already documented; AudioResult is not used as dict key anywhere |
| `Generic[DomainT]` + `abc.ABC` + `BaseModel` metaclass conflict | Low | High | Research confirmed: pydantic's `ModelMetaclass` extends `ABCMeta` — no conflict |

---

## 7. Decisions Made

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Base classes in `adapters/rest/schemas.py`, NOT domain | Domain must be framework-free; pydantic is an adapter concern |
| 2 | `DomainT = TypeVar("DomainT")` in schemas alongside base classes | TypeVar scope = where it's used |
| 3 | Domain models: `ConfigDict(frozen=True)`, NO `Field()` constraints | Domain = trusted validated state; no double-validation |
| 4 | `language: LanguageCode` kept as Literal in `MultilingualTTSRequest` | Pydantic validates it; adapters already extract bare code before construction |
| 5 | `AudioResult`: regular constructor (not `model_construct()`) | isinstance check is cheap; no deep numpy content validation |
| 6 | `HealthResponse`, `MessageResponse` stay as plain `BaseModel` | No domain counterpart; neither direction applies |
| 7 | `Self` from `typing` (Python 3.11 stdlib) | Correct return type for subclasses; no `typing_extensions` needed |

---

## 8. Success Criteria

- [ ] `isinstance(TTSRequest(...), BaseModel)` is `True`
- [ ] `TTSRequest(...).text = "x"` raises `ValidationError` (frozen)
- [ ] `AudioResult.duration_s` appears in `AudioResult.model_dump()`
- [ ] `InboundSchema` cannot be instantiated directly (`TypeError`)
- [ ] `TTSRequestSchema` is `issubclass(TTSRequestSchema, InboundSchema)` — `True`
- [ ] A schema class missing `to_domain()` raises `TypeError` at instantiation
- [ ] `uv run ty check` — 0 errors
- [ ] `uv run ruff check src/ tests/` — All checks passed
- [ ] `uv run pytest tests/unit/ --cov=src` — ≥ 95% coverage, all green
