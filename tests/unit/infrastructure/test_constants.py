"""
tests/unit/infrastructure/test_constants.py
=============================================
TDD unit tests for infrastructure/constants.py — Environment enum.

Written BEFORE the implementation exists (RED phase).

What is tested
--------------
1. All four environment values exist and are the expected strings.
2. is_debug — True for LOCAL and TESTING; False for STAGING and PRODUCTION.
3. is_deployed — True for STAGING and PRODUCTION; False for LOCAL and TESTING.
4. use_json_logs — True for deployed environments; False for local ones.
5. Enum is a str subclass — values compare equal to plain strings.
6. Can be constructed from a plain string (e.g. read from an env var).
7. Purity — the module imports ONLY from stdlib; no pydantic, no starlette,
   no fastapi, no third-party package of any kind.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _fresh_import():
    """Return the infrastructure.constants module, bypassing the cache."""
    sys.modules.pop("infrastructure.constants", None)
    return importlib.import_module("infrastructure.constants")


# ──────────────────────────────────────────────────────────────────────────────
# Module purity
# ──────────────────────────────────────────────────────────────────────────────


class TestModulePurity:
    """infrastructure.constants must have zero third-party runtime dependencies."""

    def test_importable_without_pydantic(self) -> None:
        sys.modules.pop("infrastructure.constants", None)
        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "pydantic", None)  # type: ignore[arg-type]
            mp.setitem(sys.modules, "pydantic_settings", None)  # type: ignore[arg-type]
            sys.modules.pop("infrastructure.constants", None)
            mod = importlib.import_module("infrastructure.constants")
        assert mod is not None

    def test_importable_without_fastapi(self) -> None:
        sys.modules.pop("infrastructure.constants", None)
        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "fastapi", None)  # type: ignore[arg-type]
            sys.modules.pop("infrastructure.constants", None)
            mod = importlib.import_module("infrastructure.constants")
        assert mod is not None

    def test_importable_without_starlette(self) -> None:
        sys.modules.pop("infrastructure.constants", None)
        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "starlette", None)  # type: ignore[arg-type]
            sys.modules.pop("infrastructure.constants", None)
            mod = importlib.import_module("infrastructure.constants")
        assert mod is not None

    def test_environment_class_is_present(self) -> None:
        mod = _fresh_import()
        assert hasattr(mod, "Environment"), (
            "infrastructure.constants must export an 'Environment' class"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Enum values
# ──────────────────────────────────────────────────────────────────────────────


class TestEnvironmentValues:
    """All four expected environment names must be present and correct."""

    def test_local_exists(self) -> None:
        from infrastructure.constants import Environment

        assert hasattr(Environment, "LOCAL")

    def test_staging_exists(self) -> None:
        from infrastructure.constants import Environment

        assert hasattr(Environment, "STAGING")

    def test_testing_exists(self) -> None:
        from infrastructure.constants import Environment

        assert hasattr(Environment, "TESTING")

    def test_production_exists(self) -> None:
        from infrastructure.constants import Environment

        assert hasattr(Environment, "PRODUCTION")

    def test_local_value_is_string_local(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.LOCAL == "LOCAL"

    def test_staging_value_is_string_staging(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.STAGING == "STAGING"

    def test_testing_value_is_string_testing(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.TESTING == "TESTING"

    def test_production_value_is_string_production(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.PRODUCTION == "PRODUCTION"

    def test_exactly_four_members(self) -> None:
        from infrastructure.constants import Environment

        assert len(Environment) == 4

    def test_is_str_subclass(self) -> None:
        """str enum — values must compare equal to plain strings."""
        from infrastructure.constants import Environment

        assert issubclass(Environment, str)


# ──────────────────────────────────────────────────────────────────────────────
# str-enum construction from plain strings
# ──────────────────────────────────────────────────────────────────────────────


class TestEnvironmentFromString:
    """Environment must be constructable from plain strings (e.g. from env vars)."""

    @pytest.mark.parametrize("value", ["LOCAL", "STAGING", "TESTING", "PRODUCTION"])
    def test_all_values_constructable(self, value: str) -> None:
        from infrastructure.constants import Environment

        env = Environment(value)
        assert env == value

    def test_invalid_value_raises(self) -> None:
        from infrastructure.constants import Environment

        with pytest.raises(ValueError):
            Environment("UNKNOWN")

    def test_lowercase_raises(self) -> None:
        """Values are uppercase-only — 'local' is not a valid member."""
        from infrastructure.constants import Environment

        with pytest.raises(ValueError):
            Environment("local")


# ──────────────────────────────────────────────────────────────────────────────
# is_debug property
# ──────────────────────────────────────────────────────────────────────────────


class TestIsDebug:
    """is_debug must be True for LOCAL and TESTING; False for STAGING and PRODUCTION."""

    def test_local_is_debug(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.LOCAL.is_debug is True

    def test_testing_is_debug(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.TESTING.is_debug is True

    def test_staging_is_not_debug(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.STAGING.is_debug is False

    def test_production_is_not_debug(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.PRODUCTION.is_debug is False

    def test_is_debug_returns_bool(self) -> None:
        from infrastructure.constants import Environment

        assert isinstance(Environment.LOCAL.is_debug, bool)
        assert isinstance(Environment.PRODUCTION.is_debug, bool)


# ──────────────────────────────────────────────────────────────────────────────
# is_deployed property
# ──────────────────────────────────────────────────────────────────────────────


class TestIsDeployed:
    """is_deployed must be True for STAGING and PRODUCTION; False for LOCAL and TESTING."""

    def test_staging_is_deployed(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.STAGING.is_deployed is True

    def test_production_is_deployed(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.PRODUCTION.is_deployed is True

    def test_local_is_not_deployed(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.LOCAL.is_deployed is False

    def test_testing_is_not_deployed(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.TESTING.is_deployed is False

    def test_is_debug_and_is_deployed_are_mutually_exclusive(self) -> None:
        """No environment can be both debug and deployed."""
        from infrastructure.constants import Environment

        for env in Environment:
            assert not (env.is_debug and env.is_deployed), (
                f"{env} cannot be both is_debug=True and is_deployed=True"
            )

    def test_is_deployed_returns_bool(self) -> None:
        from infrastructure.constants import Environment

        assert isinstance(Environment.STAGING.is_deployed, bool)
        assert isinstance(Environment.LOCAL.is_deployed, bool)


# ──────────────────────────────────────────────────────────────────────────────
# use_json_logs property
# ──────────────────────────────────────────────────────────────────────────────


class TestUseJsonLogs:
    """use_json_logs selects JSON output for deployed envs; plain text for local."""

    def test_staging_uses_json_logs(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.STAGING.use_json_logs is True

    def test_production_uses_json_logs(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.PRODUCTION.use_json_logs is True

    def test_local_does_not_use_json_logs(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.LOCAL.use_json_logs is False

    def test_testing_does_not_use_json_logs(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.TESTING.use_json_logs is False

    def test_use_json_logs_matches_is_deployed(self) -> None:
        """use_json_logs must always equal is_deployed — they are the same concept."""
        from infrastructure.constants import Environment

        for env in Environment:
            assert env.use_json_logs == env.is_deployed, (
                f"{env}: use_json_logs={env.use_json_logs} but is_deployed={env.is_deployed}"
            )

    def test_use_json_logs_returns_bool(self) -> None:
        from infrastructure.constants import Environment

        assert isinstance(Environment.PRODUCTION.use_json_logs, bool)
        assert isinstance(Environment.LOCAL.use_json_logs, bool)


# ──────────────────────────────────────────────────────────────────────────────
# Comparison and identity
# ──────────────────────────────────────────────────────────────────────────────


class TestEnvironmentComparison:
    """str-enum equality and identity semantics."""

    def test_local_equals_string_local(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.LOCAL == "LOCAL"

    def test_production_equals_string_production(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.PRODUCTION == "PRODUCTION"

    def test_different_envs_not_equal(self) -> None:
        from infrastructure.constants import Environment

        assert Environment.LOCAL != Environment.PRODUCTION
        assert Environment.STAGING != Environment.TESTING

    def test_enum_identity_from_string_construction(self) -> None:
        """Environment('LOCAL') must return the same singleton as Environment.LOCAL."""
        from infrastructure.constants import Environment

        assert Environment("LOCAL") is Environment.LOCAL
        assert Environment("PRODUCTION") is Environment.PRODUCTION
