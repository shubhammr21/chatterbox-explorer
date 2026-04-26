"""
src/infrastructure/constants.py
=================================
Infrastructure-layer constants for the Chatterbox TTS Explorer.

Environment enum
----------------
A str-enum that represents the deployment environment the application is
running in.  Consuming code can branch on environment properties instead
of comparing raw strings, keeping the logic in one place and making it
easy to extend.

Architecture rules
------------------
- ZERO runtime imports from any third-party package.
  This module must be importable before pydantic, fastapi, starlette, or
  any optional extra is installed.
- No HTTP concepts, no domain models, no adapter imports.
- Safe to import at any point in the startup sequence.
"""

from __future__ import annotations

from enum import StrEnum


class Environment(StrEnum):
    """Deployment environment identifier.

    Inherits from ``str`` so that values compare equal to plain strings and
    can be read directly from environment variables without explicit casting::

        env = Environment(os.environ.get("ENVIRONMENT", "LOCAL"))

    Members
    -------
    LOCAL
        Developer workstation.  Debug mode enabled; plain-text console logs.
    TESTING
        Automated test runs (CI, pytest).  Debug mode enabled; plain-text logs.
    STAGING
        Pre-production deployment.  Debug mode disabled; JSON structured logs.
    PRODUCTION
        Live deployment.  Debug mode disabled; JSON structured logs.
    """

    LOCAL = "LOCAL"
    TESTING = "TESTING"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"

    @property
    def is_debug(self) -> bool:
        """True for LOCAL and TESTING environments.

        Use to enable verbose tracebacks, reload on file change, etc.
        """
        return self in (Environment.LOCAL, Environment.TESTING)

    @property
    def is_deployed(self) -> bool:
        """True for STAGING and PRODUCTION environments.

        Use to enable production-grade safeguards (e.g. disable the
        interactive API docs, enforce stricter CORS, etc.).
        """
        return self in (Environment.STAGING, Environment.PRODUCTION)

    @property
    def use_json_logs(self) -> bool:
        """True when structured JSON logging should be used.

        JSON logs are required in deployed environments so that log
        aggregators (Datadog, CloudWatch, Grafana Loki) can ingest records
        without a custom parser.  Local and test environments use plain-text
        output for human readability.

        This property is intentionally equivalent to ``is_deployed``; it
        exists as a named concept so call sites read as intent rather than
        as a deployment-state check.
        """
        return self.is_deployed
