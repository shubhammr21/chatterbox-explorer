"""
src/infrastructure/__init__.py
================================
Infrastructure layer — application settings and dependency-injection wiring.

Packages:
    constants.py — Environment enum: deployment environment with is_debug,
                   is_deployed, and use_json_logs properties. Pure stdlib,
                   safe to import without any optional extras.
    container.py — AppContainer: declarative DI container (dependency-injector)
    settings.py  — RestSettings: pydantic-settings BaseSettings for the REST
                   adapter (loaded only when the 'rest' extra is installed).

Architecture rules enforced here:
  - This package may import from: domain, ports, services, adapters.outbound
  - This package must NOT be imported by: domain, ports, services
  - container.py must only be imported inside build_app() / build_rest_app()
    (deferred import), never at module level — preserves the compat-patch
    ordering guarantee.
  - constants.py and settings.py are safe to import at any time.
"""
