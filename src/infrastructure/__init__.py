"""
src/infrastructure/__init__.py
================================
Infrastructure layer — application settings and dependency-injection wiring.

Packages:
    config.py    — AppSettings: immutable runtime configuration resolved at startup
    container.py — AppContainer: declarative DI container (dependency-injector)

Architecture rules enforced here:
  - This package may import from: domain, ports, services, adapters.outbound
  - This package must NOT be imported by: domain, ports, services
  - container.py must only be imported inside build_app() (deferred import),
    never at module level — preserves the compat-patch ordering guarantee.
"""
