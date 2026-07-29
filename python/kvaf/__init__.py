"""Python entry point for the initial KVAF placeholder repository."""

try:
    from . import _kvaf as _native
except ImportError:
    _native = None

build_stage = None
runtime = None
meta_manager = None
access_operator = None

if _native is not None:
    build_stage = _native.build_stage
    runtime = _native.runtime
    meta_manager = _native.meta_manager
    access_operator = _native.access_operator

__all__ = ["build_stage", "runtime", "meta_manager", "access_operator"]
