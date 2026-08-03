"""Python API for the synchronous KVAF Runtime.

The default in-tree binding is an in-process CPU MockRuntime.  It exercises
the real Runtime and MetaManager lifecycle without a subprocess or a concrete
hardware AccessOperator. The optional CANN build additionally exposes
``NpuRuntime`` for the current synchronous BatchGet implementation.
"""

try:
    from . import _kvaf as _native
except ImportError as error:
    raise ImportError(
        "KVAF native bindings could not be loaded. Configure with "
        "-DKVAF_BUILD_PYTHON=ON, add build/python to PYTHONPATH, and inspect "
        f"this dynamic-loader error: {error}"
    ) from error

AccessTable = _native.AccessTable
Runtime = _native.Runtime
MockRuntime = _native.MockRuntime
create_mock_runtime = _native.create_mock_runtime
runtime = _native.runtime
meta_manager = _native.meta_manager
access_operator = _native.access_operator

if hasattr(_native, "NpuRuntime"):
    NpuRuntime = _native.NpuRuntime
    # These are fixed host/kernel ABI values from batch_get_abi.h. Keep them in
    # Python instead of module-level C++ attribute assignment so the binding
    # also loads with the target's pybind11 2.10.x / Python 3.13 combination.
    NPU_BATCH_GET_STATUS_BYTES = 384
    NPU_BATCH_GET_QUEUE_STATE_BYTES = 64
    NPU_BATCH_GET_STATUS_MAGIC = 0x4B5641464247444F

__all__ = [
    "AccessTable",
    "Runtime",
    "MockRuntime",
    "create_mock_runtime",
    "runtime",
    "meta_manager",
    "access_operator",
]

if hasattr(_native, "NpuRuntime"):
    __all__ += [
        "NpuRuntime",
        "NPU_BATCH_GET_STATUS_BYTES",
        "NPU_BATCH_GET_QUEUE_STATE_BYTES",
        "NPU_BATCH_GET_STATUS_MAGIC",
    ]
