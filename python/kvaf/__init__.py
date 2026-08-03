"""Python API for the synchronous KVAF Runtime.

The default in-tree binding is an in-process CPU MockRuntime.  It exercises
the real Runtime and MetaManager lifecycle without a subprocess or a concrete
hardware AccessOperator.  Hardware-specific Python bindings are opt-in and
will be added alongside their operator implementation.
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

if hasattr(_native, "AivBatchGetRuntime"):
    AivBatchGetRuntime = _native.AivBatchGetRuntime
    # These are fixed host/AIV ABI values from batch_get_abi.h. Keep them in
    # Python instead of module-level C++ attribute assignment so the binding
    # also loads with the target's pybind11 2.10.x / Python 3.13 combination.
    AIV_BATCH_GET_STATUS_BYTES = 384
    AIV_BATCH_GET_QUEUE_STATE_BYTES = 64
    AIV_BATCH_GET_STATUS_MAGIC = 0x4B5641464247444F

__all__ = [
    "AccessTable",
    "Runtime",
    "MockRuntime",
    "create_mock_runtime",
    "runtime",
    "meta_manager",
    "access_operator",
]

if hasattr(_native, "AivBatchGetRuntime"):
    __all__ += [
        "AivBatchGetRuntime",
        "AIV_BATCH_GET_STATUS_BYTES",
        "AIV_BATCH_GET_QUEUE_STATE_BYTES",
        "AIV_BATCH_GET_STATUS_MAGIC",
    ]
