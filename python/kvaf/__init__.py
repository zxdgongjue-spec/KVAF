"""Python API for the synchronous KVAF Runtime.

The default in-tree binding is an in-process CPU MockRuntime.  It exercises
the real Runtime and MetaManager lifecycle without a subprocess or a concrete
hardware AccessOperator.  Hardware-specific Python bindings are opt-in and
will be added alongside their operator implementation.
"""

try:
    from . import _kvaf as _native
except ImportError:
    _native = None

if _native is None:
    raise ImportError(
        "KVAF native bindings are not available. Configure with "
        "-DKVAF_BUILD_PYTHON=ON and add build/python to PYTHONPATH."
    )

AccessTable = _native.AccessTable
Runtime = _native.Runtime
MockRuntime = _native.MockRuntime
create_mock_runtime = _native.create_mock_runtime
runtime = _native.runtime
meta_manager = _native.meta_manager
access_operator = _native.access_operator

if hasattr(_native, "AivBatchGetRuntime"):
    AivBatchGetRuntime = _native.AivBatchGetRuntime
    AIV_BATCH_GET_STATUS_BYTES = _native.AIV_BATCH_GET_STATUS_BYTES
    AIV_BATCH_GET_QUEUE_STATE_BYTES = _native.AIV_BATCH_GET_QUEUE_STATE_BYTES
    AIV_BATCH_GET_STATUS_MAGIC = _native.AIV_BATCH_GET_STATUS_MAGIC

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
