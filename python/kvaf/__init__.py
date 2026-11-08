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

PutExecutionConfig = _native.PutExecutionConfig
GetExecutionConfig = _native.GetExecutionConfig
ExecutionResources = _native.ExecutionResources
AccessTable = _native.AccessTable
Runtime = _native.Runtime
MockRuntime = _native.MockRuntime
create_mock_runtime = _native.create_mock_runtime
runtime = _native.runtime
meta_manager = _native.meta_manager
access_operator = _native.access_operator

if hasattr(_native, "NpuRuntime"):
    NpuRuntime = _native.NpuRuntime
    AclGraph = _native.AclGraph
    DeviceAccessTable = _native.DeviceAccessTable

__all__ = [
    "PutExecutionConfig",
    "GetExecutionConfig",
    "ExecutionResources",
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
        "AclGraph",
        "DeviceAccessTable",
    ]
