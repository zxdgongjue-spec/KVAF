"""KVAF Runtime, per-call cache execution and independent CPU/NPU Pull APIs.

The portable binding exposes MockRuntime. CANN builds additionally expose
NpuRuntime, device access tables, graph support and the enabled Pull sources.
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
        "DeviceAccessTable",
    ]

if hasattr(_native, "AclGraph"):
    AclGraph = _native.AclGraph
    __all__.append("AclGraph")

for _name in ("CpuPullSourceDesc", "NpuPullSourceDesc", "ReadyToken", "CpuPullState",
              "PutEntry", "PutHandle",
              "CpuPullHbmSource", "NpuPullHbmSource", "CpuPullOptions", "NpuPullOptions"):
    if hasattr(_native, _name):
        globals()[_name] = getattr(_native, _name)
        __all__.append(_name)
