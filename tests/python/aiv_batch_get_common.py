"""Kernel path resolution and cleanup for hardware/performance scripts."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def kernel_paths(mode: str) -> tuple[str, str]:
    paths = {}
    errors = []
    for name in ("KVAF_BATCH_GET_KERNEL", "KVAF_BATCH_PUT_KERNEL"):
        value = os.environ.get(name, "")
        if not value:
            errors.append(f"{name} is not set")
            continue
        path = Path(value).expanduser().resolve()
        if name == "KVAF_BATCH_GET_KERNEL":
            filename = "kvaf_scatter_batch_get_kernels.o" if mode == "cpu_push_scatter" else "kvaf_gather_batch_get_kernels.o"
            if path.name in ("kvaf_execute_batch_get_kernels.o", "kvaf_sequential_batch_get_kernels.o"):
                path = path.with_name(filename)
            elif path.name in ("kvaf_scatter_batch_get_kernels.o", "kvaf_gather_batch_get_kernels.o") and path.name != filename:
                errors.append(f"{mode} requires {filename}, got {path.name}")
        paths[name] = path
        if not path.is_file():
            errors.append(f"{name} does not name an existing file for {mode}: {path}")
    if errors:
        raise ValueError("\n".join(errors) +
                         "\nBuild with cmake --build build-npu -j, then point the kernel "
                         "variables to the .o files in build-npu/bin/.")
    return str(paths["KVAF_BATCH_GET_KERNEL"]), str(paths["KVAF_BATCH_PUT_KERNEL"])


def shutdown_runtime(runtime) -> None:
    """Retain a test failure when device/transport teardown fails as well."""
    original = sys.exc_info()[1]
    if runtime.initialized:
        try:
            runtime.shutdown()
        except Exception as error:
            if original is None:
                raise
            print(f"[KVAF_TEST] shutdown also failed; preserving original exception: {error}",
                  file=sys.stderr, flush=True)


def capture_gather_graph(graph, stream, submit) -> None:
    graph.capture_begin(stream)
    try:
        submit()
    except Exception:
        try:
            graph.capture_end()
        except Exception as error:
            print(f"GATHER_GRAPH_CAPTURE_END_ERROR {error}", file=sys.stderr, flush=True)
        raise
    graph.capture_end()


def close_gather_graph(graph, stream) -> None:
    original = sys.exc_info()[1]
    try:
        stream.synchronize()
        graph.reset()
    except Exception as error:
        if original is None:
            raise
        print(f"GATHER_GRAPH_CLEANUP_ERROR preserving original exception: {error}",
              file=sys.stderr, flush=True)
