#!/usr/bin/env python3
"""Query the A5 endpoint location fields consumed by HCOMM."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys


INFO_TYPE_SDID = 26
INFO_TYPE_SERVER_ID = 27
INFO_TYPE_SUPER_POD_ID = 29


class TopologyError(RuntimeError):
    pass


def check(result: int, operation: str) -> None:
    if result != 0:
        raise TopologyError(f"{operation} failed, ret={result}")


def find_symbol(
    libraries: tuple[ctypes.CDLL, ...], name: str
) -> ctypes._CFuncPtr:
    for library in libraries:
        try:
            return getattr(library, name)
        except AttributeError:
            pass
    raise TopologyError(f"required runtime symbol is unavailable: {name}")


def query(logical_device_id: int) -> dict[str, int]:
    try:
        acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
        runtime = ctypes.CDLL("libruntime.so", mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        raise TopologyError(
            f"failed to load CANN runtime; source set_env.sh first: {error}"
        ) from error

    acl_init = find_symbol((acl, runtime), "aclInit")
    acl_init.argtypes = [ctypes.c_char_p]
    acl_init.restype = ctypes.c_int
    acl_finalize = find_symbol((acl, runtime), "aclFinalize")
    acl_finalize.argtypes = []
    acl_finalize.restype = ctypes.c_int
    set_device = find_symbol((acl, runtime), "aclrtSetDevice")
    set_device.argtypes = [ctypes.c_int32]
    set_device.restype = ctypes.c_int
    reset_device = find_symbol((acl, runtime), "aclrtResetDevice")
    reset_device.argtypes = [ctypes.c_int32]
    reset_device.restype = ctypes.c_int
    get_device = find_symbol((acl, runtime), "aclrtGetDevice")
    get_device.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    get_device.restype = ctypes.c_int
    get_physical = find_symbol(
        (runtime, acl), "aclrtGetPhyDevIdByLogicDevId"
    )
    get_physical.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    get_physical.restype = ctypes.c_int
    get_info = find_symbol((runtime, acl), "rtGetDeviceInfo")
    get_info.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    get_info.restype = ctypes.c_int

    initialized = False
    selected = False
    try:
        check(acl_init(None), "aclInit")
        initialized = True
        check(set_device(logical_device_id), "aclrtSetDevice")
        selected = True

        runtime_device = ctypes.c_int32(-1)
        check(get_device(ctypes.byref(runtime_device)), "aclrtGetDevice")
        if runtime_device.value < 0:
            raise TopologyError("aclrtGetDevice returned a negative id")

        physical = ctypes.c_int32(-1)
        check(
            get_physical(runtime_device.value, ctypes.byref(physical)),
            "aclrtGetPhyDevIdByLogicDevId",
        )
        if physical.value < 0:
            raise TopologyError(
                "aclrtGetPhyDevIdByLogicDevId returned a negative id"
            )

        def info(info_type: int, name: str) -> int:
            value = ctypes.c_int64(-1)
            check(
                get_info(
                    runtime_device.value,
                    0,
                    info_type,
                    ctypes.byref(value),
                ),
                f"rtGetDeviceInfo({name})",
            )
            if value.value < 0 or value.value > 0xFFFFFFFF:
                raise TopologyError(
                    f"rtGetDeviceInfo({name}) returned {value.value}"
                )
            return value.value

        return {
            "KVAF_LOGICAL_DEVICE_ID": logical_device_id,
            "KVAF_PHYSICAL_DEVICE_ID": physical.value,
            "KVAF_SUPER_DEVICE_ID": info(INFO_TYPE_SDID, "SDID"),
            "KVAF_SERVER_INDEX": info(INFO_TYPE_SERVER_ID, "SERVER_ID"),
            "KVAF_SUPER_POD_INDEX": info(
                INFO_TYPE_SUPER_POD_ID, "SUPER_POD_ID"
            ),
        }
    finally:
        if selected:
            reset_device(logical_device_id)
        if initialized:
            acl_finalize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logical-device-id", type=int, default=0)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    if args.logical_device_id < 0:
        print("ERROR: logical device id must be non-negative", file=sys.stderr)
        return 2
    try:
        topology = query(args.logical_device_id)
    except TopologyError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.shell:
        for key, value in topology.items():
            print(f"{key}={value}")
    else:
        print(json.dumps(topology, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
