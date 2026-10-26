#!/usr/bin/env python3
"""Build and run the NPU QoS query on a CANN Linux server using its own HCCP headers."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True, help="Logical device ID, as in DEVICE_ID")
    parser.add_argument("--eid", required=True, help="Actual local EID from PEER_LINK_LOCAL_EID_READY")
    parser.add_argument("--build-dir", type=Path, default=Path("build-npu"))
    parser.add_argument("--hccp-include", type=Path, help="Directory containing the matching installed hccp_ctx.h")
    args = parser.parse_args()
    eid = args.eid.replace(":", "").lower()
    if args.device < 0 or len(eid) != 32 or any(c not in "0123456789abcdef" for c in eid):
        parser.error("device must be nonnegative and EID must contain exactly 32 hexadecimal digits")
    root = Path(__file__).resolve().parents[2]
    build = args.build_dir.resolve()
    cache = build / "CMakeCache.txt"
    cann = None
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if line.startswith("ASCEND_CANN_PACKAGE_PATH:") and "=" in line:
                cann = Path(line.split("=", 1)[1])
                break
    if cann is None:
        value = os.environ.get("ASCEND_CANN_PACKAGE_PATH")
        if value:
            cann = Path(value)
    if cann is None or not (cann / "include/acl/acl.h").is_file():
        parser.error("CANN path unavailable; first source CANN set_env.sh and run ./build.sh npu")
    libraries = sorted({p.resolve() for p in build.rglob("libkvaf_direct_hccp_session.so")})
    if len(libraries) != 1:
        parser.error("Expected one built libkvaf_direct_hccp_session.so; run ./build.sh npu first")
    peer_library = build / "runtime/libkvaf_peer_link.a"
    if not peer_library.is_file():
        parser.error(f"Missing {peer_library}; build it with: "
                     f"cmake --build {shlex.quote(str(build))} --target kvaf_peer_link")
    if args.hccp_include:
        header = args.hccp_include.resolve() / "hccp_ctx.h"
    else:
        headers = sorted({p.resolve() for p in cann.rglob("hccp_ctx.h")})
        if len(headers) != 1:
            parser.error("Cannot select installed hccp_ctx.h; use --hccp-include with the matching SDK header directory. "
                         "Do not substitute headers from a different HCCP release.")
        header = headers[0]
    if not header.is_file():
        parser.error(f"Header does not exist: {header}")
    print(f"HCCP_HEADER={header}", file=sys.stderr)
    print(f"KVAF_LIBRARY={libraries[0]}", file=sys.stderr)
    print(f"KVAF_PEER_LIBRARY={peer_library}", file=sys.stderr)
    output = build / "diagnostics/query_npu_priority_sl"
    output.parent.mkdir(parents=True, exist_ok=True)
    lib_dirs = [libraries[0].parent, cann / "lib64"]
    command = shlex.split(os.environ.get("CXX", "g++")) + [
        "-std=c++17", "-O2", "-D_GNU_SOURCE", "-D_GLIBCXX_USE_CXX11_ABI=0",
        str(Path(__file__).with_suffix(".cc")), "-o", str(output),
        "-I" + str(root / "access_operator/include"),
        "-I" + str(cann / "include"), "-I" + str(header.parent),
    ]
    for directory in lib_dirs:
        command += ["-L" + str(directory), "-Wl,-rpath," + str(directory)]
    # The transport DSO references PeerLink. Normal AIV targets also link
    # kvaf_peer_link; this standalone executable must supply it explicitly.
    # Keep the archive after the DSO so the ELF linker can resolve its symbols.
    command += ["-lkvaf_direct_hccp_session", str(peer_library), "-lascendcl", "-ldl", "-pthread"]
    print("BUILD: " + shlex.join(command), file=sys.stderr)
    subprocess.run(command, check=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")
    return subprocess.run([str(output), str(args.device), eid], env=env).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Query failed: {exc}", file=sys.stderr)
        sys.exit(1)
