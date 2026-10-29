#!/usr/bin/env python3
"""Collect a sequential AIV compiler report without loading or replacing kernels.

Run on the NPU build host: python tests/python/collect_aiv_scatter_debug.py build-npu
Uses the existing Unix Makefiles recipe, not a reconstructed list of SDK flags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


SOURCE = "execute_sequential_batch_get_aiv.asc"
TEMP_OBJECT = "kvaf_sequential_batch_get.tmp.o"
OBJECT = "kvaf_sequential_batch_get_kernels.o"


def compile_recipe(build: Path):
    matches = []
    for makefile in build.rglob("build.make"):
        for line in makefile.read_text().splitlines():
            if not line.startswith("\t") or SOURCE not in line or TEMP_OBJECT not in line:
                continue
            argv = shlex.split(line.strip())
            cwd = build
            if argv[:1] == ["cd"]:
                if len(argv) < 4 or argv[2] != "&&":
                    raise ValueError(f"unsupported working-directory recipe in {makefile}")
                cwd = (build / argv[1]).resolve()
                argv = argv[3:]
            if any(arg in ("&&", ";", "|", ">", "<") or "$(" in arg for arg in argv):
                raise ValueError(f"unsupported shell/make expansion in {makefile}")
            if argv.count("-c") != 1 or argv.count("-o") != 1 or argv[-1] == "-o":
                raise ValueError(f"expected one compile and output argument in {makefile}")
            sources = [arg for arg in argv if Path(arg).name == SOURCE]
            output = (cwd / argv[argv.index("-o") + 1]).resolve()
            if len(sources) != 1 or output.name != TEMP_OBJECT:
                raise ValueError(f"ambiguous sequential compilation in {makefile}")
            matches.append((argv, cwd, (cwd / sources[0]).resolve(), output))
    if len(matches) != 1:
        raise ValueError(f"expected one sequential CANN compile recipe, found {len(matches)}; "
                         "use the existing Unix Makefiles build directory")
    return matches[0]


def run(argv, cwd: Path, log: Path) -> int:
    with log.open("w") as stream:
        stream.write("COMMAND " + shlex.join([str(a) for a in argv]) + "\n")
        stream.flush()
        try:
            result = subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=120, check=False)  # Never execute a shell recipe.
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(f"\nCOLLECT_ERROR {error}\n")
            code = 124
        stream.write(f"\nRETURN_CODE {code}\n")
    return code


def snapshot(path: Path, destination: Path, records: list) -> None:
    record = {"path": str(path), "resolved": str(path.resolve()), "exists": path.is_file()}
    if path.is_file():
        data = path.read_bytes()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        record.update(saved=str(destination), bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest(), mtime_ns=path.stat().st_mtime_ns)
    records.append(record)


def init_definitions(text: str) -> list[dict]:
    # Inspect active, preprocessed definitions, not a header-name whitelist or
    # the first occurrence (which is commonly a declaration or wrapper call).
    # Blank literals/comments while preserving offsets for brace matching.
    tokens = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    masked = re.sub(tokens, lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    pattern = r"\b(InitSocState(?:Impl)?)\s*\([^(){};]*\)\s*\{"
    definitions = []
    for match in re.finditer(pattern, masked):
        depth, end = 1, match.end()
        while depth and end < len(masked):
            depth += (masked[end] == "{") - (masked[end] == "}")
            end += 1
        if depth:
            continue
        start = text.rfind("\n", 0, match.start()) + 1
        path, line = "<unknown>", 1
        for previous in text[:start].splitlines():
            marker = re.match(r'^#\s+(\d+)\s+"([^"\n]+)"', previous)
            if marker:
                line, path = int(marker[1]), marker[2]
            else:
                line += 1
        definitions.append({"name": match[1], "path": path, "line": line,
                            "ii_line": text.count("\n", 0, start) + 1,
                            "text": text[start:end]})
    return definitions


def init_evidence(output: Path) -> tuple[str, list[dict]]:
    report = ["ACTIVE INITIALIZATION SOURCE (not an instruction/PC mapping):"]
    macros = output / "macros.txt"
    if macros.is_file():
        # Avoid hundreds of ASCEND header guards and unrelated library macros.
        pattern = r"^#define (?:__CCE_\w*|__DAV_\w*|__NPU_ARCH__|ASCEND_IS_AI[CV])\b"
        report.extend(["ARCHITECTURE MACROS:", "\n".join(
            line for line in macros.read_text(errors="replace").splitlines() if re.match(pattern, line))])
    preprocessed = output / "sequential.ii"
    if not preprocessed.is_file():
        return "\n\n".join(report + ["MISSING sequential.ii"]), []
    data = preprocessed.read_bytes()
    report.append(f"PREPROCESSED: {preprocessed}\nSHA256: {hashlib.sha256(data).hexdigest()}")
    definitions = init_definitions(data.decode(errors="replace"))
    for definition in definitions:
        report.append(f"DEFINITION: {definition['name']}\n"
                      f"SOURCE: {definition['path']}:{definition['line']}\n"
                      f"PREPROCESSED LINE: {definition['ii_line']}\n{definition['text']}")
    if not any(d["name"] == "InitSocStateImpl" for d in definitions):
        report.append("InitSocStateImpl DEFINITION NOT FOUND; wrapper alone is insufficient.")
    return "\n\n".join(report), definitions


def analyze_saved(output: Path) -> tuple[Path, bool]:
    # Read existing evidence only: do not invoke the compiler, load a kernel,
    # reread today's SDK headers, or replace the original collection report.
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    evidence, definitions = init_evidence(output)
    complete = any(d["name"] == "InitSocStateImpl" for d in definitions)
    report = ["KVAF saved initialization source analysis", "Not a hardware PASS or a fault-PC mapping.",
              "SAVED MANIFEST: " + str(manifest_path),
              "COMPILE COMMAND: " + shlex.join(manifest["compile_command"]),
              "ORIGINAL STEPS: " + json.dumps(manifest["steps"]), evidence,
              "INIT_SOURCE_FOUND" if complete else "INIT_SOURCE_INCOMPLETE"]
    with tempfile.NamedTemporaryFile(mode="w", prefix="init-report-", suffix=".txt",
                                     dir=output, delete=False) as stream:
        stream.write("\n\n".join(report) + "\n")
        return Path(stream.name), complete


def collect(build: Path, kernel: Path | None = None) -> tuple[Path, bool]:
    argv, cwd, source, temporary = compile_recipe(build)
    output = Path(tempfile.mkdtemp(prefix="scatter-debug-", dir=build))
    print(f"[KVAF_DEBUG] output={output}", flush=True)
    manifest = {"compile_command": argv, "compile_cwd": str(cwd), "files": [], "steps": {}}
    files = manifest["files"]
    # Save the faulting artifacts before invoking any compiler operation.
    for label, path in (("installed", build / "bin" / OBJECT), ("temporary", temporary),
                        ("linked", temporary.with_name(OBJECT))):
        snapshot(path, output / "original" / label / path.name, files)
    if kernel is not None:
        kernel = kernel.expanduser().resolve()
        if kernel.name == "kvaf_execute_batch_get_kernels.o":
            kernel = kernel.with_name(OBJECT)
        snapshot(kernel, output / "original" / "configured" / kernel.name, files)
    snapshot(source, output / "source" / source.name, files)
    for name in ("sequential_batch_get_abi.h", "batch_get_abi.h"):
        snapshot(source.parent.parent / "include" / name, output / "source" / name, files)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    compiler = Path(argv[0])
    if not compiler.is_absolute():
        compiler = ((cwd / compiler).absolute() if "/" in str(compiler) else
                    Path(shutil.which(str(compiler)) or str(cwd / compiler)))
    # Preserve the invoked symlink name (ccec vs bisheng can select driver mode).
    manifest["compiler_resolved"] = str(compiler.resolve())
    cache = (build / "CMakeCache.txt").read_text()
    paths = {}
    for line in cache.splitlines():
        match = re.match(r"(KVAF_CCEC|KVAF_LLD|KVAF_KERNEL_OPERATOR_DIR|KVAF_SIMT_API_DIR|"
                         r"ASCEND_CANN_PACKAGE_PATH|CMAKE_GENERATOR|CMAKE_HOME_DIRECTORY):[^=]+=(.*)", line)
        if match:
            paths[match[1]] = match[2]
    manifest["cache_paths"] = paths
    roots = set()
    if paths.get("ASCEND_CANN_PACKAGE_PATH"):
        roots.add(Path(paths["ASCEND_CANN_PACKAGE_PATH"]).resolve())
    # Also cover toolkit paths supplied as a non-cache CMake variable.
    for arg in argv:
        if arg.startswith("-I") and "/tools/tikcpp/" in arg:
            roots.add(Path(arg[2:].split("/tools/tikcpp/", 1)[0]).resolve())
    for i, root in enumerate(sorted(roots)):
        for name in ("version.cfg", "version.info", "ascend_toolkit_install.info",
                     "aarch64-linux/ascend_toolkit_install.info", "compiler/version.info"):
            path = root / name
            if path.is_file():
                snapshot(path, output / "versions" / str(i) / name, files)
    for component in ("driver", "firmware"):
        path = Path("/usr/local/Ascend") / component / "version.info"
        if path.is_file():
            snapshot(path, output / "versions" / component / path.name, files)

    codes = manifest["steps"]
    codes["compiler_version"] = run([str(compiler), "--version"], output, output / "compiler_version.txt")
    if paths.get("KVAF_LLD"):
        codes["linker_version"] = run([paths["KVAF_LLD"], "--version"], output, output / "linker_version.txt")
    # Preserve every original optimization/architecture/include flag. Only
    # change the compilation action and output, never write the original .o.
    base = [str(compiler)]
    skip = False
    for arg in argv[1:]:
        if skip:
            skip = False
        elif arg == "-o":
            skip = True
        elif arg != "-c":
            base.append(arg)
    # CANN 9.1 rejects device-side -S. Collect supported preprocessing evidence;
    # do not guess alternative assembly flags or claim an instruction mapping.
    for name, flags, filename in (("macros", ["-E", "-dM"], "macros.txt"),
                                  ("preprocess", ["-E"], "sequential.ii")):
        print(f"[KVAF_DEBUG] phase={name}", flush=True)
        target = output / filename
        codes[name] = run(base + flags + ["-o", str(target)], cwd, output / f"{name}.log")
        if codes[name] == 0 and (not target.is_file() or target.stat().st_size == 0):
            codes[name] = 125

    evidence, definitions = init_evidence(output)
    codes["init_impl"] = 0 if any(d["name"] == "InitSocStateImpl" for d in definitions) else 125
    headers = []
    preprocessed = output / "sequential.ii"
    if preprocessed.is_file():
        names = {"kernel_operator_common_impl.h", "kernel_operator_common_intf_impl.h",
                 "kernel_operator_common_intf.h", "kernel_operator_sys_var_impl.h",
                 "kernel_operator_data_copy_impl.h", "kernel_operator_data_copy_intf_impl.h",
                 "kernel_struct_data_copy.h"}
        included = re.findall(r'^#\s+\d+\s+"([^"\n]+)"', preprocessed.read_text(errors="replace"), re.M)
        selected = {name for name in included if Path(name).name in names}
        selected.update(d["path"] for d in definitions if d["path"] != "<unknown>")
        for path in sorted({(cwd / name).resolve() for name in selected}):
            if not path.is_file():
                continue
            saved = output / "headers" / f"{len(headers)}_{path.name}"
            snapshot(path, saved, files)
            headers.append((path, saved))

    report = ["KVAF sequential scatter compiler diagnostics", "",
              "This is a collection report, not a hardware PASS.",
              "Original .o files are preserved. Source evidence does not map the fault PC.",
              "Device-side -S is unsupported by the observed CANN 9.1; assembly is not requested.", "",
              "COMPILE CWD: " + str(cwd), "COMPILE COMMAND: " + shlex.join(argv),
              "COMPILER RESOLVED: " + str(compiler.resolve()), "CACHE PATHS: " + json.dumps(paths, indent=2), "",
              "FILES: " + json.dumps(files, indent=2), "",
              (output / "compiler_version.txt").read_text(errors="replace")]
    for record in files:
        if "/versions/" in record.get("saved", ""):
            report.extend(["VERSION FILE: " + record["path"], Path(record["saved"]).read_text(errors="replace")])
    report.append(evidence)
    for name, code in codes.items():
        log = output / f"{name}.log"
        if code and log.is_file():
            report.extend(["FAILED STEP: " + name, log.read_text(errors="replace")])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    complete = all(codes.get(name) == 0 for name in ("compiler_version", "macros", "preprocess", "init_impl"))
    report.append("STEPS: " + json.dumps(codes, indent=2))
    report.append("COLLECTION_COMPLETE" if complete else "COLLECTION_PARTIAL: inspect step logs; originals retained")
    (output / "report.txt").write_text("\n\n".join(report) + "\n")
    return output, complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path, nargs="?")
    parser.add_argument("--analyze", type=Path, metavar="SAVED_DIRECTORY",
                        help="extract actual InitSocStateImpl from saved .ii without invoking any compiler")
    parser.add_argument("--kernel", type=Path, default=os.environ.get("KVAF_BATCH_GET_KERNEL"),
                        help="also preserve the configured kernel (default: KVAF_BATCH_GET_KERNEL)")
    args = parser.parse_args()
    if args.analyze and (args.build_dir or "--kernel" in sys.argv[1:]):
        parser.error("--analyze uses only saved evidence; do not specify build_dir or --kernel")
    if not args.analyze and not args.build_dir:
        parser.error("specify build_dir or --analyze SAVED_DIRECTORY")
    try:
        if args.analyze:
            report, complete = analyze_saved(args.analyze.resolve())
        else:
            output, complete = collect(args.build_dir.resolve(), args.kernel)
            report = output / "report.txt"
    except (OSError, ValueError) as error:
        print(f"COLLECT_FAILED: {error}", file=sys.stderr)
        return 2
    print(f"{'COLLECTED' if complete else 'PARTIAL'}: {report}", flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
