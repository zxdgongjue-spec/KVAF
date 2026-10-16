"""Execute the complete current Put kernel using a deterministic CPU transport."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "access_operator/demo/aiv_batch_put"


def test_actual_put_kernel_first_seed_24_workers_wrap_replay_and_poison(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    model = Path(__file__).with_name("put_device_cpu_model.cpp").read_text()
    kernel = (DEMO / "kernel/execute_batch_put_aiv.asc").read_text()
    kernel = kernel[: kernel.index("static const struct FunLevelKType")]
    source = tmp_path / "actual_put.cpp"
    source.write_text(model.replace("// ACTUAL_PRODUCTION_KERNEL", kernel))
    (tmp_path / "simt_api").mkdir()
    (tmp_path / "kernel_operator.h").write_text("")
    (tmp_path / "simt_api/device_atomic_functions.h").write_text("")
    executable = tmp_path / "actual_put"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I",
            str(tmp_path),
            "-I",
            str(DEMO / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
    (tmp_path / "actual_put.stdout.log").write_text(result.stdout)
    assert "BATCH_PUT_ACTUAL_KERNEL_CPU_PASS scenarios=11" in result.stdout
    assert not result.stderr
