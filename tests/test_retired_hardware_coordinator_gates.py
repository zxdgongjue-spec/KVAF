#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RetiredHardwareCoordinatorGateTests(unittest.TestCase):
    COORDINATORS = (
        (
            "detached-cross-npu2",
            ROOT / "scripts/launch_a5_cross_npu2_reuse_detached.sh",
            "KVAF_A5_CROSS_NPU2_REUSE_DETACHED_COORDINATOR_DISABLED",
        ),
        (
            "cross-npu2-experiment",
            ROOT / "scripts/run_a5_cross_npu2_reuse_experiment.sh",
            "KVAF_A5_CROSS_NPU2_REUSE_EXPERIMENT_COORDINATOR_DISABLED",
        ),
        (
            "cross-npu2-risk-accepted",
            ROOT / "scripts/run_a5_direct_urma_cross_npu2_risk_accepted.sh",
            "KVAF_A5_CROSS_NPU2_RISK_ACCEPTED_COORDINATOR_DISABLED",
        ),
        (
            "npu2-clean-prepare",
            ROOT / "scripts/prepare_and_run_a5_direct_urma_npu2_clean.sh",
            "KVAF_A5_NPU2_CLEAN_PREPARE_COORDINATOR_DISABLED",
        ),
        (
            "npu2-clean-validation",
            ROOT / "scripts/run_a5_direct_urma_npu2_clean_validation.sh",
            "KVAF_A5_NPU2_CLEAN_VALIDATION_COORDINATOR_DISABLED",
        ),
        (
            "epoch-owner-validation",
            ROOT / "scripts/run_a5_direct_urma_epoch_owner_validation.sh",
            "KVAF_A5_EPOCH_OWNER_VALIDATION_COORDINATOR_DISABLED",
        ),
    )

    def test_gate_precedes_arguments_environment_source_state_and_hardware(self) -> None:
        forbidden_before_gate = (
            "SCRIPT_DIR=",
            "ROOT_DIR=",
            "\nsource ",
            "\ndocker ",
            "npu-smi",
            "/proc/asdrv_ub",
            "ASCEND_HOME_PATH",
            "KVAF_STATE_ROOT=",
            "KVAF_GLOBAL_STATE_DIR=",
            "\nmkdir ",
            "\nmktemp ",
            "\nnohup ",
            "\nexec bash",
            "$#",
            "${1",
        )
        for label, path, marker in self.COORDINATORS:
            with self.subTest(coordinator=label):
                source = path.read_text(encoding="utf-8")
                clear_env = source.index("unset BASH_ENV ENV")
                clear_functions = source.index(
                    "unset -f exit builtin 2>/dev/null || true"
                )
                gate = source.index(marker)
                fixed_exit = source.index("builtin exit 2", gate)
                false_branch = source.index("if [[ 0 == 1 ]]; then", fixed_exit)

                self.assertEqual(
                    [clear_env, clear_functions, gate, fixed_exit, false_branch],
                    sorted(
                        [clear_env, clear_functions, gate, fixed_exit, false_branch]
                    ),
                )
                pre_gate = source[:fixed_exit]
                for forbidden in forbidden_before_gate:
                    self.assertNotIn(
                        forbidden,
                        pre_gate,
                        f"{forbidden!r} ran or was inspected before the gate",
                    )
                self.assertNotIn("${KVAF_", source[gate:fixed_exit])
                self.assertTrue(
                    source.rstrip().endswith("fi\n/bin/sh -c 'exit 2'"),
                    "the constant-false branch must contain the complete retired body",
                )

    def test_exit_and_builtin_shadow_cannot_reach_retired_body(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")

        for label, path, marker in self.COORDINATORS:
            with self.subTest(coordinator=label), tempfile.TemporaryDirectory() as temporary:
                temporary_path = Path(temporary)
                bash_env = temporary_path / "bash-env"
                hook_loaded = temporary_path / "hook-loaded"
                shadow_called = temporary_path / "shadow-called"
                state_root = temporary_path / "must-not-be-created"
                bash_env.write_text(
                    'printf loaded >"${KVAF_HOOK_LOADED}"\n'
                    'exit() { printf exit >"${KVAF_SHADOW_CALLED}"; return 0; }\n'
                    'builtin() { printf builtin >"${KVAF_SHADOW_CALLED}"; return 0; }\n',
                    encoding="utf-8",
                )
                environment = os.environ.copy()
                environment.pop("ASCEND_HOME_PATH", None)
                environment.update(
                    {
                        "BASH_ENV": str(bash_env),
                        "ENV": str(bash_env),
                        "KVAF_HOOK_LOADED": str(hook_loaded),
                        "KVAF_SHADOW_CALLED": str(shadow_called),
                        "KVAF_STATE_ROOT": str(state_root),
                        "KVAF_GLOBAL_STATE_DIR": str(state_root),
                        "KVAF_A5_RESULT_ROOT": str(state_root),
                        "KVAF_A5_DIRECT_BUILD_DIR": str(state_root),
                        "KVAF_DIRECT_PROVIDER_BUILD_DIR": str(state_root),
                    }
                )
                result = subprocess.run(
                    [
                        bash,
                        "--noprofile",
                        "--norc",
                        str(path),
                        "unexpected-argument",
                    ],
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(marker, result.stderr)
                self.assertIn("CAPABILITY_LEDGER=not-implemented", result.stderr)
                self.assertTrue(hook_loaded.is_file(), "BASH_ENV was not exercised")
                self.assertFalse(
                    shadow_called.exists(),
                    "an imported exit/builtin function bypassed the fixed exit",
                )
                self.assertFalse(
                    state_root.exists(),
                    "a retired coordinator created build/state/result artifacts",
                )

    def test_recovery_entry_was_not_retired(self) -> None:
        recovery = ROOT / "scripts/recover_a5_direct_urma_state.sh"
        source = recovery.read_text(encoding="utf-8")
        self.assertNotIn("COORDINATOR_DISABLED", source)
        self.assertIn("recovery-only-v1", source)


if __name__ == "__main__":
    unittest.main()
