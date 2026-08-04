from __future__ import annotations

import hashlib
import os
import pathlib
import signal
import subprocess
import tempfile
import textwrap
import time
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
STATE = ROOT / "scripts/a5_runtime_probe_state.sh"
SUPERVISOR = ROOT / "scripts/a5_runtime_probe_supervisor.sh"
DURABLE = ROOT / "scripts/durable_state_file.py"


class RuntimeProbeStateTests(unittest.TestCase):
    def test_proc_parser_treats_zombie_and_complex_comm_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            proc_root = pathlib.Path(temporary)
            proc_entry = proc_root / "731"
            proc_entry.mkdir()
            fields_4_through_21 = " ".join(str(value) for value in range(1, 19))
            (proc_entry / "stat").write_text(
                f"731 (worker name) with close)) Z {fields_4_through_21} "
                "987654 20 21\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; '
                    'kvaf_runtime_probe_read_proc_state_and_start_ticks 731 "$2"',
                    "runtime-proc-test",
                    str(STATE),
                    str(proc_root),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.stdout.strip(), "Z 987654")

    def test_discover_worker_results_never_prove_safe_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            success = root / "discover-success.log"
            success.write_text(
                "KVAF_A5_RUNTIME_URMA_PROBE_RESULT mode=discover "
                "stage=done cleanup_proven=1 accepted=1 \n"
                "KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK\n"
                "KVAF_A5_RUNTIME_URMA_BASE_SYMBOL_SURFACE_PRESENT\n",
                encoding="utf-8",
            )
            safe_failure = root / "discover-safe-failure.log"
            safe_failure.write_text(
                "KVAF_A5_RUNTIME_URMA_PROBE_RESULT mode=discover "
                "stage=load cleanup_proven=1 accepted=0 \n"
                "KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK\n"
                "KVAF_A5_RUNTIME_URMA_PROBE_NON_ACCEPTED\n",
                encoding="utf-8",
            )
            unsafe = root / "discover-unsafe.log"
            unsafe.write_text(
                "KVAF_A5_RUNTIME_URMA_DISCOVER_CLEANUP_FAILED\n",
                encoding="utf-8",
            )
            script = (
                'source "$1"; '
                'kvaf_runtime_probe_log_is_safe_terminal "$2" "$3" discover'
            )
            self.assertNotEqual(
                subprocess.run(
                    ["bash", "-c", script, "discover-test", str(STATE), "0", str(success)]
                ).returncode,
                0,
            )
            self.assertNotEqual(
                subprocess.run(
                    [
                        "bash",
                        "-c",
                        script,
                        "discover-test",
                        str(STATE),
                        "3",
                        str(safe_failure),
                    ]
                ).returncode,
                0,
            )
            self.assertNotEqual(
                subprocess.run(
                    ["bash", "-c", script, "discover-test", str(STATE), "1", str(unsafe)]
                ).returncode,
                0,
            )
            prelaunch = root / "discover-prelaunch.log"
            prelaunch.write_text(
                "KVAF_A5_RUNTIME_URMA_PRELAUNCH_SAFE mode=discover\n",
                encoding="utf-8",
            )
            self.assertEqual(
                subprocess.run(
                    [
                        "bash",
                        "-c",
                        script,
                        "discover-test",
                        str(STATE),
                        "1",
                        str(prelaunch),
                    ]
                ).returncode,
                0,
            )

    def test_terminal_rejects_mode_and_log_digest_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            log = root / "probe.log"
            terminal = root / "probe.terminal"
            log.write_text(
                "KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK\n"
                "KVAF_A5_RUNTIME_ACCEPTED_MINIMAL_CREATE_DELETE\n"
                "KVAF_A5_RUNTIME_URMA_PROBE_RESULT mode=create-delete "
                "stage=done cleanup_proven=1 accepted=1 \n",
                encoding="utf-8",
            )
            log_sha = hashlib.sha256(log.read_bytes()).hexdigest()
            values = {
                "schema": "kvaf-a5-runtime-urma-probe-terminal-v1",
                "run_id": "runtime-urma-create-delete-20260803T120000Z-42-7",
                "mode": "create-delete",
                "boot_id": "01234567-89ab-cdef-0123-456789abcdef",
                "logical_device_id": "2",
                "physical_device_id": "2",
                "supervisor_pid": "4242",
                "supervisor_start_ticks": "9911",
                "worker_path": "/opt/kvaf/old-worker",
                "worker_sha256": "a" * 64,
                "provider_soname": "libkvaf_rt_" + "b" * 16 + ".so",
                "provider_build_id_sha256": "b" * 64,
                "expected_packaged_provider_artifact_sha256": "c" * 64,
                "log": str(log),
                "log_sha256": log_sha,
                "rc": "0",
                "reason": "worker_exited_log_synced",
                "terminal_contract": "runtime-probe-exact-terminal-v2",
                "lock_contract": (
                    "inherited-supervisor-worker-v4-held-flock-pinned-worker-fd"
                ),
                "caller_domain_contract": (
                    "undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1"
                ),
            }
            terminal.write_text(
                "".join(f"{key}={value}\n" for key, value in values.items()),
                encoding="utf-8",
            )

            def verify(expected_mode: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; '
                        "kvaf_runtime_probe_terminal_evidence_matches "
                        '"$2" "$3" "$4" "$5" 2 2 '
                        '"/opt/kvaf/old-worker" "$6" "$7" 4242 9911 '
                        '"$8" "$9" "${10}" && '
                        'kvaf_runtime_probe_log_is_safe_terminal 0 "$7" "$4"',
                        "runtime-terminal-test",
                        str(STATE),
                        str(terminal),
                        values["run_id"],
                        expected_mode,
                        values["boot_id"],
                        values["worker_sha256"],
                        str(log),
                        values["provider_soname"],
                        values["provider_build_id_sha256"],
                        values["expected_packaged_provider_artifact_sha256"],
                    ],
                    text=True,
                    capture_output=True,
                )

            self.assertEqual(verify("create-delete").returncode, 0)
            self.assertNotEqual(verify("init-device").returncode, 0)
            log.write_text(log.read_text(encoding="utf-8") + "tampered\n")
            self.assertNotEqual(verify("create-delete").returncode, 0)

            log.write_text(
                "KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK\n"
                "KVAF_A5_RUNTIME_URMA_BASE_SYMBOL_SURFACE_PRESENT\n"
                "KVAF_A5_RUNTIME_URMA_PROBE_RESULT mode=discover "
                "stage=done cleanup_proven=1 accepted=1 \n",
                encoding="utf-8",
            )
            values["run_id"] = "runtime-urma-discover-20260803T120001Z-42-8"
            values["mode"] = "discover"
            values["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
            terminal.write_text(
                "".join(f"{key}={value}\n" for key, value in values.items()),
                encoding="utf-8",
            )
            self.assertNotEqual(verify("discover").returncode, 0)
            self.assertNotEqual(verify("init-device").returncode, 0)

    @unittest.skipUnless(
        pathlib.Path("/proc/self/fd").exists(), "requires Linux procfs"
    )
    def test_supervisor_seals_ready_and_terminal_before_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            worker = root / "worker.sh"
            worker.write_text(
                "#!/usr/bin/env bash\n"
                "echo KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK\n"
                "echo KVAF_A5_RUNTIME_ACCEPTED_MINIMAL_CREATE_DELETE\n"
                "echo 'KVAF_A5_RUNTIME_URMA_PROBE_RESULT "
                "mode=create-delete stage=done cleanup_proven=1 accepted=1 '\n",
                encoding="utf-8",
            )
            worker.chmod(0o755)
            worker_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
            active_lock = root / "active.lock"
            deploy_lock = root / "deploy.lock"
            quarantine = root / "quarantine"
            active = root / "active"
            ready = root / "probe.ready"
            terminal = root / "probe.terminal"
            log = root / "probe.log"
            env = os.environ.copy()
            env.update(
                {
                    "KVAF_RUNTIME_SUPERVISOR_DURABLE_HELPER": str(DURABLE),
                    "KVAF_RUNTIME_SUPERVISOR_STATE_HELPER": str(STATE),
                    "KVAF_RUNTIME_SUPERVISOR_ACTIVE_LOCK": str(active_lock),
                    "KVAF_RUNTIME_SUPERVISOR_DEPLOY_LOCK": str(deploy_lock),
                    "KVAF_RUNTIME_SUPERVISOR_QUARANTINE": str(quarantine),
                    "KVAF_RUNTIME_SUPERVISOR_ACTIVE": str(active),
                    "KVAF_RUNTIME_SUPERVISOR_READY": str(ready),
                    "KVAF_RUNTIME_SUPERVISOR_TERMINAL": str(terminal),
                    "KVAF_RUNTIME_SUPERVISOR_LOG": str(log),
                    "KVAF_RUNTIME_SUPERVISOR_RUN_ID": (
                        "runtime-urma-create-delete-20260803T120000Z-42-7"
                    ),
                    "KVAF_RUNTIME_SUPERVISOR_MODE": "create-delete",
                    "KVAF_RUNTIME_SUPERVISOR_BOOT_ID": pathlib.Path(
                        "/proc/sys/kernel/random/boot_id"
                    ).read_text(encoding="utf-8").strip(),
                    "KVAF_RUNTIME_SUPERVISOR_LOGICAL_ID": "2",
                    "KVAF_RUNTIME_SUPERVISOR_PHYSICAL_ID": "2",
                    "KVAF_RUNTIME_SUPERVISOR_WORKER_PATH": str(worker.resolve()),
                    "KVAF_RUNTIME_SUPERVISOR_WORKER_SHA256": worker_sha,
                    "KVAF_RUNTIME_SUPERVISOR_PROVIDER_SONAME": (
                        "libkvaf_rt_1111111111111111.so"
                    ),
                    "KVAF_RUNTIME_SUPERVISOR_PROVIDER_BUILD_ID": "1" * 64,
                    "KVAF_RUNTIME_SUPERVISOR_PROVIDER_ARTIFACT_SHA256": (
                        "2" * 64
                    ),
                    "KVAF_RUNTIME_SUPERVISOR_CALLER_DOMAIN_CONTRACT": (
                        "undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1"
                    ),
                }
            )
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    'exec 8>"$1"; flock -n 8; '
                    'exec 9>"$2"; flock -n 9; exec "$3" '
                    "--provider-soname libkvaf_rt_1111111111111111.so "
                    "--provider-build-id " + "1" * 64,
                    "runtime-supervisor-test",
                    str(active_lock),
                    str(deploy_lock),
                    str(SUPERVISOR),
                ],
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(quarantine.is_file())
            self.assertTrue(active.is_file())
            self.assertTrue(ready.is_file())
            self.assertTrue(terminal.is_file())
            for state_file in (quarantine, active, ready, terminal):
                keys = [
                    line.split("=", 1)[0]
                    for line in state_file.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertEqual(
                    len(keys), len(set(keys)),
                    f"duplicate durable-state key in {state_file}",
                )
            quarantine_values = dict(
                line.split("=", 1)
                for line in quarantine.read_text(
                    encoding="utf-8"
                ).splitlines()
            )
            active_values = dict(
                line.split("=", 1)
                for line in active.read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(
                quarantine_values["reason"],
                "runtime_public_probe_supervisor_prelaunch",
            )
            self.assertEqual(
                active_values["reason"],
                "supervisor_owns_locks_and_worker_lifetime",
            )
            terminal_values = dict(
                line.split("=", 1)
                for line in terminal.read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(terminal_values["mode"], "create-delete")
            self.assertEqual(terminal_values["rc"], "0")
            self.assertEqual(
                terminal_values["log_sha256"],
                hashlib.sha256(log.read_bytes()).hexdigest(),
            )

    @unittest.skipUnless(
        pathlib.Path("/proc/self/fd").exists(), "requires Linux procfs"
    )
    def test_supervisor_executes_verified_inode_after_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            worker = root / "worker.sh"
            replacement = root / "replacement.sh"
            common = (
                "echo KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK\n"
                "echo KVAF_A5_RUNTIME_ACCEPTED_MINIMAL_CREATE_DELETE\n"
                "echo 'KVAF_A5_RUNTIME_URMA_PROBE_RESULT "
                "mode=create-delete stage=done cleanup_proven=1 accepted=1 '\n"
            )
            worker.write_text(
                "#!/usr/bin/env bash\necho WORKER_ID=verified\n" + common,
                encoding="utf-8",
            )
            replacement.write_text(
                "#!/usr/bin/env bash\necho WORKER_ID=replaced\n" + common,
                encoding="utf-8",
            )
            worker.chmod(0o755)
            replacement.chmod(0o755)
            worker_sha = hashlib.sha256(worker.read_bytes()).hexdigest()

            active_lock = root / "active.lock"
            deploy_lock = root / "deploy.lock"
            quarantine = root / "quarantine"
            active = root / "active"
            ready = root / "probe.ready"
            terminal = root / "probe.terminal"
            log = root / "probe.log"
            published = root / "ready.published"
            release = root / "ready.release"
            helper = root / "blocking_durable_helper.py"
            helper.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env python3
                    import pathlib
                    import subprocess
                    import sys
                    import time

                    subprocess.check_call([
                        sys.executable, {str(DURABLE)!r}, *sys.argv[1:]
                    ])
                    if (len(sys.argv) == 4 and sys.argv[1] == "publish" and
                            pathlib.Path(sys.argv[3]) == pathlib.Path({str(ready)!r})):
                        pathlib.Path({str(published)!r}).write_text("ready\\n")
                        while not pathlib.Path({str(release)!r}).exists():
                            time.sleep(0.01)
                    """
                ),
                encoding="utf-8",
            )
            helper.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "KVAF_RUNTIME_SUPERVISOR_DURABLE_HELPER": str(helper),
                    "KVAF_RUNTIME_SUPERVISOR_STATE_HELPER": str(STATE),
                    "KVAF_RUNTIME_SUPERVISOR_ACTIVE_LOCK": str(active_lock),
                    "KVAF_RUNTIME_SUPERVISOR_DEPLOY_LOCK": str(deploy_lock),
                    "KVAF_RUNTIME_SUPERVISOR_QUARANTINE": str(quarantine),
                    "KVAF_RUNTIME_SUPERVISOR_ACTIVE": str(active),
                    "KVAF_RUNTIME_SUPERVISOR_READY": str(ready),
                    "KVAF_RUNTIME_SUPERVISOR_TERMINAL": str(terminal),
                    "KVAF_RUNTIME_SUPERVISOR_LOG": str(log),
                    "KVAF_RUNTIME_SUPERVISOR_RUN_ID": (
                        "runtime-urma-create-delete-20260803T120100Z-42-8"
                    ),
                    "KVAF_RUNTIME_SUPERVISOR_MODE": "create-delete",
                    "KVAF_RUNTIME_SUPERVISOR_BOOT_ID": pathlib.Path(
                        "/proc/sys/kernel/random/boot_id"
                    ).read_text(encoding="utf-8").strip(),
                    "KVAF_RUNTIME_SUPERVISOR_LOGICAL_ID": "2",
                    "KVAF_RUNTIME_SUPERVISOR_PHYSICAL_ID": "2",
                    "KVAF_RUNTIME_SUPERVISOR_WORKER_PATH": str(worker.resolve()),
                    "KVAF_RUNTIME_SUPERVISOR_WORKER_SHA256": worker_sha,
                    "KVAF_RUNTIME_SUPERVISOR_PROVIDER_SONAME": (
                        "libkvaf_rt_1111111111111111.so"
                    ),
                    "KVAF_RUNTIME_SUPERVISOR_PROVIDER_BUILD_ID": "1" * 64,
                    "KVAF_RUNTIME_SUPERVISOR_PROVIDER_ARTIFACT_SHA256": "2" * 64,
                    "KVAF_RUNTIME_SUPERVISOR_CALLER_DOMAIN_CONTRACT": (
                        "undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1"
                    ),
                }
            )
            process = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    'exec 8>"$1"; flock -n 8; '
                    'exec 9>"$2"; flock -n 9; exec "$3" '
                    "--provider-soname libkvaf_rt_1111111111111111.so "
                    "--provider-build-id " + "1" * 64,
                    "runtime-supervisor-pinned-inode-test",
                    str(active_lock),
                    str(deploy_lock),
                    str(SUPERVISOR),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            stdout = ""
            stderr = ""
            communicated = False
            try:
                deadline = time.monotonic() + 10
                while not published.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    time.sleep(0.01)
                if not published.exists():
                    stdout, stderr = process.communicate(timeout=2)
                    communicated = True
                    self.fail(
                        "supervisor did not reach ready\n"
                        f"returncode={process.returncode}\n"
                        f"stdout:\n{stdout}\n"
                        f"stderr:\n{stderr}"
                    )
                os.replace(replacement, worker)
                release.write_text("continue\n", encoding="utf-8")
                stdout, stderr = process.communicate(timeout=15)
                communicated = True
                self.assertEqual(process.returncode, 0, stdout + stderr)
                self.assertIn(
                    "WORKER_ID=verified", log.read_text(encoding="utf-8")
                )
                self.assertNotIn(
                    "WORKER_ID=replaced", log.read_text(encoding="utf-8")
                )
            finally:
                # A failed assertion must not leave the supervisor or its
                # blocking durable-helper child alive, and communicate() must
                # always close both captured pipes.
                release.touch(exist_ok=True)
                if not communicated:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    try:
                        process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.communicate(timeout=2)


if __name__ == "__main__":
    unittest.main()
