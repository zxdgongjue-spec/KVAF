import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABI = ROOT / "access_operator/a5/provider/include/kvaf_urma_provider_abi.h"
PROVIDER = ROOT / "access_operator/a5/provider/urma_perftest_provider.c"
RUNTIME_PROBE = (
    ROOT / "access_operator/a5/provider/runtime_matched_urma_probe.c"
)
CONTRACT_PROBE = (
    ROOT / "access_operator/a5/provider/provider_aicpu_contract_probe.c"
)
BUILD = ROOT / "scripts/build_a5_direct_urma_provider.sh"
MANIFEST = ROOT / "scripts/a5_direct_urma_manifest_v3.sh"


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def function_body(text: str, signature: str) -> str:
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise AssertionError(f"unterminated function: {signature}")


class ProviderCapabilityLedgerGateTests(unittest.TestCase):
    def test_status_abi_is_append_only_and_explicit(self) -> None:
        abi = source(ABI)
        self.assertIn(
            "KVAF_URMA_PROVIDER_CAPABILITY_LEDGER_UNAVAILABLE = 10", abi
        )
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_RESULT_CAPABILITY_LEDGER_UNAVAILABLE = 8",
            abi,
        )
        self.assertIn(
            "KVAF_URMA_RUNTIME_PROBE_OUTCOME_CAPABILITY_LEDGER_UNAVAILABLE = 14",
            abi,
        )
        self.assertIn(
            "KVAF_URMA_OWNER_PROBE_RESULT_CAPABILITY_LEDGER_UNAVAILABLE = 4",
            abi,
        )

    def test_provider_open_stops_before_validation_lock_and_resources(self) -> None:
        body = function_body(source(PROVIDER), "int kvaf_urma_provider_open_v2(")
        gate = body.index("KVAF_URMA_PROVIDER_CAPABILITY_LEDGER_UNAVAILABLE")
        gate_return = body.index("return -1;", gate)
        config_validation = body.index("kvaf_config_is_valid(config)")
        lock = body.index("pthread_mutex_lock(&g_kvaf_lock)")
        resource_open = body.index("kvaf_parse_config(config)")

        self.assertLess(gate, gate_return)
        self.assertLess(gate_return, config_validation)
        self.assertLess(config_validation, lock)
        self.assertLess(lock, resource_open)
        self.assertNotIn("getenv", body[:gate_return])
        self.assertNotIn("config->", body[:gate_return])

    def test_epoch_owner_publishes_safe_terminal_before_claim(self) -> None:
        body = function_body(
            source(PROVIDER), "int aicpu_kvaf_urma_epoch_owner_v1("
        )
        gate = body.index("KVAF_URMA_PROVIDER_CAPABILITY_LEDGER_UNAVAILABLE")
        terminal = body.index(
            "KVAF_URMA_EPOCH_OWNER_RESULT_CAPABILITY_LEDGER_UNAVAILABLE",
            gate,
        )
        gate_return = body.index("return 0;", terminal)
        clock = body.index("CLOCK_MONOTONIC", gate_return)
        claim = body.index("kvaf_epoch_owner_acquire_claim(args)", gate_return)
        provider_open = body.index(
            "open_status = kvaf_urma_provider_open_v2(", gate_return
        )

        self.assertLess(gate, terminal)
        self.assertLess(terminal, gate_return)
        self.assertLess(gate_return, clock)
        self.assertLess(gate_return, claim)
        self.assertLess(gate_return, provider_open)
        gate_block = body[gate:gate_return]
        self.assertIn("KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID", gate_block)
        self.assertIn("KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE", gate_block)
        self.assertNotIn("KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE", gate_block)
        self.assertNotIn("getenv", body[:gate_return])

    def test_owner_probe_publishes_terminal_before_command_clock_or_sleep(
        self,
    ) -> None:
        body = function_body(
            source(PROVIDER), "int aicpu_kvaf_urma_owner_probe_v1("
        )
        gate = body.index(
            "KVAF_URMA_OWNER_PROBE_RESULT_CAPABILITY_LEDGER_UNAVAILABLE"
        )
        terminal = body.index(
            "KVAF_URMA_OWNER_PROBE_STATE_FAILED", gate
        )
        gate_return = body.index("return 0;", terminal)
        command_pointer = body.index("command =", gate_return)
        command_read = body.index("__atomic_load_n(&command->opcode", gate_return)
        clock = body.index("CLOCK_MONOTONIC", gate_return)
        sleep = body.index("kvaf_owner_probe_sleep", gate_return)

        self.assertLess(gate, terminal)
        for operation in (command_pointer, command_read, clock, sleep):
            self.assertLess(gate_return, operation)
        gate_block = body[gate:gate_return]
        self.assertIn("kvaf_owner_probe_publish", gate_block)
        self.assertIn(
            "KVAF_URMA_OWNER_PROBE_RESULT_CAPABILITY_LEDGER_UNAVAILABLE",
            gate_block,
        )
        self.assertNotIn("getenv", body[:gate_return])

    def test_runtime_probe_stops_before_loader_and_official_api(self) -> None:
        body = function_body(
            source(RUNTIME_PROBE), "int aicpu_kvaf_urma_rt_probe_v2("
        )
        identity = body.index("expected_provider_build_id_sha256")
        gate = body.index(
            "KVAF_URMA_RUNTIME_PROBE_OUTCOME_CAPABILITY_LEDGER_UNAVAILABLE"
        )
        gate_return = body.index("return 0;", gate)
        lock = body.index("pthread_mutex_trylock", gate_return)
        loader = body.index("dlopen(", gate_return)
        official_init = body.index("status = api.init(NULL)", gate_return)

        self.assertLess(identity, gate)
        self.assertLess(gate, gate_return)
        self.assertLess(gate_return, lock)
        self.assertLess(gate_return, loader)
        self.assertLess(gate_return, official_init)
        self.assertNotIn("getenv", body[:gate_return])

    def test_behavior_probe_uses_valid_requests_and_checks_no_state(self) -> None:
        contract = source(CONTRACT_PROBE)
        self.assertIn("initialize_valid_device_config", contract)
        self.assertGreaterEqual(
            contract.count("KVAF_URMA_PROVIDER_CAPABILITY_LEDGER_UNAVAILABLE"),
            2,
        )
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_RESULT_CAPABILITY_LEDGER_UNAVAILABLE",
            contract,
        )
        self.assertIn(
            "KVAF_URMA_OWNER_PROBE_RESULT_CAPABILITY_LEDGER_UNAVAILABLE",
            contract,
        )
        self.assertIn(
            "KVAF_URMA_RUNTIME_PROBE_OUTCOME_CAPABILITY_LEDGER_UNAVAILABLE",
            contract,
        )
        self.assertIn("compiled_runtime_probe_build_id", contract)
        self.assertIn("capability-gated Open unexpectedly left provider state", contract)
        self.assertIn(
            "capability-gated epoch owner unexpectedly left provider state",
            contract,
        )

    def test_build_and_verifier_bind_and_reexecute_gate_proof(self) -> None:
        build = source(BUILD)
        manifest = source(MANIFEST)
        exact_values = {
            "resource_entry_gate_schema": "capability-ledger-fail-closed-v1",
            "capability_ledger_state": "not-implemented",
            "provider_open_gate": "disabled-before-config-validation-v1",
            "owner_probe_gate": "disabled-before-command-clock-sleep-v1",
            "epoch_owner_gate": "disabled-before-claim-v1",
            "runtime_probe_gate": "disabled-before-dlopen-v1",
            "resource_gate_contract": "verified-valid-request-v1",
        }
        for key, value in exact_values.items():
            self.assertGreaterEqual(build.count(f"{key}={value}"), 2)
            self.assertIn(f"'{key}:{value}'", manifest)

        self.assertIn(
            "resource_gate_contract_probe_filename=kvaf_urma_provider_aicpu_contract_probe",
            build,
        )
        self.assertIn("resource_gate_contract_probe_sha256", build)
        self.assertIn("resource_gate_contract_output_sha256", build)
        self.assertIn('cp "${aicpu_contract_probe}"', build)
        self.assertIn('"${resource_gate_probe_path}"', manifest)
        source_check = manifest.index("sha256sum --check --strict")
        proof_run = manifest.index(
            'resource_gate_output=$(\n    "${resource_gate_probe_path}"'
        )
        self.assertLess(source_check, proof_run)
        self.assertRegex(
            manifest[proof_run:],
            re.compile(
                r"KVAF_URMA_PROVIDER_AICPU_CONTRACT=PASS.*?"
                r"resource_gate_output_actual_sha",
                re.DOTALL,
            ),
        )

    def test_exported_official_symbols_are_not_treated_as_capability(self) -> None:
        abi = source(ABI)
        runtime = source(RUNTIME_PROBE)
        self.assertIn("official implementations", abi)
        self.assertIn(
            "can export stubs, reject modes, or contain lifecycle defects",
            abi,
        )
        gate = runtime.index(
            "KVAF_URMA_RUNTIME_PROBE_OUTCOME_CAPABILITY_LEDGER_UNAVAILABLE"
        )
        loader = runtime.index("dlopen(", gate)
        self.assertLess(gate, loader)
        self.assertIn("KVAF_URMA_RUNTIME_PROBE_PUBLIC_ABI_UNPROVEN", runtime)


if __name__ == "__main__":
    unittest.main()
