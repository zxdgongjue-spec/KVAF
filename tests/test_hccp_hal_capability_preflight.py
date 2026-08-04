from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts" / "preflight_a5_hccp_hal_capabilities.sh"
PROBE = ROOT / "scripts" / "probes" / "a5_hccp_hal_capability_probe.c"


COMPLETE_HEADER = r"""
#include <stdint.h>
#include <sys/types.h>
typedef int drvError_t;
typedef int DVresult;
enum { DEVDRV_PROCESS_CP1 = 0, PROCESS_CP1 = 0 };
#define SVM_MEM_BIND_SP_GRP (1U << 1)
struct halQueryDevpidInfo {
    int hostpid;
    unsigned int devid;
    unsigned int proc_type;
};
struct res_map_info_in {
    unsigned int res_id;
    unsigned int target_proc_type;
    unsigned int res_type;
    unsigned int priv_len;
    void *priv;
};
struct res_map_info_out { unsigned long va; };
enum res_addr_type {
    RES_ADDR_TYPE_HCCP_URMA_JFC = 101,
    RES_ADDR_TYPE_HCCP_URMA_JETTY = 102,
    RES_ADDR_TYPE_HCCP_URMA_DB = 103
};
drvError_t drvDeviceGetIndexByPhyId(uint32_t, uint32_t *);
drvError_t halQueryDevpid(struct halQueryDevpidInfo, pid_t *);
DVresult halMemBindSibling(int, int, unsigned int, unsigned int,
                           unsigned int);
drvError_t halResAddrMapV2(unsigned int, struct res_map_info_in *,
                          struct res_map_info_out *);
drvError_t halResAddrUnmapV2(unsigned int, struct res_map_info_in *);
"""


ALL_SYMBOLS = (
    "drvDeviceGetIndexByPhyId",
    "halQueryDevpid",
    "halMemBindSibling",
    "halResAddrMapV2",
    "halResAddrUnmapV2",
    "RaCtxInit",
    "RaCtxDeinit",
    "RaCtxLmemRegister",
    "RaCtxLmemUnregister",
    "RaCtxRmemImport",
    "RaCtxRmemUnimport",
    "RaCtxChanCreate",
    "RaCtxChanDestroy",
    "RaCtxCqCreate",
    "RaCtxCqDestroy",
    "RaCtxQpCreate",
    "RaCtxQpDestroy",
    "RaCtxQpImport",
    "RaCtxQpUnimport",
    "RaCtxQpBind",
    "RaCtxQpUnbind",
    "RaCtxGetTpInfoList",
    "RaCtxGetTpAttr",
    "RaCtxSetTpAttr",
    "RaInit",
    "RaRdevGetHandle",
    "RaRdevInitV2",
    "RaAiQpCreate",
    "RaQpDestroy",
    "RaRegisterMr",
    "RaDeregisterMr",
    "RaMrReg",
    "RaQpConnectAsync",
    "RaGetQpStatus",
)


class HccpHalCapabilityPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "driver"
        self.header = self.root / "include" / "driver" / "ascend_hal.h"
        self.library = self.root / "lib64" / "driver" / "libascend_hal.so"
        self.ra_library = self.root / "lib64" / "common" / "libra.so"
        self.bin_dir = Path(self.tempdir.name) / "bin"
        self.header.parent.mkdir(parents=True)
        self.library.parent.mkdir(parents=True)
        self.ra_library.parent.mkdir(parents=True)
        self.bin_dir.mkdir()
        self.header.write_text(COMPLETE_HEADER, encoding="utf-8")
        self.library.write_bytes(b"not loaded by the preflight\n")
        self.ra_library.write_bytes(b"not loaded by the preflight\n")
        self.nm = self.bin_dir / "kvaf-nm"
        self.write_nm(ALL_SYMBOLS)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_nm(self, symbols: tuple[str, ...]) -> None:
        body = "\n".join(f"0000000000001000 T {name}" for name in symbols)
        self.nm.write_text(
            "#!/bin/sh\ncat <<'KVAF_NM'\n" + body + "\nKVAF_NM\n",
            encoding="utf-8",
        )
        self.nm.chmod(self.nm.stat().st_mode | stat.S_IXUSR)

    def run_preflight(self) -> subprocess.CompletedProcess[str]:
        cc = shutil.which("cc")
        if cc is None:
            self.skipTest("C compiler is unavailable")
        env = os.environ.copy()
        env.update(
            {
                "KVAF_ASCEND_DRIVER_ROOT": str(self.root),
                "KVAF_HCCP_HAL_CC": cc,
                "KVAF_HCCP_HAL_NM": str(self.nm),
            }
        )
        return subprocess.run(
            ["bash", str(PREFLIGHT)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_shell_syntax_and_probe_are_fail_closed(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(PREFLIGHT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        source = PREFLIGHT.read_text(encoding="utf-8")
        probe = PROBE.read_text(encoding="utf-8")
        for forbidden in (
            "npu-smi",
            "/dev/davinci",
            "/dev/uburma",
            "dlopen",
            "halQueryDevpid(",
            "halMemBindSibling(",
            "halResAddrMapV2(",
            "halResAddrUnmapV2(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("-fsyntax-only", source)
        self.assertIn("--defined-only", source)
        self.assertIn("unproven-not-called", source)
        self.assertIn("RaCtxQpCreate", source)
        self.assertIn("RaAiQpCreate", source)
        self.assertIn("host-owned-libra-init-aiv-direct-data-plane", source)
        self.assertIn("KVAF_HAL_PROBE_BASELINE", source)
        self.assertIn("UdmaVaInfo_private_payload", source)
        self.assertIn("hccp_hal_private_abi.h", probe)
        self.assertNotIn("#define RES_ADDR_TYPE_HCCP_URMA", probe)
        self.assertNotIn("struct res_map_info_in {", probe)
        self.assertNotIn("struct res_map_info_out {", probe)

    def test_complete_installed_surface_passes_without_calling_hal(self) -> None:
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "KVAF_HCCP_HAL_SURFACE=declarations-complete", result.stdout
        )
        self.assertIn("KVAF_HCCP_HAL_MISSING_COUNT=0", result.stdout)
        self.assertIn(
            "KVAF_HCCP_HAL_RUNTIME_SEMANTICS=unproven-not-called",
            result.stdout,
        )
        self.assertIn(
            "KVAF_A5_HCCP_HAL_DECLARATION_PREFLIGHT_OK", result.stdout
        )
        self.assertNotIn(
            "KVAF_A5_HCCP_HAL_CAPABILITY_PREFLIGHT_OK", result.stdout
        )
        self.assertIn("KVAF_HCCP_BACKEND_READY=no", result.stdout)
        self.assertIn(
            "KVAF_HCCP_PRIVATE_LAYOUT="
            "compile-time-size-and-offsets-verified-runtime-unproven",
            result.stdout,
        )
        self.assertIn(
            "KVAF_HCCP_PRIVATE_PAYLOAD_ABI="
            "source-audited-hcomm-v1-runtime-unproven",
            result.stdout,
        )
        self.assertIn("KVAF_HCCP_PRIVATE_PAYLOAD_PUBLIC_API=no", result.stdout)
        self.assertIn(
            "KVAF_HCCP_CONTROL_PLANE_CANDIDATE=libra-ra-ctx",
            result.stdout,
        )
        self.assertIn(
            "KVAF_A5_HCCP_CONTROL_PLANE_STATIC_AUDIT_OK", result.stdout
        )
        self.assertIn(
            "KVAF_HCCP_RA_PRODUCT_MODE_SUPPORT="
            "unproven-user-ctl-and-custom-sq-must-be-probed",
            result.stdout,
        )

    def test_missing_private_hal_symbol_is_reported_but_ra_audit_completes(self) -> None:
        self.write_nm(tuple(s for s in ALL_SYMBOLS if s != "halResAddrMapV2"))
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "KVAF_HCCP_HAL_LIBRARY_SYMBOL_halResAddrMapV2=missing",
            result.stdout,
        )
        self.assertIn("KVAF_HCCP_HAL_SURFACE=incomplete", result.stdout)
        self.assertNotIn("HAL_DECLARATION_PREFLIGHT_OK", result.stdout)
        self.assertIn("KVAF_HCCP_BACKEND_READY=no", result.stdout)
        self.assertIn(
            "KVAF_HCCP_CONTROL_PLANE_CANDIDATE=libra-ra-ctx",
            result.stdout,
        )

    def test_missing_official_definition_fails_without_fallback_value(self) -> None:
        self.header.write_text(
            COMPLETE_HEADER.replace(
                "    RES_ADDR_TYPE_HCCP_URMA_JETTY = 102,\n", ""
            ),
            encoding="utf-8",
        )
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "KVAF_HCCP_HAL_HEADER_CAPABILITY_"
            "RES_ADDR_TYPE_HCCP_URMA_JETTY=missing",
            result.stdout,
        )
        self.assertIn("KVAF_HCCP_HAL_SURFACE=incomplete", result.stdout)
        self.assertIn(
            "KVAF_HCCP_HAL_HEADER_DIAGNOSTIC_"
            "RES_ADDR_TYPE_HCCP_URMA_JETTY_BEGIN",
            result.stderr,
        )

    def test_missing_required_struct_member_fails_closed(self) -> None:
        self.header.write_text(
            COMPLETE_HEADER.replace("    unsigned int priv_len;\n", ""),
            encoding="utf-8",
        )
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "KVAF_HCCP_HAL_HEADER_CAPABILITY_res_map_info_in=missing",
            result.stdout,
        )

    def test_private_payload_is_not_required_from_installed_header(self) -> None:
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("KVAF_HCCP_PRIVATE_PAYLOAD_PUBLIC_API=no", result.stdout)
        self.assertIn("source-audited-hcomm-v1", result.stdout)

    def test_incompatible_public_enum_width_rejects_private_layout(self) -> None:
        self.header.write_text(
            COMPLETE_HEADER.replace(
                "enum res_addr_type {\n"
                "    RES_ADDR_TYPE_HCCP_URMA_JFC = 101,",
                "enum __attribute__((packed)) res_addr_type {\n"
                "    RES_ADDR_TYPE_HCCP_URMA_JFC = 101,",
            ),
            encoding="utf-8",
        )
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "KVAF_HCCP_HAL_HEADER_CAPABILITY_"
            "UdmaVaInfo_private_payload=missing",
            result.stdout,
        )
        self.assertIn(
            "KVAF_HCCP_PRIVATE_PAYLOAD_ABI="
            "not-selected-for-primary-backend",
            result.stdout,
        )

    def test_runtime_image_without_hal_header_still_audits_libra(self) -> None:
        self.header.unlink()
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("KVAF_HCCP_HAL_HEADER=not-found", result.stdout)
        self.assertIn(
            "KVAF_HCCP_HAL_HEADER_SURFACE="
            "unavailable-runtime-image-has-no-sdk-header",
            result.stdout,
        )
        self.assertIn(
            "KVAF_HCCP_CONTROL_PLANE_CANDIDATE=libra-ra-ctx",
            result.stdout,
        )
        self.assertIn("KVAF_HCCP_BACKEND_READY=no", result.stdout)

    def test_incomplete_ra_surface_is_inventory_not_hardware_authorization(self) -> None:
        self.write_nm(tuple(s for s in ALL_SYMBOLS if not s.startswith("RaCtx")))
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("KVAF_HCCP_RA_CTX_SURFACE=incomplete", result.stdout)
        self.assertIn(
            "KVAF_HCCP_CONTROL_PLANE_CANDIDATE=libra-legacy-ai-qp",
            result.stdout,
        )
        self.assertIn("KVAF_HCCP_BACKEND_READY=no", result.stdout)

    def test_unusable_header_is_reported_as_baseline_failure(self) -> None:
        self.header.write_text("#error broken installed header\n", encoding="utf-8")
        result = self.run_preflight()
        self.assertEqual(result.returncode, 2)
        self.assertIn("KVAF_HCCP_HAL_HEADER_BASELINE=unusable", result.stdout)
        self.assertIn("cannot compile in isolation", result.stderr)

    def test_extra_include_outside_driver_root_is_rejected(self) -> None:
        external = Path(self.tempdir.name) / "external-include"
        external.mkdir()
        cc = shutil.which("cc")
        if cc is None:
            self.skipTest("C compiler is unavailable")
        env = os.environ.copy()
        env.update(
            {
                "KVAF_ASCEND_DRIVER_ROOT": str(self.root),
                "KVAF_HCCP_HAL_CC": cc,
                "KVAF_HCCP_HAL_NM": str(self.nm),
                "KVAF_HCCP_HAL_INCLUDE_DIRS": str(external),
            }
        )
        result = subprocess.run(
            ["bash", str(PREFLIGHT)], cwd=ROOT, env=env, text=True,
            capture_output=True, check=False
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("include directory resolves outside", result.stderr)

    def test_ambiguous_installed_header_candidates_require_selection(self) -> None:
        second = self.root / "include" / "ascend_hal.h"
        second.write_text(COMPLETE_HEADER + "\n/* second file */\n", encoding="utf-8")
        result = self.run_preflight()
        self.assertEqual(result.returncode, 2)
        self.assertIn("multiple distinct ascend_hal.h candidates", result.stderr)

    def test_paths_resolving_outside_driver_root_are_rejected(self) -> None:
        external = Path(self.tempdir.name) / "external-ascend_hal.h"
        external.write_text(COMPLETE_HEADER, encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "KVAF_ASCEND_DRIVER_ROOT": str(self.root),
                "KVAF_HCCP_HAL_HEADER": str(external),
                "KVAF_HCCP_HAL_NM": str(self.nm),
            }
        )
        result = subprocess.run(
            ["bash", str(PREFLIGHT)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("resolves outside", result.stderr)


if __name__ == "__main__":
    unittest.main()
