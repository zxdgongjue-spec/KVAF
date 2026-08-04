from __future__ import annotations

import pathlib
import shlex
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "access_operator/a5/provider"


FAKE_DLFCN = r"""
#ifndef KVAF_TEST_DLFCN_H
#define KVAF_TEST_DLFCN_H
#include <stddef.h>
typedef struct {
    const char *dli_fname;
    void *dli_fbase;
    const char *dli_sname;
    void *dli_saddr;
} Dl_info;
#define RTLD_NOW 2
#define RTLD_LOCAL 0
void *dlopen(const char *, int);
void *dlsym(void *, const char *);
char *dlerror(void);
int dlclose(void *);
int dladdr(const void *, Dl_info *);
#endif
"""


def run_compiler(command: list[str]) -> None:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise AssertionError(
            "compiler command failed\n"
            f"command: {shlex.join(command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


FAKE_URMA_API = r"""
#ifndef KVAF_TEST_URMA_API_H
#define KVAF_TEST_URMA_API_H
#include <errno.h>
#include <stdint.h>
typedef int urma_status_t;
typedef struct urma_init_attr { uint32_t unused; } urma_init_attr_t;
typedef struct urma_device { uint32_t unused; } urma_device_t;
typedef struct urma_context { uint32_t unused; } urma_context_t;
typedef struct urma_jfc { uint32_t unused; } urma_jfc_t;
typedef struct urma_jfr { uint32_t unused; } urma_jfr_t;
typedef struct urma_jfc_cfg { uint32_t depth; } urma_jfc_cfg_t;
typedef struct urma_jfr_cfg {
    uint32_t depth;
    uint32_t trans_mode;
    uint32_t max_sge;
    uint32_t min_rnr_timer;
    urma_jfc_t *jfc;
    struct { uint32_t token; } token_value;
} urma_jfr_cfg_t;
#define URMA_SUCCESS 0
#define URMA_EINVAL EINVAL
#define URMA_EEXIST EEXIST
#define URMA_API_VERSION 9U
#define URMA_TM_RM 1U
#define URMA_TYPICAL_MIN_RNR_TIMER 12U
#endif
"""


HARNESS = r"""
#define dlopen kvaf_test_dlopen
#define dlsym kvaf_test_dlsym
#define dlerror kvaf_test_dlerror
#define dlclose kvaf_test_dlclose
#define dladdr kvaf_test_dladdr

#include "runtime_matched_urma_probe.c"

#include <stdio.h>
#include <stdlib.h>

enum {
    SCENARIO_DISCOVER,
    SCENARIO_INIT_DEVICE,
    SCENARIO_SUCCESS,
    SCENARIO_EEXIST,
    SCENARIO_JFR_NULL,
    SCENARIO_DELETE_JFR_FAIL,
    SCENARIO_DISCOVER_PARTIAL,
    SCENARIO_CREATE_PARTIAL,
    SCENARIO_DLOPEN_FAIL,
    SCENARIO_INIT_FAIL,
    SCENARIO_PROVIDER_ID_MISMATCH,
};

static int g_scenario;
static int g_dlopen_calls;
static int g_init_calls;
static int g_uninit_calls;
static int g_delete_context_calls;
static int g_delete_jfc_calls;
static int g_delete_jfr_calls;
static const char *g_loader_error;

static urma_status_t fake_urma_init(urma_init_attr_t *attr)
{
    (void)attr;
    ++g_init_calls;
    if (g_scenario == SCENARIO_EEXIST) return URMA_EEXIST;
    if (g_scenario == SCENARIO_INIT_FAIL) return URMA_EINVAL;
    return URMA_SUCCESS;
}

static urma_status_t fake_urma_uninit(void)
{
    ++g_uninit_calls;
    return URMA_SUCCESS;
}

static urma_device_t *fake_get_device(char *name)
{
    return name[0] == '\0' ? NULL : (urma_device_t *)(uintptr_t)0x1000;
}

static urma_context_t *fake_create_context(urma_device_t *device,
    uint32_t eid_index)
{
    (void)eid_index;
    return device == NULL ? NULL : (urma_context_t *)(uintptr_t)0x2000;
}

static urma_status_t fake_delete_context(urma_context_t *context)
{
    ++g_delete_context_calls;
    return context == NULL ? URMA_EINVAL : URMA_SUCCESS;
}

static urma_jfc_t *fake_create_jfc(urma_context_t *context,
    urma_jfc_cfg_t *config)
{
    return context == NULL || config->depth != 64 ? NULL :
        (urma_jfc_t *)(uintptr_t)0x3000;
}

static urma_status_t fake_delete_jfc(urma_jfc_t *jfc)
{
    ++g_delete_jfc_calls;
    return jfc == NULL ? URMA_EINVAL : URMA_SUCCESS;
}

static urma_jfr_t *fake_create_jfr(urma_context_t *context,
    urma_jfr_cfg_t *config)
{
    if (g_scenario == SCENARIO_JFR_NULL) {
        errno = EINVAL;
        return NULL;
    }
    return context == NULL || config->jfc == NULL ? NULL :
        (urma_jfr_t *)(uintptr_t)0x4000;
}

static urma_status_t fake_delete_jfr(urma_jfr_t *jfr)
{
    ++g_delete_jfr_calls;
    if (g_scenario == SCENARIO_DELETE_JFR_FAIL) {
        errno = EBUSY;
        return EBUSY;
    }
    return jfr == NULL ? URMA_EINVAL : URMA_SUCCESS;
}

static void *function_address(const void *address, size_t size)
{
    void *result = NULL;
    if (size != sizeof(result)) {
        abort();
    }
    memcpy(&result, address, sizeof(result));
    return result;
}

#define RETURN_FUNCTION(name, function) do { \
    if (strcmp(symbol, name) == 0) { \
        __typeof__(&function) pointer = &function; \
        return function_address(&pointer, sizeof(pointer)); \
    } \
} while (0)

void *kvaf_test_dlopen(const char *name, int flags)
{
    (void)flags;
    ++g_dlopen_calls;
    if (g_scenario == SCENARIO_DLOPEN_FAIL) {
        g_loader_error = "constructor failed after unknown side effects";
        return NULL;
    }
    return strcmp(name, "liburma.so.0") == 0 ? (void *)(uintptr_t)0x5000 : NULL;
}

void *kvaf_test_dlsym(void *handle, const char *symbol)
{
    (void)handle;
    RETURN_FUNCTION("urma_init", fake_urma_init);
    RETURN_FUNCTION("urma_uninit", fake_urma_uninit);
    RETURN_FUNCTION("urma_get_device_by_name", fake_get_device);
    RETURN_FUNCTION("urma_create_context", fake_create_context);
    RETURN_FUNCTION("urma_delete_context", fake_delete_context);
    RETURN_FUNCTION("urma_create_jfc", fake_create_jfc);
    RETURN_FUNCTION("urma_delete_jfc", fake_delete_jfc);
    if ((g_scenario == SCENARIO_DISCOVER_PARTIAL ||
         g_scenario == SCENARIO_CREATE_PARTIAL) &&
        strcmp(symbol, "urma_create_jfr") == 0) {
        g_loader_error = "optional lifecycle symbol unavailable";
        return NULL;
    }
    RETURN_FUNCTION("urma_create_jfr", fake_create_jfr);
    RETURN_FUNCTION("urma_delete_jfr", fake_delete_jfr);
    g_loader_error = "missing test symbol";
    return NULL;
}

char *kvaf_test_dlerror(void)
{
    const char *error = g_loader_error;
    g_loader_error = NULL;
    return (char *)error;
}

int kvaf_test_dlclose(void *handle)
{
    return handle == NULL ? -1 : 0;
}

int kvaf_test_dladdr(const void *address, Dl_info *info)
{
    (void)address;
    memset(info, 0, sizeof(*info));
    info->dli_fname = "/device/runtime/liburma.so.0";
    info->dli_fbase = (void *)(uintptr_t)0x6000;
    return 1;
}

static int invoke(uint32_t mode,
    struct kvaf_urma_runtime_probe_result_v2 *result)
{
    struct kvaf_urma_aicpu_runtime_probe_args_v2 args;
    memset(&args, 0, sizeof(args));
    memcpy(args.kernel_name, "aicpu_kvaf_urma_rt_probe_v2",
        sizeof("aicpu_kvaf_urma_rt_probe_v2"));
    memcpy(args.so_name, "libkvaf_urma_provider_device.so",
        sizeof("libkvaf_urma_provider_device.so"));
    memcpy(args.device_name, "udmac-test", sizeof("udmac-test"));
    args.magic = KVAF_URMA_RUNTIME_PROBE_MAGIC;
    args.abi_version = KVAF_URMA_RUNTIME_PROBE_ABI_VERSION;
    args.struct_size = sizeof(args);
    args.result_address = (uint64_t)(uintptr_t)result;
    args.queue_depth = 64;
    args.mode = mode;
    args.request_nonce[0] = UINT64_C(0x1122334455667788);
    args.request_nonce[1] = UINT64_C(0x8877665544332211);
    args.expected_provider_abi_token =
        KVAF_URMA_RUNTIME_PROBE_PROVIDER_ABI_TOKEN;
    args.logical_device_id = 2;
    memset(args.expected_provider_build_id_sha256,
        g_scenario == SCENARIO_PROVIDER_ID_MISMATCH ? 0x22 : 0x11,
        sizeof(args.expected_provider_build_id_sha256));
    return aicpu_kvaf_urma_rt_probe_v2(&args);
}

static int check(bool condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "%s\n", message);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    _Alignas(64) struct kvaf_urma_runtime_probe_result_v2 first;
    _Alignas(64) struct kvaf_urma_runtime_probe_result_v2 second;
    if (argc != 2) return 2;
    if (strcmp(argv[1], "discover") == 0) g_scenario = SCENARIO_DISCOVER;
    else if (strcmp(argv[1], "init-device") == 0) g_scenario = SCENARIO_INIT_DEVICE;
    else if (strcmp(argv[1], "success") == 0) g_scenario = SCENARIO_SUCCESS;
    else if (strcmp(argv[1], "eexist") == 0) g_scenario = SCENARIO_EEXIST;
    else if (strcmp(argv[1], "jfr-null") == 0) g_scenario = SCENARIO_JFR_NULL;
    else if (strcmp(argv[1], "delete-fail") == 0) g_scenario = SCENARIO_DELETE_JFR_FAIL;
    else if (strcmp(argv[1], "discover-partial") == 0) g_scenario = SCENARIO_DISCOVER_PARTIAL;
    else if (strcmp(argv[1], "create-partial") == 0) g_scenario = SCENARIO_CREATE_PARTIAL;
    else if (strcmp(argv[1], "dlopen-fail") == 0) g_scenario = SCENARIO_DLOPEN_FAIL;
    else if (strcmp(argv[1], "init-fail") == 0) g_scenario = SCENARIO_INIT_FAIL;
    else if (strcmp(argv[1], "provider-id-mismatch") == 0)
        g_scenario = SCENARIO_PROVIDER_ID_MISMATCH;
    else return 2;

    memset(&first, 0, sizeof(first));
    uint32_t mode = (g_scenario == SCENARIO_DISCOVER ||
                     g_scenario == SCENARIO_DISCOVER_PARTIAL) ?
        KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER :
        (g_scenario == SCENARIO_INIT_DEVICE ||
         g_scenario == SCENARIO_DLOPEN_FAIL ||
         g_scenario == SCENARIO_INIT_FAIL ||
         g_scenario == SCENARIO_PROVIDER_ID_MISMATCH ?
         KVAF_URMA_RUNTIME_PROBE_MODE_INIT_DEVICE :
         KVAF_URMA_RUNTIME_PROBE_MODE_CREATE_DELETE);
    if (invoke(mode, &first) != 0) return 1;

    if (g_scenario == SCENARIO_PROVIDER_ID_MISMATCH) {
        return check(first.outcome ==
                         KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH &&
                     first.cleanup_required == 0 &&
                     first.cleanup_completed == 0 &&
                     g_dlopen_calls == 0 && g_init_calls == 0,
                     "provider identity mismatch was not fail-closed");
    }

    /* Until a sealed capability ledger exists, every identity-bound request
     * must stop before dlopen and all official URMA entry points.  Keep the
     * fake lifecycle below compiled as implementation coverage, but do not
     * let an exported AICPU entry reach it through any mode or scenario. */
    return check(first.outcome ==
                     KVAF_URMA_RUNTIME_PROBE_OUTCOME_CAPABILITY_LEDGER_UNAVAILABLE &&
                 first.primary_outcome ==
                     KVAF_URMA_RUNTIME_PROBE_OUTCOME_CAPABILITY_LEDGER_UNAVAILABLE &&
                 first.stage == KVAF_URMA_RUNTIME_PROBE_STAGE_NONE &&
                 first.operation_status == -ENOTSUP &&
                 first.operation_errno == 0 &&
                 first.operation_errno_valid == 0 &&
                 first.cleanup_required == 0 &&
                 first.cleanup_completed == 0 &&
                 first.resolved_symbol_mask == 0 &&
                 first.flags == KVAF_URMA_RUNTIME_PROBE_PUBLIC_ABI_UNPROVEN &&
                 first.runtime_library_path[0] == '\0' &&
                 first.loader_error[0] == '\0' &&
                 g_dlopen_calls == 0 && g_init_calls == 0 &&
                 g_uninit_calls == 0 && g_delete_context_calls == 0 &&
                 g_delete_jfc_calls == 0 && g_delete_jfr_calls == 0,
                 "capability ledger gate allowed an official API side effect");

    if (g_scenario == SCENARIO_DISCOVER) {
        return check(first.outcome == KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK &&
                     first.compiled_urma_api_version == URMA_API_VERSION &&
                     first.required_symbol_mask ==
                         KVAF_URMA_RUNTIME_PROBE_BASE_SYMBOLS &&
                     first.resolved_symbol_mask ==
                         KVAF_URMA_RUNTIME_PROBE_LIFECYCLE_SYMBOLS &&
                     first.cleanup_required == KVAF_URMA_RUNTIME_PROBE_CLEANUP_LIBRARY &&
                     first.cleanup_completed == first.cleanup_required &&
                     first.request_nonce[0] == UINT64_C(0x1122334455667788) &&
                     first.request_nonce[1] == UINT64_C(0x8877665544332211) &&
                     first.provider_abi_token ==
                         KVAF_URMA_RUNTIME_PROBE_PROVIDER_ABI_TOKEN &&
                     first.mode == KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER &&
                     first.logical_device_id == 2 &&
                     first.args_size == sizeof(struct
                         kvaf_urma_aicpu_runtime_probe_args_v2) &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN) != 0 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED) != 0 &&
                     g_init_calls == 0,
                     "discover explicit-operation contract failed");
    }
    if (g_scenario == SCENARIO_DISCOVER_PARTIAL) {
        return check(first.outcome == KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK &&
                     first.required_symbol_mask ==
                         KVAF_URMA_RUNTIME_PROBE_BASE_SYMBOLS &&
                     first.resolved_symbol_mask ==
                         (KVAF_URMA_RUNTIME_PROBE_LIFECYCLE_SYMBOLS &
                          ~KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JFR) &&
                     first.cleanup_required == first.cleanup_completed &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN) != 0 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED) != 0 &&
                     g_init_calls == 0,
                     "partial official surface hid its optional gap");
    }
    if (g_scenario == SCENARIO_CREATE_PARTIAL) {
        return check(first.outcome ==
                         KVAF_URMA_RUNTIME_PROBE_OUTCOME_SYMBOL_UNAVAILABLE &&
                     first.required_symbol_mask ==
                         KVAF_URMA_RUNTIME_PROBE_LIFECYCLE_SYMBOLS &&
                     (first.resolved_symbol_mask &
                      KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JFR) == 0 &&
                     first.cleanup_required == first.cleanup_completed &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN) != 0 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED) != 0 &&
                     g_init_calls == 0,
                     "required lifecycle symbol gap was not isolated");
    }
    if (g_scenario == SCENARIO_SUCCESS) {
        return check(first.outcome == KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK &&
                     first.cleanup_required == 31 && first.cleanup_completed == 31 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN) != 0 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED) != 0 &&
                     g_uninit_calls == 1 && g_delete_jfr_calls == 1 &&
                     g_delete_jfc_calls == 1 && g_delete_context_calls == 1,
                     "success lifecycle is incomplete");
    }
    if (g_scenario == SCENARIO_INIT_DEVICE) {
        return check(first.outcome == KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK &&
                     first.cleanup_required == 24 &&
                     first.cleanup_completed == 24 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN) != 0 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED) != 0 &&
                     g_init_calls == 1 && g_uninit_calls == 1 &&
                     g_delete_context_calls == 0 &&
                     g_delete_jfc_calls == 0 && g_delete_jfr_calls == 0,
                     "init-device lifecycle is incomplete");
    }

    memset(&second, 0, sizeof(second));
    if (invoke(KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER, &second) != 0) return 1;
    if (check(second.outcome ==
              KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROCESS_POISONED &&
              (second.flags & KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED) != 0,
              "unsafe process accepted a second probe") != 0) return 1;

    if (g_scenario == SCENARIO_EEXIST) {
        return check(first.outcome ==
                     KVAF_URMA_RUNTIME_PROBE_OUTCOME_RUNTIME_ALREADY_INITIALIZED &&
                     first.cleanup_required == 16 && first.cleanup_completed == 16 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN) != 0 &&
                     g_uninit_calls == 0, "borrowed runtime ownership was violated");
    }
    if (g_scenario == SCENARIO_DLOPEN_FAIL) {
        return check(first.outcome ==
                         KVAF_URMA_RUNTIME_PROBE_OUTCOME_LIBRARY_UNAVAILABLE &&
                     first.cleanup_required == 0 &&
                     first.cleanup_completed == 0 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN) != 0 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED) != 0 &&
                     g_init_calls == 0 && g_uninit_calls == 0,
                     "failed dlopen over-proved cleanup");
    }
    if (g_scenario == SCENARIO_INIT_FAIL) {
        return check(first.outcome ==
                         KVAF_URMA_RUNTIME_PROBE_OUTCOME_INIT_FAILED &&
                     first.cleanup_required == 16 &&
                     first.cleanup_completed == 16 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN) != 0 &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED) != 0 &&
                     g_init_calls == 1 && g_uninit_calls == 0,
                     "failed urma_init over-proved cleanup");
    }
    if (g_scenario == SCENARIO_JFR_NULL) {
        return check(first.outcome ==
                     KVAF_URMA_RUNTIME_PROBE_OUTCOME_JFR_CREATE_FAILED_UNPROVEN &&
                     (first.flags &
                      KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN) != 0 &&
                     first.cleanup_required == 30 && first.cleanup_completed == 6 &&
                     g_uninit_calls == 0, "NULL-create uncertainty was hidden");
    }
    return check(first.outcome ==
                 KVAF_URMA_RUNTIME_PROBE_OUTCOME_CLEANUP_UNSAFE &&
                 first.cleanup_required == 31 && first.cleanup_completed == 0 &&
                 g_delete_jfc_calls == 0 && g_delete_context_calls == 0 &&
                 g_uninit_calls == 0, "delete failure did not stop parent cleanup");
}
"""


class RuntimeMatchedUrmaProbeCompileTests(unittest.TestCase):
    def test_provider_source_compiles_with_strict_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = pathlib.Path(temporary)
            (temporary_path / "urma_api.h").write_text(
                textwrap.dedent(FAKE_URMA_API), encoding="utf-8"
            )
            (temporary_path / "dlfcn.h").write_text(
                textwrap.dedent(FAKE_DLFCN), encoding="utf-8"
            )
            command = [
                "cc", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                "-fsyntax-only",
                (
                    '-DKVAF_RUNTIME_PROBE_PROVIDER_BUILD_ID_HEX="'
                    + "1" * 64
                    + '"'
                ),
                "-I", temporary,
                "-I", str(PROVIDER / "include"),
                str(PROVIDER / "runtime_matched_urma_probe.c"),
            ]
            run_compiler(command)


class RuntimeMatchedUrmaProbeBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = pathlib.Path(cls.temporary.name) / "runtime-probe-harness"
        harness = pathlib.Path(cls.temporary.name) / "harness.c"
        fake_header = pathlib.Path(cls.temporary.name) / "urma_api.h"
        fake_dlfcn = pathlib.Path(cls.temporary.name) / "dlfcn.h"
        harness.write_text(textwrap.dedent(HARNESS), encoding="utf-8")
        fake_header.write_text(textwrap.dedent(FAKE_URMA_API), encoding="utf-8")
        # The harness renames the loader API before including the provider.
        # Using glibc's declarations would transfer implementation-specific
        # attributes to those local fake definitions, which GCC diagnoses
        # under -Werror.  Use one controlled ABI stub for both compile tests.
        fake_dlfcn.write_text(textwrap.dedent(FAKE_DLFCN), encoding="utf-8")
        command = [
            "cc", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
            (
                '-DKVAF_RUNTIME_PROBE_PROVIDER_BUILD_ID_HEX="'
                + "1" * 64
                + '"'
            ),
            "-I", cls.temporary.name,
            "-I", str(PROVIDER),
            "-I", str(PROVIDER / "include"),
            str(harness), "-pthread", "-o", str(cls.binary),
        ]
        run_compiler(command)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_lifecycle_outcomes_and_poison_latch(self) -> None:
        for scenario in (
            "discover", "init-device", "success", "eexist", "jfr-null",
            "delete-fail", "discover-partial", "create-partial",
            "dlopen-fail", "init-fail", "provider-id-mismatch",
        ):
            with self.subTest(scenario=scenario):
                result = subprocess.run(
                    [str(self.binary), scenario], text=True,
                    capture_output=True, check=False
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
