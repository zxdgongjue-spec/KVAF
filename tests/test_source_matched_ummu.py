import pathlib
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "access_operator" / "a5" / "provider"
HEADER = PROVIDER / "include" / "kvaf_source_matched_ummu.h"
RUNTIME = PROVIDER / "source_matched_ummu.c"
COLLECTIONS = PROVIDER / "source_matched_ummu_collections.c"


class SourceMatchedUmmuTests(unittest.TestCase):
    def test_checked_contract_is_explicit_and_fail_closed(self):
        header = HEADER.read_text(encoding="utf-8")
        runtime = RUNTIME.read_text(encoding="utf-8")

        self.assertIn("KVAF_SOURCE_UMMU_CHECKED_RETRYABLE_V1", header)
        self.assertIn("KVAF_SOURCE_UMMU_PINNED_FULL_UAPI", header)
        self.assertIn("KVAF_SOURCE_UMMU_RETRY_UNSAFE", runtime)
        self.assertIn("pending_state_unknown", runtime)
        self.assertIn("KVAF_SOURCE_UMMU_FAILURE_PLBI_QUEUE_FULL", runtime)
        self.assertIn("KVAF_SOURCE_UMMU_FAILURE_PLBI_PSYNC_TIMEOUT", runtime)
        self.assertIn("KVAF_SOURCE_UMMU_FAILURE_FREE_TID_IOCTL", runtime)
        self.assertIn("KVAF_SOURCE_UMMU_FAILURE_MUNMAP", runtime)
        self.assertIn("KVAF_UMMU_FULL_UAPI_VERIFIED", runtime)
        self.assertIn("info->hw_cap", runtime)
        self.assertNotIn("#define hw_cap", runtime)
        self.assertNotIn("#define HW_CAP_KCMD_PLBI", runtime)
        self.assertIn("UMMU_IOCPLBI_VA", runtime)
        self.assertIn("UMMU_IOCPLBI_ALL", runtime)
        self.assertNotIn('dlopen("libummu', runtime)

    def test_ungrant_and_free_tid_retain_retry_state(self):
        runtime = RUNTIME.read_text(encoding="utf-8")

        self.assertIn("kvaf_retry_data_operation_locked", runtime)
        self.assertIn("kvaf_retry_retained_mappings", runtime)
        self.assertIn("KVAF_UMMU_CLEANUP_PLBI_ALL_DONE", runtime)
        self.assertIn("KVAF_UMMU_CLEANUP_KERNEL_TID_FREED", runtime)
        self.assertIn("KVAF_UMMU_CLEANUP_LOCAL_RELEASED", runtime)
        self.assertLess(
            runtime.index("kvaf_send_plbi_checked(state, PLBI_ALL_OP"),
            runtime.index("kvaf_issue_free_tid_ioctl(state)"),
        )

    def test_partial_tid_create_returns_exact_cleanup_handle(self):
        runtime = RUNTIME.read_text(encoding="utf-8")
        token = (PROVIDER / "source_matched_udma_tid.c").read_text(
            encoding="utf-8"
        )
        provider = (PROVIDER / "urma_perftest_provider.c").read_text(
            encoding="utf-8"
        )

        allocate_start = runtime.index("int ummu_allocate_tid(")
        allocate_end = runtime.index("\nint ummu_free_tid(", allocate_start)
        allocate = runtime[allocate_start:allocate_end]
        owner_lock = allocate.index(
            "pthread_mutex_lock(&g_kvaf_ummu_ctx->ctx_mutex)"
        )
        alloc_ioctl = allocate.index("ummu_get_tid(")
        registry = allocate.index("kvaf_add_state_locked(state)")
        publish = allocate.index("*tid_out = info.tid;")
        mapt_create = allocate.index("ummu_mapt_create(&info)")

        self.assertIn("*tid_out = UINT32_MAX;", allocate)
        self.assertLess(owner_lock, alloc_ioctl)
        self.assertLess(alloc_ioctl, registry)
        self.assertLess(registry, publish)
        self.assertLess(publish, mapt_create)
        self.assertEqual(
            allocate.count(
                "pthread_mutex_lock(&g_kvaf_ummu_ctx->ctx_mutex)"
            ),
            1,
            "no fallible owner lock may occur after ALLOC_TID succeeds",
        )
        self.assertIn("kvaf_free_tid_state_locked(state)", allocate)
        self.assertIn("uint32_t tid = UINT32_MAX;", token)
        self.assertIn("if (tid != UINT32_MAX)", token)
        self.assertIn("ummu_free_tid(tid)", token)
        self.assertIn("ummu_status.last_state_unknown == 0U", provider)
        self.assertIn("kvaf_source_ummu_state_is_unknown()", provider)
        self.assertIn("KVAF_URMA_PROVIDER_CLOSE_UNSAFE", provider)

    def test_builder_requires_pinned_full_uapi_and_no_system_fallback(self):
        cmake = (PROVIDER / "CMakeLists.txt").read_text(encoding="utf-8")
        script = (ROOT / "scripts" / "build_a5_direct_urma_provider.sh").read_text(
            encoding="utf-8"
        )
        for source in (cmake, script):
            self.assertIn("KVAF_UMMU_UAPI_ROOT", source)
            self.assertIn("info.hw_cap", source)
            self.assertIn("HW_CAP_EXPAN", source)
            self.assertIn("HW_CAP_DOUBLE_PLBI", source)
            self.assertIn("HW_CAP_KCMD_PLBI", source)
            self.assertIn("UMMU_IOCPLBI_VA", source)
            self.assertIn("UMMU_IOCPLBI_ALL", source)
            self.assertNotIn("KVAF_UMMU_LIBRARY", source)
        self.assertIn("KVAF_UMMU_FULL_UAPI_VERIFIED=1", cmake)
        self.assertIn("source_matched_ummu.c", cmake)
        self.assertIn("source_matched_ummu_collections.c", cmake)
        self.assertIn("udma_teardown_contract=checked-retryable-v1", script)
        self.assertIn("external liburma/UMMU/UDMA code", script)

    def test_builder_hashes_the_complete_pinned_uapi_header_tree(self):
        script = (
            ROOT / "scripts" / "build_a5_direct_urma_provider.sh"
        ).read_text(encoding="utf-8")
        roots_start = script.index("source_roots=(\n")
        roots_end = script.index("\n)\n", roots_start)
        source_roots = script[roots_start:roots_end]

        self.assertIn(
            '"${KVAF_UMMU_UAPI_ROOT}/include/uapi"', source_roots
        )
        self.assertNotIn('"${KVAF_UMMU_UAPI_HEADER}"', source_roots)
        self.assertIn("-name '*.h'", script)
        self.assertIn(
            '"${legacy_ummu_uapi_root}/"*)', script
        )

    def test_uapi_probe_is_published_and_rechecked_before_hardware(self):
        cmake = (PROVIDER / "CMakeLists.txt").read_text(encoding="utf-8")
        builder = (
            ROOT / "scripts" / "build_a5_direct_urma_provider.sh"
        ).read_text(encoding="utf-8")
        runner = (
            ROOT / "scripts" / "run_a5_direct_urma_h2d_smoke.sh"
        ).read_text(encoding="utf-8")
        probe = (PROVIDER / "ummu_uapi_probe.c").read_text(encoding="utf-8")

        self.assertIn("add_executable(kvaf_ummu_uapi_probe", cmake)
        for field in (
            "sizeof(struct ummu_tid_info)",
            "offsetof(struct ummu_tid_info, hw_cap)",
            "UMMU_IOCALLOC_TID",
            "UMMU_IOCFREE_TID",
            "UMMU_IOCPLBI_VA",
            "UMMU_IOCPLBI_ALL",
            "HW_CAP_EXPAN",
            "HW_CAP_DOUBLE_PLBI",
            "HW_CAP_KCMD_PLBI",
        ):
            self.assertIn(field, probe)
        self.assertIn("ummu_uapi_probe_filename=kvaf_ummu_uapi_probe", builder)
        self.assertIn("ummu_uapi_probe_sha256=", builder)
        self.assertIn('cp "${ummu_uapi_probe}"', builder)
        self.assertIn("manifest_value ummu_uapi_abi", runner)
        self.assertIn("manifest_value ummu_uapi_probe_sha256", runner)
        self.assertIn('probe_output=$("${probe_path}")', runner)

    def test_uapi_compile_entrypoints_define_ioctl_encoding_first(self):
        entrypoints = (
            ROOT / "scripts" / "build_a5_direct_urma_provider.sh",
            PROVIDER / "CMakeLists.txt",
            PROVIDER / "ummu_uapi_probe.c",
            RUNTIME,
        )
        for entrypoint in entrypoints:
            with self.subTest(entrypoint=entrypoint.name):
                source = entrypoint.read_text(encoding="utf-8")
                ioctl_include = source.index("#include <sys/ioctl.h>")
                ummu_include = source.index('#include "ummu_api.h"')
                self.assertLess(
                    ioctl_include,
                    ummu_include,
                    "_IOW/_IOR/_IOWR must be defined before UMMU ioctl macros",
                )

    def test_provider_advertises_checked_capability_only_after_runtime_proof(self):
        provider = (PROVIDER / "urma_perftest_provider.c").read_text(
            encoding="utf-8"
        )
        udma_header = (
            PROVIDER / "include" / "kvaf_source_matched_udma.h"
        ).read_text(encoding="utf-8")

        self.assertIn("kvaf_ummu_checked_runtime_is_armed", provider)
        self.assertIn("kvaf_source_ummu_get_status(&status)", provider)
        self.assertIn("status.capability_magic ==", provider)
        self.assertIn("status.capability_bits == expected_capability", provider)
        self.assertIn("status.last_state_unknown == 0U", provider)
        self.assertGreaterEqual(
            provider.count("!kvaf_ummu_checked_runtime_is_armed()"), 2
        )
        self.assertIn(
            "KVAF_UDMA_TEARDOWN_SOURCE_OWNED_CHECKED_RETRYABLE_V1",
            udma_header,
        )

    def test_c_collections_match_upstream_range_contract(self):
        compiler = shutil.which("cc") or shutil.which("clang")
        if compiler is None:
            self.skipTest("no C compiler")
        ummu_root = ROOT.parent / "urma_perftest" / "ubengine" / "ssapi"
        ummu_source = ummu_root / "userspace" / "ummu"
        if not (ummu_source / "ummu_seg_tree.h").is_file():
            self.skipTest("source-matched UMMU headers are not present")

        harness = textwrap.dedent(
            r"""
            #include <stdint.h>
            #include <stdlib.h>
            #include "ummu_map.h"
            #include "ummu_seg_tree.h"
            #include "kvaf_source_matched_ummu.h"

            int main(void) {
              void *map = ummu_map_create();
              uint64_t tree = 0;
              int comp = -1;
              int *first = calloc(1, sizeof(*first));
              int *second = calloc(1, sizeof(*second));
              if (!map || !first || !second) return 1;
              if (ummu_map_insert(map, 7, first) != 0) return 2;
              if (ummu_map_insert(map, 7, second) == 0) return 3;
              if (ummu_map_get(map, 7) != first) return 4;
              if (ummu_map_del(map, 7) != 0) return 5;
              free(first);
              free(second);
              ummu_map_clear(map, NULL);

              if (ummu_create_seg_tree(&tree) != 0) return 6;
              first = calloc(1, sizeof(*first));
              second = calloc(1, sizeof(*second));
              if (!first || !second) return 7;
              if (ummu_insert_seg(first, tree, 0x1000, 0xfff) != 0) return 8;
              if (ummu_insert_seg(second, tree, 0x1800, 0xfff) == 0) return 9;
              free(second);
              if (ummu_find_seg(0x1000, 0xfff, tree, &comp) != first ||
                  comp != RANGE_MATCH) return 10;
              if (ummu_find_seg(0x1800, 0x10, tree, &comp) != NULL ||
                  comp != RANGE_OVERLAP) return 11;
              if (ummu_delete_seg(0x1000, 0xfff, tree, NULL) != 0) return 12;
              if (!kvaf_source_ummu_seg_tree_is_empty(tree)) return 13;
              ummu_destroy_seg_tree(&tree, NULL);
              return tree == 0 ? 0 : 14;
            }
            """
        )
        with tempfile.TemporaryDirectory(prefix="kvaf-ummu-") as temp:
            temp_path = pathlib.Path(temp)
            harness_path = temp_path / "harness.c"
            binary_path = temp_path / "harness"
            harness_path.write_text(harness, encoding="utf-8")
            subprocess.run(
                [
                    compiler,
                    "-std=gnu11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    f"-I{PROVIDER / 'include'}",
                    f"-I{ummu_source}",
                    str(COLLECTIONS),
                    str(harness_path),
                    "-o",
                    str(binary_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(binary_path)], check=True)


if __name__ == "__main__":
    unittest.main()
