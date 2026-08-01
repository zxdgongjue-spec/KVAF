import pathlib
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "access_operator" / "a5" / "provider"
HEADER = PROVIDER / "include" / "kvaf_source_matched_ummu.h"
MAPPING_HEADER = (
    PROVIDER / "include" / "kvaf_source_matched_ummu_mapping.h"
)
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
        self.assertIn("g_kvaf_ummu_cleanup_unknown", runtime)
        self.assertIn("kvaf_source_ummu_cleanup_is_unknown", header)
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
        self.assertIn("HW_CAP_FREE_BIT", runtime)
        self.assertIn("HW_CAP_PPLBI", runtime)
        self.assertIn("ummu_plbi_f_bit_cmd", runtime)
        self.assertIn("kvaf_mark_inline_plbi_failed", runtime)
        self.assertIn("KVAF_SOURCE_UMMU_RETRY_UNSAFE", runtime)
        self.assertIn("state->mapt->positive_plbi", runtime)
        self.assertIn("data_info.lvl = MAX_LEVEL_INDEX", runtime)
        self.assertNotIn("g_kvaf_ummu_ctx->tid_cnt", runtime)
        self.assertNotIn('dlopen("libummu', runtime)

    def test_unknown_cleanup_is_process_lifetime_latched(self):
        runtime = RUNTIME.read_text(encoding="utf-8")

        record_start = runtime.index("static void kvaf_record_failure(")
        record_end = runtime.index("\nstatic void kvaf_set_pending(", record_start)
        record = runtime[record_start:record_end]
        status_start = runtime.index("int kvaf_source_ummu_get_status(")
        status_end = runtime.index(
            "\nint kvaf_source_ummu_retry_pending(", status_start
        )
        get_status = runtime[status_start:status_end]

        self.assertIn("if (state_unknown)", record)
        self.assertIn(
            "&g_kvaf_ummu_cleanup_unknown, 1U, __ATOMIC_RELEASE", record
        )
        self.assertNotIn(
            "g_kvaf_ummu_cleanup_unknown, 0U", runtime
        )
        self.assertIn(
            "kvaf_source_ummu_cleanup_is_unknown()", get_status
        )

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

    def test_mmap_ownership_is_reserved_before_mapping_and_never_allocated_on_failure(self):
        runtime = RUNTIME.read_text(encoding="utf-8")
        header = HEADER.read_text(encoding="utf-8")

        get_start = runtime.index("void *ummu_get_core_buf(")
        get_end = runtime.index("\nvoid ummu_free_core_buf(", get_start)
        get_core_buf = runtime[get_start:get_end]
        checked_start = runtime.index("static int kvaf_checked_munmap(")
        checked_end = runtime.index(
            "\nstatic int kvaf_retry_retained_mappings(", checked_start
        )
        checked_munmap = runtime[checked_start:checked_end]

        self.assertLess(get_core_buf.index("calloc("), get_core_buf.index("mmap("))
        self.assertLess(
            get_core_buf.index("mmap("),
            get_core_buf.index("kvaf_ummu_mapping_attach("),
        )
        self.assertNotIn("calloc(", checked_munmap)
        self.assertNotIn("malloc(", checked_munmap)
        self.assertNotIn("realloc(", checked_munmap)
        self.assertIn("kvaf_ummu_mapping_mark_retained(", checked_munmap)
        self.assertIn("kvaf_latch_mapping_ownership_unknown(", checked_munmap)
        self.assertIn("KVAF_SOURCE_UMMU_RETRY_UNSAFE", runtime)
        self.assertIn("preallocated ownership record", header)

    def test_intrusive_mapping_ledger_transfers_exact_ownership(self):
        compiler = shutil.which("cc") or shutil.which("clang")
        if compiler is None:
            self.skipTest("no C compiler")

        harness = textwrap.dedent(
            r"""
            #include <errno.h>
            #include <stddef.h>
            #include "kvaf_source_matched_ummu_mapping.h"

            int main(void) {
              struct kvaf_ummu_mapping_ledger ledger = {0};
              struct kvaf_ummu_owned_mapping first = {0};
              struct kvaf_ummu_owned_mapping second = {0};
              char first_address;
              char second_address;

              kvaf_ummu_mapping_attach(
                  &ledger, &first, &first_address, sizeof(first_address));
              kvaf_ummu_mapping_attach(
                  &ledger, &second, &second_address, sizeof(second_address));
              if (ledger.head != &second || second.next != &first) return 1;
              if (kvaf_ummu_mapping_find(
                      &ledger, &first_address, sizeof(first_address)) != &first) {
                return 2;
              }
              if (kvaf_ummu_mapping_mark_retained(
                      &ledger, &first_address, sizeof(first_address)) != 1) {
                return 3;
              }
              if (!first.retained || ledger.retained_count != 1) return 4;
              if (kvaf_ummu_mapping_mark_retained(
                      &ledger, &first_address, sizeof(first_address)) != 0 ||
                  ledger.retained_count != 1) {
                return 5;
              }
              if (kvaf_ummu_mapping_detach(
                      &ledger, &second_address, sizeof(second_address)) !=
                  &second) {
                return 6;
              }
              if (ledger.retained_count != 1 || ledger.head != &first) return 7;
              if (kvaf_ummu_mapping_detach(
                      &ledger, &first_address, sizeof(first_address)) != &first) {
                return 8;
              }
              if (ledger.head != NULL || ledger.retained_count != 0 ||
                  !first.retained) {
                return 9;
              }
              if (kvaf_ummu_mapping_mark_retained(
                      &ledger, &first_address, sizeof(first_address)) !=
                  -ENOENT) {
                return 10;
              }
              return 0;
            }
            """
        )
        with tempfile.TemporaryDirectory(prefix="kvaf-ummu-mapping-") as temp:
            temp_path = pathlib.Path(temp)
            harness_path = temp_path / "harness.c"
            binary_path = temp_path / "harness"
            harness_path.write_text(harness, encoding="utf-8")
            subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    f"-I{PROVIDER / 'include'}",
                    str(harness_path),
                    "-o",
                    str(binary_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(binary_path)], check=True)

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
        self.assertIn(
            "kvaf_source_ummu_cleanup_is_unknown() == 0", provider
        )
        self.assertIn("kvaf_source_ummu_state_is_unknown()", provider)
        self.assertIn("KVAF_URMA_PROVIDER_CLOSE_UNSAFE", provider)

    def test_builder_requires_pinned_full_uapi_and_no_system_fallback(self):
        cmake = (PROVIDER / "CMakeLists.txt").read_text(encoding="utf-8")
        script = (ROOT / "scripts" / "build_a5_direct_urma_provider.sh").read_text(
            encoding="utf-8"
        )
        for source in (cmake, script):
            self.assertIn("KVAF_LIBUMMU_ROOT", source)
            self.assertIn("KVAF_UMDK_ROOT", source)
            self.assertIn("info.hw_cap", source)
            self.assertIn("HW_CAP_EXPAN", source)
            self.assertIn("HW_CAP_DOUBLE_PLBI", source)
            self.assertIn("HW_CAP_KCMD_PLBI", source)
            self.assertIn("HW_CAP_FREE_BIT", source)
            self.assertIn("HW_CAP_PPLBI", source)
            self.assertIn("UMMU_IOCPLBI_VA", source)
            self.assertIn("UMMU_IOCPLBI_ALL", source)
            self.assertNotIn("KVAF_UMMU_LIBRARY", source)
        self.assertIn("KVAF_UMMU_FULL_UAPI_VERIFIED=1", cmake)
        self.assertIn("source_matched_ummu.c", cmake)
        self.assertIn("source_matched_ummu_collections.c", cmake)
        self.assertIn("udma_teardown_contract=checked-retryable-v1", script)
        self.assertIn("unexpectedly links external URMA/UMMU code", script)
        self.assertIn("umdk_libummu_pairing=verified", script)
        self.assertIn("source_layout=official-umdk-libummu-paired-v1", script)

    def test_builder_hashes_the_complete_pinned_uapi_header_tree(self):
        script = (
            ROOT / "scripts" / "build_a5_direct_urma_provider.sh"
        ).read_text(encoding="utf-8")
        roots_start = script.index("libummu_source_roots=(\n")
        roots_end = script.index("\n)\n", roots_start)
        source_roots = script[roots_start:roots_end]

        self.assertIn('"${KVAF_LIBUMMU_ROOT}/kernel_headers"', source_roots)
        self.assertIn('"${KVAF_LIBUMMU_ROOT}/include"', source_roots)
        self.assertIn('"${KVAF_LIBUMMU_ROOT}/ummu_mapt.c"', source_roots)
        self.assertIn("enumerate_source_group", script)
        self.assertIn("libummu-source.sha256", script)
        self.assertIn("libummu_source_manifest_sha256", script)

    def test_uapi_probe_is_published_and_rechecked_before_hardware(self):
        cmake = (PROVIDER / "CMakeLists.txt").read_text(encoding="utf-8")
        builder = (
            ROOT / "scripts" / "build_a5_direct_urma_provider.sh"
        ).read_text(encoding="utf-8")
        verifier = (
            ROOT / "scripts" / "a5_direct_urma_manifest_v3.sh"
        ).read_text(encoding="utf-8")
        probe = (PROVIDER / "official_ummu_uapi_probe.c").read_text(
            encoding="utf-8"
        )

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
            "HW_CAP_FREE_BIT",
            "HW_CAP_PPLBI",
        ):
            self.assertIn(field, probe)
        self.assertIn("ummu_uapi_probe_filename=kvaf_ummu_uapi_probe", builder)
        self.assertIn("ummu_uapi_probe_sha256=", builder)
        self.assertIn('cp "${ummu_uapi_probe}"', builder)
        self.assertIn("kvaf_v3_verify_probe ummu_uapi", verifier)
        self.assertIn('"${prefix}_probe_sha256"', verifier)
        self.assertIn('output=$("${path}")', verifier)

    def test_uapi_compile_entrypoints_define_ioctl_encoding_first(self):
        entrypoints = (
            ROOT / "scripts" / "build_a5_direct_urma_provider.sh",
            PROVIDER / "CMakeLists.txt",
            PROVIDER / "official_ummu_uapi_probe.c",
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
        ummu_source = ROOT.parent / "libummu"
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
