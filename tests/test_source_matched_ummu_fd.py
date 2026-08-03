import pathlib
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "access_operator" / "a5" / "provider"
FD_SOURCE = PROVIDER / "source_matched_ummu_fd.c"


class SourceMatchedUmmuFdTests(unittest.TestCase):
    def test_process_device_fd_duplicate_is_owned_unique_and_fail_closed(self):
        compiler = shutil.which("cc") or shutil.which("clang")
        if compiler is None:
            self.skipTest("no C compiler")
        fd_directory = pathlib.Path("/proc/self/fd")
        if not fd_directory.is_dir():
            fd_directory = pathlib.Path("/dev/fd")
        if not fd_directory.is_dir():
            self.skipTest("no process descriptor directory")

        harness = textwrap.dedent(
            r"""
            #define _GNU_SOURCE
            #include <errno.h>
            #include <fcntl.h>
            #include <stdio.h>
            #include <stdlib.h>
            #include <string.h>
            #include <unistd.h>

            #include "kvaf_source_matched_ummu_fd.h"

            #ifndef ENOTUNIQ
            #define ENOTUNIQ 76
            #endif

            static int unique_rw(const char *fd_dir) {
              int original = open("/dev/null", O_RDWR | O_CLOEXEC);
              int duplicate;
              int original_flags;
              if (original < 0) return 1;
              errno = EEXIST;
              duplicate = kvaf_source_ummu_duplicate_unique_process_fd(
                  "/dev/null", fd_dir);
              if (duplicate < 0 || duplicate == original || errno != 0) {
                return 2;
              }
              if ((fcntl(duplicate, F_GETFD) & FD_CLOEXEC) == 0) return 3;
              if ((fcntl(duplicate, F_GETFL) & O_ACCMODE) != O_RDWR) return 4;
              original_flags = fcntl(original, F_GETFL);
              if (original_flags < 0 ||
                  fcntl(original, F_SETFL, original_flags | O_NONBLOCK) != 0 ||
                  (fcntl(duplicate, F_GETFL) & O_NONBLOCK) == 0) return 5;
              if (close(duplicate) != 0 || fcntl(original, F_GETFD) < 0) {
                return 6;
              }
              return close(original) == 0 ? 0 : 7;
            }

            static int reject_read_only(const char *fd_dir) {
              int original = open("/dev/zero", O_RDONLY | O_CLOEXEC);
              int duplicate;
              if (original < 0) return 10;
              duplicate = kvaf_source_ummu_duplicate_unique_process_fd(
                  "/dev/zero", fd_dir);
              if (close(original) != 0) return 11;
              return duplicate == -ENOENT ? 0 : 12;
            }

            static int reject_ambiguous(const char *fd_dir) {
              int first = open("/dev/null", O_RDWR | O_CLOEXEC);
              int second = open("/dev/null", O_RDWR | O_CLOEXEC);
              int duplicate;
              if (first < 0 || second < 0) return 20;
              duplicate = kvaf_source_ummu_duplicate_unique_process_fd(
                  "/dev/null", fd_dir);
              if (close(second) != 0 || close(first) != 0) return 21;
              return duplicate == -ENOTUNIQ ? 0 : 22;
            }

            static int reject_regular_file(const char *fd_dir) {
              char path[] = "/tmp/kvaf-ummu-fd-XXXXXX";
              int original = mkstemp(path);
              int duplicate;
              if (original < 0) return 30;
              duplicate = kvaf_source_ummu_duplicate_unique_process_fd(
                  path, fd_dir);
              (void)unlink(path);
              if (close(original) != 0) return 31;
              return duplicate == -ENODEV ? 0 : 32;
            }

            static int reject_null(const char *fd_dir) {
              int result;
              errno = EACCES;
              result = kvaf_source_ummu_duplicate_unique_process_fd(
                  NULL, fd_dir);
              return result == -EINVAL ? 0 : 35;
            }

            int main(int argc, char **argv) {
              if (argc != 3) return 40;
              if (strcmp(argv[1], "unique-rw") == 0) return unique_rw(argv[2]);
              if (strcmp(argv[1], "read-only") == 0) {
                return reject_read_only(argv[2]);
              }
              if (strcmp(argv[1], "ambiguous") == 0) {
                return reject_ambiguous(argv[2]);
              }
              if (strcmp(argv[1], "regular") == 0) {
                return reject_regular_file(argv[2]);
              }
              if (strcmp(argv[1], "null") == 0) return reject_null(argv[2]);
              return 41;
            }
            """
        )
        with tempfile.TemporaryDirectory(prefix="kvaf-ummu-fd-") as temp:
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
                    str(FD_SOURCE),
                    str(harness_path),
                    "-o",
                    str(binary_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            for scenario in (
                "unique-rw",
                "read-only",
                "ambiguous",
                "regular",
                "null",
            ):
                with self.subTest(scenario=scenario):
                    subprocess.run(
                        [str(binary_path), scenario, str(fd_directory)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

    def test_context_reuse_is_strictly_gated_by_eexist(self):
        runtime = (PROVIDER / "source_matched_ummu.c").read_text(
            encoding="utf-8"
        )
        cmake = (PROVIDER / "CMakeLists.txt").read_text(encoding="utf-8")
        builder = (
            ROOT / "scripts" / "build_a5_direct_urma_provider.sh"
        ).read_text(encoding="utf-8")
        verifier = (
            ROOT / "scripts" / "a5_direct_urma_manifest_v3.sh"
        ).read_text(encoding="utf-8")
        token = (PROVIDER / "source_matched_udma_tid.c").read_text(
            encoding="utf-8"
        )

        ensure_start = runtime.index("static int kvaf_ensure_context(void)")
        ensure_end = runtime.index(
            "\nstatic void kvaf_latch_mapping_ownership_unknown", ensure_start
        )
        ensure = runtime[ensure_start:ensure_end]
        self.assertIn("if (open_error == EEXIST)", ensure)
        self.assertIn(
            "kvaf_source_ummu_duplicate_unique_process_fd(", ensure
        )
        self.assertIn('TID_DEVICE_NAME, "/proc/self/fd"', ensure)
        self.assertIn("KVAF_SOURCE_UMMU_FD_PROCESS_DUP", ensure)
        self.assertIn("KVAF_SOURCE_UMMU_FD_PRIVATE_OPEN", ensure)
        self.assertIn(
            '"KVAF_UMMU_CONTEXT_READY fd_mode=%s owner_pid=%ld', ensure
        )
        self.assertIn("errno = 0;", ensure)
        self.assertIn("source_matched_ummu_fd.c", cmake)
        self.assertIn("errno = operation_errno;", token)
        helper = FD_SOURCE.read_text(encoding="utf-8")
        self.assertIn("scan_error = entry == NULL ? errno : 0;", helper)
        self.assertIn("if (scan_error != 0)", helper)
        self.assertIn("(void)close(adopted_fd);", helper)
        self.assertIn(
            "ummu_context_fd_contract=private-open-or-unique-process-dup-v1",
            builder,
        )
        self.assertIn(
            "ummu_context_fd_contract:private-open-or-unique-process-dup-v1",
            verifier,
        )


if __name__ == "__main__":
    unittest.main()
