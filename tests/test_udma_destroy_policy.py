import pathlib
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "access_operator" / "a5" / "provider" / "include"


class UdmaDestroyPolicyTests(unittest.TestCase):
    def test_single_shot_event_debt_and_poll_policy_execute(self) -> None:
        compiler = shutil.which("cc") or shutil.which("clang")
        if compiler is None:
            self.skipTest("C compiler is required for the destroy-policy test")

        source = textwrap.dedent(
            r"""
            #include <assert.h>
            #include <errno.h>
            #include <poll.h>
            #include <stdbool.h>
            #include <stdint.h>
            #include "kvaf_udma_destroy_policy.h"

            int main(void) {
              bool kernel_created = true;
              bool delete_pending = false;
              uint32_t async_reported = 0;
              uint32_t async_acked = 0;
              unsigned int ioctl_calls = 0;
              enum kvaf_udma_destroy_decision decision;

              decision = kvaf_udma_destroy_decide(
                  kernel_created, delete_pending, 0, async_reported,
                  0, async_acked, false);
              assert(decision == KVAF_UDMA_DESTROY_ISSUE_IOCTL);
              ++ioctl_calls;
              kernel_created = false;
              delete_pending = true;
              async_reported = 2;

              decision = kvaf_udma_destroy_decide(
                  kernel_created, delete_pending, 0, async_reported,
                  0, async_acked, false);
              assert(decision == KVAF_UDMA_DESTROY_DRAIN_ASYNC);
              decision = kvaf_udma_destroy_decide(
                  kernel_created, delete_pending, 0, async_reported,
                  0, async_acked, true);
              assert(decision == KVAF_UDMA_DESTROY_RETRY_LATER);

              /* A later Close resumes the debt; it never reissues delete. */
              decision = kvaf_udma_destroy_decide(
                  kernel_created, delete_pending, 0, async_reported,
                  0, async_acked, false);
              assert(decision == KVAF_UDMA_DESTROY_DRAIN_ASYNC);
              async_acked = 2;
              decision = kvaf_udma_destroy_decide(
                  kernel_created, delete_pending, 0, async_reported,
                  0, async_acked, true);
              assert(decision == KVAF_UDMA_DESTROY_RELEASE_LOCAL);
              assert(ioctl_calls == 1);

              assert(kvaf_udma_destroy_decide(
                  false, true, 0, 1, 0, 2, false) ==
                  KVAF_UDMA_DESTROY_FAIL_UNSAFE);
              assert(kvaf_udma_destroy_decide(
                  false, true, 1, 0, 0, 0, false) ==
                  KVAF_UDMA_DESTROY_FAIL_UNSAFE);
              assert(kvaf_udma_destroy_decide(
                  true, true, 0, 0, 0, 0, false) ==
                  KVAF_UDMA_DESTROY_FAIL_UNSAFE);

              assert(kvaf_udma_async_poll_decide(0, 0, 0) ==
                  KVAF_UDMA_ASYNC_POLL_IDLE);
              assert(kvaf_udma_async_poll_decide(-1, 0, EINTR) ==
                  KVAF_UDMA_ASYNC_POLL_RETRY_LATER);
              assert(kvaf_udma_async_poll_decide(1, POLLIN, 0) ==
                  KVAF_UDMA_ASYNC_POLL_READ_ONE);
              assert(kvaf_udma_async_poll_decide(1, POLLHUP, 0) ==
                  KVAF_UDMA_ASYNC_POLL_FAIL_UNSAFE);
              assert(kvaf_udma_async_poll_decide(
                  1, (short)(POLLIN | POLLERR), 0) ==
                  KVAF_UDMA_ASYNC_POLL_FAIL_UNSAFE);
              return 0;
            }
            """
        )
        with tempfile.TemporaryDirectory(prefix="kvaf-destroy-policy-") as temp:
            temp_path = pathlib.Path(temp)
            source_path = temp_path / "destroy_policy.c"
            binary_path = temp_path / "destroy_policy"
            source_path.write_text(source, encoding="utf-8")
            subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    f"-I{INCLUDE}",
                    str(source_path),
                    "-o",
                    str(binary_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [str(binary_path)],
                check=True,
                capture_output=True,
                text=True,
            )

    def test_all_sidecars_use_the_executable_policy(self) -> None:
        for kind in ("jetty", "jfr", "jfc"):
            source = (
                ROOT
                / "access_operator"
                / "a5"
                / "provider"
                / f"source_matched_udma_{kind}.c"
            ).read_text(encoding="utf-8")
            self.assertIn("kvaf_udma_destroy_decide(", source)
            self.assertIn("KVAF_UDMA_DESTROY_ISSUE_IOCTL", source)
            self.assertIn("KVAF_UDMA_DESTROY_DRAIN_ASYNC", source)
            self.assertIn("KVAF_UDMA_DESTROY_RETRY_LATER", source)
            self.assertIn("KVAF_UDMA_DESTROY_RELEASE_LOCAL", source)
            self.assertIn("KVAF_UDMA_DESTROY_FAIL_UNSAFE", source)


if __name__ == "__main__":
    unittest.main()
