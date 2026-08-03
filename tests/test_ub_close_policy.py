import pathlib
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "access_operator" / "a5" / "provider" / "include"


class UbClosePolicyTests(unittest.TestCase):
    def test_send_accounting_and_retry_policy_execute(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler is required for the close-policy test")

        source = textwrap.dedent(
            r"""
            #include <assert.h>
            #include <errno.h>
            #include <stdint.h>
            #include "kvaf_ub_close_policy.h"

            int main(void) {
              struct kvaf_ub_send_tracker tracker;

              kvaf_ub_send_tracker_reset(&tracker);
              assert(kvaf_ub_send_tracker_is_quiesced(&tracker));
              /* JFC/JFR-only partial create: no send Jetty, no possible
               * flush barrier, and no WQE can be owned. */
              assert(kvaf_ub_send_tracker_close_is_quiesced(&tracker));

              kvaf_ub_send_tracker_require_flush_barrier(&tracker);
              assert(tracker.flush_barrier_required);
              assert(!kvaf_ub_send_tracker_close_is_quiesced(&tracker));

              kvaf_ub_send_tracker_record_post(&tracker, true);
              assert(tracker.inflight == 1U);
              assert(!kvaf_ub_send_tracker_is_quiesced(&tracker));
              kvaf_ub_send_tracker_record_completion(&tracker, false);
              assert(kvaf_ub_send_tracker_is_quiesced(&tracker));
              assert(!kvaf_ub_send_tracker_close_is_quiesced(&tracker));

              kvaf_ub_send_tracker_mark_post_unknown(&tracker);
              assert(!kvaf_ub_send_tracker_is_quiesced(&tracker));
              kvaf_ub_send_tracker_record_completion(&tracker, true);
              assert(kvaf_ub_send_tracker_is_quiesced(&tracker));
              assert(kvaf_ub_send_tracker_close_is_quiesced(&tracker));

              tracker.inflight = UINT32_MAX;
              tracker.accounting_unknown = false;
              kvaf_ub_send_tracker_record_post(&tracker, true);
              assert(tracker.accounting_unknown);

              assert(kvaf_ub_close_retry_is_proven(
                  EINTR, 0, KVAF_UB_CLOSE_RETRY_CONTROL,
                  false, false, false));
              assert(!kvaf_ub_close_retry_is_proven(
                  EINTR, 0, KVAF_UB_CLOSE_RETRY_CHECKED_DESTROY,
                  false, false, false));
              assert(kvaf_ub_close_retry_is_proven(
                  EAGAIN, 0, KVAF_UB_CLOSE_RETRY_CHECKED_DESTROY,
                  false, false, false));
              assert(kvaf_ub_close_retry_is_proven(
                  4096, 0, KVAF_UB_CLOSE_RETRY_CHECKED_DESTROY,
                  true, false, false));
              assert(!kvaf_ub_close_retry_is_proven(
                  EAGAIN, 0, KVAF_UB_CLOSE_RETRY_CONTROL,
                  true, true, false));
              assert(!kvaf_ub_close_retry_is_proven(
                  EAGAIN, 0, KVAF_UB_CLOSE_RETRY_CONTROL,
                  true, false, true));
              assert(!kvaf_ub_close_retry_is_proven(
                  EAGAIN, 0, KVAF_UB_CLOSE_RETRY_NONE,
                  true, false, false));
              return 0;
            }
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            source_path = directory_path / "ub_close_policy_test.c"
            binary_path = directory_path / "ub_close_policy_test"
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


if __name__ == "__main__":
    unittest.main()
