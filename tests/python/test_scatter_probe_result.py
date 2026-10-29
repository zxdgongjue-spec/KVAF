"""A device fault or an old binary must not look like a completed probe."""
import unittest

from aiv_batch_get_test_common import expect_scatter_probe


class ScatterProbeResultTest(unittest.TestCase):
    def test_checkpoint_marker(self):
        for stage in ("done", "flush", "load", "copy"):
            def call():
                raise RuntimeError(f"[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage={stage} sequence=1 cores=1")
            expect_scatter_probe(call, stage)

    def test_faults_are_preserved(self):
        for message in ("acl=507015", "CQ timeout", "expected_result=6",
                        "[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=flush sequence=1 cores=1"):
            error = RuntimeError(message)
            def call():
                raise error
            with self.assertRaises(RuntimeError) as caught:
                expect_scatter_probe(call, "done")
            self.assertIs(caught.exception, error)

    def test_old_host_cannot_silently_pass(self):
        with self.assertRaisesRegex(AssertionError, "rebuild Host and kernel"):
            expect_scatter_probe(lambda: None, "done")


if __name__ == "__main__":
    unittest.main()
