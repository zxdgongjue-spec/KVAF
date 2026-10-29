"""A device fault or an old binary must not look like a completed probe."""
import unittest

from aiv_batch_get_test_common import expect_scatter_probe


class ScatterProbeResultTest(unittest.TestCase):
    def test_checkpoint_marker(self):
        for stage in ("done", "flush", "load", "copy"):
            def call():
                verification = " verification=gm_ub_byte_compare verified_bytes=656" if stage == "load" else ""
                raise RuntimeError(f"[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage={stage} sequence=1 cores=1{verification}")
            expect_scatter_probe(call, stage)

    def test_legacy_load_checkpoint_is_rejected(self):
        def call():
            raise RuntimeError("[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=load sequence=1 cores=1")
        with self.assertRaisesRegex(AssertionError, "no byte verification"):
            expect_scatter_probe(call, "load")

    def test_incomplete_load_verification_is_rejected(self):
        for count in ("", " verified_bytes=0", " verified_bytes=655"):
            def call():
                raise RuntimeError("[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=load sequence=1 cores=1 "
                                   f"verification=gm_ub_byte_compare{count}")
            with self.assertRaisesRegex(AssertionError, "verified byte count"):
                expect_scatter_probe(call, "load", expected_bytes=656)

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
