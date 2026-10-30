"""A device fault or an old binary must not look like a completed probe."""
import unittest

from aiv_batch_get_test_common import expect_scatter_probe


class ScatterProbeResultTest(unittest.TestCase):
    def test_checkpoint_marker(self):
        for stage in ("done", "flush", "setup", "gm", "body", "dma", "ub", "load", "copy"):
            def call():
                verification = {"setup": " configured_bytes=656", "dma": " dma_bytes=656",
                                "load": " verification=gm_ub_byte_compare verified_bytes=656",
                                "gm": " scalar_bytes=656", "ub": " scalar_bytes=656",
                                "body": " body_bytes=640"}.get(stage, "")
                raise RuntimeError(f"[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage={stage} sequence=1 cores=1{verification}")
            expect_scatter_probe(call, stage, expected_bytes=640 if stage == "body" else 656)

    def test_setup_and_dma_require_matching_byte_count(self):
        for stage, field in (("setup", "configured_bytes"), ("dma", "dma_bytes"),
                             ("gm", "scalar_bytes"), ("ub", "scalar_bytes"), ("body", "body_bytes")):
            for count in ("", f" {field}=0", f" {field}=655", " verified_bytes=656"):
                def call():
                    raise RuntimeError(f"[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage={stage} "
                                       f"sequence=1 cores=1{count}")
                with self.subTest(stage=stage, count=count), self.assertRaisesRegex(AssertionError, "byte count"):
                    expect_scatter_probe(call, stage, expected_bytes=656)

    def test_dma_completion_is_not_load_verification(self):
        def call():
            raise RuntimeError("[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=dma sequence=1 cores=1 dma_bytes=656")
        with self.assertRaises(RuntimeError):
            expect_scatter_probe(call, "load", expected_bytes=656)

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
        for stage in ("done", "setup", "dma"):
            for message in ("acl=507015", "CQ timeout", "expected_result=6",
                            "[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=flush sequence=1 cores=1"):
                error = RuntimeError(message)
                def call():
                    raise error
                with self.assertRaises(RuntimeError) as caught:
                    expect_scatter_probe(call, stage)
                self.assertIs(caught.exception, error)

    def test_old_host_cannot_silently_pass(self):
        for stage in ("done", "setup", "dma"):
            with self.assertRaisesRegex(AssertionError, "rebuild Host and kernel"):
                expect_scatter_probe(lambda: None, stage)

    def test_diagnostic_requires_device_acknowledged_revision(self):
        for revision in ("", " diag_revision=0", " diag_revision=2", " diag_revision=10"):
            def call():
                raise RuntimeError("[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=ub "
                                   f"sequence=1 cores=1 scalar_bytes=656{revision}")
            with self.assertRaisesRegex(AssertionError, "diagnostic revision"):
                expect_scatter_probe(call, "ub", expected_bytes=656, diagnostic_revision=1)
        def call():
            raise RuntimeError("[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=ub "
                               "sequence=1 cores=1 scalar_bytes=656 diag_revision=1")
        expect_scatter_probe(call, "ub", expected_bytes=656, diagnostic_revision=1)


if __name__ == "__main__":
    unittest.main()
