#!/usr/bin/env python3
"""Static contract for the MTP forward-resident Direct-HCCP cold API.

This test deliberately has no CANN/NPU dependency.  It protects the one
integration boundary that must not drift while the MTP graph-side operator is
developed elsewhere: the deployed 112-byte BatchGet ABI stays compact-prefix
based (`miss_count`, `token_bytes`, `tokens_per_block`) and the Python API
does not reintroduce a host/GIL Post path.
"""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ABI = ROOT / "access_operator/demo/aiv_batch_get/include/batch_get_abi.h"
ACCESS = ROOT / "access_operator/src/access_operator.cc"
BINDING = ROOT / "bindings/src/npu_python_runtime.cc"
DEVICE_EXECUTOR = (
    ROOT
    / "access_operator/demo/aiv_batch_get/kernel"
    / "kvaf_mtp_layer_resident_get_device.h"
)


class MtpLayerResidentBatchGetContractTest(unittest.TestCase):
    def test_mtp_control_keeps_the_deployed_112_byte_get_abi(self) -> None:
        text = ABI.read_text()
        self.assertIn("sizeof(KvafExecuteBatchGetKernelArgs) == 112", text)
        self.assertIn("uint64_t miss_count;", text)
        self.assertIn("uint32_t token_bytes;", text)
        self.assertIn("uint32_t tokens_per_block;", text)
        self.assertIn("KvafLayerResidentBatchGetControl", text)
        self.assertIn("KvafLayerResidentBatchGetDescriptorScratch", text)
        descriptor = text.split(
            "typedef struct KvafLayerResidentBatchGetDescriptorScratch", 1
        )[1].split("} KvafLayerResidentBatchGetDescriptorScratch", 1)[0]
        self.assertIn("uint64_t miss_count;", descriptor)
        self.assertNotIn("mask", descriptor)
        self.assertNotIn("block_bytes", descriptor)
        control = text.split("typedef struct KvafLayerResidentBatchGetControl", 1)[
            1
        ].split("} KvafLayerResidentBatchGetControl", 1)[0]
        self.assertEqual(control.count("uint64_t "), 16)
        for field in (
            "channel_entity",
            "queue_state",
            "core_status",
            "output",
            "queue_leases",
            "descriptor_templates",
            "queue_count",
            "output_rows",
            "resident_row_base",
            "resident_row_count",
            "jettys_per_core",
            "token_bytes",
            "tokens_per_block",
            "descriptor_template_count",
        ):
            self.assertIn(field, control)

    def test_cold_api_is_explicit_and_blocks_eager_queue_races(self) -> None:
        text = ACCESS.read_text()
        for symbol in (
            "BindLayerResidentBatchGetControl",
            "ConfigureLayerResidentBatchGetQueueTemplates",
            "ReleaseLayerResidentBatchGetControl",
            "DescribeLayerResidentBatchGetControl",
            "ordinary BatchGet is forbidden while layer-resident control owns",
            "must finish before release",
        ):
            self.assertIn(symbol, text)
        self.assertIn("source.miss_count.bytes != sizeof(std::int32_t)", text)
        self.assertIn("aclrtSynchronizeDevice()", text)
        self.assertNotIn("source.mask", text)

    def test_python_surface_uses_miss_counts_not_masks(self) -> None:
        text = BINDING.read_text()
        self.assertIn('"bind_layer_resident_batch_get_control"', text)
        self.assertIn(
            '"configure_layer_resident_batch_get_queue_templates"', text
        )
        self.assertIn('py::arg("miss_counts")', text)
        self.assertIn('"release_layer_resident_batch_get_control"', text)
        start = text.index("ConfigureLayerResidentBatchGetQueueTemplates")
        end = text.index("void ReleaseLayerResidentBatchGetControl", start)
        self.assertIn("miss_count_view.bytes != sizeof(std::int32_t)", text[start:end])
        self.assertNotIn("mask_view", text[start:end])
        self.assertNotIn("py::sequence& masks", text[start:end])

    def test_device_executor_is_mtp_post_only_and_not_a_c8_hook(self) -> None:
        text = DEVICE_EXECUTOR.read_text()
        for symbol in (
            "TryAcquireKvafMtpLayerResidentQueueFromAicore",
            "AppendKvafMtpLayerResidentRequest",
            "PublishKvafMtpLayerResidentDescriptor",
            "BeginKvafMtpLayerResidentBatchGetPostOnly",
            "TryRetireKvafMtpLayerResidentBatchGet",
            "FinishKvafMtpLayerResidentBatchGet",
            "ReturnKvafMtpLayerResidentQueueFromAicore",
            "DataSyncBarrier<AscendC::MemDsbT::DDR>",
        ):
            self.assertIn(symbol, text)
        self.assertIn("miss_count", text)
        self.assertIn("token_bytes", text)
        self.assertIn("tokens_per_block", text)
        self.assertNotIn("block_bytes", text)
        self.assertNotIn("P10", text)
        self.assertIn("not an\n// include-time hook", text)


if __name__ == "__main__":
    unittest.main()
