#!/usr/bin/env python3
"""Source-only guard for P9's layer-local canonical-resident Get ABI.

This deliberately needs neither CANN nor an NPU.  It protects the product
boundary before an expensive OPP build: Direct-HCCP can target only the fixed
resident interval, queues become idle only after terminal CQ, and P8 private
forward/replay lifecycle types cannot leak into the P9 helper.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ABI = ROOT / "access_operator/demo/aiv_batch_get/include/batch_get_abi.h"
HELPER = ROOT / "access_operator/demo/aiv_batch_get/kernel/kvaf_layer_resident_get_device.h"
OPERATOR = ROOT / "access_operator/src/access_operator.cc"
PUBLIC = ROOT / "access_operator/include/kvaf/access_operator/access_operator.h"
PYTHON_RUNTIME = ROOT / "bindings/npu_python_runtime.cc"


def _between(text: str, start: str, end: str) -> str:
    return text[text.index(start) : text.index(end, text.index(start))]


def main() -> None:
    abi = ABI.read_text()
    helper = HELPER.read_text()
    operator = OPERATOR.read_text()
    public = PUBLIC.read_text()
    python_runtime = PYTHON_RUNTIME.read_text()

    for token in (
        "KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_WORDS = 16",
        "KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_MAGIC",
        "typedef struct KvafLayerResidentBatchGetControl",
        "typedef struct KvafLayerResidentBatchGetDescriptorScratch",
        "resident_row_base",
        "resident_row_count",
    ):
        assert token in abi, token

    for token in (
        "TryAcquireKvafLayerResidentQueueFromAicore",
        "PublishKvafLayerResidentDescriptor",
        "ValidateKvafLayerResidentOutputIndices",
        "BeginKvafLayerResidentBatchGet",
        "TryRetireKvafLayerResidentBatchGet",
        "FinishKvafLayerResidentBatchGet",
        "ReturnKvafLayerResidentQueueFromAicore",
        "BeginKvafDenseBatchGetPostOnly",
        "TryRetireKvafDenseBatchGet",
        "FinishKvafDenseBatchGet",
        "DataSyncBarrier<AscendC::MemDsbT::DDR>()",
    ):
        assert token in helper, token

    # The P9 boundary is intentionally smaller than P8: no private arena,
    # forward identity, replay identity, or payload-release state belongs in
    # the canonical-resident device helper.
    for forbidden in (
        "KvafP8",
        "private_output",
        "forward_epoch",
        "graph_replay",
        "quarantine",
    ):
        assert forbidden not in helper, forbidden

    validate = _between(
        helper,
        "__aicore__ inline bool ValidateKvafLayerResidentOutputIndices(",
        "__aicore__ inline bool PublishKvafLayerResidentDescriptor(",
    )
    for token in (
        "control.resident_row_base",
        "control.resident_row_count",
        "output_indices[earlier] == row",
    ):
        assert token in validate, token

    finish = _between(
        helper,
        "__aicore__ inline bool FinishKvafLayerResidentBatchGet(",
        "// This releases an SQ/CQ *queue*",
    )
    assert "FinishKvafDenseBatchGet" in finish
    assert "kDenseBatchGetRetired" in finish

    assert "struct LayerResidentBatchGetBinding" in public
    for token in (
        "BindLayerResidentBatchGetControl",
        "ConfigureLayerResidentBatchGetQueueTemplates",
        "ReleaseLayerResidentBatchGetControl",
    ):
        assert token in public, token
        assert token in operator, token

    for token in (
        '"bind_layer_resident_batch_get_control"',
        '"configure_layer_resident_batch_get_queue_templates"',
        '"release_layer_resident_batch_get_control"',
        '"layer_resident_batch_get_control_diagnostic"',
        "layer_resident_batch_get_control_bound_",
    ):
        assert token in python_runtime, token

    bind = _between(
        operator,
        "Status DefaultAccessOperator::BindLayerResidentBatchGetControl(",
        "Status DefaultAccessOperator::ConfigureLayerResidentBatchGetQueueTemplates(",
    )
    for token in (
        "host_control.resident_row_base = binding.resident_row_base",
        "host_control.resident_row_count = binding.resident_row_count",
        "KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_MAGIC",
        "aclrtSynchronizeDevice()",
    ):
        assert token in bind, token

    print("LAYER_RESIDENT_GET_SOURCE_CONTRACT_PASS")


if __name__ == "__main__":
    main()
