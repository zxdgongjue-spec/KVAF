#!/usr/bin/env python3
"""Source-only contract for P10's forward-owned resident Get control.

The target path is intentionally smaller than the prior streaming experiments:
one forward owns one canonical resident output arena, while queues are only
temporary Direct-HCCP transport leases.  This check runs without an NPU and
protects that boundary before an expensive CANN compile.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ABI = ROOT / "access_operator/demo/aiv_batch_get/include/batch_get_abi.h"
DEVICE = (
    ROOT
    / "access_operator/demo/aiv_batch_get/kernel/kvaf_forward_resident_get_device.h"
)
HOST = ROOT / "access_operator/src/access_operator.cc"
PUBLIC = ROOT / "access_operator/include/kvaf/access_operator/access_operator.h"


def _between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def main() -> None:
    abi = ABI.read_text()
    device = DEVICE.read_text()
    host = HOST.read_text()
    public = PUBLIC.read_text()

    for token in (
        "KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_WORDS = 16",
        "KvafForwardResidentBatchGetDescriptorScratch",
        "KvafForwardResidentBatchGetControl",
        "resident_row_base",
        "resident_row_count",
        "descriptor_template_count",
    ):
        assert token in abi, token

    for token in (
        "TryAcquireKvafForwardResidentQueueFromSimt",
        "asc_atomic_cas(owners + queue_id, 0, owner_tag)",
        "producer_id` is a stable executor work-item",
        "ValidateKvafForwardResidentOutputIndices",
        "AppendKvafForwardResidentRequest",
        "PublishKvafForwardResidentDescriptor",
        "DataSyncBarrier<AscendC::MemDsbT::DDR>()",
        "BeginKvafDenseBatchGetPostOnly",
        "TryRetireKvafDenseBatchGet",
        "FinishKvafDenseBatchGet",
        "ReturnKvafForwardResidentUnpostedQueueFromAicore",
        "ReturnKvafForwardResidentQueueFromAicore",
    ):
        assert token in device, token

    validation = _between(
        device,
        "__aicore__ inline bool ValidateKvafForwardResidentOutputIndices(",
        "// Append translates one already-admitted KV object",
    )
    assert "resident_row_base" in validation
    assert "resident_end" in validation
    assert "output_indices[earlier] == row" in validation

    queue_args = _between(
        device,
        "__aicore__ inline bool BuildKvafForwardResidentQueueArgs(",
        "// These wrappers intentionally contain no alternative HCCP protocol.",
    )
    assert "args->output = control.output" in queue_args
    assert "private_output" not in queue_args

    begin = _between(
        device,
        "__aicore__ inline bool BeginKvafForwardResidentBatchGet(",
        "__aicore__ inline void TryRetireKvafForwardResidentBatchGet(",
    )
    # The descriptor publication is not a caller-optional helper.  Begin
    # receives the local wave descriptor and must publish it (including the
    # DDR barrier in the helper) before it reloads queue arguments and can
    # invoke the established PostOnly primitive.
    assert "const KvafForwardResidentBatchGetDescriptorScratch& descriptor" in begin
    assert "PublishKvafForwardResidentDescriptor(control, lease, descriptor)" in begin
    assert begin.index("PublishKvafForwardResidentDescriptor(") < begin.index(
        "BuildKvafForwardResidentQueueArgs("
    )

    # The P10 device header must not inherit an old fixed output partition or
    # cross-forward recovery protocol.  It is acceptable for the repository
    # to retain legacy types elsewhere; this is a precise target-path guard.
    for forbidden in (
        "private_output_stride",
        "private_output_row_base",
        "forward_epoch",
        "graph_replay",
        "quarantine",
        "KvafP8",
    ):
        assert forbidden not in device, forbidden

    assert (
        "using ForwardResidentBatchGetBinding = LayerResidentBatchGetBinding;"
        in public
    )
    for token in (
        "BindForwardResidentBatchGetControl",
        "ConfigureForwardResidentBatchGetQueueTemplates",
        "ReleaseForwardResidentBatchGetControl",
        "DescribeForwardResidentBatchGetControl",
    ):
        assert token in public, token
        assert token in host, token

    host_bind = _between(
        host,
        "Status DefaultAccessOperator::BindForwardResidentBatchGetControl(",
        "Status DefaultAccessOperator::ConfigureForwardResidentBatchGetQueueTemplates(",
    )
    assert (
        "return BindLayerResidentBatchGetControl(binding, queue_leases, queue_count);"
        in host_bind
    )
    # Forward-resident deliberately aliases the hardened layer-resident raw
    # registration path.  This keeps P10's Begin-owned descriptor publication
    # while retaining the latest output-range validation and rollback fixes.
    layer_bind = _between(
        host,
        "Status DefaultAccessOperator::BindLayerResidentBatchGetControl(",
        "Status DefaultAccessOperator::ConfigureLayerResidentBatchGetQueueTemplates(",
    )
    for token in (
        "KvafBatchGetLocalBufferAcquire",
        "KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_MAGIC",
        "host_control.output = PointerValue(binding.output.data)",
        "host_control.resident_row_base = binding.resident_row_base",
        "host_control.resident_row_count = binding.resident_row_count",
        "host_control.queue_leases = PointerValue(queue_leases.data)",
    ):
        assert token in layer_bind, token
    assert "private_output_stride" not in layer_bind
    assert "bind_epoch" not in layer_bind

    host_release = _between(
        host,
        "Status DefaultAccessOperator::ReleaseLayerResidentBatchGetControl()",
        "Status DefaultAccessOperator::DescribeLayerResidentBatchGetControl(",
    )
    assert "aclrtSynchronizeDevice" in host_release
    assert "owners[queue_id] != 0" in host_release
    assert "KvafBatchGetLocalBufferRelease" in host_release

    print("FORWARD_RESIDENT_SOURCE_CONTRACT_PASS")


if __name__ == "__main__":
    main()
