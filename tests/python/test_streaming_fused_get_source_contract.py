#!/usr/bin/env python3
"""Source-only contract for the embedded streaming Direct-HCCP helper.

This intentionally runs without an NPU.  It protects the two details that a
normal C++ compiler cannot model: the global C ABI must be exposed through the
fused helper namespace, and AscendC's distinct GM address space forbids C++
aggregate assignment between GM and local records.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "access_operator/demo/aiv_batch_get/kernel/kvaf_fused_full_get_device.h"
ACCESS_OPERATOR = ROOT / "access_operator/src/access_operator.cc"


def _between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def main() -> None:
    text = HELPER.read_text()

    for token in (
        "using ::KvafStreamingBatchGetControl;",
        "using ::KvafStreamingBatchGetDescriptorScratch;",
        "LoadKvafStreamingBatchGetControlFromGmForAicore",
        "LoadKvafStreamingBatchGetControlFromGmForSimt",
        "LoadKvafStreamingBatchGetDescriptorScratchFromGm",
        "StoreKvafStreamingBatchGetDescriptorScratchToGm",
    ):
        assert token in text, token

    # P8 makes the PostOnly -> CQ lifecycle forward-owned.  The record is a
    # fixed four-cache-line entry per transport queue, so it cannot silently
    # turn a candidate burst into an unbounded ticket/ledger allocation.
    for token in (
        "struct KvafP8WaveIdentity",
        "struct KvafP8WaveRecord",
        "struct KvafP8PrivateRecord",
        "struct KvafP8ForwardControl",
        "struct KvafP8WaveDrainClaimResult",
        "static_assert(sizeof(KvafP8WaveRecord) == 256U",
        "static_assert(sizeof(KvafP8PrivateRecord) == KVAF_P8_PRIVATE_RECORD_BYTES",
        "LoadKvafP8ForwardControlFromGm",
        "StoreKvafP8ForwardControlToGm",
        "KvafP8ForwardControlMatchesIdentity",
        "MarkKvafP8ForwardAbort",
        "IsKvafP8PrivateViewReadyForQsfa",
        "kKvafP8WavePostIntent",
        "kKvafP8WavePosted",
        "kKvafP8WaveDrainClaimed",
        "kKvafP8WaveCqObserved",
        "kKvafP8WaveFinalized",
        "kKvafP8WaveQuarantined",
        "TryClaimKvafP8WaveDrainFromAicore",
        "PrepareKvafP8WavePostIntent",
        "PostKvafP8Wave",
        "TryRetireKvafP8Wave",
        "FinishKvafP8Wave",
        "PrepareKvafP8PrivateRecord",
        "FinalizeKvafP8PrivateRecord",
        "ReleaseKvafP8PrivateRecordAfterQsfaTerminal",
        "KvafP8PrivateRecordMatchesIdentity",
        "ValidateKvafP8PrivateOutputIndices",
        "ValidateKvafP8DescriptorPrivateLedger",
        "ValidateKvafP8PrivateTerminalPublication",
        "PublishKvafP8PrivateTerminalRecords",
        "PublishKvafP8BatchGetDescriptorScratch",
        "BuildKvafP8StreamingBatchGetQueueArgs",
        "ReleaseKvafP8WaveAfterPrivateReady",
        "ReleaseKvafP8UnpostedWave",
        "DataSyncBarrier<AscendC::MemDsbT::DDR>()",
    ):
        assert token in text, token

    p8_try_retire = _between(
        text,
        "__aicore__ inline bool TryRetireKvafP8Wave(",
        "// Complete a CQ-observed wave.",
    )
    assert "kCqTimeout" in p8_try_retire
    assert "kKvafP8WaveOutcomeCqTimeoutQuarantined" in p8_try_retire
    assert "record->cq_consumer_after == record->cq_consumer_before" in p8_try_retire
    assert "record->state = kKvafP8WavePosted" in p8_try_retire
    assert "record->reserved[0] = kKvafP8WaveCqObserved" in p8_try_retire

    p8_drain_claim = _between(
        text,
        "__simt_callee__ inline bool TryClaimKvafP8WaveDrainFromSimt(",
        "// P8 phase helpers",
    )
    for token in (
        "asc_atomic_cas(",
        "kKvafP8WavePosted",
        "kKvafP8WaveDrainClaimed",
        "TryClaimKvafP8WaveDrainFromAicore",
        "VF_CALL<KvafP8WaveDrainClaimSimtService>",
    ):
        assert token in p8_drain_claim, token

    p8_release = _between(
        text,
        "__aicore__ inline bool ReleaseKvafP8WaveAfterPrivateReady(",
        "// Only a *proved unposted* wave",
    )
    for token in (
        "record->private_ready_published == 0U",
        "kKvafP8WaveOutcomeReleaseCasQuarantined",
        "ValidateKvafP8PrivateTerminalPublication",
        "ReleaseKvafStreamingBatchGetQueueFromAicore",
    ):
        assert token in p8_release, token

    # Queue/descriptors are eligible for reuse after terminal CQ plus private
    # READY/FAILED publication.  The payload itself remains protected by a
    # separate fixed private record until the same-stream QSFA terminal; the
    # target design must never pin a scarce HCCP queue for the whole QSFA
    # lifetime.
    p8_private = _between(
        text,
        "__aicore__ inline bool PrepareKvafP8PrivateRecord(",
        "// AscendC treats GM and local memory",
    )
    for token in (
        "kKvafP8PrivatePending",
        "kKvafP8PrivateReady",
        "kKvafP8PrivateFailed",
        "KvafP8PrivateRecordMatchesWave",
        "ReleaseKvafP8PrivateRecordAfterQsfaTerminal",
        "private_record->state = kKvafP8PrivateQuarantined",
    ):
        assert token in p8_private, token

    # P8 must not weaken the existing streaming range guard globally.  Legacy
    # callers remain queue-private; only the P8 helper may select an absolute
    # output row after proving that the row's durable private record is owned
    # by this exact forward/replay/layer/queue/generation/post identity.
    legacy_build = _between(
        text,
        "__aicore__ inline bool BuildKvafStreamingBatchGetQueueArgs(",
        "// P8-only equivalent of the legacy queue-argument builder.",
    )
    assert "ValidateKvafStreamingBatchGetPrivateOutputIndices" in legacy_build

    p8_validate = _between(
        text,
        "__aicore__ inline bool ValidateKvafP8PrivateOutputIndices(",
        "// P8-only descriptor publication companion.",
    )
    for token in (
        "private_record_count != control.output_rows",
        "static_cast<uint64_t>(output_index) >= control.output_rows",
        "private_record.state != kKvafP8PrivatePending",
        "private_record.output_index != output_index",
        "KvafP8PrivateRecordMatchesIdentity(private_record, identity)",
        "for (uint32_t prior = 0U; prior < request; ++prior)",
    ):
        assert token in p8_validate, token
    assert "private_output_stride_bytes" not in p8_validate
    assert "private_output_row_base" not in p8_validate

    p8_publish = _between(
        text,
        "__aicore__ inline bool PublishKvafP8BatchGetDescriptorScratch(",
        "// Build the existing single-queue Direct-HCCP view",
    )
    for token in (
        "ValidateKvafP8PrivateOutputIndices",
        "StoreKvafStreamingBatchGetDescriptorScratchToGm",
        "DataSyncBarrier<AscendC::MemDsbT::DDR>()",
    ):
        assert token in p8_publish, token

    p8_build = _between(
        text,
        "__aicore__ inline bool BuildKvafP8StreamingBatchGetQueueArgs(",
        "// Begin a previously acquired lease.",
    )
    assert "ValidateKvafP8PrivateOutputIndices" in p8_build
    assert "ValidateKvafStreamingBatchGetPrivateOutputIndices" not in p8_build

    p8_ledger = _between(
        text,
        "__aicore__ inline bool ValidateKvafP8DescriptorPrivateLedger(",
        "// The only P8 private-terminal publisher.",
    )
    for token in (
        "request_source_offsets",
        "request_object_offsets",
        "request_output_indices",
        "private_record.source_offset",
        "private_record.object_offset",
        "KvafP8PrivateRecordMatchesIdentity",
    ):
        assert token in p8_ledger, token

    # The durable P8 phases must all rebuild Direct-HCCP arguments through
    # the P8 validator.  Falling back to the legacy builder at even one phase
    # would either reject a legitimate globally claimed row or silently make
    # descriptor ownership inconsistent after a graph replay.
    for start, end in (
        (
            "__aicore__ inline bool PrepareKvafP8WavePostIntent(",
            "// Execute the one irreversible step.",
        ),
        (
            "__aicore__ inline bool PostKvafP8Wave(",
            "// Probe without spinning.",
        ),
        (
            "__aicore__ inline bool TryRetireKvafP8Wave(",
            "// Complete a CQ-observed wave.",
        ),
        (
            "__aicore__ inline bool FinishKvafP8Wave(",
            "// This is the only normal release path",
        ),
    ):
        phase = _between(text, start, end)
        assert "BuildKvafP8StreamingBatchGetQueueArgs" in phase
        assert "BuildKvafStreamingBatchGetQueueArgs" not in phase

    p8_prepare = _between(
        text,
        "__aicore__ inline bool PrepareKvafP8WavePostIntent(",
        "// Execute the one irreversible step.",
    )
    for token in (
        "KvafP8ForwardControl",
        "request_source_offsets",
        "request_object_offsets",
        "request_output_indices",
        "ValidateKvafP8DescriptorPrivateLedger",
        "kKvafP8ForwardPhaseAdmitPost",
    ):
        assert token in p8_prepare, token

    p8_unposted = _between(
        text,
        "__aicore__ inline bool ReleaseKvafP8UnpostedWave(",
        "}  // namespace kvaf_fused_get",
    )
    assert "record->state == kKvafP8WavePostIntent" in p8_unposted
    assert "kKvafP8WaveOutcomePostIntentQuarantined" in p8_unposted

    # These four copies were valid in the standalone compilation context but
    # invalid when the same helper is embedded in an AscendC mixed operator.
    # Field-wise conversion preserves the exact ABI values and ordering.
    for forbidden in (
        "const KvafStreamingBatchGetControl control = *control_gm;",
        "*descriptor = *source;",
        "*scratch = descriptor;",
        "control, queue_id, *scratch))",
    ):
        assert forbidden not in text, forbidden

    for field in (
        "magic",
        "descriptor_template_count",
        "address_table",
        "private_output_row_base",
    ):
        assert f"destination->{field}" in text, field

    # A terminal Direct-HCCP CQ does not make a queue reusable by itself: the
    # owner-table CAS is the final ownership proof.  The helper must retain
    # the nonzero owner word when the CAS fails, and the cold Runtime teardown
    # must reject that table rather than releasing HCCP resources underneath
    # it.  This is source-only because injecting a CAS mismatch into a real
    # HCCP queue would intentionally corrupt the ownership protocol.
    release = _between(
        text,
        "__simt_callee__ inline bool ReleaseKvafStreamingBatchGetQueueFromSimt(",
        "__simt_vf__ __aicore__ LAUNCH_BOUND(1)\ninline void KvafStreamingBatchGetReleaseSimtService",
    )
    assert "asc_atomic_cas(owners + lease->queue_id, lease->owner_tag, 0)" in release
    assert "return false;" in release
    assert "lease->owner_tag = 0;" in release
    assert release.index("return false;") < release.index("lease->owner_tag = 0;")

    access_operator = ACCESS_OPERATOR.read_text()
    teardown = _between(
        access_operator,
        "Status DefaultAccessOperator::ReleaseStreamingBatchGetControl()",
        "Status DefaultAccessOperator::DescribeStreamingBatchGetControl(",
    )
    for token in (
        "std::vector<std::int32_t> lease_snapshot",
        "aclrtSynchronizeDevice()",
        "ACL_MEMCPY_DEVICE_TO_HOST",
        "streaming BatchGet control teardown is quarantined",
        "StatusCode::kBusy",
    ):
        assert token in teardown, token
    assert teardown.index("lease_snapshot") < teardown.index(
        "KvafBatchGetLocalBufferRelease"
    )
    assert teardown.index("aclrtSynchronizeDevice()") < teardown.index(
        "ACL_MEMCPY_DEVICE_TO_HOST"
    )

    print(
        "KVAF_STREAMING_FUSED_GET_SOURCE_CONTRACT_PASS "
        "global_abi_aliases=1 gm_local_fieldwise_copy=1 "
        "direct_hccp_abi_layout=unchanged "
        "terminal_release_cas_quarantine=1 "
        "p8_persistent_wave_state_machine=1 "
        "p8_global_private_slots=1"
    )


if __name__ == "__main__":
    main()
