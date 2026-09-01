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
    assert "record.state <= kKvafP8WaveFillingClaimed" in text

    # P8 makes the PostOnly -> CQ lifecycle forward-owned.  The record is a
    # fixed four-cache-line entry per transport queue, so it cannot silently
    # turn a candidate burst into an unbounded ticket/ledger allocation.
    for token in (
        "struct KvafP8WaveIdentity",
        "struct KvafP8WaveRecord",
        "struct KvafP8PrivateRecord",
        "struct KvafP8ForwardControl",
        "struct KvafP8WaveDrainClaimResult",
        "struct KvafP8ForwardAbortResult",
        "struct KvafP8PrivateReleaseResult",
        "struct KvafP8ForwardTerminalResult",
        "static_assert(sizeof(KvafP8WaveRecord) == 256U",
        "static_assert(sizeof(KvafP8PrivateRecord) == KVAF_P8_PRIVATE_RECORD_BYTES",
        "LoadKvafP8ForwardControlFromGm",
        "StoreKvafP8ForwardControlToGm",
        "KvafP8ForwardControlMatchesIdentity",
        "MarkKvafP8ForwardAbort",
        "MarkKvafP8ForwardAbortFromSimt",
        "MarkKvafP8ForwardAbortFromAicore",
        "IsKvafP8PrivateViewReadyForQsfa",
        "kKvafP8WaveFilling",
        "kKvafP8WaveFillingReleaseClaimed",
        "kKvafP8WaveFillingClaimed",
        "kKvafP8WavePostIntent",
        "kKvafP8WavePosted",
        "kKvafP8WaveDrainClaimed",
        "kKvafP8WaveCqObserved",
        "kKvafP8WaveFinalized",
        "kKvafP8WaveQuarantined",
        "TryClaimKvafP8WaveDrainFromAicore",
        "TryClaimKvafP8UnpostedWaveDrainFromAicore",
        "BeginKvafP8WaveFilling",
        "BeginAndPublishKvafP8WaveFilling",
        "IsKvafP8WaveRecordAllZero",
        "IsKvafP8WaveRecordAdmissibleForFilling",
        "PublishKvafP8FillingPostIntentPayloadToGm",
        "TryTransitionKvafP8FillingStateFromSimt",
        "TryTransitionKvafP8FillingStateFromAicore",
        "TryClaimKvafP8WaveFillingAdmissionFromAicore",
        "QuarantineKvafP8WaveFillingClaimedFromAicore",
        "PublishKvafP8WaveFillingClaimedPayloadToGm",
        "CommitKvafP8WaveMutablePayloadAndStateFromAicore",
        "TryCommitKvafP8WaveStateLastFromSimt",
        "CommitKvafP8ClaimedUnpostedWaveReleaseFromAicore",
        "TryCommitKvafP8ClaimedUnpostedWaveReleaseFromSimt",
        "QuarantineKvafP8FillingStateFromAicore",
        "QuarantineKvafP8PostIntentStateFromAicore",
        "ArmKvafP8WaveFillingForExternalWrites",
        "PromoteKvafP8WaveFillingToPostIntent",
        "ReleaseKvafP8FillingLeaseAfterProvenNoWork",
        "IsKvafP8WaveFillingProvenNoWork",
        "PrepareKvafP8WavePostIntent",
        "PostKvafP8Wave",
        "TryRetireKvafP8Wave",
        "FinishKvafP8Wave",
        "PrepareKvafP8PrivateRecord",
        "FinalizeKvafP8PrivateRecord",
        "ReleaseKvafP8PrivateRecordAfterQsfaTerminal",
        "ReleaseKvafP8PendingPrivateRecordAfterNoPost",
        "TryClaimKvafP8PrivateNoPostReleaseFromAicore",
        "QuarantineKvafP8PrivateReleaseClaimedFromAicore",
        "TryClaimKvafP8PrivateReleaseFromSimt",
        "TryClaimKvafP8PrivateReleaseFromAicore",
        "ClearKvafP8PrivateMappingsFromSimt",
        "ClearKvafP8PrivateMappingsFromAicore",
        "CommitKvafP8PrivateReleaseFromSimt",
        "CommitKvafP8PrivateReleaseFromAicore",
        "QuarantineKvafP8WaveAtForwardTerminal",
        "QuarantineKvafP8PrivateAtForwardTerminal",
        "KvafP8PrivateRecordMatchesIdentity",
        "ValidateKvafP8PrivateOutputIndices",
        "ValidateKvafP8DescriptorPrivateLedger",
        "ValidateKvafP8PrivateTerminalPublication",
        "PublishKvafP8PrivateTerminalRecords",
        "PublishKvafP8BatchGetDescriptorScratch",
        "BuildKvafP8StreamingBatchGetQueueArgs",
        "ReleaseKvafP8WaveAfterPrivateReady",
        "ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup",
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
    assert "static_cast<int32_t>(kKvafP8WaveFilling)" not in p8_drain_claim
    assert (
        "static_cast<int32_t>(kKvafP8WaveFillingReleaseClaimed)"
        not in p8_drain_claim
    )

    # A pre-doorbell state transition cannot use a normal GM Store: terminal
    # recovery may have already made the record QUARANTINED.  The allowed
    # edges are deliberately enumerated in a SIMT CAS endpoint.
    p8_filling_transition = _between(
        text,
        "__simt_callee__ inline bool TryTransitionKvafP8FillingStateFromSimt(",
        "// A deterministic Begin/Post rejection",
    )
    for token in (
        "asc_atomic_cas(",
        "kKvafP8WaveFillingReleaseClaimed",
        "kKvafP8WaveFillingClaimed",
        "kKvafP8WavePostIntent",
        "kKvafP8WaveQuarantined",
        "kKvafP8WaveFree",
        "VF_CALL<KvafP8FillingStateTransitionSimtService>",
        "TryTransitionKvafP8FillingStateFromAicore",
    ):
        assert token in p8_filling_transition, token

    # A local PostOnly/CQ/Finish result must never overwrite a concurrent
    # terminal state with a full GM record.  State-last commits are deliberately
    # restricted to post, drain, and owned-release edges.
    p8_state_last_edges = _between(
        text,
        "__aicore__ inline bool IsKvafP8WaveStateLastCommitEdgeAllowed(",
        "__aicore__ inline bool IsKvafP8WaveStateLastPayloadValid(",
    )
    for token in (
        "kKvafP8WavePostIntent",
        "kKvafP8WaveDrainClaimed",
        "kKvafP8WaveFinalized",
        "kKvafP8WavePosted",
        "kKvafP8WaveFree",
        "kKvafP8WaveQuarantined",
    ):
        assert token in p8_state_last_edges, token

    p8_state_last_commit = _between(
        text,
        "__aicore__ inline bool CommitKvafP8WaveMutablePayloadAndStateFromAicore(",
        "// Admission uses a separate claim",
    )
    for token in (
        "KvafP8WaveIdentityEquals",
        "LoadKvafP8WaveRecordFromGm",
        "PublishKvafP8WaveStateLastMutablePayloadToGm",
        "DataSyncBarrier<AscendC::MemDsbT::DDR>()",
        "VF_CALL<KvafP8WaveStateLastCommitSimtService>",
    ):
        assert token in p8_state_last_commit, token
    assert "StoreKvafP8WaveRecordToGm" not in p8_state_last_commit

    # The only DRAIN_CLAIMED -> FREE edge is a separately named no-post
    # release proof.  It must not broaden the regular CQ state-last API: the
    # record must still prove POST_REJECTED/no private payload, publish its
    # mutable words, and CAS the state last.
    p8_claimed_unposted_release = _between(
        text,
        "__aicore__ inline bool CommitKvafP8ClaimedUnpostedWaveReleaseFromAicore(",
        "// Admission uses a separate claim",
    )
    for token in (
        "IsKvafP8ClaimedUnpostedWaveReleasePayloadValid",
        "kKvafP8WaveDrainClaimed",
        "kKvafP8WaveFree",
        "kKvafP8WaveOutcomePostRejected",
        "kDenseBatchGetFailed",
        "PublishKvafP8WaveStateLastMutablePayloadToGm",
        "DataSyncBarrier<AscendC::MemDsbT::DDR>()",
        "VF_CALL<KvafP8ClaimedUnpostedWaveReleaseSimtService>",
    ):
        assert token in p8_claimed_unposted_release, token
    assert "StoreKvafP8WaveRecordToGm" not in p8_claimed_unposted_release
    p8_claimed_unposted_release_cas = _between(
        text,
        "TryCommitKvafP8ClaimedUnpostedWaveReleaseFromSimt(",
        "__simt_vf__ __aicore__ LAUNCH_BOUND(1)\ninline void "
        "KvafP8ClaimedUnpostedWaveReleaseSimtService(",
    )
    for token in (
        "asc_atomic_cas(",
        "kKvafP8WaveDrainClaimed",
        "kKvafP8WaveFree",
        "AscendC::Simt::ThreadFence()",
    ):
        assert token in p8_claimed_unposted_release_cas, token

    p8_admission_claim = _between(
        text,
        "__simt_callee__ inline bool TryClaimKvafP8WaveFillingAdmissionFromSimt(",
        "// A deterministic Begin/Post rejection",
    )
    for token in (
        "asc_atomic_cas(",
        "kKvafP8WaveFree",
        "kKvafP8WaveFillingClaimed",
        "TryClaimKvafP8WaveFillingAdmissionFromAicore",
        "TryQuarantineKvafP8WaveFillingClaimedFromSimt",
        "QuarantineKvafP8WaveFillingClaimedFromAicore",
    ):
        assert token in p8_admission_claim, token

    p8_post_intent_quarantine = _between(
        text,
        "__aicore__ inline bool QuarantineKvafP8PostIntentStateFromAicore(",
        "// Publish only the payload which turns an already durable FILLING record",
    )
    for token in (
        "LoadKvafP8WaveRecordFromGm",
        "KvafP8WaveIdentityEquals",
        "kKvafP8WavePostIntent",
        "kKvafP8WaveQuarantined",
        "TryTransitionKvafP8FillingStateFromAicore",
        "kKvafP8WaveOutcomePostIntentQuarantined",
    ):
        assert token in p8_post_intent_quarantine, token
    assert "StoreKvafP8WaveRecordToGm" not in p8_post_intent_quarantine

    # A deterministic Begin/Post rejection is a normal bounded defer, not an
    # ambiguous POST_INTENT.  It still receives a separate GM drain claim so
    # only one dynamic worker clears its PENDING private map entries and
    # returns the descriptor queue.  It must never share the ordinary POSTED
    # CQ-claim endpoint, because there is no CQ to poll in this branch.
    p8_no_post_claim = _between(
        text,
        "__simt_callee__ inline bool TryClaimKvafP8UnpostedWaveDrainFromSimt(",
        "// P8 phase helpers",
    )
    for token in (
        "asc_atomic_cas(",
        "kKvafP8WaveFinalized",
        "kKvafP8WaveDrainClaimed",
        "TryClaimKvafP8UnpostedWaveDrainFromAicore",
        "VF_CALL<KvafP8UnpostedWaveDrainClaimSimtService>",
    ):
        assert token in p8_no_post_claim, token

    p8_abort = _between(
        text,
        "__simt_callee__ inline bool MarkKvafP8ForwardAbortFromSimt(",
        "// Forward-terminal owns a record",
    )
    for token in (
        "asc_atomic_cas(",
        "kKvafP8ForwardAbortLowMask",
        "AscendC::Simt::ThreadFence()",
        "MarkKvafP8ForwardAbortFromAicore",
        "VF_CALL<KvafP8ForwardAbortSimtService>",
        "Compatibility-only helper",
    ):
        assert token in p8_abort, token
    # A regular GM OR remains only as a documented compatibility/setup API;
    # the P8 executor must use the SIMT-backed AIC entry point.
    assert p8_abort.index("MarkKvafP8ForwardAbortFromAicore") < p8_abort.index(
        "Compatibility-only helper"
    )

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
        "kKvafP8PrivateReleaseClaimed",
        "KvafP8PrivateRecordMatchesWave",
        "ReleaseKvafP8PrivateRecordAfterQsfaTerminal",
        "private_record->state = kKvafP8PrivateQuarantined",
    ):
        assert token in p8_private, token

    p8_private_release = _between(
        text,
        "__simt_callee__ inline bool TryClaimKvafP8PrivateReleaseFromSimt(",
        "// -------------------------------------------------------------------------\n// Forward-terminal fail-closed sweep",
    )
    for token in (
        "kKvafP8PrivateReady",
        "kKvafP8PrivateFailed",
        "kKvafP8PrivateReleaseClaimed",
        "asc_atomic_cas(",
        "ClearKvafP8PrivateMappingsFromSimt",
        "ClearKvafP8PrivateMappingsFromAicore",
        "CommitKvafP8PrivateReleaseFromSimt",
        "CommitKvafP8PrivateReleaseFromAicore",
        "private_ready[private_ready_index]",
        "static_cast<int32_t>(result->observed_state)",
        "Map entries are now proven absent",
    ):
        assert token in p8_private_release, token
    # The compatibility-local helper is deliberately not allowed to reset a
    # private record to FREE before the GM mapping CAS evidence exists.
    compat_release = _between(
        text,
        "__aicore__ inline bool ReleaseKvafP8PrivateRecordAfterQsfaTerminal(\n    KvafP8PrivateRecord*",
        "// SIMT-side ownership claim",
    )
    assert "ResetKvafP8PrivateRecord(private_record)" not in compat_release
    assert "kKvafP8PrivateReleaseClaimed" in compat_release

    # No-post cleanup is deliberately distinct from QSFA terminal release:
    # it can only claim PENDING after a proven PostRejected wave, then uses
    # the same exact source/object CAS evidence before the private row becomes
    # FREE.  This prevents a harmless deterministic no-post from leaking the
    # bounded private pool while preserving fail-closed behavior on ambiguity.
    p8_no_post_private = _between(
        text,
        "__simt_callee__ inline bool TryClaimKvafP8PrivateNoPostReleaseFromSimt(",
        "// -------------------------------------------------------------------------\n// Forward-terminal fail-closed sweep",
    )
    for token in (
        "kKvafP8PrivatePending",
        "kKvafP8PrivateReleaseClaimed",
        "KvafP8ForwardControlMatchesIdentity",
        "ClearKvafP8PrivateMappingsFromAicore",
        "CommitKvafP8PrivateReleaseFromAicore",
        "QuarantineKvafP8PrivateReleaseClaimedFromAicore",
        "ReleaseKvafP8PendingPrivateRecordAfterNoPost",
        "private_ready[private_ready_index]",
    ):
        assert token in p8_no_post_private, token

    p8_terminal = _between(
        text,
        "// Forward-terminal fail-closed sweep",
        "// AscendC treats GM and local memory",
    )
    for token in (
        "QuarantineKvafP8WaveAtForwardTerminal",
        "QuarantineKvafP8PrivateAtForwardTerminal",
        "TryQuarantineKvafP8WaveFromSimt",
        "TryQuarantineKvafP8PrivateFromSimt",
        "kKvafP8WaveOutcomeForwardTerminalQuarantined",
        "kKvafP8WaveFilling",
        "kKvafP8WaveFillingReleaseClaimed",
        "kKvafP8WaveFillingClaimed",
        "kKvafP8WaveQuarantined",
        "kKvafP8PrivateQuarantined",
        "sweep quarantines the bounded resources",
    ):
        assert token in p8_terminal, token
    assert "ReleaseKvafStreamingBatchGetQueueFromAicore" not in p8_terminal

    # CANN MIX exposes GM atomic CAS through the SIMT callable domain.  The
    # ALLOCATING terminal path used to issue its two directory CAS operations
    # from an AIC helper, which compiles as an illegal UB-pointer overload.
    # Keep the rollback best-effort but require the same VF_CALL boundary as
    # every other P8 GM atomic transition.
    p8_allocating_rollback_simt = _between(
        text,
        "__simt_callee__ inline void\n"
        "RollbackKvafP8PrivateAllocatingDirectoryClaimsFromSimt(",
        "__aicore__ inline bool QuarantineKvafP8PrivateAllocatingAtForwardTerminal(",
    )
    for token in (
        "asc_atomic_cas(",
        "source_to_slot + source_offset",
        "object_to_slot + object_offset",
        "kKvafP8PrivateDirectoryClaiming",
        "kKvafP8PrivateDirectoryUnmapped",
        "AscendC::Simt::ThreadFence()",
        "KvafP8PrivateAllocatingDirectoryRollbackSimtService",
    ):
        assert token in p8_allocating_rollback_simt, token

    p8_allocating_terminal = _between(
        text,
        "__aicore__ inline bool QuarantineKvafP8PrivateAllocatingAtForwardTerminal(",
        "// A POST worker may have made a private PENDING record visible",
    )
    assert "asc_atomic_cas(" not in p8_allocating_terminal
    assert (
        "VF_CALL<\n"
        "          KvafP8PrivateAllocatingDirectoryRollbackSimtService>"
        in p8_allocating_terminal
    )

    p8_wave_terminal = _between(
        text,
        "__aicore__ inline bool QuarantineKvafP8WaveAtForwardTerminal(",
        "__simt_callee__ inline bool TryQuarantineKvafP8PrivateFromSimt(",
    )
    claimed_index = p8_wave_terminal.index("kKvafP8WaveFillingClaimed")
    identity_index = p8_wave_terminal.index(
        "KvafP8ForwardTerminalOwnsIdentity"
    )
    assert claimed_index < identity_index
    assert "StoreKvafP8WaveRecordToGm" not in p8_wave_terminal

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

    # P8's first durable point is now FILLING rather than POST_INTENT.  It
    # appears immediately after the queue lease gains its complete identity,
    # and an explicit, durable marker changes before *any* descriptor/private
    # record/request-ledger write.  That lets an empty pre-doorbell exit prove
    # it has no externally visible work; every other pre-doorbell exit remains
    # quarantined rather than guessing from post_committed alone.
    p8_filling_begin = _between(
        text,
        "__aicore__ inline bool BeginKvafP8WaveFilling(",
        "__aicore__ inline bool KvafP8WaveFillingMatchesLeaseAndIdentity(",
    )
    for token in (
        "kKvafP8WaveFilling",
        "kKvafP8FillingNoWorkProof",
        "identity.queue_id != lease.queue_id",
        "identity.owner_tag != lease.owner_tag",
        "owners[lease.queue_id] != lease.owner_tag",
        "kKvafP8ForwardPhaseAdmitPost",
    ):
        assert token in p8_filling_begin, token

    p8_filling_admit = _between(
        text,
        "__aicore__ inline bool BeginAndPublishKvafP8WaveFilling(",
        "__aicore__ inline bool KvafP8WaveFillingMatchesLeaseAndIdentity(",
    )
    for token in (
        "BeginKvafP8WaveFilling",
        "TryClaimKvafP8WaveFillingAdmissionFromAicore",
        "PublishKvafP8WaveFillingClaimedPayloadToGm",
        "QuarantineKvafP8WaveFillingClaimedFromAicore",
        "kKvafP8WaveFillingClaimed",
        "TryTransitionKvafP8FillingStateFromAicore",
        "IsKvafP8WaveRecordValid",
        "IsKvafP8WaveRecordAdmissibleForFilling",
        "LoadKvafP8WaveRecordFromGm",
    ):
        assert token in p8_filling_admit, token
    assert "PublishKvafP8WaveRecordToGm" not in p8_filling_admit
    assert "StoreKvafP8WaveRecordToGm" not in p8_filling_admit

    p8_filling_arm = _between(
        text,
        "__aicore__ inline bool ArmKvafP8WaveFillingForExternalWrites(",
        "// A FILLING record can be returned without a HCCP drain",
    )
    for token in (
        "LoadKvafP8WaveRecordFromGm",
        "KvafP8WaveFillingMatchesLeaseAndIdentity",
        "kKvafP8FillingNoWorkProof",
        "kKvafP8FillingExposureProof",
        "record_gm->reserved[1] = kKvafP8FillingExposureProof",
        "QuarantineKvafP8FillingStateFromAicore",
        "kKvafP8WaveOutcomeFillingQuarantined",
    ):
        assert token in p8_filling_arm, token
    assert "PublishKvafP8WaveRecordToGm(record_gm, *record)" not in p8_filling_arm

    p8_filling_promote = _between(
        text,
        "__aicore__ inline bool PromoteKvafP8WaveFillingToPostIntent(",
        "// Compatibility-only direct constructor",
    )
    for token in (
        "LoadKvafP8WaveRecordFromGm",
        "KvafP8WaveFillingMatchesLeaseAndIdentity",
        "kKvafP8FillingExposureProof",
        "ValidateKvafP8DescriptorPrivateLedger",
        "SnapshotKvafP8QueueIndices",
        "PublishKvafP8FillingPostIntentPayloadToGm",
        "TryTransitionKvafP8FillingStateFromAicore",
        "record->state = kKvafP8WavePostIntent",
    ):
        assert token in p8_filling_promote, token
    assert "PublishKvafP8WaveRecordToGm(record_gm, *record)" not in p8_filling_promote
    assert "StoreKvafP8WaveRecordToGm(record_gm, *record)" not in p8_filling_promote

    p8_filling_release = _between(
        text,
        "__aicore__ inline bool ReleaseKvafP8FillingLeaseAfterProvenNoWork(",
        "__aicore__ inline bool SnapshotKvafP8QueueIndices(",
    )
    for token in (
        "LoadKvafP8WaveRecordFromGm",
        "IsKvafP8WaveFillingProvenNoWork",
        "lease.state != kStreamingBatchGetLeaseHeld",
        "lease.queue_id != identity.queue_id",
        "lease.owner_tag != identity.owner_tag",
        "KvafStreamingBatchGetReleaseSimtService",
        "kKvafP8WaveFillingReleaseClaimed",
        "TryTransitionKvafP8FillingStateFromAicore",
        "IsKvafP8WaveFillingReleaseClaimedProvenNoWork",
        "kKvafP8WaveOutcomeReleaseCasQuarantined",
        "kKvafP8WaveOutcomeFillingQuarantined",
        "record->state = kKvafP8WaveFree",
    ):
        assert token in p8_filling_release, token
    assert "PublishKvafP8WaveRecordToGm(record_gm, *record)" not in p8_filling_release
    assert "StoreKvafP8WaveRecordToGm(record_gm, *record)" not in p8_filling_release
    assert "BeginKvafDenseBatchGetPostOnly" not in p8_filling_release

    p8_unposted = _between(
        text,
        "__aicore__ inline bool ReleaseKvafP8UnpostedWave(",
        "}  // namespace kvaf_fused_get",
    )
    assert "record->state == kKvafP8WavePostIntent" in p8_unposted
    assert "kKvafP8WaveOutcomePostIntentQuarantined" in p8_unposted
    assert "record->state == kKvafP8WaveFilling" in p8_unposted
    assert "kKvafP8WaveOutcomeFillingQuarantined" in p8_unposted

    p8_claimed_unposted = _between(
        text,
        "__aicore__ inline bool ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup(",
        "// Only a *proved unposted* wave",
    )
    for token in (
        "kKvafP8WaveDrainClaimed",
        "kKvafP8WaveOutcomePostRejected",
        "record->post_committed != 0U",
        "request_source_offsets[request]",
        "request_object_offsets[request]",
        "for (uint32_t prior = 0U; prior < request; ++prior)",
        "kKvafP8PrivateFree",
        "ReleaseKvafStreamingBatchGetQueueFromAicore",
    ):
        assert token in p8_claimed_unposted, token

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
