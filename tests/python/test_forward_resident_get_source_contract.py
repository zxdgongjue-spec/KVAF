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
DENSE = ROOT / "access_operator/demo/aiv_batch_get/kernel/execute_batch_get_aiv.asc"
HOST = ROOT / "access_operator/src/access_operator.cc"
PUBLIC = ROOT / "access_operator/include/kvaf/access_operator/access_operator.h"


def _between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def main() -> None:
    abi = ABI.read_text()
    device = DEVICE.read_text()
    dense = DENSE.read_text()
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
        "BeginKvafDenseBatchGetPostOnlyScalar",
        "TryRetireKvafDenseBatchGet",
        "FinishKvafDenseBatchGet",
        "ReturnKvafForwardResidentUnpostedQueueFromAicore",
        "ReturnKvafForwardResidentQueueFromAicore",
        # The forward-resident wrapper lives in a different namespace from
        # the dense Direct-HCCP implementation.  The Begin boundary must use
        # the compact scalar helper, while the later CQ drain may still own a
        # transaction in its local wave record.
        "using kvaf_fused_get::BeginKvafDenseBatchGetPostOnlyScalar;",
    ):
        assert token in device, token

    validation = _between(
        device,
        "__aicore__ inline bool ValidateKvafForwardResidentOutputIndices(",
        "// Append translates one already-admitted KV object",
    )
    assert "resident_row_base" in validation
    assert "resident_end" in validation
    assert "ReadGmByPassDCache<int32_t>" in validation
    assert "output_indices + earlier" in validation

    # Queue acquire/release runs in SIMT while descriptor preparation and
    # PostOnly run in AICore.  Those AICore checks must not use an ordinary GM
    # load of a lease word that a preceding SIMT CAS has just changed: cached
    # zero would incorrectly reject a valid queue before descriptor publish.
    owner_load = _between(
        device,
        "__aicore__ inline int32_t ReadKvafForwardResidentQueueOwner(",
        "__aicore__ inline __gm__ KvafForwardResidentBatchGetDescriptorScratch*",
    )
    assert "ReadGmByPassDCache<int32_t>(owners + queue_id)" in owner_load
    for function_start, function_end in (
        (
            "__aicore__ inline bool LoadKvafForwardResidentQueueTemplate(",
            "// CandidateCompact owns cross-wave unique-slot admission.",
        ),
        (
            "__aicore__ inline bool PublishKvafForwardResidentDescriptor(",
            "__aicore__ inline bool BuildKvafForwardResidentQueueArgsFromDescriptor(",
        ),
        (
            "__aicore__ inline uint32_t BeginKvafForwardResidentBatchGetPostOnly(",
            "__aicore__ inline bool InitializeKvafForwardResidentPostedWaveForDrain(",
        ),
        (
            "__aicore__ inline bool InitializeKvafForwardResidentPostedWaveForDrain(",
            "__aicore__ inline void TryRetireKvafForwardResidentBatchGet(",
        ),
    ):
        queue_owner_check = _between(device, function_start, function_end)
        assert "ReadKvafForwardResidentQueueOwner(control, lease.queue_id)" in queue_owner_check
        assert "owners[lease.queue_id]" not in queue_owner_check

    queue_args = _between(
        device,
        "__aicore__ inline bool BuildKvafForwardResidentQueueArgsFromDescriptor(",
        "// Keep this full-validation reload helper for independent callers",
    )
    assert "args->output = control.output" in queue_args
    assert "private_output" not in queue_args
    assert "ValidateKvafForwardResidentOutputIndices(control, descriptor)" in queue_args

    # `Publish` retains its original three-argument source API and remains
    # the one descriptor-publish gate.  It performs complete
    # range/duplicate validation and DDR publication; Begin may only reuse
    # its local descriptor immediately after that success.
    publish = _between(
        device,
        "__aicore__ inline bool PublishKvafForwardResidentDescriptor(",
        "__aicore__ inline bool BuildKvafForwardResidentQueueArgsFromDescriptor(",
    )
    for token in (
        "ValidateKvafForwardResidentOutputIndices(control, descriptor)",
        "StoreKvafForwardResidentDescriptorToGm(scratch, descriptor)",
        "DataSyncBarrier<AscendC::MemDsbT::DDR>()",
        "control.output_rows > UINT32_MAX",
    ):
        assert token in publish, token
    assert "const KvafForwardResidentBatchGetDescriptorScratch& descriptor)" in publish
    assert "KvafExecuteBatchGetKernelArgs* published_args" not in publish
    assert publish.index("ValidateKvafForwardResidentOutputIndices(") < publish.index(
        "StoreKvafForwardResidentDescriptorToGm("
    )

    begin = _between(
        device,
        "__aicore__ inline uint32_t BeginKvafForwardResidentBatchGetPostOnly(",
        "__aicore__ inline bool InitializeKvafForwardResidentPostedWaveForDrain(",
    )
    # The descriptor publication is not a caller-optional helper.  Begin
    # receives the local descriptor and must publish it (including the DDR
    # barrier in the helper) before it invokes the established scalar
    # PostOnly primitive.  Crucially it cannot construct or pass a
    # caller-owned dense transaction while the larger C8 closure executes.
    assert "const KvafForwardResidentBatchGetDescriptorScratch& descriptor" in begin
    assert "PublishKvafForwardResidentDescriptor(control, lease, descriptor)" in begin
    assert begin.index("PublishKvafForwardResidentDescriptor(") < begin.index(
        "BeginKvafDenseBatchGetPostOnlyScalar("
    )
    assert "BuildKvafForwardResidentQueueArgsFromDescriptor(" not in begin
    for token in (
        "args.output = control.output",
        "args.queue_state =",
        "active_request_count = descriptor.active_request_count",
    ):
        assert token in begin, token
    assert begin.index("PublishKvafForwardResidentDescriptor(") < begin.index(
        "args.address_table ="
    ) < begin.index("SnapshotKvafForwardResidentQueueProgress(")
    assert "KvafForwardResidentBatchGetWave*" not in begin
    assert "KvafDenseBatchGetTransaction transaction" not in begin
    assert "BeginKvafDenseBatchGetPostOnly(" not in begin
    # A compact scalar result cannot prove that HCCP was posted.  The Begin
    # boundary must instead require the exact durable SQ producer-index
    # advance for this wave.
    assert "SnapshotKvafForwardResidentQueueProgress(" in begin
    assert "sq_producer_delta == static_cast<uint64_t>(active_request_count)" in begin
    assert "return kForwardResidentBeginPosted;" in begin
    assert "kForwardResidentBeginSqProducerDeltaMismatch" in begin
    assert "return ClassifyKvafForwardResidentScalarBeginWithoutPost(" in begin
    assert "return ClassifyKvafForwardResidentBegin(" not in begin
    exact_delta = "sq_producer_delta == static_cast<uint64_t>(active_request_count)"
    partial_delta = "if (sq_producer_delta != 0U)"
    assert begin.index(exact_delta) < begin.index(partial_delta) < begin.index(
        "return kForwardResidentBeginSqProducerDeltaMismatch;"
    ) < begin.index("return ClassifyKvafForwardResidentScalarBeginWithoutPost(")
    assert "!IsKvafForwardResidentBatchGetControlValidForAicore(control)" in begin

    rehydrate = _between(
        device,
        "__aicore__ inline bool InitializeKvafForwardResidentPostedWaveForDrain(",
        "__aicore__ inline void TryRetireKvafForwardResidentBatchGet(",
    )
    for token in (
        "IsKvafForwardResidentBatchGetControlValidForAicore(control)",
        "BuildKvafForwardResidentQueueArgsForPostedDrain(",
        "active_request_count != expected_active_request_count",
        "wave->lease.state = kForwardResidentQueuePosted",
        "wave->transaction.state = kDenseBatchGetInFlight",
        "ResetDenseBatchGetTransaction(&wave->transaction)",
    ):
        assert token in rehydrate, token
    assert "PublishKvafForwardResidentDescriptor(" not in rehydrate
    assert "BeginKvafDenseBatchGetPostOnly(" not in rehydrate
    assert "BuildKvafForwardResidentQueueArgs(" not in rehydrate
    assert "LoadKvafForwardResidentDescriptorFromGm(" not in rehydrate

    drain_args = _between(
        device,
        "__aicore__ inline bool BuildKvafForwardResidentQueueArgsForPostedDrain(",
        "// These wrappers intentionally contain no alternative HCCP protocol.",
    )
    # Rehydration must obtain CQ/PI ownership from the immutable template and
    # compare the durable descriptor release word, rather than reload and
    # re-run the full O(n^2) vector validation after a proven post.
    for token in (
        "LoadKvafForwardResidentQueueTemplate(control, lease,",
        "ReadGmByPassDCache<uint32_t>(",
        "&scratch->active_request_count",
        "published_active_request_count != expected_active_request_count",
        "control.output_rows > UINT32_MAX",
        "args->queue_state =",
        "args->output = control.output",
    ):
        assert token in drain_args, token
    assert "ValidateKvafForwardResidentOutputIndices(" not in drain_args

    assert "KvafForwardResidentBeginHasNoPost" in device
    begin_status = _between(
        device,
        "enum KvafForwardResidentBeginStatus",
        "// AscendC cannot use compiler-generated struct assignment",
    )
    assert "kForwardResidentBeginPosted = 0U" in begin_status
    assert "kForwardResidentBeginResultUnknown" in begin_status
    assert "kForwardResidentBeginSqProducerDeltaMismatch" in begin_status

    no_post = _between(
        device,
        "__aicore__ inline bool KvafForwardResidentBeginHasNoPost(",
        "// The sole HCCP PostOnly boundary.",
    )
    # Partial/ambiguous PI progress is never permitted to recycle its queue.
    assert "kForwardResidentBeginSqProducerDeltaMismatch" not in no_post
    assert "kForwardResidentBeginResultUnknown" not in no_post

    append = _between(
        device,
        "__aicore__ inline void PrepareKvafForwardResidentDescriptorVectorsForPublication(",
        "// `BeginKvafForwardResidentBatchGet`",
    )
    # AICore publishes the dense vectors that a SIMT `asc_ldcg` reader
    # materializes into HCCP WQEs.  Keep both sides on the non-DCache path;
    # ordinary stores can expose a prior descriptor generation after reuse.
    for token in (
        "PrepareKvafForwardResidentDescriptorVectorsForPublication(*descriptor)",
        "WriteGmByPassDCache<int32_t>(indices + request",
        "WriteGmByPassDCache<int32_t>(offsets + request",
        "WriteGmByPassDCache<uint8_t>(mask + request, 1U)",
        "WriteGmByPassDCache<int32_t>(output_indices + request",
    ):
        assert token in append, token
    assert append.index("PrepareKvafForwardResidentDescriptorVectorsForPublication(") < append.index(
        "WriteGmByPassDCache<int32_t>(indices + request"
    )
    prepare_vectors = _between(
        device,
        "__aicore__ inline void PrepareKvafForwardResidentDescriptorVectorsForPublication(",
        "__aicore__ inline bool AppendKvafForwardResidentRequest(",
    )
    for token in (
        "kvaf_fused_get::CleanCacheLines(",
        "descriptor.indices",
        "descriptor.offsets",
        "descriptor.mask",
        "descriptor.output_indices",
        "DataSyncBarrier<AscendC::MemDsbT::DDR>()",
    ):
        assert token in prepare_vectors, token

    store = _between(
        device,
        "__aicore__ inline bool StoreKvafForwardResidentDescriptorToGm(",
        "// The resident arena is fixed",
    )
    count_store = "WriteGmByPassDCache<uint32_t>(&destination->active_request_count"
    assert count_store in store
    assert store.index("DataSyncBarrier<AscendC::MemDsbT::DDR>()") < store.index(
        count_store
    )

    # Direct-HCCP writes HBM behind the AIV's cache.  The exact output rows
    # must be cleaned and invalidated before the doorbell; doing so after CQ
    # would risk writing stale pre-DMA data back over the payload.
    assert "PrepareDenseBatchGetOutputForInboundDma(" in dense
    dense_legacy_begin = _between(
        dense,
        "__aicore__ inline void BeginKvafDenseBatchGetPostOnly(",
        "// Forward-resident P10 needs the verified Dense PostOnly protocol",
    )
    assert dense_legacy_begin.index("ValidateDenseBatchGetDescriptors(") < dense_legacy_begin.index(
        "PrepareDenseBatchGetOutputForInboundDma("
    ) < dense_legacy_begin.index("PostDenseBatchGetRequests(")
    assert "KvafDenseBatchGetTransaction" in dense_legacy_begin
    assert "WriteGmByPassDCache<uint32_t>(&queues->initialized, 1U)" in dense_legacy_begin
    assert "WriteGmByPassDCache<uint64_t>(&queues->sq_producer_index, 0U)" in dense_legacy_begin

    # P10 never sends the large transaction record through its Begin closure.
    # Its scalar helper reuses the same validated Dense PostOnly steps, and
    # initializes queue state through bypass loads/stores so a later tile
    # cannot observe a stale DCache copy of the durable PI receipt.
    dense_scalar_begin = _between(
        dense,
        "__aicore__ inline uint64_t BeginKvafDenseBatchGetPostOnlyScalar(",
        "// Probe the one CQE",
    )
    assert "KvafDenseBatchGetTransaction" not in dense_scalar_begin
    assert dense_scalar_begin.index("ValidateDenseBatchGetDescriptors(") < dense_scalar_begin.index(
        "PrepareDenseBatchGetOutputForInboundDma("
    ) < dense_scalar_begin.index("PostDenseBatchGetRequests(")
    for token in (
        "ReadGmByPassDCache<uint32_t>(&queues->initialized)",
        "ReadGmByPassDCache<uint32_t>(&queues->magic)",
        "WriteGmByPassDCache<uint32_t>(&queues->initialized, 1U)",
        "WriteGmByPassDCache<uint64_t>(&queues->sq_producer_index, 0U)",
    ):
        assert token in dense_scalar_begin, token

    dense_commit = _between(
        dense,
        "__aicore__ inline void CommitPostScalarStage(",
        "__aicore__ inline void PostActiveRequests(",
    )
    assert "WriteGmByPassDCache<uint64_t>(&queue->sq_producer_index" in dense_commit
    assert dense_commit.index("WriteGmByPassDCache<uint64_t>(") < dense_commit.index(
        "DataSyncBarrier<MemDsbT::DDR>()"
    ) < dense_commit.index("st_dev(")

    dense_post_requests = _between(
        dense,
        "__aicore__ inline void PostDenseBatchGetRequests(",
        "// Dense P1 descriptors have already passed",
    )
    assert "ReadGmByPassDCache<uint64_t>(" in dense_post_requests
    assert "&queues[jetty].sq_producer_index" in dense_post_requests
    assert "current_heads[jetty] = queues[jetty].sq_producer_index;" not in dense_post_requests

    dense_retire = _between(
        dense,
        "__aicore__ inline void TryRetireKvafDenseBatchGet(",
        "// Terminal drain/final publication.",
    )
    for token in (
        "ReadGmByPassDCache<uint64_t>(&queue->cq_consumer_index)",
        "ReadGmByPassDCache<uint64_t>(&queue->completed_batch_count)",
        "WriteGmByPassDCache<uint64_t>(&queue->cq_consumer_index",
        "WriteGmByPassDCache<uint64_t>(&queue->completed_batch_count",
        "DataSyncBarrier<MemDsbT::DDR>()",
    ):
        assert token in dense_retire, token

    dense_finish = _between(
        dense,
        "__aicore__ inline void FinishKvafDenseBatchGet(",
        "// Compatibility wrapper retained",
    )
    for token in (
        "WriteGmByPassDCache<uint64_t>(&queue->last_error",
        "ReadGmByPassDCache<uint64_t>(&queue->completed_batch_count)",
        "ReadGmByPassDCache<uint64_t>(&queue->sq_producer_index)",
        "ReadGmByPassDCache<uint64_t>(&queue->cq_consumer_index)",
    ):
        assert token in dense_finish, token

    for token in (
        "constexpr uint32_t kDenseBatchGetReservedWqeUbBytes = 4096U;",
        "constexpr uint32_t kDenseBatchGetPostOnlyScratchCeiling = 135168U;",
        "kDenseBatchGetReservedWqeUbBytes == kDenseWqeBanksBytes",
        "kDenseBatchGetPostOnlyScratchCeiling == kActiveCountUbOffset",
    ):
        assert token in dense, token

    # Dense PostOnly is a bounded alternative to mask compaction, not an
    # alternative UB address space. Its VF materializer and MTE3 copier must
    # use the same two static get_imm() WQE banks as normal BatchGet. Passing
    # an AICore-local __ubuf__ array through VF_CALL can let SQ consume
    # stale/zero WQEs while the CQ terminal still looks successful.
    dense_fill = _between(
        dense,
        "inline void FillDenseBatchGetReadWqes(",
        "__aicore__ inline void PrepareDensePostVectorStage(",
    )
    assert "uint32_t ub_base_offset" in dense_fill
    assert "get_imm(ub_base_offset + slot * kWqeBytes)" in dense_fill
    assert "__ubuf__ uint8_t *wqe_bank" not in dense_fill

    dense_post = _between(
        dense,
        "__aicore__ inline void PrepareDensePostVectorStage(",
        "// Dense P1 descriptors have already passed",
    )
    for token in (
        "const uint32_t ub_base_offset = bank_id * kWqeBankBytes",
        "get_imm(ub_base_offset)",
        "get_imm(ub_base_offset + first_count * kWqeBytes)",
        "PrepareDensePostMte3Stage(&prepared_posts[final_bank], 0U",
    ):
        assert token in dense_post, token
    assert "wqe_banks" not in dense_post
    assert "__ubuf__ uint8_t *wqe_bank" not in dense_post
    assert "dense_wqe_words" not in dense_legacy_begin
    assert "dense_wqe_banks" not in dense_legacy_begin

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
