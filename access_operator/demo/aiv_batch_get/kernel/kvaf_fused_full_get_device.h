#ifndef KVAF_FUSED_FULL_GET_DEVICE_H_
#define KVAF_FUSED_FULL_GET_DEVICE_H_

// Device-side reuse boundary for a fused AIV operator.
//
// ExecuteBatchGet is currently built as a standalone AIV kernel in
// execute_batch_get_aiv.asc.  A MIX kernel cannot launch that global kernel
// as a child.  This header instead includes the same single-source device
// implementation with its standalone entry point suppressed.  A fused
// operator can either call the compatibility helper
// `RunKvafDenseBatchGetCore256(args, active_request_count, core)`, or use the
// explicit transaction sequence below:
//
//   KvafDenseBatchGetTransaction txn;
//   BeginKvafDenseBatchGetPostOnly(args, count, core, &txn);
//   // Run ExactScan tiles and occasionally TryRetireKvafDenseBatchGet(...).
//   FinishKvafDenseBatchGet(args, &txn);
//
// PostOnly has exactly the historical validation/WQE/SQ-doorbell semantics;
// only the CQ wait moves outside it.  There is deliberately one transaction
// per pre-bound queue: a caller must retire or finish a wave before posting a
// later wave on the same queue.  The existing queue ABI has no multi-flight
// CQ ledger, so relaxing that rule would make descriptor/WQE order ambiguous.
// Keeping one source of truth is deliberate: the Direct-HCCP WQE and CQ
// protocol must not drift between the normal BatchGet path and the fused
// path.
//
// The caller owns `core`: it is a KVAF transport-worker ordinal in
// [0, args.block_count), not a public physical-AIV identifier.  P1 uses an
// AIV-only task with lanes 0 and 1 as its two Get workers; the remaining task
// lanes run ExactScan producers.
//
// Streaming P2 uses the separate helpers below instead. There are no fixed
// Get-owner AIVs: every LI producer atomically leases an idle Direct-HCCP
// queue from an int32 queue-pool, posts one wave, continues scanning score
// tiles, probes completion, then releases the queue only after terminal CQ
// retirement. The pool has at most KVAF_AIV_BATCH_GET_MAX_AIV_CORES entries;
// it is a Direct-HCCP transport limit, not a count of LI AIV work items.

#ifndef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#define KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED 1
#include "execute_batch_get_aiv.asc"
#undef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#endif

#include "simt_api/cpp/kernel_simt_warp_level_intf.h"

namespace kvaf_fused_get {

// `batch_get_abi.h` is intentionally a C-compatible, global-namespace ABI:
// host code, the standalone AIV kernel, and a fused AscendC operator all use
// the same byte layout.  Expose the two streaming records through this device
// helper namespace as aliases, rather than creating a second C++ wrapper
// type.  That gives fused clients one coherent API while preserving the
// host/HBM ABI exactly.
using ::KvafStreamingBatchGetControl;
using ::KvafStreamingBatchGetDescriptorScratch;

// Caller-local lease state. It is intentionally not stored in HBM: the
// globally shared state is only the int32 owner table referenced by the
// graph-stable KvafStreamingBatchGetControl. `owner_tag` is positive and
// equals 1 + the caller's stable LI producer id.
enum KvafStreamingBatchGetLeaseState : uint32_t {
  kStreamingBatchGetLeaseIdle = 0U,
  kStreamingBatchGetLeaseHeld = 1U,
  kStreamingBatchGetLeasePosted = 2U,
  kStreamingBatchGetLeaseTerminal = 3U,
};

struct KvafStreamingBatchGetLease {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t state;
  uint32_t reserved0;
};

// UB hand-off record for the AIC-facing acquire/release wrappers. The raw
// atomics live in a one-thread SIMT service because CANN declares
// asc_atomic_cas as a SIMT callee in a MIX kernel. An AIC scalar caller must
// use the wrappers below rather than calling the raw SIMT helper directly.
struct KvafStreamingBatchGetLeaseResult {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t success;
  uint32_t reserved0;
};
static_assert(sizeof(KvafStreamingBatchGetLeaseResult) == 16U,
              "streaming queue lease UB hand-off ABI changed");

// A queue-owner lease alone does not serialize later CQ-drain graph phases:
// several dynamic AIV tasks can all observe the same durable POSTED wave.
// This small UB hand-off records a second CAS claim on the wave state itself.
// It keeps ownership dynamic (any worker may drain) while making one CQ wave
// single-consumer.
struct KvafP8WaveDrainClaimResult {
  uint32_t success;
  uint32_t observed_state;
};
static_assert(sizeof(KvafP8WaveDrainClaimResult) == 8U,
              "P8 drain-claim UB hand-off ABI changed");

// A caller-local wave combines the leased queue-specific argument view with
// the existing verified Direct-HCCP transaction. It is one active wave per
// lease; the existing SQ/CQ ABI deliberately does not support multiple
// in-flight waves on one queue.
struct KvafStreamingBatchGetWave {
  KvafStreamingBatchGetLease lease;
  KvafExecuteBatchGetKernelArgs queue_args;
  KvafDenseBatchGetTransaction transaction;
};

// This is deliberately caller-local ready information. It is valid only
// after Finish has made the CQ terminal; callers may then publish it to their
// own bounded source-map/ready arena. No helper here writes a persistent HBM
// block table, cache slot, or resident mapping.
struct KvafStreamingBatchGetReady {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t active_request_count;
  uint32_t ready;
  uint64_t result;
};

// P8 keeps the transport transaction in a graph-stable, forward-owned record
// rather than in an AIV-local `KvafStreamingBatchGetWave`.  The local wave is
// still rebuilt transiently when an executor phase runs, but this record is
// the only authority that survives from PostOnly to a later CQ-drain/terminal
// phase.  It is deliberately one fixed record per Direct-HCCP queue, not a
// per-candidate ticket or an unbounded completion ledger.
//
// Why all six identity fields are needed:
//   * forward_epoch and graph_replay_generation prevent an old replay from
//     adopting a later forward's private output;
//   * layer_control_index identifies the layer-private registered output;
//   * queue_id, queue_generation and post_sequence identify one SQ/CQ wave
//     within the fixed transport pool.
// `owner_tag` is retained only as the CAS ownership proof.  It is not enough
// to identify a wave after a graph replay or a terminal recovery path.
constexpr uint64_t kKvafP8WaveRecordMagic = 0x4B56414650385756ULL;  // "KVAFP8WV"
// Version 2 adds the forward-control and private-owner checks below.  The
// record remains ABI-size compatible, but an old graph-stable allocation must
// never be accepted as a current P8 transaction merely because its bytes fit.
constexpr uint32_t kKvafP8WaveRecordVersion = 2U;

enum KvafP8WaveState : uint32_t {
  // No lease, no ambiguous transport ownership.
  kKvafP8WaveFree = 0U,
  // Descriptor/ledger are materialized and visible, but no doorbell has
  // executed yet. It may be rolled back only while post_committed == 0.
  kKvafP8WavePostIntent = 1U,
  // BeginKvafDenseBatchGetPostOnly returned InFlight; a late CQ is possible.
  kKvafP8WavePosted = 2U,
  // Exactly one dynamic drain worker atomically owns this durable GM record.
  // A second worker must skip rather than probe/finish/release the same CQ.
  kKvafP8WaveDrainClaimed = 3U,
  // A CQE was observed and the transaction is eligible for Finish.
  kKvafP8WaveCqObserved = 4U,
  // Finish has published the KVAF terminal status. A private READY/FAILED
  // publication may follow, but the queue cannot be reused yet.
  kKvafP8WaveFinalized = 5U,
  // A timeout, identity mismatch, release-CAS failure, or POST_INTENT
  // ambiguity leaves the transport non-reusable until the cold recovery path
  // proves it quiescent. Never rewrite this state to FREE in an AIV kernel.
  kKvafP8WaveQuarantined = 6U,
};

enum KvafP8WaveOutcome : uint32_t {
  kKvafP8WaveOutcomeNone = 0U,
  kKvafP8WaveOutcomeSuccess = 1U,
  kKvafP8WaveOutcomeCqErrorObserved = 2U,
  kKvafP8WaveOutcomePostRejected = 3U,
  kKvafP8WaveOutcomeCqTimeoutQuarantined = 4U,
  kKvafP8WaveOutcomeIdentityQuarantined = 5U,
  kKvafP8WaveOutcomeReleaseCasQuarantined = 6U,
  kKvafP8WaveOutcomePostIntentQuarantined = 7U,
};

struct KvafP8WaveIdentity {
  uint64_t forward_epoch;
  uint64_t graph_replay_generation;
  uint32_t layer_control_index;
  uint32_t queue_id;
  uint64_t queue_generation;
  uint64_t post_sequence;
  int32_t owner_tag;
  uint32_t reserved0;
};
static_assert(sizeof(KvafP8WaveIdentity) == 48U,
              "P8 wave identity ABI changed");

// The runtime owns one graph-stable forward-control record per P8 layer.
// It is deliberately tiny: it is not a ticket or an unbounded scheduler
// ledger.  Its purpose is to let every independently launched P8 phase prove
// that the GM wave/private record belongs to *this* forward/replay/layer
// before it touches a Direct-HCCP queue or exposes a private HBM payload.
//
// `flags` is a forward-wide fail-closed gate.  When an identity, CQ or release
// failure quarantines a wave, the executor sets ABORT|PRIVATE_VIEW_DISABLED
// before QSFA.  Exact reconcile/GuaranteedRecall then owns the layer result;
// QSFA must not consume even an earlier READY speculative slot from this
// aborting forward.
enum KvafP8ForwardPhase : uint32_t {
  kKvafP8ForwardPhaseInvalid = 0U,
  kKvafP8ForwardPhaseAdmitPost = 1U,
  kKvafP8ForwardPhaseDrain = 2U,
  kKvafP8ForwardPhaseQsfaTerminal = 3U,
  kKvafP8ForwardPhaseForwardTerminal = 4U,
};

enum KvafP8ForwardFlags : uint64_t {
  kKvafP8ForwardFlagNone = 0U,
  kKvafP8ForwardFlagAbort = 1ULL << 0U,
  kKvafP8ForwardFlagPrivateViewDisabled = 1ULL << 1U,
  // Set by RuntimeResources only after this layer's exact Top-K *and*
  // Guaranteed Recall are terminal.  Speculation may not make QSFA visible
  // by setting this bit on its own.
  kKvafP8ForwardFlagExactInputReady = 1ULL << 2U,
  kKvafP8ForwardFlagQsfaTerminalObserved = 1ULL << 3U,
  kKvafP8ForwardFlagModelTerminalObserved = 1ULL << 4U,
};

struct KvafP8ForwardControl {
  uint64_t forward_epoch;
  uint64_t graph_replay_generation;
  uint32_t layer_control_index;
  uint32_t phase;
  uint64_t flags;
};
static_assert(sizeof(KvafP8ForwardControl) == KVAF_P8_FORWARD_CONTROL_BYTES,
              "P8 forward control ABI changed");

__aicore__ inline bool IsKvafP8ForwardControlValid(
    const KvafP8ForwardControl& control) {
  return control.forward_epoch != 0U &&
         control.phase >= kKvafP8ForwardPhaseAdmitPost &&
         control.phase <= kKvafP8ForwardPhaseForwardTerminal;
}

// P8 phases may run in distinct graph invocations.  Keep their GM/local
// boundary explicit rather than relying on an aggregate load that AscendC
// treats differently across address spaces.
__aicore__ inline bool LoadKvafP8ForwardControlFromGm(
    __gm__ const KvafP8ForwardControl* source,
    KvafP8ForwardControl* destination) {
  if (source == nullptr || destination == nullptr) {
    return false;
  }
  destination->forward_epoch = source->forward_epoch;
  destination->graph_replay_generation = source->graph_replay_generation;
  destination->layer_control_index = source->layer_control_index;
  destination->phase = source->phase;
  destination->flags = source->flags;
  return IsKvafP8ForwardControlValid(*destination);
}

__aicore__ inline bool StoreKvafP8ForwardControlToGm(
    __gm__ KvafP8ForwardControl* destination,
    const KvafP8ForwardControl& source) {
  if (destination == nullptr || !IsKvafP8ForwardControlValid(source)) {
    return false;
  }
  destination->forward_epoch = source.forward_epoch;
  destination->graph_replay_generation = source.graph_replay_generation;
  destination->layer_control_index = source.layer_control_index;
  destination->phase = source.phase;
  destination->flags = source.flags;
  return true;
}

__aicore__ inline bool KvafP8ForwardControlMatchesIdentity(
    const KvafP8ForwardControl& control,
    const KvafP8WaveIdentity& identity,
    KvafP8ForwardPhase expected_phase) {
  return IsKvafP8ForwardControlValid(control) &&
         control.phase == expected_phase &&
         (control.flags & (kKvafP8ForwardFlagAbort |
                           kKvafP8ForwardFlagPrivateViewDisabled)) == 0U &&
         control.forward_epoch == identity.forward_epoch &&
         control.graph_replay_generation ==
             identity.graph_replay_generation &&
         control.layer_control_index == identity.layer_control_index;
}

// A caller that cannot prove a wave belongs to the current forward must not
// leave its prior READY data visible to QSFA.  This is intentionally a small
// monotonic flag operation; cold runtime teardown owns recovery of any
// quarantined Direct-HCCP resource.
__aicore__ inline void MarkKvafP8ForwardAbort(
    __gm__ KvafP8ForwardControl* control) {
  if (control == nullptr) {
    return;
  }
  control->flags |= kKvafP8ForwardFlagAbort |
                    kKvafP8ForwardFlagPrivateViewDisabled;
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

// The P8 private view is merely a performance input to QSFA.  It becomes
// readable only once RuntimeResources has completed exact Top-K and the
// Guaranteed Recall fill for this layer.  This helper deliberately has no
// path that lets a successful speculative CQ substitute for that proof.
__aicore__ inline bool IsKvafP8PrivateViewReadyForQsfa(
    const KvafP8ForwardControl& control) {
  return IsKvafP8ForwardControlValid(control) &&
         control.phase == kKvafP8ForwardPhaseQsfaTerminal &&
         (control.flags & (kKvafP8ForwardFlagAbort |
                           kKvafP8ForwardFlagPrivateViewDisabled)) == 0U &&
         (control.flags & kKvafP8ForwardFlagExactInputReady) != 0U;
}

// Keep the record exactly four cache lines.  The reserved tail is intentional:
// P8 must not grow a variable-size device ledger when candidate bursts occur.
// Transaction is copied field-by-field at GM/local boundaries below because
// AscendC rejects aggregate copies across address spaces.
struct KvafP8WaveRecord {
  uint64_t magic;
  uint32_t version;
  uint32_t state;
  KvafP8WaveIdentity identity;
  uint64_t sq_producer_before;
  uint64_t cq_consumer_before;
  uint64_t sq_producer_after;
  uint64_t cq_consumer_after;
  uint32_t request_count;
  uint32_t outcome;
  // Set only after every private slot belonging to this CQ wave has reached
  // READY or FAILED.  This is deliberately *not* the QSFA last-reader
  // terminal: the Direct-HCCP queue/descriptor can be reused at CQ terminal,
  // while the separate per-private-slot record retains the payload lifetime
  // until QSFA is finished.
  uint32_t private_ready_published;
  uint32_t post_committed;
  KvafDenseBatchGetTransaction transaction;
  uint64_t reserved[5];
};
static_assert(sizeof(KvafDenseBatchGetTransaction) == 104U,
              "P8 requires the verified DenseBatchGet transaction ABI");
static_assert(sizeof(KvafP8WaveRecord) == 256U,
              "P8 wave record must remain four fixed cache lines");

// The HCCP queue and the private HBM payload intentionally have different
// lifetimes.  A CQ-terminal wave may return its descriptor queue immediately
// after publishing READY/FAILED, but its packed output and source/object
// mapping remain readable by QSFA until that layer's final reader completes.
// This bounded record is the durable authority for the latter lifetime.  It
// is one fixed record per private output row, never an unbounded ticket list.
enum KvafP8PrivateState : uint32_t {
  kKvafP8PrivateFree = 0U,
  kKvafP8PrivatePending = 1U,
  kKvafP8PrivateReady = 2U,
  kKvafP8PrivateFailed = 3U,
  kKvafP8PrivateQuarantined = 4U,
};

struct KvafP8PrivateRecord {
  uint64_t magic;
  uint32_t version;
  uint32_t state;
  uint64_t forward_epoch;
  uint64_t graph_replay_generation;
  uint32_t layer_control_index;
  uint32_t queue_id;
  uint64_t queue_generation;
  uint64_t post_sequence;
  uint64_t source_offset;
  uint64_t object_offset;
  int32_t output_index;
  // Keep the P8 private record coupled to the exact queue owner that filled
  // it.  This reuses the old padding so the graph-stable tensor remains
  // int64[N, 10].  `owner_tag` alone is not a complete wave identity, but it
  // closes the otherwise missing owner check when paired with generation and
  // post sequence.
  int32_t owner_tag;
};
static_assert(sizeof(KvafP8PrivateRecord) == KVAF_P8_PRIVATE_RECORD_BYTES,
              "P8 private record ABI changed");

__aicore__ inline void ResetKvafP8WaveRecord(
    KvafP8WaveRecord* record, const KvafP8WaveIdentity& identity) {
  if (record == nullptr) {
    return;
  }
  record->magic = kKvafP8WaveRecordMagic;
  record->version = kKvafP8WaveRecordVersion;
  record->state = kKvafP8WaveFree;
  record->identity = identity;
  record->sq_producer_before = 0U;
  record->cq_consumer_before = 0U;
  record->sq_producer_after = 0U;
  record->cq_consumer_after = 0U;
  record->request_count = 0U;
  record->outcome = kKvafP8WaveOutcomeNone;
  record->private_ready_published = 0U;
  record->post_committed = 0U;
  ResetDenseBatchGetTransaction(&record->transaction);
  for (uint32_t index = 0U; index < 5U; ++index) {
    record->reserved[index] = 0U;
  }
}

__aicore__ inline bool IsKvafP8WaveRecordValid(
    const KvafP8WaveRecord& record) {
  return record.magic == kKvafP8WaveRecordMagic &&
         record.version == kKvafP8WaveRecordVersion &&
         record.identity.owner_tag > 0 &&
         record.state <= kKvafP8WaveQuarantined;
}

// Explicit field-wise transfers preserve the P8 record ABI across the GM/AIC
// boundary.  Do not replace these with aggregate assignment: AscendC treats
// the two address spaces as distinct C++ types.
__aicore__ inline bool LoadKvafP8WaveRecordFromGm(
    __gm__ const KvafP8WaveRecord* source, KvafP8WaveRecord* destination) {
  if (source == nullptr || destination == nullptr) {
    return false;
  }
  destination->magic = source->magic;
  destination->version = source->version;
  destination->state = source->state;
  destination->identity.forward_epoch = source->identity.forward_epoch;
  destination->identity.graph_replay_generation =
      source->identity.graph_replay_generation;
  destination->identity.layer_control_index =
      source->identity.layer_control_index;
  destination->identity.queue_id = source->identity.queue_id;
  destination->identity.queue_generation = source->identity.queue_generation;
  destination->identity.post_sequence = source->identity.post_sequence;
  destination->identity.owner_tag = source->identity.owner_tag;
  destination->identity.reserved0 = source->identity.reserved0;
  destination->sq_producer_before = source->sq_producer_before;
  destination->cq_consumer_before = source->cq_consumer_before;
  destination->sq_producer_after = source->sq_producer_after;
  destination->cq_consumer_after = source->cq_consumer_after;
  destination->request_count = source->request_count;
  destination->outcome = source->outcome;
  destination->private_ready_published = source->private_ready_published;
  destination->post_committed = source->post_committed;
  destination->transaction.active_count = source->transaction.active_count;
  destination->transaction.total_posts = source->transaction.total_posts;
  destination->transaction.failed_request = source->transaction.failed_request;
  destination->transaction.last_wqe = source->transaction.last_wqe;
  destination->transaction.first_doorbell_cycle =
      source->transaction.first_doorbell_cycle;
  destination->transaction.profile_start_cycle =
      source->transaction.profile_start_cycle;
  destination->transaction.profile_setup_done_cycle =
      source->transaction.profile_setup_done_cycle;
  destination->transaction.profile_compact_done_cycle =
      source->transaction.profile_compact_done_cycle;
  destination->transaction.profile_post_done_cycle =
      source->transaction.profile_post_done_cycle;
  destination->transaction.profile_completion_done_cycle =
      source->transaction.profile_completion_done_cycle;
  destination->transaction.result = source->transaction.result;
  destination->transaction.completion_polls =
      source->transaction.completion_polls;
  destination->transaction.state = source->transaction.state;
  destination->transaction.profile_enabled = source->transaction.profile_enabled;
  destination->transaction.finalized = source->transaction.finalized;
  for (uint32_t index = 0U; index < 5U; ++index) {
    destination->reserved[index] = source->reserved[index];
  }
  return true;
}

__aicore__ inline bool StoreKvafP8WaveRecordToGm(
    __gm__ KvafP8WaveRecord* destination, const KvafP8WaveRecord& source) {
  if (destination == nullptr) {
    return false;
  }
  destination->magic = source.magic;
  destination->version = source.version;
  destination->state = source.state;
  destination->identity.forward_epoch = source.identity.forward_epoch;
  destination->identity.graph_replay_generation =
      source.identity.graph_replay_generation;
  destination->identity.layer_control_index = source.identity.layer_control_index;
  destination->identity.queue_id = source.identity.queue_id;
  destination->identity.queue_generation = source.identity.queue_generation;
  destination->identity.post_sequence = source.identity.post_sequence;
  destination->identity.owner_tag = source.identity.owner_tag;
  destination->identity.reserved0 = source.identity.reserved0;
  destination->sq_producer_before = source.sq_producer_before;
  destination->cq_consumer_before = source.cq_consumer_before;
  destination->sq_producer_after = source.sq_producer_after;
  destination->cq_consumer_after = source.cq_consumer_after;
  destination->request_count = source.request_count;
  destination->outcome = source.outcome;
  destination->private_ready_published = source.private_ready_published;
  destination->post_committed = source.post_committed;
  destination->transaction.active_count = source.transaction.active_count;
  destination->transaction.total_posts = source.transaction.total_posts;
  destination->transaction.failed_request = source.transaction.failed_request;
  destination->transaction.last_wqe = source.transaction.last_wqe;
  destination->transaction.first_doorbell_cycle =
      source.transaction.first_doorbell_cycle;
  destination->transaction.profile_start_cycle =
      source.transaction.profile_start_cycle;
  destination->transaction.profile_setup_done_cycle =
      source.transaction.profile_setup_done_cycle;
  destination->transaction.profile_compact_done_cycle =
      source.transaction.profile_compact_done_cycle;
  destination->transaction.profile_post_done_cycle =
      source.transaction.profile_post_done_cycle;
  destination->transaction.profile_completion_done_cycle =
      source.transaction.profile_completion_done_cycle;
  destination->transaction.result = source.transaction.result;
  destination->transaction.completion_polls = source.transaction.completion_polls;
  destination->transaction.state = source.transaction.state;
  destination->transaction.profile_enabled = source.transaction.profile_enabled;
  destination->transaction.finalized = source.transaction.finalized;
  for (uint32_t index = 0U; index < 5U; ++index) {
    destination->reserved[index] = source.reserved[index];
  }
  return true;
}

__aicore__ inline void ResetKvafP8PrivateRecord(
    KvafP8PrivateRecord* record) {
  if (record == nullptr) {
    return;
  }
  record->magic = kKvafP8WaveRecordMagic;
  record->version = kKvafP8WaveRecordVersion;
  record->state = kKvafP8PrivateFree;
  record->forward_epoch = 0U;
  record->graph_replay_generation = 0U;
  record->layer_control_index = 0U;
  record->queue_id = 0U;
  record->queue_generation = 0U;
  record->post_sequence = 0U;
  record->source_offset = 0U;
  record->object_offset = 0U;
  record->output_index = -1;
  record->owner_tag = 0;
}

__aicore__ inline bool IsKvafP8PrivateRecordValid(
    const KvafP8PrivateRecord& record) {
  return record.magic == kKvafP8WaveRecordMagic &&
         record.version == kKvafP8WaveRecordVersion &&
         record.state <= kKvafP8PrivateQuarantined;
}

// Keep all GM/local transfers field-wise.  A P8 drain phase reads an old
// record on a later graph launch; aggregate copying would either be rejected
// by AscendC or make this publication edge architecture-dependent.
__aicore__ inline bool LoadKvafP8PrivateRecordFromGm(
    __gm__ const KvafP8PrivateRecord* source,
    KvafP8PrivateRecord* destination) {
  if (source == nullptr || destination == nullptr) {
    return false;
  }
  destination->magic = source->magic;
  destination->version = source->version;
  destination->state = source->state;
  destination->forward_epoch = source->forward_epoch;
  destination->graph_replay_generation = source->graph_replay_generation;
  destination->layer_control_index = source->layer_control_index;
  destination->queue_id = source->queue_id;
  destination->queue_generation = source->queue_generation;
  destination->post_sequence = source->post_sequence;
  destination->source_offset = source->source_offset;
  destination->object_offset = source->object_offset;
  destination->output_index = source->output_index;
  destination->owner_tag = source->owner_tag;
  return true;
}

__aicore__ inline bool StoreKvafP8PrivateRecordToGm(
    __gm__ KvafP8PrivateRecord* destination,
    const KvafP8PrivateRecord& source) {
  if (destination == nullptr) {
    return false;
  }
  destination->magic = source.magic;
  destination->version = source.version;
  destination->state = source.state;
  destination->forward_epoch = source.forward_epoch;
  destination->graph_replay_generation = source.graph_replay_generation;
  destination->layer_control_index = source.layer_control_index;
  destination->queue_id = source.queue_id;
  destination->queue_generation = source.queue_generation;
  destination->post_sequence = source.post_sequence;
  destination->source_offset = source.source_offset;
  destination->object_offset = source.object_offset;
  destination->output_index = source.output_index;
  destination->owner_tag = source.owner_tag;
  return true;
}

__aicore__ inline bool PrepareKvafP8PrivateRecord(
    const KvafP8WaveIdentity& identity, uint64_t source_offset,
    uint64_t object_offset, int32_t output_index,
    KvafP8PrivateRecord* record) {
  if (record == nullptr || identity.forward_epoch == 0U ||
      identity.owner_tag <= 0 || identity.queue_generation == 0U ||
      identity.post_sequence == 0U || output_index < 0) {
    if (record != nullptr) {
      ResetKvafP8PrivateRecord(record);
      record->state = kKvafP8PrivateQuarantined;
    }
    return false;
  }
  ResetKvafP8PrivateRecord(record);
  record->state = kKvafP8PrivatePending;
  record->forward_epoch = identity.forward_epoch;
  record->graph_replay_generation = identity.graph_replay_generation;
  record->layer_control_index = identity.layer_control_index;
  record->queue_id = identity.queue_id;
  record->queue_generation = identity.queue_generation;
  record->post_sequence = identity.post_sequence;
  record->source_offset = source_offset;
  record->object_offset = object_offset;
  record->output_index = output_index;
  record->owner_tag = identity.owner_tag;
  return true;
}

__aicore__ inline bool KvafP8PrivateRecordMatchesWave(
    const KvafP8PrivateRecord& private_record,
    const KvafP8WaveRecord& wave_record) {
  return IsKvafP8PrivateRecordValid(private_record) &&
         IsKvafP8WaveRecordValid(wave_record) &&
         private_record.forward_epoch == wave_record.identity.forward_epoch &&
         private_record.graph_replay_generation ==
             wave_record.identity.graph_replay_generation &&
         private_record.layer_control_index ==
             wave_record.identity.layer_control_index &&
         private_record.queue_id == wave_record.identity.queue_id &&
         private_record.queue_generation ==
             wave_record.identity.queue_generation &&
         private_record.post_sequence == wave_record.identity.post_sequence &&
         private_record.owner_tag == wave_record.identity.owner_tag;
}

// P8 validates a candidate descriptor against the immutable identity that
// will be written into the forward-owned wave record.  Keep this separate
// from `KvafP8PrivateRecordMatchesWave`: Prepare/Post/Retire/Finish must
// validate the descriptor before there is necessarily a local wave record
// whose state is safe to inspect.
__aicore__ inline bool KvafP8PrivateRecordMatchesIdentity(
    const KvafP8PrivateRecord& private_record,
    const KvafP8WaveIdentity& identity) {
  return IsKvafP8PrivateRecordValid(private_record) &&
         private_record.forward_epoch == identity.forward_epoch &&
         private_record.graph_replay_generation ==
             identity.graph_replay_generation &&
         private_record.layer_control_index == identity.layer_control_index &&
         private_record.queue_id == identity.queue_id &&
         private_record.queue_generation == identity.queue_generation &&
         private_record.post_sequence == identity.post_sequence &&
         private_record.owner_tag == identity.owner_tag;
}

__aicore__ inline bool FinalizeKvafP8PrivateRecord(
    KvafP8PrivateRecord* private_record,
    const KvafP8WaveRecord& wave_record, bool payload_ready) {
  if (private_record == nullptr ||
      private_record->state != kKvafP8PrivatePending ||
      !KvafP8PrivateRecordMatchesWave(*private_record, wave_record)) {
    if (private_record != nullptr) {
      private_record->state = kKvafP8PrivateQuarantined;
    }
    return false;
  }
  private_record->state = payload_ready ? kKvafP8PrivateReady
                                         : kKvafP8PrivateFailed;
  return true;
}

// Only the same-stream QSFA final reader may clear this record.  It is legal
// to release the Direct-HCCP queue long before this point because the queue's
// descriptor/CQ lifetime ended at Finalized; this record protects the HBM
// payload and its source/object mapping independently.
__aicore__ inline bool ReleaseKvafP8PrivateRecordAfterQsfaTerminal(
    KvafP8PrivateRecord* private_record,
    const KvafP8ForwardControl& forward_control) {
  if (private_record == nullptr || !IsKvafP8PrivateRecordValid(*private_record) ||
      (private_record->state != kKvafP8PrivateReady &&
       private_record->state != kKvafP8PrivateFailed) ||
      !IsKvafP8ForwardControlValid(forward_control) ||
      forward_control.phase != kKvafP8ForwardPhaseQsfaTerminal ||
      !IsKvafP8PrivateViewReadyForQsfa(forward_control) ||
      (forward_control.flags & kKvafP8ForwardFlagQsfaTerminalObserved) == 0U ||
      private_record->forward_epoch != forward_control.forward_epoch ||
      private_record->graph_replay_generation !=
          forward_control.graph_replay_generation ||
      private_record->layer_control_index != forward_control.layer_control_index) {
    if (private_record != nullptr && IsKvafP8PrivateRecordValid(*private_record)) {
      private_record->state = kKvafP8PrivateQuarantined;
    }
    return false;
  }
  ResetKvafP8PrivateRecord(private_record);
  return true;
}

// AscendC treats GM and local memory as distinct C++ address spaces.  A
// compiler-generated struct copy such as `local = *gm` is rejected even when
// every field is a scalar.  Keep the control ABI identical, but transfer it
// field-by-field at the explicit GM/local boundary.  There are two entry
// points because a SIMT service must call a SIMT callee, while LI's scalar
// hook runs on AIC.
__aicore__ inline bool LoadKvafStreamingBatchGetControlFromGmForAicore(
    __gm__ const KvafStreamingBatchGetControl* source,
    KvafStreamingBatchGetControl* destination) {
  if (source == nullptr || destination == nullptr) {
    return false;
  }
  destination->magic = source->magic;
  destination->version = source->version;
  destination->channel_entity = source->channel_entity;
  destination->queue_state = source->queue_state;
  destination->core_status = source->core_status;
  destination->output = source->output;
  destination->queue_leases = source->queue_leases;
  destination->descriptor_scratch = source->descriptor_scratch;
  destination->queue_count = source->queue_count;
  destination->output_rows = source->output_rows;
  destination->private_output_stride_bytes = source->private_output_stride_bytes;
  destination->output_offset_bytes = source->output_offset_bytes;
  destination->jettys_per_core = source->jettys_per_core;
  destination->block_bytes = source->block_bytes;
  destination->descriptor_templates = source->descriptor_templates;
  destination->descriptor_template_count = source->descriptor_template_count;
  return true;
}

__simt_callee__ inline bool LoadKvafStreamingBatchGetControlFromGmForSimt(
    __gm__ const KvafStreamingBatchGetControl* source,
    KvafStreamingBatchGetControl* destination) {
  if (source == nullptr || destination == nullptr) {
    return false;
  }
  destination->magic = source->magic;
  destination->version = source->version;
  destination->channel_entity = source->channel_entity;
  destination->queue_state = source->queue_state;
  destination->core_status = source->core_status;
  destination->output = source->output;
  destination->queue_leases = source->queue_leases;
  destination->descriptor_scratch = source->descriptor_scratch;
  destination->queue_count = source->queue_count;
  destination->output_rows = source->output_rows;
  destination->private_output_stride_bytes = source->private_output_stride_bytes;
  destination->output_offset_bytes = source->output_offset_bytes;
  destination->jettys_per_core = source->jettys_per_core;
  destination->block_bytes = source->block_bytes;
  destination->descriptor_templates = source->descriptor_templates;
  destination->descriptor_template_count = source->descriptor_template_count;
  return true;
}

__aicore__ inline bool LoadKvafStreamingBatchGetDescriptorScratchFromGm(
    __gm__ const KvafStreamingBatchGetDescriptorScratch* source,
    KvafStreamingBatchGetDescriptorScratch* destination) {
  if (source == nullptr || destination == nullptr) {
    return false;
  }
  destination->address_table = source->address_table;
  destination->valid_table = source->valid_table;
  destination->indices = source->indices;
  destination->offsets = source->offsets;
  destination->mask = source->mask;
  destination->output_indices = source->output_indices;
  destination->table_capacity = source->table_capacity;
  destination->request_capacity = source->request_capacity;
  destination->active_request_count = source->active_request_count;
  destination->private_output_row_base = source->private_output_row_base;
  return true;
}

__aicore__ inline bool StoreKvafStreamingBatchGetDescriptorScratchToGm(
    __gm__ KvafStreamingBatchGetDescriptorScratch* destination,
    const KvafStreamingBatchGetDescriptorScratch& source) {
  if (destination == nullptr) {
    return false;
  }
  destination->address_table = source.address_table;
  destination->valid_table = source.valid_table;
  destination->indices = source.indices;
  destination->offsets = source.offsets;
  destination->mask = source.mask;
  destination->output_indices = source.output_indices;
  destination->table_capacity = source.table_capacity;
  destination->request_capacity = source.request_capacity;
  destination->active_request_count = source.active_request_count;
  destination->private_output_row_base = source.private_output_row_base;
  return true;
}

__simt_callee__ inline bool IsKvafStreamingBatchGetControlValidForLease(
    const KvafStreamingBatchGetControl& control) {
  return control.magic == KVAF_STREAMING_BATCH_GET_CONTROL_MAGIC &&
         control.version == KVAF_STREAMING_BATCH_GET_CONTROL_VERSION &&
         control.channel_entity != 0U && control.queue_state != 0U &&
         control.core_status != 0U && control.output != 0U &&
         control.queue_leases != 0U && control.descriptor_scratch != 0U &&
         control.descriptor_templates != 0U &&
         control.queue_count != 0U &&
         control.queue_count <= KVAF_AIV_BATCH_GET_MAX_AIV_CORES &&
         control.descriptor_template_count == control.queue_count &&
         control.output_rows != 0U &&
         control.private_output_stride_bytes != 0U &&
         control.output_offset_bytes == 0U &&
         control.private_output_stride_bytes ==
             (control.output_rows /
              control.queue_count) * KVAF_AIV_BATCH_GET_WQE_DATA_BYTES &&
         control.output_rows % control.queue_count == 0U &&
         control.jettys_per_core == 1U &&
         control.block_bytes == KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;
}

// The AIC scalar path cannot call a SIMT callee directly. Keep the same
// validation at that boundary rather than relying on a compiler-specific
// cross-pipeline call convention.
__aicore__ inline bool IsKvafStreamingBatchGetControlValidForAicore(
    const KvafStreamingBatchGetControl& control) {
  return control.magic == KVAF_STREAMING_BATCH_GET_CONTROL_MAGIC &&
         control.version == KVAF_STREAMING_BATCH_GET_CONTROL_VERSION &&
         control.channel_entity != 0U && control.queue_state != 0U &&
         control.core_status != 0U && control.output != 0U &&
         control.queue_leases != 0U && control.descriptor_scratch != 0U &&
         control.descriptor_templates != 0U &&
         control.queue_count != 0U &&
         control.queue_count <= KVAF_AIV_BATCH_GET_MAX_AIV_CORES &&
         control.descriptor_template_count == control.queue_count &&
         control.output_rows != 0U &&
         control.private_output_stride_bytes != 0U &&
         control.output_offset_bytes == 0U &&
         control.private_output_stride_bytes ==
             (control.output_rows /
              control.queue_count) * KVAF_AIV_BATCH_GET_WQE_DATA_BYTES &&
         control.output_rows % control.queue_count == 0U &&
         control.jettys_per_core == 1U &&
         control.block_bytes == KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;
}

// TryAcquire is a SIMT-safe, non-blocking claim. The producer scans every
// queue at most once starting from `start_hint`; false means the pool is full
// and the LI kernel must retain/skip this speculative candidate rather than
// spin. `producer_id` is not a physical core id and may be any stable LI task
// ordinal below INT32_MAX - 1.
__simt_callee__ inline bool TryAcquireKvafStreamingBatchGetQueueFromSimt(
    const KvafStreamingBatchGetControl& control, uint32_t producer_id,
    uint32_t start_hint, KvafStreamingBatchGetLease* lease) {
  if (lease == nullptr ||
      !IsKvafStreamingBatchGetControlValidForLease(control) ||
      producer_id >= 0x7ffffffeU) {
    return false;
  }
  const int32_t owner_tag = static_cast<int32_t>(producer_id + 1U);
  const uint32_t queue_count = static_cast<uint32_t>(control.queue_count);
  const uint32_t first_queue = start_hint % queue_count;
  __gm__ int32_t* const owners =
      reinterpret_cast<__gm__ int32_t*>(control.queue_leases);
  for (uint32_t probe = 0U; probe < queue_count; ++probe) {
    const uint32_t queue_id = (first_queue + probe) % queue_count;
    if (asc_atomic_cas(owners + queue_id, 0, owner_tag) == 0) {
      // Pair the release fence before a previous owner's CAS-to-zero with
      // this producer's first access to the queue's state and CQ memory.
      AscendC::Simt::ThreadFence();
      lease->queue_id = queue_id;
      lease->owner_tag = owner_tag;
      lease->state = kStreamingBatchGetLeaseHeld;
      lease->reserved0 = 0U;
      return true;
    }
  }
  return false;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafStreamingBatchGetAcquireSimtService(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t start_hint,
    __ubuf__ KvafStreamingBatchGetLeaseResult* result) {
  if (result == nullptr || control_gm == nullptr) {
    return;
  }
  KvafStreamingBatchGetControl control{};
  if (!LoadKvafStreamingBatchGetControlFromGmForSimt(control_gm, &control)) {
    return;
  }
  KvafStreamingBatchGetLease lease{};
  if (TryAcquireKvafStreamingBatchGetQueueFromSimt(
          control, producer_id, start_hint, &lease)) {
    result->queue_id = lease.queue_id;
    result->owner_tag = lease.owner_tag;
    result->success = 1U;
  }
}

// Supported AIC-scalar entry point for LI's ProcessVec1/ProcessTopK-side
// hook. `lease_result` must be caller-owned UB storage. `simt_event` must be
// an event id that the including LI operator reserves exclusively for this
// acquire/release service; it must not overlap a live vector/MTE dependency.
// The VF_CALL + PIPE_S/PIPE_V hand-off is required: a direct AIC call to
// asc_atomic_cas is not supported by the CANN MIX compilation model.
__aicore__ inline bool TryAcquireKvafStreamingBatchGetQueueFromAicore(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t start_hint,
    __ubuf__ KvafStreamingBatchGetLeaseResult* lease_result,
    event_t simt_event, KvafStreamingBatchGetLease* lease) {
  if (control_gm == nullptr || lease_result == nullptr || lease == nullptr) {
    return false;
  }
  lease_result->queue_id = 0U;
  lease_result->owner_tag = 0;
  lease_result->success = 0U;
  lease_result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafStreamingBatchGetAcquireSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, producer_id, start_hint,
      lease_result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (lease_result->success == 0U || lease_result->owner_tag <= 0) {
    return false;
  }
  lease->queue_id = lease_result->queue_id;
  lease->owner_tag = lease_result->owner_tag;
  lease->state = kStreamingBatchGetLeaseHeld;
  lease->reserved0 = 0U;
  return true;
}

// Return the exclusively owned per-queue descriptor record. ExactScan writes
// its already-dense candidate arrays and final private-output indices here
// before Begin. It must not write a persistent HBM block table/cache slot;
// this record only drives a speculative private-slot Get wave.
__aicore__ inline __gm__ KvafStreamingBatchGetDescriptorScratch*
GetKvafStreamingBatchGetDescriptorScratch(
    const KvafStreamingBatchGetControl& control, uint32_t queue_id) {
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      queue_id >= control.queue_count) {
    return nullptr;
  }
  return reinterpret_cast<__gm__ KvafStreamingBatchGetDescriptorScratch*>(
      control.descriptor_scratch) + queue_id;
}

// Return the immutable, layer-private template populated by the cold host
// ConfigureStreamingBatchGetQueueTemplates lifecycle call. The active
// descriptor_scratch table remains shared by the lease pool; therefore an LI
// producer must copy this template only after it has acquired its queue.
__aicore__ inline __gm__ const KvafStreamingBatchGetDescriptorScratch*
GetKvafStreamingBatchGetDescriptorTemplate(
    const KvafStreamingBatchGetControl& control, uint32_t queue_id) {
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      queue_id >= control.descriptor_template_count) {
    return nullptr;
  }
  return reinterpret_cast<__gm__ const KvafStreamingBatchGetDescriptorScratch*>(
      control.descriptor_templates) + queue_id;
}

// Load a queue's fixed template into caller-local storage. ExactScan then
// writes only the template-owned candidate arrays and active_request_count;
// Publish verifies that those graph-stable pointer/capacity fields were not
// substituted on the replay path.
__aicore__ inline bool LoadKvafStreamingBatchGetQueueTemplate(
    const KvafStreamingBatchGetControl& control,
    const KvafStreamingBatchGetLease& lease,
    KvafStreamingBatchGetDescriptorScratch* descriptor) {
  if (descriptor == nullptr || lease.state != kStreamingBatchGetLeaseHeld ||
      lease.owner_tag <= 0 || lease.queue_id >= control.queue_count) {
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners[lease.queue_id] != lease.owner_tag) {
    return false;
  }
  __gm__ const KvafStreamingBatchGetDescriptorScratch* const source =
      GetKvafStreamingBatchGetDescriptorTemplate(control, lease.queue_id);
  if (source == nullptr) {
    return false;
  }
  if (!LoadKvafStreamingBatchGetDescriptorScratchFromGm(source, descriptor)) {
    return false;
  }
  // A configured template intentionally has active_request_count == 0. It
  // becomes a valid wave only after the producer has written its own compact
  // candidate arrays and sets a nonzero count before Publish.
  return descriptor->address_table != 0U && descriptor->valid_table != 0U &&
         descriptor->indices != 0U && descriptor->offsets != 0U &&
         descriptor->mask != 0U && descriptor->output_indices != 0U &&
         descriptor->table_capacity != 0U &&
         descriptor->request_capacity != 0U &&
         descriptor->request_capacity <= kDenseRequestCapacity &&
         descriptor->active_request_count == 0U;
}

// Publish one dense descriptor record while holding its lease.
// `output_indices` must already contain absolute rows in this queue's private
// output arena; Begin does no late remapping. The expected row base is
// q * private_output_rows. Taking the lease rather than a bare queue id keeps
// a non-owner from overwriting another producer's descriptor scratch.
// `mask` is retained in the scratch ABI for diagnostics/compatibility but is
// not consumed by the dense post helper: ExactScan has already compacted the
// wave, so active_request_count is its exact valid count.
__aicore__ inline bool PublishKvafStreamingBatchGetDescriptorScratch(
    const KvafStreamingBatchGetControl& control,
    const KvafStreamingBatchGetLease& lease,
    const KvafStreamingBatchGetDescriptorScratch& descriptor) {
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      lease.state != kStreamingBatchGetLeaseHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count || descriptor.address_table == 0U ||
      descriptor.valid_table == 0U || descriptor.indices == 0U ||
      descriptor.offsets == 0U || descriptor.mask == 0U ||
      descriptor.output_indices == 0U ||
      descriptor.table_capacity == 0U || descriptor.request_capacity == 0U ||
      descriptor.request_capacity > kDenseRequestCapacity ||
      descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity) {
    return false;
  }
  const uint32_t queue_id = lease.queue_id;
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners[queue_id] != lease.owner_tag) {
    return false;
  }
  __gm__ const KvafStreamingBatchGetDescriptorScratch* const template_record =
      GetKvafStreamingBatchGetDescriptorTemplate(control, queue_id);
  if (template_record == nullptr || template_record->address_table == 0U ||
      template_record->valid_table == 0U || template_record->indices == 0U ||
      template_record->offsets == 0U || template_record->mask == 0U ||
      template_record->output_indices == 0U ||
      template_record->table_capacity != descriptor.table_capacity ||
      template_record->request_capacity != descriptor.request_capacity ||
      template_record->address_table != descriptor.address_table ||
      template_record->valid_table != descriptor.valid_table ||
      template_record->indices != descriptor.indices ||
      template_record->offsets != descriptor.offsets ||
      template_record->mask != descriptor.mask ||
      template_record->output_indices != descriptor.output_indices ||
      template_record->private_output_row_base !=
          descriptor.private_output_row_base) {
    return false;
  }
  const uint64_t private_output_rows =
      control.private_output_stride_bytes /
      KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;
  const uint64_t expected_private_base =
      static_cast<uint64_t>(queue_id) * private_output_rows;
  if (private_output_rows == 0U ||
      expected_private_base > control.output_rows ||
      private_output_rows > control.output_rows - expected_private_base ||
      descriptor.private_output_row_base != expected_private_base) {
    return false;
  }
  __gm__ KvafStreamingBatchGetDescriptorScratch* const scratch =
      GetKvafStreamingBatchGetDescriptorScratch(control, queue_id);
  if (!StoreKvafStreamingBatchGetDescriptorScratchToGm(scratch, descriptor)) {
    return false;
  }
  // The following Begin may hand the queue to Direct-HCCP immediately. Keep
  // descriptor publication ordered before its SQ WQE materialization.
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

// The legacy Direct-HCCP primitive validates an output index only against
// the whole registered output. Streaming adds this queue-private range check
// before the doorbell. It is intentionally a bounded scan over the already
// compacted wave (at most kDenseRequestCapacity entries), not a cache-table
// write or a persistent mapping update.
__aicore__ inline bool ValidateKvafStreamingBatchGetPrivateOutputIndices(
    const KvafStreamingBatchGetControl& control, uint32_t queue_id,
    const KvafStreamingBatchGetDescriptorScratch& descriptor) {
  const uint64_t private_output_rows =
      control.private_output_stride_bytes /
      KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;
  const uint64_t private_base = descriptor.private_output_row_base;
  if (private_output_rows == 0U ||
      private_base != static_cast<uint64_t>(queue_id) * private_output_rows ||
      private_base > control.output_rows ||
      private_output_rows > control.output_rows - private_base) {
    return false;
  }
  __gm__ const int32_t* const output_indices =
      reinterpret_cast<__gm__ const int32_t*>(descriptor.output_indices);
  for (uint32_t request = 0U; request < descriptor.active_request_count;
       ++request) {
    const int32_t output_index = output_indices[request];
    if (output_index < 0 ||
        static_cast<uint64_t>(output_index) < private_base ||
        static_cast<uint64_t>(output_index) >=
            private_base + private_output_rows) {
      return false;
    }
  }
  return true;
}

// P8 deliberately does not inherit the legacy queue-private output-row
// partition.  Once a CQ is terminal, P8 returns the scarce HCCP queue to the
// dynamic lease pool while the payload remains protected by a separately
// claimed private slot until QSFA's last reader.  Therefore a P8 descriptor
// may target any absolute row in the one registered output allocation.
//
// This is not a relaxed validation path.  Every output row is backed by one
// fixed private record, and that record must be PENDING for the exact
// forward/replay/layer/queue/generation/post identity about to be posted.
// The bounded duplicate scan prevents one descriptor from issuing two writes
// to the same private row.  It is at most 256 entries and runs only at the
// post boundary, never in the score-inner loop.
__aicore__ inline bool ValidateKvafP8PrivateOutputIndices(
    const KvafStreamingBatchGetControl& control,
    const KvafStreamingBatchGetDescriptorScratch& descriptor,
    const KvafP8WaveIdentity& identity,
    __gm__ const KvafP8PrivateRecord* private_records,
    uint64_t private_record_count) {
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      private_records == nullptr || private_record_count != control.output_rows ||
      identity.forward_epoch == 0U || identity.owner_tag <= 0 ||
      identity.queue_id >= control.queue_count ||
      identity.queue_generation == 0U || identity.post_sequence == 0U ||
      descriptor.output_indices == 0U || descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity ||
      descriptor.request_capacity > kDenseRequestCapacity) {
    return false;
  }
  __gm__ const int32_t* const output_indices =
      reinterpret_cast<__gm__ const int32_t*>(descriptor.output_indices);
  for (uint32_t request = 0U; request < descriptor.active_request_count;
       ++request) {
    const int32_t output_index = output_indices[request];
    if (output_index < 0 ||
        static_cast<uint64_t>(output_index) >= control.output_rows) {
      return false;
    }
    for (uint32_t prior = 0U; prior < request; ++prior) {
      if (output_indices[prior] == output_index) {
        return false;
      }
    }
    KvafP8PrivateRecord private_record{};
    if (!LoadKvafP8PrivateRecordFromGm(
            private_records + static_cast<uint64_t>(output_index),
            &private_record) ||
        private_record.state != kKvafP8PrivatePending ||
        private_record.output_index != output_index ||
        !KvafP8PrivateRecordMatchesIdentity(private_record, identity)) {
      return false;
    }
  }
  return true;
}

// P8-only descriptor publication companion.  Legacy streaming callers retain
// `PublishKvafStreamingBatchGetDescriptorScratch` and its queue-private range
// check unchanged.  P8 instead binds the descriptor to its claimed private
// records before making it visible to Direct-HCCP.
__aicore__ inline bool PublishKvafP8BatchGetDescriptorScratch(
    const KvafStreamingBatchGetControl& control,
    const KvafStreamingBatchGetLease& lease,
    const KvafP8WaveIdentity& identity,
    __gm__ const KvafP8PrivateRecord* private_records,
    uint64_t private_record_count,
    const KvafStreamingBatchGetDescriptorScratch& descriptor) {
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      lease.state != kStreamingBatchGetLeaseHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count || identity.queue_id != lease.queue_id ||
      identity.owner_tag != lease.owner_tag || descriptor.address_table == 0U ||
      descriptor.valid_table == 0U || descriptor.indices == 0U ||
      descriptor.offsets == 0U || descriptor.mask == 0U ||
      descriptor.output_indices == 0U || descriptor.table_capacity == 0U ||
      descriptor.request_capacity == 0U ||
      descriptor.request_capacity > kDenseRequestCapacity ||
      descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity) {
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners == nullptr || owners[lease.queue_id] != lease.owner_tag) {
    return false;
  }
  __gm__ const KvafStreamingBatchGetDescriptorScratch* const template_record =
      GetKvafStreamingBatchGetDescriptorTemplate(control, lease.queue_id);
  if (template_record == nullptr || template_record->address_table == 0U ||
      template_record->valid_table == 0U || template_record->indices == 0U ||
      template_record->offsets == 0U || template_record->mask == 0U ||
      template_record->output_indices == 0U ||
      template_record->table_capacity != descriptor.table_capacity ||
      template_record->request_capacity != descriptor.request_capacity ||
      template_record->address_table != descriptor.address_table ||
      template_record->valid_table != descriptor.valid_table ||
      template_record->indices != descriptor.indices ||
      template_record->offsets != descriptor.offsets ||
      template_record->mask != descriptor.mask ||
      template_record->output_indices != descriptor.output_indices ||
      template_record->private_output_row_base !=
          descriptor.private_output_row_base ||
      !ValidateKvafP8PrivateOutputIndices(
          control, descriptor, identity, private_records,
          private_record_count)) {
    return false;
  }
  __gm__ KvafStreamingBatchGetDescriptorScratch* const scratch =
      GetKvafStreamingBatchGetDescriptorScratch(control, lease.queue_id);
  if (!StoreKvafStreamingBatchGetDescriptorScratchToGm(scratch, descriptor)) {
    return false;
  }
  // Pair the private-record/descriptor publication with the later HCCP WQE
  // materialization.  The P8 caller stores its wave record and performs the
  // same DDR barrier before PostOnly, so no consumer can observe a half-built
  // descriptor or an unowned output row.
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

// Build the existing single-queue Direct-HCCP view without changing the
// normal BatchGet kernel ABI. Descriptor pointers/count are read from the
// queue-private scratch record; output/entity/state/status are derived from
// the graph-stable streaming control plus the acquired queue id.
__aicore__ inline bool BuildKvafStreamingBatchGetQueueArgs(
    const KvafStreamingBatchGetControl& control, uint32_t queue_id,
    KvafExecuteBatchGetKernelArgs* queue_args,
    uint32_t* active_request_count) {
  if (queue_args == nullptr ||
      !IsKvafStreamingBatchGetControlValidForAicore(control) ||
      queue_id >= control.queue_count || active_request_count == nullptr) {
    return false;
  }
  __gm__ const KvafStreamingBatchGetDescriptorScratch* const scratch =
      GetKvafStreamingBatchGetDescriptorScratch(control, queue_id);
  KvafStreamingBatchGetDescriptorScratch descriptor{};
  if (!LoadKvafStreamingBatchGetDescriptorScratchFromGm(scratch, &descriptor) ||
      descriptor.address_table == 0U || descriptor.valid_table == 0U ||
      descriptor.indices == 0U || descriptor.offsets == 0U ||
      descriptor.mask == 0U || descriptor.output_indices == 0U ||
      descriptor.table_capacity == 0U || descriptor.request_capacity == 0U ||
      descriptor.request_capacity > kDenseRequestCapacity ||
      descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity ||
      control.output_rows > UINT32_MAX ||
      !ValidateKvafStreamingBatchGetPrivateOutputIndices(
          control, queue_id, descriptor)) {
    return false;
  }

  queue_args->address_table = descriptor.address_table;
  queue_args->valid_table = descriptor.valid_table;
  queue_args->indices = descriptor.indices;
  queue_args->offsets = descriptor.offsets;
  queue_args->mask = descriptor.mask;
  queue_args->output_indices = descriptor.output_indices;
  // Keep the full registration base to satisfy the unchanged Direct-HCCP
  // entity check. The descriptor's validated absolute output indices select
  // this queue's private slot arena inside that allocation.
  queue_args->output = control.output;
  queue_args->core_status =
      control.core_status + queue_id * sizeof(KvafBatchGetCoreStatus);
  queue_args->channel_entity =
      control.channel_entity + queue_id * sizeof(KvafBatchGetDirectUrmaEntity);
  queue_args->queue_state =
      control.queue_state + queue_id * sizeof(KvafBatchGetQueueState);
  queue_args->table_capacity = descriptor.table_capacity;
  queue_args->request_capacity = descriptor.request_capacity;
  queue_args->output_rows = static_cast<uint32_t>(control.output_rows);
  queue_args->block_bytes = static_cast<uint32_t>(control.block_bytes);
  queue_args->block_count = 1U;
  queue_args->jettys_per_core = 1U;
  *active_request_count = descriptor.active_request_count;
  return true;
}

// P8-only equivalent of the legacy queue-argument builder.  It preserves the
// unmodified Direct-HCCP args/entity ABI, including the full registered HBM
// output base.  The sole semantic difference is deliberate: output_indices
// are validated against a forward-owned private-record arena rather than the
// legacy queue-private row interval.  The legacy builder above remains the
// only path used by existing streaming callers.
__aicore__ inline bool BuildKvafP8StreamingBatchGetQueueArgs(
    const KvafStreamingBatchGetControl& control, uint32_t queue_id,
    const KvafP8WaveIdentity& identity,
    __gm__ const KvafP8PrivateRecord* private_records,
    uint64_t private_record_count,
    KvafExecuteBatchGetKernelArgs* queue_args,
    uint32_t* active_request_count) {
  if (queue_args == nullptr || active_request_count == nullptr ||
      !IsKvafStreamingBatchGetControlValidForAicore(control) ||
      queue_id >= control.queue_count || identity.queue_id != queue_id ||
      identity.forward_epoch == 0U || identity.owner_tag <= 0 ||
      identity.queue_generation == 0U || identity.post_sequence == 0U) {
    return false;
  }
  __gm__ const KvafStreamingBatchGetDescriptorScratch* const scratch =
      GetKvafStreamingBatchGetDescriptorScratch(control, queue_id);
  KvafStreamingBatchGetDescriptorScratch descriptor{};
  if (!LoadKvafStreamingBatchGetDescriptorScratchFromGm(scratch, &descriptor) ||
      descriptor.address_table == 0U || descriptor.valid_table == 0U ||
      descriptor.indices == 0U || descriptor.offsets == 0U ||
      descriptor.mask == 0U || descriptor.output_indices == 0U ||
      descriptor.table_capacity == 0U || descriptor.request_capacity == 0U ||
      descriptor.request_capacity > kDenseRequestCapacity ||
      descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity ||
      control.output_rows > UINT32_MAX ||
      !ValidateKvafP8PrivateOutputIndices(
          control, descriptor, identity, private_records,
          private_record_count)) {
    return false;
  }

  queue_args->address_table = descriptor.address_table;
  queue_args->valid_table = descriptor.valid_table;
  queue_args->indices = descriptor.indices;
  queue_args->offsets = descriptor.offsets;
  queue_args->mask = descriptor.mask;
  queue_args->output_indices = descriptor.output_indices;
  queue_args->output = control.output;
  queue_args->core_status =
      control.core_status + queue_id * sizeof(KvafBatchGetCoreStatus);
  queue_args->channel_entity =
      control.channel_entity + queue_id * sizeof(KvafBatchGetDirectUrmaEntity);
  queue_args->queue_state =
      control.queue_state + queue_id * sizeof(KvafBatchGetQueueState);
  queue_args->table_capacity = descriptor.table_capacity;
  queue_args->request_capacity = descriptor.request_capacity;
  queue_args->output_rows = static_cast<uint32_t>(control.output_rows);
  queue_args->block_bytes = static_cast<uint32_t>(control.block_bytes);
  queue_args->block_count = 1U;
  queue_args->jettys_per_core = 1U;
  *active_request_count = descriptor.active_request_count;
  return true;
}

// A P8 caller is never allowed to invent `private_ready_published`.  This
// bounded check ties the exact descriptor rows that Direct-HCCP received to
// the durable private records that protect their payloads.  It is deliberately
// outside the score loop: a wave has at most kDenseRequestCapacity rows and
// the check happens only once after CQ terminal.
__aicore__ inline bool ValidateKvafP8PrivateTerminalPublication(
    const KvafStreamingBatchGetControl& control,
    const KvafP8ForwardControl& forward_control,
    __gm__ const KvafP8PrivateRecord* private_records,
    uint64_t private_record_count, const KvafP8WaveRecord& record) {
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      private_records == nullptr || private_record_count != control.output_rows ||
      !IsKvafP8WaveRecordValid(record) ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, record.identity, kKvafP8ForwardPhaseDrain) ||
      record.state != kKvafP8WaveFinalized ||
      record.private_ready_published == 0U ||
      record.request_count == 0U ||
      record.request_count > kDenseRequestCapacity ||
      record.identity.queue_id >= control.queue_count) {
    return false;
  }
  __gm__ const KvafStreamingBatchGetDescriptorScratch* const scratch =
      GetKvafStreamingBatchGetDescriptorScratch(control,
                                                 record.identity.queue_id);
  KvafStreamingBatchGetDescriptorScratch descriptor{};
  if (!LoadKvafStreamingBatchGetDescriptorScratchFromGm(scratch, &descriptor) ||
      descriptor.output_indices == 0U ||
      descriptor.active_request_count != record.request_count ||
      descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity ||
      descriptor.request_capacity > kDenseRequestCapacity) {
    return false;
  }
  __gm__ const int32_t* const output_indices =
      reinterpret_cast<__gm__ const int32_t*>(descriptor.output_indices);
  for (uint32_t request = 0U; request < descriptor.active_request_count;
       ++request) {
    const int32_t output_index = output_indices[request];
    if (output_index < 0 ||
        static_cast<uint64_t>(output_index) >= private_record_count) {
      return false;
    }
    for (uint32_t prior = 0U; prior < request; ++prior) {
      if (output_indices[prior] == output_index) {
        return false;
      }
    }
    KvafP8PrivateRecord private_record{};
    if (!LoadKvafP8PrivateRecordFromGm(
            private_records + static_cast<uint64_t>(output_index),
            &private_record) ||
        (private_record.state != kKvafP8PrivateReady &&
         private_record.state != kKvafP8PrivateFailed) ||
        !KvafP8PrivateRecordMatchesWave(private_record, record)) {
      return false;
    }
  }
  return true;
}

// The descriptor alone names an HBM output row and a table/offset.  P8 also
// retains the source/object ledger used to construct that descriptor so a
// self-consistent but wrong mapping cannot be posted to Direct-HCCP.  This is
// bounded by one dense wave and is checked only at POST_INTENT.
__aicore__ inline bool ValidateKvafP8DescriptorPrivateLedger(
    const KvafStreamingBatchGetControl& control,
    const KvafP8WaveIdentity& identity,
    __gm__ const KvafP8PrivateRecord* private_records,
    uint64_t private_record_count,
    __gm__ const int64_t* request_source_offsets,
    __gm__ const int64_t* request_object_offsets,
    __gm__ const int32_t* request_output_indices,
    uint32_t request_ledger_capacity, uint32_t request_count) {
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      private_records == nullptr ||
      private_record_count != control.output_rows ||
      request_source_offsets == nullptr || request_object_offsets == nullptr ||
      request_output_indices == nullptr || request_ledger_capacity == 0U ||
      request_count == 0U || request_count > request_ledger_capacity ||
      request_count > kDenseRequestCapacity || identity.queue_id >= control.queue_count) {
    return false;
  }
  __gm__ const KvafStreamingBatchGetDescriptorScratch* const scratch =
      GetKvafStreamingBatchGetDescriptorScratch(control, identity.queue_id);
  KvafStreamingBatchGetDescriptorScratch descriptor{};
  if (!LoadKvafStreamingBatchGetDescriptorScratchFromGm(scratch, &descriptor) ||
      descriptor.output_indices == 0U ||
      descriptor.active_request_count != request_count ||
      descriptor.request_capacity < request_count) {
    return false;
  }
  __gm__ const int32_t* const descriptor_output_indices =
      reinterpret_cast<__gm__ const int32_t*>(descriptor.output_indices);
  const uint64_t base =
      static_cast<uint64_t>(identity.queue_id) * request_ledger_capacity;
  for (uint32_t request = 0U; request < request_count; ++request) {
    const int32_t output_index = descriptor_output_indices[request];
    const int32_t ledger_output_index = request_output_indices[base + request];
    const int64_t ledger_source_offset = request_source_offsets[base + request];
    const int64_t ledger_object_offset = request_object_offsets[base + request];
    if (output_index < 0 || output_index != ledger_output_index ||
        ledger_source_offset < 0 || ledger_object_offset < 0 ||
        static_cast<uint64_t>(output_index) >= private_record_count) {
      return false;
    }
    KvafP8PrivateRecord private_record{};
    if (!LoadKvafP8PrivateRecordFromGm(
        private_records + static_cast<uint64_t>(output_index),
            &private_record) ||
        private_record.source_offset != static_cast<uint64_t>(ledger_source_offset) ||
        private_record.object_offset != static_cast<uint64_t>(ledger_object_offset) ||
        !KvafP8PrivateRecordMatchesIdentity(private_record, identity)) {
      return false;
    }
  }
  return true;
}

// The only P8 private-terminal publisher.  It finalizes every descriptor row
// from the terminal CQ result, publishes those records to GM, rechecks the
// bounded set, and only then sets `private_ready_published`.  A malformed
// row does not become a recoverable "best effort" result: its wave is
// quarantined by the caller and the forward disables the private P8 view.
__aicore__ inline bool PublishKvafP8PrivateTerminalRecords(
    const KvafStreamingBatchGetControl& control,
    const KvafP8ForwardControl& forward_control,
    __gm__ KvafP8PrivateRecord* private_records,
    uint64_t private_record_count, KvafP8WaveRecord* record) {
  if (private_records == nullptr || record == nullptr ||
      !IsKvafP8WaveRecordValid(*record) ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, record->identity, kKvafP8ForwardPhaseDrain) ||
      record->state != kKvafP8WaveFinalized ||
      record->request_count == 0U ||
      record->request_count > kDenseRequestCapacity ||
      record->identity.queue_id >= control.queue_count) {
    return false;
  }
  __gm__ const KvafStreamingBatchGetDescriptorScratch* const scratch =
      GetKvafStreamingBatchGetDescriptorScratch(control,
                                                 record->identity.queue_id);
  KvafStreamingBatchGetDescriptorScratch descriptor{};
  if (!LoadKvafStreamingBatchGetDescriptorScratchFromGm(scratch, &descriptor) ||
      descriptor.output_indices == 0U ||
      descriptor.active_request_count != record->request_count ||
      descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity ||
      descriptor.request_capacity > kDenseRequestCapacity ||
      private_record_count != control.output_rows) {
    return false;
  }
  const bool payload_ready =
      record->outcome == kKvafP8WaveOutcomeSuccess;
  __gm__ const int32_t* const output_indices =
      reinterpret_cast<__gm__ const int32_t*>(descriptor.output_indices);
  for (uint32_t request = 0U; request < descriptor.active_request_count;
       ++request) {
    const int32_t output_index = output_indices[request];
    if (output_index < 0 ||
        static_cast<uint64_t>(output_index) >= private_record_count) {
      return false;
    }
    for (uint32_t prior = 0U; prior < request; ++prior) {
      if (output_indices[prior] == output_index) {
        return false;
      }
    }
    KvafP8PrivateRecord private_record{};
    if (!LoadKvafP8PrivateRecordFromGm(
            private_records + static_cast<uint64_t>(output_index),
            &private_record) ||
        !FinalizeKvafP8PrivateRecord(&private_record, *record,
                                      payload_ready) ||
        !StoreKvafP8PrivateRecordToGm(
            private_records + static_cast<uint64_t>(output_index),
            private_record)) {
      return false;
    }
  }
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  record->private_ready_published = 1U;
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return ValidateKvafP8PrivateTerminalPublication(
      control, forward_control, private_records, private_record_count,
      *record);
}

// Begin a previously acquired lease. On failure it leaves the queue lease
// held but marks the wave terminal, so the caller can release it immediately
// through ReleaseKvafStreamingBatchGetQueue without risking an in-flight CQ.
__aicore__ inline bool BeginKvafStreamingBatchGet(
    const KvafStreamingBatchGetControl& control,
    const KvafStreamingBatchGetLease& lease,
    KvafStreamingBatchGetWave* wave) {
  if (wave == nullptr) {
    return false;
  }
  wave->lease = lease;
  ResetDenseBatchGetTransaction(&wave->transaction);
  if (lease.state != kStreamingBatchGetLeaseHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count) {
    wave->transaction.result = kInvalidArgs;
    wave->transaction.state = kDenseBatchGetFailed;
    wave->transaction.finalized = 1U;
    wave->lease.state = kStreamingBatchGetLeaseTerminal;
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  uint32_t active_request_count = 0U;
  if (owners[lease.queue_id] != lease.owner_tag ||
      !BuildKvafStreamingBatchGetQueueArgs(control, lease.queue_id,
                                            &wave->queue_args,
                                            &active_request_count)) {
    wave->transaction.result = kInvalidArgs;
    wave->transaction.state = kDenseBatchGetFailed;
    wave->transaction.finalized = 1U;
    wave->lease.state = kStreamingBatchGetLeaseTerminal;
    return false;
  }
  BeginKvafDenseBatchGetPostOnly(wave->queue_args, active_request_count, 0U,
                                 &wave->transaction);
  wave->lease.state =
      wave->transaction.state == kDenseBatchGetInFlight
          ? kStreamingBatchGetLeasePosted
          : kStreamingBatchGetLeaseTerminal;
  return wave->transaction.state == kDenseBatchGetInFlight;
}

// Non-blocking completion probe for a posted streaming wave. It preserves the
// normal Direct-HCCP CQ semantics and gives the LI kernel a tile boundary at
// which to publish a successful speculative payload as ready.
__aicore__ inline void TryRetireKvafStreamingBatchGet(
    KvafStreamingBatchGetWave* wave) {
  if (wave == nullptr || wave->lease.state != kStreamingBatchGetLeasePosted) {
    return;
  }
  TryRetireKvafDenseBatchGet(wave->queue_args, &wave->transaction);
  if (wave->transaction.state == kDenseBatchGetRetired ||
      wave->transaction.state == kDenseBatchGetFailed) {
    wave->lease.state = kStreamingBatchGetLeaseTerminal;
  }
}

// Finish is the terminal drain. Call this once no further LI tile work can
// hide the CQ wait, before releasing the lease. It is mathematically the same
// post/CQ path as BatchGet because it delegates to the existing transaction.
__aicore__ inline void FinishKvafStreamingBatchGet(
    KvafStreamingBatchGetWave* wave) {
  if (wave == nullptr ||
      (wave->lease.state != kStreamingBatchGetLeasePosted &&
       wave->lease.state != kStreamingBatchGetLeaseTerminal)) {
    return;
  }
  FinishKvafDenseBatchGet(wave->queue_args, &wave->transaction);
  wave->lease.state = kStreamingBatchGetLeaseTerminal;
}

// Return a ready record only after Finish has observed a terminal CQ and
// published the normal KVAF status. `ready=1` means the private output arena
// contains the dense speculative payload; a failed/timeout CQ returns false
// so the caller can leave this candidate for ExactTopK's guaranteed fallback.
__aicore__ inline bool GetKvafStreamingBatchGetReady(
    const KvafStreamingBatchGetWave& wave,
    KvafStreamingBatchGetReady* ready) {
  if (ready == nullptr || wave.lease.state != kStreamingBatchGetLeaseTerminal ||
      wave.transaction.finalized == 0U) {
    return false;
  }
  ready->queue_id = wave.lease.queue_id;
  ready->owner_tag = wave.lease.owner_tag;
  ready->active_request_count =
      static_cast<uint32_t>(wave.transaction.active_count);
  ready->result = wave.transaction.result;
  ready->ready = wave.transaction.state == kDenseBatchGetRetired &&
                 wave.transaction.result == kSuccess
                     ? 1U
                     : 0U;
  return ready->ready != 0U;
}

// Release is SIMT-safe and deliberately refuses a non-terminal wave. This is
// the queue-pool ownership boundary: no subsequent LI producer can reuse the
// SQ/CQ pair until the former owner's CQ has been retired or terminally
// failed and Finish has published the normal KVAF status.
__simt_callee__ inline bool ReleaseKvafStreamingBatchGetQueueFromSimt(
    const KvafStreamingBatchGetControl& control,
    KvafStreamingBatchGetLease* lease) {
  if (lease == nullptr || lease->state != kStreamingBatchGetLeaseTerminal ||
      lease->owner_tag <= 0 || lease->queue_id >= control.queue_count ||
      !IsKvafStreamingBatchGetControlValidForLease(control)) {
    return false;
  }
  __gm__ int32_t* const owners =
      reinterpret_cast<__gm__ int32_t*>(control.queue_leases);
  // Ensure the preceding HCCP/CQ/status accesses complete before another LI
  // producer observes this queue as free.
  AscendC::Simt::ThreadFence();
  if (asc_atomic_cas(owners + lease->queue_id, lease->owner_tag, 0) !=
      lease->owner_tag) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  lease->owner_tag = 0;
  lease->state = kStreamingBatchGetLeaseIdle;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafStreamingBatchGetReleaseSimtService(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    uint32_t queue_id, int32_t owner_tag,
    __ubuf__ KvafStreamingBatchGetLeaseResult* result) {
  if (result == nullptr || control_gm == nullptr) {
    return;
  }
  KvafStreamingBatchGetControl control{};
  if (!LoadKvafStreamingBatchGetControlFromGmForSimt(control_gm, &control)) {
    return;
  }
  KvafStreamingBatchGetLease lease{queue_id, owner_tag,
                                    kStreamingBatchGetLeaseTerminal, 0U};
  if (ReleaseKvafStreamingBatchGetQueueFromSimt(control, &lease)) {
    result->queue_id = queue_id;
    result->owner_tag = 0;
    result->success = 1U;
  }
}

// Supported AIC-scalar release companion. The caller must call
// FinishKvafStreamingBatchGet first; this wrapper enforces terminal local
// state before it asks the SIMT service to make the Direct-HCCP queue idle.
__aicore__ inline bool ReleaseKvafStreamingBatchGetQueueFromAicore(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    __ubuf__ KvafStreamingBatchGetLeaseResult* lease_result,
    event_t simt_event, KvafStreamingBatchGetWave* wave) {
  if (control_gm == nullptr || lease_result == nullptr || wave == nullptr ||
      wave->lease.state != kStreamingBatchGetLeaseTerminal ||
      wave->lease.owner_tag <= 0 || wave->transaction.finalized == 0U) {
    return false;
  }
  lease_result->queue_id = wave->lease.queue_id;
  lease_result->owner_tag = wave->lease.owner_tag;
  lease_result->success = 0U;
  lease_result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafStreamingBatchGetReleaseSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, wave->lease.queue_id,
      wave->lease.owner_tag, lease_result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (lease_result->success == 0U) {
    return false;
  }
  wave->lease.owner_tag = 0;
  wave->lease.state = kStreamingBatchGetLeaseIdle;
  return true;
}

// P8 has two independent ownership domains.  The queue-lease owner protects
// descriptor/SQ/CQ memory from another producer; this CAS protects the
// *durable wave record* from two later graph drain tasks.  It is deliberately
// a non-blocking claim: losing workers continue score work or defer instead
// of spinning.  A disrupted claimant leaves DRAIN_CLAIMED for forward
// terminal recovery, which quarantines rather than guessing whether a CQ was
// consumed.
__simt_callee__ inline bool TryClaimKvafP8WaveDrainFromSimt(
    __gm__ KvafP8WaveRecord* record,
    __ubuf__ KvafP8WaveDrainClaimResult* result) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8WaveFree;
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(kKvafP8WavePosted),
      static_cast<int32_t>(kKvafP8WaveDrainClaimed));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(kKvafP8WavePosted)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8WaveDrainClaimSimtService(
    __gm__ KvafP8WaveRecord* record,
    __ubuf__ KvafP8WaveDrainClaimResult* result) {
  (void)TryClaimKvafP8WaveDrainFromSimt(record, result);
}

// The caller must reload and validate the claimed GM record before touching
// HCCP.  It must persist either POSTED (CQ still pending), FINALIZED/FREE, or
// QUARANTINED before returning; no AIC-local `hasWave` state is authoritative.
__aicore__ inline bool TryClaimKvafP8WaveDrainFromAicore(
    __gm__ KvafP8WaveRecord* record,
    __ubuf__ KvafP8WaveDrainClaimResult* result, event_t simt_event) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8WaveFree;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8WaveDrainClaimSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  return result->success != 0U &&
         result->observed_state == kKvafP8WavePosted;
}

// P8 phase helpers ---------------------------------------------------------
//
// A P8 executor invokes these helpers in short graph-visible phases:
//
//   descriptor/ledger publish -> PreparePostIntent -> STORE+DDR barrier
//   -> PostOnly -> STORE+DDR barrier -> TryRetire/Finish -> private terminal
//   publish -> ReleaseAfterPrivateTerminal.
//
// The helpers intentionally do not store a record to GM themselves.  The
// caller owns that explicit publish edge, which prevents an implementation
// from accidentally ringing a doorbell before the descriptor and P8 record
// become visible to the same forward's drain phase.

__aicore__ inline void QuarantineKvafP8WaveRecord(
    KvafP8WaveRecord* record, KvafP8WaveOutcome outcome) {
  if (record == nullptr) {
    return;
  }
  record->state = kKvafP8WaveQuarantined;
  record->outcome = static_cast<uint32_t>(outcome);
}

__aicore__ inline bool SnapshotKvafP8QueueIndices(
    const KvafExecuteBatchGetKernelArgs& args, uint64_t* sq_producer,
    uint64_t* cq_consumer) {
  if (args.queue_state == 0U || sq_producer == nullptr ||
      cq_consumer == nullptr) {
    return false;
  }
  __gm__ const KvafBatchGetQueueState* const queue =
      reinterpret_cast<__gm__ const KvafBatchGetQueueState*>(args.queue_state);
  *sq_producer = queue->sq_producer_index;
  *cq_consumer = queue->cq_consumer_index;
  return true;
}

// Construct the durable pre-doorbell intent.  The caller must first publish
// its descriptor scratch and candidate ledger, then store this record to GM
// and issue a DDR barrier before calling PostKvafP8Wave.  A record left in
// POST_INTENT after an interrupted launch is *ambiguous*: recovery must
// quarantine its queue rather than guessing whether HCCP saw a doorbell.
__aicore__ inline bool PrepareKvafP8WavePostIntent(
    const KvafStreamingBatchGetControl& control,
    const KvafP8ForwardControl& forward_control,
    const KvafStreamingBatchGetLease& lease,
    const KvafP8WaveIdentity& identity, uint32_t request_count,
    __gm__ const KvafP8PrivateRecord* private_records,
    uint64_t private_record_count,
    __gm__ const int64_t* request_source_offsets,
    __gm__ const int64_t* request_object_offsets,
    __gm__ const int32_t* request_output_indices,
    uint32_t request_ledger_capacity,
    KvafP8WaveRecord* record) {
  if (record == nullptr || request_count == 0U ||
      request_count > kDenseRequestCapacity ||
      lease.state != kStreamingBatchGetLeaseHeld || lease.owner_tag <= 0 ||
      identity.owner_tag != lease.owner_tag || identity.queue_id != lease.queue_id ||
      identity.forward_epoch == 0U || identity.queue_generation == 0U ||
      identity.post_sequence == 0U || identity.queue_id >= control.queue_count ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, identity, kKvafP8ForwardPhaseAdmitPost)) {
    if (record != nullptr) {
      ResetKvafP8WaveRecord(record, identity);
      QuarantineKvafP8WaveRecord(record, kKvafP8WaveOutcomeIdentityQuarantined);
    }
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      owners == nullptr || owners[lease.queue_id] != lease.owner_tag) {
    ResetKvafP8WaveRecord(record, identity);
    QuarantineKvafP8WaveRecord(record, kKvafP8WaveOutcomeIdentityQuarantined);
    return false;
  }

  KvafExecuteBatchGetKernelArgs args{};
  uint32_t active_request_count = 0U;
  if (!BuildKvafP8StreamingBatchGetQueueArgs(
          control, lease.queue_id, identity, private_records,
          private_record_count, &args, &active_request_count) ||
      active_request_count != request_count ||
      !ValidateKvafP8DescriptorPrivateLedger(
          control, identity, private_records, private_record_count,
          request_source_offsets, request_object_offsets,
          request_output_indices, request_ledger_capacity, request_count)) {
    ResetKvafP8WaveRecord(record, identity);
    QuarantineKvafP8WaveRecord(record, kKvafP8WaveOutcomeIdentityQuarantined);
    return false;
  }
  ResetKvafP8WaveRecord(record, identity);
  record->request_count = request_count;
  record->state = kKvafP8WavePostIntent;
  if (!SnapshotKvafP8QueueIndices(args, &record->sq_producer_before,
                                   &record->cq_consumer_before)) {
    QuarantineKvafP8WaveRecord(record, kKvafP8WaveOutcomeIdentityQuarantined);
    return false;
  }
  return true;
}

// Execute the one irreversible step.  `record` is local in this invocation;
// caller must persist it after the call.  When Begin rejects before an
// in-flight transaction exists, the result is an explicit POST_REJECTED and
// can release the lease only through the no-post recovery path.  Once it
// returns InFlight, post_committed is durable protocol state and any later
// failure must drain or quarantine -- never roll back as if it was unposted.
__aicore__ inline bool PostKvafP8Wave(
    const KvafStreamingBatchGetControl& control,
    const KvafP8ForwardControl& forward_control,
    __gm__ const KvafP8PrivateRecord* private_records,
    uint64_t private_record_count, KvafP8WaveRecord* record) {
  if (record == nullptr || !IsKvafP8WaveRecordValid(*record) ||
      record->state != kKvafP8WavePostIntent) {
    if (record != nullptr) {
      QuarantineKvafP8WaveRecord(record,
                                  kKvafP8WaveOutcomePostIntentQuarantined);
    }
    return false;
  }
  const KvafP8WaveIdentity identity = record->identity;
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      identity.queue_id >= control.queue_count ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, identity, kKvafP8ForwardPhaseAdmitPost)) {
    QuarantineKvafP8WaveRecord(record, kKvafP8WaveOutcomeIdentityQuarantined);
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners == nullptr || owners[identity.queue_id] != identity.owner_tag) {
    QuarantineKvafP8WaveRecord(record, kKvafP8WaveOutcomeIdentityQuarantined);
    return false;
  }
  KvafExecuteBatchGetKernelArgs args{};
  uint32_t active_request_count = 0U;
  if (!BuildKvafP8StreamingBatchGetQueueArgs(
          control, identity.queue_id, identity, private_records,
          private_record_count, &args, &active_request_count) ||
      active_request_count != record->request_count) {
    record->outcome = kKvafP8WaveOutcomePostRejected;
    record->state = kKvafP8WaveFinalized;
    record->transaction.result = kInvalidArgs;
    record->transaction.state = kDenseBatchGetFailed;
    record->transaction.finalized = 1U;
    return false;
  }
  BeginKvafDenseBatchGetPostOnly(args, active_request_count, 0U,
                                 &record->transaction);
  (void)SnapshotKvafP8QueueIndices(args, &record->sq_producer_after,
                                    &record->cq_consumer_after);
  if (record->transaction.state != kDenseBatchGetInFlight) {
    // Begin returned without a post. It is only releaseable through
    // ReleaseKvafP8UnpostedWave, which re-checks post_committed == 0.
    record->outcome = kKvafP8WaveOutcomePostRejected;
    record->state = kKvafP8WaveFinalized;
    record->post_committed = 0U;
    record->transaction.finalized = 1U;
    return false;
  }
  record->post_committed = 1U;
  record->state = kKvafP8WavePosted;
  return true;
}

// Probe without spinning.  A CQ timeout is *not* a terminal-safe error: the
// underlying CQ consumer index was not advanced, so a late completion may
// still arrive.  It is therefore quarantined, not released.  A CQ error with
// an observed/consumed CQE is terminal and can later be released after Finish
// has published status and the caller has published private FAILED.
__aicore__ inline bool TryRetireKvafP8Wave(
    const KvafStreamingBatchGetControl& control,
    const KvafP8ForwardControl& forward_control,
    __gm__ const KvafP8PrivateRecord* private_records,
    uint64_t private_record_count, KvafP8WaveRecord* record) {
  if (record == nullptr || !IsKvafP8WaveRecordValid(*record) ||
      record->state != kKvafP8WaveDrainClaimed ||
      record->post_committed == 0U) {
    return false;
  }
  const KvafP8WaveIdentity identity = record->identity;
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  KvafExecuteBatchGetKernelArgs args{};
  uint32_t active_request_count = 0U;
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      identity.queue_id >= control.queue_count ||
      owners == nullptr || owners[identity.queue_id] != identity.owner_tag ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, identity, kKvafP8ForwardPhaseDrain) ||
      !BuildKvafP8StreamingBatchGetQueueArgs(
          control, identity.queue_id, identity, private_records,
          private_record_count, &args, &active_request_count) ||
      active_request_count != record->request_count) {
    QuarantineKvafP8WaveRecord(record, kKvafP8WaveOutcomeIdentityQuarantined);
    return false;
  }
  TryRetireKvafDenseBatchGet(args, &record->transaction);
  (void)SnapshotKvafP8QueueIndices(args, &record->sq_producer_after,
                                    &record->cq_consumer_after);
  if (record->transaction.state == kDenseBatchGetInFlight) {
    // The caller persists POSTED before returning.  That is the only normal
    // hand-off back to another dynamic drain worker after this CAS claim.
    record->state = kKvafP8WavePosted;
    return false;
  }
  if (record->transaction.state == kDenseBatchGetFailed &&
      record->transaction.result == kCqTimeout &&
      record->cq_consumer_after == record->cq_consumer_before) {
    QuarantineKvafP8WaveRecord(record,
                                kKvafP8WaveOutcomeCqTimeoutQuarantined);
    return false;
  }
  if (record->transaction.state == kDenseBatchGetRetired ||
      (record->transaction.state == kDenseBatchGetFailed &&
       record->cq_consumer_after != record->cq_consumer_before)) {
    // Keep the durable drain claim through Finish/private publication.  The
    // terminal CQ fact is represented by the verified transaction and this
    // diagnostic marker; changing state to a freely callable CQ_OBSERVED
    // value would let a second worker finish/release a wave it never claimed.
    record->reserved[0] = kKvafP8WaveCqObserved;
    return true;
  }
  QuarantineKvafP8WaveRecord(record, kKvafP8WaveOutcomeIdentityQuarantined);
  return false;
}

// Complete a CQ-observed wave.  This routine intentionally does not release
// the lease: the P8 caller must first publish matching per-private-slot
// READY/FAILED records.  After that the HCCP queue may be reused; QSFA's
// later final read gates only private-slot reuse, not queue reuse.
__aicore__ inline bool FinishKvafP8Wave(
    const KvafStreamingBatchGetControl& control,
    const KvafP8ForwardControl& forward_control,
    __gm__ const KvafP8PrivateRecord* private_records,
    uint64_t private_record_count, KvafP8WaveRecord* record) {
  if (record == nullptr || !IsKvafP8WaveRecordValid(*record) ||
      record->state != kKvafP8WaveDrainClaimed ||
      record->post_committed == 0U) {
    return false;
  }
  const KvafP8WaveIdentity identity = record->identity;
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  KvafExecuteBatchGetKernelArgs args{};
  uint32_t active_request_count = 0U;
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      identity.queue_id >= control.queue_count ||
      owners == nullptr || owners[identity.queue_id] != identity.owner_tag ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, identity, kKvafP8ForwardPhaseDrain) ||
      !BuildKvafP8StreamingBatchGetQueueArgs(
          control, identity.queue_id, identity, private_records,
          private_record_count, &args, &active_request_count) ||
      active_request_count != record->request_count) {
    QuarantineKvafP8WaveRecord(record, kKvafP8WaveOutcomeIdentityQuarantined);
    return false;
  }
  if (record->transaction.state == kDenseBatchGetInFlight) {
    (void)TryRetireKvafP8Wave(control, forward_control, private_records,
                               private_record_count, record);
  }
  if (record->state == kKvafP8WaveQuarantined) {
    // Publish the normal KVAF status if the transaction already became
    // locally terminal, but preserve QUARANTINED because a late CQ can still
    // be possible after timeout.
    if (record->transaction.state != kDenseBatchGetInFlight) {
      FinishKvafDenseBatchGet(args, &record->transaction);
      (void)SnapshotKvafP8QueueIndices(args, &record->sq_producer_after,
                                        &record->cq_consumer_after);
    }
    return false;
  }
  if (record->state != kKvafP8WaveDrainClaimed ||
      (record->transaction.state != kDenseBatchGetRetired &&
       record->transaction.state != kDenseBatchGetFailed)) {
    return false;
  }
  FinishKvafDenseBatchGet(args, &record->transaction);
  (void)SnapshotKvafP8QueueIndices(args, &record->sq_producer_after,
                                    &record->cq_consumer_after);
  if (record->transaction.finalized == 0U) {
    QuarantineKvafP8WaveRecord(record, kKvafP8WaveOutcomeIdentityQuarantined);
    return false;
  }
  record->state = kKvafP8WaveFinalized;
  record->outcome = record->transaction.result == kSuccess
                        ? kKvafP8WaveOutcomeSuccess
                        : kKvafP8WaveOutcomeCqErrorObserved;
  return record->outcome == kKvafP8WaveOutcomeSuccess;
}

// This is the only normal release path for a posted P8 wave.  It rejects a
// release before all rows in the wave have reached private READY/FAILED and
// quarantines on a failed owner CAS.  This releases *only* the CQ/descriptor
// queue.  The bounded KvafP8PrivateRecord array retains private HBM payloads
// until the QSFA terminal phase proves their last read is complete.
__aicore__ inline bool ReleaseKvafP8WaveAfterPrivateReady(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    const KvafP8ForwardControl& forward_control,
    __gm__ const KvafP8PrivateRecord* private_records,
    uint64_t private_record_count,
    __ubuf__ KvafStreamingBatchGetLeaseResult* lease_result,
    event_t simt_event, KvafP8WaveRecord* record) {
  if (control_gm == nullptr || lease_result == nullptr || record == nullptr ||
      !IsKvafP8WaveRecordValid(*record) ||
      record->state != kKvafP8WaveFinalized ||
      record->post_committed == 0U ||
      record->private_ready_published == 0U ||
      record->transaction.finalized == 0U ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, record->identity, kKvafP8ForwardPhaseDrain)) {
    return false;
  }
  KvafStreamingBatchGetControl control{};
  if (!LoadKvafStreamingBatchGetControlFromGmForAicore(control_gm, &control) ||
      !ValidateKvafP8PrivateTerminalPublication(
          control, forward_control, private_records, private_record_count,
          *record)) {
    QuarantineKvafP8WaveRecord(record,
                                kKvafP8WaveOutcomeIdentityQuarantined);
    return false;
  }
  KvafStreamingBatchGetWave wave{};
  wave.lease.queue_id = record->identity.queue_id;
  wave.lease.owner_tag = record->identity.owner_tag;
  wave.lease.state = kStreamingBatchGetLeaseTerminal;
  wave.lease.reserved0 = 0U;
  wave.transaction = record->transaction;
  if (!ReleaseKvafStreamingBatchGetQueueFromAicore(control_gm, lease_result,
                                                     simt_event, &wave)) {
    QuarantineKvafP8WaveRecord(record,
                                kKvafP8WaveOutcomeReleaseCasQuarantined);
    return false;
  }
  record->state = kKvafP8WaveFree;
  record->post_committed = 0U;
  return true;
}

// Only a *proved unposted* wave may use this shorter release path.  It
// exists for descriptor validation/PostOnly rejection before the SQ doorbell;
// it explicitly refuses POST_INTENT because an interrupted producer may have
// crossed the hardware visibility boundary without updating post_committed.
__aicore__ inline bool ReleaseKvafP8UnpostedWave(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    const KvafP8ForwardControl& forward_control,
    __ubuf__ KvafStreamingBatchGetLeaseResult* lease_result,
    event_t simt_event, KvafP8WaveRecord* record) {
  if (control_gm == nullptr || lease_result == nullptr || record == nullptr ||
      !IsKvafP8WaveRecordValid(*record) || record->post_committed != 0U ||
      record->state != kKvafP8WaveFinalized ||
      record->outcome != kKvafP8WaveOutcomePostRejected ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, record->identity, kKvafP8ForwardPhaseDrain)) {
    if (record != nullptr && IsKvafP8WaveRecordValid(*record) &&
        record->state == kKvafP8WavePostIntent) {
      QuarantineKvafP8WaveRecord(record,
                                  kKvafP8WaveOutcomePostIntentQuarantined);
    }
    return false;
  }
  KvafStreamingBatchGetWave wave{};
  wave.lease.queue_id = record->identity.queue_id;
  wave.lease.owner_tag = record->identity.owner_tag;
  wave.lease.state = kStreamingBatchGetLeaseTerminal;
  wave.lease.reserved0 = 0U;
  wave.transaction = record->transaction;
  if (!ReleaseKvafStreamingBatchGetQueueFromAicore(control_gm, lease_result,
                                                     simt_event, &wave)) {
    QuarantineKvafP8WaveRecord(record,
                                kKvafP8WaveOutcomeReleaseCasQuarantined);
    return false;
  }
  record->state = kKvafP8WaveFree;
  return true;
}

}  // namespace kvaf_fused_get

#endif  // KVAF_FUSED_FULL_GET_DEVICE_H_
