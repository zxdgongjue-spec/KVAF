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

// Error handling is concurrent too.  Abort is a monotonic, shared forward
// gate; it cannot be implemented by a normal GM read/modify/write because a
// second worker could otherwise erase a first worker's ABORT bit.  The raw
// CAS is intentionally kept in SIMT for the same CANN MIX restriction as the
// queue and drain claims above.
struct KvafP8ForwardAbortResult {
  uint32_t success;
  uint32_t observed_flags_low;
};
static_assert(sizeof(KvafP8ForwardAbortResult) == 8U,
              "P8 forward-abort UB hand-off ABI changed");

// A private payload survives the Direct-HCCP queue.  Its final reader must
// claim the row before it clears the two mapping directories; otherwise two
// graph/replay workers can both clear or recycle the same physical slot.
// `source_observed` and `object_observed` retain the CAS evidence for a
// failure diagnostic.  A partial map clear is never rolled back or freed: the
// row is quarantined and Exact/Guaranteed Recall remains authoritative.
struct KvafP8PrivateReleaseResult {
  uint32_t success;
  uint32_t observed_state;
  int32_t source_observed;
  int32_t object_observed;
};
static_assert(sizeof(KvafP8PrivateReleaseResult) == 16U,
              "P8 private-release UB hand-off ABI changed");

// Forward-terminal is a fail-closed sweep, not a second normal drain path.
// It only classifies an already durable record; it never returns a possibly
// in-flight Direct-HCCP queue or an unread private payload to a hot pool.
struct KvafP8ForwardTerminalResult {
  uint32_t success;
  uint32_t observed_state;
};
static_assert(sizeof(KvafP8ForwardTerminalResult) == 8U,
              "P8 forward-terminal UB hand-off ABI changed");

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
  // The lease has been acquired and its complete forward/replay/queue
  // identity has been durably published, but no descriptor, private record,
  // request ledger, or HCCP doorbell may have been exposed yet.  This value
  // is deliberately appended: the graph-stable record ABI and all legacy
  // state numbers above remain unchanged.
  kKvafP8WaveFilling = 7U,
  // A proven-empty FILLING record has one worker that owns the final
  // owner-table CAS-to-zero.  It exists solely to close the race with a
  // forward-terminal sweep: the worker must commit this state to FREE with a
  // second CAS, never overwrite a concurrent QUARANTINED terminal result.
  kKvafP8WaveFillingReleaseClaimed = 8U,
  // Admission ownership is claimed *before* a producer writes the new wave
  // identity or any mutable record payload.  The prior record can be either
  // a valid FREE entry or an all-zero first-use allocation, so a terminal
  // sweep must treat this state as quarantinable even when the identity has
  // not been initialized yet.  Once the payload is DDR-visible, the producer
  // commits CLAIMED -> FILLING with a second CAS.  Appending this value keeps
  // the graph-stable 256-byte record ABI and all existing state numbers.
  kKvafP8WaveFillingClaimed = 9U,
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
  kKvafP8WaveOutcomeForwardTerminalQuarantined = 8U,
  // A FILLING record was not proved empty, could not be matched to its
  // lease/identity, or could not prove its owner-CAS release.  It is sticky
  // until the cold recovery path resets the graph-stable arena.
  kKvafP8WaveOutcomeFillingQuarantined = 9U,
};

// `KvafP8WaveRecord::reserved[1]` is a two-value durable proof while the
// record is in FILLING.  A worker obtains `NoWork` immediately after its
// queue-lease record is published.  It must atomically publish `Exposure`
// before it writes *any* descriptor, private-slot record, or request ledger.
// Therefore an interrupted FILLING record can return a queue only with the
// NoWork proof; every other pre-doorbell exit is quarantined.
constexpr uint64_t kKvafP8FillingNoWorkProof = 0x5038464E4F574F52ULL;
constexpr uint64_t kKvafP8FillingExposureProof = 0x5038464558504F53ULL;

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

// `owner_tag` proves a queue-lease CAS, but it is not a complete transaction
// identity.  Every P8 transition that crosses a graph-visible boundary must
// compare all identity fields, including the reserved word (which is required
// to be deterministic for replay).  Do not weaken this to owner_tag/queue_id:
// that would allow an old forward or graph replay to adopt a later wave.
__aicore__ inline bool KvafP8WaveIdentityEquals(
    const KvafP8WaveIdentity& left, const KvafP8WaveIdentity& right) {
  return left.forward_epoch == right.forward_epoch &&
         left.graph_replay_generation == right.graph_replay_generation &&
         left.layer_control_index == right.layer_control_index &&
         left.queue_id == right.queue_id &&
         left.queue_generation == right.queue_generation &&
         left.post_sequence == right.post_sequence &&
         left.owner_tag == right.owner_tag && left.reserved0 == right.reserved0;
}

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

// P8 keeps the graph-wide arena lease token in the otherwise unused high
// 32 bits of `flags`.  The low 32 bits remain the graph-visible flag ABI
// above and are still the only bits touched by the concurrent abort CAS.
// This is deliberately a generation token, not another forward identity.
// Terminal-before-reuse is the primary correctness rule; the token is a
// no-extra-tensor defensive CAS check if a graph-ordering violation would
// otherwise let a delayed terminal observe a later lease.
constexpr uint32_t kKvafP8ForwardLeaseGenerationShift = 32U;
constexpr uint64_t kKvafP8ForwardRuntimeFlagsMask = 0xFFFFFFFFULL;

struct KvafP8ForwardControl {
  uint64_t forward_epoch;
  uint64_t graph_replay_generation;
  uint32_t layer_control_index;
  uint32_t phase;
  uint64_t flags;
};
static_assert(sizeof(KvafP8ForwardControl) == KVAF_P8_FORWARD_CONTROL_BYTES,
              "P8 forward control ABI changed");

__aicore__ inline uint32_t GetKvafP8ForwardLeaseGeneration(
    const KvafP8ForwardControl& control) {
  return static_cast<uint32_t>(
      control.flags >> kKvafP8ForwardLeaseGenerationShift);
}

// Admission is the only writer of the high word and runs before P8 workers
// are launched.  Write that word independently so a concurrent low-word
// ABORT CAS can never be accidentally erased by a stale 64-bit RMW.
__aicore__ inline bool StoreKvafP8ForwardLeaseGenerationToGm(
    __gm__ KvafP8ForwardControl* control, uint32_t generation) {
  if (control == nullptr || generation == 0U) {
    return false;
  }
  __gm__ uint32_t* const high_word =
      reinterpret_cast<__gm__ uint32_t*>(&(control->flags)) + 1U;
  *high_word = generation;
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return *high_word == generation;
}

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
// leave its prior READY data visible to QSFA.  Only the low 32 bits currently
// contain P8 flags, including the two sticky abort bits.  Preserve the high
// half so this remains ABI compatible if a future control-plane flag uses it.
constexpr uint32_t kKvafP8ForwardAbortLowMask =
    static_cast<uint32_t>(kKvafP8ForwardFlagAbort |
                          kKvafP8ForwardFlagPrivateViewDisabled);

__simt_callee__ inline bool MarkKvafP8ForwardAbortFromSimt(
    __gm__ KvafP8ForwardControl* control,
    __ubuf__ KvafP8ForwardAbortResult* result) {
  if (result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_flags_low = 0U;
  if (control == nullptr) {
    return false;
  }
  // flags is naturally 8-byte aligned; its low word is independently CASed
  // because all sticky P8 abort bits are deliberately assigned there.
  __gm__ int32_t* const flags_low =
      reinterpret_cast<__gm__ int32_t*>(&(control->flags));
  int32_t expected = *flags_low;
  constexpr uint32_t kAbortAttempts = 32U;
  for (uint32_t attempt = 0U; attempt < kAbortAttempts; ++attempt) {
    const int32_t desired = expected |
        static_cast<int32_t>(kKvafP8ForwardAbortLowMask);
    const int32_t observed = asc_atomic_cas(flags_low, expected, desired);
    result->observed_flags_low = static_cast<uint32_t>(observed);
    if (observed == expected) {
      // The release fence makes the fail-closed gate visible before any
      // subsequent quarantine record is published by this worker.
      AscendC::Simt::ThreadFence();
      result->success = 1U;
      result->observed_flags_low =
          static_cast<uint32_t>(desired);
      return true;
    }
    if ((static_cast<uint32_t>(observed) & kKvafP8ForwardAbortLowMask) ==
        kKvafP8ForwardAbortLowMask) {
      // Another worker already made this forward fail closed.  Do not write
      // a stale `desired` value back and accidentally erase its nonsticky
      // control flags.
      AscendC::Simt::ThreadFence();
      result->success = 1U;
      result->observed_flags_low = static_cast<uint32_t>(observed);
      return true;
    }
    expected = observed;
  }
  // Bounded failure is still fail-closed for the caller: it must retain or
  // quarantine its resource and never continue to PostOnly.  A later terminal
  // sweep owns cold recovery rather than spinning inside an AIV task.
  return false;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8ForwardAbortSimtService(
    __gm__ KvafP8ForwardControl* control,
    __ubuf__ KvafP8ForwardAbortResult* result) {
  (void)MarkKvafP8ForwardAbortFromSimt(control, result);
}

// Supported AIC entry point.  A P8 executor must use this form, then reload
// forward control before it performs another irreversible operation.  The
// previous one-argument helper is retained below only for externally
// serialized setup code; it is not safe for concurrent device workers.
__aicore__ inline bool MarkKvafP8ForwardAbortFromAicore(
    __gm__ KvafP8ForwardControl* control,
    __ubuf__ KvafP8ForwardAbortResult* result, event_t simt_event) {
  if (control == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_flags_low = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8ForwardAbortSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U ||
      (result->observed_flags_low & kKvafP8ForwardAbortLowMask) !=
          kKvafP8ForwardAbortLowMask) {
    return false;
  }
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

// Compatibility-only helper.  It is safe only when the caller owns the
// control word exclusively (for example, pre-launch initialization).  Device
// executor failure paths must use MarkKvafP8ForwardAbortFromAicore above.
__aicore__ inline void MarkKvafP8ForwardAbort(
    __gm__ KvafP8ForwardControl* control) {
  if (control == nullptr) {
    return;
  }
  control->flags |= kKvafP8ForwardFlagAbort |
                    kKvafP8ForwardFlagPrivateViewDisabled;
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

// Forward-terminal owns a record even when the forward already carries ABORT.
// Normal Post/Drain helpers intentionally reject such a control word, but the
// terminal sweep must be able to match and quarantine the durable identity.
__aicore__ inline bool KvafP8ForwardTerminalOwnsIdentity(
    const KvafP8ForwardControl& control,
    const KvafP8WaveIdentity& identity) {
  return IsKvafP8ForwardControlValid(control) &&
         control.phase == kKvafP8ForwardPhaseForwardTerminal &&
         identity.owner_tag > 0 && identity.queue_generation != 0U &&
         identity.post_sequence != 0U &&
         control.forward_epoch == identity.forward_epoch &&
         control.graph_replay_generation == identity.graph_replay_generation &&
         control.layer_control_index == identity.layer_control_index;
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
  // reserved[0] is the CQ-observed diagnostic marker.  reserved[1] is the
  // FILLING no-work/exposure proof below; it is intentionally inside the
  // existing fixed 256-byte ABI rather than a new side ledger.  The remaining
  // entries stay zero unless a future version documents a graph-visible use.
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
  // The only transition out of READY/FAILED on the normal path.  It gives
  // one QSFA-terminal worker ownership of map removal and prevents a second
  // replay worker from freeing/reusing the slot underneath that removal.
  // Keep QUARANTINED=4 unchanged: this extends semantics without changing the
  // graph-stable private-record ABI or interpreting old quarantine bytes as a
  // reusable state.
  kKvafP8PrivateReleaseClaimed = 5U,
  // A worker owns this row before it has made a fully identified PENDING
  // record visible.  It exists to close FREE -> PENDING publication: writing
  // identity while the row still says FREE lets another worker claim it,
  // while writing PENDING before identity lets ForwardTerminal quarantine an
  // unknown owner.  ALLOCATING is non-consumable; terminal treats it as an
  // unresolved resource and quarantines it rather than trying to reuse it.
  kKvafP8PrivateAllocating = 6U,
  // Both source/object directory entries are claimed (-2) and their producer
  // exclusively owns the subsequent -2 -> positive-map publication. A
  // terminal sweep must not mutate this state or either directory: doing so
  // between a producer check and its CAS would leave a stale positive map.
  // It latches the arena unhealthy instead, leaving the bounded row for cold
  // recovery. This state is never consumable by QSFA.
  kKvafP8PrivateMapPublishing = 7U,
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
         record.state <= kKvafP8WaveFillingClaimed;
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
  // `state` is the publication word.  Generic writes are retained for cold
  // setup/diagnostics only; hot P8 paths use the exact expected-state CAS
  // helpers below.  Even here, never make a partially written identity look
  // consumable by exposing its state first.
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
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  destination->state = source.state;
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
         record.state <= kKvafP8PrivateMapPublishing;
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
  // Payload/identity are always visible before a state consumer can observe
  // the terminal state.  Initial FREE->ALLOCATING/PENDING publication does
  // not use this generic store: it uses the explicit CAS protocol below.
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
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  destination->state = source.state;
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

// The private output row has one additional publication protocol beyond the
// normal READY/FAILED transitions.  A state CAS first reserves the physical
// row, payload/identity are written while it is ALLOCATING, and only after
// both source/object maps are positive does the producer CAS it to PENDING.
// This makes an interrupted allocation durable and terminal-visible without
// allowing a stale full-record Store to revive the slot.
__simt_callee__ inline bool TryClaimKvafP8PrivateAllocatingFromSimt(
    __gm__ KvafP8PrivateRecord* record) {
  if (record == nullptr) {
    return false;
  }
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(kKvafP8PrivateFree),
      static_cast<int32_t>(kKvafP8PrivateAllocating));
  if (observed != static_cast<int32_t>(kKvafP8PrivateFree)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  return true;
}

// SIMT owns the narrow reservation and immutable-payload write.  The P8
// executor performs its DDR publication barrier from its enclosing AICore
// phase before it lets a second SIMT service expose either directory entry.
// Keeping that barrier outside this SIMT callable is intentional: CANN does
// not permit a SIMT VF to call the AICore-only payload publisher.
__simt_callee__ inline bool BeginKvafP8PrivateAllocatingPayloadFromSimt(
    __gm__ KvafP8PrivateRecord* record,
    const KvafP8WaveIdentity& identity, uint64_t source_offset,
    uint64_t object_offset, int32_t output_index) {
  if (record == nullptr || identity.forward_epoch == 0U ||
      identity.owner_tag <= 0 || identity.queue_generation == 0U ||
      identity.post_sequence == 0U || output_index < 0) {
    return false;
  }
  if (!TryClaimKvafP8PrivateAllocatingFromSimt(record)) {
    return false;
  }

  // Do not write state here.  A racing ForwardTerminal may permanently win
  // ALLOCATING -> QUARANTINED; immutable field writes cannot revive it, and
  // the later map-commit service will refuse to publish positive directory
  // entries unless the state is still ALLOCATING.
  record->magic = kKvafP8WaveRecordMagic;
  record->version = kKvafP8WaveRecordVersion;
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
  AscendC::Simt::ThreadFence();
  return true;
}

__simt_callee__ inline bool TryPromoteKvafP8PrivateAllocatingToPendingFromSimt(
    __gm__ KvafP8PrivateRecord* record) {
  if (record == nullptr) {
    return false;
  }
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(kKvafP8PrivateAllocating),
      static_cast<int32_t>(kKvafP8PrivatePending));
  if (observed != static_cast<int32_t>(kKvafP8PrivateAllocating)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  return true;
}

// This producer-only transition closes the window between transient (-2)
// directory claims and their positive publication. ForwardTerminal may
// quarantine ALLOCATING before this CAS, but deliberately never changes
// MAP_PUBLISHING while the producer owns the two directory words.
__simt_callee__ inline bool TryClaimKvafP8PrivateMapPublishingFromSimt(
    __gm__ KvafP8PrivateRecord* record) {
  if (record == nullptr) {
    return false;
  }
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(kKvafP8PrivateAllocating),
      static_cast<int32_t>(kKvafP8PrivateMapPublishing));
  if (observed != static_cast<int32_t>(kKvafP8PrivateAllocating)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  return true;
}

__simt_callee__ inline bool
TryPromoteKvafP8PrivateMapPublishingToPendingFromSimt(
    __gm__ KvafP8PrivateRecord* record) {
  if (record == nullptr) {
    return false;
  }
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(kKvafP8PrivateMapPublishing),
      static_cast<int32_t>(kKvafP8PrivatePending));
  if (observed != static_cast<int32_t>(kKvafP8PrivateMapPublishing)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  return true;
}

// `ALLOCATING` deliberately has no consumer semantics.  A producer uses this
// narrow transition on every pre-PENDING failure instead of full-storing a
// local record: ForwardTerminal may have won ALLOCATING -> QUARANTINED while
// the producer was publishing identity or claiming a directory entry.
__simt_callee__ inline bool TryQuarantineKvafP8PrivateAllocatingFromSimt(
    __gm__ KvafP8PrivateRecord* record) {
  if (record == nullptr) {
    return false;
  }
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(kKvafP8PrivateAllocating),
      static_cast<int32_t>(kKvafP8PrivateQuarantined));
  if (observed != static_cast<int32_t>(kKvafP8PrivateAllocating)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  return true;
}

// Publish the complete immutable ownership payload without touching state.
// The caller must have already claimed FREE -> ALLOCATING and must still
// complete the final ALLOCATING -> PENDING CAS itself.  Keeping the state
// unchanged here is what makes a racing terminal quarantine win permanently.
__aicore__ inline bool PublishKvafP8PrivateAllocatingPayloadToGm(
    __gm__ KvafP8PrivateRecord* destination,
    const KvafP8PrivateRecord& pending_record) {
  if (destination == nullptr ||
      !IsKvafP8PrivateRecordValid(pending_record) ||
      pending_record.state != kKvafP8PrivatePending ||
      pending_record.forward_epoch == 0U ||
      pending_record.owner_tag <= 0 ||
      pending_record.queue_generation == 0U ||
      pending_record.post_sequence == 0U ||
      pending_record.output_index < 0) {
    return false;
  }
  destination->magic = pending_record.magic;
  destination->version = pending_record.version;
  destination->forward_epoch = pending_record.forward_epoch;
  destination->graph_replay_generation =
      pending_record.graph_replay_generation;
  destination->layer_control_index = pending_record.layer_control_index;
  destination->queue_id = pending_record.queue_id;
  destination->queue_generation = pending_record.queue_generation;
  destination->post_sequence = pending_record.post_sequence;
  destination->source_offset = pending_record.source_offset;
  destination->object_offset = pending_record.object_offset;
  destination->output_index = pending_record.output_index;
  destination->owner_tag = pending_record.owner_tag;
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
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

__aicore__ inline bool HasKvafP8PrivateRecordCompleteIdentity(
    const KvafP8PrivateRecord& private_record) {
  return IsKvafP8PrivateRecordValid(private_record) &&
         private_record.owner_tag > 0 &&
         private_record.queue_generation != 0U &&
         private_record.post_sequence != 0U;
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

// Publish a CQ-terminal private state without ever writing a full local
// record back to GM.  ForwardTerminal may concurrently seal PENDING as
// QUARANTINED; the state CAS is therefore the only legal publication edge.
// The immutable identity was already published by the ALLOCATING protocol and
// is rechecked both before and after the CAS.
__simt_callee__ inline bool TryCommitKvafP8PrivateTerminalFromSimt(
    __gm__ KvafP8PrivateRecord* record, uint32_t desired_state,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  if (record == nullptr || result == nullptr ||
      (desired_state != kKvafP8PrivateReady &&
       desired_state != kKvafP8PrivateFailed)) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  result->source_observed = -1;
  result->object_observed = -1;
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(kKvafP8PrivatePending),
      static_cast<int32_t>(desired_state));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(kKvafP8PrivatePending)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8PrivateTerminalCommitSimtService(
    __gm__ KvafP8PrivateRecord* record, uint32_t desired_state,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  (void)TryCommitKvafP8PrivateTerminalFromSimt(record, desired_state, result);
}

__aicore__ inline bool CommitKvafP8PrivateTerminalFromAicore(
    __gm__ KvafP8PrivateRecord* record_gm,
    const KvafP8WaveIdentity& expected_identity, bool payload_ready,
    __ubuf__ KvafP8PrivateReleaseResult* result, event_t simt_event,
    KvafP8PrivateRecord* committed_record) {
  if (record_gm == nullptr || result == nullptr) {
    return false;
  }
  KvafP8PrivateRecord observed{};
  if (!LoadKvafP8PrivateRecordFromGm(record_gm, &observed) ||
      !HasKvafP8PrivateRecordCompleteIdentity(observed) ||
      observed.state != kKvafP8PrivatePending ||
      !KvafP8PrivateRecordMatchesIdentity(observed, expected_identity)) {
    return false;
  }
  const uint32_t desired_state = payload_ready ? kKvafP8PrivateReady
                                                : kKvafP8PrivateFailed;
  // Identity/payload are immutable for this edge, but retain an explicit DDR
  // ordering boundary before the state becomes visible to QSFA/release.
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  result->source_observed = -1;
  result->object_observed = -1;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8PrivateTerminalCommitSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record_gm, desired_state, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U ||
      result->observed_state != kKvafP8PrivatePending) {
    return false;
  }
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  KvafP8PrivateRecord committed{};
  if (!LoadKvafP8PrivateRecordFromGm(record_gm, &committed) ||
      !HasKvafP8PrivateRecordCompleteIdentity(committed) ||
      committed.state != desired_state ||
      !KvafP8PrivateRecordMatchesIdentity(committed, expected_identity)) {
    return false;
  }
  if (committed_record != nullptr) {
    *committed_record = committed;
  }
  return true;
}

// Compatibility-local validation used by older fused callers.  It deliberately
// stops at RELEASE_CLAIMED rather than setting FREE: a local copy cannot prove
// that the two GM mapping directories were atomically cleared.  New P8 device
// executors must call the GM/AIC helper below, which performs
// READY/FAILED -> RELEASE_CLAIMED CAS, verifies both mapping clears, and only
// then commits RELEASE_CLAIMED -> FREE.
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
  private_record->state = kKvafP8PrivateReleaseClaimed;
  return true;
}

// SIMT-side ownership claim for the long-lived HBM payload.  This is separate
// from the P8 wave drain claim: the Direct-HCCP queue can be released at CQ
// terminal while this row remains readable by QSFA.
__simt_callee__ inline bool TryClaimKvafP8PrivateReleaseFromSimt(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  if (result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  result->source_observed = -1;
  result->object_observed = -1;
  if (record == nullptr) {
    return false;
  }
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(kKvafP8PrivateReady),
      static_cast<int32_t>(kKvafP8PrivateReleaseClaimed));
  if (observed != static_cast<int32_t>(kKvafP8PrivateReady)) {
    observed = asc_atomic_cas(
        state, static_cast<int32_t>(kKvafP8PrivateFailed),
        static_cast<int32_t>(kKvafP8PrivateReleaseClaimed));
  }
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(kKvafP8PrivateReady) &&
      observed != static_cast<int32_t>(kKvafP8PrivateFailed)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8PrivateReleaseClaimSimtService(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  (void)TryClaimKvafP8PrivateReleaseFromSimt(record, result);
}

__aicore__ inline bool TryClaimKvafP8PrivateReleaseFromAicore(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result, event_t simt_event) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  result->source_observed = -1;
  result->object_observed = -1;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8PrivateReleaseClaimSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  return result->success != 0U &&
         (result->observed_state == kKvafP8PrivateReady ||
          result->observed_state == kKvafP8PrivateFailed);
}

// Clear both directory entries only while a RELEASE_CLAIMED record holds the
// payload slot.  These are two different GM arrays, so no hardware operation
// can clear them as one word.  If either CAS is not exact, the caller must
// quarantine the record rather than resetting it or trying to compensate.
// In particular, a partial source clear is safer than an unsafe reuse.
__simt_callee__ inline bool ClearKvafP8PrivateMappingsFromSimt(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    uint64_t source_offset, uint64_t object_offset, int32_t output_index,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  if (result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->source_observed = -1;
  result->object_observed = -1;
  if (source_to_slot == nullptr || object_to_slot == nullptr ||
      output_index < 0) {
    return false;
  }
  const int32_t source_observed = asc_atomic_cas(
      source_to_slot + source_offset, output_index, -1);
  result->source_observed = source_observed;
  if (source_observed != output_index) {
    return false;
  }
  const int32_t object_observed = asc_atomic_cas(
      object_to_slot + object_offset, output_index, -1);
  result->object_observed = object_observed;
  if (object_observed != output_index) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8PrivateMappingClearSimtService(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    uint64_t source_offset, uint64_t object_offset, int32_t output_index,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  (void)ClearKvafP8PrivateMappingsFromSimt(
      source_to_slot, object_to_slot, source_offset, object_offset,
      output_index, result);
}

__aicore__ inline bool ClearKvafP8PrivateMappingsFromAicore(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    uint64_t source_offset, uint64_t object_offset, int32_t output_index,
    __ubuf__ KvafP8PrivateReleaseResult* result, event_t simt_event) {
  if (source_to_slot == nullptr || object_to_slot == nullptr ||
      output_index < 0 || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->source_observed = -1;
  result->object_observed = -1;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8PrivateMappingClearSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, source_to_slot, object_to_slot,
      source_offset, object_offset, output_index, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  return result->success != 0U &&
         result->source_observed == output_index &&
         result->object_observed == output_index;
}

__simt_callee__ inline bool CommitKvafP8PrivateReleaseFromSimt(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  if (result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  if (record == nullptr) {
    return false;
  }
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(kKvafP8PrivateReleaseClaimed),
      static_cast<int32_t>(kKvafP8PrivateFree));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(kKvafP8PrivateReleaseClaimed)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8PrivateReleaseCommitSimtService(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  (void)CommitKvafP8PrivateReleaseFromSimt(record, result);
}

__aicore__ inline bool CommitKvafP8PrivateReleaseFromAicore(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result, event_t simt_event) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8PrivateReleaseCommitSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  return result->success != 0U &&
         result->observed_state == kKvafP8PrivateReleaseClaimed;
}

// A failed RELEASE_CLAIMED -> FREE commit must never be silently treated as
// a reusable slot.  In the normal case the state is still RELEASE_CLAIMED, so
// a second CAS can seal it as QUARANTINED.  If that CAS sees any other state,
// the caller has lost ownership proof and must leave the row unavailable for
// forward-terminal recovery rather than overwrite a potentially newer owner.
__simt_callee__ inline bool TryQuarantineKvafP8PrivateReleaseClaimedFromSimt(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(kKvafP8PrivateReleaseClaimed),
      static_cast<int32_t>(kKvafP8PrivateQuarantined));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(kKvafP8PrivateReleaseClaimed)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8PrivateReleaseFailureQuarantineSimtService(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  (void)TryQuarantineKvafP8PrivateReleaseClaimedFromSimt(record, result);
}

__aicore__ inline bool QuarantineKvafP8PrivateReleaseClaimedFromAicore(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result, event_t simt_event) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<
      KvafP8PrivateReleaseFailureQuarantineSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  return result->success != 0U &&
         result->observed_state == kKvafP8PrivateReleaseClaimed;
}

// The complete normal release protocol.  `private_ready_index` is supplied by
// the executor's fixed row partition and must equal record.output_index.  The
// helper intentionally does not accept capacities: callers validate all
// offsets against graph-stable tensor shapes before they enter this endpoint.
__aicore__ inline bool ReleaseKvafP8PrivateRecordAfterQsfaTerminal(
    __gm__ KvafP8PrivateRecord* record_gm,
    const KvafP8ForwardControl& forward_control,
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    __gm__ int32_t* private_ready, uint64_t private_ready_index,
    __ubuf__ KvafP8PrivateReleaseResult* result, event_t simt_event) {
  if (record_gm == nullptr || source_to_slot == nullptr ||
      object_to_slot == nullptr || private_ready == nullptr ||
      result == nullptr) {
    return false;
  }
  if (!TryClaimKvafP8PrivateReleaseFromAicore(
          record_gm, result, simt_event)) {
    return false;
  }
  KvafP8PrivateRecord record{};
  if (!LoadKvafP8PrivateRecordFromGm(record_gm, &record) ||
      !HasKvafP8PrivateRecordCompleteIdentity(record) ||
      record.state != kKvafP8PrivateReleaseClaimed ||
      record.output_index < 0 ||
      static_cast<uint64_t>(record.output_index) != private_ready_index ||
      private_ready[private_ready_index] !=
          static_cast<int32_t>(result->observed_state) ||
      !IsKvafP8PrivateViewReadyForQsfa(forward_control) ||
      (forward_control.flags & kKvafP8ForwardFlagQsfaTerminalObserved) == 0U ||
      record.forward_epoch != forward_control.forward_epoch ||
      record.graph_replay_generation !=
          forward_control.graph_replay_generation ||
      record.layer_control_index != forward_control.layer_control_index) {
    // The claim already moved READY/FAILED -> RELEASE_CLAIMED.  Do not
    // resurrect a terminal transition with a full local Store; narrow-CAS
    // that exact state to QUARANTINED instead.
    (void)QuarantineKvafP8PrivateReleaseClaimedFromAicore(
        record_gm, result, simt_event);
    private_ready[private_ready_index] = static_cast<int32_t>(
        kKvafP8PrivateQuarantined);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    return false;
  }
  if (!ClearKvafP8PrivateMappingsFromAicore(
          source_to_slot, object_to_slot, record.source_offset,
          record.object_offset, record.output_index, result, simt_event)) {
    // A partial CAS is deliberately terminally non-reusable.  It may leave
    // one directory entry cleared, but no next claim can safely infer that the
    // other entry or payload is reusable.
    (void)QuarantineKvafP8PrivateReleaseClaimedFromAicore(
        record_gm, result, simt_event);
    private_ready[private_ready_index] = static_cast<int32_t>(
        kKvafP8PrivateQuarantined);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    return false;
  }
  // Map entries are now proven absent.  Make the ready word non-readable
  // before the final state CAS; a subsequent claimant cannot observe FREE
  // until this write and the preceding mapping clears have crossed the fence.
  private_ready[private_ready_index] =
      static_cast<int32_t>(kKvafP8PrivateFree);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  if (!CommitKvafP8PrivateReleaseFromAicore(record_gm, result, simt_event)) {
    // No map points to the row any longer, but its ownership transition was
    // not proved.  Seal RELEASE_CLAIMED as QUARANTINED when ownership is still
    // ours; otherwise retain the unknown record for forward-terminal recovery.
    (void)QuarantineKvafP8PrivateReleaseClaimedFromAicore(
        record_gm, result, simt_event);
    private_ready[private_ready_index] = static_cast<int32_t>(
        kKvafP8PrivateQuarantined);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    return false;
  }
  return true;
}

// A POST_REJECTED wave has no HCCP payload and therefore never reaches QSFA.
// Its private records are nevertheless already PENDING and visible through
// source/object mapping directories.  Releasing them with the normal QSFA
// endpoint would be incorrect: that endpoint requires READY/FAILED plus the
// QSFA terminal flag.  This is the narrowly scoped no-post counterpart.  It
// is legal only after a dynamic drain worker claimed the durable rejected wave
// and revalidated the same forward/replay/layer identity in DRAIN phase.
//
// The transition remains exactly as strict as the QSFA path:
// PENDING -> RELEASE_CLAIMED -> exact map CAS clear -> FREE.  A partial clear
// or a stale identity quarantines the row; it is never "repaired" locally.
__simt_callee__ inline bool TryClaimKvafP8PrivateNoPostReleaseFromSimt(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  result->source_observed = -1;
  result->object_observed = -1;
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(kKvafP8PrivatePending),
      static_cast<int32_t>(kKvafP8PrivateReleaseClaimed));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(kKvafP8PrivatePending)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8PrivateNoPostReleaseClaimSimtService(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result) {
  (void)TryClaimKvafP8PrivateNoPostReleaseFromSimt(record, result);
}

__aicore__ inline bool TryClaimKvafP8PrivateNoPostReleaseFromAicore(
    __gm__ KvafP8PrivateRecord* record,
    __ubuf__ KvafP8PrivateReleaseResult* result, event_t simt_event) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  result->source_observed = -1;
  result->object_observed = -1;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8PrivateNoPostReleaseClaimSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  return result->success != 0U &&
         result->observed_state == kKvafP8PrivatePending;
}

__aicore__ inline bool ReleaseKvafP8PendingPrivateRecordAfterNoPost(
    __gm__ KvafP8PrivateRecord* record_gm,
    const KvafP8ForwardControl& forward_control,
    const KvafP8WaveIdentity& identity,
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    __gm__ int32_t* private_ready, uint64_t private_ready_index,
    __ubuf__ KvafP8PrivateReleaseResult* result, event_t simt_event) {
  if (record_gm == nullptr || source_to_slot == nullptr ||
      object_to_slot == nullptr || private_ready == nullptr ||
      result == nullptr ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, identity, kKvafP8ForwardPhaseDrain)) {
    return false;
  }
  if (!TryClaimKvafP8PrivateNoPostReleaseFromAicore(
          record_gm, result, simt_event)) {
    return false;
  }
  KvafP8PrivateRecord record{};
  if (!LoadKvafP8PrivateRecordFromGm(record_gm, &record) ||
      !HasKvafP8PrivateRecordCompleteIdentity(record) ||
      record.state != kKvafP8PrivateReleaseClaimed ||
      !KvafP8PrivateRecordMatchesIdentity(record, identity) ||
      record.output_index < 0 ||
      static_cast<uint64_t>(record.output_index) != private_ready_index ||
      private_ready[private_ready_index] !=
          static_cast<int32_t>(kKvafP8PrivatePending)) {
    (void)QuarantineKvafP8PrivateReleaseClaimedFromAicore(
        record_gm, result, simt_event);
    private_ready[private_ready_index] = static_cast<int32_t>(
        kKvafP8PrivateQuarantined);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    return false;
  }
  if (!ClearKvafP8PrivateMappingsFromAicore(
          source_to_slot, object_to_slot, record.source_offset,
          record.object_offset, record.output_index, result, simt_event)) {
    (void)QuarantineKvafP8PrivateReleaseClaimedFromAicore(
        record_gm, result, simt_event);
    private_ready[private_ready_index] = static_cast<int32_t>(
        kKvafP8PrivateQuarantined);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    return false;
  }
  private_ready[private_ready_index] = static_cast<int32_t>(
      kKvafP8PrivateFree);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  if (!CommitKvafP8PrivateReleaseFromAicore(record_gm, result, simt_event)) {
    (void)QuarantineKvafP8PrivateReleaseClaimedFromAicore(
        record_gm, result, simt_event);
    private_ready[private_ready_index] = static_cast<int32_t>(
        kKvafP8PrivateQuarantined);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    return false;
  }
  return true;
}

// -------------------------------------------------------------------------
// Forward-terminal fail-closed sweep
// -------------------------------------------------------------------------
// A forward terminal is entered only after RuntimeResources has stopped new
// P8 admission for this graph-stable arena.  It is deliberately not a second
// CQ-drain algorithm: if a PostOnly/CQ ownership transition was interrupted,
// neither a device worker nor runtime teardown may guess that the queue or
// output is reusable.  Exact/Guaranteed Recall preserves math while this
// sweep quarantines the bounded resources for cold recovery.

__simt_callee__ inline bool TryQuarantineKvafP8WaveFromSimt(
    __gm__ KvafP8WaveRecord* record, uint32_t expected_state,
    __ubuf__ KvafP8ForwardTerminalResult* result) {
  if (result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8WaveFree;
  if (record == nullptr || expected_state == kKvafP8WaveFree ||
      expected_state == kKvafP8WaveQuarantined) {
    return false;
  }
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(expected_state),
      static_cast<int32_t>(kKvafP8WaveQuarantined));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(expected_state)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8WaveForwardTerminalSimtService(
    __gm__ KvafP8WaveRecord* record, uint32_t expected_state,
    __ubuf__ KvafP8ForwardTerminalResult* result) {
  (void)TryQuarantineKvafP8WaveFromSimt(record, expected_state, result);
}

// Returns true only when this exact forward/replay/layer owns a record which
// is already FREE/QUARANTINED or has just been atomically quarantined.  This
// deliberately includes durable FILLING (`kKvafP8WaveFilling`) and
// FILLING_RELEASE_CLAIMED (`kKvafP8WaveFillingReleaseClaimed`) records:
// terminal recovery cannot infer whether an interrupted producer crossed the
// descriptor/private/ledger exposure edge, or whether a release claimant will
// reach its final FREE CAS, so it quarantines rather than guessing or
// releasing a lease.  The caller must treat false as a foreign/corrupt/racing
// record and must not free
// its queue lease.  The mandatory pre-terminal admission gate rules out a new
// forward's ABA reuse between the pre-CAS identity read and post-CAS reload.
__aicore__ inline bool QuarantineKvafP8WaveAtForwardTerminal(
    __gm__ KvafP8WaveRecord* record_gm,
    const KvafP8ForwardControl& forward_control,
    __ubuf__ KvafP8ForwardTerminalResult* result, event_t simt_event) {
  if (record_gm == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8WaveFree;
  KvafP8WaveRecord record{};
  if (!LoadKvafP8WaveRecordFromGm(record_gm, &record)) {
    return false;
  }
  // FILLING_CLAIMED is deliberately entered before the new identity is
  // field-wise visible.  It is the one terminal state whose ownership cannot
  // be checked against ForwardControl; terminal must seal it by state alone
  // rather than leave a partially initialized queue slot reusable.
  if (record.state == kKvafP8WaveFillingClaimed) {
    set_flag(PIPE_S, PIPE_V, simt_event);
    wait_flag(PIPE_S, PIPE_V, simt_event);
    AscendC::Simt::VF_CALL<KvafP8WaveForwardTerminalSimtService>(
        AscendC::Simt::Dim3{1U, 1U, 1U}, record_gm,
        kKvafP8WaveFillingClaimed, result);
    set_flag(PIPE_V, PIPE_S, simt_event);
    wait_flag(PIPE_V, PIPE_S, simt_event);
    KvafP8WaveRecord quarantined{};
    return result->success != 0U &&
           result->observed_state == kKvafP8WaveFillingClaimed &&
           LoadKvafP8WaveRecordFromGm(record_gm, &quarantined) &&
           quarantined.state == kKvafP8WaveQuarantined;
  }
  if (!IsKvafP8WaveRecordValid(record) ||
      !KvafP8ForwardTerminalOwnsIdentity(forward_control, record.identity)) {
    return false;
  }
  if (record.state == kKvafP8WaveFree ||
      record.state == kKvafP8WaveQuarantined) {
    result->success = 1U;
    result->observed_state = record.state;
    return true;
  }
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8WaveForwardTerminalSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record_gm, record.state, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U || result->observed_state != record.state ||
      !LoadKvafP8WaveRecordFromGm(record_gm, &record) ||
      !IsKvafP8WaveRecordValid(record) ||
      record.state != kKvafP8WaveQuarantined ||
      !KvafP8ForwardTerminalOwnsIdentity(forward_control, record.identity)) {
    return false;
  }
  // The CAS above owns only the state word.  Do not full-store `record` here:
  // a producer which was publishing mutable post/drain payload when terminal
  // won may still have a newer local snapshot.  State is already terminal;
  // publish just the diagnostic outcome.
  record_gm->outcome = kKvafP8WaveOutcomeForwardTerminalQuarantined;
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

__simt_callee__ inline bool TryQuarantineKvafP8PrivateFromSimt(
    __gm__ KvafP8PrivateRecord* record, uint32_t expected_state,
    __ubuf__ KvafP8ForwardTerminalResult* result) {
  if (result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  if (record == nullptr || expected_state == kKvafP8PrivateFree ||
      expected_state == kKvafP8PrivateQuarantined) {
    return false;
  }
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(expected_state),
      static_cast<int32_t>(kKvafP8PrivateQuarantined));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(expected_state)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8PrivateForwardTerminalSimtService(
    __gm__ KvafP8PrivateRecord* record, uint32_t expected_state,
    __ubuf__ KvafP8ForwardTerminalResult* result) {
  (void)TryQuarantineKvafP8PrivateFromSimt(record, expected_state, result);
}

// ALLOCATING is intentionally allowed to have an incomplete identity: it is
// the only state whose state-CAS happens before its ownership fields are
// published.  The P8 arena lease rules out a concurrent forward, so a forward
// terminal sweep may always seal that raw state.  If the producer had already
// reached the map-claim edge, its immutable offsets are now durable and the
// sweep rolls back only the transient -2 markers.  It never clears a
// non-negative mapping: that would make an uncertain payload reusable.
constexpr int32_t kKvafP8PrivateDirectoryUnmapped = -1;
constexpr int32_t kKvafP8PrivateDirectoryClaiming = -2;

// The ALLOCATING terminal path rolls back only transient -2 directory
// claims.  It runs after the private-record state has already transitioned to
// QUARANTINED, so this remains deliberately best-effort: a missing/changed
// entry is left unavailable rather than being repaired or reused.  Crucially,
// the two GM CAS operations belong to a SIMT callee.  CANN's MIX compiler
// exposes asc_atomic_cas only in that address-space/domain; calling it from
// the AIC wrapper below either selects the UB overload or fails name lookup.
__simt_callee__ inline void
RollbackKvafP8PrivateAllocatingDirectoryClaimsFromSimt(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    uint64_t source_offset, uint64_t object_offset) {
  if (source_to_slot != nullptr) {
    (void)asc_atomic_cas(source_to_slot + source_offset,
                         kKvafP8PrivateDirectoryClaiming,
                         kKvafP8PrivateDirectoryUnmapped);
  }
  if (object_to_slot != nullptr) {
    (void)asc_atomic_cas(object_to_slot + object_offset,
                         kKvafP8PrivateDirectoryClaiming,
                         kKvafP8PrivateDirectoryUnmapped);
  }
  AscendC::Simt::ThreadFence();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8PrivateAllocatingDirectoryRollbackSimtService(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    uint64_t source_offset, uint64_t object_offset) {
  RollbackKvafP8PrivateAllocatingDirectoryClaimsFromSimt(
      source_to_slot, object_to_slot, source_offset, object_offset);
}

__aicore__ inline bool QuarantineKvafP8PrivateAllocatingAtForwardTerminal(
    __gm__ KvafP8PrivateRecord* record_gm,
    const KvafP8ForwardControl& forward_control,
    __gm__ int32_t* source_to_slot, uint64_t source_to_slot_count,
    __gm__ int32_t* object_to_slot, uint64_t object_to_slot_count,
    __gm__ int32_t* private_ready, uint64_t private_ready_index,
    __ubuf__ KvafP8ForwardTerminalResult* result, event_t simt_event) {
  if (record_gm == nullptr || private_ready == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  KvafP8PrivateRecord before{};
  if (!LoadKvafP8PrivateRecordFromGm(record_gm, &before) ||
      before.state != kKvafP8PrivateAllocating) {
    return false;
  }
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8PrivateForwardTerminalSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record_gm,
      kKvafP8PrivateAllocating, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  KvafP8PrivateRecord after{};
  if (result->success == 0U ||
      result->observed_state != kKvafP8PrivateAllocating ||
      !LoadKvafP8PrivateRecordFromGm(record_gm, &after) ||
      after.state != kKvafP8PrivateQuarantined) {
    return false;
  }
  // A complete identity proves the producer passed its identity publish
  // barrier.  Only then may its bounded map offsets be dereferenced.
  if (HasKvafP8PrivateRecordCompleteIdentity(after) &&
      after.output_index >= 0 &&
      static_cast<uint64_t>(after.output_index) == private_ready_index &&
      after.source_offset < source_to_slot_count &&
      after.object_offset < object_to_slot_count &&
      IsKvafP8ForwardControlValid(forward_control) &&
      forward_control.phase == kKvafP8ForwardPhaseForwardTerminal &&
      after.forward_epoch == forward_control.forward_epoch &&
      after.graph_replay_generation ==
          forward_control.graph_replay_generation &&
      after.layer_control_index == forward_control.layer_control_index) {
    if (source_to_slot != nullptr || object_to_slot != nullptr) {
      set_flag(PIPE_S, PIPE_V, simt_event);
      wait_flag(PIPE_S, PIPE_V, simt_event);
      AscendC::Simt::VF_CALL<
          KvafP8PrivateAllocatingDirectoryRollbackSimtService>(
          AscendC::Simt::Dim3{1U, 1U, 1U}, source_to_slot, object_to_slot,
          after.source_offset, after.object_offset);
      set_flag(PIPE_V, PIPE_S, simt_event);
      wait_flag(PIPE_V, PIPE_S, simt_event);
    }
  }
  private_ready[private_ready_index] = static_cast<int32_t>(
      kKvafP8PrivateQuarantined);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

// A POST worker may have made a private PENDING record visible while its wave
// is still FILLING or POST_INTENT, then lose a control/identity proof before
// it can ring the doorbell.  That is not a ForwardTerminal operation, but it
// has the same ownership requirement: never full-store an AIC-local record
// over a concurrent terminal transition.  This narrow CAS can only seal the
// exact PENDING record owned by `identity`; anything else remains unavailable
// for the graph terminal sweep/cold recovery path.
__aicore__ inline bool QuarantineKvafP8PrivatePendingFromAicore(
    __gm__ KvafP8PrivateRecord* record_gm,
    const KvafP8WaveIdentity& identity,
    __gm__ int32_t* private_ready, uint64_t private_ready_index,
    __ubuf__ KvafP8ForwardTerminalResult* result, event_t simt_event) {
  if (record_gm == nullptr || private_ready == nullptr || result == nullptr) {
    return false;
  }
  KvafP8PrivateRecord before{};
  if (!LoadKvafP8PrivateRecordFromGm(record_gm, &before) ||
      !IsKvafP8PrivateRecordValid(before) ||
      before.state != kKvafP8PrivatePending ||
      !KvafP8PrivateRecordMatchesIdentity(before, identity) ||
      before.output_index < 0 ||
      static_cast<uint64_t>(before.output_index) != private_ready_index) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8PrivateForwardTerminalSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record_gm,
      kKvafP8PrivatePending, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  KvafP8PrivateRecord after{};
  if (result->success == 0U ||
      result->observed_state != kKvafP8PrivatePending ||
      !LoadKvafP8PrivateRecordFromGm(record_gm, &after) ||
      !IsKvafP8PrivateRecordValid(after) ||
      after.state != kKvafP8PrivateQuarantined ||
      !KvafP8PrivateRecordMatchesIdentity(after, identity) ||
      after.output_index < 0 ||
      static_cast<uint64_t>(after.output_index) != private_ready_index) {
    return false;
  }
  private_ready[private_ready_index] = static_cast<int32_t>(
      kKvafP8PrivateQuarantined);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

__aicore__ inline bool KvafP8PrivateRecordBelongsToForwardTerminal(
    const KvafP8PrivateRecord& record,
    const KvafP8ForwardControl& forward_control) {
  return IsKvafP8PrivateRecordValid(record) &&
         HasKvafP8PrivateRecordCompleteIdentity(record) &&
         IsKvafP8ForwardControlValid(forward_control) &&
         forward_control.phase == kKvafP8ForwardPhaseForwardTerminal &&
         record.forward_epoch == forward_control.forward_epoch &&
         record.graph_replay_generation ==
             forward_control.graph_replay_generation &&
         record.layer_control_index == forward_control.layer_control_index;
}

// Private terminal cleanup deliberately leaves source/object mappings intact.
// A stale map paired with QUARANTINED is a fail-closed diagnostic; clearing it
// after an interrupted Post/CQ could make a still-DMA-writing payload reusable.
__aicore__ inline bool QuarantineKvafP8PrivateAtForwardTerminal(
    __gm__ KvafP8PrivateRecord* record_gm,
    const KvafP8ForwardControl& forward_control,
    __gm__ int32_t* private_ready, uint64_t private_ready_index,
    __ubuf__ KvafP8ForwardTerminalResult* result, event_t simt_event) {
  if (record_gm == nullptr || private_ready == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8PrivateFree;
  KvafP8PrivateRecord record{};
  if (!LoadKvafP8PrivateRecordFromGm(record_gm, &record) ||
      !KvafP8PrivateRecordBelongsToForwardTerminal(record, forward_control) ||
      record.output_index < 0 ||
      static_cast<uint64_t>(record.output_index) != private_ready_index) {
    return false;
  }
  if (record.state == kKvafP8PrivateFree ||
      record.state == kKvafP8PrivateQuarantined) {
    if (record.state == kKvafP8PrivateQuarantined) {
      private_ready[private_ready_index] = static_cast<int32_t>(
          kKvafP8PrivateQuarantined);
      AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    result->success = 1U;
    result->observed_state = record.state;
    return true;
  }
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8PrivateForwardTerminalSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record_gm, record.state, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U || result->observed_state != record.state ||
      !LoadKvafP8PrivateRecordFromGm(record_gm, &record) ||
      !KvafP8PrivateRecordBelongsToForwardTerminal(record, forward_control) ||
      record.state != kKvafP8PrivateQuarantined ||
      record.output_index < 0 ||
      static_cast<uint64_t>(record.output_index) != private_ready_index) {
    return false;
  }
  private_ready[private_ready_index] = static_cast<int32_t>(
      kKvafP8PrivateQuarantined);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
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
    uint64_t private_record_count,
    __ubuf__ KvafP8PrivateReleaseResult* transition_result,
    event_t simt_event, KvafP8WaveRecord* record) {
  if (private_records == nullptr || record == nullptr ||
      transition_result == nullptr ||
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
        !KvafP8PrivateRecordMatchesWave(private_record, *record) ||
        !CommitKvafP8PrivateTerminalFromAicore(
            private_records + static_cast<uint64_t>(output_index),
            record->identity, payload_ready, transition_result, simt_event,
            &private_record)) {
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
// consumed.  The CAS accepts POSTED only: FILLING and POST_INTENT are
// pre-doorbell states and must never be treated as a CQ-drainable wave.
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

// FILLING and its admission-claim predecessor have narrow transitions that
// must not be implemented as a normal GM Store.  Forward-terminal recovery may race an empty producer after it
// has acquired a queue lease.  A full-record write after that recovery wins
// would silently resurrect a QUARANTINED record.  Keep the permitted edges
// deliberately narrow and use the same single-thread SIMT CAS service as the
// queue and CQ-drain ownership protocols:
//
//   FILLING -> POST_INTENT              (descriptor/ledger are published)
//   FILLING -> FILLING_RELEASE_CLAIMED  (proved empty, one release owner)
//   FILLING -> QUARANTINED              (failed closed)
//   FILLING_RELEASE_CLAIMED -> FREE     (owner-table CAS succeeded)
//   FILLING_RELEASE_CLAIMED -> QUARANTINED (release failure/terminal)
//   POST_INTENT -> QUARANTINED          (pre-PostOnly control failure)
//   FILLING_CLAIMED -> FILLING          (identity/payload is published)
//   FILLING_CLAIMED -> QUARANTINED      (admission/terminal failure)
//
// No other state may use this endpoint.  In particular, CQ drain continues
// to accept POSTED only, so a durable FILLING record can never be mistaken
// for a doorbelled transaction.
__simt_callee__ inline bool TryTransitionKvafP8FillingStateFromSimt(
    __gm__ KvafP8WaveRecord* record, uint32_t expected_state,
    uint32_t desired_state, __ubuf__ KvafP8WaveDrainClaimResult* result) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8WaveFree;
  const bool is_filling_edge =
      expected_state == kKvafP8WaveFilling &&
      (desired_state == kKvafP8WavePostIntent ||
       desired_state == kKvafP8WaveFillingReleaseClaimed ||
       desired_state == kKvafP8WaveQuarantined);
  const bool is_release_claimed_edge =
      expected_state == kKvafP8WaveFillingReleaseClaimed &&
      (desired_state == kKvafP8WaveFree ||
       desired_state == kKvafP8WaveQuarantined);
  const bool is_post_intent_quarantine_edge =
      expected_state == kKvafP8WavePostIntent &&
      desired_state == kKvafP8WaveQuarantined;
  const bool is_admission_claimed_edge =
      expected_state == kKvafP8WaveFillingClaimed &&
      (desired_state == kKvafP8WaveFilling ||
       desired_state == kKvafP8WaveQuarantined);
  if (!is_filling_edge && !is_release_claimed_edge &&
      !is_post_intent_quarantine_edge && !is_admission_claimed_edge) {
    return false;
  }
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(expected_state),
      static_cast<int32_t>(desired_state));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(expected_state)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8FillingStateTransitionSimtService(
    __gm__ KvafP8WaveRecord* record, uint32_t expected_state,
    uint32_t desired_state, __ubuf__ KvafP8WaveDrainClaimResult* result) {
  (void)TryTransitionKvafP8FillingStateFromSimt(
      record, expected_state, desired_state, result);
}

__aicore__ inline bool TryTransitionKvafP8FillingStateFromAicore(
    __gm__ KvafP8WaveRecord* record, uint32_t expected_state,
    uint32_t desired_state, __ubuf__ KvafP8WaveDrainClaimResult* result,
    event_t simt_event) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8WaveFree;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8FillingStateTransitionSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record, expected_state,
      desired_state, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U || result->observed_state != expected_state) {
    return false;
  }
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

// State-last commit protocol ------------------------------------------------
//
// A P8 wave is durable GM state shared by the producer, dynamic CQ drain
// workers, and the forward-terminal sweep.  Therefore a worker must never
// turn a locally-mutated `KvafP8WaveRecord` into truth with a full GM Store:
// that could overwrite a terminal QUARANTINED state which won while the
// worker was inside PostOnly/Retire/Finish.  The narrow helpers below provide
// the only transition boundary for the post/drain/release states used by P8:
//
//   1. reload and prove the complete identity plus expected state;
//   2. write only mutable payload words and issue a DDR publish barrier;
//   3. CAS the state word from expected -> desired; and
//   4. reload to prove the committed state still belongs to this identity.
//
// The payload-first ordering is intentional.  A competing terminal CAS may
// win after step 2, in which case this worker's final CAS fails and cannot
// resurrect the wave.  It must not write another field after that failure.
// Fields that establish the admitted wave (magic/version/identity, request
// count, and the pre-Post queue indices) are immutable at these boundaries.

__aicore__ inline bool IsKvafP8WaveStateLastCommitEdgeAllowed(
    uint32_t expected_state, uint32_t desired_state) {
  const bool post_intent_edge =
      expected_state == kKvafP8WavePostIntent &&
      (desired_state == kKvafP8WavePosted ||
       desired_state == kKvafP8WaveFinalized ||
       desired_state == kKvafP8WaveQuarantined);
  const bool drain_claimed_edge =
      expected_state == kKvafP8WaveDrainClaimed &&
      (desired_state == kKvafP8WavePosted ||
       desired_state == kKvafP8WaveFinalized ||
       desired_state == kKvafP8WaveQuarantined);
  const bool finalized_edge =
      expected_state == kKvafP8WaveFinalized &&
      (desired_state == kKvafP8WaveFree ||
       desired_state == kKvafP8WaveQuarantined);
  return post_intent_edge || drain_claimed_edge || finalized_edge;
}

__aicore__ inline bool IsKvafP8WaveStateLastPayloadValid(
    const KvafP8WaveRecord& observed, const KvafP8WaveRecord& payload,
    uint32_t expected_state, uint32_t desired_state,
    const KvafP8WaveIdentity& expected_identity) {
  if (!IsKvafP8WaveRecordValid(observed) ||
      !IsKvafP8WaveRecordValid(payload) ||
      observed.state != expected_state || payload.state != desired_state ||
      !KvafP8WaveIdentityEquals(observed.identity, expected_identity) ||
      !KvafP8WaveIdentityEquals(payload.identity, expected_identity) ||
      observed.magic != payload.magic || observed.version != payload.version ||
      observed.sq_producer_before != payload.sq_producer_before ||
      observed.cq_consumer_before != payload.cq_consumer_before ||
      observed.request_count != payload.request_count) {
    return false;
  }
  // reserved[0] is the CQ-observed marker and is mutable with the drain
  // payload.  The FILLING proof and remaining reserved words must never be
  // republished by a post/drain worker from a stale local record.
  for (uint32_t index = 1U; index < 5U; ++index) {
    if (observed.reserved[index] != payload.reserved[index]) {
      return false;
    }
  }
  if (desired_state == kKvafP8WavePosted) {
    return payload.post_committed == 1U &&
           payload.transaction.state == kDenseBatchGetInFlight &&
           payload.transaction.finalized == 0U;
  }
  if (desired_state == kKvafP8WaveFinalized) {
    return payload.transaction.finalized != 0U;
  }
  if (desired_state == kKvafP8WaveFree) {
    // This edge is legal only after the caller has obtained the exact
    // owner-table release proof.  It cannot free a possibly posted wave.
    return payload.post_committed == 0U &&
           payload.transaction.finalized != 0U;
  }
  // QUARANTINED is terminal by state alone.  The caller supplies the
  // diagnostic outcome in `payload`, but no non-terminal transaction fact is
  // used to make a quarantined record reusable.
  return desired_state == kKvafP8WaveQuarantined;
}

// Publish only fields that are allowed to change after POST_INTENT.  In
// particular, this deliberately does not touch `state`, identity, request
// count, pre-Post queue indices, FILLING proof, or any fixed-record ABI
// layout.  It is private to the state-last commit helper below.
__aicore__ inline bool PublishKvafP8WaveStateLastMutablePayloadToGm(
    __gm__ KvafP8WaveRecord* record_gm,
    const KvafP8WaveRecord& payload) {
  if (record_gm == nullptr) {
    return false;
  }
  record_gm->sq_producer_after = payload.sq_producer_after;
  record_gm->cq_consumer_after = payload.cq_consumer_after;
  record_gm->outcome = payload.outcome;
  record_gm->private_ready_published = payload.private_ready_published;
  record_gm->post_committed = payload.post_committed;
  record_gm->transaction.active_count = payload.transaction.active_count;
  record_gm->transaction.total_posts = payload.transaction.total_posts;
  record_gm->transaction.failed_request = payload.transaction.failed_request;
  record_gm->transaction.last_wqe = payload.transaction.last_wqe;
  record_gm->transaction.first_doorbell_cycle =
      payload.transaction.first_doorbell_cycle;
  record_gm->transaction.profile_start_cycle =
      payload.transaction.profile_start_cycle;
  record_gm->transaction.profile_setup_done_cycle =
      payload.transaction.profile_setup_done_cycle;
  record_gm->transaction.profile_compact_done_cycle =
      payload.transaction.profile_compact_done_cycle;
  record_gm->transaction.profile_post_done_cycle =
      payload.transaction.profile_post_done_cycle;
  record_gm->transaction.profile_completion_done_cycle =
      payload.transaction.profile_completion_done_cycle;
  record_gm->transaction.result = payload.transaction.result;
  record_gm->transaction.completion_polls =
      payload.transaction.completion_polls;
  record_gm->transaction.state = payload.transaction.state;
  record_gm->transaction.profile_enabled =
      payload.transaction.profile_enabled;
  record_gm->transaction.finalized = payload.transaction.finalized;
  record_gm->reserved[0] = payload.reserved[0];
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

__simt_callee__ inline bool TryCommitKvafP8WaveStateLastFromSimt(
    __gm__ KvafP8WaveRecord* record, uint32_t expected_state,
    uint32_t desired_state, __ubuf__ KvafP8WaveDrainClaimResult* result) {
  const bool post_intent_edge =
      expected_state == kKvafP8WavePostIntent &&
      (desired_state == kKvafP8WavePosted ||
       desired_state == kKvafP8WaveFinalized ||
       desired_state == kKvafP8WaveQuarantined);
  const bool drain_claimed_edge =
      expected_state == kKvafP8WaveDrainClaimed &&
      (desired_state == kKvafP8WavePosted ||
       desired_state == kKvafP8WaveFinalized ||
       desired_state == kKvafP8WaveQuarantined);
  const bool finalized_edge =
      expected_state == kKvafP8WaveFinalized &&
      (desired_state == kKvafP8WaveFree ||
       desired_state == kKvafP8WaveQuarantined);
  if (record == nullptr || result == nullptr ||
      (!post_intent_edge && !drain_claimed_edge && !finalized_edge)) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8WaveFree;
  __gm__ int32_t* const state =
      reinterpret_cast<__gm__ int32_t*>(&(record->state));
  const int32_t observed = asc_atomic_cas(
      state, static_cast<int32_t>(expected_state),
      static_cast<int32_t>(desired_state));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(expected_state)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8WaveStateLastCommitSimtService(
    __gm__ KvafP8WaveRecord* record, uint32_t expected_state,
    uint32_t desired_state, __ubuf__ KvafP8WaveDrainClaimResult* result) {
  (void)TryCommitKvafP8WaveStateLastFromSimt(
      record, expected_state, desired_state, result);
}

__aicore__ inline bool CommitKvafP8WaveMutablePayloadAndStateFromAicore(
    __gm__ KvafP8WaveRecord* record_gm,
    const KvafP8WaveIdentity& expected_identity, uint32_t expected_state,
    uint32_t desired_state, const KvafP8WaveRecord& payload,
    __ubuf__ KvafP8WaveDrainClaimResult* transition_result,
    event_t simt_event, KvafP8WaveRecord* committed_record) {
  if (record_gm == nullptr || transition_result == nullptr ||
      !IsKvafP8WaveStateLastCommitEdgeAllowed(expected_state, desired_state)) {
    return false;
  }
  KvafP8WaveRecord observed{};
  if (!LoadKvafP8WaveRecordFromGm(record_gm, &observed) ||
      !IsKvafP8WaveStateLastPayloadValid(
          observed, payload, expected_state, desired_state,
          expected_identity) ||
      !PublishKvafP8WaveStateLastMutablePayloadToGm(record_gm, payload)) {
    return false;
  }
  transition_result->success = 0U;
  transition_result->observed_state = kKvafP8WaveFree;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8WaveStateLastCommitSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record_gm, expected_state,
      desired_state, transition_result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (transition_result->success == 0U ||
      transition_result->observed_state != expected_state) {
    return false;
  }
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  KvafP8WaveRecord committed{};
  if (!LoadKvafP8WaveRecordFromGm(record_gm, &committed) ||
      !IsKvafP8WaveRecordValid(committed) ||
      committed.state != desired_state ||
      !KvafP8WaveIdentityEquals(committed.identity, expected_identity)) {
    return false;
  }
  if (committed_record != nullptr) {
    (void)LoadKvafP8WaveRecordFromGm(record_gm, committed_record);
  }
  return true;
}

// The normal state-last helper intentionally does not allow
// DRAIN_CLAIMED -> FREE: that broad edge would make it too easy for a CQ
// drain to recycle a queue before a possibly-posted transaction has reached
// its terminal/private-publication proof.  There is exactly one exception.
// A worker which atomically claimed a durable POST_REJECTED FINALIZED record,
// released every associated PENDING private row, and then successfully
// released the queue owner may make its record reusable.  Keep that exception
// in this dedicated proof so native P8 code cannot fall back to a raw full
// record Store after `ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup`.
//
// The input `payload` is the caller-local record returned by that release
// helper: it has state FREE, but its immutable identity/ledger evidence is
// retained.  We publish only mutable transaction/outcome words, issue the
// DDR barrier, then CAS the durable state last.  A terminal quarantine that
// wins the CAS keeps the record non-reusable; this helper never resurrects it.
__aicore__ inline bool IsKvafP8ClaimedUnpostedWaveReleasePayloadValid(
    const KvafP8WaveRecord& observed, const KvafP8WaveRecord& payload,
    const KvafP8WaveIdentity& expected_identity) {
  if (!IsKvafP8WaveStateLastPayloadValid(
          observed, payload, kKvafP8WaveDrainClaimed, kKvafP8WaveFree,
          expected_identity)) {
    return false;
  }
  // A deterministic no-post wave is the only safe DRAIN_CLAIMED -> FREE
  // case.  It cannot have published a usable private payload or observed a
  // CQ.  `total_posts` is deliberately not required to be zero: PostOnly
  // computes it before descriptor validation, even when it never rings SQ.
  return observed.post_committed == 0U &&
         payload.post_committed == 0U &&
         observed.outcome == kKvafP8WaveOutcomePostRejected &&
         payload.outcome == kKvafP8WaveOutcomePostRejected &&
         observed.private_ready_published == 0U &&
         payload.private_ready_published == 0U &&
         observed.reserved[0] == 0U && payload.reserved[0] == 0U &&
         observed.transaction.state == kDenseBatchGetFailed &&
         payload.transaction.state == kDenseBatchGetFailed &&
         observed.transaction.finalized != 0U &&
         payload.transaction.finalized != 0U;
}

__simt_callee__ inline bool
TryCommitKvafP8ClaimedUnpostedWaveReleaseFromSimt(
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
      state, static_cast<int32_t>(kKvafP8WaveDrainClaimed),
      static_cast<int32_t>(kKvafP8WaveFree));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(kKvafP8WaveDrainClaimed)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8ClaimedUnpostedWaveReleaseSimtService(
    __gm__ KvafP8WaveRecord* record,
    __ubuf__ KvafP8WaveDrainClaimResult* result) {
  (void)TryCommitKvafP8ClaimedUnpostedWaveReleaseFromSimt(record, result);
}

// State-last durable completion for the sole deterministic no-post release
// edge.  It preserves the P8 record ABI and exposes no generic
// DRAIN_CLAIMED -> FREE transition.
__aicore__ inline bool CommitKvafP8ClaimedUnpostedWaveReleaseFromAicore(
    __gm__ KvafP8WaveRecord* record_gm,
    const KvafP8WaveIdentity& expected_identity,
    const KvafP8WaveRecord& payload,
    __ubuf__ KvafP8WaveDrainClaimResult* transition_result,
    event_t simt_event, KvafP8WaveRecord* committed_record) {
  if (record_gm == nullptr || transition_result == nullptr) {
    return false;
  }
  KvafP8WaveRecord observed{};
  if (!LoadKvafP8WaveRecordFromGm(record_gm, &observed) ||
      !IsKvafP8ClaimedUnpostedWaveReleasePayloadValid(
          observed, payload, expected_identity) ||
      !PublishKvafP8WaveStateLastMutablePayloadToGm(record_gm, payload)) {
    return false;
  }
  transition_result->success = 0U;
  transition_result->observed_state = kKvafP8WaveFree;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8ClaimedUnpostedWaveReleaseSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record_gm, transition_result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (transition_result->success == 0U ||
      transition_result->observed_state != kKvafP8WaveDrainClaimed) {
    return false;
  }
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  KvafP8WaveRecord committed{};
  if (!LoadKvafP8WaveRecordFromGm(record_gm, &committed) ||
      !IsKvafP8WaveRecordValid(committed) ||
      committed.state != kKvafP8WaveFree ||
      !KvafP8WaveIdentityEquals(committed.identity, expected_identity) ||
      committed.magic != payload.magic ||
      committed.version != payload.version ||
      committed.sq_producer_before != payload.sq_producer_before ||
      committed.cq_consumer_before != payload.cq_consumer_before ||
      committed.request_count != payload.request_count ||
      committed.sq_producer_after != payload.sq_producer_after ||
      committed.cq_consumer_after != payload.cq_consumer_after ||
      committed.outcome != kKvafP8WaveOutcomePostRejected ||
      committed.private_ready_published != 0U ||
      committed.post_committed != 0U ||
      committed.reserved[0] != 0U ||
      committed.transaction.state != kDenseBatchGetFailed ||
      committed.transaction.finalized == 0U) {
    return false;
  }
  if (committed_record != nullptr) {
    (void)LoadKvafP8WaveRecordFromGm(record_gm, committed_record);
  }
  return true;
}

// Admission uses a separate claim because a first-use graph-stable slot is
// all zero and therefore has no complete identity to validate yet.  Claiming
// only the state word first prevents two dynamic producers from both seeing
// FREE and then full-storing different identities into the same queue record.
// A forward terminal may always CAS CLAIMED -> QUARANTINED without inspecting
// identity; this is what closes the identity-publication interruption window.
__simt_callee__ inline bool TryClaimKvafP8WaveFillingAdmissionFromSimt(
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
      state, static_cast<int32_t>(kKvafP8WaveFree),
      static_cast<int32_t>(kKvafP8WaveFillingClaimed));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(kKvafP8WaveFree)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8WaveFillingAdmissionClaimSimtService(
    __gm__ KvafP8WaveRecord* record,
    __ubuf__ KvafP8WaveDrainClaimResult* result) {
  (void)TryClaimKvafP8WaveFillingAdmissionFromSimt(record, result);
}

__aicore__ inline bool TryClaimKvafP8WaveFillingAdmissionFromAicore(
    __gm__ KvafP8WaveRecord* record,
    __ubuf__ KvafP8WaveDrainClaimResult* result, event_t simt_event) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8WaveFree;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8WaveFillingAdmissionClaimSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U ||
      result->observed_state != kKvafP8WaveFree) {
    return false;
  }
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

__simt_callee__ inline bool TryQuarantineKvafP8WaveFillingClaimedFromSimt(
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
      state, static_cast<int32_t>(kKvafP8WaveFillingClaimed),
      static_cast<int32_t>(kKvafP8WaveQuarantined));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(kKvafP8WaveFillingClaimed)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8WaveFillingClaimedQuarantineSimtService(
    __gm__ KvafP8WaveRecord* record,
    __ubuf__ KvafP8WaveDrainClaimResult* result) {
  (void)TryQuarantineKvafP8WaveFillingClaimedFromSimt(record, result);
}

// This deliberately does not update outcome or identity after the CAS.  A
// producer may have been interrupted halfway through its field-wise payload
// publication, so the only safe terminal fact is the QUARANTINED state word.
__aicore__ inline bool QuarantineKvafP8WaveFillingClaimedFromAicore(
    __gm__ KvafP8WaveRecord* record,
    __ubuf__ KvafP8WaveDrainClaimResult* result, event_t simt_event) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8WaveFree;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8WaveFillingClaimedQuarantineSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U ||
      result->observed_state != kKvafP8WaveFillingClaimed) {
    return false;
  }
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

// A deterministic Begin/Post rejection is different from POST_INTENT
// ambiguity.  The former leaves a durable FINALIZED record with
// `post_committed == 0` and no CQ can exist; retaining that fact lets a later
// dynamic drain worker return the fixed descriptor queue *and* the PENDING
// private rows without turning an ordinary no-post into permanent capacity
// loss.  It still needs its own GM CAS: two later graph invocations must not
// both clear the same source/object directory entries.
//
// The state CAS deliberately does not inspect `outcome` or `post_committed`.
// Those fields are loaded and verified by the caller after the claim.  This is
// the same fail-closed pattern as a normal CQ drain: a corrupt FINALIZED
// record becomes QUARANTINED, never a candidate for an unsafe release.
__simt_callee__ inline bool TryClaimKvafP8UnpostedWaveDrainFromSimt(
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
      state, static_cast<int32_t>(kKvafP8WaveFinalized),
      static_cast<int32_t>(kKvafP8WaveDrainClaimed));
  result->observed_state = static_cast<uint32_t>(observed);
  if (observed != static_cast<int32_t>(kKvafP8WaveFinalized)) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  result->success = 1U;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafP8UnpostedWaveDrainClaimSimtService(
    __gm__ KvafP8WaveRecord* record,
    __ubuf__ KvafP8WaveDrainClaimResult* result) {
  (void)TryClaimKvafP8UnpostedWaveDrainFromSimt(record, result);
}

__aicore__ inline bool TryClaimKvafP8UnpostedWaveDrainFromAicore(
    __gm__ KvafP8WaveRecord* record,
    __ubuf__ KvafP8WaveDrainClaimResult* result, event_t simt_event) {
  if (record == nullptr || result == nullptr) {
    return false;
  }
  result->success = 0U;
  result->observed_state = kKvafP8WaveFree;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafP8UnpostedWaveDrainClaimSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, record, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  return result->success != 0U &&
         result->observed_state == kKvafP8WaveFinalized;
}

// P8 phase helpers ---------------------------------------------------------
//
// A P8 executor invokes these helpers in short graph-visible phases.  Every
// phase that changes the durable wave state must use the state-last CAS
// helpers below; there is intentionally no generic full-record GM publisher
// for PostOnly/CQ/Finish paths.  This prevents an AIC-local drain snapshot
// from overwriting a terminal quarantine after the snapshot was taken.

__aicore__ inline void QuarantineKvafP8WaveRecord(
    KvafP8WaveRecord* record, KvafP8WaveOutcome outcome) {
  if (record == nullptr) {
    return;
  }
  record->state = kKvafP8WaveQuarantined;
  record->outcome = static_cast<uint32_t>(outcome);
}

// Fail-close one of the two pre-doorbell FILLING states without a full-record
// Store.  The CAS is the ownership boundary; publishing `outcome` afterwards
// is safe only for the worker that won that CAS.  A loser must not write any
// field because its observed record may already be the forward-terminal
// QUARANTINED state.
__aicore__ inline bool QuarantineKvafP8FillingStateFromAicore(
    __gm__ KvafP8WaveRecord* record_gm, uint32_t expected_state,
    KvafP8WaveOutcome outcome,
    __ubuf__ KvafP8WaveDrainClaimResult* transition_result,
    event_t simt_event, KvafP8WaveRecord* record) {
  if (!TryTransitionKvafP8FillingStateFromAicore(
          record_gm, expected_state, kKvafP8WaveQuarantined,
          transition_result, simt_event)) {
    return false;
  }
  record_gm->outcome = static_cast<uint32_t>(outcome);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  if (record != nullptr) {
    record->state = kKvafP8WaveQuarantined;
    record->outcome = static_cast<uint32_t>(outcome);
  }
  return true;
}

// A successful FILLING -> POST_INTENT commit makes descriptor/private/ledger
// state externally visible, but it has not rung the HCCP doorbell yet.  If
// the executor cannot reload a matching AdmitPost control word immediately
// before PostOnly, it must make that ambiguity durable.  Do not use a normal
// full-record Store here: ForwardTerminal may have won the same race.  The
// GM reload + complete identity check prevents a stale producer from
// quarantining a later forward's wave that reused the queue id.
__aicore__ inline bool QuarantineKvafP8PostIntentStateFromAicore(
    __gm__ KvafP8WaveRecord* record_gm,
    const KvafP8WaveIdentity& expected_identity,
    __ubuf__ KvafP8WaveDrainClaimResult* transition_result,
    event_t simt_event, KvafP8WaveRecord* record) {
  KvafP8WaveRecord observed{};
  if (record_gm == nullptr || transition_result == nullptr ||
      !LoadKvafP8WaveRecordFromGm(record_gm, &observed) ||
      !IsKvafP8WaveRecordValid(observed) ||
      observed.state != kKvafP8WavePostIntent ||
      !KvafP8WaveIdentityEquals(observed.identity, expected_identity)) {
    return false;
  }
  if (!TryTransitionKvafP8FillingStateFromAicore(
          record_gm, kKvafP8WavePostIntent, kKvafP8WaveQuarantined,
          transition_result, simt_event)) {
    return false;
  }
  record_gm->outcome =
      static_cast<uint32_t>(kKvafP8WaveOutcomePostIntentQuarantined);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  if (record != nullptr) {
    record->state = kKvafP8WaveQuarantined;
    record->outcome =
        static_cast<uint32_t>(kKvafP8WaveOutcomePostIntentQuarantined);
  }
  return true;
}

// Publish only the payload which turns an already durable FILLING record
// into a ready POST_INTENT candidate.  Do not write `state`, `outcome`, or
// identity here: a terminal sweep may have CASed FILLING to QUARANTINED while
// validation was running.  The following FILLING -> POST_INTENT CAS decides
// whether this payload becomes live; if it loses, the record stays
// QUARANTINED and no HCCP Post is allowed.
__aicore__ inline bool PublishKvafP8FillingPostIntentPayloadToGm(
    __gm__ KvafP8WaveRecord* record_gm,
    const KvafP8WaveRecord& record) {
  if (record_gm == nullptr || !IsKvafP8WaveRecordValid(record) ||
      record.state != kKvafP8WaveFilling ||
      record.reserved[1] != kKvafP8FillingExposureProof) {
    return false;
  }
  record_gm->sq_producer_before = record.sq_producer_before;
  record_gm->cq_consumer_before = record.cq_consumer_before;
  record_gm->request_count = record.request_count;
  // Clear the no-work/exposure proof before the state CAS.  The state remains
  // FILLING until that CAS succeeds, so this can only make a racing terminal
  // more conservative; it can never make it release a descriptor-bearing
  // wave as empty.
  record_gm->reserved[1] = 0U;
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

// `FILLING` exists specifically to close the old lease-acquire -> descriptor
// publication hole.  The owner table CAS is necessary but not sufficient:
// if a dynamic task exits after the CAS and before a durable record exists,
// no later drain/terminal phase can decide whether it is safe to reuse that
// queue.  `BeginAndPublishKvafP8WaveFilling` claims the state word first,
// then publishes this fully initialized record before touching any externally
// visible descriptor, private record, or request-ledger byte.
__aicore__ inline bool BeginKvafP8WaveFilling(
    const KvafStreamingBatchGetControl& control,
    const KvafP8ForwardControl& forward_control,
    const KvafStreamingBatchGetLease& lease,
    const KvafP8WaveIdentity& identity, KvafP8WaveRecord* record) {
  if (record == nullptr) {
    return false;
  }
  ResetKvafP8WaveRecord(record, identity);
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      lease.state != kStreamingBatchGetLeaseHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count || identity.forward_epoch == 0U ||
      identity.queue_generation == 0U || identity.post_sequence == 0U ||
      identity.reserved0 != 0U || identity.queue_id != lease.queue_id ||
      identity.owner_tag != lease.owner_tag ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, identity, kKvafP8ForwardPhaseAdmitPost)) {
    QuarantineKvafP8WaveRecord(record,
                                kKvafP8WaveOutcomeFillingQuarantined);
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners == nullptr || owners[lease.queue_id] != lease.owner_tag) {
    QuarantineKvafP8WaveRecord(record,
                                kKvafP8WaveOutcomeFillingQuarantined);
    return false;
  }
  record->state = kKvafP8WaveFilling;
  // This marker is the only normal proof that no external P8 work has been
  // exposed.  ArmKvafP8WaveFillingForExternalWrites clears that proof before
  // any descriptor/private/ledger mutation.
  record->reserved[1] = kKvafP8FillingNoWorkProof;
  return true;
}

// A graph-stable wave slot is admitted only from a fresh all-zero allocation
// or a previously completed, valid FREE record.  Do not overwrite a malformed
// or non-FREE record: after a process interruption it may still describe a
// queue/private lifetime which only cold recovery can prove quiescent.
__aicore__ inline bool IsKvafP8WaveRecordAllZero(
    const KvafP8WaveRecord& record) {
  if (record.magic != 0U || record.version != 0U || record.state != 0U ||
      record.identity.forward_epoch != 0U ||
      record.identity.graph_replay_generation != 0U ||
      record.identity.layer_control_index != 0U || record.identity.queue_id != 0U ||
      record.identity.queue_generation != 0U ||
      record.identity.post_sequence != 0U || record.identity.owner_tag != 0 ||
      record.identity.reserved0 != 0U || record.sq_producer_before != 0U ||
      record.cq_consumer_before != 0U || record.sq_producer_after != 0U ||
      record.cq_consumer_after != 0U || record.request_count != 0U ||
      record.outcome != 0U || record.private_ready_published != 0U ||
      record.post_committed != 0U || record.transaction.active_count != 0U ||
      record.transaction.total_posts != 0U ||
      record.transaction.failed_request != 0U ||
      record.transaction.last_wqe != 0U ||
      record.transaction.first_doorbell_cycle != 0U ||
      record.transaction.profile_start_cycle != 0U ||
      record.transaction.profile_setup_done_cycle != 0U ||
      record.transaction.profile_compact_done_cycle != 0U ||
      record.transaction.profile_post_done_cycle != 0U ||
      record.transaction.profile_completion_done_cycle != 0U ||
      record.transaction.result != 0U ||
      record.transaction.completion_polls != 0U ||
      record.transaction.state != 0U ||
      record.transaction.profile_enabled != 0U ||
      record.transaction.finalized != 0U) {
    return false;
  }
  for (uint32_t index = 0U; index < 5U; ++index) {
    if (record.reserved[index] != 0U) {
      return false;
    }
  }
  return true;
}

__aicore__ inline bool IsKvafP8WaveRecordAdmissibleForFilling(
    const KvafP8WaveRecord& record) {
  return IsKvafP8WaveRecordAllZero(record) ||
         (IsKvafP8WaveRecordValid(record) &&
          record.state == kKvafP8WaveFree);
}

// Publish a complete FILLING payload while the state word is already held in
// FILLING_CLAIMED.  This is deliberately field-wise and omits `state`: a
// terminal worker can CAS CLAIMED -> QUARANTINED at any point, and this
// producer must never revive that terminal result with a stale full-record
// Store.  If terminal wins while this helper is writing, its state stays
// QUARANTINED and the final CLAIMED -> FILLING CAS below fails closed.
__aicore__ inline bool PublishKvafP8WaveFillingClaimedPayloadToGm(
    __gm__ KvafP8WaveRecord* record_gm,
    const KvafP8WaveRecord& record) {
  if (record_gm == nullptr || !IsKvafP8WaveRecordValid(record) ||
      record.state != kKvafP8WaveFilling) {
    return false;
  }
  record_gm->magic = record.magic;
  record_gm->version = record.version;
  record_gm->identity.forward_epoch = record.identity.forward_epoch;
  record_gm->identity.graph_replay_generation =
      record.identity.graph_replay_generation;
  record_gm->identity.layer_control_index = record.identity.layer_control_index;
  record_gm->identity.queue_id = record.identity.queue_id;
  record_gm->identity.queue_generation = record.identity.queue_generation;
  record_gm->identity.post_sequence = record.identity.post_sequence;
  record_gm->identity.owner_tag = record.identity.owner_tag;
  record_gm->identity.reserved0 = record.identity.reserved0;
  record_gm->sq_producer_before = record.sq_producer_before;
  record_gm->cq_consumer_before = record.cq_consumer_before;
  record_gm->sq_producer_after = record.sq_producer_after;
  record_gm->cq_consumer_after = record.cq_consumer_after;
  record_gm->request_count = record.request_count;
  record_gm->outcome = record.outcome;
  record_gm->private_ready_published = record.private_ready_published;
  record_gm->post_committed = record.post_committed;
  record_gm->transaction.active_count = record.transaction.active_count;
  record_gm->transaction.total_posts = record.transaction.total_posts;
  record_gm->transaction.failed_request = record.transaction.failed_request;
  record_gm->transaction.last_wqe = record.transaction.last_wqe;
  record_gm->transaction.first_doorbell_cycle =
      record.transaction.first_doorbell_cycle;
  record_gm->transaction.profile_start_cycle =
      record.transaction.profile_start_cycle;
  record_gm->transaction.profile_setup_done_cycle =
      record.transaction.profile_setup_done_cycle;
  record_gm->transaction.profile_compact_done_cycle =
      record.transaction.profile_compact_done_cycle;
  record_gm->transaction.profile_post_done_cycle =
      record.transaction.profile_post_done_cycle;
  record_gm->transaction.profile_completion_done_cycle =
      record.transaction.profile_completion_done_cycle;
  record_gm->transaction.result = record.transaction.result;
  record_gm->transaction.completion_polls =
      record.transaction.completion_polls;
  record_gm->transaction.state = record.transaction.state;
  record_gm->transaction.profile_enabled =
      record.transaction.profile_enabled;
  record_gm->transaction.finalized = record.transaction.finalized;
  for (uint32_t index = 0U; index < 5U; ++index) {
    record_gm->reserved[index] = record.reserved[index];
  }
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

// Preferred P8 admission entry point.  It first claims FREE/all-zero state
// atomically, then writes identity/payload, then commits CLAIMED -> FILLING.
// This closes the old check-FREE then full-Store race between two dynamic
// producers and gives ForwardTerminal a state it can quarantine even before
// the new identity is complete.
__aicore__ inline bool BeginAndPublishKvafP8WaveFilling(
    __gm__ KvafP8WaveRecord* record_gm,
    const KvafStreamingBatchGetControl& control,
    const KvafP8ForwardControl& forward_control,
    const KvafStreamingBatchGetLease& lease,
    const KvafP8WaveIdentity& identity,
    __ubuf__ KvafP8WaveDrainClaimResult* transition_result,
    event_t simt_event, KvafP8WaveRecord* record) {
  KvafP8WaveRecord prior{};
  if (record_gm == nullptr ||
      !LoadKvafP8WaveRecordFromGm(record_gm, &prior) ||
      !IsKvafP8WaveRecordAdmissibleForFilling(prior)) {
    // A non-FREE or malformed slot is not overwritten.  The caller must set
    // the forward-abort gate and retain its held lease for terminal/cold
    // recovery rather than risking an ABA overwrite.
    return false;
  }
  // This is the only write to the state word before the complete identity is
  // visible.  A terminal sweep may now safely quarantine the slot without
  // trying to interpret partial payload bytes.
  if (!TryClaimKvafP8WaveFillingAdmissionFromAicore(
          record_gm, transition_result, simt_event)) {
    return false;
  }
  const bool begun = BeginKvafP8WaveFilling(
      control, forward_control, lease, identity, record);
  if (!begun || record == nullptr || !IsKvafP8WaveRecordValid(*record) ||
      !PublishKvafP8WaveFillingClaimedPayloadToGm(record_gm, *record)) {
    (void)QuarantineKvafP8WaveFillingClaimedFromAicore(
        record_gm, transition_result, simt_event);
    return false;
  }
  // State is the final admission commit edge.  No field is written after a
  // failed CAS because the terminal worker may own the QUARANTINED record.
  if (!TryTransitionKvafP8FillingStateFromAicore(
          record_gm, kKvafP8WaveFillingClaimed, kKvafP8WaveFilling,
          transition_result, simt_event)) {
    return false;
  }
  KvafP8WaveRecord committed{};
  if (!LoadKvafP8WaveRecordFromGm(record_gm, &committed) ||
      !IsKvafP8WaveRecordValid(committed) ||
      committed.state != kKvafP8WaveFilling ||
      !KvafP8WaveIdentityEquals(committed.identity, identity)) {
    return false;
  }
  (void)LoadKvafP8WaveRecordFromGm(record_gm, record);
  return true;
}

__aicore__ inline bool KvafP8WaveFillingMatchesLeaseAndIdentity(
    const KvafStreamingBatchGetControl& control,
    const KvafP8ForwardControl& forward_control,
    const KvafStreamingBatchGetLease& lease,
    const KvafP8WaveIdentity& identity,
    const KvafP8WaveRecord& record) {
  if (!IsKvafP8WaveRecordValid(record) ||
      record.state != kKvafP8WaveFilling ||
      !KvafP8WaveIdentityEquals(record.identity, identity) ||
      !IsKvafStreamingBatchGetControlValidForAicore(control) ||
      lease.state != kStreamingBatchGetLeaseHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count ||
      identity.forward_epoch == 0U || identity.queue_generation == 0U ||
      identity.post_sequence == 0U || identity.reserved0 != 0U ||
      identity.queue_id != lease.queue_id ||
      identity.owner_tag != lease.owner_tag ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, identity, kKvafP8ForwardPhaseAdmitPost)) {
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  return owners != nullptr && owners[lease.queue_id] == lease.owner_tag;
}

// This is the mandatory pre-exposure edge.  It must be called and durably
// published *before* writing descriptor scratch, private-slot records, or
// request-ledger entries.  The marker lets a later no-work exit distinguish
// a genuinely empty FILLING record from one whose producer was interrupted
// after making any external state visible.
__aicore__ inline bool ArmKvafP8WaveFillingForExternalWrites(
    __gm__ KvafP8WaveRecord* record_gm,
    const KvafStreamingBatchGetControl& control,
    const KvafP8ForwardControl& forward_control,
    const KvafStreamingBatchGetLease& lease,
    const KvafP8WaveIdentity& identity,
    __ubuf__ KvafP8WaveDrainClaimResult* transition_result,
    event_t simt_event, KvafP8WaveRecord* record) {
  if (record_gm == nullptr || record == nullptr ||
      !LoadKvafP8WaveRecordFromGm(record_gm, record)) {
    return false;
  }
  if (!KvafP8WaveFillingMatchesLeaseAndIdentity(
          control, forward_control, lease, identity, *record) ||
      record->reserved[1] != kKvafP8FillingNoWorkProof) {
    if (IsKvafP8WaveRecordValid(*record) &&
        record->state == kKvafP8WaveFilling) {
      (void)QuarantineKvafP8FillingStateFromAicore(
          record_gm, kKvafP8WaveFilling,
          kKvafP8WaveOutcomeFillingQuarantined, transition_result,
          simt_event, record);
    }
    return false;
  }
  // This one-word publish is intentionally not a full-record Store.  A
  // racing terminal may already have changed state to QUARANTINED; writing
  // the exposure proof does not revive it, and the later FILLING ->
  // POST_INTENT CAS will then fail closed before any doorbell is possible.
  record->reserved[1] = kKvafP8FillingExposureProof;
  record_gm->reserved[1] = kKvafP8FillingExposureProof;
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

// A FILLING record can be returned without a HCCP drain only while the
// durable no-work proof remains intact.  It is deliberately stronger than
// checking `post_committed == 0`: descriptor/private/ledger bytes may have
// been exposed without a doorbell, in which case another graph invocation
// cannot prove they are harmless.  Those paths must quarantine instead.
__aicore__ inline bool IsKvafP8WaveFillingStateProvenNoWork(
    const KvafP8WaveRecord& record, uint32_t expected_state) {
  if (!IsKvafP8WaveRecordValid(record) ||
      (expected_state != kKvafP8WaveFilling &&
       expected_state != kKvafP8WaveFillingReleaseClaimed) ||
      record.state != expected_state ||
      record.identity.forward_epoch == 0U ||
      record.identity.queue_generation == 0U ||
      record.identity.post_sequence == 0U || record.identity.owner_tag <= 0 ||
      record.identity.reserved0 != 0U ||
      record.request_count != 0U ||
      record.outcome != kKvafP8WaveOutcomeNone ||
      record.private_ready_published != 0U || record.post_committed != 0U ||
      record.reserved[1] != kKvafP8FillingNoWorkProof ||
      record.sq_producer_before != 0U || record.cq_consumer_before != 0U ||
      record.sq_producer_after != 0U || record.cq_consumer_after != 0U ||
      record.transaction.active_count != 0U ||
      record.transaction.total_posts != 0U ||
      record.transaction.failed_request != UINT64_MAX ||
      record.transaction.last_wqe != 0U ||
      record.transaction.first_doorbell_cycle != 0U ||
      record.transaction.profile_start_cycle != 0U ||
      record.transaction.profile_setup_done_cycle != 0U ||
      record.transaction.profile_compact_done_cycle != 0U ||
      record.transaction.profile_post_done_cycle != 0U ||
      record.transaction.profile_completion_done_cycle != 0U ||
      record.transaction.result != kSuccess ||
      record.transaction.completion_polls != 0U ||
      record.transaction.state != kDenseBatchGetIdle ||
      record.transaction.profile_enabled != 0U ||
      record.transaction.finalized != 0U) {
    return false;
  }
  for (uint32_t index = 0U; index < 5U; ++index) {
    if (index != 1U && record.reserved[index] != 0U) {
      return false;
    }
  }
  return true;
}

__aicore__ inline bool IsKvafP8WaveFillingProvenNoWork(
    const KvafP8WaveRecord& record) {
  return IsKvafP8WaveFillingStateProvenNoWork(
      record, kKvafP8WaveFilling);
}

__aicore__ inline bool IsKvafP8WaveFillingReleaseClaimedProvenNoWork(
    const KvafP8WaveRecord& record) {
  return IsKvafP8WaveFillingStateProvenNoWork(
      record, kKvafP8WaveFillingReleaseClaimed);
}

// This is the only normal pre-doorbell release endpoint.  It loads the
// durable FILLING record, re-proves exact forward/lease ownership and the
// no-work marker, then first CASes FILLING -> FILLING_RELEASE_CLAIMED before
// it performs the owner-table CAS.  The second state CAS to FREE is required
// even after a successful owner-table release: a forward-terminal sweep may
// have quarantined the claimed record in the interim, and this helper must
// never overwrite that durable outcome.  A failed proof or CAS is durably
// QUARANTINED; the caller must then set the shared forward-abort gate rather
// than treating a local `hasWave` flag as authority.
__aicore__ inline bool ReleaseKvafP8FillingLeaseAfterProvenNoWork(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    const KvafP8ForwardControl& forward_control,
    const KvafStreamingBatchGetLease& lease,
    __ubuf__ KvafStreamingBatchGetLeaseResult* lease_result,
    __ubuf__ KvafP8WaveDrainClaimResult* transition_result,
    event_t simt_event, __gm__ KvafP8WaveRecord* record_gm,
    KvafP8WaveRecord* record) {
  if (control_gm == nullptr || lease_result == nullptr ||
      transition_result == nullptr || record_gm == nullptr || record == nullptr ||
      !LoadKvafP8WaveRecordFromGm(record_gm, record)) {
    return false;
  }
  KvafStreamingBatchGetControl control{};
  const KvafP8WaveIdentity identity = record->identity;
  const bool identity_matches_forward =
      IsKvafP8ForwardControlValid(forward_control) &&
      forward_control.phase == kKvafP8ForwardPhaseAdmitPost &&
      forward_control.forward_epoch == identity.forward_epoch &&
      forward_control.graph_replay_generation ==
          identity.graph_replay_generation &&
      forward_control.layer_control_index == identity.layer_control_index;
  if (!LoadKvafStreamingBatchGetControlFromGmForAicore(control_gm, &control) ||
      !IsKvafP8WaveFillingProvenNoWork(*record) ||
      !identity_matches_forward || lease.state != kStreamingBatchGetLeaseHeld ||
      lease.queue_id != identity.queue_id ||
      lease.owner_tag != identity.owner_tag ||
      identity.queue_id >= control.queue_count) {
    if (IsKvafP8WaveRecordValid(*record) &&
        record->state == kKvafP8WaveFilling) {
      (void)QuarantineKvafP8FillingStateFromAicore(
          record_gm, kKvafP8WaveFilling,
          kKvafP8WaveOutcomeFillingQuarantined, transition_result,
          simt_event, record);
    }
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners == nullptr || owners[identity.queue_id] != identity.owner_tag) {
    (void)QuarantineKvafP8FillingStateFromAicore(
        record_gm, kKvafP8WaveFilling,
        kKvafP8WaveOutcomeFillingQuarantined, transition_result,
        simt_event, record);
    return false;
  }

  // Claim the only normal pre-doorbell release ownership before touching the
  // queue-owner table.  If a terminal sweep won first, this fails closed and
  // no producer-side write is permitted afterwards.
  if (!TryTransitionKvafP8FillingStateFromAicore(
          record_gm, kKvafP8WaveFilling,
          kKvafP8WaveFillingReleaseClaimed, transition_result,
          simt_event)) {
    return false;
  }
  record->state = kKvafP8WaveFillingReleaseClaimed;

  // Re-load after the claim.  This proves that no terminal/quarantine state
  // or identity mutation is hidden behind an AIC-local copy before the
  // irreversible owner-table CAS.
  if (!LoadKvafP8WaveRecordFromGm(record_gm, record) ||
      !IsKvafP8WaveFillingReleaseClaimedProvenNoWork(*record) ||
      !KvafP8WaveIdentityEquals(record->identity, identity) ||
      !identity_matches_forward || lease.state != kStreamingBatchGetLeaseHeld ||
      lease.queue_id != identity.queue_id ||
      lease.owner_tag != identity.owner_tag || owners == nullptr ||
      owners[identity.queue_id] != identity.owner_tag) {
    (void)QuarantineKvafP8FillingStateFromAicore(
        record_gm, kKvafP8WaveFillingReleaseClaimed,
        kKvafP8WaveOutcomeFillingQuarantined, transition_result,
        simt_event, record);
    return false;
  }
  lease_result->queue_id = identity.queue_id;
  lease_result->owner_tag = identity.owner_tag;
  lease_result->success = 0U;
  lease_result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafStreamingBatchGetReleaseSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, identity.queue_id,
      identity.owner_tag, lease_result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (lease_result->success == 0U) {
    (void)QuarantineKvafP8FillingStateFromAicore(
        record_gm, kKvafP8WaveFillingReleaseClaimed,
        kKvafP8WaveOutcomeReleaseCasQuarantined, transition_result,
        simt_event, record);
    return false;
  }
  // Do not full-store the local record here: a terminal worker could have
  // changed RELEASE_CLAIMED to QUARANTINED after the owner table was cleared.
  // Only this exact CAS may make the record reusable.
  if (!TryTransitionKvafP8FillingStateFromAicore(
          record_gm, kKvafP8WaveFillingReleaseClaimed,
          kKvafP8WaveFree, transition_result, simt_event)) {
    return false;
  }
  record->state = kKvafP8WaveFree;
  record->reserved[1] = 0U;
  return true;
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

// FILLING -> POST_INTENT is the only supported P8 pre-doorbell transition.
// `ArmKvafP8WaveFillingForExternalWrites` must already have durably changed
// the no-work marker before the caller materializes descriptor scratch,
// private PENDING rows, or its request ledger.  This helper reloads the GM
// record and proves all of the following before it lets PostOnly see that
// work:
//   * the state is exactly FILLING;
//   * every identity word and the held queue lease still match;
//   * the owner table still contains the exact owner tag;
//   * the external-write marker was durably armed; and
//   * descriptor/private/ledger values agree on the same identity.
//
// Any failure after the external-write marker is visible is conservatively
// persisted as QUARANTINED.  It must not call a no-post release helper: a
// later terminal sweep owns cold recovery and the forward abort gate owns
// mathematical fallback.
__aicore__ inline bool PromoteKvafP8WaveFillingToPostIntent(
    __gm__ KvafP8WaveRecord* record_gm,
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
    __ubuf__ KvafP8WaveDrainClaimResult* transition_result,
    event_t simt_event, KvafP8WaveRecord* record) {
  if (record_gm == nullptr || transition_result == nullptr || record == nullptr ||
      !LoadKvafP8WaveRecordFromGm(record_gm, record)) {
    return false;
  }
  if (!KvafP8WaveFillingMatchesLeaseAndIdentity(
          control, forward_control, lease, identity, *record) ||
      record->reserved[1] != kKvafP8FillingExposureProof ||
      request_count == 0U || request_count > kDenseRequestCapacity) {
    if (IsKvafP8WaveRecordValid(*record) &&
        record->state == kKvafP8WaveFilling) {
      (void)QuarantineKvafP8FillingStateFromAicore(
          record_gm, kKvafP8WaveFilling,
          kKvafP8WaveOutcomeFillingQuarantined, transition_result,
          simt_event, record);
    }
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
          request_output_indices, request_ledger_capacity, request_count) ||
      !SnapshotKvafP8QueueIndices(args, &record->sq_producer_before,
                                   &record->cq_consumer_before)) {
    (void)QuarantineKvafP8FillingStateFromAicore(
        record_gm, kKvafP8WaveFilling,
        kKvafP8WaveOutcomeFillingQuarantined, transition_result,
        simt_event, record);
    return false;
  }
  record->request_count = request_count;
  if (!PublishKvafP8FillingPostIntentPayloadToGm(record_gm, *record)) {
    (void)QuarantineKvafP8FillingStateFromAicore(
        record_gm, kKvafP8WaveFilling,
        kKvafP8WaveOutcomeFillingQuarantined, transition_result,
        simt_event, record);
    return false;
  }
  // State publication is the final commit edge.  The payload and every
  // descriptor/private/ledger byte are already DDR-visible; a terminal
  // quarantine which races this CAS wins, and PostOnly is then forbidden.
  if (!TryTransitionKvafP8FillingStateFromAicore(
          record_gm, kKvafP8WaveFilling, kKvafP8WavePostIntent,
          transition_result, simt_event)) {
    return false;
  }
  record->reserved[1] = 0U;
  record->state = kKvafP8WavePostIntent;
  return true;
}

// Compatibility-only direct constructor for pre-P8 callers.  New P8 code
// must call BeginKvafP8WaveFilling -> Publish -> ArmForExternalWrites ->
// PromoteKvafP8WaveFillingToPostIntent so a lease is never invisible across
// a graph replay/exception boundary.  A record left in POST_INTENT after an
// interrupted launch is still ambiguous and must be quarantined.
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

// The deterministic no-post companion to ReleaseKvafP8WaveAfterPrivateReady.
// It is intentionally separate from the historical unposted helper below:
// this endpoint receives a record that has already been atomically claimed by
// a P8 dynamic drain worker (FINALIZED -> DRAIN_CLAIMED), and therefore makes
// the single-consumer ownership proof explicit.  Every PENDING private row
// must first have completed ReleaseKvafP8PendingPrivateRecordAfterNoPost;
// otherwise this helper quarantines the wave rather than returning its queue.
__aicore__ inline bool ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    const KvafP8ForwardControl& forward_control,
    __gm__ const KvafP8PrivateRecord* private_records,
    uint64_t private_record_count,
    __gm__ const int64_t* request_source_offsets,
    __gm__ const int64_t* request_object_offsets,
    __gm__ const int32_t* request_output_indices,
    uint32_t request_ledger_capacity,
    __ubuf__ KvafStreamingBatchGetLeaseResult* lease_result,
    event_t simt_event, KvafP8WaveRecord* record) {
  if (control_gm == nullptr || private_records == nullptr ||
      request_source_offsets == nullptr || request_object_offsets == nullptr ||
      request_output_indices == nullptr || lease_result == nullptr ||
      record == nullptr || !IsKvafP8WaveRecordValid(*record) ||
      record->state != kKvafP8WaveDrainClaimed ||
      record->post_committed != 0U ||
      record->outcome != kKvafP8WaveOutcomePostRejected ||
      record->transaction.finalized == 0U ||
      !KvafP8ForwardControlMatchesIdentity(
          forward_control, record->identity, kKvafP8ForwardPhaseDrain)) {
    if (record != nullptr && IsKvafP8WaveRecordValid(*record)) {
      QuarantineKvafP8WaveRecord(record,
                                  kKvafP8WaveOutcomeIdentityQuarantined);
    }
    return false;
  }
  // Load the actual control after the basic identity gate so a stale caller
  // cannot use a later runtime's queue table to free this record.  We cannot
  // reuse ValidateKvafP8DescriptorPrivateLedger here: that pre-doorbell
  // validator correctly requires PENDING rows, whereas every no-post row has
  // already passed its exact mapping-clear protocol and is now FREE.
  KvafStreamingBatchGetControl control{};
  if (!LoadKvafStreamingBatchGetControlFromGmForAicore(control_gm, &control) ||
      !IsKvafStreamingBatchGetControlValidForAicore(control) ||
      record->identity.queue_id >= control.queue_count ||
      private_record_count != control.output_rows ||
      record->request_count == 0U ||
      record->request_count > request_ledger_capacity ||
      record->request_count > kDenseRequestCapacity) {
    QuarantineKvafP8WaveRecord(record,
                                kKvafP8WaveOutcomeIdentityQuarantined);
    return false;
  }
  for (uint32_t request = 0U; request < record->request_count; ++request) {
    const int32_t output_index = request_output_indices[request];
    if (output_index < 0 ||
        static_cast<uint64_t>(output_index) >= private_record_count) {
      QuarantineKvafP8WaveRecord(record,
                                  kKvafP8WaveOutcomeIdentityQuarantined);
      return false;
    }
    KvafP8PrivateRecord private_record{};
    if (!LoadKvafP8PrivateRecordFromGm(
            private_records + static_cast<uint64_t>(output_index),
            &private_record) ||
        !KvafP8PrivateRecordMatchesIdentity(private_record,
                                             record->identity) ||
        private_record.output_index != output_index ||
        private_record.state != kKvafP8PrivateFree ||
        private_record.source_offset != static_cast<uint64_t>(
            request_source_offsets[request]) ||
        private_record.object_offset != static_cast<uint64_t>(
            request_object_offsets[request])) {
      QuarantineKvafP8WaveRecord(record,
                                  kKvafP8WaveOutcomeIdentityQuarantined);
      return false;
    }
    for (uint32_t prior = 0U; prior < request; ++prior) {
      if (request_output_indices[prior] == output_index) {
        QuarantineKvafP8WaveRecord(record,
                                    kKvafP8WaveOutcomeIdentityQuarantined);
        return false;
      }
    }
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
    if (record != nullptr && IsKvafP8WaveRecordValid(*record) &&
        (record->state == kKvafP8WaveFilling ||
         record->state == kKvafP8WaveFillingReleaseClaimed)) {
      QuarantineKvafP8WaveRecord(record,
                                  kKvafP8WaveOutcomeFillingQuarantined);
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
