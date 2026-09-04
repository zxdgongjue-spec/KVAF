#ifndef KVAF_FORWARD_RESIDENT_GET_DEVICE_H_
#define KVAF_FORWARD_RESIDENT_GET_DEVICE_H_

#include <climits>

// Forward-owned Direct-HCCP companion for a native streaming C8 pipeline.
//
// ExactScoreScan/CandidateCompact assigns each admitted object one fixed,
// absolute resident row in the enclosing forward arena.  A dynamic queue
// lease then protects only one SQ/CQ + descriptor record while the existing
// dense Direct-HCCP primitive performs the transfer:
//
//   claim queue -> fill compact descriptor -> publish -> PostOnly
//       -> scan more C8 tiles -> TryRetire -> Finish -> return queue
//
// There is deliberately no fixed Get-AIV partition, no private per-queue row
// partition, and no cross-forward/replay lifecycle.  A resident row
// is assigned once for this forward and is never released or remapped here.
// The enclosing fused operator drains every posted wave before it hands the
// final attention-slot map to QSFA.

// The standalone dense kernel exposes reusable helpers when this exact guard
// suppresses its global entrypoint.  Preserve a guard owned by an enclosing
// source-derived C8 target rather than unconditionally undefining it.
#ifndef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#define KVAF_FORWARD_RESIDENT_OWNS_DENSE_EMBED_GUARD 1
#define KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED 1
#endif
#include "execute_batch_get_aiv.asc"
#ifdef KVAF_FORWARD_RESIDENT_OWNS_DENSE_EMBED_GUARD
#undef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#undef KVAF_FORWARD_RESIDENT_OWNS_DENSE_EMBED_GUARD
#endif

#include "simt_api/cpp/kernel_simt_warp_level_intf.h"

namespace kvaf_forward_resident_get {

// Wire ABI types intentionally stay in the global namespace.  Only the
// verified dense Direct-HCCP implementation is namespaced.
using ::KvafBatchGetCoreStatus;
using ::KvafBatchGetDirectUrmaEntity;
using ::KvafBatchGetQueueState;
using ::KvafExecuteBatchGetKernelArgs;
using ::KvafForwardResidentBatchGetControl;
using ::KvafForwardResidentBatchGetDescriptorScratch;
using kvaf_fused_get::BeginKvafDenseBatchGetPostOnly;
using kvaf_fused_get::FinishKvafDenseBatchGet;
using kvaf_fused_get::kDenseBatchGetFailed;
using kvaf_fused_get::kDenseBatchGetIdle;
using kvaf_fused_get::kDenseBatchGetInFlight;
using kvaf_fused_get::kDenseBatchGetRetired;
using kvaf_fused_get::kInvalidArgs;
using kvaf_fused_get::kInvalidEntity;
using kvaf_fused_get::kInvalidLookup;
using kvaf_fused_get::kInvalidOutput;
using kvaf_fused_get::kInvalidQueue;
using kvaf_fused_get::kQueueFull;
using kvaf_fused_get::kSuccess;
using kvaf_fused_get::KvafDenseBatchGetTransaction;
using kvaf_fused_get::ResetDenseBatchGetTransaction;
using kvaf_fused_get::TryRetireKvafDenseBatchGet;

enum KvafForwardResidentQueueLeaseState : uint32_t {
  kForwardResidentQueueIdle = 0U,
  kForwardResidentQueueHeld = 1U,
  kForwardResidentQueuePosted = 2U,
  kForwardResidentQueueTerminal = 3U,
};

// This identifies a temporary transport owner only.  It does not identify a
// resident row and it does not survive one forward-owned fused invocation.
struct KvafForwardResidentQueueLease {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t state;
  uint32_t reserved0;
};

struct KvafForwardResidentQueueLeaseResult {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t success;
  uint32_t reserved0;
};
static_assert(sizeof(KvafForwardResidentQueueLeaseResult) == 16U,
              "forward-resident queue lease hand-off ABI changed");

struct KvafForwardResidentBatchGetWave {
  KvafForwardResidentQueueLease lease;
  KvafExecuteBatchGetKernelArgs queue_args;
  KvafDenseBatchGetTransaction transaction;
};

struct KvafForwardResidentBatchGetTerminal {
  uint32_t queue_id;
  uint32_t active_request_count;
  uint32_t success;
  uint32_t reserved0;
  uint64_t result;
};
static_assert(sizeof(KvafForwardResidentBatchGetTerminal) == 24U,
              "forward-resident terminal ABI changed");

// `BeginKvafDenseBatchGetPostOnly` mutates a large transaction record while it
// constructs WQEs and rings the doorbell.  A P10 companion must not use that
// record as an out-parameter across its larger score-executor closure: the
// AICore compiler can otherwise lose a nested writeback even though the HCCP
// side effects already happened.  The Begin boundary therefore returns this
// small scalar result only.  A successful caller rehydrates its drain record
// from the immutable control + the exact descriptor it just published.
enum KvafForwardResidentBeginStatus : uint32_t {
  kForwardResidentBeginPosted = 0U,
  kForwardResidentBeginInvalidArgs = 1U,
  kForwardResidentBeginInvalidEntity = 2U,
  kForwardResidentBeginInvalidQueue = 3U,
  kForwardResidentBeginInvalidLookup = 4U,
  kForwardResidentBeginQueueFull = 5U,
  kForwardResidentBeginInvalidOutput = 6U,
  // These cannot prove whether the doorbell was reached.  Callers must keep
  // the queue leased and fail the forward rather than recycle it.
  kForwardResidentBeginResultUnknown = 7U,
  kForwardResidentBeginStateIdle = 8U,
  kForwardResidentBeginStateRetired = 9U,
  kForwardResidentBeginStateFailedSuccess = 10U,
  kForwardResidentBeginStateUnknown = 11U,
  // A nonzero but nonexact producer-index advance proves that descriptor
  // publication may have reached HCCP.  It is deliberately not releasable.
  kForwardResidentBeginSqProducerDeltaMismatch = 12U,
};

// AscendC cannot use compiler-generated struct assignment across the GM/local
// address-space boundary.  Keep the AIC and SIMT loaders separate because the
// CANN MIX ABI requires a SIMT callee to call only SIMT callees.
__aicore__ inline bool LoadKvafForwardResidentBatchGetControlFromGmForAicore(
    __gm__ const KvafForwardResidentBatchGetControl* source,
    KvafForwardResidentBatchGetControl* destination) {
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
  destination->resident_row_base = source->resident_row_base;
  destination->resident_row_count = source->resident_row_count;
  destination->jettys_per_core = source->jettys_per_core;
  destination->block_bytes = source->block_bytes;
  destination->descriptor_templates = source->descriptor_templates;
  destination->descriptor_template_count = source->descriptor_template_count;
  return true;
}

__simt_callee__ inline bool LoadKvafForwardResidentBatchGetControlFromGmForSimt(
    __gm__ const KvafForwardResidentBatchGetControl* source,
    KvafForwardResidentBatchGetControl* destination) {
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
  destination->resident_row_base = source->resident_row_base;
  destination->resident_row_count = source->resident_row_count;
  destination->jettys_per_core = source->jettys_per_core;
  destination->block_bytes = source->block_bytes;
  destination->descriptor_templates = source->descriptor_templates;
  destination->descriptor_template_count = source->descriptor_template_count;
  return true;
}

__aicore__ inline bool LoadKvafForwardResidentDescriptorFromGm(
    __gm__ const KvafForwardResidentBatchGetDescriptorScratch* source,
    KvafForwardResidentBatchGetDescriptorScratch* destination) {
  if (source == nullptr || destination == nullptr) {
    return false;
  }
  // The dense SIMT WQE materializer consumes descriptor vectors with
  // `asc_ldcg`.  Keep the AICore reload on the same non-DCache path so a
  // reused descriptor record cannot mix one generation's header with another
  // generation's vector addresses/count.
  destination->address_table = ReadGmByPassDCache<uint64_t>(
      const_cast<__gm__ uint64_t*>(&source->address_table));
  destination->valid_table = ReadGmByPassDCache<uint64_t>(
      const_cast<__gm__ uint64_t*>(&source->valid_table));
  destination->indices = ReadGmByPassDCache<uint64_t>(
      const_cast<__gm__ uint64_t*>(&source->indices));
  destination->offsets = ReadGmByPassDCache<uint64_t>(
      const_cast<__gm__ uint64_t*>(&source->offsets));
  destination->mask = ReadGmByPassDCache<uint64_t>(
      const_cast<__gm__ uint64_t*>(&source->mask));
  destination->output_indices = ReadGmByPassDCache<uint64_t>(
      const_cast<__gm__ uint64_t*>(&source->output_indices));
  destination->table_capacity = ReadGmByPassDCache<uint32_t>(
      const_cast<__gm__ uint32_t*>(&source->table_capacity));
  destination->request_capacity = ReadGmByPassDCache<uint32_t>(
      const_cast<__gm__ uint32_t*>(&source->request_capacity));
  destination->active_request_count = ReadGmByPassDCache<uint32_t>(
      const_cast<__gm__ uint32_t*>(&source->active_request_count));
  destination->reserved0 = ReadGmByPassDCache<uint32_t>(
      const_cast<__gm__ uint32_t*>(&source->reserved0));
  return true;
}

__aicore__ inline bool StoreKvafForwardResidentDescriptorToGm(
    __gm__ KvafForwardResidentBatchGetDescriptorScratch* destination,
    const KvafForwardResidentBatchGetDescriptorScratch& source) {
  if (destination == nullptr) {
    return false;
  }
  // Publish immutable descriptor fields first.  `active_request_count` is
  // the release word for this scratch record and must never become visible
  // before its vector pointers and capacities are globally visible.
  WriteGmByPassDCache<uint64_t>(&destination->address_table,
                                source.address_table);
  WriteGmByPassDCache<uint64_t>(&destination->valid_table,
                                source.valid_table);
  WriteGmByPassDCache<uint64_t>(&destination->indices, source.indices);
  WriteGmByPassDCache<uint64_t>(&destination->offsets, source.offsets);
  WriteGmByPassDCache<uint64_t>(&destination->mask, source.mask);
  WriteGmByPassDCache<uint64_t>(&destination->output_indices,
                                source.output_indices);
  WriteGmByPassDCache<uint32_t>(&destination->table_capacity,
                                source.table_capacity);
  WriteGmByPassDCache<uint32_t>(&destination->request_capacity,
                                source.request_capacity);
  WriteGmByPassDCache<uint32_t>(&destination->reserved0, source.reserved0);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  WriteGmByPassDCache<uint32_t>(&destination->active_request_count,
                                source.active_request_count);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

// The resident arena is fixed at cold bind.  It need not be divisible by the
// number of HCCP queues: every queue is permitted to target any absolute row
// in this one arena, subject to CandidateCompact's unique-slot invariant.
__aicore__ inline bool IsKvafForwardResidentBatchGetControlValidForAicore(
    const KvafForwardResidentBatchGetControl& control) {
  return control.magic == KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_MAGIC &&
         control.version == KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_VERSION &&
         control.channel_entity != 0U && control.queue_state != 0U &&
         control.core_status != 0U && control.output != 0U &&
         control.queue_leases != 0U && control.descriptor_scratch != 0U &&
         control.descriptor_templates != 0U && control.queue_count != 0U &&
         control.queue_count <= KVAF_AIV_BATCH_GET_MAX_AIV_CORES &&
         control.descriptor_template_count == control.queue_count &&
         control.output_rows != 0U && control.resident_row_count != 0U &&
         control.resident_row_base < control.output_rows &&
         control.resident_row_count <=
             control.output_rows - control.resident_row_base &&
         control.jettys_per_core == 1U &&
         control.block_bytes == KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;
}

__simt_callee__ inline bool IsKvafForwardResidentBatchGetControlValidForLease(
    const KvafForwardResidentBatchGetControl& control) {
  return control.magic == KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_MAGIC &&
         control.version == KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_VERSION &&
         control.channel_entity != 0U && control.queue_state != 0U &&
         control.core_status != 0U && control.output != 0U &&
         control.queue_leases != 0U && control.descriptor_scratch != 0U &&
         control.descriptor_templates != 0U && control.queue_count != 0U &&
         control.queue_count <= KVAF_AIV_BATCH_GET_MAX_AIV_CORES &&
         control.descriptor_template_count == control.queue_count &&
         control.output_rows != 0U && control.resident_row_count != 0U &&
         control.resident_row_base < control.output_rows &&
         control.resident_row_count <=
             control.output_rows - control.resident_row_base &&
         control.jettys_per_core == 1U &&
         control.block_bytes == KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;
}

__aicore__ inline __gm__ KvafForwardResidentBatchGetDescriptorScratch*
GetKvafForwardResidentDescriptorScratch(
    const KvafForwardResidentBatchGetControl& control, uint32_t queue_id) {
  if (!IsKvafForwardResidentBatchGetControlValidForAicore(control) ||
      queue_id >= control.queue_count) {
    return nullptr;
  }
  return reinterpret_cast<__gm__ KvafForwardResidentBatchGetDescriptorScratch*>(
             control.descriptor_scratch) +
         queue_id;
}

__aicore__ inline __gm__ const KvafForwardResidentBatchGetDescriptorScratch*
GetKvafForwardResidentDescriptorTemplate(
    const KvafForwardResidentBatchGetControl& control, uint32_t queue_id) {
  if (!IsKvafForwardResidentBatchGetControlValidForAicore(control) ||
      queue_id >= control.descriptor_template_count) {
    return nullptr;
  }
  return reinterpret_cast<
      __gm__ const KvafForwardResidentBatchGetDescriptorScratch*>(
      control.descriptor_templates) + queue_id;
}

// Queue ownership is dynamic. `producer_id` is a stable executor work-item
// ordinal, never a physical AIV number. A false return means all bounded
// transport queues are busy; the caller defers the candidate to final recall
// rather than spinning or permanently reserving a Get core.
__simt_callee__ inline bool TryAcquireKvafForwardResidentQueueFromSimt(
    const KvafForwardResidentBatchGetControl& control, uint32_t producer_id,
    uint32_t start_hint, KvafForwardResidentQueueLease* lease) {
  if (lease == nullptr ||
      !IsKvafForwardResidentBatchGetControlValidForLease(control) ||
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
      AscendC::Simt::ThreadFence();
      lease->queue_id = queue_id;
      lease->owner_tag = owner_tag;
      lease->state = kForwardResidentQueueHeld;
      lease->reserved0 = 0U;
      return true;
    }
  }
  return false;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafForwardResidentAcquireSimtService(
    __gm__ const KvafForwardResidentBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t start_hint,
    __ubuf__ KvafForwardResidentQueueLeaseResult* result) {
  if (control_gm == nullptr || result == nullptr) {
    return;
  }
  KvafForwardResidentBatchGetControl control{};
  if (!LoadKvafForwardResidentBatchGetControlFromGmForSimt(control_gm,
                                                           &control)) {
    return;
  }
  KvafForwardResidentQueueLease lease{};
  if (TryAcquireKvafForwardResidentQueueFromSimt(control, producer_id,
                                                 start_hint, &lease)) {
    result->queue_id = lease.queue_id;
    result->owner_tag = lease.owner_tag;
    result->success = 1U;
    result->reserved0 = 0U;
  }
}

__aicore__ inline bool TryAcquireKvafForwardResidentQueueFromAicore(
    __gm__ const KvafForwardResidentBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t start_hint,
    __ubuf__ KvafForwardResidentQueueLeaseResult* result, event_t simt_event,
    KvafForwardResidentQueueLease* lease) {
  if (control_gm == nullptr || result == nullptr || lease == nullptr) {
    return false;
  }
  result->queue_id = 0U;
  result->owner_tag = 0;
  result->success = 0U;
  result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafForwardResidentAcquireSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, producer_id, start_hint,
      result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U || result->owner_tag <= 0) {
    return false;
  }
  lease->queue_id = result->queue_id;
  lease->owner_tag = result->owner_tag;
  lease->state = kForwardResidentQueueHeld;
  lease->reserved0 = 0U;
  return true;
}

__aicore__ inline bool LoadKvafForwardResidentQueueTemplate(
    const KvafForwardResidentBatchGetControl& control,
    const KvafForwardResidentQueueLease& lease,
    KvafForwardResidentBatchGetDescriptorScratch* descriptor) {
  if (descriptor == nullptr || lease.state != kForwardResidentQueueHeld ||
      lease.owner_tag <= 0 || lease.queue_id >= control.queue_count ||
      !IsKvafForwardResidentBatchGetControlValidForAicore(control)) {
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners[lease.queue_id] != lease.owner_tag) {
    return false;
  }
  __gm__ const KvafForwardResidentBatchGetDescriptorScratch* const source =
      GetKvafForwardResidentDescriptorTemplate(control, lease.queue_id);
  if (!LoadKvafForwardResidentDescriptorFromGm(source, descriptor)) {
    return false;
  }
  return descriptor->address_table != 0U && descriptor->valid_table != 0U &&
         descriptor->indices != 0U && descriptor->offsets != 0U &&
         descriptor->mask != 0U && descriptor->output_indices != 0U &&
         descriptor->table_capacity != 0U && descriptor->request_capacity != 0U &&
         descriptor->request_capacity <=
             KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY &&
         descriptor->active_request_count == 0U && descriptor->reserved0 == 0U;
}

// CandidateCompact owns cross-wave unique-slot admission.  The validation at
// the doorbell boundary verifies its local half: every output index is an
// absolute row of this one forward arena, and one dense wave cannot write the
// same row twice.
__aicore__ inline bool ValidateKvafForwardResidentOutputIndices(
    const KvafForwardResidentBatchGetControl& control,
    const KvafForwardResidentBatchGetDescriptorScratch& descriptor) {
  if (descriptor.output_indices == 0U || descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity ||
      descriptor.request_capacity > KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY) {
    return false;
  }
  __gm__ const int32_t* const output_indices =
      reinterpret_cast<__gm__ const int32_t*>(descriptor.output_indices);
  const uint64_t resident_end =
      control.resident_row_base + control.resident_row_count;
  for (uint32_t request = 0U; request < descriptor.active_request_count;
       ++request) {
    const int32_t row = ReadGmByPassDCache<int32_t>(
        const_cast<__gm__ int32_t*>(output_indices + request));
    if (row < 0 || static_cast<uint64_t>(row) < control.resident_row_base ||
        static_cast<uint64_t>(row) >= resident_end) {
      return false;
    }
    for (uint32_t earlier = 0U; earlier < request; ++earlier) {
      if (ReadGmByPassDCache<int32_t>(
              const_cast<__gm__ int32_t*>(output_indices + earlier)) == row) {
        return false;
      }
    }
  }
  return true;
}

// Append translates one already-admitted KV object into the dense Direct-HCCP
// vector ABI.  It does not allocate or reclaim a resident row; the enclosing
// Stage-B executor must have fixed the row before this method is called.
//
// The vectors are reused queue-local GM storage.  Before the first bypass
// write of a new dense wave, evict any old AICore DCache lines for the whole
// bounded vector range.  A later clean of an old dirty cache line would
// otherwise be able to overwrite a newer bypass-published descriptor entry.
__aicore__ inline void PrepareKvafForwardResidentDescriptorVectorsForPublication(
    const KvafForwardResidentBatchGetDescriptorScratch& descriptor) {
  if (descriptor.indices == 0U || descriptor.offsets == 0U ||
      descriptor.mask == 0U || descriptor.output_indices == 0U ||
      descriptor.request_capacity == 0U) {
    return;
  }
  const uint64_t count = descriptor.request_capacity;
  kvaf_fused_get::CleanCacheLines(
      reinterpret_cast<__gm__ uint8_t*>(descriptor.indices),
      count * sizeof(int32_t));
  kvaf_fused_get::CleanCacheLines(
      reinterpret_cast<__gm__ uint8_t*>(descriptor.offsets),
      count * sizeof(int32_t));
  kvaf_fused_get::CleanCacheLines(
      reinterpret_cast<__gm__ uint8_t*>(descriptor.mask), count);
  kvaf_fused_get::CleanCacheLines(
      reinterpret_cast<__gm__ uint8_t*>(descriptor.output_indices),
      count * sizeof(int32_t));
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

__aicore__ inline bool AppendKvafForwardResidentRequest(
    const KvafForwardResidentBatchGetControl& control,
    KvafForwardResidentBatchGetDescriptorScratch* descriptor,
    uint32_t kv_object, uint32_t rows_per_entry, int32_t output_row) {
  if (descriptor == nullptr || rows_per_entry == 0U || output_row < 0 ||
      !IsKvafForwardResidentBatchGetControlValidForAicore(control) ||
      descriptor->address_table == 0U || descriptor->valid_table == 0U ||
      descriptor->indices == 0U || descriptor->offsets == 0U ||
      descriptor->mask == 0U || descriptor->output_indices == 0U ||
      descriptor->table_capacity == 0U || descriptor->request_capacity == 0U ||
      descriptor->request_capacity > KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY ||
      descriptor->active_request_count >= descriptor->request_capacity ||
      descriptor->reserved0 != 0U) {
    return false;
  }
  const uint64_t resident_end =
      control.resident_row_base + control.resident_row_count;
  if (static_cast<uint64_t>(output_row) < control.resident_row_base ||
      static_cast<uint64_t>(output_row) >= resident_end) {
    return false;
  }
  const uint32_t table_index = kv_object / rows_per_entry;
  if (table_index >= descriptor->table_capacity) {
    return false;
  }
  const uint64_t byte_offset =
      static_cast<uint64_t>(kv_object % rows_per_entry) * control.block_bytes;
  if (byte_offset > static_cast<uint64_t>(INT32_MAX)) {
    return false;
  }
  const uint32_t request = descriptor->active_request_count;
  __gm__ int32_t* const indices =
      reinterpret_cast<__gm__ int32_t*>(descriptor->indices);
  __gm__ int32_t* const offsets =
      reinterpret_cast<__gm__ int32_t*>(descriptor->offsets);
  __gm__ uint8_t* const mask =
      reinterpret_cast<__gm__ uint8_t*>(descriptor->mask);
  __gm__ int32_t* const output_indices =
      reinterpret_cast<__gm__ int32_t*>(descriptor->output_indices);
  if (request == 0U) {
    PrepareKvafForwardResidentDescriptorVectorsForPublication(*descriptor);
  }
  for (uint32_t earlier = 0U; earlier < request; ++earlier) {
    if (ReadGmByPassDCache<int32_t>(output_indices + earlier) == output_row) {
      return false;
    }
  }
  // The dense transport's SIMT WQE materializer reads these exact vectors
  // with `asc_ldcg`, not normal GM indexing.  Ordinary AICore cached stores
  // can leave an earlier wave visible to that reader after descriptor reuse.
  WriteGmByPassDCache<int32_t>(indices + request,
                               static_cast<int32_t>(table_index));
  WriteGmByPassDCache<int32_t>(offsets + request,
                               static_cast<int32_t>(byte_offset));
  WriteGmByPassDCache<uint8_t>(mask + request, 1U);
  WriteGmByPassDCache<int32_t>(output_indices + request, output_row);
  descriptor->active_request_count = request + 1U;
  return true;
}

// `BeginKvafForwardResidentBatchGet` is the only caller-facing transition
// from a filled local descriptor to a possible HCCP doorbell.  Keep this
// helper separate only to make the DDR visibility boundary auditable: Begin
// calls it itself, so a caller cannot accidentally reuse an older queue
// scratch record by omitting a standalone publish step.
__aicore__ inline bool PublishKvafForwardResidentDescriptor(
    const KvafForwardResidentBatchGetControl& control,
    const KvafForwardResidentQueueLease& lease,
    const KvafForwardResidentBatchGetDescriptorScratch& descriptor) {
  if (!IsKvafForwardResidentBatchGetControlValidForAicore(control) ||
      lease.state != kForwardResidentQueueHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count || descriptor.address_table == 0U ||
      descriptor.valid_table == 0U || descriptor.indices == 0U ||
      descriptor.offsets == 0U || descriptor.mask == 0U ||
      descriptor.output_indices == 0U || descriptor.table_capacity == 0U ||
      descriptor.request_capacity == 0U ||
      descriptor.request_capacity > KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY ||
      descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity ||
      descriptor.reserved0 != 0U ||
      !ValidateKvafForwardResidentOutputIndices(control, descriptor)) {
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners[lease.queue_id] != lease.owner_tag) {
    return false;
  }
  __gm__ const KvafForwardResidentBatchGetDescriptorScratch* const template_record =
      GetKvafForwardResidentDescriptorTemplate(control, lease.queue_id);
  if (template_record == nullptr ||
      template_record->address_table != descriptor.address_table ||
      template_record->valid_table != descriptor.valid_table ||
      template_record->indices != descriptor.indices ||
      template_record->offsets != descriptor.offsets ||
      template_record->mask != descriptor.mask ||
      template_record->output_indices != descriptor.output_indices ||
      template_record->table_capacity != descriptor.table_capacity ||
      template_record->request_capacity != descriptor.request_capacity ||
      template_record->active_request_count != 0U ||
      template_record->reserved0 != 0U) {
    return false;
  }
  __gm__ KvafForwardResidentBatchGetDescriptorScratch* const scratch =
      GetKvafForwardResidentDescriptorScratch(control, lease.queue_id);
  if (!StoreKvafForwardResidentDescriptorToGm(scratch, descriptor)) {
    return false;
  }
  // Direct-HCCP may materialize WQEs immediately after Begin; make all
  // descriptor vectors/count/output indices visible before the doorbell.
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

__aicore__ inline bool BuildKvafForwardResidentQueueArgsFromDescriptor(
    const KvafForwardResidentBatchGetControl& control, uint32_t queue_id,
    const KvafForwardResidentBatchGetDescriptorScratch& descriptor,
    KvafExecuteBatchGetKernelArgs* args, uint32_t* active_request_count) {
  if (args == nullptr || active_request_count == nullptr ||
      !IsKvafForwardResidentBatchGetControlValidForAicore(control) ||
      queue_id >= control.queue_count || control.output_rows > UINT32_MAX ||
      descriptor.address_table == 0U || descriptor.valid_table == 0U ||
      descriptor.indices == 0U || descriptor.offsets == 0U ||
      descriptor.mask == 0U || descriptor.output_indices == 0U ||
      descriptor.table_capacity == 0U || descriptor.request_capacity == 0U ||
      descriptor.request_capacity > KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY ||
      descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity ||
      descriptor.reserved0 != 0U ||
      !ValidateKvafForwardResidentOutputIndices(control, descriptor)) {
    return false;
  }
  args->address_table = descriptor.address_table;
  args->valid_table = descriptor.valid_table;
  args->indices = descriptor.indices;
  args->offsets = descriptor.offsets;
  args->mask = descriptor.mask;
  args->output_indices = descriptor.output_indices;
  // Preserve the full LocalBufferAcquire base required by the unmodified
  // Direct-HCCP entity; descriptors select absolute resident rows within it.
  args->output = control.output;
  args->core_status =
      control.core_status + queue_id * sizeof(KvafBatchGetCoreStatus);
  args->channel_entity = control.channel_entity +
                         queue_id * sizeof(KvafBatchGetDirectUrmaEntity);
  args->queue_state =
      control.queue_state + queue_id * sizeof(KvafBatchGetQueueState);
  args->table_capacity = descriptor.table_capacity;
  args->request_capacity = descriptor.request_capacity;
  args->output_rows = static_cast<uint32_t>(control.output_rows);
  args->block_bytes = static_cast<uint32_t>(control.block_bytes);
  args->block_count = 1U;
  args->jettys_per_core = 1U;
  *active_request_count = descriptor.active_request_count;
  return true;
}

// P10's Begin/rehydrate pair already owns one exact local descriptor.  It
// calls the descriptor form above and never reloads a potentially unrelated
// GM record through a mutable wave out-parameter.  Keep this reload helper
// for callers that intentionally start from queue-owned descriptor scratch.
__aicore__ inline bool BuildKvafForwardResidentQueueArgs(
    const KvafForwardResidentBatchGetControl& control, uint32_t queue_id,
    KvafExecuteBatchGetKernelArgs* args, uint32_t* active_request_count) {
  if (!IsKvafForwardResidentBatchGetControlValidForAicore(control) ||
      queue_id >= control.queue_count) {
    return false;
  }
  __gm__ const KvafForwardResidentBatchGetDescriptorScratch* const scratch =
      GetKvafForwardResidentDescriptorScratch(control, queue_id);
  KvafForwardResidentBatchGetDescriptorScratch descriptor{};
  return LoadKvafForwardResidentDescriptorFromGm(scratch, &descriptor) &&
      BuildKvafForwardResidentQueueArgsFromDescriptor(
          control, queue_id, descriptor, args, active_request_count);
}

// These wrappers intentionally contain no alternative HCCP protocol.  They
// delegate to the unchanged dense PostOnly / nonblocking CQ probe / terminal
// Finish ABI once a dynamic queue and descriptor have been admitted.
// Read the durable SQ producer receipt outside the large PostOnly closure.
// Direct bypass loads plus a DDR DSB make this independent of the compiler's
// writeback of the local dense transaction record.
__aicore__ inline bool SnapshotKvafForwardResidentQueueProgress(
    const KvafExecuteBatchGetKernelArgs& args, uint32_t* initialized,
    uint64_t* sq_producer_index) {
  if (args.queue_state == 0U || initialized == nullptr ||
      sq_producer_index == nullptr) {
    return false;
  }
  __gm__ KvafBatchGetQueueState* const queue =
      reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  *initialized = ReadGmByPassDCache<uint32_t>(&queue->initialized);
  *sq_producer_index =
      ReadGmByPassDCache<uint64_t>(&queue->sq_producer_index);
  return true;
}

// Transaction scalar output may prove a validation failure, but may not prove
// that a post happened.  POSTED is returned only from an exact durable SQ PI
// delta in BeginKvafForwardResidentBatchGetPostOnly below.
__aicore__ inline uint32_t ClassifyKvafForwardResidentBeginWithoutPost(
    uint64_t result, uint32_t state) {
  switch (result) {
    case kInvalidArgs:
      return kForwardResidentBeginInvalidArgs;
    case kInvalidEntity:
      return kForwardResidentBeginInvalidEntity;
    case kInvalidQueue:
      return kForwardResidentBeginInvalidQueue;
    case kInvalidLookup:
      return kForwardResidentBeginInvalidLookup;
    case kQueueFull:
      return kForwardResidentBeginQueueFull;
    case kInvalidOutput:
      return kForwardResidentBeginInvalidOutput;
    default:
      break;
  }
  if (result != kSuccess) {
    return kForwardResidentBeginResultUnknown;
  }
  switch (state) {
    case kDenseBatchGetIdle:
      return kForwardResidentBeginStateIdle;
    case kDenseBatchGetRetired:
      return kForwardResidentBeginStateRetired;
    case kDenseBatchGetFailed:
      return kForwardResidentBeginStateFailedSuccess;
    default:
      return kForwardResidentBeginStateUnknown;
  }
}

// All returned non-POSTED statuses below arise before `PostDense...` is
// invoked.  Only these statuses may safely release a held queue lease.  An
// unknown scalar result is intentionally not considered proof of no-post.
__aicore__ inline bool KvafForwardResidentBeginHasNoPost(uint32_t status) {
  return status == kForwardResidentBeginInvalidArgs ||
      status == kForwardResidentBeginInvalidEntity ||
      status == kForwardResidentBeginInvalidQueue ||
      status == kForwardResidentBeginInvalidLookup ||
      status == kForwardResidentBeginQueueFull ||
      status == kForwardResidentBeginInvalidOutput;
}

// The sole HCCP PostOnly boundary.  It intentionally has no wave output
// parameter: the dense transaction is local to this closure and its outcome
// crosses the compiler/domain boundary as one scalar.  On POSTED, the caller
// must call `InitializeKvafForwardResidentPostedWaveForDrain` before it polls
// CQ.  That routine reloads the already-published queue scratch; it does not
// modify it.
__aicore__ inline uint32_t BeginKvafForwardResidentBatchGetPostOnly(
    const KvafForwardResidentBatchGetControl& control,
    const KvafForwardResidentQueueLease& lease,
    const KvafForwardResidentBatchGetDescriptorScratch& descriptor) {
  uint32_t active_request_count = 0U;
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (!IsKvafForwardResidentBatchGetControlValidForAicore(control) ||
      lease.state != kForwardResidentQueueHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count ||
      owners[lease.queue_id] != lease.owner_tag ||
      // Publish is deliberately inside Begin.  It stores this wave's exact
      // descriptor, executes the GM/DDR visibility barrier, and only then
      // lets the unchanged dense primitive materialize WQEs/doorbell.  A
      // previous valid scratch record can therefore never be posted merely
      // because a caller skipped a separate helper call.
      !PublishKvafForwardResidentDescriptor(control, lease, descriptor)) {
    return kForwardResidentBeginInvalidArgs;
  }

  // Construct the short-lived dense ABI record locally.  No caller-owned
  // wave receives an address-space write from the large PostOnly closure.
  KvafExecuteBatchGetKernelArgs args{};
  if (!BuildKvafForwardResidentQueueArgsFromDescriptor(
          control, lease.queue_id, descriptor, &args,
          &active_request_count)) {
    return kForwardResidentBeginInvalidArgs;
  }
  uint32_t sq_initialized_before = 0U;
  uint64_t sq_producer_before = 0U;
  if (!SnapshotKvafForwardResidentQueueProgress(
          args, &sq_initialized_before, &sq_producer_before)) {
    return kForwardResidentBeginInvalidArgs;
  }
  KvafDenseBatchGetTransaction transaction{};
  BeginKvafDenseBatchGetPostOnly(args, active_request_count, 0U,
                                 &transaction);
  uint32_t sq_initialized_after = 0U;
  uint64_t sq_producer_after = 0U;
  if (!SnapshotKvafForwardResidentQueueProgress(
          args, &sq_initialized_after, &sq_producer_after)) {
    return kForwardResidentBeginResultUnknown;
  }

  // First use initializes PI to zero, then advances it once per exact dense
  // request.  Unsigned subtraction intentionally remains correct across a
  // 64-bit producer-index wrap.  With the held queue lease, any nonzero
  // nonexact delta is ambiguous/partial commit and must retain the lease.
  if (sq_initialized_after != 0U) {
    const uint64_t sq_producer_baseline =
        sq_initialized_before != 0U ? sq_producer_before : 0U;
    const uint64_t sq_producer_delta =
        sq_producer_after - sq_producer_baseline;
    if (sq_producer_delta == static_cast<uint64_t>(active_request_count)) {
      return kForwardResidentBeginPosted;
    }
    if (sq_producer_delta != 0U) {
      return kForwardResidentBeginSqProducerDeltaMismatch;
    }
  }
  return ClassifyKvafForwardResidentBeginWithoutPost(transaction.result,
                                                     transaction.state);
}

// Rebuild the CQ-drain record from inputs that are immutable for the held
// queue.  This is called only after the scalar Begin result proves POSTED;
// it never constructs a descriptor, publishes GM state, or rings a doorbell.
// P10 does not enable standalone BatchGet profiling, so rehydration leaves
// those optional telemetry fields at their ResetDense... values instead of
// inventing timestamps that were not observed here.
__aicore__ inline bool InitializeKvafForwardResidentPostedWaveForDrain(
    const KvafForwardResidentBatchGetControl& control,
    const KvafForwardResidentQueueLease& lease,
    uint32_t expected_active_request_count,
    KvafForwardResidentBatchGetWave* wave) {
  if (wave == nullptr || expected_active_request_count == 0U ||
      !IsKvafForwardResidentBatchGetControlValidForAicore(control) ||
      lease.state != kForwardResidentQueueHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count) {
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners[lease.queue_id] != lease.owner_tag) {
    return false;
  }
  KvafExecuteBatchGetKernelArgs args{};
  uint32_t active_request_count = 0U;
  if (!BuildKvafForwardResidentQueueArgs(
          control, lease.queue_id, &args, &active_request_count) ||
      active_request_count != expected_active_request_count) {
    return false;
  }

  wave->lease.queue_id = lease.queue_id;
  wave->lease.owner_tag = lease.owner_tag;
  wave->lease.state = kForwardResidentQueuePosted;
  wave->lease.reserved0 = 0U;
  wave->queue_args.address_table = args.address_table;
  wave->queue_args.valid_table = args.valid_table;
  wave->queue_args.indices = args.indices;
  wave->queue_args.offsets = args.offsets;
  wave->queue_args.mask = args.mask;
  wave->queue_args.output_indices = args.output_indices;
  wave->queue_args.output = args.output;
  wave->queue_args.core_status = args.core_status;
  wave->queue_args.channel_entity = args.channel_entity;
  wave->queue_args.queue_state = args.queue_state;
  wave->queue_args.table_capacity = args.table_capacity;
  wave->queue_args.request_capacity = args.request_capacity;
  wave->queue_args.output_rows = args.output_rows;
  wave->queue_args.block_bytes = args.block_bytes;
  wave->queue_args.block_count = args.block_count;
  wave->queue_args.jettys_per_core = args.jettys_per_core;

  ResetDenseBatchGetTransaction(&wave->transaction);
  wave->transaction.active_count = active_request_count;
  wave->transaction.total_posts =
      (static_cast<uint64_t>(active_request_count) +
       kvaf_fused_get::kWqesPerPost - 1U) /
      kvaf_fused_get::kWqesPerPost;
  wave->transaction.result = kSuccess;
  wave->transaction.state = kDenseBatchGetInFlight;
  return true;
}

__aicore__ inline void TryRetireKvafForwardResidentBatchGet(
    KvafForwardResidentBatchGetWave* wave) {
  if (wave == nullptr || wave->lease.state != kForwardResidentQueuePosted) {
    return;
  }
  TryRetireKvafDenseBatchGet(wave->queue_args, &wave->transaction);
  if (wave->transaction.state == kDenseBatchGetRetired ||
      wave->transaction.state == kDenseBatchGetFailed) {
    wave->lease.state = kForwardResidentQueueTerminal;
  }
}

__aicore__ inline bool FinishKvafForwardResidentBatchGet(
    KvafForwardResidentBatchGetWave* wave,
    KvafForwardResidentBatchGetTerminal* terminal) {
  if (wave == nullptr || terminal == nullptr ||
      (wave->lease.state != kForwardResidentQueuePosted &&
       wave->lease.state != kForwardResidentQueueTerminal)) {
    return false;
  }
  FinishKvafDenseBatchGet(wave->queue_args, &wave->transaction);
  wave->lease.state = kForwardResidentQueueTerminal;
  terminal->queue_id = wave->lease.queue_id;
  terminal->active_request_count =
      static_cast<uint32_t>(wave->transaction.active_count);
  terminal->success =
      wave->transaction.state == kDenseBatchGetRetired &&
              wave->transaction.result == kSuccess
          ? 1U
          : 0U;
  terminal->reserved0 = 0U;
  terminal->result = wave->transaction.result;
  return terminal->success != 0U;
}

// A held lease can be returned only if descriptor publication/PostOnly never
// happened.  It is intentionally separate from terminal return: after a
// possible doorbell, only Finish may make a queue reusable.
__simt_callee__ inline bool ReturnKvafForwardResidentUnpostedQueueFromSimt(
    const KvafForwardResidentBatchGetControl& control,
    KvafForwardResidentQueueLease* lease) {
  if (lease == nullptr || lease->state != kForwardResidentQueueHeld ||
      lease->owner_tag <= 0 || lease->queue_id >= control.queue_count ||
      !IsKvafForwardResidentBatchGetControlValidForLease(control)) {
    return false;
  }
  __gm__ int32_t* const owners =
      reinterpret_cast<__gm__ int32_t*>(control.queue_leases);
  AscendC::Simt::ThreadFence();
  if (asc_atomic_cas(owners + lease->queue_id, lease->owner_tag, 0) !=
      lease->owner_tag) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  lease->owner_tag = 0;
  lease->state = kForwardResidentQueueIdle;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafForwardResidentReturnUnpostedSimtService(
    __gm__ const KvafForwardResidentBatchGetControl* control_gm,
    uint32_t queue_id, int32_t owner_tag,
    __ubuf__ KvafForwardResidentQueueLeaseResult* result) {
  if (control_gm == nullptr || result == nullptr) {
    return;
  }
  KvafForwardResidentBatchGetControl control{};
  if (!LoadKvafForwardResidentBatchGetControlFromGmForSimt(control_gm,
                                                           &control)) {
    return;
  }
  KvafForwardResidentQueueLease lease{queue_id, owner_tag,
                                       kForwardResidentQueueHeld, 0U};
  if (ReturnKvafForwardResidentUnpostedQueueFromSimt(control, &lease)) {
    result->queue_id = queue_id;
    result->owner_tag = 0;
    result->success = 1U;
    result->reserved0 = 0U;
  }
}

__aicore__ inline bool ReturnKvafForwardResidentUnpostedQueueFromAicore(
    __gm__ const KvafForwardResidentBatchGetControl* control_gm,
    __ubuf__ KvafForwardResidentQueueLeaseResult* result, event_t simt_event,
    KvafForwardResidentQueueLease* lease) {
  if (control_gm == nullptr || result == nullptr || lease == nullptr ||
      lease->state != kForwardResidentQueueHeld || lease->owner_tag <= 0) {
    return false;
  }
  result->queue_id = lease->queue_id;
  result->owner_tag = lease->owner_tag;
  result->success = 0U;
  result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafForwardResidentReturnUnpostedSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, lease->queue_id,
      lease->owner_tag, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U) {
    return false;
  }
  lease->owner_tag = 0;
  lease->state = kForwardResidentQueueIdle;
  return true;
}

// Returning a terminal queue never releases, clears, or remaps resident rows.
// The enclosing forward keeps them intact until QSFA has consumed its final
// attention-slot map.
__simt_callee__ inline bool ReturnKvafForwardResidentQueueFromSimt(
    const KvafForwardResidentBatchGetControl& control,
    KvafForwardResidentQueueLease* lease) {
  if (lease == nullptr || lease->state != kForwardResidentQueueTerminal ||
      lease->owner_tag <= 0 || lease->queue_id >= control.queue_count ||
      !IsKvafForwardResidentBatchGetControlValidForLease(control)) {
    return false;
  }
  __gm__ int32_t* const owners =
      reinterpret_cast<__gm__ int32_t*>(control.queue_leases);
  AscendC::Simt::ThreadFence();
  if (asc_atomic_cas(owners + lease->queue_id, lease->owner_tag, 0) !=
      lease->owner_tag) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  lease->owner_tag = 0;
  lease->state = kForwardResidentQueueIdle;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafForwardResidentReturnSimtService(
    __gm__ const KvafForwardResidentBatchGetControl* control_gm,
    uint32_t queue_id, int32_t owner_tag,
    __ubuf__ KvafForwardResidentQueueLeaseResult* result) {
  if (control_gm == nullptr || result == nullptr) {
    return;
  }
  KvafForwardResidentBatchGetControl control{};
  if (!LoadKvafForwardResidentBatchGetControlFromGmForSimt(control_gm,
                                                           &control)) {
    return;
  }
  KvafForwardResidentQueueLease lease{queue_id, owner_tag,
                                       kForwardResidentQueueTerminal, 0U};
  if (ReturnKvafForwardResidentQueueFromSimt(control, &lease)) {
    result->queue_id = queue_id;
    result->owner_tag = 0;
    result->success = 1U;
    result->reserved0 = 0U;
  }
}

__aicore__ inline bool ReturnKvafForwardResidentQueueFromAicore(
    __gm__ const KvafForwardResidentBatchGetControl* control_gm,
    __ubuf__ KvafForwardResidentQueueLeaseResult* result, event_t simt_event,
    KvafForwardResidentBatchGetWave* wave) {
  if (control_gm == nullptr || result == nullptr || wave == nullptr ||
      wave->lease.state != kForwardResidentQueueTerminal ||
      wave->lease.owner_tag <= 0 || wave->transaction.finalized == 0U) {
    return false;
  }
  result->queue_id = wave->lease.queue_id;
  result->owner_tag = wave->lease.owner_tag;
  result->success = 0U;
  result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafForwardResidentReturnSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, wave->lease.queue_id,
      wave->lease.owner_tag, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U) {
    return false;
  }
  wave->lease.owner_tag = 0;
  wave->lease.state = kForwardResidentQueueIdle;
  return true;
}

}  // namespace kvaf_forward_resident_get

#endif  // KVAF_FORWARD_RESIDENT_GET_DEVICE_H_
