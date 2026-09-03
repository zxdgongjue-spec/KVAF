#ifndef KVAF_MTP_LAYER_RESIDENT_GET_DEVICE_H_
#define KVAF_MTP_LAYER_RESIDENT_GET_DEVICE_H_

#include <climits>

// MTP forward-resident device API.
//
// This is intentionally a companion *executor* building block, not an
// include-time hook in the existing LightningIndexer/C8 kernel.  The executor
// consumes CandidateSeed-derived compact requests from a graph-visible MTP
// control, posts Direct-HCCP reads without a host/GIL submission, intermittently
// probes CQ while later work runs, and drains before the enclosing forward
// exposes its fixed resident rows to QSFA.
//
// The standalone MTP BatchGet kernel remains unchanged when compiled normally.
// This header only suppresses that standalone entrypoint while importing its
// verified WQE/validation primitives into a separate executor operator.  It
// preserves the deployed 112-byte ABI: miss_count + token_bytes +
// tokens_per_block, never a second experimental descriptor layout.
#ifndef KVAF_MTP_FORWARD_RESIDENT_DEVICE_EMBEDDED
#define KVAF_MTP_LAYER_RESIDENT_OWNS_EMBED_GUARD 1
#define KVAF_MTP_FORWARD_RESIDENT_DEVICE_EMBEDDED 1
#endif
#include "execute_batch_get_aiv.asc"
#ifdef KVAF_MTP_LAYER_RESIDENT_OWNS_EMBED_GUARD
#undef KVAF_MTP_FORWARD_RESIDENT_DEVICE_EMBEDDED
#undef KVAF_MTP_LAYER_RESIDENT_OWNS_EMBED_GUARD
#endif

#include "simt_api/cpp/kernel_simt_warp_level_intf.h"

namespace kvaf_mtp_layer_resident_get {

using ::KvafBatchGetCoreStatus;
using ::KvafBatchGetDirectUrmaEntity;
using ::KvafBatchGetQueueState;
using ::KvafExecuteBatchGetKernelArgs;
using ::KvafLayerResidentBatchGetControl;
using ::KvafLayerResidentBatchGetDescriptorScratch;

// Queue ownership is dynamically leased by a score/executor work item.  It is
// never a physical-AIV assignment and never owns a resident slot.  One lease
// admits one SQ/CQ wave only; the caller must obtain a CQ terminal before it
// can return the queue to the pool.
enum KvafMtpLayerResidentQueueLeaseState : uint32_t {
  kKvafMtpLayerResidentQueueIdle = 0U,
  kKvafMtpLayerResidentQueueHeld = 1U,
  kKvafMtpLayerResidentQueuePosted = 2U,
  kKvafMtpLayerResidentQueueTerminal = 3U,
};

struct KvafMtpLayerResidentQueueLease {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t state;
  uint32_t reserved0;
};

struct KvafMtpLayerResidentQueueLeaseResult {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t success;
  uint32_t reserved0;
};
static_assert(sizeof(KvafMtpLayerResidentQueueLeaseResult) == 16U,
              "MTP layer-resident queue lease ABI changed");

enum KvafMtpLayerResidentTransactionState : uint32_t {
  kKvafMtpLayerResidentTransactionIdle = 0U,
  kKvafMtpLayerResidentTransactionInFlight = 1U,
  kKvafMtpLayerResidentTransactionRetired = 2U,
  kKvafMtpLayerResidentTransactionFailed = 3U,
};

// This record remains AIV-local for one executor worker.  The MTP product
// contract does not permit a forward to return while a wave is unresolved, so
// no cross-forward/replay arena or ABA identity protocol is needed here.
struct KvafMtpLayerResidentTransaction {
  uint64_t active_count;
  uint64_t total_posts;
  uint64_t failed_request;
  uint64_t last_wqe;
  uint64_t result;
  uint32_t completion_polls;
  uint32_t state;
  uint32_t finalized;
  uint32_t reserved0;
};

struct KvafMtpLayerResidentBatchGetWave {
  KvafMtpLayerResidentQueueLease lease;
  KvafExecuteBatchGetKernelArgs queue_args;
  KvafMtpLayerResidentTransaction transaction;
};

struct KvafMtpLayerResidentBatchGetTerminal {
  uint32_t queue_id;
  uint32_t active_request_count;
  uint32_t success;
  uint32_t reserved0;
  uint64_t result;
};
static_assert(sizeof(KvafMtpLayerResidentBatchGetTerminal) == 24U,
              "MTP layer-resident terminal ABI changed");

__aicore__ inline void ResetKvafMtpLayerResidentTransaction(
    KvafMtpLayerResidentTransaction* transaction) {
  if (transaction == nullptr) {
    return;
  }
  transaction->active_count = 0U;
  transaction->total_posts = 0U;
  transaction->failed_request = UINT64_MAX;
  transaction->last_wqe = 0U;
  transaction->result = kSuccess;
  transaction->completion_polls = 0U;
  transaction->state = kKvafMtpLayerResidentTransactionIdle;
  transaction->finalized = 0U;
  transaction->reserved0 = 0U;
}

// Explicit scalar loads avoid compiler-generated GM/local struct copies in
// AscendC.  AICore and SIMT call paths remain separate so an AICore caller
// never invokes a SIMT-only atomic directly.
__aicore__ inline bool LoadKvafMtpLayerResidentControlForAicore(
    __gm__ const KvafLayerResidentBatchGetControl* source,
    KvafLayerResidentBatchGetControl* destination) {
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
  destination->descriptor_templates = source->descriptor_templates;
  destination->queue_count = source->queue_count;
  destination->output_rows = source->output_rows;
  destination->resident_row_base = source->resident_row_base;
  destination->resident_row_count = source->resident_row_count;
  destination->jettys_per_core = source->jettys_per_core;
  destination->token_bytes = source->token_bytes;
  destination->tokens_per_block = source->tokens_per_block;
  destination->descriptor_template_count = source->descriptor_template_count;
  return true;
}

__simt_callee__ inline bool LoadKvafMtpLayerResidentControlForSimt(
    __gm__ const KvafLayerResidentBatchGetControl* source,
    KvafLayerResidentBatchGetControl* destination) {
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
  destination->descriptor_templates = source->descriptor_templates;
  destination->queue_count = source->queue_count;
  destination->output_rows = source->output_rows;
  destination->resident_row_base = source->resident_row_base;
  destination->resident_row_count = source->resident_row_count;
  destination->jettys_per_core = source->jettys_per_core;
  destination->token_bytes = source->token_bytes;
  destination->tokens_per_block = source->tokens_per_block;
  destination->descriptor_template_count = source->descriptor_template_count;
  return true;
}

__aicore__ inline bool LoadKvafMtpLayerResidentDescriptorForAicore(
    __gm__ const KvafLayerResidentBatchGetDescriptorScratch* source,
    KvafLayerResidentBatchGetDescriptorScratch* destination) {
  if (source == nullptr || destination == nullptr) {
    return false;
  }
  destination->address_table = source->address_table;
  destination->valid_table = source->valid_table;
  destination->indices = source->indices;
  destination->offsets = source->offsets;
  destination->miss_count = source->miss_count;
  destination->output_indices = source->output_indices;
  destination->table_capacity = source->table_capacity;
  destination->request_capacity = source->request_capacity;
  destination->token_bytes = source->token_bytes;
  destination->tokens_per_block = source->tokens_per_block;
  return true;
}

__aicore__ inline bool IsKvafMtpLayerResidentControlValid(
    const KvafLayerResidentBatchGetControl& control) {
  return control.magic == KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_MAGIC &&
         control.version == KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_VERSION &&
         control.channel_entity != 0U && control.queue_state != 0U &&
         control.core_status != 0U && control.output != 0U &&
         control.queue_leases != 0U && control.descriptor_templates != 0U &&
         control.queue_count != 0U &&
         control.queue_count <= KVAF_AIV_BATCH_GET_MAX_AIV_CORES &&
         control.descriptor_template_count == control.queue_count &&
         control.output_rows != 0U && control.resident_row_count != 0U &&
         control.resident_row_base < control.output_rows &&
         control.resident_row_count <=
             control.output_rows - control.resident_row_base &&
         control.jettys_per_core == 1U && control.token_bytes != 0U &&
         control.tokens_per_block != 0U;
}

__simt_callee__ inline bool IsKvafMtpLayerResidentControlValidForSimt(
    const KvafLayerResidentBatchGetControl& control) {
  return control.magic == KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_MAGIC &&
         control.version == KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_VERSION &&
         control.queue_leases != 0U && control.queue_count != 0U &&
         control.queue_count <= KVAF_AIV_BATCH_GET_MAX_AIV_CORES &&
         control.jettys_per_core == 1U;
}

__aicore__ inline __gm__ const KvafLayerResidentBatchGetDescriptorScratch*
GetKvafMtpLayerResidentTemplate(
    const KvafLayerResidentBatchGetControl& control, uint32_t queue_id) {
  if (!IsKvafMtpLayerResidentControlValid(control) ||
      queue_id >= control.descriptor_template_count) {
    return nullptr;
  }
  return reinterpret_cast<
      __gm__ const KvafLayerResidentBatchGetDescriptorScratch*>(
      control.descriptor_templates) + queue_id;
}

// SIMT owns the atomic compare-and-swap.  `producer_id` is a stable task
// ordinal supplied by the executor, not a physical core ID.
__simt_callee__ inline bool TryAcquireKvafMtpLayerResidentQueueFromSimt(
    const KvafLayerResidentBatchGetControl& control, uint32_t producer_id,
    uint32_t start_hint, KvafMtpLayerResidentQueueLease* lease) {
  if (lease == nullptr ||
      !IsKvafMtpLayerResidentControlValidForSimt(control) ||
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
      lease->state = kKvafMtpLayerResidentQueueHeld;
      lease->reserved0 = 0U;
      return true;
    }
  }
  return false;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafMtpLayerResidentAcquireSimtService(
    __gm__ const KvafLayerResidentBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t start_hint,
    __ubuf__ KvafMtpLayerResidentQueueLeaseResult* result) {
  if (control_gm == nullptr || result == nullptr) {
    return;
  }
  KvafLayerResidentBatchGetControl control{};
  if (!LoadKvafMtpLayerResidentControlForSimt(control_gm, &control)) {
    return;
  }
  KvafMtpLayerResidentQueueLease lease{};
  if (TryAcquireKvafMtpLayerResidentQueueFromSimt(
          control, producer_id, start_hint, &lease)) {
    result->queue_id = lease.queue_id;
    result->owner_tag = lease.owner_tag;
    result->success = 1U;
    result->reserved0 = 0U;
  }
}

__aicore__ inline bool TryAcquireKvafMtpLayerResidentQueueFromAicore(
    __gm__ const KvafLayerResidentBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t start_hint,
    __ubuf__ KvafMtpLayerResidentQueueLeaseResult* result,
    event_t simt_event, KvafMtpLayerResidentQueueLease* lease) {
  if (control_gm == nullptr || result == nullptr || lease == nullptr) {
    return false;
  }
  result->queue_id = 0U;
  result->owner_tag = 0;
  result->success = 0U;
  result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafMtpLayerResidentAcquireSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, producer_id, start_hint,
      result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U || result->owner_tag <= 0) {
    return false;
  }
  lease->queue_id = result->queue_id;
  lease->owner_tag = result->owner_tag;
  lease->state = kKvafMtpLayerResidentQueueHeld;
  lease->reserved0 = 0U;
  return true;
}

__aicore__ inline bool LoadKvafMtpLayerResidentQueueTemplate(
    const KvafLayerResidentBatchGetControl& control,
    const KvafMtpLayerResidentQueueLease& lease,
    KvafLayerResidentBatchGetDescriptorScratch* descriptor) {
  if (descriptor == nullptr ||
      lease.state != kKvafMtpLayerResidentQueueHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count ||
      !IsKvafMtpLayerResidentControlValid(control)) {
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners[lease.queue_id] != lease.owner_tag ||
      !LoadKvafMtpLayerResidentDescriptorForAicore(
          GetKvafMtpLayerResidentTemplate(control, lease.queue_id),
          descriptor)) {
    return false;
  }
  return descriptor->address_table != 0U && descriptor->valid_table != 0U &&
         descriptor->indices != 0U && descriptor->offsets != 0U &&
         descriptor->miss_count != 0U && descriptor->output_indices != 0U &&
         descriptor->table_capacity != 0U && descriptor->request_capacity != 0U &&
         descriptor->token_bytes == control.token_bytes &&
         descriptor->tokens_per_block == control.tokens_per_block;
}

// CandidateCompact supplies unique, absolute canonical resident rows.  This
// helper only materializes one compact MTP request; it never changes Top-K or
// remaps a resident row.  `kv_object` is indexed in logical token units.
__aicore__ inline bool AppendKvafMtpLayerResidentRequest(
    const KvafLayerResidentBatchGetControl& control,
    KvafLayerResidentBatchGetDescriptorScratch* descriptor,
    uint32_t kv_object, int32_t output_row, uint32_t* active_request_count) {
  if (descriptor == nullptr || active_request_count == nullptr ||
      !IsKvafMtpLayerResidentControlValid(control) ||
      descriptor->tokens_per_block == 0U || descriptor->token_bytes == 0U ||
      descriptor->request_capacity == 0U ||
      *active_request_count >= descriptor->request_capacity || output_row < 0) {
    return false;
  }
  const uint64_t resident_end =
      control.resident_row_base + control.resident_row_count;
  if (static_cast<uint64_t>(output_row) < control.resident_row_base ||
      static_cast<uint64_t>(output_row) >= resident_end) {
    return false;
  }
  const uint32_t table_index = kv_object / descriptor->tokens_per_block;
  const uint64_t byte_offset =
      static_cast<uint64_t>(kv_object % descriptor->tokens_per_block) *
      descriptor->token_bytes;
  if (table_index >= descriptor->table_capacity ||
      byte_offset > static_cast<uint64_t>(INT32_MAX)) {
    return false;
  }
  __gm__ int32_t* const output_indices =
      reinterpret_cast<__gm__ int32_t*>(descriptor->output_indices);
  for (uint32_t prior = 0U; prior < *active_request_count; ++prior) {
    if (output_indices[prior] == output_row) {
      return false;
    }
  }
  const uint32_t request = *active_request_count;
  reinterpret_cast<__gm__ int32_t*>(descriptor->indices)[request] =
      static_cast<int32_t>(table_index);
  reinterpret_cast<__gm__ int32_t*>(descriptor->offsets)[request] =
      static_cast<int32_t>(byte_offset);
  output_indices[request] = output_row;
  *active_request_count = request + 1U;
  return true;
}

__aicore__ inline bool ValidateKvafMtpLayerResidentOutputIndices(
    const KvafLayerResidentBatchGetControl& control,
    const KvafLayerResidentBatchGetDescriptorScratch& descriptor,
    uint32_t active_request_count) {
  if (!IsKvafMtpLayerResidentControlValid(control) ||
      descriptor.output_indices == 0U || active_request_count == 0U ||
      active_request_count > descriptor.request_capacity) {
    return false;
  }
  const uint64_t resident_end =
      control.resident_row_base + control.resident_row_count;
  __gm__ const int32_t* const output_indices =
      reinterpret_cast<__gm__ const int32_t*>(descriptor.output_indices);
  for (uint32_t request = 0U; request < active_request_count; ++request) {
    const int32_t row = output_indices[request];
    if (row < 0 || static_cast<uint64_t>(row) < control.resident_row_base ||
        static_cast<uint64_t>(row) >= resident_end) {
      return false;
    }
    for (uint32_t prior = 0U; prior < request; ++prior) {
      if (output_indices[prior] == row) {
        return false;
      }
    }
  }
  return true;
}

// Writes the compact-prefix counter only after CandidateCompact has filled
// indices/offsets/output_indices.  The DDR barrier is the descriptor-to-WQE
// publish boundary: HCCP must not observe a half-written candidate wave.
__aicore__ inline bool PublishKvafMtpLayerResidentDescriptor(
    const KvafLayerResidentBatchGetControl& control,
    const KvafMtpLayerResidentQueueLease& lease,
    const KvafLayerResidentBatchGetDescriptorScratch& descriptor,
    uint32_t active_request_count) {
  if (lease.state != kKvafMtpLayerResidentQueueHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count || active_request_count == 0U ||
      active_request_count > descriptor.request_capacity ||
      !ValidateKvafMtpLayerResidentOutputIndices(
          control, descriptor, active_request_count)) {
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners[lease.queue_id] != lease.owner_tag) {
    return false;
  }
  KvafLayerResidentBatchGetDescriptorScratch template_record{};
  if (!LoadKvafMtpLayerResidentDescriptorForAicore(
          GetKvafMtpLayerResidentTemplate(control, lease.queue_id),
          &template_record) ||
      template_record.address_table != descriptor.address_table ||
      template_record.valid_table != descriptor.valid_table ||
      template_record.indices != descriptor.indices ||
      template_record.offsets != descriptor.offsets ||
      template_record.miss_count != descriptor.miss_count ||
      template_record.output_indices != descriptor.output_indices ||
      template_record.table_capacity != descriptor.table_capacity ||
      template_record.request_capacity != descriptor.request_capacity ||
      template_record.token_bytes != descriptor.token_bytes ||
      template_record.tokens_per_block != descriptor.tokens_per_block) {
    return false;
  }
  *reinterpret_cast<__gm__ int32_t*>(descriptor.miss_count) =
      static_cast<int32_t>(active_request_count);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

__aicore__ inline bool BuildKvafMtpLayerResidentQueueArgs(
    const KvafLayerResidentBatchGetControl& control, uint32_t queue_id,
    KvafExecuteBatchGetKernelArgs* args, uint32_t* active_request_count) {
  if (args == nullptr || active_request_count == nullptr ||
      !IsKvafMtpLayerResidentControlValid(control) ||
      queue_id >= control.queue_count ||
      control.output_rows > UINT32_MAX) {
    return false;
  }
  KvafLayerResidentBatchGetDescriptorScratch descriptor{};
  if (!LoadKvafMtpLayerResidentDescriptorForAicore(
          GetKvafMtpLayerResidentTemplate(control, queue_id), &descriptor) ||
      descriptor.miss_count == 0U || descriptor.token_bytes != control.token_bytes ||
      descriptor.tokens_per_block != control.tokens_per_block) {
    return false;
  }
  const int32_t count = *reinterpret_cast<__gm__ const int32_t*>(
      descriptor.miss_count);
  if (count <= 0 || static_cast<uint32_t>(count) > descriptor.request_capacity ||
      !ValidateKvafMtpLayerResidentOutputIndices(
          control, descriptor, static_cast<uint32_t>(count))) {
    return false;
  }
  *args = KvafExecuteBatchGetKernelArgs{};
  args->address_table = descriptor.address_table;
  args->valid_table = descriptor.valid_table;
  args->indices = descriptor.indices;
  args->offsets = descriptor.offsets;
  args->miss_count = descriptor.miss_count;
  args->output_indices = descriptor.output_indices;
  args->output = control.output;
  args->core_status =
      control.core_status + queue_id * sizeof(KvafBatchGetCoreStatus);
  args->channel_entity = control.channel_entity +
      queue_id * sizeof(KvafBatchGetDirectUrmaEntity);
  args->queue_state = control.queue_state +
      queue_id * sizeof(KvafBatchGetQueueState);
  args->table_capacity = descriptor.table_capacity;
  args->request_capacity = descriptor.request_capacity;
  args->output_rows = static_cast<uint32_t>(control.output_rows);
  args->token_bytes = descriptor.token_bytes;
  args->tokens_per_block = descriptor.tokens_per_block;
  args->block_count = 1U;
  args->jettys_per_core = 1U;
  *active_request_count = static_cast<uint32_t>(count);
  return true;
}

// Exactly the MTP standalone kernel's WQE/validation path through its last
// doorbell, with the CQ wait deliberately omitted.  Keep this only in the
// separate executor operator: RunBatchGetValidation owns the existing 160KiB
// BatchGet UB layout and must not be injected into C8's opaque hot loop.
__aicore__ inline void BeginKvafMtpLayerResidentBatchGetPostOnly(
    const KvafExecuteBatchGetKernelArgs& args, uint32_t active_request_count,
    KvafMtpLayerResidentTransaction* transaction) {
  if (transaction == nullptr) {
    return;
  }
  ResetKvafMtpLayerResidentTransaction(transaction);
  if (args.block_count != 1U || args.jettys_per_core != 1U ||
      args.address_table == 0U || args.valid_table == 0U ||
      args.indices == 0U || args.offsets == 0U || args.miss_count == 0U ||
      args.output_indices == 0U || args.output == 0U ||
      args.core_status == 0U || args.channel_entity == 0U ||
      args.queue_state == 0U || args.table_capacity == 0U ||
      args.request_capacity == 0U || active_request_count == 0U ||
      active_request_count > args.request_capacity || args.output_rows == 0U ||
      args.token_bytes == 0U || args.tokens_per_block == 0U) {
    transaction->result = kInvalidArgs;
    transaction->state = kKvafMtpLayerResidentTransactionFailed;
    transaction->finalized = 1U;
    return;
  }
  __gm__ KvafBatchGetCoreStatus* const status =
      reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
  __gm__ KvafBatchGetQueueState* const queue =
      reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
  __gm__ const KvafBatchGetDirectUrmaEntity* const entity =
      reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(
          args.channel_entity);
  transaction->active_count = active_request_count;
  transaction->total_posts =
      (active_request_count + kWqesPerPost - 1U) / kWqesPerPost;
  if (entity->magic != kEntityMagic || entity->version != kEntityVersion ||
      entity->local_hbm_address == 0U || entity->local_hbm_address != args.output ||
      entity->local_hbm_size < args.token_bytes || entity->local_token_id == 0U ||
      entity->remote_host_ddr_address == 0U ||
      entity->remote_host_ddr_size < args.token_bytes || entity->sq_va == 0U ||
      entity->sq_db_va == 0U || entity->sq_depth < 2U ||
      entity->wqe_size != kWqeBytes || entity->cq_va == 0U ||
      entity->cq_db_va == 0U || entity->cq_depth == 0U ||
      entity->cqe_size < kCqeProbeBytes) {
    transaction->result = kInvalidEntity;
  }
  if (transaction->result == kSuccess && queue->initialized != 0U &&
      queue->magic != kQueueMagic) {
    transaction->result = kInvalidQueue;
  }
  if (transaction->result == kSuccess && queue->initialized == 0U) {
    queue->magic = kQueueMagic;
    queue->initialized = 1U;
    queue->sq_producer_index = 0U;
    queue->cq_consumer_index = 0U;
    queue->completed_batch_count = 0U;
    queue->last_error = 0U;
  }
  __gm__ const uint64_t* const address_table =
      reinterpret_cast<__gm__ const uint64_t*>(args.address_table);
  __gm__ const uint8_t* const valid_table =
      reinterpret_cast<__gm__ const uint8_t*>(args.valid_table);
  __gm__ const int32_t* const indices =
      reinterpret_cast<__gm__ const int32_t*>(args.indices);
  __gm__ const int32_t* const offsets =
      reinterpret_cast<__gm__ const int32_t*>(args.offsets);
  __gm__ const int32_t* const output_indices =
      reinterpret_cast<__gm__ const int32_t*>(args.output_indices);
  if (transaction->result == kSuccess) {
    RunBatchGetValidation(
        address_table, valid_table, indices, offsets, output_indices, 0U,
        active_request_count, args.table_capacity, args.output_rows,
        args.token_bytes, args.tokens_per_block,
        entity->remote_host_ddr_address, entity->remote_host_ddr_size,
        entity->local_hbm_size, &transaction->result,
        &transaction->failed_request);
  }
  if (transaction->result == kSuccess &&
      ActiveRequestsForJetty(active_request_count, 0U, 1U) >= entity->sq_depth) {
    transaction->result = kQueueFull;
  }
  if (transaction->result == kSuccess) {
    uint64_t first_doorbell_cycle = 0U;
    ConfigureUbToGmCopy();
    PostDirectRequests(indices, output_indices, offsets, address_table, 0U,
                       active_request_count, args.token_bytes, 1U, entity,
                       queue, 0U, &first_doorbell_cycle,
                       &transaction->last_wqe);
  }
  transaction->state = transaction->result == kSuccess
                           ? kKvafMtpLayerResidentTransactionInFlight
                           : kKvafMtpLayerResidentTransactionFailed;
  if (transaction->state == kKvafMtpLayerResidentTransactionFailed) {
    WriteStatus(status, transaction->result, transaction->active_count,
                queue->completed_batch_count, queue->sq_producer_index,
                queue->cq_consumer_index, transaction->failed_request,
                transaction->last_wqe);
    transaction->finalized = 1U;
  }
}

// Nonblocking CQ probe.  A not-ready CQ has no side effect so the executor
// can run this between subsequent score-receipt batches without creating an
// artificial wait bubble.
__aicore__ inline void TryRetireKvafMtpLayerResidentBatchGetPostOnly(
    const KvafExecuteBatchGetKernelArgs& args,
    KvafMtpLayerResidentTransaction* transaction) {
  if (transaction == nullptr ||
      transaction->state != kKvafMtpLayerResidentTransactionInFlight) {
    return;
  }
  __gm__ KvafBatchGetQueueState* const queue =
      reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
  __gm__ const KvafBatchGetDirectUrmaEntity* const entity =
      reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(
          args.channel_entity);
  const uint32_t slot =
      static_cast<uint32_t>(queue->cq_consumer_index % entity->cq_depth);
  __gm__ uint8_t* const cqe_address =
      reinterpret_cast<__gm__ uint8_t*>(entity->cq_va) +
      static_cast<uint64_t>(slot) * entity->cqe_size;
  CleanCacheLines(cqe_address, kCqeProbeBytes);
  __asm__ __volatile__("");
  const uint32_t cqe_word = *reinterpret_cast<__gm__ uint32_t*>(cqe_address);
  __asm__ __volatile__("");
  ++transaction->completion_polls;
  const uint8_t owner = static_cast<uint8_t>((cqe_word >> 2U) & 1U);
  const uint8_t expected = static_cast<uint8_t>(
      (queue->cq_consumer_index / entity->cq_depth) & 1U);
  if ((owner ^ expected) == 0U) {
    if (transaction->completion_polls >= kMaxCqPolls) {
      transaction->result = kCqTimeout;
      transaction->state = kKvafMtpLayerResidentTransactionFailed;
    }
    return;
  }
  const uint8_t sub_status = static_cast<uint8_t>((cqe_word >> 16U) & 0xffU);
  const uint8_t status = static_cast<uint8_t>((cqe_word >> 24U) & 0xffU);
  ++queue->cq_consumer_index;
  st_dev(static_cast<uint32_t>(queue->cq_consumer_index) & kCqConsumerMask,
         reinterpret_cast<__gm__ uint32_t*>(entity->cq_db_va), 0);
  transaction->result = status == 0U && sub_status == 0U ? kSuccess : kCqError;
  if (transaction->result == kSuccess) {
    ++queue->completed_batch_count;
    transaction->state = kKvafMtpLayerResidentTransactionRetired;
  } else {
    transaction->state = kKvafMtpLayerResidentTransactionFailed;
  }
}

__aicore__ inline bool FinishKvafMtpLayerResidentBatchGet(
    KvafMtpLayerResidentBatchGetWave* wave,
    KvafMtpLayerResidentBatchGetTerminal* terminal) {
  if (wave == nullptr || terminal == nullptr ||
      (wave->lease.state != kKvafMtpLayerResidentQueuePosted &&
       wave->lease.state != kKvafMtpLayerResidentQueueTerminal)) {
    return false;
  }
  while (wave->transaction.state == kKvafMtpLayerResidentTransactionInFlight) {
    TryRetireKvafMtpLayerResidentBatchGetPostOnly(wave->queue_args,
                                                  &wave->transaction);
  }
  __gm__ KvafBatchGetCoreStatus* const status =
      reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(
          wave->queue_args.core_status);
  __gm__ KvafBatchGetQueueState* const queue =
      reinterpret_cast<__gm__ KvafBatchGetQueueState*>(
          wave->queue_args.queue_state);
  queue->last_error = wave->transaction.result;
  WriteStatus(status, wave->transaction.result, wave->transaction.active_count,
              queue->completed_batch_count, queue->sq_producer_index,
              queue->cq_consumer_index, wave->transaction.failed_request,
              wave->transaction.last_wqe);
  wave->transaction.finalized = 1U;
  wave->lease.state = kKvafMtpLayerResidentQueueTerminal;
  terminal->queue_id = wave->lease.queue_id;
  terminal->active_request_count =
      static_cast<uint32_t>(wave->transaction.active_count);
  terminal->success =
      wave->transaction.state == kKvafMtpLayerResidentTransactionRetired &&
              wave->transaction.result == kSuccess
          ? 1U
          : 0U;
  terminal->reserved0 = 0U;
  terminal->result = wave->transaction.result;
  return terminal->success != 0U;
}

__aicore__ inline bool BeginKvafMtpLayerResidentBatchGet(
    const KvafLayerResidentBatchGetControl& control,
    const KvafMtpLayerResidentQueueLease& lease,
    KvafMtpLayerResidentBatchGetWave* wave) {
  if (wave == nullptr) {
    return false;
  }
  // A failed pre-doorbell validation remains Held and must be returned through
  // ReturnKvafMtpLayerResidentUnpostedQueueFromAicore.  In particular, do not
  // pretend it is a CQ terminal: no WQE has been made visible to HCCP yet.
  wave->lease = lease;
  ResetKvafMtpLayerResidentTransaction(&wave->transaction);
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  uint32_t active_request_count = 0U;
  if (lease.state != kKvafMtpLayerResidentQueueHeld ||
      lease.owner_tag <= 0 || lease.queue_id >= control.queue_count ||
      owners[lease.queue_id] != lease.owner_tag ||
      !BuildKvafMtpLayerResidentQueueArgs(control, lease.queue_id,
                                           &wave->queue_args,
                                           &active_request_count)) {
    wave->transaction.result = kInvalidArgs;
    wave->transaction.state = kKvafMtpLayerResidentTransactionFailed;
    wave->transaction.finalized = 1U;
    return false;
  }
  BeginKvafMtpLayerResidentBatchGetPostOnly(
      wave->queue_args, active_request_count, &wave->transaction);
  if (wave->transaction.state != kKvafMtpLayerResidentTransactionInFlight) {
    // All errors above PostDirectRequests are deterministic pre-doorbell
    // validation failures.  Preserve Held so the caller uses the unposted
    // return path instead of racing a fabricated terminal state.
    return false;
  }
  wave->lease.state = kKvafMtpLayerResidentQueuePosted;
  return true;
}

__aicore__ inline void TryRetireKvafMtpLayerResidentBatchGet(
    KvafMtpLayerResidentBatchGetWave* wave) {
  if (wave == nullptr ||
      wave->lease.state != kKvafMtpLayerResidentQueuePosted) {
    return;
  }
  TryRetireKvafMtpLayerResidentBatchGetPostOnly(wave->queue_args,
                                                &wave->transaction);
  if (wave->transaction.state == kKvafMtpLayerResidentTransactionRetired ||
      wave->transaction.state == kKvafMtpLayerResidentTransactionFailed) {
    wave->lease.state = kKvafMtpLayerResidentQueueTerminal;
  }
}

// The two release paths are deliberately separate.  A Held lease has never
// been doorbelled; a Terminal lease has a CQ result and an initialized queue.
// Neither releases resident rows: their ownership lasts for the full forward.
__simt_callee__ inline bool ReturnKvafMtpLayerResidentUnpostedQueueFromSimt(
    const KvafLayerResidentBatchGetControl& control,
    KvafMtpLayerResidentQueueLease* lease) {
  if (lease == nullptr || lease->state != kKvafMtpLayerResidentQueueHeld ||
      lease->owner_tag <= 0 || lease->queue_id >= control.queue_count ||
      !IsKvafMtpLayerResidentControlValidForSimt(control)) {
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
  lease->state = kKvafMtpLayerResidentQueueIdle;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafMtpLayerResidentReturnUnpostedSimtService(
    __gm__ const KvafLayerResidentBatchGetControl* control_gm,
    uint32_t queue_id, int32_t owner_tag,
    __ubuf__ KvafMtpLayerResidentQueueLeaseResult* result) {
  if (control_gm == nullptr || result == nullptr) {
    return;
  }
  KvafLayerResidentBatchGetControl control{};
  if (!LoadKvafMtpLayerResidentControlForSimt(control_gm, &control)) {
    return;
  }
  KvafMtpLayerResidentQueueLease lease{
      queue_id, owner_tag, kKvafMtpLayerResidentQueueHeld, 0U};
  if (ReturnKvafMtpLayerResidentUnpostedQueueFromSimt(control, &lease)) {
    result->queue_id = queue_id;
    result->owner_tag = 0;
    result->success = 1U;
    result->reserved0 = 0U;
  }
}

__aicore__ inline bool ReturnKvafMtpLayerResidentUnpostedQueueFromAicore(
    __gm__ const KvafLayerResidentBatchGetControl* control_gm,
    __ubuf__ KvafMtpLayerResidentQueueLeaseResult* result, event_t simt_event,
    KvafMtpLayerResidentQueueLease* lease) {
  if (control_gm == nullptr || result == nullptr || lease == nullptr ||
      lease->state != kKvafMtpLayerResidentQueueHeld ||
      lease->owner_tag <= 0) {
    return false;
  }
  result->queue_id = lease->queue_id;
  result->owner_tag = lease->owner_tag;
  result->success = 0U;
  result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafMtpLayerResidentReturnUnpostedSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, lease->queue_id,
      lease->owner_tag, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U) {
    return false;
  }
  lease->owner_tag = 0;
  lease->state = kKvafMtpLayerResidentQueueIdle;
  return true;
}

__simt_callee__ inline bool ReturnKvafMtpLayerResidentQueueFromSimt(
    const KvafLayerResidentBatchGetControl& control,
    KvafMtpLayerResidentQueueLease* lease) {
  if (lease == nullptr || lease->state != kKvafMtpLayerResidentQueueTerminal ||
      lease->owner_tag <= 0 || lease->queue_id >= control.queue_count ||
      !IsKvafMtpLayerResidentControlValidForSimt(control)) {
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
  lease->state = kKvafMtpLayerResidentQueueIdle;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafMtpLayerResidentReturnSimtService(
    __gm__ const KvafLayerResidentBatchGetControl* control_gm,
    uint32_t queue_id, int32_t owner_tag,
    __ubuf__ KvafMtpLayerResidentQueueLeaseResult* result) {
  if (control_gm == nullptr || result == nullptr) {
    return;
  }
  KvafLayerResidentBatchGetControl control{};
  if (!LoadKvafMtpLayerResidentControlForSimt(control_gm, &control)) {
    return;
  }
  KvafMtpLayerResidentQueueLease lease{
      queue_id, owner_tag, kKvafMtpLayerResidentQueueTerminal, 0U};
  if (ReturnKvafMtpLayerResidentQueueFromSimt(control, &lease)) {
    result->queue_id = queue_id;
    result->owner_tag = 0;
    result->success = 1U;
    result->reserved0 = 0U;
  }
}

__aicore__ inline bool ReturnKvafMtpLayerResidentQueueFromAicore(
    __gm__ const KvafLayerResidentBatchGetControl* control_gm,
    __ubuf__ KvafMtpLayerResidentQueueLeaseResult* result, event_t simt_event,
    KvafMtpLayerResidentBatchGetWave* wave) {
  if (control_gm == nullptr || result == nullptr || wave == nullptr ||
      wave->lease.state != kKvafMtpLayerResidentQueueTerminal ||
      wave->lease.owner_tag <= 0 || wave->transaction.finalized == 0U) {
    return false;
  }
  result->queue_id = wave->lease.queue_id;
  result->owner_tag = wave->lease.owner_tag;
  result->success = 0U;
  result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafMtpLayerResidentReturnSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, wave->lease.queue_id,
      wave->lease.owner_tag, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U) {
    return false;
  }
  wave->lease.owner_tag = 0;
  wave->lease.state = kKvafMtpLayerResidentQueueIdle;
  return true;
}

}  // namespace kvaf_mtp_layer_resident_get

#endif  // KVAF_MTP_LAYER_RESIDENT_GET_DEVICE_H_
