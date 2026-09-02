#ifndef KVAF_LAYER_RESIDENT_GET_DEVICE_H_
#define KVAF_LAYER_RESIDENT_GET_DEVICE_H_

// Device-side Direct-HCCP building block for the P9 product boundary:
//
//   ExactScoreScan / CandidateCompact
//       -> post directly into pre-reserved canonical resident rows
//       -> continue C8 scoring while CQ is in flight
//       -> Exact Top-K + Guaranteed Recall into the remaining rows
//       -> drain every post before this fused operator returns
//       -> QSFA reads only the final resident rows.
//
// This is intentionally *not* a P8 private-arena helper.  There are no
// forward/replay/epoch fields, no private-slot directory, no payload release,
// and no cross-layer lifetime.  The only dynamic ownership below is a bounded
// Direct-HCCP SQ/CQ queue lease inside one fused invocation.  A terminal queue
// is returned to the invocation-local pool after CQ, while the output remains
// in its fixed canonical resident slot for QSFA.

// execute_batch_get_aiv.asc contains both the reusable dense Direct-HCCP
// helpers and, unless this *exact* guard is present, a standalone global
// kernel entrypoint.  P9 imports only the helpers into a source-derived C8
// operator, so it must use the guard owned by that header.  Preserve a guard
// supplied by an enclosing target rather than unconditionally undefining it.
#ifndef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#define KVAF_LAYER_RESIDENT_OWNS_DENSE_EMBED_GUARD 1
#define KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED 1
#endif
#include "execute_batch_get_aiv.asc"
#ifdef KVAF_LAYER_RESIDENT_OWNS_DENSE_EMBED_GUARD
#undef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#undef KVAF_LAYER_RESIDENT_OWNS_DENSE_EMBED_GUARD
#endif

#include "simt_api/cpp/kernel_simt_warp_level_intf.h"

namespace kvaf_layer_resident_get {

using ::KvafLayerResidentBatchGetControl;
using ::KvafLayerResidentBatchGetDescriptorScratch;
// Keep the dependency on the existing dense Direct-HCCP primitive explicit:
// no global using-directive is allowed across the P9/C8 boundary.
using kvaf_fused_get::BeginKvafDenseBatchGetPostOnly;
using kvaf_fused_get::FinishKvafDenseBatchGet;
using kvaf_fused_get::kDenseBatchGetFailed;
using kvaf_fused_get::kDenseBatchGetInFlight;
using kvaf_fused_get::kDenseBatchGetRetired;
using kvaf_fused_get::kInvalidArgs;
using kvaf_fused_get::kSuccess;
using kvaf_fused_get::KvafBatchGetCoreStatus;
using kvaf_fused_get::KvafBatchGetDirectUrmaEntity;
using kvaf_fused_get::KvafBatchGetQueueState;
using kvaf_fused_get::KvafDenseBatchGetTransaction;
using kvaf_fused_get::KvafExecuteBatchGetKernelArgs;
using kvaf_fused_get::ResetDenseBatchGetTransaction;
using kvaf_fused_get::TryRetireKvafDenseBatchGet;

enum KvafLayerResidentQueueLeaseState : uint32_t {
  kLayerResidentQueueIdle = 0U,
  kLayerResidentQueueHeld = 1U,
  kLayerResidentQueuePosted = 2U,
  kLayerResidentQueueTerminal = 3U,
};

// This identifies only a temporary SQ/CQ owner.  It is never a resident-slot
// owner and is valid only until the enclosing fused operator has observed a
// terminal CQ for that queue.
struct KvafLayerResidentQueueLease {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t state;
  uint32_t reserved0;
};

struct KvafLayerResidentQueueLeaseResult {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t success;
  uint32_t reserved0;
};
static_assert(sizeof(KvafLayerResidentQueueLeaseResult) == 16U,
              "layer-resident queue lease hand-off ABI changed");

struct KvafLayerResidentBatchGetWave {
  KvafLayerResidentQueueLease lease;
  KvafExecuteBatchGetKernelArgs queue_args;
  KvafDenseBatchGetTransaction transaction;
};

struct KvafLayerResidentBatchGetTerminal {
  uint32_t queue_id;
  uint32_t active_request_count;
  uint32_t success;
  uint32_t reserved0;
  uint64_t result;
};
static_assert(sizeof(KvafLayerResidentBatchGetTerminal) == 24U,
              "layer-resident terminal ABI changed");

// Explicit scalar loads avoid an illegal compiler-generated GM/local struct
// copy in AscendC.  Keep this in the P9 header rather than reusing the P8
// streaming loader: the controls deliberately have different semantics.
__aicore__ inline bool LoadKvafLayerResidentBatchGetControlFromGmForAicore(
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

__simt_callee__ inline bool LoadKvafLayerResidentBatchGetControlFromGmForSimt(
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

__aicore__ inline bool LoadKvafLayerResidentDescriptorFromGm(
    __gm__ const KvafLayerResidentBatchGetDescriptorScratch* source,
    KvafLayerResidentBatchGetDescriptorScratch* destination) {
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
  destination->reserved0 = source->reserved0;
  return true;
}

__aicore__ inline bool StoreKvafLayerResidentDescriptorToGm(
    __gm__ KvafLayerResidentBatchGetDescriptorScratch* destination,
    const KvafLayerResidentBatchGetDescriptorScratch& source) {
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
  destination->reserved0 = source.reserved0;
  return true;
}

// The range is fixed when the layer control is cold-bound.  It must fit the
// full registered output, but it need not (and must not) be partitioned by
// transport queue: CandidateCompact owns the unique slot assignment.
__aicore__ inline bool IsKvafLayerResidentBatchGetControlValidForAicore(
    const KvafLayerResidentBatchGetControl& control) {
  return control.magic == KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_MAGIC &&
         control.version == KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_VERSION &&
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

__simt_callee__ inline bool IsKvafLayerResidentBatchGetControlValidForLease(
    const KvafLayerResidentBatchGetControl& control) {
  return control.magic == KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_MAGIC &&
         control.version == KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_VERSION &&
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

__aicore__ inline __gm__ KvafLayerResidentBatchGetDescriptorScratch*
GetKvafLayerResidentDescriptorScratch(
    const KvafLayerResidentBatchGetControl& control, uint32_t queue_id) {
  if (!IsKvafLayerResidentBatchGetControlValidForAicore(control) ||
      queue_id >= control.queue_count) {
    return nullptr;
  }
  return reinterpret_cast<__gm__ KvafLayerResidentBatchGetDescriptorScratch*>(
      control.descriptor_scratch) + queue_id;
}

__aicore__ inline __gm__ const KvafLayerResidentBatchGetDescriptorScratch*
GetKvafLayerResidentDescriptorTemplate(
    const KvafLayerResidentBatchGetControl& control, uint32_t queue_id) {
  if (!IsKvafLayerResidentBatchGetControlValidForAicore(control) ||
      queue_id >= control.descriptor_template_count) {
    return nullptr;
  }
  return reinterpret_cast<
      __gm__ const KvafLayerResidentBatchGetDescriptorScratch*>(
      control.descriptor_templates) + queue_id;
}

// Queue ownership is dynamic: producer_id is a stable LI task ordinal, not a
// physical-AIV allocation.  A false return means all bounded transport queues
// are busy; the caller keeps the candidate for later exact recall rather than
// spinning or reserving a dedicated Get core.
__simt_callee__ inline bool TryAcquireKvafLayerResidentQueueFromSimt(
    const KvafLayerResidentBatchGetControl& control, uint32_t producer_id,
    uint32_t start_hint, KvafLayerResidentQueueLease* lease) {
  if (lease == nullptr ||
      !IsKvafLayerResidentBatchGetControlValidForLease(control) ||
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
      lease->state = kLayerResidentQueueHeld;
      lease->reserved0 = 0U;
      return true;
    }
  }
  return false;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafLayerResidentAcquireSimtService(
    __gm__ const KvafLayerResidentBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t start_hint,
    __ubuf__ KvafLayerResidentQueueLeaseResult* result) {
  if (control_gm == nullptr || result == nullptr) {
    return;
  }
  KvafLayerResidentBatchGetControl control{};
  if (!LoadKvafLayerResidentBatchGetControlFromGmForSimt(control_gm,
                                                         &control)) {
    return;
  }
  KvafLayerResidentQueueLease lease{};
  if (TryAcquireKvafLayerResidentQueueFromSimt(control, producer_id,
                                               start_hint, &lease)) {
    result->queue_id = lease.queue_id;
    result->owner_tag = lease.owner_tag;
    result->success = 1U;
    result->reserved0 = 0U;
  }
}

__aicore__ inline bool TryAcquireKvafLayerResidentQueueFromAicore(
    __gm__ const KvafLayerResidentBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t start_hint,
    __ubuf__ KvafLayerResidentQueueLeaseResult* result, event_t simt_event,
    KvafLayerResidentQueueLease* lease) {
  if (control_gm == nullptr || result == nullptr || lease == nullptr) {
    return false;
  }
  result->queue_id = 0U;
  result->owner_tag = 0;
  result->success = 0U;
  result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafLayerResidentAcquireSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, producer_id, start_hint,
      result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U || result->owner_tag <= 0) {
    return false;
  }
  lease->queue_id = result->queue_id;
  lease->owner_tag = result->owner_tag;
  lease->state = kLayerResidentQueueHeld;
  lease->reserved0 = 0U;
  return true;
}

__aicore__ inline bool LoadKvafLayerResidentQueueTemplate(
    const KvafLayerResidentBatchGetControl& control,
    const KvafLayerResidentQueueLease& lease,
    KvafLayerResidentBatchGetDescriptorScratch* descriptor) {
  if (descriptor == nullptr || lease.state != kLayerResidentQueueHeld ||
      lease.owner_tag <= 0 || lease.queue_id >= control.queue_count ||
      !IsKvafLayerResidentBatchGetControlValidForAicore(control)) {
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners[lease.queue_id] != lease.owner_tag) {
    return false;
  }
  __gm__ const KvafLayerResidentBatchGetDescriptorScratch* const source =
      GetKvafLayerResidentDescriptorTemplate(control, lease.queue_id);
  if (!LoadKvafLayerResidentDescriptorFromGm(source, descriptor)) {
    return false;
  }
  return descriptor->address_table != 0U && descriptor->valid_table != 0U &&
         descriptor->indices != 0U && descriptor->offsets != 0U &&
         descriptor->mask != 0U && descriptor->output_indices != 0U &&
         descriptor->table_capacity != 0U && descriptor->request_capacity != 0U &&
         descriptor->request_capacity <= KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY &&
         descriptor->active_request_count == 0U && descriptor->reserved0 == 0U;
}

// CandidateCompact owns cross-wave uniqueness.  This function verifies the
// local half of that contract at the last safe point before HCCP sees the
// descriptor: all rows must be canonical resident rows and no one dense wave
// may issue two writes to the same row.
__aicore__ inline bool ValidateKvafLayerResidentOutputIndices(
    const KvafLayerResidentBatchGetControl& control,
    const KvafLayerResidentBatchGetDescriptorScratch& descriptor) {
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
    const int32_t row = output_indices[request];
    if (row < 0 || static_cast<uint64_t>(row) < control.resident_row_base ||
        static_cast<uint64_t>(row) >= resident_end) {
      return false;
    }
    for (uint32_t earlier = 0U; earlier < request; ++earlier) {
      if (output_indices[earlier] == row) {
        return false;
      }
    }
  }
  return true;
}

__aicore__ inline bool PublishKvafLayerResidentDescriptor(
    const KvafLayerResidentBatchGetControl& control,
    const KvafLayerResidentQueueLease& lease,
    const KvafLayerResidentBatchGetDescriptorScratch& descriptor) {
  if (!IsKvafLayerResidentBatchGetControlValidForAicore(control) ||
      lease.state != kLayerResidentQueueHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count || descriptor.address_table == 0U ||
      descriptor.valid_table == 0U || descriptor.indices == 0U ||
      descriptor.offsets == 0U || descriptor.mask == 0U ||
      descriptor.output_indices == 0U || descriptor.table_capacity == 0U ||
      descriptor.request_capacity == 0U ||
      descriptor.request_capacity > KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY ||
      descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity ||
      descriptor.reserved0 != 0U ||
      !ValidateKvafLayerResidentOutputIndices(control, descriptor)) {
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners[lease.queue_id] != lease.owner_tag) {
    return false;
  }
  __gm__ const KvafLayerResidentBatchGetDescriptorScratch* const template_record =
      GetKvafLayerResidentDescriptorTemplate(control, lease.queue_id);
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
  __gm__ KvafLayerResidentBatchGetDescriptorScratch* const scratch =
      GetKvafLayerResidentDescriptorScratch(control, lease.queue_id);
  if (!StoreKvafLayerResidentDescriptorToGm(scratch, descriptor)) {
    return false;
  }
  // The direct HCCP WQE can read descriptor arrays immediately after Begin.
  // Publish the fully written candidate vectors/count/rows before doorbell.
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

__aicore__ inline bool BuildKvafLayerResidentQueueArgs(
    const KvafLayerResidentBatchGetControl& control, uint32_t queue_id,
    KvafExecuteBatchGetKernelArgs* args, uint32_t* active_request_count) {
  if (args == nullptr || active_request_count == nullptr ||
      !IsKvafLayerResidentBatchGetControlValidForAicore(control) ||
      queue_id >= control.queue_count || control.output_rows > UINT32_MAX) {
    return false;
  }
  __gm__ const KvafLayerResidentBatchGetDescriptorScratch* const scratch =
      GetKvafLayerResidentDescriptorScratch(control, queue_id);
  KvafLayerResidentBatchGetDescriptorScratch descriptor{};
  if (!LoadKvafLayerResidentDescriptorFromGm(scratch, &descriptor) ||
      descriptor.address_table == 0U || descriptor.valid_table == 0U ||
      descriptor.indices == 0U || descriptor.offsets == 0U ||
      descriptor.mask == 0U || descriptor.output_indices == 0U ||
      descriptor.table_capacity == 0U || descriptor.request_capacity == 0U ||
      descriptor.request_capacity > KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY ||
      descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity ||
      descriptor.reserved0 != 0U ||
      !ValidateKvafLayerResidentOutputIndices(control, descriptor)) {
    return false;
  }
  args->address_table = descriptor.address_table;
  args->valid_table = descriptor.valid_table;
  args->indices = descriptor.indices;
  args->offsets = descriptor.offsets;
  args->mask = descriptor.mask;
  args->output_indices = descriptor.output_indices;
  // Preserve the complete LocalBufferAcquire base required by the existing
  // Direct-HCCP entity. The descriptor's absolute canonical row selects the
  // bounded resident interval inside that registered allocation.
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

// The three methods below are thin wrappers around the proven dense
// PostOnly/TryRetire/Finish primitive. They only change the destination-row
// contract; they do not create a second HCCP protocol.
__aicore__ inline bool BeginKvafLayerResidentBatchGet(
    const KvafLayerResidentBatchGetControl& control,
    const KvafLayerResidentQueueLease& lease,
    KvafLayerResidentBatchGetWave* wave) {
  if (wave == nullptr) {
    return false;
  }
  wave->lease = lease;
  ResetDenseBatchGetTransaction(&wave->transaction);
  uint32_t active_request_count = 0U;
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (lease.state != kLayerResidentQueueHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count ||
      owners[lease.queue_id] != lease.owner_tag ||
      !BuildKvafLayerResidentQueueArgs(control, lease.queue_id,
                                       &wave->queue_args,
                                       &active_request_count)) {
    wave->transaction.result = kInvalidArgs;
    wave->transaction.state = kDenseBatchGetFailed;
    wave->transaction.finalized = 1U;
    wave->lease.state = kLayerResidentQueueTerminal;
    return false;
  }
  BeginKvafDenseBatchGetPostOnly(wave->queue_args, active_request_count, 0U,
                                 &wave->transaction);
  wave->lease.state = wave->transaction.state == kDenseBatchGetInFlight
                          ? kLayerResidentQueuePosted
                          : kLayerResidentQueueTerminal;
  return wave->lease.state == kLayerResidentQueuePosted;
}

__aicore__ inline void TryRetireKvafLayerResidentBatchGet(
    KvafLayerResidentBatchGetWave* wave) {
  if (wave == nullptr || wave->lease.state != kLayerResidentQueuePosted) {
    return;
  }
  TryRetireKvafDenseBatchGet(wave->queue_args, &wave->transaction);
  if (wave->transaction.state == kDenseBatchGetRetired ||
      wave->transaction.state == kDenseBatchGetFailed) {
    wave->lease.state = kLayerResidentQueueTerminal;
  }
}

__aicore__ inline bool FinishKvafLayerResidentBatchGet(
    KvafLayerResidentBatchGetWave* wave,
    KvafLayerResidentBatchGetTerminal* terminal) {
  if (wave == nullptr || terminal == nullptr ||
      (wave->lease.state != kLayerResidentQueuePosted &&
       wave->lease.state != kLayerResidentQueueTerminal)) {
    return false;
  }
  FinishKvafDenseBatchGet(wave->queue_args, &wave->transaction);
  wave->lease.state = kLayerResidentQueueTerminal;
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

// This releases an SQ/CQ *queue* after its terminal CQ; it never releases,
// clears, or remaps a resident row.  The final fused operator calls it only
// after `Finish` and joins every wave before exposing attention_slots/QSFA.
__simt_callee__ inline bool ReturnKvafLayerResidentQueueFromSimt(
    const KvafLayerResidentBatchGetControl& control,
    KvafLayerResidentQueueLease* lease) {
  if (lease == nullptr || lease->state != kLayerResidentQueueTerminal ||
      lease->owner_tag <= 0 || lease->queue_id >= control.queue_count ||
      !IsKvafLayerResidentBatchGetControlValidForLease(control)) {
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
  lease->state = kLayerResidentQueueIdle;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafLayerResidentReturnSimtService(
    __gm__ const KvafLayerResidentBatchGetControl* control_gm,
    uint32_t queue_id, int32_t owner_tag,
    __ubuf__ KvafLayerResidentQueueLeaseResult* result) {
  if (control_gm == nullptr || result == nullptr) {
    return;
  }
  KvafLayerResidentBatchGetControl control{};
  if (!LoadKvafLayerResidentBatchGetControlFromGmForSimt(control_gm,
                                                         &control)) {
    return;
  }
  KvafLayerResidentQueueLease lease{queue_id, owner_tag,
                                     kLayerResidentQueueTerminal, 0U};
  if (ReturnKvafLayerResidentQueueFromSimt(control, &lease)) {
    result->queue_id = queue_id;
    result->owner_tag = 0;
    result->success = 1U;
    result->reserved0 = 0U;
  }
}

__aicore__ inline bool ReturnKvafLayerResidentQueueFromAicore(
    __gm__ const KvafLayerResidentBatchGetControl* control_gm,
    __ubuf__ KvafLayerResidentQueueLeaseResult* result, event_t simt_event,
    KvafLayerResidentBatchGetWave* wave) {
  if (control_gm == nullptr || result == nullptr || wave == nullptr ||
      wave->lease.state != kLayerResidentQueueTerminal ||
      wave->lease.owner_tag <= 0 || wave->transaction.finalized == 0U) {
    return false;
  }
  result->queue_id = wave->lease.queue_id;
  result->owner_tag = wave->lease.owner_tag;
  result->success = 0U;
  result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafLayerResidentReturnSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, wave->lease.queue_id,
      wave->lease.owner_tag, result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (result->success == 0U) {
    return false;
  }
  wave->lease.owner_tag = 0;
  wave->lease.state = kLayerResidentQueueIdle;
  return true;
}

}  // namespace kvaf_layer_resident_get

#endif  // KVAF_LAYER_RESIDENT_GET_DEVICE_H_
