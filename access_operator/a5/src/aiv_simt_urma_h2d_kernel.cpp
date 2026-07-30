#include "kvaf/access_operator/a5/aiv_simt_urma_h2d_launcher.h"

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "hcomm/hcomm.h"

namespace kvaf::access::a5 {
namespace {

using kvaf::access::DeviceBatchGetParams;
using kvaf::access::DeviceBatchState;
using kvaf::access::DeviceBatchStatus;
using kvaf::access::DeviceCompletion;
using kvaf::access::TransferState;
using kvaf::access::UrmaH2DDescriptor;

__simt_callee__ inline void SetDescriptorFailure(
    __gm__ UrmaH2DDescriptor* descriptor, DeviceBatchStatus status) {
  descriptor->state = static_cast<std::uint32_t>(TransferState::kFailed);
  descriptor->status = static_cast<std::uint32_t>(status);
}

__aicore__ inline void SetDescriptorFailureAiv(
    __gm__ UrmaH2DDescriptor* descriptor, DeviceBatchStatus status) {
  descriptor->state = static_cast<std::uint32_t>(TransferState::kFailed);
  descriptor->status = static_cast<std::uint32_t>(status);
}

__simt_vf__ __launch_bounds__(kAivUrmaSimtLanes) inline void
BuildH2DDescriptorsSimt(__gm__ const DeviceBatchGetParams* params) {
  const std::uint32_t request = threadIdx.x;
  if (request >= params->entry_count) {
    return;
  }

  auto* descriptors = reinterpret_cast<__gm__ UrmaH2DDescriptor*>(
      params->descriptors);
  __gm__ UrmaH2DDescriptor* descriptor = &descriptors[request];
  descriptor->abi_version = kAivUrmaAbiVersion;
  descriptor->state = static_cast<std::uint32_t>(TransferState::kEmpty);
  descriptor->logical_index = request;
  descriptor->flags = 0;
  descriptor->remote_addr = 0;
  descriptor->local_addr = 0;
  descriptor->bytes = 0;
  descriptor->generation = params->generation;
  descriptor->status =
      static_cast<std::uint32_t>(DeviceBatchStatus::kOk);
  descriptor->reserved0 = 0;
  descriptor->user_tag = request;

  if (params->mask != 0) {
    auto* mask = reinterpret_cast<__gm__ const std::uint8_t*>(params->mask);
    if (mask[request] == 0) {
      descriptor->state =
          static_cast<std::uint32_t>(TransferState::kSkipped);
      return;
    }
  }

  auto* indices =
      reinterpret_cast<__gm__ const std::uint32_t*>(params->indices);
  const std::uint32_t table_index = indices[request];
  if (table_index >= params->table.capacity) {
    SetDescriptorFailure(descriptor,
                         DeviceBatchStatus::kIndexOutOfRange);
    return;
  }

  if (params->table.valid != 0) {
    auto* valid =
        reinterpret_cast<__gm__ const std::uint8_t*>(params->table.valid);
    if (valid[table_index] == 0) {
      SetDescriptorFailure(
          descriptor, DeviceBatchStatus::kAccessTableEntryInvalid);
      return;
    }
  }

  auto* remote_addresses = reinterpret_cast<__gm__ const std::uint64_t*>(
      params->table.remote_addresses);
  auto* lengths = reinterpret_cast<__gm__ const std::uint64_t*>(
      params->table.lengths);
  auto* output_offsets = reinterpret_cast<__gm__ const std::uint64_t*>(
      params->output_offsets);
  const std::uint64_t remote = remote_addresses[table_index];
  const std::uint64_t bytes = lengths[table_index];
  const std::uint64_t output_offset = output_offsets[request];
  const std::uint64_t local_address = params->output_base + output_offset;

  if (remote == 0 || bytes == 0 || bytes > kAivUrmaMaxTransferBytes ||
      remote + bytes < remote) {
    SetDescriptorFailure(descriptor,
                         DeviceBatchStatus::kRemoteRangeInvalid);
    return;
  }
  if (output_offset > params->output_bytes ||
      bytes > params->output_bytes - output_offset ||
      local_address < params->output_base ||
      bytes > ~static_cast<std::uint64_t>(0) - local_address) {
    SetDescriptorFailure(descriptor,
                         DeviceBatchStatus::kOutputRangeInvalid);
    return;
  }

  descriptor->remote_addr = remote;
  descriptor->local_addr = local_address;
  descriptor->bytes = bytes;
  descriptor->state = static_cast<std::uint32_t>(TransferState::kReady);
}

__aicore__ inline void SetCompletionFailure(
    __gm__ DeviceCompletion* completion, DeviceBatchStatus status,
    std::uint32_t first_failed_index) {
  completion->state =
      static_cast<std::uint32_t>(DeviceBatchState::kFailed);
  completion->status = static_cast<std::uint32_t>(status);
  completion->first_failed_index = first_failed_index;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
KvafAivSimtUrmaH2DKernel(
    __gm__ const DeviceBatchGetParams* params,
    __gm__ DeviceCompletion* completion, std::uint64_t hcomm_channel) {
  completion->abi_version = kAivUrmaAbiVersion;
  completion->state =
      static_cast<std::uint32_t>(DeviceBatchState::kPlanning);
  completion->status = static_cast<std::uint32_t>(DeviceBatchStatus::kOk);
  completion->planned_count = 0;
  completion->submitted_count = 0;
  completion->completed_count = 0;
  completion->first_failed_index = kNoFailedIndex;
  completion->reserved0 = 0;
  completion->generation = 0;
  completion->reserved[0] = 0;
  completion->reserved[1] = 0;
  completion->reserved[2] = 0;

  if (params == nullptr) {
    SetCompletionFailure(completion, DeviceBatchStatus::kNullAddress, 0);
    return;
  }
  completion->generation = params->generation;
  if (params->table.abi_version != kAivUrmaAbiVersion) {
    SetCompletionFailure(completion, DeviceBatchStatus::kAbiMismatch, 0);
    return;
  }
  if (params->table.generation != params->generation) {
    SetCompletionFailure(
        completion, DeviceBatchStatus::kGenerationMismatch, 0);
    return;
  }
  if (params->entry_count > kAivUrmaMaxDescriptors ||
      params->entry_count > params->descriptor_capacity) {
    SetCompletionFailure(completion, DeviceBatchStatus::kCountOutOfRange, 0);
    return;
  }
  if (params->entry_count != 0 &&
      (params->table.remote_addresses == 0 || params->table.lengths == 0 ||
       params->indices == 0 || params->output_offsets == 0 ||
       params->descriptors == 0 || params->output_base == 0 ||
       hcomm_channel == 0)) {
    SetCompletionFailure(completion, DeviceBatchStatus::kNullAddress, 0);
    return;
  }

  asc_vf_call<BuildH2DDescriptorsSimt>(
      cce::dim3(kAivUrmaSimtLanes), params);
  AscendC::PipeBarrier<PIPE_ALL>();

  auto* descriptors = reinterpret_cast<__gm__ UrmaH2DDescriptor*>(
      params->descriptors);
  std::uint32_t ready_count = 0;
  for (std::uint32_t request = 0; request < params->entry_count; ++request) {
    const std::uint32_t state = descriptors[request].state;
    if (state == static_cast<std::uint32_t>(TransferState::kFailed)) {
      SetCompletionFailure(
          completion,
          static_cast<DeviceBatchStatus>(descriptors[request].status),
          request);
      return;
    }
    if (state == static_cast<std::uint32_t>(TransferState::kReady)) {
      ++ready_count;
    }
  }

  completion->planned_count = ready_count;
  completion->state =
      static_cast<std::uint32_t>(DeviceBatchState::kPlanned);
  if (ready_count == 0) {
    completion->state =
        static_cast<std::uint32_t>(DeviceBatchState::kComplete);
    return;
  }

  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECOUT> hcomm_buffer;
  pipe.InitBuffer(hcomm_buffer, AscendC::HCOMM_URMA_TMP_BUF_SIZE);
  auto local_buffer = hcomm_buffer.Get<std::uint8_t>();
  auto* local_buffer_address = reinterpret_cast<__ubuf__ std::uint8_t*>(
      local_buffer.GetPhyAddr());
  for (std::uint32_t offset = 0;
       offset < AscendC::HCOMM_URMA_TMP_BUF_SIZE; ++offset) {
    local_buffer_address[offset] = 0;
  }

  AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> hcomm;
  if (hcomm.Init(local_buffer_address,
                 AscendC::HCOMM_URMA_TMP_BUF_SIZE) != 0) {
    SetCompletionFailure(completion,
                         DeviceBatchStatus::kHcommInitFailed,
                         kNoFailedIndex);
    return;
  }

  completion->state =
      static_cast<std::uint32_t>(DeviceBatchState::kSubmitting);
  std::uint32_t enqueued_count = 0;
  std::uint32_t failed_request = kNoFailedIndex;
  for (std::uint32_t request = 0; request < params->entry_count; ++request) {
    __gm__ UrmaH2DDescriptor* descriptor = &descriptors[request];
    if (descriptor->state !=
        static_cast<std::uint32_t>(TransferState::kReady)) {
      continue;
    }
    const auto result = hcomm.ReadNbi<false>(
        static_cast<AscendC::ChannelHandle>(hcomm_channel),
        reinterpret_cast<GM_ADDR>(descriptor->local_addr),
        reinterpret_cast<GM_ADDR>(descriptor->remote_addr),
        descriptor->bytes);
    if (result != 0) {
      SetDescriptorFailureAiv(descriptor,
                              DeviceBatchStatus::kHcommReadFailed);
      failed_request = request;
      break;
    }
    descriptor->state =
        static_cast<std::uint32_t>(TransferState::kSubmitted);
    ++enqueued_count;
    ++completion->submitted_count;
  }

  if (enqueued_count != 0) {
    if (hcomm.Commit(
            static_cast<AscendC::ChannelHandle>(hcomm_channel)) != 0) {
      SetCompletionFailure(completion,
                           DeviceBatchStatus::kHcommCommitFailed,
                           kNoFailedIndex);
      return;
    }
    if (hcomm.Drain(
            static_cast<AscendC::ChannelHandle>(hcomm_channel)) != 0) {
      SetCompletionFailure(completion,
                           DeviceBatchStatus::kHcommDrainFailed,
                           kNoFailedIndex);
      return;
    }
  }

  for (std::uint32_t request = 0; request < params->entry_count; ++request) {
    if (descriptors[request].state ==
        static_cast<std::uint32_t>(TransferState::kSubmitted)) {
      descriptors[request].state =
          static_cast<std::uint32_t>(TransferState::kComplete);
      ++completion->completed_count;
    }
  }

  if (failed_request != kNoFailedIndex) {
    SetCompletionFailure(completion,
                         DeviceBatchStatus::kHcommReadFailed,
                         failed_request);
    return;
  }

  completion->state =
      static_cast<std::uint32_t>(DeviceBatchState::kComplete);
}

}  // namespace

void LaunchAivSimtUrmaH2D(
    void* stream, const DeviceBatchGetParams* device_params,
    DeviceCompletion* device_completion, std::uint64_t hcomm_channel) {
  KvafAivSimtUrmaH2DKernel<<<1, nullptr, stream>>>(
      device_params, device_completion, hcomm_channel);
}

}  // namespace kvaf::access::a5
