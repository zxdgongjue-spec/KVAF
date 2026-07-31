#include "kvaf/access_operator/a5/aiv_simt_urma_h2d_launcher.h"

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace kvaf::access::a5 {
namespace {

using kvaf::access::AivDirectUrmaChannel;
using kvaf::access::DeviceBatchGetParams;
using kvaf::access::DeviceBatchState;
using kvaf::access::DeviceBatchStatus;
using kvaf::access::DeviceCompletion;
using kvaf::access::TransferState;
using kvaf::access::UrmaH2DDescriptor;

constexpr std::uint32_t kUrmaReadOpcode = 6U;
constexpr std::uint32_t kUrmaStrongOrder = 2U;
constexpr std::uint32_t kUrmaComplete = 1U << 5U;
constexpr std::uint32_t kUrmaDefaultFlags =
    kUrmaStrongOrder | kUrmaComplete;
constexpr std::uint32_t kUrmaWqeWords = 16U;
constexpr std::uint32_t kUrmaCqeWords = 16U;
constexpr std::uint32_t kUrmaQueueEntryBytes = 64U;
constexpr std::uint32_t kUrmaCqMaxRetries = 1000000U;
constexpr std::uint32_t kUrmaCqTimeout = 0xffffffffU;
constexpr std::uint32_t kUrmaScratchBytes = 512U;

__simt_callee__ inline void SetDescriptorFailure(
    __gm__ UrmaH2DDescriptor* descriptor, DeviceBatchStatus status) {
  descriptor->state = static_cast<std::uint32_t>(TransferState::kFailed);
  descriptor->status = static_cast<std::uint32_t>(status);
}

__aicore__ inline void SetDescriptorFailureAiv(
    __gm__ UrmaH2DDescriptor* descriptor, DeviceBatchStatus status,
    std::uint32_t detail = 0) {
  descriptor->state = static_cast<std::uint32_t>(TransferState::kFailed);
  descriptor->status = static_cast<std::uint32_t>(status);
  descriptor->reserved0 = detail;
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
    std::uint32_t first_failed_index, std::uint32_t detail = 0) {
  completion->state =
      static_cast<std::uint32_t>(DeviceBatchState::kFailed);
  completion->status = static_cast<std::uint32_t>(status);
  completion->first_failed_index = first_failed_index;
  completion->reserved0 = detail;
}

__aicore__ inline bool IsPowerOfTwo(std::uint32_t value) {
  return value != 0 && (value & (value - 1U)) == 0;
}

__aicore__ inline bool RangeContains(
    std::uint64_t base, std::uint64_t capacity,
    std::uint64_t address, std::uint64_t bytes) {
  return address >= base && bytes <= capacity &&
         address - base <= capacity - bytes;
}

__aicore__ inline void InvalidateGlobalCacheLine(
    std::uint64_t address) {
  const std::uint64_t aligned =
      address / AscendC::CACHE_LINE_SIZE *
      AscendC::CACHE_LINE_SIZE;
  AscendC::GlobalTensor<std::uint8_t> cache_line;
  cache_line.SetGlobalBuffer(
      reinterpret_cast<__gm__ std::uint8_t*>(aligned));
  __asm__ __volatile__("");
  AscendC::DataCacheCleanAndInvalid<
      std::uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
      AscendC::DcciDst::CACHELINE_OUT>(cache_line[0]);
  __asm__ __volatile__("");
}

__aicore__ inline bool DirectChannelIsValid(
    __gm__ const AivDirectUrmaChannel* channel) {
  return channel != nullptr &&
         channel->abi_version == kAivUrmaAbiVersion &&
         channel->sq_base != 0 && channel->sq_doorbell != 0 &&
         channel->cq_base != 0 && channel->cq_doorbell != 0 &&
         channel->sq_head_address != 0 &&
         channel->sq_tail_address != 0 &&
         channel->cq_tail_address != 0 &&
         channel->remote_base != 0 && channel->remote_bytes != 0 &&
         channel->local_base != 0 && channel->local_bytes != 0 &&
         channel->wqe_size == kUrmaQueueEntryBytes &&
         channel->cqe_size == kUrmaQueueEntryBytes &&
         IsPowerOfTwo(channel->sq_depth) &&
         IsPowerOfTwo(channel->cq_depth) &&
         channel->tp_id < (1U << 24U) &&
         channel->remote_token_id < (1U << 20U) &&
         (channel->remote_eid_low != 0 ||
          channel->remote_eid_high != 0) &&
         channel->sq_head == channel->cq_tail &&
         channel->wqe_count == channel->cq_tail;
}

__aicore__ inline void FillReadWqe(
    __ubuf__ std::uint32_t* words,
    __gm__ const AivDirectUrmaChannel* channel,
    std::uint32_t sq_head, std::uint64_t remote_address,
    std::uint64_t local_address, std::uint32_t bytes) {
  for (std::uint32_t index = 0; index < kUrmaWqeWords; ++index) {
    words[index] = 0;
  }

  const std::uint32_t owner =
      ((sq_head / channel->sq_depth) & 1U) == 0 ? 1U : 0U;
  const std::uint32_t token_enabled =
      (channel->flags & kAivUrmaChannelTokenEnabled) != 0 ? 1U : 0U;
  words[0] = (kUrmaDefaultFlags << 16U) |
             (token_enabled << 28U) |
             (1U << 29U) |  // remote object is a Jetty/segment id
             (owner << 31U);
  words[1] = kUrmaReadOpcode << 8U;
  words[2] = channel->tp_id | (1U << 24U);
  words[3] = channel->remote_token_id;
  words[4] = static_cast<std::uint32_t>(channel->remote_eid_low);
  words[5] =
      static_cast<std::uint32_t>(channel->remote_eid_low >> 32U);
  words[6] = static_cast<std::uint32_t>(channel->remote_eid_high);
  words[7] =
      static_cast<std::uint32_t>(channel->remote_eid_high >> 32U);
  words[8] =
      token_enabled != 0 ? channel->remote_token_value : 0U;
  words[10] = static_cast<std::uint32_t>(remote_address);
  words[11] = static_cast<std::uint32_t>(remote_address >> 32U);
  words[12] = bytes;
  // Match udma_fill_read_sqe(), SHMEM and UB_MicroBenchmark_new: the local
  // SGE token field remains zero and the registered local VA is carried in
  // words 14-15.
  words[13] = 0;
  words[14] = static_cast<std::uint32_t>(local_address);
  words[15] = static_cast<std::uint32_t>(local_address >> 32U);
}

__aicore__ inline bool PostDirectRead(
    __gm__ AivDirectUrmaChannel* channel,
    const AscendC::LocalTensor<std::uint32_t>& wqe_tensor,
    __ubuf__ std::uint32_t* wqe_words,
    std::uint64_t remote_address, std::uint64_t local_address,
    std::uint32_t bytes) {
  const std::uint32_t sq_head = channel->sq_head;
  const std::uint32_t wqe_count = channel->wqe_count;
  const std::uint32_t outstanding = wqe_count - channel->cq_tail;
  if (outstanding >= channel->sq_depth ||
      outstanding >= channel->cq_depth) {
    return false;
  }

  FillReadWqe(wqe_words, channel, sq_head, remote_address,
              local_address, bytes);
  const std::uint64_t sqe_address =
      channel->sq_base +
      static_cast<std::uint64_t>(channel->wqe_size) *
          (sq_head & (channel->sq_depth - 1U));
  AscendC::GlobalTensor<std::uint32_t> sqe_global;
  sqe_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ std::uint32_t*>(sqe_address));
  AscendC::SyncAction<AscendC::HardEvent::S_MTE3>();
  AscendC::DataCopy(sqe_global, wqe_tensor, kUrmaWqeWords);
  AscendC::SyncAction<AscendC::HardEvent::MTE3_S>();

  const std::uint32_t next_head = sq_head + 1U;
  const std::uint32_t next_wqe_count = wqe_count + 1U;
  const std::uint64_t packed_head =
      static_cast<std::uint64_t>(next_head) |
      (static_cast<std::uint64_t>(next_wqe_count) << 32U);
  st_dev(packed_head,
         reinterpret_cast<__gm__ std::uint64_t*>(
             channel->sq_head_address),
         0);
  channel->sq_head = next_head;
  channel->wqe_count = next_wqe_count;
  st_dev(next_head,
         reinterpret_cast<__gm__ std::uint32_t*>(
             channel->sq_doorbell),
         0);
  return true;
}

__aicore__ inline std::uint32_t PollDirectCq(
    __gm__ AivDirectUrmaChannel* channel,
    const AscendC::LocalTensor<std::uint32_t>& cqe_tensor,
    __ubuf__ std::uint32_t* cqe_words,
    std::uint32_t expected_tail) {
  std::uint32_t current_tail = channel->cq_tail;
  while (current_tail != expected_tail) {
    const std::uint64_t cqe_address =
        channel->cq_base +
        static_cast<std::uint64_t>(channel->cqe_size) *
            (current_tail & (channel->cq_depth - 1U));
    AscendC::GlobalTensor<std::uint32_t> cqe_global;
    cqe_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::uint32_t*>(cqe_address));

    const std::uint32_t expected_phase =
        (current_tail / channel->cq_depth) & 1U;
    bool valid = false;
    for (std::uint32_t retry = 0; retry < kUrmaCqMaxRetries;
         ++retry) {
      // The UDMA engine updates CQ memory outside the AIV cache. Match the
      // validated UB_MicroBenchmark path and invalidate the cache line before
      // each observation of the owner/phase bit.
      InvalidateGlobalCacheLine(cqe_address);
      AscendC::SyncAction<AscendC::HardEvent::S_MTE2>();
      AscendC::DataCopy(cqe_tensor, cqe_global, kUrmaCqeWords);
      AscendC::SyncAction<AscendC::HardEvent::MTE2_S>();
      const std::uint32_t owner = (cqe_words[0] >> 2U) & 1U;
      if ((expected_phase ^ owner) != 0) {
        valid = true;
        break;
      }
    }
    if (!valid) {
      return kUrmaCqTimeout;
    }

    const std::uint32_t substatus =
        (cqe_words[0] >> 16U) & 0xffU;
    const std::uint32_t status =
        (cqe_words[0] >> 24U) & 0xffU;
    if (status != 0 || substatus != 0) {
      return (status << 8U) | substatus;
    }
    ++current_tail;
  }

  channel->cq_tail = current_tail;
  st_dev(current_tail,
         reinterpret_cast<__gm__ std::uint32_t*>(
             channel->cq_tail_address),
         0);
  st_dev(current_tail & 0x003fffffU,
         reinterpret_cast<__gm__ std::uint32_t*>(
             channel->cq_doorbell),
         0);
  st_dev(current_tail,
         reinterpret_cast<__gm__ std::uint32_t*>(
             channel->sq_tail_address),
         0);
  return 0;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
KvafAivSimtUrmaH2DKernel(
    __gm__ const DeviceBatchGetParams* params,
    __gm__ DeviceCompletion* completion,
    __gm__ AivDirectUrmaChannel* direct_channel) {
  if (completion == nullptr) {
    return;
  }
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
       direct_channel == nullptr)) {
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
  if (!DirectChannelIsValid(direct_channel)) {
    SetCompletionFailure(
        completion, DeviceBatchStatus::kDirectChannelInvalid,
        kNoFailedIndex);
    return;
  }

  // Preflight every descriptor against the exact local and remote MRs before
  // ringing the first SQ doorbell. This preserves the planner's no-submit-on-
  // validation-error contract.
  for (std::uint32_t request = 0; request < params->entry_count; ++request) {
    __gm__ UrmaH2DDescriptor* descriptor = &descriptors[request];
    if (descriptor->state !=
        static_cast<std::uint32_t>(TransferState::kReady)) {
      continue;
    }
    if (!RangeContains(direct_channel->remote_base,
                       direct_channel->remote_bytes,
                       descriptor->remote_addr, descriptor->bytes)) {
      SetDescriptorFailureAiv(
          descriptor, DeviceBatchStatus::kDirectRemoteRangeInvalid);
      SetCompletionFailure(
          completion, DeviceBatchStatus::kDirectRemoteRangeInvalid,
          request);
      return;
    }
    if (!RangeContains(direct_channel->local_base,
                       direct_channel->local_bytes,
                       descriptor->local_addr, descriptor->bytes)) {
      SetDescriptorFailureAiv(
          descriptor, DeviceBatchStatus::kDirectLocalRangeInvalid);
      SetCompletionFailure(
          completion, DeviceBatchStatus::kDirectLocalRangeInvalid,
          request);
      return;
    }
  }

  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECOUT> direct_scratch;
  pipe.InitBuffer(direct_scratch, kUrmaScratchBytes);
  auto wqe_tensor = direct_scratch.Get<std::uint32_t>();
  auto cqe_tensor = wqe_tensor[kUrmaWqeWords];
  auto* wqe_words = reinterpret_cast<__ubuf__ std::uint32_t*>(
      wqe_tensor.GetPhyAddr());
  auto* cqe_words = reinterpret_cast<__ubuf__ std::uint32_t*>(
      cqe_tensor.GetPhyAddr());

  completion->state =
      static_cast<std::uint32_t>(DeviceBatchState::kSubmitting);
  for (std::uint32_t request = 0; request < params->entry_count; ++request) {
    __gm__ UrmaH2DDescriptor* descriptor = &descriptors[request];
    if (descriptor->state !=
        static_cast<std::uint32_t>(TransferState::kReady)) {
      continue;
    }

    if (!PostDirectRead(
            direct_channel, wqe_tensor, wqe_words,
            descriptor->remote_addr, descriptor->local_addr,
            static_cast<std::uint32_t>(descriptor->bytes))) {
      SetDescriptorFailureAiv(
          descriptor, DeviceBatchStatus::kDirectChannelInvalid);
      SetCompletionFailure(
          completion, DeviceBatchStatus::kDirectChannelInvalid, request);
      return;
    }
    descriptor->state =
        static_cast<std::uint32_t>(TransferState::kSubmitted);
    ++completion->submitted_count;

    const std::uint32_t cq_result = PollDirectCq(
        direct_channel, cqe_tensor, cqe_words,
        direct_channel->wqe_count);
    if (cq_result != 0) {
      const DeviceBatchStatus status =
          cq_result == kUrmaCqTimeout
              ? DeviceBatchStatus::kDirectCqTimeout
              : DeviceBatchStatus::kDirectCqError;
      SetDescriptorFailureAiv(descriptor, status, cq_result);
      SetCompletionFailure(completion, status, request, cq_result);
      return;
    }
    descriptor->state =
        static_cast<std::uint32_t>(TransferState::kComplete);
    ++completion->completed_count;
  }

  completion->state =
      static_cast<std::uint32_t>(DeviceBatchState::kComplete);
}

}  // namespace

void LaunchAivSimtUrmaH2D(
    void* stream, const DeviceBatchGetParams* device_params,
    DeviceCompletion* device_completion,
    AivDirectUrmaChannel* direct_channel) {
  KvafAivSimtUrmaH2DKernel<<<1, nullptr, stream>>>(
      device_params, device_completion, direct_channel);
}

}  // namespace kvaf::access::a5
