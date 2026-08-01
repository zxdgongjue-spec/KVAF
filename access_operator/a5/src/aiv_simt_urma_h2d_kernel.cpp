#include "kvaf/access_operator/a5/aiv_simt_urma_h2d_launcher.h"

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "kvaf_urma_provider_abi.h"

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
  auto* descriptors = reinterpret_cast<__gm__ UrmaH2DDescriptor*>(
      params->descriptors);
  // entry_count is a runtime workload dimension and may be much larger than
  // the fixed SIMT lane count.  Each lane therefore owns a strided subset;
  // one asc_vf_call still plans the complete batch.
  for (std::uint32_t request = threadIdx.x;
       request < params->entry_count;
       request += kAivUrmaSimtLanes) {
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
      auto* mask =
          reinterpret_cast<__gm__ const std::uint8_t*>(params->mask);
      if (mask[request] == 0) {
        descriptor->state =
            static_cast<std::uint32_t>(TransferState::kSkipped);
        continue;
      }
    }

    auto* indices =
        reinterpret_cast<__gm__ const std::uint32_t*>(params->indices);
    const std::uint32_t table_index = indices[request];
    if (table_index >= params->table.capacity) {
      SetDescriptorFailure(descriptor,
                           DeviceBatchStatus::kIndexOutOfRange);
      continue;
    }

    if (params->table.valid != 0) {
      auto* valid =
          reinterpret_cast<__gm__ const std::uint8_t*>(params->table.valid);
      if (valid[table_index] == 0) {
        SetDescriptorFailure(
            descriptor, DeviceBatchStatus::kAccessTableEntryInvalid);
        continue;
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
      continue;
    }
    if (output_offset > params->output_bytes ||
        bytes > params->output_bytes - output_offset ||
        local_address < params->output_base ||
        bytes > ~static_cast<std::uint64_t>(0) - local_address) {
      SetDescriptorFailure(descriptor,
                           DeviceBatchStatus::kOutputRangeInvalid);
      continue;
    }

    descriptor->remote_addr = remote;
    descriptor->local_addr = local_address;
    descriptor->bytes = bytes;
    descriptor->state = static_cast<std::uint32_t>(TransferState::kReady);
  }
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

__aicore__ inline void CleanProbeCacheLine(std::uint64_t address) {
  const std::uint64_t aligned =
      address / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE;
  AscendC::GlobalTensor<std::uint8_t> cache_line;
  cache_line.SetGlobalBuffer(
      reinterpret_cast<__gm__ std::uint8_t*>(aligned));
  AscendC::PipeBarrier<PIPE_ALL>();
  __asm__ __volatile__("");
  AscendC::DataCacheCleanAndInvalid<
      std::uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
      AscendC::DcciDst::CACHELINE_OUT>(cache_line[0]);
  __asm__ __volatile__("");
}

__aicore__ inline void PublishOwnerProbeAivResult(
    __gm__ AicpuOwnerProbeAivResult* result,
    AicpuOwnerProbeAivStatus status, std::uint64_t session_id,
    std::uint64_t observed_sequence, std::uint32_t observed_owner_state,
    std::uint32_t poll_count) {
  result->magic = kAicpuOwnerProbeAivResultMagic;
  result->abi_version = kAicpuOwnerProbeAivResultAbiVersion;
  result->struct_size = sizeof(AicpuOwnerProbeAivResult);
  result->session_id = session_id;
  result->observed_sequence = observed_sequence;
  result->observed_owner_state = observed_owner_state;
  result->poll_count = poll_count;
  result->reserved[0] = 0;
  result->reserved[1] = 0;
  result->reserved[2] = 0;
  // Status is the commit field for the Host-side result validation.
  result->status = static_cast<std::uint32_t>(status);
  CleanProbeCacheLine(reinterpret_cast<std::uint64_t>(result));
}

__aicore__ inline void InvalidateGlobalCacheRange(
    std::uint64_t address, std::uint64_t bytes) {
  for (std::uint64_t offset = 0; offset < bytes;
       offset += AscendC::CACHE_LINE_SIZE) {
    InvalidateGlobalCacheLine(address + offset);
  }
}

__aicore__ inline void PublishEpochOwnerAivResult(
    __gm__ AicpuEpochOwnerAivResult* result,
    AicpuEpochOwnerAivStatus status, std::uint64_t epoch,
    std::uint64_t session_id, std::uint32_t owner_state,
    std::uint32_t poll_count, std::uint64_t sequence,
    std::uint32_t sq_head = 0, std::uint32_t sq_tail = 0,
    std::uint32_t cq_tail = 0, std::uint32_t wqe_count = 0) {
  result->magic = kAicpuEpochOwnerAivResultMagic;
  result->abi_version = kAicpuEpochOwnerAivResultAbiVersion;
  result->struct_size = sizeof(AicpuEpochOwnerAivResult);
  result->epoch = epoch;
  result->session_id = session_id;
  result->observed_owner_state = owner_state;
  result->poll_count = poll_count;
  result->observed_sq_head = sq_head;
  result->observed_sq_tail = sq_tail;
  result->observed_cq_tail = cq_tail;
  result->observed_wqe_count = wqe_count;
  result->observed_sequence = sequence;
  result->status = static_cast<std::uint32_t>(status);
  CleanProbeCacheLine(reinterpret_cast<std::uint64_t>(result));
}

__aicore__ inline bool EpochOwnerStatusIdentityIsValid(
    __gm__ const kvaf_urma_epoch_owner_status_v1* owner_status,
    std::uint64_t epoch, std::uint64_t session_id) {
  return owner_status->magic == KVAF_URMA_EPOCH_OWNER_MAGIC &&
         owner_status->abi_version ==
             KVAF_URMA_EPOCH_OWNER_ABI_VERSION &&
         owner_status->struct_size == sizeof(*owner_status) &&
         owner_status->epoch == epoch &&
         owner_status->session_id == session_id;
}

__aicore__ inline bool EpochOwnerStateIsTerminal(std::uint32_t state) {
  return state == KVAF_URMA_EPOCH_OWNER_STATE_CLOSING ||
         state == KVAF_URMA_EPOCH_OWNER_STATE_CLOSE_RETRYING ||
         state == KVAF_URMA_EPOCH_OWNER_STATE_CLOSED ||
         state == KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
KvafAivEpochOwnerWaitReadyKernel(
    __gm__ const kvaf_urma_epoch_owner_status_v1* owner_status,
    __gm__ const kvaf_urma_provider_result_v2* open_result,
    __gm__ AicpuEpochOwnerAivResult* aiv_result,
    std::uint64_t epoch, std::uint64_t session_id,
    std::uint32_t max_polls) {
  if (owner_status == nullptr || open_result == nullptr ||
      aiv_result == nullptr || epoch == 0 || session_id == 0 ||
      max_polls == 0) {
    if (aiv_result != nullptr) {
      PublishEpochOwnerAivResult(
          aiv_result, AicpuEpochOwnerAivStatus::kInvalidOwnerStatus,
          epoch, session_id, 0, 0, 0);
    }
    return;
  }

  for (std::uint32_t poll = 0; poll < max_polls; ++poll) {
    const std::uint32_t observed_polls = poll + 1U;
    InvalidateGlobalCacheLine(
        reinterpret_cast<std::uint64_t>(owner_status));
    const std::uint32_t state = owner_status->state;
    if (state == KVAF_URMA_EPOCH_OWNER_STATE_EMPTY) {
      continue;
    }
    if (!EpochOwnerStatusIdentityIsValid(
            owner_status, epoch, session_id)) {
      PublishEpochOwnerAivResult(
          aiv_result, AicpuEpochOwnerAivStatus::kInvalidOwnerStatus,
          epoch, session_id, state, observed_polls,
          owner_status->sequence);
      return;
    }
    if (state == KVAF_URMA_EPOCH_OWNER_STATE_READY) {
      const std::uint32_t required_flags =
          KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
          KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE;
      InvalidateGlobalCacheRange(
          reinterpret_cast<std::uint64_t>(open_result),
          sizeof(*open_result));
      if (owner_status->result_code !=
              KVAF_URMA_EPOCH_OWNER_RESULT_OK ||
          (owner_status->flags & required_flags) != required_flags ||
          open_result->magic != KVAF_URMA_PROVIDER_MAGIC ||
          open_result->abi_version != KVAF_URMA_PROVIDER_ABI_VERSION ||
          open_result->struct_size != sizeof(*open_result) ||
          open_result->status != KVAF_URMA_PROVIDER_OK ||
          open_result->provider_status != 0 ||
          open_result->session_id != session_id) {
        PublishEpochOwnerAivResult(
            aiv_result, AicpuEpochOwnerAivStatus::kInvalidOwnerStatus,
            epoch, session_id, state, observed_polls,
            owner_status->sequence);
        return;
      }
      PublishEpochOwnerAivResult(
          aiv_result, AicpuEpochOwnerAivStatus::kReadyObserved,
          epoch, session_id, state, observed_polls,
          owner_status->sequence);
      return;
    }
    if (EpochOwnerStateIsTerminal(state) ||
        state == KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING) {
      PublishEpochOwnerAivResult(
          aiv_result, AicpuEpochOwnerAivStatus::kOwnerTerminal,
          epoch, session_id, state, observed_polls,
          owner_status->sequence);
      return;
    }
  }

  InvalidateGlobalCacheLine(
      reinterpret_cast<std::uint64_t>(owner_status));
  PublishEpochOwnerAivResult(
      aiv_result, AicpuEpochOwnerAivStatus::kTimedOut,
      epoch, session_id, owner_status->state, max_polls,
      owner_status->sequence);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
KvafAivEpochOwnerShutdownKernel(
    __gm__ kvaf_urma_epoch_owner_command_v1* command,
    __gm__ const kvaf_urma_epoch_owner_status_v1* owner_status,
    __gm__ AicpuEpochOwnerAivResult* aiv_result,
    std::uint64_t packed_sq_head_address,
    std::uint64_t sq_tail_address, std::uint64_t cq_tail_address,
    std::uint64_t epoch, std::uint64_t session_id,
    std::uint32_t max_polls) {
  if (command == nullptr || owner_status == nullptr ||
      aiv_result == nullptr || packed_sq_head_address == 0 ||
      sq_tail_address == 0 || cq_tail_address == 0 || epoch == 0 ||
      session_id == 0 || max_polls == 0 ||
      packed_sq_head_address % alignof(std::uint64_t) != 0 ||
      sq_tail_address % alignof(std::uint32_t) != 0 ||
      cq_tail_address % alignof(std::uint32_t) != 0) {
    if (aiv_result != nullptr) {
      PublishEpochOwnerAivResult(
          aiv_result, AicpuEpochOwnerAivStatus::kInvalidOwnerStatus,
          epoch, session_id, 0, 0, 0);
    }
    return;
  }

  for (std::uint32_t poll = 0; poll < max_polls; ++poll) {
    const std::uint32_t observed_polls = poll + 1U;
    InvalidateGlobalCacheLine(
        reinterpret_cast<std::uint64_t>(owner_status));
    const std::uint32_t state = owner_status->state;
    // The control stream is intentionally independent from the AICPU owner
    // stream, so it may run before the owner has replaced the Host-initialized
    // all-zero status line.  EMPTY exposes no resource and is the only state
    // for which an incomplete identity header is expected.  Once the owner
    // publishes OPENING (or any later state), the release publication must
    // make the complete epoch/session header visible and a mismatch is fatal.
    if (state == KVAF_URMA_EPOCH_OWNER_STATE_EMPTY) {
      continue;
    }
    if (!EpochOwnerStatusIdentityIsValid(
            owner_status, epoch, session_id)) {
      PublishEpochOwnerAivResult(
          aiv_result, AicpuEpochOwnerAivStatus::kInvalidOwnerStatus,
          epoch, session_id, state, observed_polls,
          owner_status->sequence);
      return;
    }
    const bool accepts_shutdown =
        state == KVAF_URMA_EPOCH_OWNER_STATE_READY ||
        state == KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING;
    if (accepts_shutdown) {
      const std::uint32_t required_flags =
          KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
          KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE;
      if ((owner_status->flags & required_flags) != required_flags) {
        PublishEpochOwnerAivResult(
            aiv_result, AicpuEpochOwnerAivStatus::kInvalidOwnerStatus,
            epoch, session_id, state, observed_polls,
            owner_status->sequence);
        return;
      }

      InvalidateGlobalCacheLine(packed_sq_head_address);
      InvalidateGlobalCacheLine(sq_tail_address);
      InvalidateGlobalCacheLine(cq_tail_address);
      const std::uint64_t packed_sq_head =
          *reinterpret_cast<__gm__ const std::uint64_t*>(
              packed_sq_head_address);
      const std::uint32_t sq_head =
          static_cast<std::uint32_t>(packed_sq_head);
      const std::uint32_t wqe_count =
          static_cast<std::uint32_t>(packed_sq_head >> 32U);
      const std::uint32_t sq_tail =
          *reinterpret_cast<__gm__ const std::uint32_t*>(sq_tail_address);
      const std::uint32_t cq_tail =
          *reinterpret_cast<__gm__ const std::uint32_t*>(cq_tail_address);
      if (sq_head != sq_tail || sq_tail != cq_tail ||
          cq_tail != wqe_count) {
        PublishEpochOwnerAivResult(
            aiv_result, AicpuEpochOwnerAivStatus::kQueueNotQuiesced,
            epoch, session_id, state, observed_polls,
            owner_status->sequence, sq_head, sq_tail, cq_tail,
            wqe_count);
        return;
      }

      command->magic = KVAF_URMA_EPOCH_OWNER_MAGIC;
      command->abi_version = KVAF_URMA_EPOCH_OWNER_ABI_VERSION;
      command->struct_size = sizeof(*command);
      command->epoch = epoch;
      command->session_id = session_id;
      command->sequence = KVAF_URMA_EPOCH_OWNER_SHUTDOWN_SEQUENCE;
      command->final_sq_head = sq_head;
      command->final_sq_tail = sq_tail;
      command->final_cq_tail = cq_tail;
      command->final_wqe_count = wqe_count;
      command->reserved[0] = 0;
      AscendC::PipeBarrier<PIPE_ALL>();
      command->opcode =
          KVAF_URMA_EPOCH_OWNER_COMMAND_SHUTDOWN_QUIESCED;
      CleanProbeCacheLine(reinterpret_cast<std::uint64_t>(command));
      PublishEpochOwnerAivResult(
          aiv_result, AicpuEpochOwnerAivStatus::kShutdownSubmitted,
          epoch, session_id, state, observed_polls,
          KVAF_URMA_EPOCH_OWNER_SHUTDOWN_SEQUENCE,
          sq_head, sq_tail, cq_tail,
          wqe_count);
      return;
    }
    if (EpochOwnerStateIsTerminal(state)) {
      PublishEpochOwnerAivResult(
          aiv_result, AicpuEpochOwnerAivStatus::kOwnerTerminal,
          epoch, session_id, state, observed_polls,
          owner_status->sequence);
      return;
    }
  }

  InvalidateGlobalCacheLine(
      reinterpret_cast<std::uint64_t>(owner_status));
  PublishEpochOwnerAivResult(
      aiv_result, AicpuEpochOwnerAivStatus::kTimedOut,
      epoch, session_id, owner_status->state, max_polls,
      owner_status->sequence);
}

// This kernel deliberately performs no URMA/EID/topology operation.  Its only
// purpose is to prove that a bounded AICPU invocation can coexist with an AIV
// task on another stream and that their two cache-line records are coherent.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
KvafAicpuOwnerConcurrencyProbeKernel(
    __gm__ kvaf_urma_owner_probe_command_v1* command,
    __gm__ const kvaf_urma_owner_probe_status_v1* owner_status,
    __gm__ AicpuOwnerProbeAivResult* result, std::uint64_t session_id,
    std::uint32_t max_ready_polls) {
  if (command == nullptr || owner_status == nullptr || result == nullptr ||
      session_id == 0 || max_ready_polls == 0) {
    if (result != nullptr) {
      PublishOwnerProbeAivResult(
          result, AicpuOwnerProbeAivStatus::kInvalidOwnerStatus,
          session_id, 0, KVAF_URMA_OWNER_PROBE_STATE_EMPTY, 0);
    }
    return;
  }

  for (std::uint32_t poll = 0; poll < max_ready_polls; ++poll) {
    const std::uint32_t observed_polls = poll + 1U;
    InvalidateGlobalCacheLine(
        reinterpret_cast<std::uint64_t>(owner_status));
    const std::uint32_t state = owner_status->state;
    if (state == KVAF_URMA_OWNER_PROBE_STATE_READY) {
      if (owner_status->magic != KVAF_URMA_OWNER_PROBE_MAGIC ||
          owner_status->abi_version !=
              KVAF_URMA_OWNER_PROBE_ABI_VERSION ||
          owner_status->struct_size != sizeof(*owner_status) ||
          owner_status->session_id != session_id ||
          owner_status->result_code != KVAF_URMA_OWNER_PROBE_RESULT_OK) {
        PublishOwnerProbeAivResult(
            result, AicpuOwnerProbeAivStatus::kInvalidOwnerStatus,
            session_id, owner_status->sequence, state, observed_polls);
        return;
      }

      command->magic = KVAF_URMA_OWNER_PROBE_MAGIC;
      command->abi_version = KVAF_URMA_OWNER_PROBE_ABI_VERSION;
      command->struct_size = sizeof(*command);
      command->session_id = session_id;
      command->sequence = KVAF_URMA_OWNER_PROBE_RELEASE_SEQUENCE;
      command->reserved[0] = 0;
      command->reserved[1] = 0;
      command->reserved[2] = 0;
      command->reserved[3] = 0;
      AscendC::PipeBarrier<PIPE_ALL>();
      // Publish the opcode last, then clean the complete producer-owned line.
      command->opcode = KVAF_URMA_OWNER_PROBE_COMMAND_RELEASE;
      CleanProbeCacheLine(reinterpret_cast<std::uint64_t>(command));
      PublishOwnerProbeAivResult(
          result, AicpuOwnerProbeAivStatus::kReleaseSubmitted,
          session_id, owner_status->sequence, state, observed_polls);
      return;
    }
    if (state == KVAF_URMA_OWNER_PROBE_STATE_RELEASED ||
        state == KVAF_URMA_OWNER_PROBE_STATE_TIMED_OUT ||
        state == KVAF_URMA_OWNER_PROBE_STATE_FAILED) {
      PublishOwnerProbeAivResult(
          result, AicpuOwnerProbeAivStatus::kOwnerTerminalBeforeReady,
          session_id, owner_status->sequence, state, observed_polls);
      return;
    }
  }

  InvalidateGlobalCacheLine(
      reinterpret_cast<std::uint64_t>(owner_status));
  PublishOwnerProbeAivResult(
      result, AicpuOwnerProbeAivStatus::kReadyTimedOut, session_id,
      owner_status->sequence, owner_status->state, max_ready_polls);
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
  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
  AscendC::DataCopy(sqe_global, wqe_tensor, kUrmaWqeWords);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

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
  std::uint32_t first_error = 0;
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
      AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
      AscendC::DataCopy(cqe_tensor, cqe_global, kUrmaCqeWords);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
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
    if ((status != 0 || substatus != 0) && first_error == 0) {
      // A CQE with an error is still a completed/consumable CQE.  Remember
      // the first error but continue draining the complete submitted wave so
      // teardown never inherits avoidable outstanding work.
      first_error = (status << 8U) | substatus;
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
  return first_error;
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
  // Queue counters are absolute uint32 values.  Never allow one batch to
  // cross the rollover boundary: all-or-nothing validation must happen before
  // the first WQE is visible.  A higher layer rebuilds the transport epoch
  // when this explicit rollover guard trips.
  constexpr std::uint32_t kMaxQueueCounter =
      ~static_cast<std::uint32_t>(0);
  if (ready_count > kMaxQueueCounter - direct_channel->sq_head ||
      ready_count > kMaxQueueCounter - direct_channel->wqe_count ||
      ready_count > kMaxQueueCounter - direct_channel->cq_tail) {
    SetCompletionFailure(
        completion, DeviceBatchStatus::kDirectCounterOverflow,
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
  const std::uint32_t wave_capacity =
      direct_channel->sq_depth < direct_channel->cq_depth
          ? direct_channel->sq_depth
          : direct_channel->cq_depth;
  std::uint32_t request_cursor = 0;
  while (completion->completed_count < ready_count) {
    const std::uint32_t wave_first = request_cursor;
    std::uint32_t wave_submitted = 0;
    std::uint32_t first_submitted = kNoFailedIndex;
    while (request_cursor < params->entry_count &&
           wave_submitted < wave_capacity) {
      __gm__ UrmaH2DDescriptor* descriptor =
          &descriptors[request_cursor];
      if (descriptor->state ==
          static_cast<std::uint32_t>(TransferState::kReady)) {
        if (!PostDirectRead(
                direct_channel, wqe_tensor, wqe_words,
                descriptor->remote_addr, descriptor->local_addr,
                static_cast<std::uint32_t>(descriptor->bytes))) {
          // The capacity calculation makes this path unreachable without an
          // externally corrupted queue state.  If earlier WQEs in this wave
          // were already posted, drain them before reporting the corruption.
          const std::uint32_t drain_result =
              wave_submitted == 0
                  ? 0
                  : PollDirectCq(
                        direct_channel, cqe_tensor, cqe_words,
                        direct_channel->wqe_count);
          const DeviceBatchStatus status =
              drain_result == kUrmaCqTimeout
                  ? DeviceBatchStatus::kDirectCqTimeout
                  : (drain_result != 0
                         ? DeviceBatchStatus::kDirectCqError
                         : DeviceBatchStatus::kDirectChannelInvalid);
          SetDescriptorFailureAiv(descriptor, status, drain_result);
          SetCompletionFailure(
              completion, status, request_cursor, drain_result);
          return;
        }
        if (first_submitted == kNoFailedIndex) {
          first_submitted = request_cursor;
        }
        descriptor->state =
            static_cast<std::uint32_t>(TransferState::kSubmitted);
        ++completion->submitted_count;
        ++wave_submitted;
      }
      ++request_cursor;
    }

    if (wave_submitted == 0) {
      SetCompletionFailure(
          completion, DeviceBatchStatus::kDirectChannelInvalid,
          kNoFailedIndex);
      return;
    }
    ++completion->reserved[0];
    const std::uint32_t cq_result = PollDirectCq(
        direct_channel, cqe_tensor, cqe_words,
        direct_channel->wqe_count);
    if (cq_result != 0) {
      const DeviceBatchStatus status =
          cq_result == kUrmaCqTimeout
              ? DeviceBatchStatus::kDirectCqTimeout
              : DeviceBatchStatus::kDirectCqError;
      for (std::uint32_t request = wave_first;
           request < request_cursor; ++request) {
        __gm__ UrmaH2DDescriptor* descriptor = &descriptors[request];
        if (descriptor->state ==
            static_cast<std::uint32_t>(TransferState::kSubmitted)) {
          SetDescriptorFailureAiv(descriptor, status, cq_result);
        }
      }
      SetCompletionFailure(
          completion, status, first_submitted, cq_result);
      return;
    }
    for (std::uint32_t request = wave_first;
         request < request_cursor; ++request) {
      __gm__ UrmaH2DDescriptor* descriptor = &descriptors[request];
      if (descriptor->state ==
          static_cast<std::uint32_t>(TransferState::kSubmitted)) {
        descriptor->state =
            static_cast<std::uint32_t>(TransferState::kComplete);
        ++completion->completed_count;
      }
    }
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

void LaunchAicpuOwnerConcurrencyProbe(
    void* stream, void* command, const void* status,
    AicpuOwnerProbeAivResult* result, std::uint64_t session_id,
    std::uint32_t max_ready_polls) {
  KvafAicpuOwnerConcurrencyProbeKernel<<<1, nullptr, stream>>>(
      static_cast<kvaf_urma_owner_probe_command_v1*>(command),
      static_cast<const kvaf_urma_owner_probe_status_v1*>(status), result,
      session_id, max_ready_polls);
}

void LaunchAivEpochOwnerWaitReady(
    void* stream, const void* owner_status, const void* open_result,
    AicpuEpochOwnerAivResult* aiv_result, std::uint64_t epoch,
    std::uint64_t session_id, std::uint32_t max_polls) {
  KvafAivEpochOwnerWaitReadyKernel<<<1, nullptr, stream>>>(
      static_cast<const kvaf_urma_epoch_owner_status_v1*>(owner_status),
      static_cast<const kvaf_urma_provider_result_v2*>(open_result),
      aiv_result, epoch, session_id, max_polls);
}

void LaunchAivEpochOwnerShutdown(
    void* stream, void* command, const void* owner_status,
    AicpuEpochOwnerAivResult* aiv_result,
    std::uint64_t packed_sq_head_address,
    std::uint64_t sq_tail_address, std::uint64_t cq_tail_address,
    std::uint64_t epoch, std::uint64_t session_id,
    std::uint32_t max_polls) {
  KvafAivEpochOwnerShutdownKernel<<<1, nullptr, stream>>>(
      static_cast<kvaf_urma_epoch_owner_command_v1*>(command),
      static_cast<const kvaf_urma_epoch_owner_status_v1*>(owner_status),
      aiv_result, packed_sq_head_address, sq_tail_address,
      cq_tail_address, epoch, session_id, max_polls);
}

}  // namespace kvaf::access::a5
