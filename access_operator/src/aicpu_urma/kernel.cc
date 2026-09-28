#include "wqe.h"
#include "gather_read.h"

#include "batch_get_abi.h"
#include "batch_put_internal.h"

#include <cstdint>
#include <cstring>
#if defined(KVAF_AICPU_URMA_HOST_TEST)
#include <atomic>
#endif

#if !defined(__aarch64__) && !defined(KVAF_AICPU_URMA_HOST_TEST)
#error "The AICPU URMA kernel must be compiled for aarch64"
#endif

namespace {
using kvaf::aicpu_urma::EncodeWqe;
using kvaf::aicpu_urma::Endpoint;
using kvaf::aicpu_urma::InRange;
using kvaf::aicpu_urma::Transfer;

constexpr std::uint32_t kEntityMagic = 0x4b564155U;
constexpr std::uint32_t kQueueMagic = 0x4b564151U;
constexpr std::uint32_t kPollLimit = 10000000U;

void Barrier() {
#if defined(KVAF_AICPU_URMA_HOST_TEST)
  std::atomic_thread_fence(std::memory_order_seq_cst);
#else
  asm volatile("dsb sy" ::: "memory");
#endif
}

void Flush(const void* address, std::uint64_t bytes) {
#if defined(KVAF_AICPU_URMA_HOST_TEST)
  (void)address;
  (void)bytes;
#else
  const auto first = reinterpret_cast<std::uintptr_t>(address) & ~std::uintptr_t{63};
  const auto end = reinterpret_cast<std::uintptr_t>(address) + bytes;
  for (auto line = first; line < end; line += 64U)
    asm volatile("dc civac, %0" :: "r"(line) : "memory");
  Barrier();
#endif
}

void Ring(volatile std::uint32_t* doorbell, std::uint32_t value) {
#if defined(KVAF_AICPU_URMA_HOST_TEST)
  __atomic_store_n(const_cast<std::uint32_t*>(doorbell), value, __ATOMIC_RELEASE);
#else
  *doorbell = value;
#endif
}

std::uint32_t LoadCompletion(volatile std::uint32_t* completion) {
#if defined(KVAF_AICPU_URMA_HOST_TEST)
  return __atomic_load_n(const_cast<std::uint32_t*>(completion), __ATOMIC_ACQUIRE);
#else
  return *completion;
#endif
}

template <typename Entity>
bool ValidEntity(const Entity& entity) {
  return entity.magic == kEntityMagic && entity.version == 1U &&
         entity.remote_host_ddr_address != 0U && entity.remote_host_ddr_size != 0U &&
         entity.remote_token_id != 0U && entity.remote_token_value != 0U &&
         entity.sq_va != 0U && entity.sq_db_va != 0U && entity.sq_depth >= 2U &&
         entity.wqe_size == 64U && entity.cq_va != 0U && entity.cq_db_va != 0U &&
         entity.cq_depth != 0U && entity.cqe_size >= 32U;
}

template <typename Queue>
bool ValidQueue(const Queue& queue) {
  return queue.initialized == 0U || queue.magic == kQueueMagic;
}

template <typename Entity, typename Queue>
bool Valid(const Entity& entity, const Queue& queue) {
  return ValidEntity(entity) && ValidQueue(queue);
}

template <typename Queue>
void Initialize(Queue& queue) {
  if (queue.initialized != 0U) return;
  queue.magic = kQueueMagic;
  queue.initialized = 1U;
  queue.sq_producer_index = 0U;
  queue.cq_consumer_index = 0U;
  queue.completed_batch_count = 0U;
  queue.last_error = 0U;
}

template <typename Entity, typename Queue>
std::uint32_t WaitCompletion(const Entity& entity, Queue& queue,
                             std::uint32_t* cqe_word) {
  const auto slot = queue.cq_consumer_index % entity.cq_depth;
  auto* completion = reinterpret_cast<volatile std::uint32_t*>(
      entity.cq_va + slot * entity.cqe_size);
  const auto expected = static_cast<std::uint32_t>(
      (queue.cq_consumer_index / entity.cq_depth) & 1U);
  for (std::uint32_t poll = 0U; poll < kPollLimit; ++poll) {
    Flush(const_cast<const std::uint32_t*>(completion), 32U);
    const std::uint32_t word = LoadCompletion(completion);
    if (((word >> 2U) & 1U) == expected) continue;
    if (cqe_word != nullptr) *cqe_word = word;
    ++queue.cq_consumer_index;
    Ring(reinterpret_cast<volatile std::uint32_t*>(entity.cq_db_va),
         static_cast<std::uint32_t>(queue.cq_consumer_index) & 0x003fffffU);
    Barrier();
    return ((word >> 16U) & 0xffffU) == 0U ? 0U : 8U;
  }
  return 7U;
}

template <typename Entity, typename Queue>
std::uint32_t Submit(const Entity& entity, Queue& queue,
                     const Transfer* transfers, std::uint32_t count,
                     std::uint32_t opcode, std::uint32_t* cqe_word = nullptr,
                     std::uint64_t* last_wqe_protocol = nullptr) {
  if (cqe_word != nullptr) *cqe_word = 0U;
  if (count == 0U) return 0U;
  if (count >= entity.sq_depth) return 6U;
  Endpoint endpoint{entity.tp_id, entity.remote_token_id,
                    entity.remote_token_value, {}};
  std::memcpy(endpoint.remote_eid, entity.remote_eid, sizeof(endpoint.remote_eid));
  for (std::uint32_t index = 0U; index < count; ++index) {
    const std::uint64_t producer = queue.sq_producer_index + index;
    const auto slot = producer % entity.sq_depth;
    std::uint32_t words[16];
    EncodeWqe(words, endpoint, transfers[index], opcode,
              static_cast<std::uint32_t>((producer / entity.sq_depth) & 1U),
              index + 1U == count);
    if (last_wqe_protocol != nullptr && index + 1U == count)
      *last_wqe_protocol = words[0];
    auto* destination = reinterpret_cast<void*>(entity.sq_va + slot * 64U);
    std::memcpy(destination, words, sizeof(words));
    Flush(destination, sizeof(words));
  }
  queue.sq_producer_index += count;
  Ring(reinterpret_cast<volatile std::uint32_t*>(entity.sq_db_va),
       static_cast<std::uint32_t>(queue.sq_producer_index));
  Barrier();
  return WaitCompletion(entity, queue, cqe_word);
}

template <typename Entity, typename Queue>
std::uint32_t SubmitNotify(const Entity& entity, Queue& queue,
                            std::uint64_t command, std::uint64_t notify,
                            std::uint64_t local, std::uint64_t sequence,
                            std::uint32_t* cqe_word) {
  if (entity.sq_depth < 4U) return 6U;
  Endpoint endpoint{entity.tp_id, entity.remote_token_id,
                    entity.remote_token_value, {}};
  std::memcpy(endpoint.remote_eid, entity.remote_eid, sizeof(endpoint.remote_eid));
  std::uint32_t words[32];
  EncodeNotifyWqe(words, endpoint, command, notify, local, entity.local_token_id,
                  sequence, static_cast<std::uint32_t>(
                      (queue.sq_producer_index / entity.sq_depth) & 1U));
  for (std::uint32_t block = 0U; block < 2U; ++block) {
    const auto slot = (queue.sq_producer_index + block) % entity.sq_depth;
    auto* destination = reinterpret_cast<void*>(entity.sq_va + slot * 64U);
    std::memcpy(destination, &words[block * 16U], 64U);
    Flush(destination, 64U);
  }
  queue.sq_producer_index += 2U;
  Ring(reinterpret_cast<volatile std::uint32_t*>(entity.sq_db_va),
       static_cast<std::uint32_t>(queue.sq_producer_index));
  Barrier();
  return WaitCompletion(entity, queue, cqe_word);
}

template <typename Entity, typename Queue>
std::uint32_t Post(const Entity& entity, Queue& queue, Transfer* transfers,
                   std::uint32_t& count, std::uint32_t opcode,
                   std::uint32_t* cqe_word = nullptr,
                   std::uint64_t* last_wqe_protocol = nullptr) {
  const auto result = Submit(entity, queue, transfers, count, opcode, cqe_word,
                             last_wqe_protocol);
  if (result == 0U) count = 0U;
  return result;
}

bool MakeGetTransfer(const KvafExecuteBatchGetKernelArgs& args,
                     const KvafBatchGetDirectUrmaEntity& entity,
                     std::uint32_t index, Transfer* transfer) {
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(args.address_table);
  const auto* valid = reinterpret_cast<const std::uint8_t*>(args.valid_table);
  const auto* indices = reinterpret_cast<const std::int32_t*>(args.indices);
  const auto* offsets = reinterpret_cast<const std::int32_t*>(args.offsets);
  const auto* output_indices = reinterpret_cast<const std::int32_t*>(args.output_indices);
  const auto table_index = indices[index];
  const auto offset = offsets[index];
  const auto output_index = output_indices[index];
  const std::uint64_t block_bytes = static_cast<std::uint64_t>(args.token_bytes) * args.tokens_per_block;
  if (table_index < 0 || static_cast<std::uint32_t>(table_index) >= args.table_capacity ||
      valid[table_index] == 0U || offset < 0 || block_bytes < args.token_bytes ||
      static_cast<std::uint64_t>(offset) > block_bytes - args.token_bytes ||
      output_index < 0 || static_cast<std::uint32_t>(output_index) >= args.output_rows ||
      !InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size,
               addresses[table_index], block_bytes)) return false;
  const std::uint64_t output_offset = static_cast<std::uint64_t>(output_index) * args.token_bytes;
  if (args.output > UINT64_MAX - output_offset ||
      !InRange(entity.local_hbm_address, entity.local_hbm_size,
               args.output + output_offset, args.token_bytes)) return false;
  *transfer = {addresses[table_index] + static_cast<std::uint32_t>(offset),
               args.output + output_offset, args.token_bytes, entity.local_token_id};
  return true;
}

}  // namespace

extern "C" __attribute__((visibility("default")))
std::uint32_t KvafAicpuUrmaBatchPut(KvafExecuteBatchPutLaunchArgs* args) {
  if (args == nullptr || args->request.core_status == 0U) return 1U;
  const auto& request = args->request;
  auto& status = *reinterpret_cast<KvafBatchPutCoreStatus*>(request.core_status);
  status = {};
  status.magic = KVAF_AIV_BATCH_PUT_STATUS_MAGIC;
  status.result = KVAF_BATCH_PUT_CORE_INVALID_ARGUMENT;
  status.failed_request_index = UINT64_MAX;
  if (request.row_count == 0U || request.aiv_core_count != 1U ||
      args->reserved0 == 0U || args->reserved0 > request.row_count ||
      args->jettys_per_core != 1U || args->channel_entity == 0U || args->queue_state == 0U ||
      request.source_descriptors == 0U || request.source_indices == 0U ||
      request.input_offsets == 0U || request.lengths == 0U ||
      request.destination_addresses == 0U) {
    Flush(&status, sizeof(status));
    return 0U;
  }
  auto& entity = *reinterpret_cast<KvafBatchPutDirectUrmaEntity*>(args->channel_entity);
  auto& queue = *reinterpret_cast<KvafBatchPutQueueState*>(args->queue_state);
  if (!ValidEntity(entity)) {
    status.result = KVAF_BATCH_PUT_CORE_INVALID_ENTITY;
    Flush(&status, sizeof(status));
    return 0U;
  }
  if (!ValidQueue(queue)) {
    status.result = KVAF_BATCH_PUT_CORE_INVALID_QUEUE;
    Flush(&status, sizeof(status));
    return 0U;
  }
  Initialize(queue);
  status.result = KVAF_BATCH_PUT_CORE_SUCCESS;
  const auto* sources = reinterpret_cast<const KvafBatchPutSourceDescriptor*>(request.source_descriptors);
  const auto* source_indices = reinterpret_cast<const std::uint32_t*>(request.source_indices);
  const auto* offsets = reinterpret_cast<const std::uint64_t*>(request.input_offsets);
  const auto* lengths = reinterpret_cast<const std::uint64_t*>(request.lengths);
  const auto* destinations = reinterpret_cast<const std::uint64_t*>(request.destination_addresses);
  Flush(sources, static_cast<std::uint64_t>(args->reserved0) * sizeof(*sources));
  Flush(source_indices, static_cast<std::uint64_t>(request.row_count) * sizeof(*source_indices));
  Flush(offsets, static_cast<std::uint64_t>(request.row_count) * sizeof(*offsets));
  Flush(lengths, static_cast<std::uint64_t>(request.row_count) * sizeof(*lengths));
  Flush(destinations, static_cast<std::uint64_t>(request.row_count) * sizeof(*destinations));
  Transfer transfers[32];
  std::uint32_t count = 0U;
  std::uint32_t cqe_word = 0U;
  for (std::uint32_t index = 0U; index < request.row_count; ++index) {
    if (source_indices[index] >= args->reserved0) {
      status.result = KVAF_BATCH_PUT_CORE_INVALID_SOURCE_RANGE;
      status.failed_request_index = index;
      Flush(&status, sizeof(status));
      return 0U;
    }
    const auto& source = sources[source_indices[index]];
    const auto length = lengths[index];
    if (length == 0U || length > UINT32_MAX) {
      status.result = KVAF_BATCH_PUT_CORE_INVALID_LENGTH;
      status.failed_request_index = index;
      Flush(&status, sizeof(status));
      return 0U;
    }
    if (source.base_address > UINT64_MAX - offsets[index] || source.local_token_id == 0U) {
      status.result = KVAF_BATCH_PUT_CORE_INVALID_SOURCE_RANGE;
      status.failed_request_index = index;
      Flush(&status, sizeof(status));
      return 0U;
    }
    if (!InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size,
                 destinations[index], length)) {
      status.result = KVAF_BATCH_PUT_CORE_INVALID_DESTINATION_RANGE;
      status.failed_request_index = index;
      Flush(&status, sizeof(status));
      return 0U;
    }
    const auto local = source.base_address + offsets[index];
    Flush(reinterpret_cast<void*>(local), length);
    transfers[count++] = {destinations[index], local,
                          static_cast<std::uint32_t>(length), source.local_token_id};
    if (count == 32U || index + 1U == request.row_count) {
      status.result = Post(entity, queue, transfers, count, kvaf::aicpu_urma::kWriteOpcode,
                           &cqe_word, &status.last_wqe_protocol);
      if (status.result == KVAF_BATCH_PUT_CORE_CQ_ERROR) {
        status.cq_status = (cqe_word >> 24U) & 0xffU;
        status.cq_sub_status = (cqe_word >> 16U) & 0xffU;
        status.reserved[0] = cqe_word;
      }
      if (status.result != 0U) break;
    }
    status.processed_request_count = index + 1U;
  }
  status.sq_producer_index = queue.sq_producer_index;
  status.cq_consumer_index = queue.cq_consumer_index;
  if (status.result == 0U) status.completed_batch_count = 1U;
  Flush(&queue, sizeof(queue));
  Flush(&status, sizeof(status));
  return 0U;
}

extern "C" __attribute__((visibility("default")))
std::uint32_t KvafAicpuUrmaBatchGet(KvafExecuteBatchGetKernelArgs* args) {
  if (args == nullptr || args->block_count != 1U || args->jettys_per_core != 1U ||
      args->channel_entity == 0U || args->queue_state == 0U || args->core_status == 0U)
    return 1U;
  auto& entity = *reinterpret_cast<KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
  auto& queue = *reinterpret_cast<KvafBatchGetQueueState*>(args->queue_state);
  auto& status = *reinterpret_cast<KvafBatchGetCoreStatus*>(args->core_status);
  status = {};
  status.magic = KVAF_AIV_BATCH_GET_STATUS_MAGIC;
  status.result = 0U;
  status.failed_request_index = UINT64_MAX;
  if (!Valid(entity, queue) || args->address_table == 0U || args->valid_table == 0U ||
      args->indices == 0U || args->offsets == 0U || args->mask == 0U ||
      args->output_indices == 0U || args->output == 0U || args->token_bytes == 0U ||
      args->tokens_per_block == 0U ||
      entity.local_token_id == 0U || entity.local_hbm_address != args->output) {
    status.result = 2U;
    Flush(&status, sizeof(status));
    return 1U;
  }
  Initialize(queue);
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(args->address_table);
  const auto* valid = reinterpret_cast<const std::uint8_t*>(args->valid_table);
  const auto* indices = reinterpret_cast<const std::int32_t*>(args->indices);
  const auto* offsets = reinterpret_cast<const std::int32_t*>(args->offsets);
  const auto* mask = reinterpret_cast<const std::uint8_t*>(args->mask);
  const auto* output_indices = reinterpret_cast<const std::int32_t*>(args->output_indices);
  Flush(addresses, static_cast<std::uint64_t>(args->table_capacity) * sizeof(*addresses));
  Flush(valid, args->table_capacity);
  Flush(indices, static_cast<std::uint64_t>(args->request_capacity) * sizeof(*indices));
  Flush(offsets, static_cast<std::uint64_t>(args->request_capacity) * sizeof(*offsets));
  Flush(mask, args->request_capacity);
  Flush(output_indices, static_cast<std::uint64_t>(args->request_capacity) * sizeof(*output_indices));
  Transfer transfer{};
  for (std::uint32_t index = 0U; index < args->request_capacity; ++index) {
    if (mask[index] != 0U && !MakeGetTransfer(*args, entity, index, &transfer)) {
      status.result = 2U;
      status.failed_request_index = index;
      Flush(&status, sizeof(status));
      return 1U;
    }
  }
  Transfer transfers[32];
  std::uint32_t count = 0U;
  for (std::uint32_t index = 0U; index < args->request_capacity; ++index) {
    if (mask[index] == 0U) continue;
    MakeGetTransfer(*args, entity, index, &transfer);
    Flush(reinterpret_cast<void*>(transfer.local_address), transfer.length);
    transfers[count++] = transfer;
    if (count == 32U || index + 1U == args->request_capacity) {
      status.result = Post(entity, queue, transfers, count, kvaf::aicpu_urma::kReadOpcode);
      if (status.result != 0U) break;
    }
    ++status.processed_request_count;
  }
  if (status.result == 0U && count != 0U)
    status.result = Post(entity, queue, transfers, count, kvaf::aicpu_urma::kReadOpcode);
  if (status.result == 0U) {
    Flush(reinterpret_cast<void*>(args->output),
          static_cast<std::uint64_t>(args->output_rows) * args->token_bytes);
    status.completed_batch_count = 1U;
  }
  status.sq_producer_index = queue.sq_producer_index;
  status.cq_consumer_index = queue.cq_consumer_index;
  Flush(&queue, sizeof(queue));
  Flush(&status, sizeof(status));
  return status.result == 0U ? 0U : 1U;
}

extern "C" __attribute__((visibility("default")))
std::uint32_t KvafAicpuUrmaGatherRead(kvaf::aicpu_urma::GatherReadArgs* args) {
  using kvaf::aicpu_urma::GatherReadStatus;
  if (args == nullptr || args->channel_entity == 0U || args->queue_state == 0U ||
      args->core_status == 0U) return 1U;
  auto& entity = *reinterpret_cast<KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
  auto& queue = *reinterpret_cast<KvafBatchGetQueueState*>(args->queue_state);
  auto& status = *reinterpret_cast<GatherReadStatus*>(args->core_status);
  status = {};
  status.magic = kvaf::aicpu_urma::kGatherStatusMagic;
  status.sequence = args->sequence;
  status.result = 2U;
  if (!Valid(entity, queue) || entity.local_token_id == 0U ||
      args->sequence == 0U || args->bytes == 0U ||
      args->bytes > UINT32_MAX - kvaf::aicpu_urma::kGatherScratchBytes ||
      args->output != entity.local_hbm_address ||
      !InRange(entity.local_hbm_address, entity.local_hbm_size, args->output,
               ((static_cast<std::uint64_t>(args->bytes) + 63U) & ~std::uint64_t{63U}) +
                   kvaf::aicpu_urma::kGatherScratchBytes) ||
      !InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size,
               args->remote_staging, args->bytes) ||
      !InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size,
               args->remote_command, 8U) ||
      !InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size,
               args->remote_notify, 8U) ||
      !InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size,
               args->remote_ack, 8U)) {
    Flush(&status, sizeof(status));
    return 1U;
  }
  Initialize(queue);
  const auto scratch = args->output +
      ((static_cast<std::uint64_t>(args->bytes) + 63U) & ~std::uint64_t{63U});
  auto* ack = reinterpret_cast<std::uint64_t*>(scratch + 8U);
  *ack = 0U;
  *reinterpret_cast<std::uint64_t*>(scratch + 16U) = args->sequence;
  Flush(reinterpret_cast<void*>(scratch), kvaf::aicpu_urma::kGatherScratchBytes);
  status.phase = 1U;
  std::uint32_t cqe_word = 0U;
  status.result = SubmitNotify(entity, queue, args->remote_command,
                               args->remote_notify, scratch + 16U, args->sequence,
                               &cqe_word);
  status.cqe_word = cqe_word;
  if (status.result == 0U) {
    status.phase = 2U;
    const Transfer check{args->remote_ack, scratch + 8U, 8U, entity.local_token_id};
    for (std::uint32_t poll = 0U; poll < 10000U; ++poll) {
      Flush(ack, 8U);
      status.result = Submit(entity, queue, &check, 1U, kvaf::aicpu_urma::kReadOpcode,
                             &cqe_word);
      status.cqe_word = cqe_word;
      if (status.result != 0U) break;
      Flush(ack, 8U);
      status.ack_seen = *ack;
      if (status.ack_seen == args->sequence) break;
    }
    if (status.result == 0U && status.ack_seen != args->sequence)
      status.result = 9U;
  }
  if (status.result == 0U) {
    status.phase = 3U;
    Flush(reinterpret_cast<void*>(args->output), args->bytes);
    Transfer data[16];
    const auto count = kvaf::aicpu_urma::PlanContiguousRead(
        data, args->remote_staging, args->output, args->bytes, entity.local_token_id);
    status.result = Submit(entity, queue, data, count, kvaf::aicpu_urma::kReadOpcode,
                           &cqe_word);
    status.cqe_word = cqe_word;
    if (status.result == 0U) Flush(reinterpret_cast<void*>(args->output), args->bytes);
  }
  status.sq_producer_index = queue.sq_producer_index;
  status.cq_consumer_index = queue.cq_consumer_index;
  Flush(&queue, sizeof(queue));
  Flush(&status, sizeof(status));
  return status.result == 0U ? 0U : 1U;
}
