#include "wqe.h"
#include "gather_read.h"
#include "cpu_gather.h"

#include "batch_get_abi.h"
#include "batch_put_internal.h"

#include <cstdint>
#include <cstring>
#include <chrono>
#include <algorithm>
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

std::uint64_t NowNs() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

void Barrier() {
#if defined(KVAF_AICPU_URMA_HOST_TEST)
  std::atomic_thread_fence(std::memory_order_seq_cst);
#else
  asm volatile("dsb sy" ::: "memory");
#endif
}

void FlushLines(const void* address, std::uint64_t bytes) {
#if defined(KVAF_AICPU_URMA_HOST_TEST)
  (void)address;
  (void)bytes;
#else
  const auto first = reinterpret_cast<std::uintptr_t>(address) & ~std::uintptr_t{63};
  const auto end = reinterpret_cast<std::uintptr_t>(address) + bytes;
  for (auto line = first; line < end; line += 64U)
    asm volatile("dc civac, %0" :: "r"(line) : "memory");
#endif
}

void Flush(const void* address, std::uint64_t bytes) {
  FlushLines(address, bytes);
  Barrier();
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
                     std::uint64_t* last_wqe_protocol = nullptr,
                     std::uint64_t* post_ns = nullptr,
                     std::uint64_t* cq_ns = nullptr,
                     bool maintain_sq_cache = true) {
  if (cqe_word != nullptr) *cqe_word = 0U;
  if (post_ns != nullptr) *post_ns = 0U;
  if (cq_ns != nullptr) *cq_ns = 0U;
  if (count == 0U) return 0U;
  if (count >= entity.sq_depth) return 6U;
  const bool timed = post_ns != nullptr || cq_ns != nullptr;
  const auto post_start_ns = timed ? NowNs() : 0U;
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
    if (maintain_sq_cache) Flush(destination, sizeof(words));
  }
  queue.sq_producer_index += count;
  if (!maintain_sq_cache) Barrier();
  Ring(reinterpret_cast<volatile std::uint32_t*>(entity.sq_db_va),
       static_cast<std::uint32_t>(queue.sq_producer_index));
  Barrier();
  const auto cq_start_ns = timed ? NowNs() : 0U;
  if (post_ns != nullptr) *post_ns = cq_start_ns - post_start_ns;
  const auto result = WaitCompletion(entity, queue, cqe_word);
  if (cq_ns != nullptr) *cq_ns = NowNs() - cq_start_ns;
  return result;
}

template <typename Entity, typename Queue>
std::uint32_t SubmitNotify(const Entity& entity, Queue& queue,
                            std::uint64_t command, std::uint64_t notify,
                            std::uint64_t local, std::uint64_t sequence,
                            std::uint32_t* cqe_word,
                            std::uint64_t* post_ns,
                            std::uint64_t* cq_ns) {
  if (entity.sq_depth < 4U) return 6U;
  const auto post_start_ns = NowNs();
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
  }
  queue.sq_producer_index += 2U;
  Barrier();
  Ring(reinterpret_cast<volatile std::uint32_t*>(entity.sq_db_va),
       static_cast<std::uint32_t>(queue.sq_producer_index));
  Barrier();
  const auto cq_start_ns = NowNs();
  *post_ns = cq_start_ns - post_start_ns;
  const auto result = WaitCompletion(entity, queue, cqe_word);
  *cq_ns = NowNs() - cq_start_ns;
  return result;
}

template <typename Entity>
void WriteSqBlocks(const Entity& entity, std::uint64_t producer,
                   const std::uint32_t* words, std::uint32_t blocks) {
  const auto slot = static_cast<std::uint32_t>(producer % entity.sq_depth);
  const auto first = std::min(blocks, entity.sq_depth - slot);
  std::memcpy(reinterpret_cast<void*>(entity.sq_va + slot * 64U),
              words, first * 64U);
  if (first < blocks)
    std::memcpy(reinterpret_cast<void*>(entity.sq_va), words + first * 16U,
                (blocks - first) * 64U);
}

template <typename Entity, typename Queue>
std::uint32_t SubmitScatter(const Entity& entity, Queue& queue,
                            const kvaf::aicpu_urma::ScatterTransfer* transfers,
                            std::uint32_t count, std::uint32_t* cqe_word,
                            std::uint64_t* post_ns, std::uint64_t* cq_ns) {
  if (cqe_word != nullptr) *cqe_word = 0U;
  if (post_ns != nullptr) *post_ns = 0U;
  if (cq_ns != nullptr) *cq_ns = 0U;
  if (count == 0U) return 0U;
  std::uint32_t blocks = 0U;
  for (std::uint32_t index = 0U; index < count; ++index) {
    if (transfers[index].sge_count == 0U ||
        transfers[index].sge_count > kvaf::kGatherMaxSglSize) return 6U;
    blocks += kvaf::aicpu_urma::ScatterWqebbs(transfers[index].sge_count);
  }
  if (blocks >= entity.sq_depth) return 6U;
  Endpoint endpoint{entity.tp_id, entity.remote_token_id,
                    entity.remote_token_value, {}};
  std::memcpy(endpoint.remote_eid, entity.remote_eid, sizeof(endpoint.remote_eid));
  const auto post_start_ns = NowNs();
  std::uint64_t producer = queue.sq_producer_index;
  constexpr std::uint32_t kStagingBlocks = 8U * kvaf::aicpu_urma::kMaxScatterWqebbs;
  std::uint32_t words[kStagingBlocks * 16U];
  std::uint32_t staged_blocks = 0U;
  std::uint64_t staged_producer = producer;
  for (std::uint32_t index = 0U; index < count; ++index) {
    const auto wqebbs = kvaf::aicpu_urma::ScatterWqebbs(transfers[index].sge_count);
    if (staged_blocks + wqebbs > kStagingBlocks) {
      WriteSqBlocks(entity, staged_producer, words, staged_blocks);
      staged_producer = producer;
      staged_blocks = 0U;
    }
    kvaf::aicpu_urma::EncodeScatterReadWqe(
        words + staged_blocks * 16U, endpoint, transfers[index],
        static_cast<std::uint32_t>((producer / entity.sq_depth) & 1U),
        index + 1U == count);
    staged_blocks += wqebbs;
    producer += wqebbs;
  }
  if (staged_blocks != 0U) WriteSqBlocks(entity, staged_producer, words, staged_blocks);
  Barrier();
  queue.sq_producer_index = producer;
  Ring(reinterpret_cast<volatile std::uint32_t*>(entity.sq_db_va),
       static_cast<std::uint32_t>(producer));
  Barrier();
  const auto cq_start_ns = NowNs();
  if (post_ns != nullptr) *post_ns = cq_start_ns - post_start_ns;
  const auto result = WaitCompletion(entity, queue, cqe_word);
  if (cq_ns != nullptr) *cq_ns = NowNs() - cq_start_ns;
  return result;
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
  if (args->output_extent > UINT64_MAX - 63U) {
    Barrier();
    return 1U;
  }
  const std::uint64_t scratch_offset =
      (args->output_extent + 63U) & ~std::uint64_t{63U};
  const bool direct = args->source_span_count != 0U ||
                      args->remote_descriptor != 0U;
  const std::uint64_t source_bytes =
      kvaf::peer::GatherDescriptorBytes(args->source_span_count);
  if (!Valid(entity, queue) || entity.local_token_id == 0U ||
      args->sequence == 0U || args->bytes == 0U ||
      args->bytes > UINT32_MAX - kvaf::aicpu_urma::kGatherScratchBytes ||
      args->output_spans == 0U || args->output_span_count == 0U ||
      args->output_span_count > 65536U || args->sgl_size == 0U ||
      args->output_extent == 0U || args->output > UINT64_MAX - scratch_offset ||
      scratch_offset > UINT64_MAX - kvaf::aicpu_urma::kGatherScratchBytes ||
      args->output != entity.local_hbm_address ||
      !InRange(entity.local_hbm_address, entity.local_hbm_size, args->output,
               scratch_offset + kvaf::aicpu_urma::kGatherScratchBytes) ||
      !InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size,
               args->remote_staging, args->bytes) ||
      !InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size,
               args->remote_command, 8U) ||
      !InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size,
               args->remote_notify, 8U) ||
      !InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size,
               args->remote_ack, sizeof(kvaf::peer::GatherControl) - 8U) ||
      (direct && (args->sequence >= kvaf::peer::kGatherErrorBit ||
                  args->source_span_count == 0U ||
                  args->source_span_count > 65536U ||
                  args->remote_descriptor == 0U ||
                  args->source_descriptor < args->output ||
                  args->source_descriptor - args->output != scratch_offset +
                      kvaf::aicpu_urma::kGatherScratchBytes ||
                  !InRange(entity.local_hbm_address, entity.local_hbm_size,
                           args->source_descriptor, source_bytes) ||
                  !InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size,
                           args->remote_descriptor, source_bytes)))) {
    Barrier();
    return 1U;
  }
  const auto* output_spans =
      reinterpret_cast<const kvaf::GatherOutputSpan*>(args->output_spans);
  std::uint64_t output_bytes = 0U;
  for (std::uint32_t index = 0U; index < args->output_span_count; ++index) {
    const auto& span = output_spans[index];
    if (span.reserved != 0U || span.length == 0U ||
        span.offset > args->output_extent ||
        span.length > args->output_extent - span.offset ||
        span.length > args->bytes - output_bytes) {
      Barrier();
      return 1U;
    }
    output_bytes += span.length;
  }
  if (output_bytes != args->bytes) {
    Barrier();
    return 1U;
  }
  Initialize(queue);
  status.result = 0U;
  const auto scratch = args->output + scratch_offset;
  auto* ack = reinterpret_cast<std::uint64_t*>(scratch + 8U);
  *ack = 0U;
  *reinterpret_cast<std::uint64_t*>(scratch + 16U) =
      direct ? args->remote_descriptor : args->sequence;
  Barrier();
  std::uint32_t cqe_word = 0U;
  if (direct) {
    status.phase = 1U;
    const auto source_start_ns = NowNs();
    const Transfer descriptor{args->remote_descriptor, args->source_descriptor,
                              static_cast<std::uint32_t>(source_bytes), entity.local_token_id};
    status.result = Submit(entity, queue, &descriptor, 1U,
                           kvaf::aicpu_urma::kWriteOpcode, &cqe_word, nullptr,
                           &status.source_write_post_ns, &status.source_write_cq_ns,
                           false);
    status.source_write_ns = NowNs() - source_start_ns;
    status.cqe_word = cqe_word;
  }
  if (status.result != 0U) {
    status.sq_producer_index = queue.sq_producer_index;
    status.cq_consumer_index = queue.cq_consumer_index;
    Barrier();
    return 1U;
  }
  status.phase = direct ? 2U : 1U;
  const auto notify_start_ns = NowNs();
  status.result = SubmitNotify(entity, queue, args->remote_command,
                               args->remote_notify, scratch + 16U, args->sequence,
                               &cqe_word, &status.notify_post_ns,
                               &status.notify_cq_ns);
  status.notify_ns = NowNs() - notify_start_ns;
  status.cqe_word = cqe_word;
  if (status.result == 0U) {
    status.phase = direct ? 3U : 2U;
    const Transfer check{args->remote_ack, scratch + 8U,
                         sizeof(kvaf::peer::GatherControl) - 8U,
                         entity.local_token_id};
    const auto ack_start_ns = NowNs();
    for (std::uint32_t poll = 0U; poll < 10000U; ++poll) {
      std::uint64_t post_ns = 0U;
      std::uint64_t cq_ns = 0U;
      status.result = Submit(entity, queue, &check, 1U, kvaf::aicpu_urma::kReadOpcode,
                             &cqe_word, nullptr, &post_ns, &cq_ns, false);
      status.ack_post_ns += post_ns;
      status.ack_cq_ns += cq_ns;
      ++status.ack_polls;
      status.cqe_word = cqe_word;
      if (status.result != 0U) break;
      status.ack_seen = *ack;
      if (direct && status.ack_started_ns == 0U &&
          status.ack_seen == kvaf::peer::GatherStartedAck(args->sequence))
        status.ack_started_ns = NowNs() - ack_start_ns;
      if (status.ack_seen == args->sequence ||
          status.ack_seen == (args->sequence | kvaf::peer::kGatherErrorBit)) break;
    }
    status.ack_poll_ns = NowNs() - ack_start_ns;
    if (status.result == 0U &&
        status.ack_seen == (args->sequence | kvaf::peer::kGatherErrorBit))
      status.result = 10U;
    if (status.result == 0U && status.ack_seen == args->sequence)
      std::memcpy(&status.cpu, reinterpret_cast<const void*>(scratch + 24U),
                  sizeof(status.cpu));
    if (status.result == 0U && status.ack_seen != args->sequence)
      status.result = 9U;
  }
  if (status.result == 0U) {
    status.phase = direct ? 4U : 3U;
    const std::uint32_t wqe_sgl_size = args->sgl_size < kvaf::kGatherMaxSglSize
        ? args->sgl_size : kvaf::kGatherMaxSglSize;
    kvaf::aicpu_urma::ScatterTransfer data[64]{};
    const std::uint32_t max_batch = kvaf::aicpu_urma::MaxScatterBatch(entity.sq_depth);
    if (max_batch == 0U) status.result = 6U;
    std::uint32_t batch_count = 0U;
    std::uint32_t span_index = 0U;
    std::uint32_t span_offset = 0U;
    std::uint64_t remote_offset = 0U;
    auto plan_start_ns = NowNs();
    while (span_index < args->output_span_count && status.result == 0U) {
      auto& transfer = data[batch_count];
      transfer = {};
      transfer.remote_address = args->remote_staging + remote_offset;
      while (span_index < args->output_span_count &&
             transfer.sge_count < wqe_sgl_size &&
             transfer.bytes < kvaf::aicpu_urma::kMaxReadBytes) {
        const auto& span = output_spans[span_index];
        const auto available = span.length - span_offset;
        const auto capacity = kvaf::aicpu_urma::kMaxReadBytes - transfer.bytes;
        const auto length = available < capacity ? available : capacity;
        transfer.sges[transfer.sge_count++] = {
            args->output + span.offset + span_offset, length, entity.local_token_id};
        transfer.bytes += length;
        remote_offset += length;
        span_offset += length;
        if (span_offset == span.length) {
          ++span_index;
          span_offset = 0U;
        }
      }
      ++batch_count;
      if (batch_count == max_batch || span_index == args->output_span_count) {
        status.scatter_plan_ns += NowNs() - plan_start_ns;
        status.read_wqes += batch_count;
        std::uint64_t post_ns = 0U;
        std::uint64_t cq_ns = 0U;
        status.result = SubmitScatter(entity, queue, data, batch_count, &cqe_word,
                                      &post_ns, &cq_ns);
        status.scatter_post_ns += post_ns;
        status.scatter_cq_ns += cq_ns;
        status.cqe_word = cqe_word;
        batch_count = 0U;
        plan_start_ns = NowNs();
      }
    }
    status.cqe_word = cqe_word;
  }
  status.sq_producer_index = queue.sq_producer_index;
  status.cq_consumer_index = queue.cq_consumer_index;
  Barrier();
  return status.result == 0U ? 0U : 1U;
}
