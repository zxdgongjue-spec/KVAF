#include "batch_get_abi.h"
#include "batch_put_internal.h"
#include "aicpu_urma/wqe.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

extern "C" std::uint32_t KvafAicpuUrmaBatchPut(KvafExecuteBatchPutLaunchArgs* args);
extern "C" std::uint32_t KvafAicpuUrmaBatchGet(KvafExecuteBatchGetKernelArgs* args);

template <typename Entity>
void Setup(Entity& entity, std::array<std::uint32_t, 1024>& sq,
           std::array<std::uint32_t, 512>& cq, std::uint32_t& sq_doorbell,
           std::uint32_t& cq_doorbell, void* remote, std::size_t bytes) {
  entity.magic = 0x4b564155U;
  entity.version = 1U;
  entity.remote_host_ddr_address = reinterpret_cast<std::uint64_t>(remote);
  entity.remote_host_ddr_size = bytes;
  entity.remote_token_id = 3U;
  entity.remote_token_value = 4U;
  entity.tp_id = 5U;
  entity.sq_va = reinterpret_cast<std::uint64_t>(sq.data());
  entity.sq_db_va = reinterpret_cast<std::uint64_t>(&sq_doorbell);
  entity.sq_depth = 64U;
  entity.wqe_size = 64U;
  entity.cq_va = reinterpret_cast<std::uint64_t>(cq.data());
  entity.cq_db_va = reinterpret_cast<std::uint64_t>(&cq_doorbell);
  entity.cq_depth = 64U;
  entity.cqe_size = 32U;
  cq[0] = cq[8] = 4U;
}

void Put() {
  std::array<std::uint64_t, 33> remote{}, source{};
  std::array<std::uint64_t, 33> offsets{}, lengths{}, destinations{};
  std::array<std::uint32_t, 33> source_indices{};
  std::array<std::uint32_t, 1024> sq{};
  std::array<std::uint32_t, 512> cq{};
  std::uint32_t sq_doorbell = 0, cq_doorbell = 0;
  KvafBatchPutDirectUrmaEntity entity{};
  Setup(entity, sq, cq, sq_doorbell, cq_doorbell, remote.data(), sizeof(remote));
  KvafBatchPutQueueState queue{};
  KvafBatchPutCoreStatus status{};
  KvafBatchPutSourceDescriptor descriptor{reinterpret_cast<std::uint64_t>(source.data()), 7U, 0U};
  for (std::size_t index = 0; index < lengths.size(); ++index) {
    offsets[index] = index * 8;
    lengths[index] = 8;
    destinations[index] = reinterpret_cast<std::uint64_t>(&remote[index]);
  }
  KvafExecuteBatchPutLaunchArgs args{};
  args.request = {reinterpret_cast<std::uint64_t>(&descriptor),
                  reinterpret_cast<std::uint64_t>(source_indices.data()),
                  reinterpret_cast<std::uint64_t>(offsets.data()),
                  reinterpret_cast<std::uint64_t>(lengths.data()),
                  reinterpret_cast<std::uint64_t>(destinations.data()),
                  reinterpret_cast<std::uint64_t>(&status), 33, 1};
  args.channel_entity = reinterpret_cast<std::uint64_t>(&entity);
  args.queue_state = reinterpret_cast<std::uint64_t>(&queue);
  args.jettys_per_core = 1;
  args.reserved0 = 1;
  assert(KvafAicpuUrmaBatchPut(&args) == 0U);
  assert(status.magic == KVAF_AIV_BATCH_PUT_STATUS_MAGIC && status.result == 0U);
  assert(status.processed_request_count == 33U && status.failed_request_index == UINT64_MAX);
  assert(status.cq_status == 0U && status.cq_sub_status == 0U && status.reserved[0] == 0U);
  assert(sq_doorbell == 33U && cq_doorbell == 2U);
  assert(sq[0] == 0x30150000U && sq[32 * 16] == 0x303e0000U);
  assert(sq[1] == kvaf::aicpu_urma::kWriteOpcode << 8U);
  assert(sq[32 * 16 + 1] == kvaf::aicpu_urma::kWriteOpcode << 8U);
  source_indices[0] = 2U;
  assert(KvafAicpuUrmaBatchPut(&args) == 0U &&
         status.result == KVAF_BATCH_PUT_CORE_INVALID_SOURCE_RANGE &&
         status.failed_request_index == 0U);
  assert(sq_doorbell == 33U);
  args.reserved0 = 0U;
  assert(KvafAicpuUrmaBatchPut(&args) == 0U &&
         status.result == KVAF_BATCH_PUT_CORE_INVALID_ARGUMENT);
  assert(sq_doorbell == 33U);
  args.reserved0 = 1U;
  source_indices[0] = 0U;
  entity.magic = 0U;
  assert(KvafAicpuUrmaBatchPut(&args) == 0U &&
         status.result == KVAF_BATCH_PUT_CORE_INVALID_ENTITY);
  entity.magic = 0x4b564155U;
  queue.magic = 0U;
  assert(KvafAicpuUrmaBatchPut(&args) == 0U &&
         status.result == KVAF_BATCH_PUT_CORE_INVALID_QUEUE);
  assert(sq_doorbell == 33U);
  queue = {};
  cq[0] = (3U << 24U) | (5U << 16U) | 4U;
  assert(KvafAicpuUrmaBatchPut(&args) == 0U &&
         status.result == KVAF_BATCH_PUT_CORE_CQ_ERROR &&
         status.cq_status == 3U && status.cq_sub_status == 5U &&
         status.reserved[0] == cq[0]);
  queue = {};
  cq[0] = 4U;
  cq[8] = (7U << 24U) | (9U << 16U) | 4U;
  assert(KvafAicpuUrmaBatchPut(&args) == 0U &&
         status.result == KVAF_BATCH_PUT_CORE_CQ_ERROR &&
         status.processed_request_count == 32U &&
         status.cq_status == 7U && status.cq_sub_status == 9U &&
         status.reserved[0] == cq[8]);
  queue = {};
  queue.magic = 0x4b564151U;
  queue.initialized = 1U;
  queue.sq_producer_index = entity.sq_depth;
  cq[8] = 4U;
  assert(KvafAicpuUrmaBatchPut(&args) == 0U && status.result == 0U);
  assert(sq[0] == 0xb0150000U && sq[32 * 16] == 0xb03e0000U);
}

void Get() {
  std::array<std::uint64_t, 2> remote{}, output{};
  std::array<std::uint64_t, 1> addresses{reinterpret_cast<std::uint64_t>(remote.data())};
  std::array<std::uint8_t, 1> valid{1};
  std::array<std::int32_t, 3> indices{0, 0, 0}, offsets{0, 0, 8};
  std::array<std::uint8_t, 3> mask{1, 0, 1};
  std::array<std::int32_t, 3> output_indices{0, 0, 1};
  std::array<std::uint32_t, 1024> sq{};
  std::array<std::uint32_t, 512> cq{};
  std::uint32_t sq_doorbell = 0, cq_doorbell = 0;
  KvafBatchGetDirectUrmaEntity entity{};
  Setup(entity, sq, cq, sq_doorbell, cq_doorbell, remote.data(), sizeof(remote));
  entity.local_hbm_address = reinterpret_cast<std::uint64_t>(output.data());
  entity.local_hbm_size = sizeof(output);
  entity.local_token_id = 7U;
  KvafBatchGetQueueState queue{};
  KvafBatchGetCoreStatus status{};
  KvafExecuteBatchGetKernelArgs args{};
  args.address_table = reinterpret_cast<std::uint64_t>(addresses.data());
  args.valid_table = reinterpret_cast<std::uint64_t>(valid.data());
  args.indices = reinterpret_cast<std::uint64_t>(indices.data());
  args.offsets = reinterpret_cast<std::uint64_t>(offsets.data());
  args.mask = reinterpret_cast<std::uint64_t>(mask.data());
  args.output_indices = reinterpret_cast<std::uint64_t>(output_indices.data());
  args.output = reinterpret_cast<std::uint64_t>(output.data());
  args.core_status = reinterpret_cast<std::uint64_t>(&status);
  args.channel_entity = reinterpret_cast<std::uint64_t>(&entity);
  args.queue_state = reinterpret_cast<std::uint64_t>(&queue);
  args.table_capacity = 1;
  args.request_capacity = 3;
  args.output_rows = 2;
  args.token_bytes = 8;
  args.tokens_per_block = 2;
  args.block_count = 1;
  args.jettys_per_core = 1;
  assert(KvafAicpuUrmaBatchGet(&args) == 0U);
  assert(status.magic == KVAF_AIV_BATCH_GET_STATUS_MAGIC && status.result == 0U);
  assert(status.processed_request_count == 2U && sq_doorbell == 2U && cq_doorbell == 1U);
  assert(sq[1] == kvaf::aicpu_urma::kReadOpcode << 8U);
  assert(sq[16 + 13] == 7U);
  indices[0] = 5;
  assert(KvafAicpuUrmaBatchGet(&args) != 0U && status.failed_request_index == 0U);
  assert(sq_doorbell == 2U);
  indices[0] = 0;
  indices[2] = 5;
  assert(KvafAicpuUrmaBatchGet(&args) != 0U && status.failed_request_index == 2U);
  assert(sq_doorbell == 2U);
}

int main() {
  Put();
  Get();
  std::cout << "AICPU URMA kernel simulation: PASS\n";
}
