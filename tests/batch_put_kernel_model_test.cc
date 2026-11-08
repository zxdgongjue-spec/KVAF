#include "kernel_operator.h"

#include <array>
#include <iostream>
#include <vector>

#define __attribute__(...)
#include "../access_operator/demo/aiv_batch_put/kernel/execute_batch_put_aiv.asc"
#undef __attribute__

#define CHECK(c)                                        \
  do {                                                  \
    if (!(c)) {                                         \
      std::cerr << "line " << __LINE__ << ": " #c "\n"; \
      std::abort();                                     \
    }                                                   \
  } while (false)

namespace aiv_model {
std::vector<std::uint32_t> sq(32768 * 16);
alignas(64) std::array<std::uint32_t, 8 * 16> cq{};
std::uint32_t sq_db = 0, cq_db = 0;
std::uint64_t produced = 0, completed = 0, cq_produced = 0, cq_consumed = 0;
std::uintptr_t source_base = 0, destination_base = 0;
std::size_t buffer_bytes = 0;
void Doorbell(std::uint32_t value, std::uint32_t* address) {
  if (address == &cq_db) {
    CHECK(cq_consumed < cq_produced);
    ++cq_consumed;
    cq_db = value;
    return;
  }
  CHECK(address == &sq_db);
  produced += value - sq_db;
  sq_db = value;
}
void Poll(void* address) {
  const auto probe = reinterpret_cast<std::uintptr_t>(address);
  CHECK(probe >= reinterpret_cast<std::uintptr_t>(cq.data()) &&
        probe < reinterpret_cast<std::uintptr_t>(cq.data()) + sizeof(cq));
  if (cq_consumed != cq_produced) return;
  while (completed < produced) {
    const auto* wqe = sq.data() + (completed++ % 32768) * 16;
    CHECK(((wqe[1] >> 8U) & 255U) == 3U);
    const auto source = wqe[14] | (std::uint64_t(wqe[15]) << 32U);
    const auto destination = wqe[10] | (std::uint64_t(wqe[11]) << 32U);
    CHECK(source >= source_base && source - source_base <= buffer_bytes);
    CHECK(destination >= destination_base && destination - destination_base <= buffer_bytes);
    CHECK(wqe[12] <= buffer_bytes - (source - source_base));
    CHECK(wqe[12] <= buffer_bytes - (destination - destination_base));
    std::memcpy(reinterpret_cast<void*>(destination), reinterpret_cast<void*>(source), wqe[12]);
    if (wqe[0] & (1U << 21U)) {
      cq[(cq_produced % 8) * 16] = (1U - ((cq_produced / 8) & 1U)) << 2U;
      ++cq_produced;
      return;
    }
  }
}
}  // namespace aiv_model

int main() {
  constexpr std::size_t rows = 257, bytes = 16;
  std::array<std::uint8_t, rows * bytes> source{}, destination{};
  std::array<std::uint64_t, rows> offsets{}, lengths{}, addresses{};
  for (std::size_t i = 0; i < source.size(); ++i) source[i] = (i * 7 + 1) % 251;
  for (std::size_t i = 0; i < rows; ++i) {
    offsets[i] = i * bytes;
    lengths[i] = bytes;
    addresses[i] = 0x1000 + (rows - i - 1) * bytes;
  }
  aiv_model::source_base = reinterpret_cast<std::uintptr_t>(source.data());
  aiv_model::destination_base = reinterpret_cast<std::uintptr_t>(destination.data());
  aiv_model::buffer_bytes = source.size();
  KvafBatchPutDirectUrmaEntity entity{};
  entity.magic = 0x4b564155;
  entity.version = 2;
  entity.local_hbm_address = aiv_model::source_base;
  entity.local_hbm_size = source.size();
  entity.remote_host_ddr_address = 0x1000;
  entity.remote_host_ddr_size = destination.size();
  entity.registered_base = aiv_model::destination_base;
  entity.local_token_id = entity.local_token_value = 1;
  entity.remote_token_id = entity.remote_token_value = 1;
  entity.sq_va = reinterpret_cast<std::uintptr_t>(aiv_model::sq.data());
  entity.sq_db_va = reinterpret_cast<std::uintptr_t>(&aiv_model::sq_db);
  entity.sq_depth = 32768;
  entity.wqe_size = 64;
  entity.cq_va = reinterpret_cast<std::uintptr_t>(aiv_model::cq.data());
  entity.cq_db_va = reinterpret_cast<std::uintptr_t>(&aiv_model::cq_db);
  entity.cq_depth = 8;
  entity.cqe_size = 64;
  KvafBatchPutQueueState queue{};
  KvafBatchPutCoreStatus status{};
  KvafExecuteBatchPutLaunchArgs args{};
  args.request = {aiv_model::source_base,
                  reinterpret_cast<std::uintptr_t>(offsets.data()),
                  reinterpret_cast<std::uintptr_t>(lengths.data()),
                  reinterpret_cast<std::uintptr_t>(addresses.data()),
                  reinterpret_cast<std::uintptr_t>(&status),
                  source.size(),
                  rows,
                  1};
  args.channel_entity = reinterpret_cast<std::uintptr_t>(&entity);
  args.queue_state = reinterpret_cast<std::uintptr_t>(&queue);
  args.jettys_per_core = 1;
  // Run the production kernel, consume its WRITE WQEs, and verify actual payload.
  // Reuse the queue to catch state/credit regressions across submissions.
  for (unsigned replay = 0; replay < 2; ++replay) {
    destination.fill(0);
    KvafExecuteBatchPutAivKernelV2(args);
    CHECK(status.magic == KVAF_AIV_BATCH_PUT_STATUS_MAGIC && status.result == 0);
    CHECK(aiv_model::completed == rows * (replay + 1));
    for (std::size_t i = 0; i < rows; ++i)
      CHECK(std::memcmp(source.data() + i * bytes, destination.data() + (rows - i - 1) * bytes,
                        bytes) == 0);
  }
  entity.registered_base = UINT64_MAX - 1;
  KvafExecuteBatchPutAivKernelV2(args);
  CHECK(status.result == KVAF_BATCH_PUT_CORE_INVALID_ENTITY);
  CHECK(aiv_model::produced == 2 * rows);
  std::cout << "AIV Put payload, canonical address translation and binding overflow passed\n";
}
