// Execute production URMA/UBMEM control flow with synchronous DMA and delayed
// CQ checkpoints. Device timing and cache visibility still require A5 tests.
#include "kernel_operator.h"
#include "batch_get_abi.h"
#include <algorithm>
#include <array>
#include <deque>
#include <iostream>
#include <vector>

#define __attribute__(...)
#include "../access_operator/algorithms/get/direct/urma_aiv/kernel/execute_batch_get_aiv.asc"
#include "../access_operator/algorithms/get/direct/ubmem_aiv/kernel/execute_batch_get_ubmem.asc"
#undef __attribute__

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::abort(); \
} } while (false)

namespace aiv_model {
struct Queue {
  std::vector<uint32_t> sq = std::vector<uint32_t>(32768 * 16);
  alignas(64) std::array<uint32_t, 8 * 16> cq{};
  uint32_t sq_db{}, cq_db{}, cq_depth{8};
  uint64_t producer{}, completed{}, cq_producer{}, cq_consumer{}, peak_live{};
  uint64_t remote_bias{};
  std::deque<uint64_t> checkpoints;
  bool error{}, timeout{};
};
std::vector<Queue> hardware;
uint64_t doorbells{};
void Doorbell(uint32_t value, uint32_t* address) {
  for (auto& q : hardware) {
    if (address == &q.cq_db) {
      CHECK(q.cq_consumer < q.cq_producer);
      ++q.cq_consumer;
      CHECK(value == (q.cq_consumer & 0x003fffffU));
      q.cq_db = value;
      return;
    }
    if (address != &q.sq_db) continue;
    ++doorbells;
    const uint32_t added = value - q.sq_db;
    CHECK(added && added <= 128);
    CHECK(q.producer + added - q.completed < 32768);
    for (uint32_t i = 0; i < added; ++i) {
      const auto pi = q.producer++;
      const auto* wqe = q.sq.data() + (pi % 32768) * 16;
      CHECK((wqe[0] >> 31U) == 1U - ((pi / 32768) & 1U));
      if (wqe[0] & (1U << 21U)) q.checkpoints.push_back(q.producer);
    }
    CHECK(q.checkpoints.size() <= q.cq_depth);
    q.peak_live = std::max(q.peak_live, q.producer - q.completed);
    q.sq_db = value;
    return;
  }
  CHECK(false);
}
void Poll(void* address) {
  const auto probe = reinterpret_cast<uintptr_t>(address);
  for (auto& q : hardware) {
    const auto begin = reinterpret_cast<uintptr_t>(q.cq.data());
    if (probe < begin || probe >= begin + sizeof(q.cq)) continue;
    if (q.timeout || q.checkpoints.empty() || q.cq_producer != q.cq_consumer) return;
    const auto end = q.checkpoints.front();
    q.checkpoints.pop_front();
    while (q.completed < end) {
      const auto* wqe = q.sq.data() + (q.completed++ % 32768) * 16;
      const uint64_t src = wqe[10] | (uint64_t(wqe[11]) << 32U);
      const uint64_t dst = wqe[14] | (uint64_t(wqe[15]) << 32U);
      std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src - q.remote_bias), wqe[12]);
    }
    const uint32_t owner = 1U - ((q.cq_producer / q.cq_depth) & 1U);
    q.cq[(q.cq_producer % q.cq_depth) * 16] = (owner << 2U) | (q.error ? 1U << 24U : 0U);
    ++q.cq_producer;
    return;
  }
  CHECK(false);
}
}  // namespace aiv_model

void CheckEvents() {
  for (auto& from : model_events) for (auto& to : from) for (bool event : to) CHECK(!event);
  for (auto& channel : AscendC::hard_events) for (bool event : channel) CHECK(!event);
}
struct Fixture {
  std::vector<uint8_t> source = std::vector<uint8_t>(257), output, valid;
  std::vector<uint64_t> addresses;
  std::vector<int32_t> indices, offsets, rows;
  std::vector<KvafBatchGetQueueState> queues;
  std::vector<KvafBatchGetDirectUrmaEntity> entities;
  std::vector<KvafBatchGetCoreStatus> reports;
  KvafExecuteBatchGetKernelArgs args{};
  int32_t count{};
  Fixture(uint32_t capacity, uint32_t cores, uint32_t jettys, uint32_t budget, uint32_t cq_depth = 8)
      : output(capacity), valid(source.size(), 1), addresses(source.size()),
        indices(capacity), offsets(capacity), rows(capacity), queues(cores * jettys),
        entities(cores * jettys), reports(cores) {
    aiv_model::hardware.clear(); aiv_model::hardware.resize(cores * jettys);
    for (uint32_t i = 0; i < source.size(); ++i) {
      source[i] = i % 251; addresses[i] = reinterpret_cast<uint64_t>(source.data() + i);
    }
    for (uint32_t i = 0; i < capacity; ++i) { indices[i] = i % source.size(); rows[i] = capacity - 1 - i; }
    for (uint32_t i = 0; i < entities.size(); ++i) {
      auto& e = entities[i]; auto& q = aiv_model::hardware[i]; q.cq_depth = cq_depth;
      e.magic = 0x4b564155U; e.version = KVAF_AIV_DIRECT_URMA_ENTITY_VERSION;
      e.remote_host_ddr_address = e.registered_base = reinterpret_cast<uint64_t>(source.data());
      // Distinct per-path registrations/tokens must survive queue windows and
      // ring wraparound while all paths retain the same logical source region.
      q.remote_bias = (uint64_t{i} + 1U) << 32U;
      e.registered_base += q.remote_bias;
      e.remote_token_id = i + 1U; e.remote_token_value = i + 100U;
      e.remote_host_ddr_size = source.size();
      e.local_hbm_address = reinterpret_cast<uint64_t>(output.data()); e.local_hbm_size = output.size();
      e.local_token_id = 1; e.sq_depth = 32768; e.wqe_size = 64;
      e.sq_va = reinterpret_cast<uint64_t>(q.sq.data()); e.sq_db_va = reinterpret_cast<uint64_t>(&q.sq_db);
      e.cq_va = reinterpret_cast<uint64_t>(q.cq.data()); e.cq_db_va = reinterpret_cast<uint64_t>(&q.cq_db);
      e.cq_depth = cq_depth; e.cqe_size = 64;
    }
    for (auto& r : reports) r.profile.magic = KVAF_AIV_BATCH_GET_PROFILE_MAGIC;
    args.address_table = reinterpret_cast<uint64_t>(addresses.data()); args.valid_table = reinterpret_cast<uint64_t>(valid.data());
    args.indices = reinterpret_cast<uint64_t>(indices.data()); args.offsets = reinterpret_cast<uint64_t>(offsets.data());
    args.output_indices = reinterpret_cast<uint64_t>(rows.data()); args.miss_count = reinterpret_cast<uint64_t>(&count);
    args.output = reinterpret_cast<uint64_t>(output.data()); args.core_status = reinterpret_cast<uint64_t>(reports.data());
    args.queue_state = reinterpret_cast<uint64_t>(queues.data()); args.channel_entity = reinterpret_cast<uint64_t>(entities.data());
    args.table_capacity = source.size(); args.request_capacity = capacity; args.output_rows = capacity;
    args.token_bytes = args.tokens_per_block = 1; args.block_count = cores; args.jettys_per_core = jettys;
    args.batch_get_max_inflight_total = budget;
  }
  void CheckOutput(uint32_t n) {
    for (uint32_t i = 0; i < output.size(); ++i)
      CHECK(output[rows[i]] == (i < n ? source[indices[i]] : 0xa5));
    CheckEvents();
  }
  void Run(int32_t n, uint64_t expected = 0) {
    count = n; std::fill(output.begin(), output.end(), 0xa5);
    for (uint32_t c = 0; c < args.block_count; ++c) {
      aiv_model::core = c; KvafExecuteBatchGetAivKernelV3(args);
      CHECK(reports[c].magic == KVAF_AIV_BATCH_GET_STATUS_MAGIC); CHECK(reports[c].result == expected);
      if (!expected) {
        CHECK(reports[c].profile.submitted_group_count == reports[c].profile.completed_group_count);
        if (args.batch_get_max_inflight_total && n > args.batch_get_max_inflight_total)
          CHECK(reports[c].profile.peak_submitted_not_drained <= kvaf_batch_get::CoreBudget(
              args.batch_get_max_inflight_total, args.block_count, c));
      }
    }
    if (expected) return;
    CheckOutput(n);
    for (const auto& q : aiv_model::hardware) {
      CHECK(q.producer == q.completed); CHECK(q.cq_producer == q.cq_consumer); CHECK(q.checkpoints.empty());
    }
  }
  void RunUbmem(int32_t n, uint64_t expected = 0) {
    count = n; std::fill(output.begin(), output.end(), 0xa5);
    std::vector<kvaf::ubmem::CoreStatus> statuses(args.block_count);
    kvaf::ubmem::BatchArgs a{};
    a.remote_base = a.mapped_base = reinterpret_cast<uint64_t>(source.data()); a.remote_bytes = source.size();
    a.local_base = args.output; a.local_bytes = output.size(); a.addresses = args.address_table;
    a.valid = args.valid_table; a.indices = args.indices; a.offsets = args.offsets; a.output_indices = args.output_indices;
    a.miss_count = args.miss_count; a.statuses = reinterpret_cast<uint64_t>(statuses.data());
    a.blocks = args.block_count; a.capacity = args.request_capacity; a.table_entries = args.table_capacity;
    a.token_bytes = a.tokens_per_block = 1; a.profile = 2;
    for (uint32_t c = 0; c < a.blocks; ++c) {
      aiv_model::core = c; KvafExecuteBatchGetUbmemKernelV2(a);
      CHECK(statuses[c].result == expected);
      if (!expected) CHECK(statuses[c].completed == uint64_t(n) * (c + 1) / a.blocks - uint64_t(n) * c / a.blocks);
    }
    if (!expected) CheckOutput(n); else CheckEvents();
  }
};

int main() {
  for (const auto shape : {std::array<uint32_t, 3>{1, 1, 0}, {2, 2, 0}, {3, 3, 0},
                            {2, 2, 2048}, {3, 3, 2049}}) {
    Fixture f(200000, shape[0], shape[1], shape[2]);
    for (int32_t n : {0, 1, 127, 128, 129, 2048, 2049, 32767, 32768, 32769, 65536, 65537,
                       130814, 130815, 200000}) f.Run(n);
    f.Run(200000); f.Run(1); f.Run(0);
  }
  { Fixture f(1000000, 2, 2, 0); f.Run(1000000); }
  { Fixture f(200000, 1, 2, 0, 1); f.Run(200000); f.Run(200000); }
  { Fixture f(65536, 2, 2, 0); f.Run(65536); for (const auto& r : f.reports) CHECK(r.profile.submitted_group_count == 1); }
  // An explicit unsafe budget retains the previous fail-fast behavior.
  { Fixture f(65536, 1, 1, 65536); const auto before = aiv_model::doorbells;
    f.Run(65536, kQueueFull); CHECK(before == aiv_model::doorbells); }
  for (bool timeout : {false, true}) {
    Fixture f(200000, 1, 2, 0);
    aiv_model::hardware[1].timeout = timeout; aiv_model::hardware[1].error = !timeout;
    f.Run(200000, timeout ? kCqTimeout : kCqError); CHECK(f.queues[0].poisoned && f.queues[1].poisoned);
    const auto before = aiv_model::doorbells; f.Run(1, kInvalidQueue); CHECK(before == aiv_model::doorbells);
  }
  { Fixture f(1, 1, 1, 0); f.Run(-1, kInvalidArgs); f.Run(2, kInvalidArgs); }
  { Fixture f(1025, 3, 3, 7); f.Run(1025); }
  { Fixture f(200000, 3, 1, 0);
    for (int32_t n : {0, 1, 31, 32, 33, 65536, 65537, 200000}) f.RunUbmem(n);
    f.RunUbmem(-1, 1); f.RunUbmem(200001, 1);
    // Every core reports only errors in its own assigned slice.
    f.args.block_count = 1;
    f.rows[0] = f.output.size(); f.RunUbmem(1, 3);
    f.rows[0] = f.output.size() - 1; f.indices[0] = f.source.size(); f.RunUbmem(1, 2);
  }
  std::cout << "Direct Get: queue windows, large batches, URMA rings/CQ faults and UBMEM bounds passed\n";
}
