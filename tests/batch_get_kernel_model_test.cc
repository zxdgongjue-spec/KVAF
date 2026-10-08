#include "kernel_operator.h"
#include "batch_get_abi.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <vector>

// The Ascend metadata section is not a native Mach-O/ELF section. All device
// code otherwise comes directly from the production source, without a copy.
#define __attribute__(...)
#include "../access_operator/demo/aiv_batch_get/kernel/execute_batch_get_aiv.asc"
#undef __attribute__

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

namespace aiv_model {
struct Queue {
  std::vector<uint32_t> sq = std::vector<uint32_t>(32768 * 16);
  alignas(64) std::array<uint32_t, 8 * 16> cq{};
  uint32_t sq_db = 0, cq_db = 0;
  uint64_t producer = 0, completed = 0, cq_producer = 0, cq_consumer = 0;
  std::deque<uint64_t> checkpoints;
  bool timeout = false, error = false;
};
std::vector<Queue> hardware;
uint64_t doorbells = 0;

void Doorbell(uint32_t value, uint32_t* address) {
  for (auto& queue : hardware) {
    if (address == &queue.cq_db) {
      CHECK(queue.cq_consumer < queue.cq_producer);
      ++queue.cq_consumer;
      CHECK(value == (queue.cq_consumer & 0x003fffffU));
      queue.cq_db = value;
      return;
    }
    if (address != &queue.sq_db) continue;
    ++doorbells;
    const uint32_t added = value - queue.sq_db;
    CHECK(added > 0 && added <= 128);
    CHECK(queue.producer + added - queue.completed < 32768);
    for (uint32_t i = 0; i < added; ++i) {
      const uint64_t pi = queue.producer++;
      const uint32_t* wqe = queue.sq.data() + (pi % 32768) * 16;
      CHECK((wqe[0] >> 31U) == 1U - ((pi / 32768) & 1U));
      const bool signal = (wqe[0] & (1U << 21U)) != 0;
      CHECK(((wqe[0] >> 16U) & 3U) == (signal ? 2U : 1U));
      if (signal) queue.checkpoints.push_back(queue.producer);
    }
    queue.sq_db = value;
    return;
  }
  CHECK(false);
}

void Poll(void* address) {
  const auto probe = reinterpret_cast<uintptr_t>(address);
  for (auto& queue : hardware) {
    const auto begin = reinterpret_cast<uintptr_t>(queue.cq.data());
    if (probe < begin || probe >= begin + sizeof(queue.cq)) continue;
    // Deliver a checkpoint only when polled. This forces the initial two
    // groups to coexist and leaves the younger group outstanding during refill.
    if (queue.timeout || queue.checkpoints.empty() ||
        queue.cq_consumer != queue.cq_producer) return;
    const uint64_t end = queue.checkpoints.front();
    queue.checkpoints.pop_front();
    while (queue.completed < end) {
      const uint32_t* wqe = queue.sq.data() + (queue.completed++ % 32768) * 16;
      const uint64_t source = wqe[10] | (uint64_t(wqe[11]) << 32U);
      const uint64_t destination = wqe[14] | (uint64_t(wqe[15]) << 32U);
      std::memcpy(reinterpret_cast<void*>(destination),
                  reinterpret_cast<void*>(source), wqe[12]);
    }
    const uint32_t owner = 1U - ((queue.cq_producer / 8) & 1U);
    queue.cq[(queue.cq_producer % 8) * 16] =
        (owner << 2U) | (queue.error ? 1U << 24U : 0U);
    ++queue.cq_producer;
    return;
  }
  CHECK(false);
}
}  // namespace aiv_model

struct Fixture {
  uint32_t capacity, cores, jettys;
  std::array<uint8_t, 16 * 16> remote;
  std::array<uint64_t, 16> addresses;
  std::array<uint8_t, 16> valid;
  std::vector<uint8_t> output;
  std::vector<int32_t> indices, offsets, rows;
  std::vector<KvafBatchGetQueueState> queues;
  std::vector<KvafBatchGetDirectUrmaEntity> entities;
  std::vector<KvafBatchGetCoreStatus> statuses;
  std::vector<uint64_t> raw_remote, raw_local;
  bool raw;
  int32_t misses = 0;
  KvafExecuteBatchGetKernelArgs args{};

  Fixture(uint32_t count, uint32_t c, uint32_t j, uint32_t budget, bool raw_mode = false)
      : capacity(count), cores(c), jettys(j), output(count * 16),
        indices(count), offsets(count), rows(count), queues(c * j),
        entities(c * j), statuses(c), raw_remote(count), raw_local(count), raw(raw_mode) {
    aiv_model::hardware.clear();
    aiv_model::hardware.resize(c * j);
    for (uint32_t i = 0; i < remote.size(); ++i) remote[i] = (i * 7 + 1) % 251;
    for (uint32_t i = 0; i < 16; ++i) {
      addresses[i] = reinterpret_cast<uint64_t>(remote.data() + 16 * i);
      valid[i] = 1;
    }
    for (uint32_t i = 0; i < count; ++i) {
      indices[i] = i % 16;
      rows[i] = count - 1 - i;
    }
    for (uint32_t i = 0; i < c * j; ++i) {
      auto& entity = entities[i];
      auto& hardware = aiv_model::hardware[i];
      entity.magic = 0x4b564155U;
      entity.version = 1;
      entity.remote_host_ddr_address = reinterpret_cast<uint64_t>(remote.data());
      entity.remote_host_ddr_size = remote.size();
      entity.local_hbm_address = reinterpret_cast<uint64_t>(output.data());
      entity.local_hbm_size = output.size();
      entity.local_token_id = 1;
      entity.sq_va = reinterpret_cast<uint64_t>(hardware.sq.data());
      entity.sq_db_va = reinterpret_cast<uint64_t>(&hardware.sq_db);
      entity.sq_depth = 32768;
      entity.wqe_size = 64;
      entity.cq_va = reinterpret_cast<uint64_t>(hardware.cq.data());
      entity.cq_db_va = reinterpret_cast<uint64_t>(&hardware.cq_db);
      entity.cq_depth = 8;
      entity.cqe_size = 64;
    }
    for (auto& status : statuses) status.profile.magic = KVAF_AIV_BATCH_GET_PROFILE_MAGIC;
    args.address_table = reinterpret_cast<uint64_t>(addresses.data());
    args.valid_table = reinterpret_cast<uint64_t>(valid.data());
    args.indices = reinterpret_cast<uint64_t>(indices.data());
    args.offsets = reinterpret_cast<uint64_t>(offsets.data());
    args.output_indices = reinterpret_cast<uint64_t>(rows.data());
    args.miss_count = reinterpret_cast<uint64_t>(&misses);
    args.output = reinterpret_cast<uint64_t>(output.data());
    args.channel_entity = reinterpret_cast<uint64_t>(entities.data());
    args.queue_state = reinterpret_cast<uint64_t>(queues.data());
    args.core_status = reinterpret_cast<uint64_t>(statuses.data());
    args.table_capacity = 16;
    args.request_capacity = args.output_rows = count;
    args.token_bytes = 16;
    args.tokens_per_block = 1;
    args.block_count = c;
    args.jettys_per_core = j;
    args.batch_get_max_inflight_total = budget;
    if (raw) {
      // Neither transfer endpoint is a token row/block boundary. The kernel
      // must preserve the supplied addresses byte-for-byte.
      args.token_bytes = 7;
      for (uint32_t i = 0; i < count; ++i) {
        raw_remote[i] = addresses[indices[i]] + 3;
        raw_local[i] = args.output + rows[i] * 16 + 1;
      }
      args.remote_addresses = reinterpret_cast<uint64_t>(raw_remote.data());
      args.local_addresses = reinterpret_cast<uint64_t>(raw_local.data());
      args.address_table = args.valid_table = args.indices = args.offsets = 0;
      args.output_indices = args.table_capacity = args.output_rows = args.tokens_per_block = 0;
    }
  }

  void Run(uint32_t count, uint64_t expected = 0) {
    misses = count;
    std::fill(output.begin(), output.end(), 0);
    uint64_t cq_before = 0, cq_after = 0, profile_cq = 0;
    const bool profiling = statuses[0].profile.magic == KVAF_AIV_BATCH_GET_PROFILE_MAGIC;
    const auto before = queues;
    for (const auto& queue : queues) cq_before += queue.cq_consumer_index;
    for (uint32_t c = 0; c < cores; ++c) {
      aiv_model::core = c;
      KvafExecuteBatchGetAivKernelV3(args);
      CHECK(statuses[c].result == expected);
      if (expected || !profiling) continue;
      const auto& profile = statuses[c].profile;
      CHECK(profile.submitted_group_count == profile.completed_group_count);
      CHECK(profile.kernel_start_cycle <= profile.setup_done_cycle);
      CHECK(profile.setup_done_cycle <= profile.post_done_cycle);
      CHECK(profile.post_done_cycle <= profile.completion_done_cycle);
      CHECK(profile.completion_done_cycle <= profile.kernel_finish_cycle);
      profile_cq += profile.completion_count;
      if (args.batch_get_max_inflight_total && count > args.batch_get_max_inflight_total) {
        CHECK(profile.peak_submitted_not_drained <=
              kvaf_batch_get::CoreBudget(args.batch_get_max_inflight_total, cores, c));
      }
    }
    if (expected) return;
    std::vector<uint8_t> expected_output(output.size(), 0);
    for (uint32_t i = 0; i < count; ++i) {
      const uint64_t remote_address = raw ? raw_remote[i] : addresses[indices[i]] + offsets[i];
      const uint64_t local_address = raw ? raw_local[i] : args.output + rows[i] * args.token_bytes;
      std::memcpy(expected_output.data() + (local_address - args.output),
                  reinterpret_cast<const void*>(remote_address), args.token_bytes);
    }
    CHECK(output == expected_output);
    for (uint32_t i = 0; i < cores * jettys; ++i) {
      const auto& hardware = aiv_model::hardware[i];
      CHECK(hardware.checkpoints.empty());
      CHECK(hardware.producer == hardware.completed);
      CHECK(hardware.cq_producer == hardware.cq_consumer);
      CHECK(queues[i].sq_producer_index == hardware.producer);
      CHECK(queues[i].cq_consumer_index == hardware.cq_consumer);
      CHECK(queues[i].completed_batch_count == before[i].completed_batch_count +
            (queues[i].sq_producer_index != before[i].sq_producer_index));
      cq_after += queues[i].cq_consumer_index;
    }
    if (profiling) CHECK(cq_after - cq_before == profile_cq);
    for (auto& source : model_events) {
      for (auto& target : source) {
        for (bool event : target) CHECK(!event);
      }
    }
  }
};

void RunMode(bool raw) {
  for (const auto& shape : {std::array<uint32_t, 3>{1, 1, 2048}, {1, 16, 2048},
                           {1, 64, 2048}, {4, 4, 2048}, {3, 3, 2049}, {1, 1, 2},
                           {1, 16, 0}}) {
    Fixture f(8193, shape[0], shape[1], shape[2], raw);
    for (int repeat = 0; repeat < 2; ++repeat) {
      for (uint32_t count : {0U, 1U, 127U, 128U, 129U, 1024U, 2047U,
                             2048U, 2049U, 3072U, 4096U, 8193U}) f.Run(count);
    }
  }
  { Fixture f(65537, 1, 1, 2048, raw); f.Run(65537); f.Run(1); f.Run(0); }
  { Fixture f(65537, 1, 1, 0, raw); f.Run(65537, kQueueFull); CHECK(aiv_model::hardware[0].producer == 0); }
  { Fixture f(2049, 1, 1, 1, raw); f.Run(2049, kInvalidArgs); }
  {
    Fixture f(65535, 1, 1, 65534, raw);
    f.Run(65535, kQueueFull);  // first group fits, second would overwrite SQ
    CHECK(aiv_model::hardware[0].producer == 32767);
    CHECK(f.queues[0].poisoned);
    f.Run(1, kInvalidQueue);
  }
  {
    Fixture f(8193, 1, 1, 2048, raw);
    f.statuses[0].profile.magic = 0;
    f.Run(2049);
    CHECK(f.statuses[0].profile.magic == 0);
  }
  {
    Fixture f(8193, 1, 1, 2048, raw);
    auto& q = f.queues[0];
    auto& h = aiv_model::hardware[0];
    q.magic = 0x4b564151U;
    q.initialized = 1;
    q.sq_producer_index = h.producer = h.completed = (1ULL << 32) - 64;
    h.sq_db = static_cast<uint32_t>(h.producer);
    q.cq_consumer_index = h.cq_producer = h.cq_consumer = (1ULL << 22) - 1;
    f.Run(8193);  // hardware doorbell counters wrap; software indices do not
  }
  for (bool timeout : {false, true}) {
    Fixture f(8193, 1, 2, 2048, raw);
    aiv_model::hardware[1].timeout = timeout;
    aiv_model::hardware[1].error = !timeout;
    f.Run(8193, timeout ? kCqTimeout : kCqError);
    CHECK(f.queues[0].poisoned && f.queues[1].poisoned);
    CHECK(f.statuses[0].profile.submitted_group_count == 2);
    CHECK(f.statuses[0].profile.completed_group_count == 0);
    const uint64_t before = aiv_model::doorbells;
    f.Run(8193, kInvalidQueue);
    CHECK(aiv_model::doorbells == before);
  }
  if (raw) {
    Fixture f(257, 1, 2, 256, true);
    f.Run(257);
    // Replay with changed source/destination contents at fixed array addresses.
    std::reverse(f.raw_remote.begin(), f.raw_remote.end());
    f.Run(257);
    std::reverse(f.raw_local.begin(), f.raw_local.end());
    f.Run(257);
    f.raw_remote.back() = f.raw_local.back() = UINT64_MAX;
    f.Run(256);  // ignored suffix must not be dereferenced
    f.Run(0);
    f.Run(258, kInvalidArgs);
    f.Run(UINT32_MAX, kInvalidArgs);
    const uint64_t before = aiv_model::doorbells;
    f.args.local_addresses = 0;
    f.Run(1, kInvalidArgs);
    f.args.local_addresses = reinterpret_cast<uint64_t>(f.raw_local.data());
    f.args.indices = reinterpret_cast<uint64_t>(f.indices.data());
    f.Run(1, kInvalidArgs);  // reject mixed addressing modes
    CHECK(aiv_model::doorbells == before);
  }
}

int main() {
  RunMode(false);
  RunMode(true);
  std::cout << "BatchGet kernel CPU model: payload, CQ/SQ wrap, tail, replay and faults passed\n";
}
