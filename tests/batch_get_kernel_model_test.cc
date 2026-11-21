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
#include "../access_operator/demo/aiv_batch_get/kernel/execute_batch_get_ranges_aiv.asc"
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
  uint32_t delay_polls = 0;
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
    if (queue.delay_polls) { --queue.delay_polls; return; }
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
  int32_t misses = 0;
  KvafExecuteBatchGetKernelArgs args{};

  Fixture(uint32_t count, uint32_t c, uint32_t j, uint32_t budget)
      : capacity(count), cores(c), jettys(j), output(count * 16),
        indices(count), offsets(count), rows(count), queues(c * j),
        entities(c * j), statuses(c) {
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
      KvafExecuteBatchGetAivKernelV2(args);
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
    for (uint32_t i = 0; i < capacity; ++i) {
      for (uint32_t byte = 0; byte < 16; ++byte) {
        CHECK(output[rows[i] * 16 + byte] == (i < count ? remote[indices[i] * 16 + byte] : 0));
      }
    }
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

void CheckRangeKernelModel() {
  // Exercise the production range kernel against the same delayed-DMA model.
  for (uint32_t cores : {1U, 2U, 4U}) for (uint32_t jettys : {1U, 2U, 4U}) {
    constexpr uint32_t capacity = 257;
    Fixture f(capacity, cores, jettys, 2 * cores);
    std::vector<uint8_t> packed(256 * 1024);
    for (std::size_t i = 0; i < packed.size(); ++i) packed[i] = (i * 19 + i / 997) % 251;
    std::vector<int64_t> starts(capacity), sizes(capacity), targets(capacity);
    uint64_t bytes = 11;
    for (uint32_t i = 0; i < capacity; ++i) {
      sizes[i] = i % 17 == 0 ? 150 * 1024 : (i % 5 == 0 ? 65537 : 1 + i % 1968);
      starts[i] = i % 2 ? 512 : 510;  // aligned and unaligned ranges
      targets[i] = bytes;
      bytes += sizes[i] + 7;  // guard gaps prove tails don't write padding
      f.indices[i] = 0;
    }
    f.output.resize(bytes + 17);
    f.addresses[0] = reinterpret_cast<uint64_t>(packed.data());
    std::array<uint64_t, 16> table_lengths{};
    table_lengths[0] = packed.size();
    for (auto& e : f.entities) {
      e.remote_host_ddr_address = reinterpret_cast<uint64_t>(packed.data());
      e.remote_host_ddr_size = packed.size();
      e.local_hbm_address = reinterpret_cast<uint64_t>(f.output.data());
      e.local_hbm_size = f.output.size();
    }
    KvafExecuteBatchGetRangesKernelArgs a{};
    a.address_table = reinterpret_cast<uint64_t>(f.addresses.data());
    a.valid_table = reinterpret_cast<uint64_t>(f.valid.data());
    a.table_lengths = reinterpret_cast<uint64_t>(table_lengths.data());
    a.indices = reinterpret_cast<uint64_t>(f.indices.data());
    a.source_offsets = reinterpret_cast<uint64_t>(starts.data());
    a.lengths = reinterpret_cast<uint64_t>(sizes.data());
    a.request_count = reinterpret_cast<uint64_t>(&f.misses);
    a.output_offsets = reinterpret_cast<uint64_t>(targets.data());
    a.output = reinterpret_cast<uint64_t>(f.output.data());
    a.core_status = reinterpret_cast<uint64_t>(f.statuses.data());
    a.channel_entity = reinterpret_cast<uint64_t>(f.entities.data());
    a.queue_state = reinterpret_cast<uint64_t>(f.queues.data());
    a.table_capacity = 16;
    a.request_capacity = capacity;
    a.block_count = cores;
    a.jettys_per_core = jettys;
    // Tight budgets force one object to span multiple waves; zero and larger
    // budgets cover multi-Jetty posting and 128-WQE post boundaries.
    for (uint32_t chunk : {0U, 65536U, 1024U * 1024, UINT32_MAX}) {
      a.read_chunk_bytes = chunk;
      const uint64_t read_bytes = chunk ? chunk : UINT32_MAX;
      for (uint32_t budget : {2 * cores, 128 * cores, 0U}) {
        a.max_inflight_total = budget;
        for (uint32_t count : {0U, 1U, 3U, capacity, 0U, 129U}) {
          f.misses = count;
          std::fill(f.output.begin(), f.output.end(), 0xa5);
          auto expected = f.output;
          uint64_t reads = 0, expected_reads = 0, logical = 0;
          for (uint32_t i = 0; i < count; ++i) {
            std::memcpy(expected.data() + targets[i], packed.data() + starts[i], sizes[i]);
            expected_reads += (sizes[i] - 1) / read_bytes + 1;
          }
          for (uint32_t c = 0; c < cores; ++c) {
            aiv_model::core = c;
            KvafExecuteBatchGetRangesAivKernelV2(a);
            CHECK(f.statuses[c].result == 0);
            logical += f.statuses[c].processed_request_count;
          }
          for (const auto& q : f.queues) reads += q.reserved[0];
          CHECK(f.output == expected);
          CHECK(logical == count && reads == expected_reads);
          for (const auto& q : aiv_model::hardware)
            CHECK(q.producer == q.completed && q.checkpoints.empty());
        }
      }
    }
    if (cores != 1) continue;
    // Bad active values fail before posting; inactive garbage is never read.
    f.misses = 1;
    starts[1] = sizes[1] = targets[1] = -1;
    aiv_model::core = 0;
    KvafExecuteBatchGetRangesAivKernelV2(a);
    CHECK(f.statuses[0].result == 0);
    const auto before = aiv_model::doorbells;
    auto fail = [&](uint64_t code) {
      KvafExecuteBatchGetRangesAivKernelV2(a);
      CHECK(f.statuses[0].result == code);
      CHECK(aiv_model::doorbells == before);
    };
    f.misses = -1; fail(kInvalidArgs);
    f.misses = capacity + 1; fail(kInvalidArgs);
    f.misses = 1;
    starts[0] = -1; fail(kInvalidLookup);
    starts[0] = INT64_MAX; fail(kInvalidLookup);
    starts[0] = 0;
    const auto length = sizes[0];
    sizes[0] = 0; fail(kInvalidLookup);
    sizes[0] = INT64_MAX; fail(kInvalidLookup);
    sizes[0] = length;
    targets[0] = INT64_MAX; fail(kInvalidOutput);
    targets[0] = -1; fail(kInvalidOutput);
    targets[0] = 0;
    f.indices[0] = 16; fail(kInvalidLookup);
    f.indices[0] = 0;
    f.valid[0] = 0; fail(kInvalidLookup);
    f.valid[0] = 1;
    table_lengths[0] = 4; fail(kInvalidLookup);
    table_lengths[0] = packed.size();
    // A CQ fault poisons every submitted queue; subsequent launches cannot
    // accidentally reuse live requests or stale completion records.
    aiv_model::hardware[0].error = true;
    KvafExecuteBatchGetRangesAivKernelV2(a);
    CHECK(f.statuses[0].result == kCqError && f.queues[0].poisoned);
    const auto error_posts = aiv_model::doorbells;
    KvafExecuteBatchGetRangesAivKernelV2(a);
    CHECK(f.statuses[0].result == kInvalidQueue && aiv_model::doorbells == error_posts);
  }
  // Large tensors use one READ by default. Explicit smaller chunks must resume
  // across waves/posts and preserve payload, including SQ/CQ owner wrap.
  for (const int64_t bytes : {int64_t{1024 * 1024}, int64_t{32 * 1024 * 1024 + 17}})
  for (uint32_t chunk : {0U, 65536U, 1024U * 1024, 4U * 1024 * 1024, UINT32_MAX})
  for (uint32_t budget : {2U, 128U, 0U}) {
    Fixture f(1, 1, 1, budget);
    std::vector<uint8_t> source(bytes);
    for (int64_t i = 0; i < bytes; ++i) source[i] = (i * 7 + i / 997) % 251;
    f.output.resize(bytes + 32, 0xa5);
    f.addresses[0] = reinterpret_cast<uint64_t>(source.data());
    f.indices[0] = 0;
    f.misses = 1;
    int64_t zero = 0;
    uint64_t table_length = bytes;
    auto& e = f.entities[0];
    e.remote_host_ddr_address = f.addresses[0];
    e.remote_host_ddr_size = bytes;
    e.local_hbm_address = reinterpret_cast<uint64_t>(f.output.data());
    e.local_hbm_size = f.output.size();
    auto& q = f.queues[0];
    q.magic = kQueueMagic; q.initialized = 1;
    q.sq_producer_index = 32765; q.cq_consumer_index = 7;
    auto& hw = aiv_model::hardware[0];
    hw.producer = hw.completed = hw.sq_db = 32765;
    hw.cq_producer = hw.cq_consumer = hw.cq_db = 7;
    // Entries 0..6 belong to the consumed first epoch. Leaving them zero
    // would falsely look like fresh second-epoch CQEs during timeout checks.
    for (uint32_t slot = 0; slot < 7; ++slot) hw.cq[slot * 16] = 1U << 2U;
    hw.delay_polls = kMaxCqPolls + 17;  // must outlast the old fixed allowance
    KvafExecuteBatchGetRangesKernelArgs a{};
    a.address_table = reinterpret_cast<uint64_t>(f.addresses.data());
    a.valid_table = reinterpret_cast<uint64_t>(f.valid.data());
    a.table_lengths = reinterpret_cast<uint64_t>(&table_length);
    a.indices = reinterpret_cast<uint64_t>(f.indices.data());
    a.source_offsets = a.output_offsets = reinterpret_cast<uint64_t>(&zero);
    a.lengths = reinterpret_cast<uint64_t>(&bytes);
    a.request_count = reinterpret_cast<uint64_t>(&f.misses);
    a.output = e.local_hbm_address;
    a.core_status = reinterpret_cast<uint64_t>(f.statuses.data());
    a.channel_entity = reinterpret_cast<uint64_t>(f.entities.data());
    a.queue_state = reinterpret_cast<uint64_t>(f.queues.data());
    a.table_capacity = a.request_capacity = a.block_count = a.jettys_per_core = 1;
    a.max_inflight_total = budget;
    a.read_chunk_bytes = chunk;
    const uint64_t read_bytes = chunk ? chunk : UINT32_MAX;
    const uint64_t expected_reads = (bytes - 1) / read_bytes + 1;
    aiv_model::core = 0;
    KvafExecuteBatchGetRangesAivKernelV2(a);
    CHECK(f.statuses[0].result == 0 && f.statuses[0].processed_request_count == 1);
    CHECK(q.reserved[0] == expected_reads && q.sq_producer_index == 32765 + expected_reads);
    CHECK(std::equal(source.begin(), source.end(), f.output.begin()));
    CHECK(std::all_of(f.output.begin() + bytes, f.output.end(), [](uint8_t b) { return b == 0xa5; }));
    CHECK(hw.producer == hw.completed && hw.cq_producer == hw.cq_consumer);
    if (bytes == 1024 * 1024 && chunk == 0 && budget == 2) {
      hw.timeout = true;
      KvafExecuteBatchGetRangesAivKernelV2(a);
      CHECK(f.statuses[0].result == kCqTimeout && q.poisoned);
    }
  }
  std::cout << "RANGE_GET_MODEL_PASS: 1MiB/32MiB tensors, one READ by default, configurable chunks, payload/guards, READ counts, prefixes, bounds, SQ/CQ wrap and CQ faults\n";
}

int main() {
  CheckRangeKernelModel();
  for (const auto& shape : {std::array<uint32_t, 3>{1, 1, 2048}, {1, 16, 2048},
                           {1, 64, 2048}, {4, 4, 2048}, {3, 3, 2049}, {1, 1, 2},
                           {1, 16, 0}}) {
    Fixture f(8193, shape[0], shape[1], shape[2]);
    for (int repeat = 0; repeat < 2; ++repeat) {
      for (uint32_t count : {0U, 1U, 127U, 128U, 129U, 1024U, 2047U,
                             2048U, 2049U, 3072U, 4096U, 8193U}) f.Run(count);
    }
  }
  { Fixture f(65537, 1, 1, 2048); f.Run(65537); f.Run(1); f.Run(0); }
  { Fixture f(65537, 1, 1, 0); f.Run(65537, kQueueFull); CHECK(aiv_model::hardware[0].producer == 0); }
  { Fixture f(2049, 1, 1, 1); f.Run(2049, kInvalidArgs); }
  {
    Fixture f(65535, 1, 1, 65534);
    f.Run(65535, kQueueFull);  // first group fits, second would overwrite SQ
    CHECK(aiv_model::hardware[0].producer == 32767);
    CHECK(f.queues[0].poisoned);
    f.Run(1, kInvalidQueue);
  }
  {
    Fixture f(8193, 1, 1, 2048);
    f.statuses[0].profile.magic = 0;
    f.Run(2049);
    CHECK(f.statuses[0].profile.magic == 0);
  }
  {
    Fixture f(8193, 1, 1, 2048);
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
    Fixture f(8193, 1, 2, 2048);
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
  std::cout << "BatchGet kernel CPU model: payload, CQ/SQ wrap, tail, replay and faults passed\n";
}
