#include "kernel_operator.h"
#include "batch_get_abi.h"

#include <algorithm>
#include <array>
#include <cstddef>
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
  uint32_t defer_polls = 0;
  uint32_t tp_id = 0, remote_token_id = 0, remote_token_value = 0, local_token_id = 0;
  std::array<uint32_t, 4> remote_eid{};
};
std::vector<Queue> hardware;
uint64_t doorbells = 0;
enum class EventKind { kVector, kSqCopy, kSqDoorbell, kCqProbe };
struct Event { EventKind kind; uint32_t queue; };
std::vector<Event> trace;

void VectorCall() {
  ++vector_calls;
  trace.push_back({EventKind::kVector, UINT32_MAX});
}

void SqCopy(void* address, uint64_t bytes) {
  const auto destination = reinterpret_cast<uintptr_t>(address);
  for (uint32_t i = 0; i < hardware.size(); ++i) {
    const auto begin = reinterpret_cast<uintptr_t>(hardware[i].sq.data());
    const auto end = begin + hardware[i].sq.size() * sizeof(uint32_t);
    if (destination >= begin && destination < end) {
      CHECK(destination + bytes <= end);
      trace.push_back({EventKind::kSqCopy, i});
      return;
    }
  }
  CHECK(false);
}

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
    trace.push_back({EventKind::kSqDoorbell,
                     static_cast<uint32_t>(&queue - hardware.data())});
    const uint32_t added = value - queue.sq_db;
    CHECK(added > 0 && added <= 128);
    CHECK(queue.producer + added - queue.completed < 32768);
    for (uint32_t i = 0; i < added; ++i) {
      const uint64_t pi = queue.producer++;
      const uint32_t* wqe = queue.sq.data() + (pi % 32768) * 16;
      CHECK((wqe[0] >> 31U) == 1U - ((pi / 32768) & 1U));
      const bool signal = (wqe[0] & (1U << 21U)) != 0;
      CHECK(((wqe[0] >> 16U) & 3U) == (signal ? 2U : 1U));
      CHECK(wqe[1] == (0x6U << 8U));
      CHECK(wqe[2] == ((queue.tp_id & 0x00ffffffU) | 0x01000000U));
      CHECK(wqe[3] == (queue.remote_token_id & 0x000fffffU));
      for (uint32_t word = 0; word < 4; ++word) {
        CHECK(wqe[4 + word] == queue.remote_eid[word]);
      }
      CHECK(wqe[8] == queue.remote_token_value);
      CHECK(wqe[13] == queue.local_token_id);
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
    trace.push_back({EventKind::kCqProbe,
                     static_cast<uint32_t>(&queue - hardware.data())});
    if (queue.defer_polls != 0) { --queue.defer_polls; return; }
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
    aiv_model::trace.clear();
    aiv_model::vector_calls = 0;
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
      entity.tp_id = hardware.tp_id = 0x100U + i;
      entity.remote_token_id = hardware.remote_token_id = 11;
      entity.remote_token_value = hardware.remote_token_value = 22;
      entity.local_token_id = hardware.local_token_id = 0x200U + i;
      for (uint32_t byte = 0; byte < 16; ++byte) {
        entity.remote_eid[byte] = static_cast<uint8_t>(i * 17 + byte);
      }
      std::memcpy(hardware.remote_eid.data(), entity.remote_eid, 16);
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
    const bool profiling = statuses[0].profile.magic == KVAF_AIV_BATCH_GET_PROFILE_MAGIC ||
        statuses[0].profile.magic == KVAF_AIV_BATCH_GET_JETTY_TIMELINE_MAGIC;
    const auto before = queues;
    for (const auto& queue : queues) cq_before += queue.cq_consumer_index;
    for (uint32_t c = 0; c < cores; ++c) {
      aiv_model::core = c;
      const uint64_t doorbells_before = aiv_model::doorbells;
      KvafExecuteBatchGetAivKernelV3(args);
      CHECK(statuses[c].result == expected);
      if (expected || !profiling) continue;
      const auto& profile = statuses[c].profile;
      CHECK(profile.total_posts == aiv_model::doorbells - doorbells_before);
      CHECK(profile.profile_version == 3);
      CHECK(profile.small_post_max_requests == kvaf_batch_get::kSmallPostMaxRequests);
      uint64_t read_sum = 0, post_sum = 0, checkpoint_sum = 0;
      for (uint32_t j = 0; j < KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE; ++j) {
        const auto& trace = profile.jettys[j];
        read_sum += trace.submitted_reads;
        post_sum += trace.submitted_posts;
        checkpoint_sum += trace.completed_checkpoints;
        if (j < jettys) {
          CHECK(trace.submitted_reads == queues[c * jettys + j].sq_producer_index -
                before[c * jettys + j].sq_producer_index);
        }
        if (trace.submitted_reads == 0) {
          CHECK(trace.first_doorbell_cycle == 0 && trace.last_cqe_seen_cycle == 0);
          CHECK(trace.submitted_posts == 0 && trace.completed_checkpoints == 0);
        } else {
          CHECK(trace.first_doorbell_cycle >= profile.kernel_start_cycle);
          CHECK(trace.first_doorbell_cycle <= trace.last_doorbell_cycle);
          CHECK(trace.last_doorbell_cycle <= trace.last_cqe_seen_cycle);
          CHECK(trace.first_cqe_seen_cycle <= trace.last_cqe_seen_cycle);
          CHECK(trace.last_cqe_seen_cycle <= profile.completion_done_cycle);
        }
      }
      CHECK(read_sum == profile.active_requests);
      CHECK(post_sum == profile.total_posts);
      CHECK(checkpoint_sum == profile.completion_count);
      CHECK(profile.submitted_group_count == profile.completed_group_count);
      CHECK(profile.kernel_start_cycle <= profile.setup_done_cycle);
      CHECK(profile.setup_done_cycle <= profile.post_done_cycle);
      CHECK(profile.post_done_cycle <= profile.completion_done_cycle);
      CHECK(profile.completion_done_cycle <= profile.kernel_finish_cycle);
      if (profile.active_requests != 0) {
        CHECK(profile.first_doorbell_cycle >= profile.kernel_start_cycle);
        CHECK(profile.first_doorbell_cycle <= profile.post_done_cycle);
      } else {
        CHECK(profile.first_doorbell_cycle == 0);
      }
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

int main() {
  {
    Fixture f(257, 2, 2, 2048);
    f.Run(32);
    const bool small_posts = kvaf_batch_get::kSmallPostMaxRequests >= 32;
    CHECK(aiv_model::vector_calls == 2);  // one fill per core, even with two Jettys
    for (uint32_t c = 0; c < 2; ++c) {
      CHECK(f.statuses[c].profile.total_posts == (small_posts ? 2 : 1));
      CHECK(f.statuses[c].profile.completion_count == (small_posts ? 2 : 1));
      CHECK(f.statuses[c].profile.small_post_group_count == (small_posts ? 1 : 0));
      CHECK(f.statuses[c].profile.fused_group_count == (small_posts ? 1 : 0));
      CHECK(f.statuses[c].profile.wqe_build_calls == 1);
      CHECK(f.queues[c * 2].sq_producer_index == (small_posts ? 8 : 16));
      CHECK(f.queues[c * 2 + 1].sq_producer_index == (small_posts ? 8 : 0));
    }
    if (small_posts) {
      for (uint32_t c = 0; c < 2; ++c) {
        std::size_t first_doorbell = aiv_model::trace.size();
        std::size_t second_copy = aiv_model::trace.size();
        for (std::size_t i = 0; i < aiv_model::trace.size(); ++i) {
          const auto& event = aiv_model::trace[i];
          if (event.kind == aiv_model::EventKind::kSqDoorbell && event.queue == c * 2) {
            first_doorbell = i;
          }
          if (event.kind == aiv_model::EventKind::kSqCopy && event.queue == c * 2 + 1) {
            second_copy = i;
          }
        }
        CHECK(first_doorbell < aiv_model::trace.size());
        CHECK(second_copy < aiv_model::trace.size());
        CHECK(first_doorbell < second_copy);
      }
    }
    // Replay on the same queues while changing both the request count and
    // the selected post width, including the experiment's default cutoff.
    for (uint32_t count : {0U, 1U, 31U, 33U, 64U, 127U, 128U, 129U,
                           255U, 256U, 257U, 32U}) f.Run(count);
  }
  for (const auto& shape : {std::array<uint32_t, 2>{2, 4}, {3, 3}, {1, 64}}) {
    Fixture f(513, shape[0], shape[1], 2048);
    for (uint32_t count : {0U, 1U, 3U, 31U, 32U, 33U, 128U, 255U,
                           256U, 257U, 512U, 513U}) f.Run(count);
  }
  {
    Fixture f(257, 1, 4, 4);
    for (uint32_t count : {32U, 33U, 255U, 256U, 257U, 1U, 0U}) f.Run(count);
  }
  {
    Fixture f(257, 2, 2, 2048);
    for (uint32_t i = 0; i < 4; ++i) {
      auto& q = f.queues[i];
      auto& h = aiv_model::hardware[i];
      q.magic = 0x4b564151U;
      q.initialized = 1;
      const uint64_t heads[] = {32764, 65534, 131069, 0};
      q.sq_producer_index = h.producer = h.completed = heads[i];
      h.sq_db = static_cast<uint32_t>(heads[i]);
      q.cq_consumer_index = h.cq_producer = h.cq_consumer = 7;
    }
    for (int replay = 0; replay < 20; ++replay) f.Run(33);
  }
  if (kvaf_batch_get::kSmallPostMaxRequests >= 32) {
    for (bool timeout : {false, true}) {
      Fixture f(257, 1, 2, 2048);
      aiv_model::hardware[1].timeout = timeout;
      aiv_model::hardware[1].error = !timeout;
      f.Run(32, timeout ? kCqTimeout : kCqError);
      CHECK(f.queues[0].poisoned && f.queues[1].poisoned);
      const uint64_t before = aiv_model::doorbells;
      f.Run(32, kInvalidQueue);
      CHECK(aiv_model::doorbells == before);
    }
  }
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
  {
    Fixture f(512, 1, 2, 2048);
    f.statuses[0].profile.magic = KVAF_AIV_BATCH_GET_JETTY_TIMELINE_MAGIC;
    aiv_model::hardware[0].defer_polls = 8;
    f.Run(512);
    const auto& p = f.statuses[0].profile;
    CHECK(p.jetty_timeline_enabled == 1);
    CHECK(p.jettys[0].submitted_reads == 256 && p.jettys[1].submitted_reads == 256);
    CHECK(p.jettys[1].first_cqe_seen_cycle < p.jettys[0].first_cqe_seen_cycle);
    CHECK(p.jettys[0].probe_active_cycles > 0 && p.jettys[1].probe_active_cycles > 0);
    CHECK(p.jetty_completion_wait_cycles[0] == 0);
    CHECK(p.fused_group_count == 0 && p.wqe_build_calls == 4);
    f.Run(0);  // no timestamps/counters leak from the previous replay
  }
  {
    Fixture f(513, 2, 2, 4);
    for (auto& status : f.statuses) status.profile.magic = KVAF_AIV_BATCH_GET_JETTY_TIMELINE_MAGIC;
    for (uint32_t n : {32U, 33U, 256U, 257U, 513U, 0U}) f.Run(n);
  }
  for (bool timeout : {false, true}) {
    Fixture f(512, 1, 2, 2048);
    f.statuses[0].profile.magic = KVAF_AIV_BATCH_GET_JETTY_TIMELINE_MAGIC;
    aiv_model::hardware[1].timeout = timeout;
    aiv_model::hardware[1].error = !timeout;
    f.Run(512, timeout ? kCqTimeout : kCqError);
    CHECK(f.queues[0].poisoned && f.queues[1].poisoned);
    const auto before = aiv_model::doorbells;
    f.Run(512, kInvalidQueue);
    CHECK(aiv_model::doorbells == before);
  }
  std::cout << "BatchGet kernel CPU model: small-post cutoff="
            << kvaf_batch_get::kSmallPostMaxRequests
            << ", payload, CQ/SQ wrap, tail, replay and faults passed\n";
}
