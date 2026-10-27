#include "kernel_operator.h"
#include "batch_put_internal.h"

#include <array>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <vector>

#define __attribute__(...)
#include "../access_operator/demo/aiv_batch_put/kernel/execute_batch_put_aiv.asc"
#undef __attribute__

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

namespace aiv_model {
struct Queue {
  std::array<uint32_t, 256 * 16> sq{};
  alignas(64) std::array<uint32_t, 8 * 16> cq{};
  uint32_t sq_db = 0, cq_db = 0;
  uint64_t producer = 0, completed = 0, cq_producer = 0;
  std::deque<uint64_t> signals;
};
std::vector<Queue> hardware;
void Doorbell(uint32_t value, uint32_t* address) {
  for (auto& queue : hardware) {
    if (address == &queue.cq_db) { queue.cq_db = value; return; }
    if (address != &queue.sq_db) continue;
    const uint32_t added = value - queue.sq_db;
    CHECK(added > 0 && added <= 32);
    for (uint32_t i = 0; i < added; ++i) {
      const auto* wqe = queue.sq.data() + (queue.producer++ % 256) * 16;
      if (wqe[0] & (1U << 21U)) queue.signals.push_back(queue.producer);
    }
    queue.sq_db = value;
    return;
  }
  CHECK(false);
}
void Poll(void* address) {
  const auto probe = reinterpret_cast<uintptr_t>(address);
  for (auto& queue : hardware) {
    const auto base = reinterpret_cast<uintptr_t>(queue.cq.data());
    if (probe < base || probe >= base + sizeof(queue.cq)) continue;
    if (queue.signals.empty()) return;
    const auto end = queue.signals.front();
    queue.signals.pop_front();
    while (queue.completed < end) {
      const auto* wqe = queue.sq.data() + (queue.completed++ % 256) * 16;
      const uint64_t target = wqe[10] | (uint64_t(wqe[11]) << 32U);
      const uint64_t source = wqe[14] | (uint64_t(wqe[15]) << 32U);
      std::memcpy(reinterpret_cast<void*>(target), reinterpret_cast<void*>(source), wqe[12]);
    }
    queue.cq[(queue.cq_producer % 8) * 16] =
        (1U - ((queue.cq_producer / 8) & 1U)) << 2U;
    ++queue.cq_producer;
    return;
  }
  CHECK(false);
}
}  // namespace aiv_model

void Run(uint32_t count, uint32_t cores, uint32_t jettys, bool profiling) {
  std::vector<uint8_t> source(count * 16), target(count * 16);
  for (std::size_t i = 0; i < source.size(); ++i) source[i] = (i * 17) % 251;
  std::vector<uint64_t> offsets(count), lengths(count), addresses(count);
  for (uint32_t i = 0; i < count; ++i) {
    offsets[i] = i * 16;
    lengths[i] = i % 16 + 1;
    addresses[i] = reinterpret_cast<uint64_t>(target.data() + i * 16);
  }
  aiv_model::hardware.clear();
  aiv_model::hardware.resize(cores * jettys);
  aiv_model::cycle = 0;
  std::vector<KvafBatchPutDirectUrmaEntity> entities(cores * jettys);
  std::vector<KvafBatchPutQueueState> queues(cores * jettys);
  std::vector<KvafBatchPutCoreStatus> statuses(cores);
  for (uint32_t i = 0; i < cores * jettys; ++i) {
    auto& e = entities[i];
    auto& q = aiv_model::hardware[i];
    e.magic = 0x4b564155U;
    e.version = 1;
    e.local_hbm_address = reinterpret_cast<uint64_t>(source.data());
    e.local_hbm_size = source.size();
    e.remote_host_ddr_address = reinterpret_cast<uint64_t>(target.data());
    e.remote_host_ddr_size = target.size();
    e.local_token_id = e.local_token_value = e.remote_token_id = e.remote_token_value = 1;
    e.sq_va = reinterpret_cast<uint64_t>(q.sq.data());
    e.sq_db_va = reinterpret_cast<uint64_t>(&q.sq_db);
    e.sq_depth = 256;
    e.wqe_size = 64;
    e.cq_va = reinterpret_cast<uint64_t>(q.cq.data());
    e.cq_db_va = reinterpret_cast<uint64_t>(&q.cq_db);
    e.cq_depth = 8;
    e.cqe_size = 64;
  }
  KvafExecuteBatchPutLaunchArgs args{};
  args.request = {reinterpret_cast<uint64_t>(source.data()),
      reinterpret_cast<uint64_t>(offsets.data()), reinterpret_cast<uint64_t>(lengths.data()),
      reinterpret_cast<uint64_t>(addresses.data()), reinterpret_cast<uint64_t>(statuses.data()),
      source.size(), count, cores};
  args.channel_entity = reinterpret_cast<uint64_t>(entities.data());
  args.queue_state = reinterpret_cast<uint64_t>(queues.data());
  args.jettys_per_core = jettys;
  args.profile_enabled = profiling;
  for (uint32_t core = 0; core < cores; ++core) {
    aiv_model::core = core;
    KvafExecuteBatchPutAivKernel(args);
    const auto& status = statuses[core];
    CHECK(status.result == 0);
    const uint64_t active = uint64_t(count) * (core + 1) / cores - uint64_t(count) * core / cores;
    CHECK(status.processed_request_count == active);
    const auto& p = status.profile;
    if (!profiling) { CHECK(p.magic == 0 && aiv_model::cycle == 0); continue; }
    CHECK(p.magic == KVAF_AIV_BATCH_PUT_PROFILE_MAGIC);
    CHECK(p.kernel_start_cycle <= p.setup_done_cycle);
    CHECK(p.setup_done_cycle <= p.post_done_cycle);
    CHECK(p.post_done_cycle <= p.completion_done_cycle);
    CHECK(p.completion_done_cycle <= p.kernel_finish_cycle);
    CHECK(p.total_posts == (active + 31) / 32);
    CHECK(p.used_jettys == std::min<uint64_t>(jettys, p.total_posts));
    if (active) {
      CHECK(p.setup_done_cycle <= p.first_doorbell_cycle);
      CHECK(p.first_doorbell_cycle < p.post_done_cycle);
      CHECK(p.post_done_cycle < p.completion_done_cycle);
    } else { CHECK(p.first_doorbell_cycle == 0); }
  }
  for (uint32_t row = 0; row < count; ++row) {
    for (uint32_t byte = 0; byte < 16; ++byte) {
      CHECK(target[row * 16 + byte] == (byte < lengths[row] ? source[row * 16 + byte] : 0));
    }
  }
}

int main() {
  for (bool profiling : {false, true}) {
    for (uint32_t rows : {1U, 32U, 33U, 66U, 129U}) {
      for (uint32_t cores : {1U, 2U, 4U}) {
        for (uint32_t jettys : {1U, 2U}) Run(rows, cores, jettys, profiling);
      }
    }
  }
  std::cout << "BatchPut actual kernel model: 60 shapes, profile on/off, payload and timelines PASS\n";
}
