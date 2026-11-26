#include "kernel_operator.h"
#include "batch_get_abi.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include <array>
#include <cstdio>
#include <vector>

// Compile the production kernel as host C++; omit only the device metadata section.
#define __attribute__(...)
#include "execute_batch_get_aiv.asc"
#undef __attribute__

namespace {
void Check(bool ok, const char* message) {
  if (!ok) { std::fprintf(stderr, "FAILED: %s\n", message); std::exit(1); }
}
constexpr uint64_t kCanonicalDdr = 0x100000;
std::array<uint32_t, 16> ddr;
struct Jetty {
  uint32_t fe{}, sq_db{}, cq_db{}, signals{};
  std::vector<uint32_t> sq = std::vector<uint32_t>(32768 * 16);
  alignas(64) std::array<uint32_t, 8 * 16> cq{};
  uint64_t registered_base() const { return kCanonicalDdr * (fe + 2); }
};
std::vector<Jetty> jettys;
}

namespace aiv_model {
void Poll(void*) {}
void Doorbell(uint32_t value, uint32_t* address) {
  for (auto& jetty : jettys) {
    if (address == &jetty.cq_db) { jetty.cq_db = value; return; }
    if (address != &jetty.sq_db) continue;
    for (uint32_t pi = jetty.sq_db; pi < value; ++pi) {
      const auto* words = jetty.sq.data() + (pi % 32768) * 16;
      Check((words[2] & 0xffffff) == jetty.fe + 30 && words[3] == jetty.fe + 10 &&
            words[8] == jetty.fe + 20 && words[13] == jetty.fe + 40, "path-specific TP and tokens");
      Check(words[4] == jetty.fe + 50, "path-specific WQE EID");
      const auto source = (uint64_t(words[11]) << 32) | words[10];
      const auto dest = (uint64_t(words[15]) << 32) | words[14];
      const auto offset = source - jetty.registered_base();
      Check(offset < sizeof(ddr) && offset % sizeof(uint32_t) == 0 && words[12] == sizeof(uint32_t),
            "FE translates the logical address to its own registered base");
      std::memcpy(reinterpret_cast<void*>(dest), &ddr[offset / sizeof(uint32_t)], words[12]);
      if (words[0] & (1U << 21U)) {
        const auto signal = jetty.signals++;
        jetty.cq[(signal % 8) * 16] = (1U - ((signal / 8) & 1U)) << 2U;
      }
    }
    jetty.sq_db = value;
    return;
  }
  Check(false, "unknown Jetty doorbell");
}
}

void Run(uint32_t cores, uint32_t per_core, uint32_t fes, const std::vector<unsigned>& expected,
         const std::vector<uint32_t>& core_fe_map = {}) {
  const auto workers = cores * per_core;
  const uint32_t count = workers * 128;
  jettys.clear();
  jettys.resize(workers);
  std::array<uint64_t, 16> addresses{};
  std::array<uint8_t, 16> valid{};
  for (unsigned i = 0; i < ddr.size(); ++i) {
    ddr[i] = 1000 + i;
    addresses[i] = kCanonicalDdr + i * sizeof(uint32_t);
    valid[i] = 1;
  }
  std::vector<int32_t> indices(count), offsets(count), rows(count);
  std::vector<uint32_t> output(count);
  for (unsigned i = 0; i < count; ++i) { indices[i] = i % ddr.size(); rows[i] = count - 1 - i; }
  std::vector<KvafBatchGetDirectUrmaEntity> entities(workers);
  std::vector<KvafBatchGetQueueState> queues(workers);
  std::vector<KvafBatchGetCoreStatus> statuses(cores);
  for (unsigned i = 0; i < workers; ++i) {
    auto& j = jettys[i];
    j.fe = expected.at(i);
    const auto path = kvaf::direct_hccp::GetPathForWorker(i, cores, per_core, 1, fes,
                                                          core_fe_map.empty() ? nullptr : core_fe_map.data());
    Check(path == j.fe, "global Jetty to CPU FE mapping");
    auto& e = entities[i];
    e.magic = kEntityMagic;
    e.version = kEntityVersion;
    e.registered_base = j.registered_base();
    e.remote_host_ddr_address = kCanonicalDdr;
    e.remote_host_ddr_size = sizeof(ddr);
    e.local_hbm_address = reinterpret_cast<uint64_t>(output.data());
    e.local_hbm_size = count * sizeof(uint32_t);
    e.remote_token_id = path + 10; e.remote_token_value = path + 20;
    e.tp_id = path + 30; e.local_token_id = path + 40;
    e.remote_eid[0] = path + 50;
    e.sq_va = reinterpret_cast<uint64_t>(j.sq.data());
    e.sq_db_va = reinterpret_cast<uint64_t>(&j.sq_db);
    e.cq_va = reinterpret_cast<uint64_t>(j.cq.data());
    e.cq_db_va = reinterpret_cast<uint64_t>(&j.cq_db);
    e.sq_depth = 32768; e.cq_depth = 8; e.cqe_size = e.wqe_size = 64;
  }
  int32_t misses = count;
  KvafExecuteBatchGetKernelArgs args{};
  args.address_table = reinterpret_cast<uint64_t>(addresses.data());
  args.valid_table = reinterpret_cast<uint64_t>(valid.data());
  args.indices = reinterpret_cast<uint64_t>(indices.data());
  args.offsets = reinterpret_cast<uint64_t>(offsets.data());
  args.output_indices = reinterpret_cast<uint64_t>(rows.data());
  args.output = reinterpret_cast<uint64_t>(output.data());
  args.miss_count = reinterpret_cast<uint64_t>(&misses);
  args.core_status = reinterpret_cast<uint64_t>(statuses.data());
  args.channel_entity = reinterpret_cast<uint64_t>(entities.data());
  args.queue_state = reinterpret_cast<uint64_t>(queues.data());
  args.table_capacity = addresses.size();
  args.request_capacity = args.output_rows = count;
  args.token_bytes = sizeof(uint32_t); args.tokens_per_block = 1;
  args.block_count = cores; args.jettys_per_core = per_core;
  args.batch_get_max_inflight_total = 2048;
  for (int repeat = 1; repeat <= 2; ++repeat) {
    std::fill(output.begin(), output.end(), 0);
    for (aiv_model::core = 0; aiv_model::core < cores; ++aiv_model::core) {
      KvafExecuteBatchGetAivKernelV3(args);
      Check(statuses[aiv_model::core].result == kSuccess, "kernel completion");
    }
    for (const auto& q : queues) Check(q.sq_producer_index == repeat * 128U, "READs per Jetty");
    for (unsigned i = 0; i < count; ++i) Check(output[rows[i]] == ddr[indices[i]], "payload and output row");
  }
}

int main() {
  Run(1, 2, 1, {0, 0});
  Run(1, 2, 2, {0, 1});
  Run(2, 1, 2, {0, 1});
  Run(2, 2, 3, {0, 1, 2, 0});
  Run(3, 2, 3, {0, 1, 2, 0, 1, 2});
  Run(3, 2, 3, {0, 0, 0, 0, 1, 1}, {0, 0, 1});
  Run(3, 2, 3, {2, 2, 2, 2, 1, 1}, {2, 2, 1});
}
