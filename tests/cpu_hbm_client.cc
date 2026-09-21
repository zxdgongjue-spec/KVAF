#include "cpu_urma/cpu_urma_transport.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <sys/mman.h>
#include <unordered_set>
#include <unistd.h>
#include <vector>

namespace {

struct Options {
  std::string peer;
  std::size_t blocks{1};
  std::size_t block_bytes{656};
  std::size_t iterations{1};
  std::size_t warmup{1};
  std::uint32_t wqes_per_doorbell{0};
  std::uint64_t host_space_bytes{0};
  std::uint64_t seed{0x6b766166ULL};
  std::string ub_device;
  bool has_eid_index{false};
  std::uint32_t eid_index{0};
  std::uint32_t device_id{0};
  std::uint32_t jetty_count{4};
  std::uint32_t remote_sgl_size{1};
  std::uint32_t prepare_thread_count{4};
};

bool ParseU64(const char* text, std::uint64_t* value) {
  if (text == nullptr || value == nullptr || *text == '\0') return false;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 0);
  if (end == text || *end != '\0') return false;
  *value = static_cast<std::uint64_t>(parsed);
  return true;
}

bool ParseU32(const char* text, std::uint32_t* value) {
  std::uint64_t parsed = 0;
  if (value == nullptr || !ParseU64(text, &parsed) ||
      parsed > std::numeric_limits<std::uint32_t>::max()) return false;
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

void Usage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s HOST[:PORT] [PORT] [--blocks N] [--block-bytes N] "
               "[--host-space-bytes N[K|M|G|T]] [--iterations N] [--warmup N] "
               "[--doorbell-wqes N] "
               "[--seed N] [--ub-device NAME] "
               "[--eid-index N] [--device-id N] [--jettys N] "
               "[--remote-sgl-size N] [--prepare-threads N]\n",
               program);
}

bool ParseBytes(const char* text, std::uint64_t* value) {
  if (text == nullptr || value == nullptr || *text == '\0') return false;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 0);
  if (end == text) return false;
  std::uint64_t multiplier = 1U;
  if (*end != '\0') {
    if (end[1] != '\0') return false;
    switch (*end) {
      case 'k': case 'K': multiplier = 1ULL << 10U; break;
      case 'm': case 'M': multiplier = 1ULL << 20U; break;
      case 'g': case 'G': multiplier = 1ULL << 30U; break;
      case 't': case 'T': multiplier = 1ULL << 40U; break;
      default: return false;
    }
  }
  if (parsed > std::numeric_limits<std::uint64_t>::max() / multiplier) return false;
  *value = static_cast<std::uint64_t>(parsed) * multiplier;
  return true;
}

bool Parse(int argc, char** argv, Options* options) {
  if (argc < 2 || options == nullptr) return false;
  options->peer = argv[1];
  int index = 2;
  if (index < argc && argv[index][0] != '-') {
    options->peer += ":";
    options->peer += argv[index++];
  } else if (options->peer.find(':') == std::string::npos) {
    options->peer += ":29888";
  }
  while (index < argc) {
    const std::string arg(argv[index++]);
    if (arg == "--help" || arg == "-h") return false;
    if (index >= argc) return false;
    const char* value = argv[index++];
    std::uint64_t number = 0;
    if (arg == "--blocks" && ParseU64(value, &number)) options->blocks = number;
    else if (arg == "--block-bytes" && ParseU64(value, &number)) options->block_bytes = number;
    else if (arg == "--host-space-bytes" && ParseBytes(value, &options->host_space_bytes)) {}
    else if (arg == "--doorbell-wqes" && ParseU32(value, &options->wqes_per_doorbell)) {}
    else if (arg == "--iterations" && ParseU64(value, &number)) options->iterations = number;
    else if (arg == "--warmup" && ParseU64(value, &number)) options->warmup = number;
    else if (arg == "--seed" && ParseU64(value, &options->seed)) {}
    else if (arg == "--ub-device") options->ub_device = value;
    else if (arg == "--eid-index" && ParseU32(value, &options->eid_index)) options->has_eid_index = true;
    else if (arg == "--device-id" && ParseU32(value, &options->device_id)) {}
    else if (arg == "--jettys" && ParseU32(value, &options->jetty_count)) {}
    else if (arg == "--remote-sgl-size" && ParseU32(value, &options->remote_sgl_size)) {}
    else if (arg == "--prepare-threads" && ParseU32(value, &options->prepare_thread_count)) {}
    else return false;
  }
  return options->blocks != 0 && options->block_bytes != 0 && options->iterations != 0 &&
         options->wqes_per_doorbell <= 1024U &&
         options->jetty_count >= 1U && options->jetty_count <= 16U &&
         options->remote_sgl_size != 0U &&
         options->prepare_thread_count >= 1U && options->prepare_thread_count <= 64U &&
         options->blocks <= std::numeric_limits<std::uint64_t>::max() / options->block_bytes &&
         options->block_bytes <= std::numeric_limits<std::uint32_t>::max();
}

struct Stats {
  std::uint64_t batches{0};
  std::uint64_t wrs{0};
  std::uint64_t blocks{0};
  std::uint64_t bytes{0};
  std::uint64_t sum_ns{0};
  std::uint64_t min_ns{std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t max_ns{0};
  std::uint64_t wall_ns{0};
  std::uint64_t plan_ns{0};
  std::uint64_t prepare_ns{0};
  std::uint64_t post_ns{0};
  std::uint64_t poll_ns{0};
  std::uint64_t setup_ns{0};
  std::uint64_t dispatch_ns{0};
  std::uint64_t total_ns{0};
  std::uint64_t phase_samples{0};
};

struct HostBlock {
  std::uint8_t* data{nullptr};
};

struct MappedRegion {
  std::uint8_t* data{nullptr};
  std::size_t bytes{0};

  MappedRegion() = default;
  ~MappedRegion() {
    if (data != nullptr) (void)munmap(data, bytes);
  }
  MappedRegion(const MappedRegion&) = delete;
  MappedRegion& operator=(const MappedRegion&) = delete;
};

bool RunBatchPass(kvaf::CpuUrmaTransport* transport, bool write,
                std::vector<HostBlock>* blocks,
                const std::vector<std::size_t>& order, std::size_t block_bytes,
                std::size_t iterations, Stats* stats,
                std::uint32_t remote_sgl_size = 1U) {
  if (transport == nullptr || blocks == nullptr || stats == nullptr) {
    return false;
  }
  const auto wall_start = std::chrono::steady_clock::now();
  const bool coalesced_write = write && remote_sgl_size > 1U;
  const bool use_coalesced_order = remote_sgl_size > 1U;
  std::vector<std::size_t> positions;
  std::vector<std::uint64_t> remote_offsets;
  if (use_coalesced_order) {
    positions.resize(order.size());
    for (std::size_t i = 0U; i < order.size(); ++i) positions[i] = i;
    std::sort(positions.begin(), positions.end(), [&](std::size_t a, std::size_t b) {
      return reinterpret_cast<std::uintptr_t>((*blocks)[order[a]].data) <
             reinterpret_cast<std::uintptr_t>((*blocks)[order[b]].data);
    });
    if (coalesced_write) remote_offsets.resize(order.size());
  }
  std::vector<kvaf::CpuUrmaTransport::HostSge> requests(order.size());
  for (std::size_t i = 0U; i < order.size(); ++i) {
    const std::size_t position = use_coalesced_order ? positions[i] : i;
    requests[i] = {(*blocks)[order[position]].data, block_bytes};
    if (coalesced_write) remote_offsets[i] = static_cast<std::uint64_t>(i) * block_bytes;
  }
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    const kvaf::Status status = write
        ? (coalesced_write
               ? transport->WriteBatch(remote_offsets.data(), requests.data(), requests.size())
               : transport->WriteSglBatch(0U, requests.data(), requests.size()))
        : transport->ReadSglBatch(0U, requests.data(), requests.size());
    if (!status.ok()) {
      std::fprintf(stderr, "[HBM_TEST][CPU] %s failed: %s\n",
                   write ? "WRITE" : "READ", status.message().c_str());
      return false;
    }
    const auto timing = transport->last_batch_timing();
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    ++stats->batches;
    stats->wrs += timing.wr_count;
    stats->blocks += requests.size();
    stats->bytes += static_cast<std::uint64_t>(requests.size()) * block_bytes;
    stats->sum_ns += elapsed;
    stats->min_ns = std::min(stats->min_ns, elapsed);
    stats->max_ns = std::max(stats->max_ns, elapsed);
    stats->plan_ns += timing.plan_ns;
    stats->prepare_ns += timing.prepare_ns;
    stats->post_ns += timing.post_ns;
    stats->poll_ns += timing.poll_ns;
    stats->setup_ns += timing.setup_ns;
    stats->dispatch_ns += timing.dispatch_ns;
    stats->total_ns += timing.total_ns;
    ++stats->phase_samples;
  }
  stats->wall_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - wall_start).count());
  return true;
}

void PrintStats(const char* name, const Stats& stats) {
  const double bandwidth = stats.wall_ns == 0
      ? 0.0 : stats.bytes * 1e9 / stats.wall_ns / (1024.0 * 1024.0);
  const double average = stats.batches == 0 ? 0.0 : stats.sum_ns / static_cast<double>(stats.batches);
  std::printf("[HBM_TEST][CPU] %-5s bytes=%llu blocks=%llu wrs=%llu batches=%llu "
              "bandwidth=%.2f MiB/s avg_batch_latency=%.3f us min=%.3f us max=%.3f us\n",
              name, static_cast<unsigned long long>(stats.bytes),
              static_cast<unsigned long long>(stats.blocks),
              static_cast<unsigned long long>(stats.wrs),
              static_cast<unsigned long long>(stats.batches), bandwidth, average / 1000.0,
              stats.min_ns / 1000.0, stats.max_ns / 1000.0);
  if (stats.phase_samples != 0U) {
    const double samples = static_cast<double>(stats.phase_samples);
    std::printf("[HBM_TEST][CPU] %-5s_PHASE avg_plan=%.3f us avg_setup=%.3f us "
                "avg_prepare=%.3f us avg_post=%.3f us avg_poll=%.3f us\n",
                name, stats.plan_ns / samples / 1000.0,
                stats.setup_ns / samples / 1000.0,
                stats.prepare_ns / samples / 1000.0,
                stats.post_ns / samples / 1000.0,
                stats.poll_ns / samples / 1000.0);
    const auto covered_ns = stats.plan_ns + stats.setup_ns +
        stats.dispatch_ns + stats.poll_ns;
    const auto other_ns = stats.total_ns > covered_ns
        ? stats.total_ns - covered_ns : 0U;
    std::printf("[HBM_TEST][CPU] %-5s_WALL avg_plan=%.3f us avg_setup=%.3f us "
                "avg_submit_wall=%.3f us avg_poll_wait=%.3f us avg_other=%.3f us "
                "avg_total=%.3f us\n",
                name, stats.plan_ns / samples / 1000.0,
                stats.setup_ns / samples / 1000.0,
                stats.dispatch_ns / samples / 1000.0,
                stats.poll_ns / samples / 1000.0,
                other_ns / samples / 1000.0,
                stats.total_ns / samples / 1000.0);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    Usage(argv[0]);
    return argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") ? 0 : 2;
  }
  // OpenMP may read its policy before main; re-exec so ACTIVE takes effect.
  const char* wait_policy = std::getenv("OMP_WAIT_POLICY");
  if (wait_policy == nullptr || std::strcmp(wait_policy, "ACTIVE") != 0) {
    if (setenv("OMP_WAIT_POLICY", "ACTIVE", 1) == 0) {
#if defined(__linux__)
      execv("/proc/self/exe", argv);
#else
      execvp(argv[0], argv);
#endif
    }
    std::perror("[HBM_TEST][CPU] FAILED stage=omp_active_startup");
    return 1;
  }
  const std::uint64_t total_bytes = static_cast<std::uint64_t>(options.blocks) * options.block_bytes;
  const std::uint64_t host_space_bytes =
      options.host_space_bytes == 0U ? total_bytes : options.host_space_bytes;
  if (host_space_bytes < total_bytes ||
      host_space_bytes / options.block_bytes < options.blocks ||
      host_space_bytes > std::numeric_limits<std::size_t>::max()) {
    std::fprintf(stderr,
                 "[HBM_TEST][CPU] invalid host address space bytes=%llu for blocks=%zu block_bytes=%zu\n",
                 static_cast<unsigned long long>(host_space_bytes), options.blocks,
                 options.block_bytes);
    return 2;
  }
  std::fprintf(stderr,
               "[HBM_TEST][CPU] START peer=%s blocks=%zu block_bytes=%zu payload_bytes=%llu "
               "host_address_space_bytes=%llu "
               "iterations=%zu warmup=%zu "
               "remote_device=%u jettys=%u prepare_threads=%u remote_sgl_size=%u "
               "write_order=%s omp_wait_policy=ACTIVE\n",
               options.peer.c_str(), options.blocks, options.block_bytes,
               static_cast<unsigned long long>(total_bytes),
               static_cast<unsigned long long>(host_space_bytes), options.iterations,
               options.warmup,
               options.device_id, options.jetty_count, options.prepare_thread_count,
               options.remote_sgl_size,
               options.remote_sgl_size > 1U ? "local-address" : "random");
  std::fprintf(stderr, "[HBM_TEST][CPU] DATA_PATH=direct-registered-host-memory\n");

  kvaf::CpuUrmaTransportOptions transport_options;
  transport_options.peer_address = options.peer;
  transport_options.ub_device = options.ub_device;
  transport_options.has_eid_index = options.has_eid_index;
  transport_options.eid_index = options.eid_index;
  transport_options.region_bytes = total_bytes;
  transport_options.remote_hbm = true;
  transport_options.remote_device_id = options.device_id;
  transport_options.remote_eid_index = options.has_eid_index ? options.eid_index : UINT32_MAX;
  transport_options.wqes_per_doorbell = options.wqes_per_doorbell;
  transport_options.jetty_count = options.jetty_count;
  transport_options.remote_sgl_size = options.remote_sgl_size;
  transport_options.prepare_thread_count = options.prepare_thread_count;
  kvaf::CpuUrmaTransport transport;
  kvaf::Status status = transport.Initialize(transport_options);
  if (!status.ok()) {
    std::fprintf(stderr, "[HBM_TEST][CPU] FAILED stage=transport_initialize: %s\n",
                 status.message().c_str());
    return 1;
  }
  std::fprintf(stderr, "[HBM_TEST][CPU] CHANNEL_READY remote_bytes=%llu\n",
               static_cast<unsigned long long>(transport.region_bytes()));

  // Reserve the requested virtual address space, but touch only the selected
  // blocks. This keeps the HBM allocation at payload_bytes while allowing a
  // 100G host address-space placement test.
  std::mt19937_64 rng(options.seed);
  const std::size_t page_bytes = 4096U;
  if (static_cast<std::size_t>(host_space_bytes) >
      std::numeric_limits<std::size_t>::max() - (page_bytes - 1U)) {
    std::fprintf(stderr, "[HBM_TEST][CPU] host address space size overflows page rounding\n");
    (void)transport.Shutdown();
    return 1;
  }
  const std::size_t allocation_bytes =
      (static_cast<std::size_t>(host_space_bytes) + page_bytes - 1U) /
      page_bytes * page_bytes;
  int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
  mmap_flags |= MAP_NORESERVE;
#endif
  void* storage_address = mmap(nullptr, allocation_bytes, PROT_READ | PROT_WRITE,
                               mmap_flags, -1, 0);
  MappedRegion storage;
  if (storage_address == MAP_FAILED) {
    std::fprintf(stderr, "[HBM_TEST][CPU] host address-space reservation failed bytes=%zu\n",
                 allocation_bytes);
    (void)transport.Shutdown();
    return 1;
  }
  storage.data = static_cast<std::uint8_t*>(storage_address);
  storage.bytes = allocation_bytes;
  std::vector<HostBlock> blocks(options.blocks);
  std::vector<std::vector<std::uint8_t>> expected(
      options.blocks, std::vector<std::uint8_t>(options.block_bytes));
  const std::uint64_t slot_count = host_space_bytes / options.block_bytes;
  std::uniform_int_distribution<std::uint64_t> slot_distribution(0U, slot_count - 1U);
  std::unordered_set<std::uint64_t> used_slots;
  used_slots.reserve(options.blocks * 2U);
  for (std::size_t i = 0; i < options.blocks; ++i) {
    std::uint64_t slot = 0U;
    do {
      slot = slot_distribution(rng);
    } while (!used_slots.insert(slot).second);
    blocks[i].data = storage.data +
                     static_cast<std::size_t>(slot * options.block_bytes);
    for (std::size_t j = 0; j < options.block_bytes; ++j) {
      blocks[i].data[j] = static_cast<std::uint8_t>(rng());
      expected[i][j] = blocks[i].data[j];
    }
  }
  // Fault in the selected pages before registering the encompassing segment.
  // Providers that validate VA-to-page mappings during registration can then
  // accept the sparse 100G virtual range without requiring all pages to be
  // physically resident.
  status = transport.RegisterDirectHostMemory(
      storage.data, static_cast<std::size_t>(host_space_bytes));
  if (!status.ok()) {
    std::fprintf(stderr, "[HBM_TEST][CPU] FAILED stage=host_region_register: %s\n",
                 status.message().c_str());
    (void)transport.Shutdown();
    return 1;
  }
  std::vector<std::size_t> order(options.blocks);
  for (std::size_t i = 0; i < options.blocks; ++i) order[i] = i;
  std::shuffle(order.begin(), order.end(), rng);

  Stats warmup_stats;
  const auto run_pass = [&](bool write, std::size_t iterations, Stats* stats) {
    return RunBatchPass(&transport, write, &blocks, order, options.block_bytes,
                      iterations, stats, options.remote_sgl_size);
  };
  if (options.warmup != 0 && !run_pass(true, options.warmup, &warmup_stats)) {
    (void)transport.Shutdown();
    return 1;
  }
  Stats write_stats;
  if (!run_pass(true, options.iterations, &write_stats)) {
    (void)transport.Shutdown();
    return 1;
  }
  for (auto& block : blocks) std::fill_n(block.data, options.block_bytes, 0);
  Stats read_stats;
  if (!run_pass(false, options.iterations, &read_stats)) {
    (void)transport.Shutdown();
    return 1;
  }
  bool verified = true;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    verified = verified && std::equal(blocks[i].data,
                                      blocks[i].data + options.block_bytes,
                                      expected[i].begin());
  }
  PrintStats("WRITE", write_stats);
  PrintStats("READ", read_stats);
  std::printf("[HBM_TEST][CPU] %s blocks=%zu block_bytes=%zu seed=%llu\n",
              verified ? "PASS" : "FAIL", options.blocks, options.block_bytes,
              static_cast<unsigned long long>(options.seed));
  (void)transport.Shutdown();
  return verified ? 0 : 1;
}
