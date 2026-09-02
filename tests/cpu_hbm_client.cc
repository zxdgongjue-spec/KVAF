#include "cpu_urma_transport.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string peer;
  std::size_t blocks{1};
  std::size_t block_bytes{656};
  std::size_t iterations{1};
  std::size_t warmup{1};
  std::uint64_t seed{0x6b766166ULL};
  std::string ub_device;
  bool has_eid_index{false};
  std::uint32_t eid_index{0};
  std::uint32_t device_id{0};
  std::size_t sgl_size{1};
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
               "[--iterations N] [--warmup N] [--seed N] [--ub-device NAME] "
               "[--eid-index N] [--device-id N] [--sgl-size N]\n",
               program);
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
    else if (arg == "--bytes" && ParseU64(value, &number)) {
      options->blocks = 1;
      options->block_bytes = number;
    }
    else if (arg == "--iterations" && ParseU64(value, &number)) options->iterations = number;
    else if (arg == "--warmup" && ParseU64(value, &number)) options->warmup = number;
    else if (arg == "--seed" && ParseU64(value, &options->seed)) {}
    else if (arg == "--ub-device") options->ub_device = value;
    else if (arg == "--eid-index" && ParseU32(value, &options->eid_index)) options->has_eid_index = true;
    else if (arg == "--device-id" && ParseU32(value, &options->device_id)) {}
    else if (arg == "--sgl-size" && ParseU64(value, &number)) options->sgl_size = number;
    else return false;
  }
  return options->blocks != 0 && options->block_bytes != 0 && options->iterations != 0 &&
         options->blocks <= std::numeric_limits<std::uint64_t>::max() / options->block_bytes &&
         options->block_bytes <= std::numeric_limits<std::uint32_t>::max();
}

struct Stats {
  std::uint64_t ops{0};
  std::uint64_t blocks{0};
  std::uint64_t bytes{0};
  std::uint64_t sum_ns{0};
  std::uint64_t min_ns{std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t max_ns{0};
  std::uint64_t wall_ns{0};
};

struct HostBlock {
  std::uint8_t* data{nullptr};
};

bool RunPass(kvaf::CpuUrmaTransport* transport, bool write,
             std::vector<HostBlock>* blocks,
             const std::vector<std::size_t>& order, std::size_t block_bytes,
             std::size_t iterations, std::size_t sgl_size, Stats* stats) {
  if (transport == nullptr || blocks == nullptr || stats == nullptr) return false;
  const auto wall_start = std::chrono::steady_clock::now();
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    for (std::size_t position = 0; position < order.size();) {
      const std::size_t sgl_count = std::min(sgl_size, order.size() - position);
      std::vector<kvaf::CpuUrmaTransport::HostSge> sges(sgl_count);
      std::uint64_t sgl_bytes = 0U;
      for (std::size_t i = 0; i < sgl_count; ++i) {
        const std::size_t block_index = order[position + i];
        sges[i] = {(*blocks)[block_index].data, block_bytes};
        sgl_bytes += block_bytes;
      }
      const auto start = std::chrono::steady_clock::now();
      const std::uint64_t offset = static_cast<std::uint64_t>(position) * block_bytes;
      const kvaf::Status status = write
          ? transport->WriteSgl(offset, sges.data(), sges.size())
          : transport->ReadSgl(offset, sges.data(), sges.size());
      if (!status.ok()) {
        std::fprintf(stderr, "[HBM_TEST][CPU] %s failed sgl_position=%zu: %s\n",
                     write ? "WRITE" : "READ", position, status.message().c_str());
        return false;
      }
      const auto elapsed = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start).count());
      ++stats->ops;
      stats->blocks += sgl_count;
      stats->bytes += sgl_bytes;
      stats->sum_ns += elapsed;
      stats->min_ns = std::min(stats->min_ns, elapsed);
      stats->max_ns = std::max(stats->max_ns, elapsed);
      position += sgl_count;
    }
  }
  stats->wall_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - wall_start).count());
  return true;
}

void PrintStats(const char* name, const Stats& stats) {
  const double bandwidth = stats.wall_ns == 0
      ? 0.0 : stats.bytes * 1e9 / stats.wall_ns / (1024.0 * 1024.0);
  const double average = stats.ops == 0 ? 0.0 : stats.sum_ns / static_cast<double>(stats.ops);
  std::printf("[HBM_TEST][CPU] %-5s bytes=%llu blocks=%llu wrs=%llu bandwidth=%.2f MiB/s "
              "avg_wr_latency=%.3f us min=%.3f us max=%.3f us\n",
              name, static_cast<unsigned long long>(stats.bytes),
              static_cast<unsigned long long>(stats.blocks),
              static_cast<unsigned long long>(stats.ops), bandwidth, average / 1000.0,
              stats.min_ns / 1000.0, stats.max_ns / 1000.0);
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    Usage(argv[0]);
    return argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") ? 0 : 2;
  }
  const std::uint64_t total_bytes = static_cast<std::uint64_t>(options.blocks) * options.block_bytes;
  std::fprintf(stderr,
               "[HBM_TEST][CPU] START peer=%s blocks=%zu block_bytes=%zu total_bytes=%llu "
               "iterations=%zu warmup=%zu remote_device=%u\n",
               options.peer.c_str(), options.blocks, options.block_bytes,
               static_cast<unsigned long long>(total_bytes), options.iterations, options.warmup,
               options.device_id);
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
  kvaf::CpuUrmaTransport transport;
  kvaf::Status status = transport.Initialize(transport_options);
  if (!status.ok()) {
    std::fprintf(stderr, "[HBM_TEST][CPU] FAILED stage=transport_initialize: %s\n",
                 status.message().c_str());
    return 1;
  }
  std::fprintf(stderr, "[HBM_TEST][CPU] CHANNEL_READY remote_bytes=%llu\n",
               static_cast<unsigned long long>(transport.region_bytes()));
  if (options.sgl_size == 0U || options.sgl_size > transport.max_sge()) {
    std::fprintf(stderr,
                 "[HBM_TEST][CPU] FAILED stage=sgl_size requested=%zu max_sge=%u\n",
                 options.sgl_size, transport.max_sge());
    (void)transport.Shutdown();
    return 1;
  }
  std::fprintf(stderr, "[HBM_TEST][CPU] SGL_SIZE=%zu\n", options.sgl_size);

  // Separate allocations intentionally produce non-contiguous/random host addresses.
  std::mt19937_64 rng(options.seed);
  const std::size_t page_bytes = 4096U;
  const std::size_t allocation_bytes =
      (total_bytes + page_bytes - 1U) / page_bytes * page_bytes;
  std::unique_ptr<std::uint8_t, decltype(&std::free)> storage(nullptr, &std::free);
  void* storage_address = nullptr;
  if (posix_memalign(&storage_address, page_bytes, allocation_bytes) != 0 ||
      storage_address == nullptr) {
    std::fprintf(stderr, "[HBM_TEST][CPU] host region allocation failed bytes=%zu\n",
                 allocation_bytes);
    (void)transport.Shutdown();
    return 1;
  }
  storage.reset(static_cast<std::uint8_t*>(storage_address));
  status = transport.RegisterDirectHostMemory(storage.get(), total_bytes);
  if (!status.ok()) {
    std::fprintf(stderr, "[HBM_TEST][CPU] FAILED stage=host_region_register: %s\n",
                 status.message().c_str());
    (void)transport.Shutdown();
    return 1;
  }
  std::vector<HostBlock> blocks(options.blocks);
  std::vector<std::vector<std::uint8_t>> expected(
      options.blocks, std::vector<std::uint8_t>(options.block_bytes));
  std::vector<std::size_t> slots(options.blocks);
  for (std::size_t i = 0; i < options.blocks; ++i) slots[i] = i;
  std::shuffle(slots.begin(), slots.end(), rng);
  for (std::size_t i = 0; i < options.blocks; ++i) {
    blocks[i].data = storage.get() + slots[i] * options.block_bytes;
    for (std::size_t j = 0; j < options.block_bytes; ++j) {
      blocks[i].data[j] = static_cast<std::uint8_t>(rng());
      expected[i][j] = blocks[i].data[j];
    }
  }
  std::vector<std::size_t> order(options.blocks);
  for (std::size_t i = 0; i < options.blocks; ++i) order[i] = i;
  std::shuffle(order.begin(), order.end(), rng);

  Stats warmup_stats;
  if (options.warmup != 0 && !RunPass(&transport, true, &blocks, order, options.block_bytes,
                                      options.warmup, options.sgl_size, &warmup_stats)) {
    (void)transport.Shutdown();
    return 1;
  }
  Stats write_stats;
  if (!RunPass(&transport, true, &blocks, order, options.block_bytes,
               options.iterations, options.sgl_size, &write_stats)) {
    (void)transport.Shutdown();
    return 1;
  }
  for (auto& block : blocks) std::fill_n(block.data, options.block_bytes, 0);
  Stats read_stats;
  if (!RunPass(&transport, false, &blocks, order, options.block_bytes,
               options.iterations, options.sgl_size, &read_stats)) {
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
