#include "cpu_runtime.h"

#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <random>
#include <vector>

namespace {

struct Options {
  std::string peer;
  std::uint64_t bytes{4096U};
  std::uint64_t slot_size{4096U};
  std::string ub_device;
  std::uint32_t eid_index{0U};
  bool has_eid_index{false};
  std::uint32_t device_id{0U};
  std::size_t blocks{1U};
  std::size_t block_bytes{656U};
  std::size_t iterations{1U};
  std::size_t warmup{0U};
};

bool ParseU64(const char* text, std::uint64_t* value) {
  if (text == nullptr || value == nullptr || text[0] == '\0') return false;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 0);
  if (end == text || *end != '\0') return false;
  *value = static_cast<std::uint64_t>(parsed);
  return true;
}

bool ParseU32(const char* text, std::uint32_t* value) {
  std::uint64_t parsed = 0U;
  if (!ParseU64(text, &parsed) || parsed > UINT32_MAX || value == nullptr) return false;
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

void Usage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s --peer HOST:PORT [--bytes N] [--slot-size N] "
               "[--ub-device NAME] [--eid-index N] [--device-id N] "
               "[--blocks N] [--block-bytes N] [--iterations N] [--warmup N]\n",
               program);
}

bool Parse(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") return false;
    if (arg == "--") continue;
    const auto value = [&](const char** out) {
      if (i + 1 >= argc) return false;
      *out = argv[++i];
      return true;
    };
    const char* text = nullptr;
    if (arg == "--peer") {
      if (!value(&text)) return false;
      options->peer = text;
    } else if (arg == "--bytes") {
      if (!value(&text) || !ParseU64(text, &options->bytes)) return false;
    } else if (arg == "--slot-size") {
      if (!value(&text) || !ParseU64(text, &options->slot_size)) return false;
    } else if (arg == "--ub-device") {
      if (!value(&text)) return false;
      options->ub_device = text;
    } else if (arg == "--eid-index") {
      if (!value(&text) || !ParseU32(text, &options->eid_index)) return false;
      options->has_eid_index = true;
    } else if (arg == "--device-id") {
      if (!value(&text) || !ParseU32(text, &options->device_id)) return false;
    } else if (arg == "--blocks") {
      std::uint64_t parsed = 0; if (!value(&text) || !ParseU64(text, &parsed) || parsed == 0 || parsed > SIZE_MAX) return false; options->blocks = static_cast<std::size_t>(parsed);
    } else if (arg == "--block-bytes") {
      std::uint64_t parsed = 0; if (!value(&text) || !ParseU64(text, &parsed) || parsed == 0 || parsed > SIZE_MAX) return false; options->block_bytes = static_cast<std::size_t>(parsed); options->slot_size = options->block_bytes;
    } else if (arg == "--iterations") {
      std::uint64_t parsed = 0; if (!value(&text) || !ParseU64(text, &parsed) || parsed == 0 || parsed > SIZE_MAX) return false; options->iterations = static_cast<std::size_t>(parsed);
    } else if (arg == "--warmup") {
      std::uint64_t parsed = 0; if (!value(&text) || !ParseU64(text, &parsed) || parsed > SIZE_MAX) return false; options->warmup = static_cast<std::size_t>(parsed);
    } else {
      return false;
    }
  }
  if (options->block_bytes == 0U) options->block_bytes = options->slot_size;
  if (options->blocks > SIZE_MAX / options->block_bytes) return false;
  options->bytes = static_cast<std::uint64_t>(options->blocks * options->block_bytes);
  return !options->peer.empty() && options->bytes != 0U && options->slot_size != 0U &&
         options->bytes % options->slot_size == 0U;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!Parse(argc, argv, &options)) {
    Usage(argv[0]);
    return argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") ? 0 : 2;
  }

  kvaf::CpuRuntime runtime;
  if (!runtime.LinkTo(kvaf::LinkTarget{options.peer}).ok()) return 1;
  kvaf::CpuRuntimeClientOptions client;
  client.hbm_bytes = options.bytes;
  client.slot_size = options.slot_size;
  client.peer_address = options.peer;
  client.ub_device = options.ub_device;
  client.eid_index = options.eid_index;
  client.has_eid_index = options.has_eid_index;
  client.device_id = options.device_id;
  kvaf::Status status = runtime.InitializeClient(client);
  if (!status.ok()) {
    std::fprintf(stderr, "[KVAF_CPU_NPU_E2E] initialize failed: %s\n", status.message().c_str());
    return 1;
  }

  std::mt19937_64 rng(1U);
  std::vector<std::uint64_t> keys(options.blocks);
  std::vector<std::vector<std::uint8_t>> inputs(options.blocks, std::vector<std::uint8_t>(options.block_bytes));
  for (std::size_t i = 0; i < options.blocks; ++i) {
    keys[i] = static_cast<std::uint64_t>(i + 1U);
    for (auto& byte : inputs[i]) byte = static_cast<std::uint8_t>(rng());
  }
  kvaf::BatchPutRequest put;
  put.groups.reserve(options.blocks);
  for (std::size_t i = 0; i < options.blocks; ++i) {
    put.groups.push_back(kvaf::BatchPutGroup{{keys[i]}, kvaf::ConstBufferView{inputs[i].data(), inputs[i].size(), kvaf::DeviceKind::kCpu}, {0U}, {inputs[i].size()}});
  }
  kvaf::AccessTableHandle table;
  const auto put_start = std::chrono::steady_clock::now();
  status = runtime.core().BatchPut(put);
  const auto put_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - put_start).count();
  if (!status.ok()) { std::fprintf(stderr, "[KVAF_CPU_NPU_E2E] batch put failed: %s\n", status.message().c_str()); return 1; }
  status = runtime.core().PrepareAccessTable(kvaf::AccessTablePrepareRequest{keys}, &table);
  if (!status.ok()) return 1;
  std::vector<std::int32_t> indices(options.blocks), offsets(options.blocks, 0), output_indices(options.blocks);
  for (std::size_t i = 0; i < options.blocks; ++i) { indices[i] = static_cast<std::int32_t>(i); output_indices[i] = static_cast<std::int32_t>(i); }
  std::vector<std::uint8_t> output(options.blocks * options.block_bytes, 0U);
  kvaf::BatchGetRequest get;
  get.access_table = table;
  get.indices = kvaf::ConstBufferView{indices.data(), indices.size() * sizeof(std::int32_t), kvaf::DeviceKind::kCpu};
  get.offsets = kvaf::ConstBufferView{offsets.data(), offsets.size() * sizeof(std::int32_t), kvaf::DeviceKind::kCpu};
  get.output_indices = kvaf::ConstBufferView{output_indices.data(), output_indices.size() * sizeof(std::int32_t), kvaf::DeviceKind::kCpu};
  get.output = kvaf::MutableBufferView{output.data(), output.size(), kvaf::DeviceKind::kCpu};
  get.output_block_bytes = options.block_bytes;
  for (std::size_t i = 0; i < options.warmup; ++i) { status = runtime.core().BatchGet(get); if (!status.ok()) break; }
  if (!status.ok()) { std::fprintf(stderr, "[KVAF_CPU_NPU_E2E] warmup batch get failed: %s\n", status.message().c_str()); return 1; }
  const auto timed_get = [&]() {
    const auto start = std::chrono::steady_clock::now(); kvaf::Status result;
    for (std::size_t i = 0; i < options.iterations; ++i) result = runtime.core().BatchGet(get);
    return std::pair<kvaf::Status, long long>{result, std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()};
  };
  const std::pair<kvaf::Status, long long> put_result{status, put_ns};
  std::fill(output.begin(), output.end(), 0U);
  auto get_result = timed_get();
  if (!get_result.first.ok()) { std::fprintf(stderr, "[KVAF_CPU_NPU_E2E] batch get failed: %s\n", get_result.first.message().c_str()); return 1; }
  bool verified = get_result.first.ok();
  for (std::size_t i = 0; verified && i < options.blocks; ++i) verified = std::memcmp(output.data() + i * options.block_bytes, inputs[i].data(), options.block_bytes) == 0;
  (void)runtime.core().ReleaseAccessTable(table);
  (void)runtime.core().BatchFree(kvaf::BatchFreeRequest{keys});
  (void)runtime.Shutdown();
  if (!verified) {
    std::fprintf(stderr, "[KVAF_CPU_NPU_E2E] batch get/verify failed: %s\n", status.message().c_str());
    return 1;
  }
  const double put_bw = static_cast<double>(options.blocks * options.block_bytes * options.iterations) * 1e9 / static_cast<double>(put_result.second) / (1024.0 * 1024.0);
  const double get_bw = static_cast<double>(options.blocks * options.block_bytes * options.iterations) * 1e9 / static_cast<double>(get_result.second) / (1024.0 * 1024.0);
  std::fprintf(stdout, "[KVAF_CPU_NPU_E2E] PASS blocks=%zu block_bytes=%zu put_bandwidth=%.2fMiB/s get_bandwidth=%.2fMiB/s\n", options.blocks, options.block_bytes, put_bw, get_bw);
  return 0;
}
