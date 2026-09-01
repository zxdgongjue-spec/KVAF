#include "cpu_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
               "[--ub-device NAME] [--eid-index N] [--device-id N]\n",
               program);
}

bool Parse(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--") continue;
    if (arg == "--help" || arg == "-h") return false;
    if (i + 1 >= argc) return false;
    if (arg == "--peer") options->peer = argv[++i];
    else if (arg == "--bytes" && !ParseU64(argv[++i], &options->bytes)) return false;
    else if (arg == "--slot-size" && !ParseU64(argv[++i], &options->slot_size)) return false;
    else if (arg == "--ub-device") options->ub_device = argv[++i];
    else if (arg == "--eid-index") {
      if (!ParseU32(argv[++i], &options->eid_index)) return false;
      options->has_eid_index = true;
    }
    else if (arg == "--device-id" && !ParseU32(argv[++i], &options->device_id)) return false;
    else return false;
  }
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

  const std::uint64_t key = 1U;
  std::vector<std::uint8_t> input(options.slot_size);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<std::uint8_t>((i * 17U + 3U) & 0xffU);
  kvaf::BatchPutRequest put;
  put.groups.push_back(kvaf::BatchPutGroup{{key}, kvaf::ConstBufferView{input.data(), input.size(), kvaf::DeviceKind::kCpu}, {0U}, {input.size()}});
  status = runtime.core().BatchPut(put);
  if (!status.ok()) {
    std::fprintf(stderr, "[KVAF_CPU_NPU_E2E] batch put failed: %s\n", status.message().c_str());
    return 1;
  }

  kvaf::AccessTableHandle table;
  status = runtime.core().PrepareAccessTable(kvaf::AccessTablePrepareRequest{{key}}, &table);
  if (!status.ok()) return 1;
  std::int32_t index = 0;
  std::int32_t offset = 0;
  std::int32_t output_index = 0;
  std::vector<std::uint8_t> output(options.slot_size, 0U);
  kvaf::BatchGetRequest get;
  get.access_table = table;
  get.indices = kvaf::ConstBufferView{&index, sizeof(index), kvaf::DeviceKind::kCpu};
  get.offsets = kvaf::ConstBufferView{&offset, sizeof(offset), kvaf::DeviceKind::kCpu};
  get.output_indices = kvaf::ConstBufferView{&output_index, sizeof(output_index), kvaf::DeviceKind::kCpu};
  get.output = kvaf::MutableBufferView{output.data(), output.size(), kvaf::DeviceKind::kCpu};
  get.output_block_bytes = options.slot_size;
  status = runtime.core().BatchGet(get);
  const bool verified = status.ok() && output == input;
  (void)runtime.core().ReleaseAccessTable(table);
  (void)runtime.core().BatchFree(kvaf::BatchFreeRequest{{key}});
  (void)runtime.Shutdown();
  if (!verified) {
    std::fprintf(stderr, "[KVAF_CPU_NPU_E2E] batch get/verify failed: %s\n", status.message().c_str());
    return 1;
  }
  std::fprintf(stdout, "[KVAF_CPU_NPU_E2E] PASS bytes=%llu\n",
               static_cast<unsigned long long>(options.slot_size));
  return 0;
}
