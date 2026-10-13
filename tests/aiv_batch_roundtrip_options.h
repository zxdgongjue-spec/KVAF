#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace kvaf::aiv_batch_roundtrip {

inline std::string CannRoot() {
#if defined(KVAF_AIV_NATIVE_CANN_ROOT)
  return KVAF_AIV_NATIVE_CANN_ROOT;
#else
  const char* root = std::getenv("ASCEND_HOME_PATH");
  return root != nullptr && root[0] != '\0' ? root : "/usr/local/Ascend/ascend-toolkit/latest";
#endif
}

inline bool ConfigureCannEnvironment() {
  const char* current = std::getenv("ASCEND_HOME_PATH");
  return (current != nullptr && current[0] != '\0') ||
         setenv("ASCEND_HOME_PATH", CannRoot().c_str(), 1) == 0;
}

inline std::string KernelObject(const char* name) {
#if defined(KVAF_AIV_NATIVE_BUILD_DIR)
  return std::string(KVAF_AIV_NATIVE_BUILD_DIR) + "/bin/" + name;
#else
  return std::string("build-aiv-native/bin/") + name;
#endif
}

struct Options {
  std::string peer{"127.0.0.1:28999"};
  std::string put_kernel{KernelObject("kvaf_execute_batch_put_kernels.o")};
  std::string get_kernel{KernelObject("kvaf_execute_batch_get_kernels.o")};
  std::uint32_t device_id{0U};
  std::string peer_device{"cpu"};
  std::uint32_t peer_device_id{0U};
  std::uint32_t blocks{1U};
  std::uint32_t block_bytes{4096U};
  std::uint32_t token_bytes{256U};
  std::uint32_t iterations{1U};
};

inline bool ParseNumber(const char* text, std::uint32_t* value) {
  if (text == nullptr || value == nullptr || text[0] == '\0' || text[0] == '-') return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long long number = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || number > UINT32_MAX) return false;
  *value = static_cast<std::uint32_t>(number);
  return true;
}

inline bool Parse(int argc, const char* const* argv, Options* options) {
  if (argc < 1 || argv == nullptr || options == nullptr) return false;
  Options parsed = *options;
  bool peer_device_id_set = false;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) return false;
    const std::string flag = argv[index];
    const char* value = argv[index + 1];
    if (flag == "--peer") parsed.peer = value;
    else if (flag == "--peer-device") parsed.peer_device = value;
    else if (flag == "--put-kernel") parsed.put_kernel = value;
    else if (flag == "--get-kernel") parsed.get_kernel = value;
    else {
      std::uint32_t number = 0U;
      if (!ParseNumber(value, &number)) return false;
      if (flag == "--device-id") parsed.device_id = number;
      else if (flag == "--peer-device-id") {
        parsed.peer_device_id = number;
        peer_device_id_set = true;
      } else if (flag == "--blocks") parsed.blocks = number;
      else if (flag == "--block-bytes") parsed.block_bytes = number;
      else if (flag == "--token-bytes") parsed.token_bytes = number;
      else if (flag == "--iterations") parsed.iterations = number;
      else return false;
    }
  }
  if (parsed.peer.empty() || parsed.put_kernel.empty() || parsed.get_kernel.empty() ||
      (parsed.peer_device != "cpu" && parsed.peer_device != "npu") ||
      (peer_device_id_set && parsed.peer_device != "npu") ||
      parsed.peer_device_id > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
      parsed.device_id > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
      parsed.blocks == 0U || parsed.block_bytes == 0U || parsed.token_bytes == 0U ||
      parsed.iterations == 0U || parsed.block_bytes % parsed.token_bytes != 0U ||
      parsed.block_bytes > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) return false;
  const std::uint64_t requests = static_cast<std::uint64_t>(parsed.blocks) *
                                 (parsed.block_bytes / parsed.token_bytes);
  const std::uint64_t bytes = static_cast<std::uint64_t>(parsed.blocks) * parsed.block_bytes;
  if (requests > 16384U || bytes > UINT32_MAX ||
      bytes > std::numeric_limits<std::size_t>::max()) return false;
  *options = parsed;
  return true;
}

}  // namespace kvaf::aiv_batch_roundtrip
