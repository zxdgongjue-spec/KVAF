#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace kvaf::aicpu_urma_client {

inline std::string DefaultCannRoot() {
#if defined(KVAF_AICPU_URMA_CANN_ROOT)
  const char* root = KVAF_AICPU_URMA_CANN_ROOT;
#else
  const char* root = std::getenv("ASCEND_CANN_PACKAGE_PATH");
  if (root == nullptr || root[0] == '\0') root = std::getenv("ASCEND_HOME_PATH");
  if (root == nullptr || root[0] == '\0') root = "/usr/local/Ascend/ascend-toolkit/latest";
#endif
  return root;
}

inline std::string DefaultKernelJson() {
  return DefaultCannRoot() + "/opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json";
}

inline bool ConfigureCannEnvironment() {
  const char* current = std::getenv("ASCEND_HOME_PATH");
  if (current != nullptr && current[0] != '\0') return true;
  return setenv("ASCEND_HOME_PATH", DefaultCannRoot().c_str(), 1) == 0;
}

struct Options {
  std::string peer{"127.0.0.1:28999"};
  std::string kernel_json{DefaultKernelJson()};
  std::uint32_t device_id{0};
  std::uint32_t blocks{8};
  std::uint32_t block_bytes{4096};
  std::uint32_t token_bytes{block_bytes};
  std::uint32_t warmup{1};
  std::uint32_t iterations{3};
};

inline bool Number(const char* text, std::uint64_t* value) {
  if (text == nullptr || text[0] == '\0' || text[0] == '-') return false;
  char* end = nullptr;
  errno = 0;
  const auto parsed = std::strtoull(text, &end, 0);
  if (errno != 0 || end == text || *end != '\0') return false;
  *value = static_cast<std::uint64_t>(parsed);
  return true;
}

inline bool Parse(int argc, const char* const* argv, Options* options) {
  if (argc < 1 || argv == nullptr || options == nullptr) return false;
  Options parsed;
  bool token_bytes_set = false;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) return false;
    const std::string flag = argv[index];
    const char* value = argv[index + 1];
    if (flag == "--peer") parsed.peer = value;
    else if (flag == "--kernel-json") parsed.kernel_json = value;
    else {
      std::uint64_t number = 0;
      if (!Number(value, &number)) return false;
      if (number > UINT32_MAX) return false;
      if (flag == "--device-id") parsed.device_id = static_cast<std::uint32_t>(number);
      else if (flag == "--blocks") parsed.blocks = static_cast<std::uint32_t>(number);
      else if (flag == "--block-bytes") parsed.block_bytes = static_cast<std::uint32_t>(number);
      else if (flag == "--token-bytes") {
        parsed.token_bytes = static_cast<std::uint32_t>(number);
        token_bytes_set = true;
      } else if (flag == "--warmup") parsed.warmup = static_cast<std::uint32_t>(number);
      else if (flag == "--iterations") parsed.iterations = static_cast<std::uint32_t>(number);
      else return false;
    }
  }
  if (!token_bytes_set) parsed.token_bytes = parsed.block_bytes;
  if (parsed.peer.empty() || parsed.kernel_json.empty() || parsed.blocks == 0 ||
      parsed.device_id > INT32_MAX || parsed.token_bytes == 0 ||
      parsed.block_bytes == 0 || parsed.iterations == 0 ||
      parsed.block_bytes % parsed.token_bytes != 0 ||
      parsed.block_bytes > INT32_MAX ||
      parsed.blocks > static_cast<std::uint32_t>(INT32_MAX) /
                          (parsed.block_bytes / parsed.token_bytes) ||
      parsed.blocks > std::numeric_limits<std::size_t>::max() / parsed.block_bytes) return false;
  const std::uint64_t payload_bytes = static_cast<std::uint64_t>(parsed.blocks) * parsed.block_bytes;
  if (payload_bytes > (1ULL << 32U)) return false;
  *options = parsed;
  return true;
}

}  // namespace kvaf::aicpu_urma_client
