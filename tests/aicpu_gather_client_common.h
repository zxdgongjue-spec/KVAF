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

inline bool Number(const char* text, std::uint64_t* value) {
  if (text == nullptr || text[0] == '\0' || text[0] == '-') return false;
  char* end = nullptr;
  errno = 0;
  const auto parsed = std::strtoull(text, &end, 0);
  if (errno != 0 || end == text || *end != '\0') return false;
  *value = static_cast<std::uint64_t>(parsed);
  return true;
}

}  // namespace kvaf::aicpu_urma_client
