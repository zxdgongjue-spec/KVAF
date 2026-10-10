#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace kvaf::diagnostics {
// Opt-in only: production behavior is unchanged without this exact value.
inline bool Enabled() {
  const char* value = std::getenv("KVAF_DIAGNOSTICS");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool ReadUnsigned(const char* name, std::uint32_t fallback,
                         std::uint32_t* output) {
  *output = fallback;
  const char* value = std::getenv(name);
  if (value == nullptr) return true;
  if (*value == '\0') return false;
  std::uint32_t parsed = 0;
  for (const char* cursor = value; *cursor; ++cursor) {
    if (*cursor < '0' || *cursor > '9') return false;
    const auto digit = static_cast<std::uint32_t>(*cursor - '0');
    if (parsed > (std::numeric_limits<std::uint32_t>::max() - digit) / 10U)
      return false;
    parsed = parsed * 10U + digit;
  }
  *output = parsed;
  return true;
}
}  // namespace kvaf::diagnostics
