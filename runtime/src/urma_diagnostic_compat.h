#pragma once

#include <string>

namespace kvaf::detail {

// URMA query APIs do not receive the caller's buffer size. A symbol being
// present (or the header compiling) does not establish the runtime ABI.
// The default CPU build can combine pinned headers with a system liburma,
// so only an explicitly confirmed deployment may use these diagnostic calls.
constexpr bool CpuUrmaQueryAbiConfirmed() {
#if defined(KVAF_CPU_URMA_QUERY_ABI_CONFIRMED) && KVAF_CPU_URMA_QUERY_ABI_CONFIRMED
  return true;
#else
  return false;
#endif
}

// UMDK 24.12's public JFS flags have no order_type member. Keep the field
// optional without assuming that reserved bits have the newer ABI's meaning.
template <typename Flags>
auto DiagnosticJfsOrderType(const Flags& flags, int)
    -> decltype(flags.bs.order_type, std::string()) {
  return std::to_string(static_cast<unsigned int>(flags.bs.order_type));
}

template <typename Flags>
std::string DiagnosticJfsOrderType(const Flags&, long) {
  return "unavailable";
}

}  // namespace kvaf::detail
