#pragma once

#include <string>

namespace kvaf::detail {

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
