// Compile and exercise the production reader with both public header shapes.
// No URMA installation is needed for this old-header regression test.
#include <cstdint>
#include <cstdio>

#include "urma_diagnostic_compat.h"

struct LegacyJfsFlags {
  struct {
    std::uint32_t lock_free : 1;
    std::uint32_t error_suspend : 1;
    std::uint32_t outorder_comp : 1;
    std::uint32_t sub_trans_mode : 8;
    std::uint32_t rc_share_tp : 1;
    std::uint32_t reserved : 20;
  } bs;
};

struct CurrentJfsFlags {
  struct {
    std::uint32_t lock_free : 1;
    std::uint32_t error_suspend : 1;
    std::uint32_t outorder_comp : 1;
    std::uint32_t order_type : 8;
    std::uint32_t reserved : 21;
  } bs;
};

int main() {
  LegacyJfsFlags legacy{};
  // Neither the old field at this position nor reserved bits are order_type.
  legacy.bs.sub_trans_mode = 2;
  legacy.bs.reserved = 3;
  CurrentJfsFlags current{};
  if (kvaf::detail::DiagnosticJfsOrderType(legacy, 0) != "unavailable" ||
      kvaf::detail::DiagnosticJfsOrderType(current, 0) != "0") {
    std::fputs("missing field was confused with an actual zero value\n", stderr);
    return 1;
  }
  current.bs.order_type = 2;
  if (kvaf::detail::DiagnosticJfsOrderType(current, 0) != "2") {
    std::fputs("available bitfield was not read\n", stderr);
    return 1;
  }
  std::puts("URMA diagnostic header compatibility tests passed");
  return 0;
}
