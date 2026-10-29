#pragma once

#include <cstddef>
#include <cstdint>

namespace kvaf::urma {
// Device-attribute ABI used by the deployed UB SDK.
struct UbDeviceCapabilities {
  std::uint32_t feature;
  std::uint32_t max_jfc, max_jfs, max_jfr, max_jetty, max_jetty_grp, max_jetty_in_jetty_grp;
  std::uint32_t max_jfc_depth, max_jfs_depth, max_jfr_depth, max_jfs_inline_len;
  std::uint32_t max_jfs_sge, max_jfs_rsge, max_jfr_sge;
  std::uint64_t max_msg_size;
  std::uint32_t max_read_size, max_write_size, max_cas_size, max_swap_size;
  std::uint32_t max_fetch_and_add_size, max_fetch_and_sub_size, max_fetch_and_and_size;
  std::uint32_t max_fetch_and_or_size, max_fetch_and_xor_size, atomic_feature;
  std::uint16_t trans_mode, reserved, congestion_ctrl_alg;
  std::uint32_t ceq_cnt, max_tp_in_tpg, max_eid_cnt;
  std::uint64_t page_size_cap;
  std::uint32_t max_oor_cnt, mn, max_netaddr_cnt;
  std::uint32_t rm_order_cap, rc_order_cap, rm_tp_cap, rc_tp_cap, um_tp_cap, tp_feature;
  struct PriorityInfo { std::uint32_t sl, tp_type; } priority_info[16];
};
struct UbDeviceAttributes {
  std::uint8_t guid[16];
  UbDeviceCapabilities dev_cap;
  std::uint8_t port_cnt;
  struct PortInfo {
    std::uint32_t max_mtu, state, active_width, active_speed, active_mtu;
  } port_attr[8];
  std::uint32_t reserved_jetty_id_min, reserved_jetty_id_max;
};
static_assert(sizeof(UbDeviceCapabilities) == 304 && sizeof(UbDeviceAttributes) == 496);
static_assert(offsetof(UbDeviceAttributes, dev_cap) + offsetof(UbDeviceCapabilities, max_jfs_sge) == 60);
static_assert(offsetof(UbDeviceAttributes, dev_cap) + offsetof(UbDeviceCapabilities, max_msg_size) == 72);
static_assert(offsetof(UbDeviceAttributes, port_attr) == 324);
static_assert(offsetof(UbDeviceAttributes, reserved_jetty_id_max) == 488);
}  // namespace kvaf::urma
