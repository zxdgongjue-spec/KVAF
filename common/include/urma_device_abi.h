#pragma once

#include <cstddef>
#include <cstdint>
#include <urma_api.h>

namespace kvaf::urma {
// Expanded UB device ABI used by the deployed SDK. Older public UMDK headers
// stop the capabilities record at max_netaddr_cnt and place the port records
// earlier. urma_query_device copies the library's complete record, not the
// caller header's sizeof(urma_device_attr_t).
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
static_assert(offsetof(UbDeviceAttributes, port_attr) == 324);
static_assert(offsetof(UbDeviceAttributes, reserved_jetty_id_max) == 488);

// Retain native-header support as well as the explicit expanded ABI. Only the
// common SGE/message-limit prefix is consumed by CPU push.
union DeviceAttributeStorage {
  urma_device_attr_t native;
  UbDeviceAttributes ub;
};
static_assert(offsetof(urma_device_attr_t, dev_cap) == offsetof(UbDeviceAttributes, dev_cap));
static_assert(offsetof(urma_device_cap_t, max_jfs_sge) == offsetof(UbDeviceCapabilities, max_jfs_sge));
static_assert(offsetof(urma_device_cap_t, max_msg_size) == offsetof(UbDeviceCapabilities, max_msg_size));
}  // namespace kvaf::urma
