#pragma once

#include <cstdint>

namespace kvaf {

// TODO: 与最终 Key 编码和上层 tensor/host 表示统一。
using KeyId = std::uint64_t;

// TODO: 与远端 DDR 地址表示和 UBA 定义统一。
using Address = std::uint64_t;

// 仅表示当前占位接口可见的执行设备，不代表最终 backend 选择定义。
enum class DeviceKind {
  kCpu = 0,
  kNpu,
};

// 当前只保留句柄形态；内部字段和生命周期语义后续确定。
struct Event {
  std::uint64_t opaque{0};
};

struct AccessTableHandle {
  std::uint64_t opaque{0};
};

}  // namespace kvaf
