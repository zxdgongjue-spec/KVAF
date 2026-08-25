#pragma once

#include <cstddef>
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

// 远端 DDR 区域。base 是可供 AccessOperator 使用的绝对地址（UBA）。
// Runtime 通过 DdrMemoryAllocator 创建并持有该区域的生命周期。
struct RemoteRegionDesc {
  Address base{0};
  std::uint64_t length{0};
  std::uint64_t slot_size{0};
  DeviceKind device{DeviceKind::kCpu};
  // One Runtime owns one physical region. MetaManager may subdivide it into
  // fixed per-layer ranges so different transformer layers never interleave
  // their DDR allocations.
  std::uint32_t layer_count{1};
};

// 调用方拥有的只读/可写 buffer 描述。Runtime 不申请、释放或注册这些
// BatchGet 输入输出 buffer；它们必须在调用返回前保持有效。
struct ConstBufferView {
  const void* data{nullptr};
  std::size_t bytes{0};
  DeviceKind device{DeviceKind::kCpu};

  bool empty() const { return data == nullptr && bytes == 0; }
  bool valid() const { return data != nullptr && bytes != 0; }

  template <typename Element>
  bool has_elements() const {
    return valid() && bytes % sizeof(Element) == 0;
  }
};

struct MutableBufferView {
  void* data{nullptr};
  std::size_t bytes{0};
  DeviceKind device{DeviceKind::kCpu};

  bool empty() const { return data == nullptr && bytes == 0; }
  bool valid() const { return data != nullptr && bytes != 0; }

  template <typename Element>
  bool has_elements() const {
    return valid() && bytes % sizeof(Element) == 0;
  }
};

// 当前同步 Runtime 不暴露 Event；保留类型以免破坏已有占位调用方。
struct Event {
  std::uint64_t opaque{0};
};

// AccessTable 的所有权与 Key 稳定性锁由 Runtime 保存。opaque 为 0 的
// Handle 无效；ReleaseAccessTable 后不得再次使用。
struct AccessTableHandle {
  std::uint64_t opaque{0};
};

}  // namespace kvaf
