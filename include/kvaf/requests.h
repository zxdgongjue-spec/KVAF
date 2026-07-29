#pragma once

#include <cstddef>
#include <cstdint>

#include "kvaf/types.h"

namespace kvaf {

// 这些结构当前只固定模块之间的类型名称和依赖方向。
// 字段应在对应接口细化时补充，不在占位仓中提前固化实现细节。
struct BatchGetRequest {
  const void* opaque{nullptr};
  std::size_t size{0};
};

struct BatchPutRequest {
  const void* opaque{nullptr};
  std::size_t size{0};
};

struct DeleteRequest {
  const void* opaque{nullptr};
  std::size_t size{0};
};

struct LookupRequest {
  const void* opaque{nullptr};
  std::size_t size{0};
};

struct MemoryRegionDesc {
  Address base{0};
  std::uint64_t length{0};
  DeviceKind device{DeviceKind::kCpu};
};

}  // namespace kvaf
