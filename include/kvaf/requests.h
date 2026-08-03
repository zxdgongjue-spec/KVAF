#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kvaf/types.h"

namespace kvaf {

// Runtime 初始化时由 Runtime 自己申请并注册的远端 DDR 资源规格。
// BatchGet 的调用方 buffer 不属于这里：它们由上层在每次调用前准备。
struct RuntimeInitRequest {
  std::uint64_t ddr_bytes{0};
  std::uint64_t slot_size{0};
  DeviceKind ddr_device{DeviceKind::kCpu};
};

// 以扁平 Key 顺序准备 AccessTable。POC 的 Operator 使用一维表；上层
// 需要保留自己的逻辑 shape，并以 indices 指向这张表。
struct AccessTablePrepareRequest {
  std::vector<KeyId> flat_keys;
};

// Runtime 内部 AccessTable 的只读宿主视图。view 只在关联 Handle 有效时
// 可用；NPU 调用方可在图外将其中 address/valid 拷贝到自己的 Device buffer。
struct AccessTableView {
  const Address* addresses{nullptr};
  const std::uint64_t* lengths{nullptr};
  const std::uint8_t* valid{nullptr};
  std::size_t size{0};
};

// NPU BatchGet 使用的、由调用方预先申请且已完成链路注册的表 buffer。
// Runtime 不拥有这些 buffer，也不在它们上面执行隐式 Host/Device 拷贝。
struct AccessTableDeviceView {
  ConstBufferView addresses;
  ConstBufferView valid;

  bool empty() const {
    return addresses.data == nullptr && addresses.bytes == 0 &&
           valid.data == nullptr && valid.bytes == 0;
  }
};

struct BatchGetRequest {
  AccessTableHandle access_table;

  // indices/output_indices 为 int32[n]；mask 为空时所有请求均有效。
  ConstBufferView indices;
  ConstBufferView mask;
  ConstBufferView output_indices;
  MutableBufferView output;
  std::uint64_t output_block_bytes{0};

  // CPU path 留空。NPU path 必须由调用方提供与 handle 对应的稳定 Device 表。
  AccessTableDeviceView device_access_table;
};

struct BatchPutRequest {
  std::vector<KeyId> keys;
  std::vector<ConstBufferView> blocks;
};

struct BatchFreeRequest {
  std::vector<KeyId> keys;
};

// Runtime 向 AccessOperator 提交的同步 BatchGet 描述。
struct BatchGetExecRequest {
  AccessTableView host_access_table;
  const BatchGetRequest* request{nullptr};
};

// Runtime 向 AccessOperator 提交的同步 BatchPut 描述。put_desc 的具体
// 类型由 MetaManager 定义，因此该请求在 access_operator.h 中完整定义。
struct MemoryRegionDesc {
  Address base{0};
  std::uint64_t length{0};
  DeviceKind device{DeviceKind::kCpu};
};

}  // namespace kvaf
