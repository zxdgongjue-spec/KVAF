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
// 可用；绑定层可将其中 address/valid materialize 为直接传给 NPU BatchGet
// 的 DeviceAccessTable。
struct AccessTableView {
  const Address* addresses{nullptr};
  const std::uint64_t* lengths{nullptr};
  const std::uint8_t* valid{nullptr};
  std::size_t size{0};
};

// NPU BatchGet 使用的 device AccessTable buffer。C++ Runtime 不拥有这些
// buffer；Python binding 可从 AccessTableHandle materialize 一个直接可用的
// DeviceAccessTable。其 entry 数可以大于 Host AccessTable，以支持固定容量
// 的调用方 Tensor。
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

  // CPU path 留空。NPU path 必须提供与 handle 对应的稳定 Device 表。
  AccessTableDeviceView device_access_table;
};

struct BatchPutRequest {
  std::vector<KeyId> keys;

  // A caller-owned, contiguous buffer containing all source blocks.  It may
  // reside on CPU or NPU memory; Runtime never packs, copies, owns, or
  // releases it. input_offsets[i] and lengths[i] describe the source bytes
  // for keys[i].
  ConstBufferView input;
  std::vector<std::uint64_t> input_offsets;
  std::vector<std::uint64_t> lengths;
};

// BatchPut V1 is synchronous, but the Direct-HCCP launch still needs the
// caller's ACL stream so preceding producer work on that stream is ordered
// before the remote WRITE. stream is ignored by CPU/mock implementations.
struct BatchPutExecutionOptions {
  void* stream{nullptr};
};

struct BatchFreeRequest {
  std::vector<KeyId> keys;
};

// 一次 BatchGet 提交使用的调用方 stream。Runtime 不解释 stream 的具体
// 类型；当前 CANN Adapter 将它解释为 aclrtStream。默认只向 stream enqueue，
// 不替调用方同步；同一 stream 上的后续 Torch/ACL 操作天然按序执行。跨 stream
// 依赖、buffer 生命周期与显式 synchronize 由调用方/框架负责。
struct BatchGetExecutionOptions {
  void* stream{nullptr};
  bool synchronize{false};
};

// Runtime 向 AccessOperator 提交的 BatchGet 描述。
struct BatchGetExecRequest {
  AccessTableView host_access_table;
  const BatchGetRequest* request{nullptr};
  BatchGetExecutionOptions execution;
};

// Runtime 向 AccessOperator 提交的同步 BatchPut 描述。put_desc 的具体
// 类型由 MetaManager 定义，因此该请求在 access_operator.h 中完整定义。
struct MemoryRegionDesc {
  Address base{0};
  std::uint64_t length{0};
  DeviceKind device{DeviceKind::kCpu};
};

}  // namespace kvaf
