#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "types.h"

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
  // CPU BatchGet 必须提供有效 handle。NPU BatchGet 使用 DeviceAccessTable
  // 重载时也会提供 handle，以便 Runtime 验证 Host 表生命周期；
  // 直接传入 addresses/valid Tensor 时 handle 可为空。
  AccessTableHandle access_table;

  // indices/offsets/output_indices 为 int32[n]；offsets 是表项内的字节偏移。
  // miss_count 为 int32[1]，只有三个请求数组的 [0, miss_count) 前缀有效。
  // NPU path 由 output_block_bytes 和 tokens_per_block 定义单 token 搬运大小
  // 与表项布局；CPU path 使用实际 AccessTable entry 长度校验 offset。
  ConstBufferView indices;
  ConstBufferView offsets;
  ConstBufferView miss_count;
  ConstBufferView output_indices;
  MutableBufferView output;
  std::uint64_t output_block_bytes{0};
  std::uint64_t tokens_per_block{0};

  // CPU path 留空。NPU path 必须提供调用方持有的稳定 Device 表。
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
  // Caller guarantees contiguous, correctly typed NPU tensors and their lifetime.
  // The gather fast path is eager only; storage and stream are read each call.
  bool trusted_npu_request{false};
};

// Runtime 向 AccessOperator 提交的 BatchGet 描述。
struct BatchGetExecRequest {
  // 直接传入 NPU addresses/valid Tensor 时 Host view 可为空。
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
