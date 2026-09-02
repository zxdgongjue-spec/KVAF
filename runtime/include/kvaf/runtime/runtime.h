#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "kvaf/access_operator/access_operator.h"
#include "kvaf/meta_manager/meta_manager.h"
#include "kvaf/requests.h"
#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf {

// Runtime 通过该接口申请并释放内部远端 DDR。硬件实现可以在 Allocate 中
// 使用 ACL/HComm；测试实现可以使用普通 Host 内存。返回的 base 必须可被
// AccessOperator 解释为远端 UBA。
class DdrMemoryAllocator {
 public:
  virtual ~DdrMemoryAllocator() = default;

  virtual Status Allocate(const RuntimeInitRequest& request,
                          RemoteRegionDesc* remote_region) = 0;
  virtual Status Release(const RemoteRegionDesc& remote_region) = 0;
};

// 为功能测试和 CPU 数据面提供的 Host DDR allocator。NPU 部署应注入
// 与实际 ACL/HComm 注册流程匹配的 allocator。
std::unique_ptr<DdrMemoryAllocator> CreateHostDdrMemoryAllocator();

// Runtime 是同步 POC 的控制面入口：它拥有 DDR 生命周期，编排
// MetaManager 的 Put/Lock/Lookup/Free 语义，并把数据面请求交给 Operator。
class Runtime {
 public:
  Runtime(MetaManager& meta_manager, AccessOperator& access_operator,
          DdrMemoryAllocator& ddr_allocator);

  ~Runtime();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  Status Initialize(const RuntimeInitRequest& request);
  Status Shutdown();

  bool initialized() const { return initialized_; }
  const RemoteRegionDesc& remote_region() const { return remote_region_; }

  Status PrepareAccessTable(const AccessTablePrepareRequest& request,
                            AccessTableHandle* handle);
  Status GetAccessTableView(const AccessTableHandle& handle,
                            AccessTableView* view) const;
  Status ReleaseAccessTable(const AccessTableHandle& handle);

  Status BatchGet(
      const BatchGetRequest& request,
      const BatchGetExecutionOptions& execution = BatchGetExecutionOptions{});
  Status BatchPut(const BatchPutRequest& request,
                  BatchPutExecutionOptions execution = {});
  Status BatchFree(const BatchFreeRequest& request);

 protected:
  MetaManager& meta_manager_;
  AccessOperator& access_operator_;
  DdrMemoryAllocator& ddr_allocator_;

 private:
  struct AccessTableState {
    LockContext lock_context;
    std::vector<Address> addresses;
    std::vector<std::uint64_t> lengths;
    std::vector<std::uint8_t> valid;

    // NPU BatchGet requires every valid lookup entry to have the same length.
    // The expected 128 * output_block_bytes size is checked per BatchGet
    // without rescanning the table.
    bool has_valid_entries{false};
    bool has_uniform_valid_length{true};
    std::uint64_t uniform_valid_length{0};
  };

  AccessTableState* FindAccessTable(const AccessTableHandle& handle);
  const AccessTableState* FindAccessTable(
      const AccessTableHandle& handle) const;

  bool initialized_{false};
  RemoteRegionDesc remote_region_;
  std::uint64_t next_access_table_id_{1};
  std::unordered_map<std::uint64_t, std::unique_ptr<AccessTableState>>
      access_tables_;
};

}  // namespace kvaf
