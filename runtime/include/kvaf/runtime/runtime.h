#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "kvaf/access_operator/access_operator.h"
#include "kvaf/dsa_row/requests.h"
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

  // DSA_ROW M0 -------------------------------------------------------------
  // These APIs are intentionally independent from the legacy KeyId based
  // MetaManager path above.  They implement the fixed row-major directory
  // contract used by DSA cache offload and are synchronous on the CPU
  // reference backend.
  Status InitializeDsaRowIndex(const DsaRowInitRequest& request);
  Status GetDsaRowDirectoryView(DsaRowDirectoryView* view) const;
  Status GetDsaRowCommittedBlockCount(std::uint32_t pool_row,
                                      std::uint32_t* count) const;
  Status BindResidentPoolRow(const RowBindRequest& request);
  Status AbortBindResidentPoolRow(std::uint32_t pool_row);

  Status PrepareFullBlockGroupPut(
      const std::vector<KvafPutBlockTarget>& targets,
      const std::vector<RowCountCommitSpec>& row_commits,
      DsaPutPlanHandle* plan);
  Status WriteFullBlockGroupLayer(
      const DsaPutPlanHandle& plan,
      const std::vector<KvafPutBlockItem>& layer_blocks,
      const BoundLayerCacheView& layer_cache, void* stream = nullptr);
  Status SealFullBlockGroupPut(const DsaPutPlanHandle& plan,
                               DsaSealResult* result);
  Status CommitFullBlockGroupPut(DsaPutPlanHandle* plan,
                                 PutLocalResult* result);
  Status AbortFullBlockGroupPut(DsaPutPlanHandle* plan,
                                PutLocalResult* result);

  Status PrepareUnifiedAccess(const PrepareUnifiedAccessRequest& request,
                              PrepareUnifiedAccessResult* result);
  Status PrepareDirectBatchGet(const DsaRowAccessHandle& access,
                               const DirectGetRequest& request,
                               DsaDirectGetPlanHandle* plan);
  Status ExecutePreparedDirectBatchGet(const DsaDirectGetPlanHandle& plan,
                                       DirectGetCompletion* completion);
  Status SettleDirectBatchGet(DsaDirectGetPlanHandle* plan,
                              const DirectGetCompletion& completion,
                              GetLocalResult* result);
  Status AbortPreparedDirectBatchGet(DsaDirectGetPlanHandle* plan,
                                     GetLocalResult* result);
  Status ReleaseUnifiedAccess(DsaRowAccessHandle* access);
  Status CloseResidentPoolRow(std::uint32_t pool_row,
                              CloseLocalResult* result);

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

    // The current NPU BatchGet ABI has one block_bytes value per launch, not
    // one length per lookup entry. Compute this table-wide invariant once
    // while preparing the table rather than rescanning it for every BatchGet.
    bool has_valid_entries{false};
    bool has_uniform_valid_length{true};
    std::uint64_t uniform_valid_length{0};
  };

  class DsaRowState;
  struct DsaRowStateDeleter {
    void operator()(DsaRowState* state) const;
  };

  AccessTableState* FindAccessTable(const AccessTableHandle& handle);
  const AccessTableState* FindAccessTable(
      const AccessTableHandle& handle) const;
  Status CheckDsaRowShutdown() const;

  bool initialized_{false};
  RemoteRegionDesc remote_region_;
  std::uint64_t next_access_table_id_{1};
  std::unordered_map<std::uint64_t, std::unique_ptr<AccessTableState>>
      access_tables_;
  std::unique_ptr<DsaRowState, DsaRowStateDeleter> dsa_row_state_;
  // The M0 DSA directory owns the complete region slot map.  Keep legacy
  // writes and DSA_ROW mutually exclusive until the allocator is unified.
  bool legacy_data_plane_used_{false};
};

}  // namespace kvaf
