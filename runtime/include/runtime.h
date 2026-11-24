#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "access_operator.h"
#include "meta_manager.h"
#include "requests.h"
#include "status.h"
#include "types.h"
#include "cpu_pull_executor.h"

namespace kvaf {

// One write record for ordinary Put and CPU Pull. Handles retain completed CPU
// records; ordinary Put reuses a private record to keep steady-state allocation flat.
struct ConnectorPutState {
  std::shared_ptr<const int> owner;
  PutContext transaction;
  PutAccessDesc allocation;
  RemoteRegionDesc region;
  CpuPullRequest request;
  PutCompletion completion;
  // Public handle progress can be read while a Core operation is running.
  // Transaction decisions and all other mutations remain under put_mutex_.
  std::atomic<CpuPullState> state{CpuPullState::Prepared};
  std::atomic<bool> submitted{false}, abort_requested{false};
  std::optional<bool> decision;
  std::optional<PutExecutionConfig> config;
  bool cpu{false}, metadata_finalized{false}, input_bound{false};
};
class PutHandle {
 public:
  std::uint64_t id() const;
  const std::vector<PutAllocation>& entries() const;
  CpuPullState state() const;
  bool submitted() const;
  bool aborting() const;
 private:
  friend class RuntimeCore;
  std::shared_ptr<ConnectorPutState> state_;
};


// Runtime roles share Link lifecycle but do not necessarily support the same
// data-plane operations. A target intentionally carries only host:port for
// the first CPU--NPU deployment; EID and UB routing stay inside KVAF.
struct LinkTarget {
  std::string address;
};

using LinkId = std::uint64_t;

struct RuntimeCapabilities {
  bool batch_get{false};
  bool batch_put{false};
  bool incoming_links{false};
  bool outgoing_links{false};
  bool cpu_offload{false};
  bool npu_pull{false};
};

// Role-neutral Runtime interface. RuntimeService dispatches incoming control
// connections through this interface; concrete role runtimes own all hardware
// resources and decide which Link directions they currently support.
class Runtime {
 public:
  virtual ~Runtime() = default;

  virtual Status LinkTo(const LinkTarget& target) = 0;
  virtual Status AcceptLink(const LinkTarget& peer) = 0;
  virtual Status ReleaseLink(LinkId link_id) = 0;
  virtual Status ServeConnection(int connection_fd) = 0;
  virtual Status Shutdown() = 0;
  virtual bool initialized() const = 0;
  virtual RuntimeCapabilities capabilities() const = 0;
};

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

// RuntimeCore 是同步 POC 的控制面实现：它拥有 DDR 生命周期，编排
// MetaManager 的 Put/Lock/Lookup/Free 语义，并把数据面请求交给 Operator。
// 角色 Runtime 通过组合它实现当前 NPU KV 访问能力。
class RuntimeCore {
 public:
  RuntimeCore(MetaManager& meta_manager, AccessOperator& access_operator,
              DdrMemoryAllocator& ddr_allocator);

  ~RuntimeCore();

  RuntimeCore(const RuntimeCore&) = delete;
  RuntimeCore& operator=(const RuntimeCore&) = delete;

  Status Initialize(const RuntimeInitRequest& request);
  Status Shutdown();

  bool initialized() const { return initialized_; }
  bool needs_cleanup() const { return needs_cleanup_; }
  const RemoteRegionDesc& remote_region() const { return remote_region_; }

  Status PrepareAccessTable(const AccessTablePrepareRequest& request,
                            AccessTableHandle* handle);
  Status GetAccessTableView(const AccessTableHandle& handle,
                            AccessTableView* view) const;
  Status ReleaseAccessTable(const AccessTableHandle& handle);

  Status BatchGet(
      const BatchGetRequest& request,
      const BatchGetExecutionOptions& execution = BatchGetExecutionOptions{},
      BatchGetHostTiming* host_timing = nullptr);
  Status BatchPut(const BatchPutRequest& request,
                  BatchPutExecutionOptions execution = {});
  Status BatchFree(const BatchFreeRequest& request);
  Status DrainGet();
  Status RecoverGet(GetExecutionConfig);
  bool HasPendingGet() const;
  // Resolve the already submitted Put, without replay or closing the Runtime.
  Status RecoverPut(PutExecutionConfig);
  bool HasPendingPut() const;
  Status BeginGetGraph(GetExecutionConfig, const GetGraphOptions&, std::uint64_t*);
  Status EndGetGraph(std::uint64_t, void* stream, bool discard_result = false);
  Status BeginGetOutput(GetExecutionConfig, const GetOutputOptions&, std::uint64_t*);
  Status EndGetOutput(std::uint64_t);
  Status BeginPutInput(PutExecutionConfig, const PutInputOptions&, std::uint64_t*);
  Status EndPutInput(std::uint64_t);

  Status SetCpuPullExecutor(CpuPullExecutor*);
  bool pending_offload() const;
  Status PreparePut(const std::vector<KeyId>&, const std::vector<std::uint64_t>&, PutHandle*);
  Status SubmitCpuPull(const ReadyToken&, const PutHandle&, Uint64ArrayView, Uint64ArrayView, Uint64ArrayView);
  Status SubmitCpuPullAndWait(const ReadyToken&, const PutHandle&, Uint64ArrayView, Uint64ArrayView,
                              Uint64ArrayView, std::uint32_t, CpuPullSubmitTiming* = nullptr);
  Status QueryCpuPull(const PutHandle&, CpuPullState*);
  Status WaitCpuPull(const PutHandle&, std::uint32_t, CpuPullState*);
  Status CancelCpuPull(const ReadyToken&);
  Status CommitPut(const PutHandle&);
  Status AbortPut(const PutHandle&);
  Status BatchCpuPull(const std::vector<KeyId>&, const ReadyToken&, Uint64ArrayView,
                      const std::vector<std::uint64_t>&, std::uint32_t, CpuPullProfile* = nullptr);
  Status ResolveCpuPull(std::uint32_t);
  Status LastOffloadChecksum(std::uint64_t*) const;


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

    // NPU BatchGet requires every valid lookup entry to match its configured
    // tokens_per_block * output_block_bytes layout. Compute the invariant once
    // while preparing the table rather than rescanning it for every BatchGet.
    bool has_valid_entries{false};
    bool has_uniform_valid_length{true};
    std::uint64_t uniform_valid_length{0};
  };

  AccessTableState* FindAccessTable(const AccessTableHandle& handle);
  const AccessTableState* FindAccessTable(
      const AccessTableHandle& handle) const;

  Status DrainGetLocked();
  struct PutTraceRow;
  Status FinalizePutLocked(PutTraceRow* timing = nullptr);
  void FlushPutTrace();
  struct PutTraceRow {
    const char* event{"batch_put"};
    const char* path{"direct"};
    int result{};
    std::uint64_t total_ns{}, lock_ns{}, validation_ns{}, prepare_ns{}, metadata_prepare_ns{};
    std::uint64_t execute_ns{}, metadata_finalize_ns{}, acknowledge_ns{}, resolve_ns{};
    PutDispatchTiming dispatch;
    PutProviderTiming provider;
  };
  std::string put_trace_path_;
  std::vector<PutTraceRow> put_trace_rows_;
  std::uint64_t put_trace_index_{};
  Status SubmitCpuPullImpl(const ReadyToken&, const PutHandle&, Uint64ArrayView, Uint64ArrayView,
                           Uint64ArrayView, const std::uint32_t*, CpuPullSubmitTiming*);
  Status QueryCpuPullLocked(const PutHandle&, CpuPullState*);
  Status FinishConnectorPutLocked(const PutHandle&, bool commit);
  CpuPullExecutor* cpu_pull_{nullptr};
  std::shared_ptr<const int> connector_owner_{std::make_shared<const int>(0)};
  std::shared_ptr<ConnectorPutState> ordinary_write_{std::make_shared<ConnectorPutState>()};
  PutHandle batch_pull_put_;
  bool closing_{false};
  mutable std::shared_mutex lifecycle_mutex_;
  mutable std::mutex table_mutex_;
  mutable std::mutex put_mutex_;
  mutable std::mutex get_mutex_;
  std::shared_ptr<AccessTableState> pending_get_table_;
  std::unordered_map<std::uint64_t, std::shared_ptr<AccessTableState>> graph_tables_;
  std::shared_ptr<ConnectorPutState> pending_put_;
  std::atomic<bool> initialized_{false};
  std::atomic<bool> needs_cleanup_{false};
  RemoteRegionDesc remote_region_;
  std::uint64_t next_access_table_id_{1};
  std::unordered_map<std::uint64_t, std::shared_ptr<AccessTableState>>
      access_tables_;
};

}  // namespace kvaf
