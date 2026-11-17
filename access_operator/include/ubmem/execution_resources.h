#pragma once

#include "../access_operator.h"
#include "execution_view.h"
#include "ubmem/batch_abi.h"
#include "ubmem/device_mapping.h"
#include <acl/acl.h>
#include <mutex>
#include <unordered_map>

namespace kvaf::ubmem {
struct OperatorOptions {
  std::string batch_get_kernel_object_path;
  std::string batch_put_kernel_object_path;
  std::uint32_t device_id{0};
  std::uint32_t cores{2};
  std::shared_ptr<peer::PeerLink> peer_link;
  std::shared_ptr<DeviceMappingCache> mappings;
};

// Reusable storage belongs to the resource owner, never to the kernel algorithm.
struct ExecutionSlot {
  enum class Pending { kNone, kEvent, kStream };
  aclrtEvent event{};
  void* statuses{};
  void* host_statuses{};
  void* metadata{};
  std::size_t capacity{};
};
class UbmemResources;
struct UbmemInvocation final : ExecutionState {
  explicit UbmemInvocation(const UbmemResources* value) : owner(value) {}
  const UbmemResources* owner;
  std::unordered_map<aclrtStream, ExecutionSlot::Pending> pending;
  bool graph_active{}, graph_recorded{};
  GetGraphOptions graph_options;
  aclrtStream profile_stream{};
  aclrtStream put_stream{};
  std::uint64_t put_rows{};
  bool put_issued{};
  PutCompletion put_completion;
  aclrtStream prepared_stream{};
};

// Borrowed concrete launch views. No allocator, registration, session teardown,
// or operator pointer is reachable through these views.
struct MappedRegionView {
  std::uint64_t base{}, bytes{};
};
struct AivKernelView {
  aclrtFuncHandle function{};
  std::uint32_t cores{};
};
struct MappedExecutionView {
  ExecutionViewCommon common;
  MappedRegionView memory;
  AivKernelView kernel;
  bool profiling{}, profile_detail{};
  ExecutionSlot* slot{};
  ExecutionHealth* health{};
};
class UbmemPutView {
 public:
  UbmemPutView() = default;
 private:
  friend class UbmemResources;
  friend class UbmemPutOperator;
  explicit UbmemPutView(MappedExecutionView view) : execution_(view) {}
  MappedExecutionView execution_;
};
class UbmemGetView {
 public:
  UbmemGetView() = default;
 private:
  friend class UbmemResources;
  friend class UbmemGetOperator;
  explicit UbmemGetView(MappedExecutionView view) : execution_(view) {}
  MappedExecutionView execution_;
};

class UbmemPutOperator {
 public:
  static Status Launch(const BatchPutExecRequest&, const UbmemPutView&, UbmemInvocation&);
};
class UbmemGetOperator {
 public:
  static Status Launch(const BatchGetExecRequest&, const UbmemGetView&, UbmemInvocation&);
};

// Owns mapping, modules and stream slots independently of the compatibility
// AccessOperator. The manager can initialize, warm and close it directly.
class UbmemResources final : public ExecutionResourceOwner {
 public:
  explicit UbmemResources(OperatorOptions value) : options(std::move(value)) { views.Bind(invocation); }
  ~UbmemResources() override;
  Status Initialize(const RemoteRegionDesc&) override;
  Status Warmup(const ExecutionWarmup&) override;
  Status Close() override;
  // Preparation requires the caller to hold the exclusive resource lease.
  Status PreparePutResources(aclrtStream, std::uint32_t capacity, UbmemPutView*);
  Status PrepareGetResources(aclrtStream, UbmemGetView*);
  Status Bind(ExecutionState*);
  std::unique_ptr<ExecutionState> CreateState() const;
 private:
  friend class AccessOperator;  // Compatibility adapter, not a kernel algorithm.
  Status LoadKernel(unsigned, const std::string&, const char*);
  Status Slot(aclrtStream, ExecutionSlot**);
  Status ReserveMetadata(ExecutionSlot&, std::uint32_t capacity);
  MappedExecutionView View(unsigned, aclrtStream, ExecutionSlot&);
  ExecutionViewState views;
  ExecutionHealth health;
  mutable std::mutex mutex;
  OperatorOptions options;
  std::shared_ptr<DeviceMapping> mapping;
  RemoteRegionDesc remote{};
  aclrtBinHandle binaries[2]{};
  aclrtFuncHandle functions[2]{};
  std::vector<char> binary_bytes[2];
  std::unordered_map<aclrtStream, ExecutionSlot> streams;
  bool ready{}, retained{}, profiling{}, profile_detail{};
  // Keeps initialization and Close valid before binding and after unbinding.
  UbmemInvocation default_invocation{this};
  // Borrowed while the execution context holds its resource lease. Close uses
  // this state to fence work before releasing any physical storage.
  UbmemInvocation* invocation{&default_invocation};
};
}  // namespace kvaf::ubmem
