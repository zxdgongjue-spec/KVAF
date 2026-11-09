#pragma once

#include <mutex>
#include "aicpu_access_operator.h"
#include "aicpu_urma/session.h"

namespace kvaf {
namespace aicpu_urma { class BatchPutTransport; class BatchGetTransport; }
namespace cpu_push { class CpuPushTransport; }

// Per-invocation state is held by ExecutionContext; no communication handle is
// owned here. The resource owner only borrows the currently bound invocation.
struct AicpuInvocation final : ExecutionState {
  explicit AicpuInvocation(const AicpuExecutionResources* value) : owner(value) {}
  const AicpuExecutionResources* owner;
  PutCompletion put_completion;
  aicpu_urma::UmdkSession::WriteState write;
  std::unique_ptr<ExecutionState> put, get, push;
};

// These are concrete, non-owning views. They cannot be constructed with a
// mismatched session/transport pair or converted across execution directions.
class UrmaAicpuPutView {
  friend class AicpuExecutionResources;
  friend class UrmaAicpuPutOperator;
  const AicpuExecutionResources* owner_{};
  aicpu_urma::BatchPutTransport* transfers_{};
  aicpu_urma::UmdkSession* session_{};
  const RemoteRegionDesc* region_{};
};
class UrmaAicpuGatherView {
  friend class AicpuExecutionResources;
  friend class UrmaAicpuGatherOperator;
  const AicpuExecutionResources* owner_{};
  aicpu_urma::BatchGetTransport* gather_{};
};
class UrmaAicpuScatterView {
  friend class AicpuExecutionResources;
  friend class UrmaAicpuScatterOperator;
  const AicpuExecutionResources* owner_{};
  cpu_push::CpuPushTransport* push_{};
};

// Communication sessions, registrations and execution-slot storage live here,
// independently of the operator adapter. Close never calls back into an operator.
class AicpuExecutionResources final : public ExecutionResourceOwner {
 public:
  explicit AicpuExecutionResources(UmdkAccessOperatorConfig);
  ~AicpuExecutionResources() override;
  Status Initialize(const RemoteRegionDesc&) override;
  Status Warmup(const ExecutionWarmup&) override;
  Status Close() override;
  std::unique_ptr<ExecutionState> CreateInvocation();
  Status Bind(ExecutionState*);
  AicpuInvocation& invocation();
  bool IsBound(const AicpuInvocation&) const;
  bool managed_completion() const;
  bool scatter() const;
  std::mutex& put_mutex();
  std::mutex& get_mutex();
  // Preparation and Launch run under the same exclusive direction lease.
  Status PreparePut(const BatchPutExecRequest&);
  Status PrepareGet(const BatchGetExecRequest&);
  Status ReleasePutBuffers();
  Status ReleaseGetBuffers();
  Status ResolvePut(PutCompletion*);
  Status AcknowledgePut();
  Status DrainGet();
  UrmaAicpuPutView PutView();
  UrmaAicpuGatherView GatherView();
  UrmaAicpuScatterView ScatterView();
 private:
  struct State;
  std::unique_ptr<State> state_;
};

// Algorithms receive already prepared resource views and explicit call state.
// They neither create nor destroy sessions, register memory, or own workspaces.
class UrmaAicpuPutOperator {
 public:
  static Status Launch(const BatchPutExecRequest&, const UrmaAicpuPutView&, AicpuInvocation&);
};
class UrmaAicpuGatherOperator {
 public:
  static Status Launch(const BatchGetExecRequest&, const UrmaAicpuGatherView&, AicpuInvocation&);
};
class UrmaAicpuScatterOperator {
 public:
  static Status Launch(const BatchGetExecRequest&, const UrmaAicpuScatterView&, AicpuInvocation&);
};
}  // namespace kvaf
