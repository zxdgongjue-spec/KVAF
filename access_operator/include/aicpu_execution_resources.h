#pragma once

#include <mutex>
#include "aicpu_access_operator.h"
#include "aicpu_urma/session.h"
#include "aicpu_urma/launch_view.h"

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

// Same common contract across all directions; provider pieces remain typed.
struct UrmaAicpuPutView { ExecutionViewCommon common; aicpu_urma::PutLaunchView launch; };
struct UrmaAicpuGatherView { ExecutionViewCommon common; aicpu_urma::GatherLaunchView launch; };
struct UrmaAicpuScatterView { ExecutionViewCommon common; aicpu_urma::ScatterLaunchView launch; };

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
  bool managed_completion() const;
  bool scatter() const;
  std::mutex& put_mutex();
  std::mutex& get_mutex();
  // Preparation and Launch run under the same exclusive direction lease.
  Status PreparePutResources(const BatchPutExecRequest&);
  Status PrepareGetResources(const BatchGetExecRequest&);
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
