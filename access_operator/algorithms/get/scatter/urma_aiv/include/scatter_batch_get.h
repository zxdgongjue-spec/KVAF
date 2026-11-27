#pragma once
#include <acl/acl.h>
#include "access_operator.h"
#include "execution_view.h"
#include "scatter_batch_get_abi.h"

extern "C" int32_t KvafScatterBatchGetLoadKernel(const char* path);
extern "C" int32_t KvafScatterBatchGetLaunch(const KvafScatterBatchGetArgs* args, void* stream);
namespace kvaf {
struct ScatterResources;

struct ScatterInvocation final : ExecutionState {
  explicit ScatterInvocation(const ScatterResources* resources) : owner(resources) {}
  bool HasPendingResources() const { return pending != nullptr; }
  const ScatterResources* owner;
  aclrtStream pending{};
  bool prepared{}, captured{}, recovery_fenced{};
};

struct ScatterProgress {
  ExecutionHealth health;
  // All nodes of this graph share the transport and status slab.
  struct Graph {
    aclrtStream stream{};
    aclmdlRI id{};
    uint32_t nodes{}, leases{};
    bool checked{}, retiring{};
  } graph;
  uint64_t sequence{};
  bool profile_ready{};
};

struct ScatterLaunchView {
  ExecutionViewCommon common;
  const KvafScatterBatchGetArgs& args;
  ScatterProgress& progress;
  uint32_t cores, token_bytes, jettys_per_core;
  void* status_device;
  void* status_host;
  uint64_t mailbox_address;
  bool initialized;

  uint64_t StatusBytes() const { return cores * sizeof(KvafSequentialStatus); }
  uint64_t ProfileBytes() const { return cores * sizeof(ScatterBatchGetCoreProfile); }
  Status AclFailure(const char* phase, aclError code) const;
};

class ScatterHostOperator {
 public:
  static Status Execute(const BatchGetExecRequest&, const ScatterLaunchView&, ScatterInvocation& active,
                        ScatterInvocation& eager);
  static Status Drain(const ScatterLaunchView&, const ScatterInvocation& active, ScatterInvocation& eager,
                      GetProviderTiming* timing = nullptr);
  static Status PrepareGraph(const GetGraphOptions&, const ScatterLaunchView&, ScatterInvocation& node,
                             const ScatterInvocation& eager);
  static Status CheckGraph(void*, const ScatterLaunchView&, const ScatterInvocation& node);
  static Status FinishGraph(void*, bool discard, const ScatterLaunchView&, ScatterInvocation& node);
  static Status CopyProfiles(std::vector<ScatterBatchGetCoreProfile>*, const ScatterLaunchView&,
                             const ScatterInvocation& active, ScatterInvocation& eager);
};
}  // namespace kvaf
