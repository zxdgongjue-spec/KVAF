#pragma once
#include <acl/acl.h>
#include "access_operator.h"
#include "execution_view.h"
#include "kvaf_aiv_batch_get.h"
#include "gather_batch_get_abi.h"

extern "C" int32_t KvafGatherBatchGetLoadKernel(const char* path);
extern "C" int32_t KvafGatherBatchGetLaunch(const KvafGatherBatchGetArgs* args, void* stream);
namespace kvaf {
namespace sequential_detail {
inline const char* ProbeName(uint32_t probe) {
  switch (probe) {
    case KVAF_GATHER_PROBE_ACK: return "ack";
    case KVAF_GATHER_PROBE_WQE: return "wqe";
    case KVAF_GATHER_PROBE_SQ: return "sq";
    case KVAF_GATHER_PROBE_FIRST_READ: return "first_read";
    default: return "none";
  }
}
}  // namespace sequential_detail

struct GatherResources;

struct GatherInvocation final : ExecutionState {
  explicit GatherInvocation(const GatherResources* resources) : owner(resources) {}
  bool HasPendingResources() const { return pending || output_buffer; }
  const GatherResources* owner;
  aclrtStream pending{};
  KvafBatchGetLocalBuffer* output_buffer{};
  bool pending_profile{}, profile_valid{}, pending_status_check{}, head_start{}, recovery_fenced{};
  uint64_t profile_begin_ns{};
  GatherBatchGetProfile latest_profile{};
  struct Graph {
    aclrtStream stream{};
    uint64_t initial_sequence{}, output_entity{};
    bool captured{};
  } graph;
};

// All launches share the status slab and its sequence. Recovery is scoped to
// that physical slot, while individual graph bindings stay in the invocation.
struct GatherProgress {
  ExecutionHealth health;
  GatherInvocation* live_graph{};
  uint64_t sequence{};
};

class GatherOutputRegistration {
 public:
  int Acquire(const MutableBufferView&, KvafBatchGetLocalBuffer**, uint64_t* entity);
  void Release(KvafBatchGetLocalBuffer*&);
  void Preserve(KvafBatchGetLocalBuffer*&);
 private:
  friend struct GatherResources;
  KvafBatchGetTransport* transport_{};
  KvafBatchGetLocalBuffer* retained_{};
};

struct GatherLaunchView {
  ExecutionViewCommon common;
  const KvafGatherBatchGetArgs& args;
  GatherProgress& progress;
  GatherOutputRegistration& outputs;
  uint32_t cores, token_bytes;
  void* status_device;
  void* status_host;
  void* queues;
  void* profile_device;
  void* profile_host;
  bool initialized, profiling, check_async_status;

  uint64_t StatusBytes() const { return cores * sizeof(KvafSequentialStatus); }
  uint64_t ProfileBytes() const { return cores * sizeof(KvafGatherBatchGetCoreProfile); }
  Status AclFailure(const char* phase, aclError code) const;
};

class GatherHostOperator {
 public:
  static Status Execute(const BatchGetExecRequest&, const GatherLaunchView&, GatherInvocation& active,
                        GatherInvocation& eager);
  static Status Drain(const GatherLaunchView&, const GatherInvocation& active, GatherInvocation& eager);
  static Status PrepareGraph(const GetGraphOptions&, const GatherLaunchView&, GatherInvocation& node,
                             GatherInvocation& eager);
  static Status CheckGraph(void*, const GatherLaunchView&, const GatherInvocation& node);
  static Status FinishGraph(void*, bool discard, const GatherLaunchView&, GatherInvocation& node);
  static Status CopyProfile(GatherBatchGetProfile*, const GatherLaunchView&, const GatherInvocation& active,
                            const GatherInvocation& eager);
};
}  // namespace kvaf
