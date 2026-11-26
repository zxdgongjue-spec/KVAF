#pragma once
#include "sequential_batch_get.h"
#include "gather_batch_get_abi.h"

extern "C" int32_t KvafGatherBatchGetLoadKernel(const char* path);
extern "C" int32_t KvafGatherBatchGetLaunch(const KvafGatherBatchGetArgs* args, void* stream);
namespace kvaf {
struct GatherLaunchView {
  uint32_t cores, token_bytes;
  uint64_t status_address, queue_address;
};
Status PrepareGatherBatchGetArguments(const BatchGetExecRequest&, GatherLaunchView,
                                      KvafGatherBatchGetArgs&);
std::unique_ptr<SequentialBatchGet> CreateGatherBatchGet(AccessOperatorConfig config);
}
