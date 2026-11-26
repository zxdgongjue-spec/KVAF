#pragma once
#include "sequential_batch_get.h"
#include "scatter_batch_get_abi.h"

extern "C" int32_t KvafScatterBatchGetLoadKernel(const char* path);
extern "C" int32_t KvafScatterBatchGetLaunch(const KvafScatterBatchGetArgs* args, void* stream);
namespace kvaf {
struct ScatterLaunchView {
  uint32_t cores, token_bytes, jettys_per_core;
  uint64_t status_address, mailbox_address;
};
Status PrepareScatterBatchGetArguments(const BatchGetExecRequest&, ScatterLaunchView,
                                       KvafScatterBatchGetArgs&);
std::unique_ptr<SequentialBatchGet> CreateScatterBatchGet(AccessOperatorConfig config);
}
