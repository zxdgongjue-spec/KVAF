#pragma once
#include "sequential_batch_get.h"
#include "gather_batch_get_abi.h"

extern "C" int32_t KvafGatherBatchGetLoadKernel(const char* path);
extern "C" int32_t KvafGatherBatchGetLaunch(const KvafGatherBatchGetArgs* args, void* stream);
namespace kvaf {
std::unique_ptr<SequentialBatchGet> CreateGatherBatchGet(AccessOperatorConfig config);
}
