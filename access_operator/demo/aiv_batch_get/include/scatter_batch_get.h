#pragma once
#include "sequential_batch_get.h"
#include "scatter_batch_get_abi.h"

extern "C" int32_t KvafScatterBatchGetLoadKernel(const char* path);
extern "C" int32_t KvafScatterBatchGetLaunch(const KvafScatterBatchGetArgs* args, void* stream);
namespace kvaf {
std::unique_ptr<SequentialBatchGet> CreateScatterBatchGet(AccessOperatorConfig config);
}
