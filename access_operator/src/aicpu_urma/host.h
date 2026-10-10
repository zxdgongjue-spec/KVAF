#pragma once

#include <cstdint>

#include "acl/acl.h"
#include "batch_get_abi.h"
#include "batch_put_internal.h"
#include "gather_read.h"
#include "sq_copy_benchmark.h"

namespace kvaf::aicpu_urma {

bool Load(const char* json_path);
int32_t LaunchPut(const KvafExecuteBatchPutLaunchArgs* arguments, aclrtStream stream);
int32_t LaunchGet(const KvafExecuteBatchGetKernelArgs* arguments, aclrtStream stream);
int32_t LaunchGatherRead(const GatherReadArgs* arguments, aclrtStream stream);
int32_t LaunchSqCopyBenchmark(const SqCopyBenchmarkArgs* arguments, aclrtStream stream);

}  // namespace kvaf::aicpu_urma
