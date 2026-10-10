#pragma once

#include "acl/acl.h"
#include "aicpu_urma/sq_copy_benchmark.h"

namespace kvaf::aiv_sq_copy {

constexpr std::uint32_t kLocalMemoryBytes = 16U * 1024U;

bool Load(const char* object_path);
int32_t LaunchCopy(const aicpu_urma::SqCopyBenchmarkArgs* args, aclrtStream stream);
int32_t LaunchClock(const aicpu_urma::SqCopyBenchmarkArgs* args, aclrtStream stream);

}
