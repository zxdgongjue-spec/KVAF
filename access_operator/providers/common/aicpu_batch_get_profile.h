#pragma once

#include <acl/acl.h>
#include "access_operator.h"
#include "batch_get_profile.h"

namespace kvaf::detail {
inline Status AllocateAicpuProfiles(void** device, void** host, std::size_t bytes) {
  if (!*device && (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
                   aclrtMemset(*device, bytes, 0, bytes) != ACL_SUCCESS))
    return Status::Internal("cannot allocate AICPU profiling buffer");
  if (!*host && aclrtMallocHost(host, bytes) != ACL_SUCCESS)
    return Status::Internal("cannot allocate AICPU profiling staging");
  return Status::OK();
}

// As with AIV snapshots, the caller completes the stream before this D2H.
inline Status CopyAicpuProfiles(void* device, void* host, std::uint32_t workers,
                                 std::vector<BatchGetKernelProfile>* profiles) {
  if (!profiles) return Status::InvalidArgument("profile output is null");
  const auto bytes = workers * sizeof(BatchGetWorkerProfile);
  if (aclrtMemcpy(host, bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
    return Status::Internal("cannot read AICPU kernel profiling");
  profiles->clear();
  const auto* source = static_cast<const BatchGetWorkerProfile*>(host);
  for (unsigned w = 0; w < workers; ++w) {
    if (!source[w].enter_ns || source[w].exit_ns < source[w].enter_ns)
      return Status::InvalidState("no completed AICPU kernel profile");
    BatchGetKernelProfile p{};
    p.device_start_ns = source[w].enter_ns;
    p.device_finish_ns = source[w].exit_ns;
    p.active_requests = source[w].spans;
    p.aicpu_profile = source[w];
    profiles->push_back(p);
  }
  return Status::OK();
}
}  // namespace kvaf::detail
