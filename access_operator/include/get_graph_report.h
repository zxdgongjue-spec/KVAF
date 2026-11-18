#pragma once

#include <cstddef>
#include <acl/acl.h>
#include "status.h"

namespace kvaf {
// Owned by the captured node. The dispatcher destroys it only after fencing
// the replay stream, or retains the node if completion is unresolved.
struct GetGraphReport {
  void* device{};
  void* host{};
  Status Allocate(std::size_t bytes, int initial = 0, bool stage = true) {
    if (aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
      return Status::OutOfMemory("cannot allocate Get graph report");
    if (aclrtMemset(device, bytes, initial, bytes) != ACL_SUCCESS)
      return Status::Internal("cannot initialize Get graph report");
    if (stage && aclrtMallocHost(&host, bytes) != ACL_SUCCESS)
      return Status::OutOfMemory("cannot allocate Get graph report staging");
    return Status::OK();
  }
  GetGraphReport() = default;
  GetGraphReport(const GetGraphReport&) = delete;
  GetGraphReport& operator=(const GetGraphReport&) = delete;
  ~GetGraphReport() {
    if (device) (void)aclrtFree(device);
    if (host) (void)aclrtFreeHost(host);
  }
};
}  // namespace kvaf
