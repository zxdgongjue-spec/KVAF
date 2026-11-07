#pragma once

#include <acl/acl.h>
#include <string>
#include <vector>

#include "status.h"

namespace kvaf::ubmem {

// An instance is confined to the creating NPU context. The operator drains
// outstanding work before Unload; a poisoned operator retains this instance.
class Kernel {
 public:
  Status Load(const std::string& path, const char* name);
  Status Unload();
  aclError Launch(const void* arguments, size_t bytes, uint32_t workers,
                  aclrtStream stream) const;

 private:
  uint32_t local_memory_bytes_{0};
  aclrtBinHandle binary_{nullptr};
  aclrtFuncHandle function_{nullptr};
  std::vector<char> bytes_;
};

}  // namespace kvaf::ubmem
