#pragma once
#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "status.h"
namespace kvaf::aicpu_urma {
bool LoadUmdkKernel(const char* json);
// Load and check all selected entry points without an EID or remote allocation.
Status PrepareUmdkKernel(const std::string& json, bool scatter);
int32_t LaunchUmdkKernel(const char* name, const void* args, std::size_t bytes, aclrtStream stream,
                         std::uint32_t blocks = 1U);
}  // namespace kvaf::aicpu_urma
