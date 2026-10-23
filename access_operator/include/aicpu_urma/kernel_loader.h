#pragma once
#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
namespace kvaf::aicpu_urma {
bool LoadUmdkKernel(const char* json);
int32_t LaunchUmdkKernel(const char* name, const void* args, std::size_t bytes, aclrtStream stream,
                         std::uint32_t blocks = 1U);
}  // namespace kvaf::aicpu_urma
