#pragma once
#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "status.h"
namespace kvaf::aicpu_urma {
bool LoadUmdkKernel(const char* json);
enum class UmdkKernelKind { Control, Put, Gather, Scatter, CpuSglWrite, CpuSglPush };
// Load the shared binary once, resolving only this algorithm's required entries.
Status PrepareUmdkKernel(const std::string& json, UmdkKernelKind kind);
int32_t LaunchUmdkKernel(const char* name, const void* args, std::size_t bytes, aclrtStream stream,
                         std::uint32_t blocks = 1U);
}  // namespace kvaf::aicpu_urma
