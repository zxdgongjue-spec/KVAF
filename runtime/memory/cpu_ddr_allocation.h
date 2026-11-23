#pragma once

#include <cstddef>

#include "cpu_runtime.h"

namespace kvaf {

const char* CpuMemoryPageModeName(CpuMemoryPageMode mode);

// Keep any allocation owned by the caller if prefaulting fails. The mapping
// length is tracked separately from the logical length used for registration.
Status AllocateCpuDdr(CpuMemoryPageMode mode, std::size_t bytes, void*& memory,
                      std::size_t& mapped_bytes);
Status ReleaseCpuDdr(void*& memory, std::size_t& mapped_bytes);

}  // namespace kvaf
