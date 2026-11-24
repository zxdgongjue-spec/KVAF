#pragma once

#include <cstddef>

#include "cpu_runtime.h"

namespace kvaf {

std::string CpuMemoryPageModeName(CpuMemoryPageMode mode);
// Explicit HugeTLB backing is rounded to this size. Heap/THP uses the logical
// length (alignment 1); invalid page modes return zero.
std::size_t CpuDdrBackingAlignment(CpuMemoryPageMode mode);

// Keep any allocation owned by the caller if prefaulting fails. The mapping
// length is tracked separately from the logical length used for registration.
Status AllocateCpuDdr(CpuMemoryPageMode mode, std::size_t bytes, void*& memory,
                      std::size_t& mapped_bytes);
Status ReleaseCpuDdr(void*& memory, std::size_t& mapped_bytes);

}  // namespace kvaf
