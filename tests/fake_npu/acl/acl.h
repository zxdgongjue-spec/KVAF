#pragma once
// Minimal CPU-only provider double for the HBM service lifecycle tests.
#include <cstddef>
#include <cstdint>
using aclError = int;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 1;
int aclrtSetDevice(std::int32_t device);
int aclrtMalloc(void** data, std::size_t bytes, int policy);
int aclrtFree(void* data);
