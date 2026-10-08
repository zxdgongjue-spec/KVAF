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

// Host launcher ABI doubles. Hardware service tests do not use these APIs.
using aclrtStream = void*;
using aclrtBinary = void*;
using aclrtBinHandle = void*;
using aclrtFuncHandle = void*;
enum {
  ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE,
  ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE,
  ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US,
  ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE,
  ACL_RT_ENGINE_TYPE_AIV,
};
struct aclrtLaunchKernelAttr {
  int id;
  struct {
    int schemMode;
    uint32_t dynUBufSize;
    struct { uint32_t timeoutLow, timeoutHigh; } timeoutUs;
    int engineType;
  } value;
};
struct aclrtLaunchKernelCfg { uint32_t numAttrs; aclrtLaunchKernelAttr* attrs; };
aclrtBinary aclrtCreateBinary(const void*, size_t);
int aclrtBinaryLoad(aclrtBinary, aclrtBinHandle*);
int aclrtDestroyBinary(aclrtBinary);
int aclrtBinaryGetFunction(aclrtBinHandle, const char*, aclrtFuncHandle*);
int aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, uint32_t, aclrtStream,
                                 const aclrtLaunchKernelCfg*, void*, size_t, void*, size_t);
