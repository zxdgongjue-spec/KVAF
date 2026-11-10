#pragma once
// Declarations for CPU-only Host compilation/lifecycle tests. This is not a
// replacement for compiling against the installed CANN headers on the target.
#include <cstddef>
#include <cstdint>
using aclError = int;
using aclrtStream = void*;
using aclrtBinary = void*;
using aclrtBinHandle = void*;
using aclrtFuncHandle = void*;
using aclmdlRI = void*;
enum aclmdlRICaptureStatus { ACL_MODEL_RI_CAPTURE_STATUS_NONE, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE };
enum { ACL_SUCCESS = 0, ACL_MEM_MALLOC_HUGE_FIRST = 0, ACL_MEM_MALLOC_NORMAL_ONLY = 1 };
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST };
int aclrtGetDevice(int32_t*);
int aclrtSetDevice(int32_t);
int aclrtMalloc(void**, size_t, int);
int aclrtFree(void*);
int aclrtMallocHost(void**, size_t);
int aclrtFreeHost(void*);
int aclrtMemset(void*, size_t, int32_t, size_t);
int aclrtMemsetAsync(void*, size_t, int32_t, size_t, aclrtStream);
int aclrtMemcpy(void*, size_t, const void*, size_t, aclrtMemcpyKind);
int aclrtMemcpyAsync(void*, size_t, const void*, size_t, aclrtMemcpyKind, aclrtStream);
int aclrtCreateStream(aclrtStream*);
int aclrtDestroyStream(aclrtStream);
int aclrtSynchronizeStream(aclrtStream);
int aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus*, aclmdlRI*);
enum {
  ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE, ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE,
  ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US, ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE,
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
