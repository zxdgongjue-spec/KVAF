#pragma once

#include <cstddef>

using aclError = int;
using aclrtStream = void*;
constexpr aclError ACL_SUCCESS = 0;
enum aclrtMemMallocPolicy { ACL_MEM_MALLOC_HUGE_FIRST };
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST };

extern "C" {
aclError aclrtMalloc(void** output, std::size_t bytes, aclrtMemMallocPolicy policy);
aclError aclrtFree(void* pointer);
aclError aclrtMallocHost(void** output, std::size_t bytes);
aclError aclrtFreeHost(void* pointer);
aclError aclrtMemset(void* output, std::size_t capacity, int value, std::size_t bytes);
aclError aclrtMemcpy(void* output, std::size_t capacity, const void* input, std::size_t bytes,
                     aclrtMemcpyKind kind);
aclError aclrtMemcpyAsync(void* output, std::size_t capacity, const void* input, std::size_t bytes,
                          aclrtMemcpyKind kind, aclrtStream stream);
aclError aclrtCreateStream(aclrtStream* stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtSynchronizeStream(aclrtStream stream);
}

using aclrtBinHandle = void*;
using aclrtFuncHandle = void*;
constexpr int ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT = 1;
constexpr int ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE = 1;
struct aclrtLaunchKernelAttr {
  int id;
  struct {
    unsigned timeout;
  } value;
};
struct aclrtLaunchKernelCfg {
  unsigned numAttrs;
  aclrtLaunchKernelAttr* attrs;
};
struct aclrtBinaryLoadOption {
  int type;
  struct {
    int cpuKernelMode;
  } value;
};
struct aclrtBinaryLoadOptions {
  unsigned numOpt;
  aclrtBinaryLoadOption* options;
};
extern "C" {
aclError aclrtBinaryLoadFromFile(const char*, const aclrtBinaryLoadOptions*, aclrtBinHandle*);
aclError aclrtBinaryGetFunction(aclrtBinHandle, const char*, aclrtFuncHandle*);
aclError aclrtBinaryUnLoad(aclrtBinHandle);
aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, unsigned, aclrtStream,
                                       const aclrtLaunchKernelCfg*, void*, std::size_t, void*,
                                       std::size_t);
}
