#pragma once
// Minimal host contract model. This is not an SDK compatibility header.
#include <cstddef>
#include <cstdint>
using aclError = int;
using aclrtStream = void*;
using aclrtBinHandle = void*;
using aclrtFuncHandle = void*;
constexpr aclError ACL_SUCCESS = 0;
enum aclrtMemMallocPolicy { ACL_MEM_MALLOC_NORMAL_ONLY };
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST };
aclError aclrtGetDevice(int32_t* device);
aclError aclrtMalloc(void** address, size_t bytes, aclrtMemMallocPolicy);
aclError aclrtMallocHost(void** address, size_t bytes);
aclError aclrtFree(void* address);
aclError aclrtFreeHost(void* address);
aclError aclrtMemcpy(void* destination, size_t capacity, const void* source,
                     size_t bytes, aclrtMemcpyKind);
aclError aclrtMemcpyAsync(void* destination, size_t capacity, const void* source,
                          size_t bytes, aclrtMemcpyKind, aclrtStream);
aclError aclrtMemsetAsync(void* destination, size_t capacity, int value,
                          size_t bytes, aclrtStream);
aclError aclrtSynchronizeStream(aclrtStream);
aclError aclrtSynchronizeDevice();
