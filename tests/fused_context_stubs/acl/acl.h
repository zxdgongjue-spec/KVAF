#pragma once

// Only the ACL surface used by DefaultAccessOperator. This header is private
// to the portable host lifecycle test; no production target includes it.
#include <cstddef>
#include <cstdint>

using aclError = int;
using aclrtStream = void*;
constexpr aclError ACL_SUCCESS = 0;
enum aclrtMemMallocPolicy { ACL_MEM_MALLOC_HUGE_FIRST, ACL_MEM_MALLOC_NORMAL_ONLY };
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST };

extern "C" {
aclError aclrtMalloc(void**, std::size_t, aclrtMemMallocPolicy);
aclError aclrtFree(void*);
aclError aclrtMemset(void*, std::size_t, int32_t, std::size_t);
aclError aclrtMemcpy(void*, std::size_t, const void*, std::size_t, aclrtMemcpyKind);
aclError aclrtSynchronizeDevice();
aclError aclrtSynchronizeStream(aclrtStream);
aclError aclrtSetDevice(int32_t);
aclError aclrtCreateStream(aclrtStream*);
aclError aclrtDestroyStream(aclrtStream);
}
