#pragma once

#include <cstddef>

// Host-test declarations only. Never on a production target's include path.
using aclError = int;
using aclrtStream = void*;
constexpr aclError ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 2;
aclError aclrtMalloc(void**, std::size_t, int);
aclError aclrtFree(void*);
aclError aclrtMemcpy(void*, std::size_t, const void*, std::size_t, int);
aclError aclrtMemset(void*, std::size_t, int, std::size_t);
aclError aclrtCreateStream(aclrtStream*);
aclError aclrtDestroyStream(aclrtStream);
aclError aclrtSetDevice(int);
