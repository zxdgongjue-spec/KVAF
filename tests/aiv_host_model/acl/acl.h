#pragma once
// Only for the Host operator unit test. Does not model CANN execution/timing.
#include <cstddef>
using aclError = int;
using aclrtStream = void*;
using aclmdlRI = void*;
enum aclmdlRICaptureStatus { ACL_MODEL_RI_CAPTURE_STATUS_NONE, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE };
enum { ACL_SUCCESS = 0, ACL_MEM_MALLOC_HUGE_FIRST = 0,
       ACL_MEMCPY_HOST_TO_DEVICE = 1, ACL_MEMCPY_DEVICE_TO_HOST = 2 };
aclError aclrtMalloc(void**, std::size_t, int);
aclError aclrtFree(void*);
aclError aclrtMemset(void*, std::size_t, int, std::size_t);
aclError aclrtMemsetAsync(void*, std::size_t, int, std::size_t, aclrtStream);
aclError aclrtMemcpy(void*, std::size_t, const void*, std::size_t, int);
aclError aclrtSynchronizeStream(aclrtStream);
aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus*, aclmdlRI*);
