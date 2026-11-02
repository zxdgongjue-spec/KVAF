#pragma once
// Test declarations only. Production builds use the installed CANN headers.
#include <cstddef>
#include <cstdint>
using aclError = int;
using aclrtStream = void*;
using aclrtEvent = void*;
constexpr int ACL_SUCCESS = 0;
constexpr std::uint32_t ACL_EVENT_TIME_LINE = 8;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 2;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
extern "C" {
aclError aclrtSetDevice(int32_t);
aclError aclrtMemcpy(void*, std::size_t, const void*, std::size_t, int);
aclError aclrtGetDevice(int32_t*);
aclError aclrtMalloc(void**, std::size_t, int);
aclError aclrtFree(void*);
aclError aclrtMemset(void*, std::size_t, int, std::size_t);
aclError aclrtCreateStream(aclrtStream*);
aclError aclrtDestroyStream(aclrtStream);
aclError aclrtSynchronizeStream(aclrtStream);
aclError aclrtCreateEvent(aclrtEvent*);
aclError aclrtCreateEventWithFlag(aclrtEvent*, std::uint32_t);
aclError aclrtEventElapsedTime(float*, aclrtEvent, aclrtEvent);
aclError aclrtRecordEvent(aclrtEvent, aclrtStream);
aclError aclrtSynchronizeEvent(aclrtEvent);
aclError aclrtDestroyEvent(aclrtEvent);
}
