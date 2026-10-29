#pragma once
// Test declarations only. Production builds use the installed CANN headers.
#include <cstddef>
#include <cstdint>
using aclError = int;
using aclrtStream = void*;
using aclrtEvent = void*;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0;
extern "C" {
aclError aclrtGetDevice(int32_t*);
aclError aclrtMalloc(void**, std::size_t, int);
aclError aclrtFree(void*);
aclError aclrtMemset(void*, std::size_t, int, std::size_t);
aclError aclrtCreateStream(aclrtStream*);
aclError aclrtDestroyStream(aclrtStream);
aclError aclrtCreateEvent(aclrtEvent*);
aclError aclrtRecordEvent(aclrtEvent, aclrtStream);
aclError aclrtSynchronizeEvent(aclrtEvent);
aclError aclrtDestroyEvent(aclrtEvent);
}
