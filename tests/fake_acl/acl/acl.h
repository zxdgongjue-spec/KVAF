#pragma once

// Test-only ACL surface. Never added to production target include paths.
#include <cstddef>
#include <cstdint>

using aclError = int;
using aclrtStream = void*;
using aclrtEvent = void*;
constexpr aclError ACL_SUCCESS = 0;
constexpr std::uint32_t ACL_EVENT_SYNC = 1;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 2;
constexpr int ACL_MEMCPY_DEVICE_TO_DEVICE = 3;

aclError aclrtCreateStream(aclrtStream* stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtCreateEventExWithFlag(aclrtEvent* event, std::uint32_t flag);
aclError aclrtDestroyEvent(aclrtEvent event);
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);
aclError aclrtSynchronizeStream(aclrtStream stream);
aclError aclrtMalloc(void** pointer, std::size_t bytes, int policy);
aclError aclrtFree(void* pointer);
aclError aclrtMemset(void* pointer, std::size_t capacity, int value, std::size_t bytes);
aclError aclrtMemcpy(void* destination, std::size_t capacity, const void* source,
                    std::size_t bytes, int kind);
