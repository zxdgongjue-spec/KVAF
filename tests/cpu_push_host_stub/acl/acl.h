#pragma once

#include "../../aicpu_loader_stub/acl/acl.h"

enum aclrtMemMallocPolicy { ACL_MEM_MALLOC_NORMAL_ONLY };
enum aclrtMemcpyKind { ACL_MEMCPY_DEVICE_TO_HOST };
aclError aclrtMalloc(void**, std::size_t, aclrtMemMallocPolicy);
aclError aclrtMallocHost(void**, std::size_t);
aclError aclrtFree(void*);
aclError aclrtFreeHost(void*);
aclError aclrtMemset(void*, std::size_t, int, std::size_t);
aclError aclrtMemsetAsync(void*, std::size_t, int, std::size_t, aclrtStream);
aclError aclrtMemcpyAsync(void*, std::size_t, const void*, std::size_t, aclrtMemcpyKind, aclrtStream);
aclError aclrtSynchronizeStream(aclrtStream);
