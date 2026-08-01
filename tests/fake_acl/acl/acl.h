#ifndef KVAF_TEST_FAKE_ACL_H
#define KVAF_TEST_FAKE_ACL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int aclError;
typedef void* aclrtStream;

enum {
  ACL_SUCCESS = 0,
  ACL_MEM_MALLOC_HUGE_FIRST = 0,
  ACL_MEMCPY_HOST_TO_DEVICE = 0,
  ACL_MEMCPY_DEVICE_TO_HOST = 1,
};

aclError aclInit(const char* config_path);
aclError aclFinalize(void);
aclError aclrtSetDevice(int32_t device_id);
aclError aclrtResetDevice(int32_t device_id);
aclError aclrtCreateStream(aclrtStream* stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtSynchronizeStream(aclrtStream stream);
aclError aclrtMalloc(void** address, size_t bytes, int policy);
aclError aclrtFree(void* address);
aclError aclrtMemcpy(void* destination, size_t destination_bytes,
                     const void* source, size_t source_bytes, int kind);

#ifdef __cplusplus
}
#endif

#endif
