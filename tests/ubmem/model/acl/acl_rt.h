#pragma once

// Only the ACL VMM surface used by DeviceMapping. Host tests model resource
// ownership and aliasing, not Runtime routing or device memory accessibility.
#include <cstddef>
#include <cstdint>
using aclError = int;
using aclrtDrvMemHandle = void*;
constexpr aclError ACL_SUCCESS = 0;
enum aclrtMemSharedHandleType { ACL_MEM_SHARE_HANDLE_TYPE_FABRIC = 0x2 };
extern "C" {
const char* aclGetRecentErrMsg();
aclError aclrtReserveMemAddress(void**, std::size_t, std::size_t, void*, std::uint64_t);
aclError aclrtReleaseMemAddress(void*);
aclError aclrtMemImportFromShareableHandleV2(void*, aclrtMemSharedHandleType, std::uint64_t, aclrtDrvMemHandle*);
aclError aclrtMapMem(void*, std::size_t, std::size_t, aclrtDrvMemHandle, std::uint64_t);
aclError aclrtUnmapMem(void*);
aclError aclrtFreePhysical(aclrtDrvMemHandle);
}
