#pragma once

// Private ABI surface for the portable allocator lifecycle test. Production
// targets must include the CANN SDK headers instead of this directory.
#include <cstddef>
#include <cstdint>

using aclError = int;
using aclrtDrvMemHandle = void*;
constexpr aclError ACL_SUCCESS = 0;
enum aclrtMemHandleType { ACL_MEM_HANDLE_TYPE_NONE };
enum aclrtMemAllocationType { ACL_MEM_ALLOCATION_TYPE_PINNED };
enum aclrtMemAttr { ACL_DDR_MEM_P2P_HUGE, ACL_DDR_MEM_P2P_NORMAL };
enum aclrtMemLocationType {
  ACL_MEM_LOCATION_TYPE_HOST_NUMA,
  ACL_MEM_LOCATION_TYPE_DEVICE
};
enum aclrtMemGranularityOptions { ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM };
enum aclrtMemAccessFlags { ACL_RT_MEM_ACCESS_FLAGS_READWRITE };
enum aclrtHacType { ACL_RT_HAC_TYPE_AIV };
enum aclrtHostMemMapCapability {
  ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED, ACL_RT_HOST_MEM_MAP_SUPPORTED
};
constexpr std::uint32_t ACL_HOST_REG_MAPPED = 0x2U;
struct aclrtMemLocation {
  aclrtMemLocationType type;
  std::uint32_t id;
};
struct aclrtPhysicalMemProp {
  aclrtMemHandleType handleType;
  aclrtMemAllocationType allocationType;
  aclrtMemAttr memAttr;
  aclrtMemLocation location;
};
struct aclrtMemAccessDesc {
  aclrtMemLocation location;
  aclrtMemAccessFlags flags;
};

extern "C" {
aclError aclrtGetDevice(std::int32_t*);
aclError aclrtSetDevice(std::int32_t);
aclError aclsysGetVersionNum(char*, std::int32_t*);
aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp*,
    aclrtMemGranularityOptions, std::size_t*);
aclError aclrtMallocPhysical(aclrtDrvMemHandle*, std::size_t,
    const aclrtPhysicalMemProp*, std::uint64_t);
aclError aclrtReserveMemAddress(void**, std::size_t, std::size_t, void*, std::uint64_t);
aclError aclrtMapMem(void*, std::size_t, std::size_t, aclrtDrvMemHandle, std::uint64_t);
aclError aclrtMemSetAccess(void*, std::size_t, aclrtMemAccessDesc*, std::size_t);
aclError aclrtSynchronizeDevice();
aclError aclrtUnmapMem(void*);
aclError aclrtFreePhysical(aclrtDrvMemHandle);
aclError aclrtReleaseMemAddress(void*);
aclError aclrtHostMemMapCapabilities(std::uint32_t, aclrtHacType, aclrtHostMemMapCapability*);
aclError aclrtHostRegisterV2(void*, std::uint64_t, std::uint32_t);
aclError aclrtHostGetDevicePointer(void*, void**, std::uint32_t);
aclError aclrtHostUnregister(void*);
}
