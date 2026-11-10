#pragma once
// Minimal ACL surface for testing the real loader's preflight logic on a host.
#include <cstddef>
#include <cstdint>
using aclError = int;
using aclrtBinHandle = void*;
using aclrtFuncHandle = void*;
using aclrtStream = void*;
constexpr aclError ACL_SUCCESS = 0;
constexpr int ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT = 0;
constexpr int ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE = 0;
struct aclrtLaunchKernelAttr { int id; struct { std::uint32_t timeout; } value; };
struct aclrtLaunchKernelCfg { aclrtLaunchKernelAttr* attrs; std::size_t numAttrs; };
struct aclrtBinaryLoadOption { int type; struct { int cpuKernelMode; } value; };
struct aclrtBinaryLoadOptions { aclrtBinaryLoadOption* options; std::size_t numOpt; };
aclError aclrtBinaryLoadFromFile(const char*, const aclrtBinaryLoadOptions*, aclrtBinHandle*);
aclError aclrtBinaryUnLoad(aclrtBinHandle);
aclError aclrtBinaryGetFunction(aclrtBinHandle, const char*, aclrtFuncHandle*);
aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, std::uint32_t, aclrtStream,
    const aclrtLaunchKernelCfg*, const void*, std::size_t, void*, std::size_t);
