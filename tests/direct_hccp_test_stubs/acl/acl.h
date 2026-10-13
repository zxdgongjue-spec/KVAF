#pragma once

#include <cstdint>

// Only for the Host control-plane test, which never opens an ACL device.
constexpr int ACL_SUCCESS = 0;
extern "C" int aclrtGetDevice(std::int32_t* device);
