// SPDX-License-Identifier: Apache-2.0
#pragma once

// Private to the allocator host test: no production target includes this file.
#include <cstddef>

using aclError = int;
constexpr aclError ACL_SUCCESS = 0;

extern "C" {
aclError aclrtMallocHost(void** address, std::size_t bytes);
aclError aclrtFreeHost(void* address);
}
