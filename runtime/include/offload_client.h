#pragma once
#include <string>
#include "offload_protocol.h"
#include "status.h"

namespace kvaf::offload {
bool ReadBytes(int fd, void* buffer, std::size_t bytes);
bool WriteBytes(int fd, const void* buffer, std::size_t bytes);
void SetSocketTimeout(int fd);
}
