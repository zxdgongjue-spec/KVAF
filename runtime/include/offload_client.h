#pragma once
#include <string>
#include "offload_protocol.h"
#include "status.h"

namespace kvaf::offload {
// Bounded TCP exchange; a failed exchange may have executed on the peer.
Status Exchange(const std::string& address, Request request,
                const std::vector<Span>& spans, Response* response);
bool ReadBytes(int fd, void* buffer, std::size_t bytes);
bool WriteBytes(int fd, const void* buffer, std::size_t bytes);
void SetSocketTimeout(int fd);
}
