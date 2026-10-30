#include "cpu_backends.h"
#include "ubmem/hal.h"

#include <sys/socket.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace kvaf {
using namespace ubmem;
namespace {
template <typename T>
bool Transfer(int fd, T* object, std::size_t offset, bool write) {
  auto* data = reinterpret_cast<std::uint8_t*>(object);
  while (offset < sizeof(T)) {
    const auto n = write ? send(fd, data + offset, sizeof(T) - offset, MSG_NOSIGNAL)
                         : recv(fd, data + offset, sizeof(T) - offset, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    offset += static_cast<std::size_t>(n);
  }
  return true;
}
}  // namespace

struct UbmemCpuBackend::Impl {
  std::shared_ptr<Hal> hal = std::make_shared<Hal>();
  std::unordered_map<std::uint64_t, std::unique_ptr<Mapping>> regions;
  std::uint64_t& next_session;
  std::string library;
  Impl(const CpuRuntimeOptions& options, std::uint64_t& next)
      : next_session(next), library(options.hal_library), numa(options.numa_node) {}
  int numa = -1;

  Status Allocate(int fd, const peer::MessageHeader& header) {
    peer::AllocateUbmemRegionRequest request{};
    request.header = header;
    if (!peer::IsValidHeader(header, peer::Command::kAllocateUbmemRegion, sizeof(request)) ||
        !Transfer(fd, &request, sizeof(header), false)) {
      return Status::InvalidArgument("invalid UBmem allocation record");
    }
    peer::AllocateUbmemRegionResponse response{};
    response.header = peer::MakeHeader(peer::Command::kAllocateUbmemRegion, sizeof(response));
    response.header.command = peer::ResponseCommand(peer::Command::kAllocateUbmemRegion);
    auto region = std::make_unique<Mapping>(hal);
    FabricHandle handle{};
    const bool invalid = request.slot_size == 0 || request.ddr_bytes == 0 ||
        request.ddr_bytes > UINT64_MAX - (kPageBytes - 1) ||
        request.ddr_bytes % request.slot_size != 0;
    const auto mapping_bytes = invalid ? 0 : (request.ddr_bytes + kPageBytes - 1) / kPageBytes * kPageBytes;
    Status status = invalid ? Status::InvalidArgument("DDR bytes must be a multiple of slot_size")
                            : hal->Load(library);
    if (status.ok()) status = region->AllocateHost(mapping_bytes, numa, &handle);
    response.result = static_cast<int>(status.ok() ? peer::Result::kOk : peer::Result::kResourceError);
    if (status.ok()) {
      response.session_id = next_session++;
      response.peer_base = region->address();
      response.peer_length = request.ddr_bytes;
      response.mapping_bytes = region->bytes();
      std::memcpy(response.fabric_handle, handle.data(), handle.size());
      regions.emplace(response.session_id, std::move(region));
      std::fprintf(stderr, "[KVAF_UBMEM] DDR_EXPORTED session=%llu bytes=%llu numa=%d\n",
          static_cast<unsigned long long>(response.session_id),
          static_cast<unsigned long long>(response.peer_length), numa);
    } else {
      std::snprintf(response.error_message, sizeof(response.error_message), "%s", status.message().c_str());
    }
    if (!Transfer(fd, &response, 0, true)) {
      if (response.session_id != 0) regions.erase(response.session_id);
      return Status::Internal("failed to send UBmem allocation response");
    }
    return status;
  }
};

UbmemCpuBackend::UbmemCpuBackend(const CpuRuntimeOptions& options, std::uint64_t& next)
    : impl_(std::make_unique<Impl>(options, next)) {}
UbmemCpuBackend::~UbmemCpuBackend() { (void)Shutdown(); }
Status UbmemCpuBackend::ReleaseLink(LinkId id) {
  auto found = impl_->regions.find(id);
  if (found == impl_->regions.end()) return Status::NotFound("UBmem session was not found");
  Status status = found->second->Close();
  if (status.ok()) impl_->regions.erase(found);
  return status;
}
Status UbmemCpuBackend::Shutdown() {
  for (auto it = impl_->regions.begin(); it != impl_->regions.end();) {
    Status status = it->second->Close();
    if (!status.ok()) return status;
    it = impl_->regions.erase(it);
  }
  return Status::OK();
}

Status UbmemCpuBackend::ServeConnection(int fd, const peer::MessageHeader& header) {
  return impl_->Allocate(fd, header);
}

}  // namespace kvaf
