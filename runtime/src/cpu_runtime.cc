#include "cpu_backends.h"

#include <sys/socket.h>
#include <cerrno>
#include <cstdio>

namespace kvaf {
namespace {
bool Read(int fd, void* data, std::size_t bytes) {
  auto* out = static_cast<char*>(data);
  while (bytes) {
    const auto n = recv(fd, out, bytes, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    out += n; bytes -= n;
  }
  return true;
}
template <class Response>
Status Reply(int fd, peer::Command command, peer::Result result) {
  Response response{};
  response.header = peer::MakeHeader(command, sizeof(response));
  response.header.command = peer::ResponseCommand(command);
  response.result = static_cast<int>(result);
  const auto* data = reinterpret_cast<const char*>(&response);
  std::size_t left = sizeof(response);
  while (left) {
    const auto n = send(fd, data, left, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return Status::Internal("cannot send CPU response");
    data += n; left -= n;
  }
  return Status::OK();
}
}  // namespace

struct CpuRuntime::Impl {
  explicit Impl(CpuRuntimeOptions value)
      : options(std::move(value)), ubmem(options, next_session) {}
  CpuRuntimeOptions options;
  std::uint64_t next_session{1};
  UbmemCpuBackend ubmem;
#ifdef KVAF_CPU_HAS_URMA
  UrmaCpuBackend urma;
#endif
};
CpuRuntime::CpuRuntime() = default;
CpuRuntime::~CpuRuntime() { (void)Shutdown(); }
Status CpuRuntime::Initialize(const CpuRuntimeOptions& options) {
  if (impl_) return Status::InvalidState("CpuRuntime already initialized");
  if ((options.has_eid_index && options.ub_device.empty()) || options.numa_node < -1)
    return Status::InvalidArgument("invalid CPU runtime options");
  impl_ = std::make_unique<Impl>(options);
  return Status::OK();
}
Status CpuRuntime::LinkTo(const LinkTarget&) {
  return Status::NotImplemented("CpuRuntime accepts incoming links");
}
Status CpuRuntime::AcceptLink(const LinkTarget&) {
  return initialized() ? Status::OK() : Status::NotInitialized("CpuRuntime not initialized");
}
Status CpuRuntime::ReleaseLink(LinkId id) {
  if (!impl_) return Status::NotInitialized("CpuRuntime not initialized");
  auto status = impl_->ubmem.ReleaseLink(id);
#ifdef KVAF_CPU_HAS_URMA
  if (status.code() == StatusCode::kNotFound && impl_->urma.initialized())
    status = impl_->urma.ReleaseLink(id);
#endif
  return status;
}
Status CpuRuntime::ServeConnection(int fd) {
  if (!impl_) return Status::NotInitialized("CpuRuntime not initialized");
  peer::MessageHeader header{};
  if (!Read(fd, &header, sizeof(header)) || header.magic != peer::kProtocolMagic ||
      header.version != peer::kProtocolVersion)
    return Status::InvalidArgument("invalid CPU control header");
  const auto command = static_cast<peer::Command>(header.command);
  if (command == peer::Command::kAllocateUbmemRegion ||
      command == peer::Command::kCheckUbmemRange || command == peer::Command::kFillUbmemRange)
    return impl_->ubmem.ServeConnection(fd, header);
  if (command == peer::Command::kReleaseRegion) {
    peer::ReleaseRegionRequest request{};
    if (!peer::IsValidHeader(header, command, sizeof(request)) ||
        !Read(fd, &request.session_id, sizeof(request.session_id)))
      return Reply<peer::StatusResponse>(fd, command, peer::Result::kInvalidRequest);
    const auto status = ReleaseLink(request.session_id);
    return Reply<peer::StatusResponse>(fd, command, status.ok() ? peer::Result::kOk :
        status.code() == StatusCode::kNotFound ? peer::Result::kNotFound : peer::Result::kResourceError);
  }
#ifdef KVAF_CPU_HAS_URMA
  auto status = impl_->urma.initialized() ? Status::OK()
      : impl_->urma.Initialize(impl_->options, impl_->next_session);
  if (status.ok()) return impl_->urma.ServeConnection(fd, header);
  std::fprintf(stderr, "[KVAF_CPU] %s\n", status.message().c_str());
#endif
  // Match the requested response layout even when this backend is unavailable.
  switch (command) {
    case peer::Command::kAllocateRegion:
      return Reply<peer::AllocateRegionResponse>(fd, command, peer::Result::kResourceError);
    case peer::Command::kGetCandidateCount:
      return Reply<peer::CandidateCountResponse>(fd, command, peer::Result::kResourceError);
    case peer::Command::kGetCandidateInfo:
      return Reply<peer::CandidateInfoResponse>(fd, command, peer::Result::kResourceError);
    case peer::Command::kCreateChannel:
      return Reply<peer::CreateChannelResponse>(fd, command, peer::Result::kResourceError);
    default: return Reply<peer::StatusResponse>(fd, command, peer::Result::kInvalidRequest);
  }
}
Status CpuRuntime::Shutdown() {
  if (!impl_) return Status::OK();
  auto status = impl_->ubmem.Shutdown();
  if (!status.ok()) return status;
  impl_.reset();
  return Status::OK();
}
bool CpuRuntime::initialized() const { return impl_ != nullptr; }
RuntimeCapabilities CpuRuntime::capabilities() const { return {false, false, true, false}; }
}  // namespace kvaf
