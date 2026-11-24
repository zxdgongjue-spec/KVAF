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
Status WriteReply(int fd, const Response& response) {
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
template <class Response>
Status Reply(int fd, peer::Command command, peer::Result result,
             uint16_t version = peer::kProtocolVersion) {
  Response response{};
  response.header = peer::MakeHeader(command, sizeof(response), version);
  response.header.command = peer::ResponseCommand(command);
  response.result = static_cast<int>(result);
  return WriteReply(fd, response);
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
CpuRuntime::~CpuRuntime() { if (!Shutdown().ok()) (void)impl_.release(); }
Status CpuRuntime::Initialize(const CpuRuntimeOptions& options) {
  if (impl_) return Status::InvalidState("CpuRuntime already initialized");
  if (options.has_eid_index && options.ub_device.empty())
    return Status::InvalidArgument("invalid CPU runtime options");
  if (options.gather_threads > 64U)
    return Status::InvalidArgument("CpuRuntime gather_threads must be in [1,64] or auto");
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
  if (fd < 0) return Status::InvalidArgument("CpuRuntime connection fd is invalid");
  peer::MessageHeader header{};
  if (!Read(fd, &header, sizeof(header)) || header.magic != peer::kProtocolMagic)
    return Status::InvalidArgument("invalid CPU control header");
  const auto command = static_cast<peer::Command>(header.command);
  if (!peer::IsSupportedProtocolVersion(command, header.version))
    return Status::InvalidArgument("invalid CPU control protocol version");
  if (command == peer::Command::kAllocateUbmemRegion)
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
    case peer::Command::kOpenCpuPushGet:
    case peer::Command::kOpenAivCpuGatherPushGet:
      if (header.version == peer::kAivCpuPushJettyProtocolVersion) {
        peer::OpenAivCpuPushJettyResponse response{};
        response.base.header = peer::MakeHeader(command, sizeof(response), header.version);
        response.base.header.command = peer::ResponseCommand(command);
        response.base.result = static_cast<int32_t>(peer::Result::kResourceError);
        return WriteReply(fd, response);
      }
      if (header.version == peer::kAivCpuPushProtocolVersion)
        return Reply<peer::OpenAivCpuPushGetResponse>(fd, command, peer::Result::kResourceError,
                                                      header.version);
      return Reply<peer::OpenCpuPushGetResponse>(fd, command, peer::Result::kResourceError);
    case peer::Command::kEnableDirectGather:
    case peer::Command::kEnableGatherLanes:
    case peer::Command::kStopGatherLanes:
    case peer::Command::kStartCpuPushGet:
    case peer::Command::kStopCpuPushGet:
      return Reply<peer::StatusResponse>(fd, command, peer::Result::kResourceError, header.version);
    default: return Reply<peer::StatusResponse>(fd, command, peer::Result::kInvalidRequest);
  }
}
Status CpuRuntime::Shutdown() {
  if (!impl_) return Status::OK();
  auto status = impl_->ubmem.Shutdown();
  if (!status.ok()) return status;
#ifdef KVAF_CPU_HAS_URMA
  status = impl_->urma.Shutdown();
  if (!status.ok()) return status;
#endif
  impl_.reset();
  return Status::OK();
}
bool CpuRuntime::initialized() const { return impl_ != nullptr; }
RuntimeCapabilities CpuRuntime::capabilities() const { return {false, false, true, false}; }
}  // namespace kvaf
