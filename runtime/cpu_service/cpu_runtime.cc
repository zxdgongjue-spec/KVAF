#include "cpu_service/backends/cpu_backends.h"
#include "memory/execution_cpu_backend.h"
#include "remote_pull_service.h"
#include <mutex>
#ifdef KVAF_CPU_HAS_URMA
#include "cpu_service/urma/cpu_urma_discovery.h"
#endif

#include <sys/socket.h>
#include <cerrno>
#include <cstdio>
#include <array>
#include <cstring>

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
Status Reply(int fd, peer::Command command, peer::Result result,
             uint16_t version = peer::kProtocolVersion) {
  // All fixed-size CPU replies start with the same header and result.
  peer::StatusResponse status{};
  status.header = peer::MakeHeader(command, sizeof(Response), version);
  status.header.command = peer::ResponseCommand(command);
  status.result = static_cast<int>(result);
  std::array<char, sizeof(Response)> response{};
  static_assert(sizeof(Response) >= sizeof(status));
  std::memcpy(response.data(), &status, sizeof(status));
  const char* data = response.data();
  std::size_t left = response.size();
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
      : options(std::move(value)),
#ifdef KVAF_CPU_HAS_URMA
        discovery(std::make_shared<CpuUrmaDiscovery>(options.ub_device, options.has_eid_index,
                                                     options.eid_index)),
#endif
        ubmem(options, next_session),
        execution(CreateExecutionService(options, discovery)),
        pull([this](std::uint64_t id, remote_pull::RegionInfo* info) {
          return execution->DescribeCpuPullRegion(id, info);
        }, [this](const remote_pull::RegionInfo& info) {
          return execution->AcquireCpuPullBackend(info);
        }) {}
  CpuRuntimeOptions options;
  std::shared_ptr<CpuUrmaDiscovery> discovery;
  std::uint64_t next_session{1};
  UbmemCpuBackend ubmem;
  std::unique_ptr<execution::CacheRegionService> execution;
  remote_pull::Service pull;
  std::recursive_mutex legacy_mutex;
#ifdef KVAF_CPU_HAS_URMA
  UrmaCpuBackend urma;
#endif
};
CpuRuntime::CpuRuntime() = default;
CpuRuntime::~CpuRuntime() { if (!Shutdown().ok()) (void)impl_.release(); }
Status CpuRuntime::Initialize(const CpuRuntimeOptions& options) {
  std::unique_lock<std::shared_mutex> life(lifecycle_);
  if (impl_) return Status::InvalidState("CpuRuntime already initialized");
  if (options.has_eid_index && options.ub_device.empty())
    return Status::InvalidArgument("CPU eid_index requires ub_device");
  if (options.has_eid_index && options.eid_index == UINT32_MAX)
    return Status::InvalidArgument("CPU eid_index cannot be the wildcard UINT32_MAX");
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
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  return ReleaseLinkUnlocked(id);
}
Status CpuRuntime::ReleaseLinkUnlocked(LinkId id) {
  if (!impl_) return Status::NotInitialized("CpuRuntime not initialized");
  std::lock_guard<std::recursive_mutex> lock(impl_->legacy_mutex);
  auto status = impl_->ubmem.ReleaseLink(id);
#ifdef KVAF_CPU_HAS_URMA
  if (status.code() == StatusCode::kNotFound && impl_->urma.initialized())
    status = impl_->urma.ReleaseLink(id);
#endif
  return status;
}
Status CpuRuntime::ServeConnection(int fd) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  if (!impl_) return Status::NotInitialized("CpuRuntime not initialized");
  if (fd < 0) return Status::InvalidArgument("CpuRuntime connection fd is invalid");
  peer::MessageHeader header{};
  if (!Read(fd, &header, sizeof(header)))
    return Status::InvalidArgument("incomplete CPU control header");
  if (header.magic == remote_pull::kMagic) {
    impl_->pull.Serve(fd, header);
    return Status::OK();
  }
  const auto command = static_cast<peer::Command>(header.command);
  if (header.magic != peer::kProtocolMagic || !peer::IsSupportedProtocolVersion(command, header.version))
    return Status::InvalidArgument("invalid CPU control header");
  if (header.command == execution::kCommand) {
    if (header.bytes < sizeof(header) || header.bytes > 16384)
      return Status::InvalidArgument("invalid execution packet size");
    std::vector<std::uint8_t> bytes(header.bytes - sizeof(header));
    if (!Read(fd, bytes.data(), bytes.size()))
      return Status::Internal("incomplete execution packet");
    execution::Request request;
    execution::Response response;
    response.status = execution::Decode(bytes, &request);
    if (response.status.ok()) response = impl_->execution->Handle(request);
    bytes = execution::Encode(response);
    auto reply = peer::MakeHeader(static_cast<peer::Command>(execution::kCommand),
                                  sizeof(header) + bytes.size());
    reply.command = execution::kCommand | 0x8000;
    bytes.insert(bytes.begin(), reinterpret_cast<const std::uint8_t*>(&reply),
                 reinterpret_cast<const std::uint8_t*>(&reply) + sizeof(reply));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
      auto n = send(fd, bytes.data() + offset, bytes.size() - offset, MSG_NOSIGNAL);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) return Status::Internal("execution reply lost; record retained for query");
      offset += n;
    }
    return Status::OK();
  }
  std::lock_guard<std::recursive_mutex> lock(impl_->legacy_mutex);
  if (command == peer::Command::kAllocateUbmemRegion)
    return impl_->ubmem.ServeConnection(fd, header);
  if (command == peer::Command::kReleaseRegion) {
    peer::ReleaseRegionRequest request{};
    if (!peer::IsValidHeader(header, command, sizeof(request)) ||
        !Read(fd, &request.session_id, sizeof(request.session_id)))
      return Reply<peer::StatusResponse>(fd, command, peer::Result::kInvalidRequest);
    const auto status = ReleaseLinkUnlocked(request.session_id);
    return Reply<peer::StatusResponse>(fd, command, status.ok() ? peer::Result::kOk :
        status.code() == StatusCode::kNotFound ? peer::Result::kNotFound : peer::Result::kResourceError);
  }
#ifdef KVAF_CPU_HAS_URMA
  auto& backend = impl_->urma;
  auto status = backend.initialized() ? Status::OK()
      : backend.Initialize(impl_->options, impl_->next_session, impl_->discovery);
  if (status.ok()) return backend.ServeConnection(fd, header);
  std::fprintf(stderr, "[KVAF_CPU] %s\n", status.message().c_str());
#endif
  // Match the requested response layout even when this backend is unavailable.
  switch (command) {
    case peer::Command::kAllocateRegion:
    case peer::Command::kPrepareGatherRoute:
    case peer::Command::kPrepareGatherWorkspace:
      return Reply<peer::AllocateRegionResponse>(fd, command, peer::Result::kResourceError);
    case peer::Command::kGetCacheCandidateCount:
    case peer::Command::kGetCandidateCount:
      return Reply<peer::CandidateCountResponse>(fd, command, peer::Result::kResourceError);
    case peer::Command::kGetCandidateInfo:
      if (header.bytes == sizeof(peer::CandidateEidRequest))
        return Reply<peer::CandidateEidResponse>(fd, command, peer::Result::kResourceError);
      return Reply<peer::CandidateInfoResponse>(fd, command, peer::Result::kResourceError);
    case peer::Command::kCreateChannel:
      return Reply<peer::CreateChannelResponse>(fd, command, peer::Result::kResourceError);
    case peer::Command::kOpenCpuPushMailbox:
      return Reply<peer::OpenCpuPushMailboxResponse>(fd, command, peer::Result::kResourceError);
    case peer::Command::kOpenCpuPushGet:
    case peer::Command::kOpenAivCpuGatherPushGet:
      if (header.version == peer::kAivCpuPushJettyProtocolVersion)
        return Reply<peer::OpenAivCpuPushJettyResponse>(fd, command, peer::Result::kResourceError, header.version);
      return Reply<peer::OpenAivCpuPushGetResponse>(fd, command, peer::Result::kResourceError, header.version);
    case peer::Command::kPrepareGatherWindow:
      return Reply<peer::PrepareGatherWindowResponse>(fd, command, peer::Result::kResourceError);
    case peer::Command::kEnableDirectGather:
    case peer::Command::kEnableGatherLanes:
    case peer::Command::kEnableAivGatherLanes:
    case peer::Command::kStopGatherLanes:
    case peer::Command::kStartCpuPushGet:
    case peer::Command::kStopCpuPushGet:
      return Reply<peer::StatusResponse>(fd, command, peer::Result::kResourceError, header.version);
    default: return Reply<peer::StatusResponse>(fd, command, peer::Result::kInvalidRequest);
  }
}
Status CpuRuntime::Shutdown() {
  std::unique_lock<std::shared_mutex> life(lifecycle_);
  if (!impl_) return Status::OK();
  // Keep the service reachable for Query/Cancel/Detach after a failed shutdown.
  // Registered sources retain region leases even when they are idle.
  auto status = impl_->pull.Shutdown();
  if (!status.ok()) return status;
  status = impl_->execution->Shutdown();
  if (!status.ok()) return status;
  status = impl_->ubmem.Shutdown();
  if (!status.ok()) return status;
#ifdef KVAF_CPU_HAS_URMA
  status = impl_->urma.Shutdown();
  if (!status.ok()) return status;
#endif
  impl_.reset();
  return Status::OK();
}
bool CpuRuntime::initialized() const {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  return impl_ != nullptr;
}
RuntimeCapabilities CpuRuntime::capabilities() const { return {false, false, true, false}; }
}  // namespace kvaf
