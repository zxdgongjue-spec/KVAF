#include "memory/execution_cpu_backend.h"
#include "cpu_service/backends/cpu_backends.h"
#include "memory/cpu_ddr_allocation.h"
#include "ubmem/hal.h"
#include "offload_service.h"
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>
namespace kvaf {
namespace {
class PoolAllocation final : public execution::Allocation {
 public:
  ~PoolAllocation() override {
    if (!Close().ok()) (void)mapping.release();
  }
  Status Close() override {
#ifdef KVAF_CPU_HAS_URMA
    if (urma_memory) {
      auto s = urma_memory->Close();
      if (!s.ok()) return s;
      urma_memory.reset();
    }
#endif
    if (mapping) {
      auto s = mapping->Close();
      if (!s.ok()) return s;
      mapping.reset();
    }
    return ReleaseCpuDdr(memory, mapped_bytes);
  }
#ifdef KVAF_CPU_HAS_URMA
  Status PrepareCpuPull(std::uint32_t candidate, std::uint32_t token,
                        std::uint64_t bytes, peer::PeerRegion* out) override {
    return urma_memory ? urma_memory->PrepareCpuPull(candidate, token, bytes, out)
                       : Status::NotImplemented("allocation has no CPU URMA owner");
  }
  std::unique_ptr<offload::Backend> CreateCpuPullBackend(std::shared_ptr<void> lease) override {
    return urma_memory ? urma_memory->CreateCpuPullBackend(std::move(lease)) : nullptr;
  }
#endif
  void* memory = nullptr;
  std::size_t mapped_bytes = 0U;
  std::unique_ptr<ubmem::Mapping> mapping;
#ifdef KVAF_CPU_HAS_URMA
  std::shared_ptr<CpuUrmaMemoryPool> urma_memory;
#endif
};
class FabricSession final : public execution::Session {
 public:
  Status Open(const execution::Request&, execution::Allocation& pool,
              peer::PeerRegion* out) override {
    if (!pool.descriptor.fabric_mapping_bytes)
      return Status::InvalidArgument("region is not Fabric-exported");
    *out = pool.descriptor;
    return Status::OK();
  }
  Status Call(const std::vector<std::uint8_t>&, std::vector<std::uint8_t>*) override {
    return Status::InvalidArgument("Fabric sessions have no URMA commands");
  }
  Status Close() override { return Status::OK(); }
};
#ifdef KVAF_CPU_HAS_URMA
class UrmaSession final : public execution::Session {
 public:
  UrmaSession(CpuRuntimeOptions options, std::shared_ptr<CpuUrmaDiscovery> discovery)
      : options_(std::move(options)), discovery_(std::move(discovery)) {}
  Status Open(const execution::Request& q, execution::Allocation& pool,
              peer::PeerRegion* out) override {
    if (!q.payload.empty() && q.payload.size() != peer::kEidBytes)
      return Status::InvalidArgument("invalid cache session EID");
    profile_ = q;
    if (q.cpu_threads) options_.gather_threads = q.cpu_threads;
    auto* allocation = dynamic_cast<PoolAllocation*>(&pool);
    if (!allocation || !allocation->urma_memory)
      return Status::InvalidState("allocation has no shared CPU URMA memory owner");
    auto s = backend_.Initialize(options_, next_, discovery_, allocation->urma_memory);
    if (s.ok())
      s = backend_.AttachRegion(reinterpret_cast<void*>(pool.descriptor.base),
                                pool.descriptor.length, 1, q.candidate, out);
    if (s.ok() && !q.payload.empty() && std::memcmp(q.payload.data(), out->eid.data(), out->eid.size()) != 0)
      return Status::InvalidState("CPU cache registration differs from the selected EID");
    return s;
  }
  Status Call(const std::vector<std::uint8_t>& in, std::vector<std::uint8_t>* out) override {
    if (in.size() < sizeof(peer::MessageHeader))
      return Status::InvalidArgument("truncated session command");
    peer::MessageHeader header{};
    std::memcpy(&header, in.data(), sizeof(header));
    const auto command = static_cast<peer::Command>(header.command);
    if ((command == peer::Command::kEnableDirectGather ||
         command == peer::Command::kEnableGatherLanes ||
         command == peer::Command::kEnableAivGatherLanes ||
         command == peer::Command::kStopGatherLanes ||
         command == peer::Command::kPrepareGatherWorkspace ||
         command == peer::Command::kPrepareGatherWindow) &&
        (profile_.put || profile_.algorithm != GetAlgorithm::Gather))
      return Status::InvalidArgument("session has no Gather profile");
    if ((command == peer::Command::kOpenCpuPushGet || command == peer::Command::kOpenAivCpuGatherPushGet ||
         command == peer::Command::kStartCpuPushGet ||
         command == peer::Command::kStopCpuPushGet) &&
        (profile_.put || profile_.algorithm != GetAlgorithm::Scatter))
      return Status::InvalidArgument("session has no Scatter profile");
    return backend_.Call(in, out);
  }
  Status Close() override { return backend_.Shutdown(); }

 private:
  CpuRuntimeOptions options_;
  std::shared_ptr<CpuUrmaDiscovery> discovery_;
  execution::Request profile_;
  std::uint64_t next_ = 1;
  UrmaCpuBackend backend_;
};
#endif
}  // namespace
std::unique_ptr<execution::CacheRegionService> CreateExecutionService(
    const CpuRuntimeOptions& options, std::shared_ptr<CpuUrmaDiscovery> discovery) {
  auto allocate = [options, discovery](const execution::Request& q,
                            std::unique_ptr<execution::Allocation>* out) -> Status {
    auto pool = std::make_unique<PoolAllocation>();
    auto* p = pool.get();
    *out = std::move(pool);
    const auto bytes = q.logical_bytes + q.auxiliary_bytes;
    if (bytes > std::numeric_limits<std::size_t>::max())
      return Status::InvalidArgument("region too large");
    if (q.mode != TransportMode::Urma) {
      if (bytes > UINT64_MAX - (ubmem::kPageBytes - 1))
        return Status::InvalidArgument("mapping size overflow");
      std::shared_ptr<ubmem::Hal> hal;
      auto s = ubmem::AcquireProcessHal(options.hal_library, &hal);
      if (!s.ok()) return s;
      p->mapping = std::make_unique<ubmem::Mapping>(hal);
      s = p->mapping->AllocateHost(
          (bytes + ubmem::kPageBytes - 1) / ubmem::kPageBytes * ubmem::kPageBytes,
          &p->descriptor.fabric_handle);
      if (!s.ok()) return s;
      p->descriptor.base = p->mapping->address();
      p->descriptor.fabric_mapping_bytes = p->mapping->bytes();
    } else {
      auto s = AllocateCpuDdr(options.memory_page_mode, bytes, p->memory, p->mapped_bytes);
      if (!s.ok()) return s;
      p->descriptor.base = reinterpret_cast<std::uintptr_t>(p->memory);
      std::fprintf(stderr, "[KVAF_CPU] DDR_ALLOCATED bytes=%" PRIu64 " backing_bytes=%" PRIu64
                   " page_mode=%s memory=%p\n", bytes,
                   p->mapped_bytes != 0U ? static_cast<std::uint64_t>(p->mapped_bytes) : bytes,
                   CpuMemoryPageModeName(options.memory_page_mode).c_str(), p->memory);
    }
    p->descriptor.length = bytes;
#ifdef KVAF_CPU_HAS_URMA
    if (q.mode != TransportMode::Ubmem)
      p->urma_memory = std::make_shared<CpuUrmaMemoryPool>(options, discovery,
          reinterpret_cast<void*>(p->descriptor.base), bytes, q.epoch, q.request);
#endif
    // Initialize auxiliary controls once, never on each registration of this pool.
    if (q.auxiliary_bytes)
      std::memset(reinterpret_cast<void*>(p->descriptor.base + q.logical_bytes), 0,
                  q.auxiliary_bytes);
    return Status::OK();
  };
  auto create = [options, discovery](const execution::Request& q) -> std::unique_ptr<execution::Session> {
    if (q.transport == Transport::Ubmem) return std::make_unique<FabricSession>();
#ifdef KVAF_CPU_HAS_URMA
    return std::make_unique<UrmaSession>(options, discovery);
#else
    return {};
#endif
  };
  return std::make_unique<execution::CacheRegionService>(execution::kFabric
#ifdef KVAF_CPU_HAS_URMA
                                                             | execution::kUrma | execution::kCpuPull
#endif
                                                         ,
                                                         std::move(allocate), std::move(create));
}
}  // namespace kvaf
