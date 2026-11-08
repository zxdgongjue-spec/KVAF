#include "execution_cpu_backend.h"
#include "cpu_backends.h"
#include "ubmem/hal.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#if defined(__linux__)
#include <sys/mman.h>
#endif
namespace kvaf {
namespace {
class PoolAllocation final : public execution::Allocation {
 public:
  ~PoolAllocation() override {
    if (!Close().ok()) (void)mapping.release();
  }
  Status Close() override {
    if (mapping) {
      auto s = mapping->Close();
      if (!s.ok()) return s;
      mapping.reset();
    }
    if (memory) {
      std::free(memory);
      memory = nullptr;
    }
    return Status::OK();
  }
  void* memory = nullptr;
  std::unique_ptr<ubmem::Mapping> mapping;
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
  explicit UrmaSession(CpuRuntimeOptions options) : options_(std::move(options)) {}
  Status Open(const execution::Request& q, execution::Allocation& pool,
              peer::PeerRegion* out) override {
    profile_ = q;
    if (q.cpu_threads) options_.gather_threads = q.cpu_threads;
    auto s = backend_.Initialize(options_, next_);
    if (s.ok())
      s = backend_.AttachRegion(reinterpret_cast<void*>(pool.descriptor.base),
                                pool.descriptor.length, 1, q.candidate, out);
    return s;
  }
  Status Call(const std::vector<std::uint8_t>& in, std::vector<std::uint8_t>* out) override {
    if (in.size() < sizeof(peer::MessageHeader))
      return Status::InvalidArgument("truncated session command");
    peer::MessageHeader header{};
    std::memcpy(&header, in.data(), sizeof(header));
    const auto command = static_cast<peer::Command>(header.command);
    if ((command == peer::Command::kEnableDirectGather ||
         command == peer::Command::kEnableGatherLanes) &&
        (profile_.put || profile_.algorithm != GetAlgorithm::Gather))
      return Status::InvalidArgument("session has no Gather profile");
    if ((command == peer::Command::kOpenCpuPushGet || command == peer::Command::kStartCpuPushGet ||
         command == peer::Command::kStopCpuPushGet) &&
        (profile_.put || profile_.algorithm != GetAlgorithm::Scatter))
      return Status::InvalidArgument("session has no Scatter profile");
    return backend_.Call(in, out);
  }
  Status Close() override { return backend_.Shutdown(); }

 private:
  CpuRuntimeOptions options_;
  execution::Request profile_;
  std::uint64_t next_ = 1;
  UrmaCpuBackend backend_;
};
#endif
}  // namespace
std::unique_ptr<execution::CacheRegionService> CreateExecutionService(
    const CpuRuntimeOptions& options) {
  auto allocate = [options](const execution::Request& q,
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
      auto hal = std::make_shared<ubmem::Hal>();
      auto s = hal->Load(options.hal_library);
      if (!s.ok()) return s;
      p->mapping = std::make_unique<ubmem::Mapping>(hal);
      s = p->mapping->AllocateHost(
          (bytes + ubmem::kPageBytes - 1) / ubmem::kPageBytes * ubmem::kPageBytes,
          &p->descriptor.fabric_handle);
      if (!s.ok()) return s;
      p->descriptor.base = p->mapping->address();
      p->descriptor.fabric_mapping_bytes = p->mapping->bytes();
    } else {
      const std::size_t alignment =
          options.memory_page_mode == CpuMemoryPageMode::kTransparentHugePage ? 2U * 1024U * 1024U
                                                                              : 4096U;
      if (posix_memalign(&p->memory, alignment, bytes))
        return Status::Internal("DDR allocation failed");
#if defined(__linux__)
      if (options.memory_page_mode == CpuMemoryPageMode::kTransparentHugePage &&
          madvise(p->memory, bytes, MADV_HUGEPAGE))
        return Status::Internal("DDR huge-page advice failed");
#endif
      if (options.memory_page_mode == CpuMemoryPageMode::kTransparentHugePage)
        std::memset(p->memory, 0, bytes);
      p->descriptor.base = reinterpret_cast<std::uintptr_t>(p->memory);
    }
    p->descriptor.length = bytes;
    // Initialize auxiliary controls once, never on each registration of this pool.
    if (q.auxiliary_bytes)
      std::memset(reinterpret_cast<void*>(p->descriptor.base + q.logical_bytes), 0,
                  q.auxiliary_bytes);
    return Status::OK();
  };
  auto create = [options](const execution::Request& q) -> std::unique_ptr<execution::Session> {
    if (q.transport == Transport::Ubmem) return std::make_unique<FabricSession>();
#ifdef KVAF_CPU_HAS_URMA
    return std::make_unique<UrmaSession>(options);
#else
    return {};
#endif
  };
  return std::make_unique<execution::CacheRegionService>(execution::kFabric
#ifdef KVAF_CPU_HAS_URMA
                                                             | execution::kUrma
#endif
                                                         ,
                                                         std::move(allocate), std::move(create));
}
}  // namespace kvaf
