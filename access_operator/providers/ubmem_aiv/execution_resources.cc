#include "ubmem/execution_resources.h"
#include "peer_link.h"
#include <cstdio>
#include <fstream>
#include <limits>

namespace kvaf::ubmem {
Status UbmemResources::LoadKernel(unsigned index, const std::string& path, const char* name) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() <= 0) return Status::NotFound("cannot read UBmem kernel: " + path);
  auto& bytes = binary_bytes[index];
  bytes.resize(static_cast<std::size_t>(file.tellg()));
  file.seekg(0);
  if (!file.read(bytes.data(), bytes.size())) return Status::Internal("incomplete kernel object");
  aclrtBinary input = aclrtCreateBinary(bytes.data(), bytes.size());
  if (!input) return Status::Internal("aclrtCreateBinary failed");
  auto status = Acl("aclrtBinaryLoad", aclrtBinaryLoad(input, &binaries[index]));
  aclrtDestroyBinary(input);
  if (status.ok()) status = Acl("aclrtBinaryGetFunction",
      aclrtBinaryGetFunction(binaries[index], name, &functions[index]));
  return status;
}
Status UbmemResources::Slot(aclrtStream stream, ExecutionSlot** result) {
  if (!ready || health.failed) return Status::InvalidState("UBMEM resources are uninitialized or failed");
  if (!stream) return Status::InvalidArgument("UBmem requires a caller-owned ACL stream");
  auto& state = streams[stream];
  // Keep partial allocations in the owner so failed cleanup is retryable.
  auto status = state.event ? Status::OK() : Acl("aclrtCreateEvent", aclrtCreateEvent(&state.event));
  if (status.ok() && !state.statuses)
    status = Acl("aclrtMalloc(status)", aclrtMalloc(
        &state.statuses, options.cores * sizeof(CoreStatus), ACL_MEM_MALLOC_NORMAL_ONLY));
  if (status.ok() && !state.host_statuses)
    status = Acl("aclrtMallocHost(status)", aclrtMallocHost(
        &state.host_statuses, options.cores * sizeof(CoreStatus)));
  if (status.ok()) *result = &state;
  return status;
}
Status UbmemResources::Initialize(const RemoteRegionDesc& region) {
  if (mapping || options.cores == 0 || options.cores > 48 ||
      !options.peer_link || region.device != DeviceKind::kCpu || region.base == 0 ||
      region.length > UINT64_MAX - region.base) {
    return Status::InvalidArgument("invalid UBmem configuration/remote region");
  }
  auto status = Acl("aclrtSetDevice", aclrtSetDevice(options.device_id));
  peer::PeerRegion peer;
  if (status.ok()) status = options.peer_link->LookupRegion(region.base, &peer);
  if (status.ok()) status = options.peer_link->RetainUbmemMapping(region.base);
  if (status.ok()) {
    remote = region;
    retained = true;
    if (options.mappings)
      status = options.mappings->Acquire(options.device_id, peer.fabric_handle,
                                                peer.fabric_mapping_bytes, &mapping);
    else {
      mapping = std::make_shared<DeviceMapping>();
      status = mapping->Import(peer.fabric_handle, peer.fabric_mapping_bytes);
    }
  }
  if (status.ok() && !options.batch_get_kernel_object_path.empty()) status = LoadKernel(0, options.batch_get_kernel_object_path,
      "KvafExecuteBatchGetUbmemKernelV2");
  if (status.ok() && !options.batch_put_kernel_object_path.empty())
    status = LoadKernel(1, options.batch_put_kernel_object_path,
        "KvafExecuteBatchPutUbmemKernelV2");
  if (!status.ok()) { (void)Close(); return status; }
  ready = true;
  health.failed = false;
  views.Initialize(region, peer.region_id, peer.epoch);
  std::fprintf(stderr, "[KVAF_UBMEM] DEVICE_MAPPING_READY transport=ubmem mapping=acl ub_bytes=%u bytes=%llu cores=%u acl_device=%u "
      "host_base=0x%llx mapped_base=0x%llx\n",
      kMteBufferBytes,
      static_cast<unsigned long long>(region.length), options.cores,
      options.device_id,
      static_cast<unsigned long long>(region.base),
      static_cast<unsigned long long>(mapping->address()));
  return Status::OK();
}
Status UbmemResources::Close() {
  std::lock_guard<std::mutex> lock(mutex);
  if (invocation->graph_active) return Status::Busy("release UBMEM graph lease before shutdown");
  views.Close();
  ready = false;
  profiling = false;
  invocation->profile_stream = nullptr;
  for (auto& entry : streams) {
    if (invocation->pending[entry.first] == ExecutionSlot::Pending::kStream) {
      auto status = Acl("aclrtSynchronizeStream(recovery)", aclrtSynchronizeStream(entry.first));
      if (!status.ok()) return status;
    } else if (invocation->pending[entry.first] == ExecutionSlot::Pending::kEvent) {
      auto status = Acl("aclrtSynchronizeEvent(shutdown)", aclrtSynchronizeEvent(entry.second.event));
      if (!status.ok()) return status;
    }
    invocation->pending[entry.first] = ExecutionSlot::Pending::kNone;
  }
  for (auto& entry : streams) {
    auto& s = entry.second;
    if (s.metadata) {
      auto status = Acl("aclrtFree(metadata)", aclrtFree(s.metadata));
      if (!status.ok()) return status;
      s.metadata = nullptr; s.capacity = 0;
    }
    if (s.statuses) {
      auto status = Acl("aclrtFree(status)", aclrtFree(s.statuses));
      if (!status.ok()) return status;
      s.statuses = nullptr;
    }
    if (s.host_statuses) {
      auto status = Acl("aclrtFreeHost(status)", aclrtFreeHost(s.host_statuses));
      if (!status.ok()) return status;
      s.host_statuses = nullptr;
    }
    if (s.event) {
      auto status = Acl("aclrtDestroyEvent", aclrtDestroyEvent(s.event));
      if (!status.ok()) return status;
      s.event = nullptr;
    }
  }
  streams.clear();
  for (unsigned i = 0; i < 2; ++i) {
    if (binaries[i]) {
      auto status = Acl("aclrtBinaryUnLoad", aclrtBinaryUnLoad(binaries[i]));
      if (!status.ok()) return status;
      binaries[i] = nullptr;
      functions[i] = nullptr;
    }
  }
  if (mapping) {
    auto status = options.mappings ? options.mappings->Release(&mapping)
                                          : mapping->Close();
    if (!status.ok()) return status;
    mapping.reset();
  }
  if (retained) {
    options.peer_link->ReleaseUbmemMapping(remote.base);
    retained = false;
  }
  return Status::OK();
}

UbmemResources::~UbmemResources() {
  if (!Close().ok()) {
    // Failed fencing must not unmap storage still used by a device. Raw slot
    // allocations/modules remain retained as well; normal shutdown is retryable.
    struct RetainedDependencies {
      OperatorOptions options;
      std::shared_ptr<DeviceMapping> mapping;
      std::vector<char> binaries[2];
    };
    (void)new RetainedDependencies{std::move(options), std::move(mapping),
                                  {std::move(binary_bytes[0]), std::move(binary_bytes[1])}};
  }
}
Status UbmemResources::ReserveMetadata(ExecutionSlot& slot, std::uint32_t capacity) {
  if (std::size_t(capacity) > std::numeric_limits<std::size_t>::max() / (3 * sizeof(std::uint64_t)))
    return Status::InvalidArgument("UBMEM metadata size overflow");
  const auto bytes = std::size_t(capacity) * 3 * sizeof(std::uint64_t);
  if (slot.capacity >= bytes) return Status::OK();
  void* allocation = nullptr;
  auto s = Acl("aclrtMalloc(metadata)", aclrtMalloc(&allocation, bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
  if (!s.ok()) return s;
  if (slot.metadata) aclrtFree(slot.metadata);
  slot.metadata = allocation;
  slot.capacity = bytes;
  return Status::OK();
}
MappedExecutionView UbmemResources::View(unsigned index, aclrtStream stream, ExecutionSlot& slot) {
  invocation->prepared_stream = stream;
  return {views.View(index ? ExecutionDirection::Put : ExecutionDirection::Get),
          {mapping->address(), remote.length}, {functions[index], options.cores},
          profiling, profile_detail, &slot, &health};
}
Status UbmemResources::PreparePutResources(aclrtStream stream, std::uint32_t capacity, UbmemPutView* out) {
  if (!out) return Status::InvalidArgument("null UBMEM Put view");
  views.Prepare();
  if (!functions[1]) return Status::NotImplemented("UBmem BatchPut kernel is not configured");
  ExecutionSlot* slot = nullptr;
  auto s = Slot(stream, &slot);
  if (s.ok()) s = ReserveMetadata(*slot, capacity);
  if (s.ok()) *out = UbmemPutView(View(1, stream, *slot));
  return s;
}
Status UbmemResources::PrepareGetResources(aclrtStream stream, UbmemGetView* out) {
  if (!out) return Status::InvalidArgument("null UBMEM Get view");
  views.Prepare();
  if (!functions[0]) return Status::NotImplemented("UBmem BatchGet kernel is not configured");
  ExecutionSlot* slot = nullptr;
  auto s = Slot(stream, &slot);
  if (s.ok()) *out = UbmemGetView(View(0, stream, *slot));
  return s;
}
Status UbmemResources::Warmup(const ExecutionWarmup& request) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!ready || health.failed) return Status::NotInitialized("UBMEM resources not initialized");
  if (!request.stream) return Status::OK();
  views.Prepare();
  ExecutionSlot* slot = nullptr;
  auto s = Slot(static_cast<aclrtStream>(request.stream), &slot);
  if (!s.ok() || !functions[1] || !request.request_capacity) return s;
  return ReserveMetadata(*slot, request.request_capacity);
}
std::unique_ptr<ExecutionState> UbmemResources::CreateState() const {
  return std::make_unique<UbmemInvocation>(this);
}
Status UbmemResources::Bind(ExecutionState* state) {
  std::lock_guard<std::mutex> lock(mutex);
  auto* typed = state ? dynamic_cast<UbmemInvocation*>(state) : &default_invocation;
  if (!typed || typed->owner != this)
    return Status::InvalidArgument("UBMEM execution state belongs to another resource");
  views.Bind(typed);
  invocation = typed;
  return Status::OK();
}
}  // namespace kvaf::ubmem
