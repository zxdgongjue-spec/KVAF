#include "ubmem/access_operator.h"
#include "ubmem/batch_abi.h"
#include "ubmem/hal.h"
#include "peer_link.h"
#include "meta_manager.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <cstdio>
#include <fstream>
#include <limits>
#include <unordered_map>

namespace kvaf::ubmem {
namespace {
Status Acl(const char* operation, aclError code) {
  return code == ACL_SUCCESS ? Status::OK() : HalError(operation, code);
}
std::uint64_t AddressOf(const void* ptr) { return reinterpret_cast<std::uintptr_t>(ptr); }
bool DeviceBuffer(const ConstBufferView& buffer, std::size_t unit) {
  return buffer.valid() && buffer.device == DeviceKind::kNpu && buffer.bytes % unit == 0;
}
Status DriverDevice(std::uint32_t user_device, std::uint32_t* device) {
  void* library = dlopen("libruntime.so", RTLD_NOW | RTLD_LOCAL);
  if (!library) return Status::NotFound("cannot load libruntime.so for HAL logical-device resolution");
  using Resolve = int(*)(int32_t, int32_t*);
  auto resolve = reinterpret_cast<Resolve>(dlsym(library, "rtGetLogicDevIdByUserDevId"));
  int32_t result = -1;
  const int ret = resolve ? resolve(user_device, &result) : -1;
  dlclose(library);
  if (ret != 0 || result < 0) return HalError("rtGetLogicDevIdByUserDevId", ret);
  *device = result;
  return Status::OK();
}
}  // namespace

struct AccessOperator::Impl {
  explicit Impl(OperatorOptions value) : options(std::move(value)) {}
  struct StreamState {
    aclrtEvent event{};
    void* statuses{};
    void* metadata{};
    std::size_t capacity{};
    bool recorded{};
    bool needs_stream_drain{};
  };
  OperatorOptions options;
  std::shared_ptr<Hal> hal = std::make_shared<Hal>();
  std::unique_ptr<Mapping> mapping;
  RemoteRegionDesc remote{};
  aclrtBinHandle binary{};
  aclrtFuncHandle function{};
  std::vector<char> binary_bytes;
  std::unordered_map<aclrtStream, StreamState> streams;
  bool ready{};
  bool poisoned{};
  bool retained{};

  Status LoadKernel() {
    std::ifstream file(options.kernel_object_path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) return Status::NotFound("cannot read UBmem kernel object");
    const auto size = file.tellg();
    binary_bytes.resize(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!file.read(binary_bytes.data(), size)) return Status::Internal("cannot read complete UBmem kernel object");
    aclrtBinary input = aclrtCreateBinary(binary_bytes.data(), binary_bytes.size());
    if (input == nullptr) return Status::Internal("aclrtCreateBinary failed");
    auto status = Acl("aclrtBinaryLoad", aclrtBinaryLoad(input, &binary));
    aclrtDestroyBinary(input);
    if (status.ok()) status = Acl("aclrtBinaryGetFunction",
        aclrtBinaryGetFunction(binary, "KvafUbmemBatchKernel", &function));
    return status;
  }

  Status State(aclrtStream stream, StreamState** result) {
    if (!ready || poisoned) return Status::InvalidState("UBmem operator is uninitialized or failed; rebuild it");
    if (stream == nullptr) return Status::InvalidArgument("UBmem requires a caller-owned ACL stream");
    auto [it, inserted] = streams.try_emplace(stream);
    auto& state = it->second;
    if (inserted) {
      auto status = Acl("aclrtCreateEvent", aclrtCreateEvent(&state.event));
      if (status.ok()) status = Acl("aclrtMalloc(status)", aclrtMalloc(
          &state.statuses, options.cores * sizeof(CoreStatus), ACL_MEM_MALLOC_NORMAL_ONLY));
      if (!status.ok()) {
        if (state.event) aclrtDestroyEvent(state.event);
        streams.erase(it);
        return status;
      }
    }
    *result = &state;
    return Status::OK();
  }

  BatchArgs Args(Operation operation, void* local, std::size_t bytes, StreamState& state) const {
    BatchArgs a{};
    a.magic = kBatchMagic;
    a.operation = operation;
    a.remote_base = remote.base;
    a.remote_bytes = remote.length;
    a.mapped_base = mapping->address();
    a.local_base = AddressOf(local);
    a.local_bytes = bytes;
    a.statuses = AddressOf(state.statuses);
    a.blocks = options.cores;
    return a;
  }

  Status Launch(BatchArgs& args, aclrtStream stream, StreamState& state, bool synchronize) {
    aclrtLaunchKernelAttr attrs[3]{};
    attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attrs[0].value.schemMode = 1;
    attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attrs[1].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
    attrs[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attrs[2].value.timeoutUs.timeoutLow = 180000000U;
    aclrtLaunchKernelCfg cfg{}; cfg.numAttrs = 3; cfg.attrs = attrs;
    auto status = Acl("aclrtLaunchKernelWithHostArgs", aclrtLaunchKernelWithHostArgs(
        function, options.cores, stream, &cfg, &args, sizeof(args), nullptr, 0));
    if (!status.ok()) return status;
    state.needs_stream_drain = true;
    status = Acl("aclrtRecordEvent", aclrtRecordEvent(state.event, stream));
    if (!status.ok()) {
      poisoned = true;
      (void)aclrtSynchronizeStream(stream);
      return status;
    }
    state.recorded = true;
    state.needs_stream_drain = false;
    if (!synchronize) return Status::OK();
    status = Acl("aclrtSynchronizeEvent", aclrtSynchronizeEvent(state.event));
    if (!status.ok()) { poisoned = true; return status; }
    std::vector<CoreStatus> reports(options.cores);
    const auto size = reports.size() * sizeof(CoreStatus);
    status = Acl("aclrtMemcpy(status)", aclrtMemcpy(reports.data(), size, state.statuses, size,
                                                   ACL_MEMCPY_DEVICE_TO_HOST));
    if (!status.ok()) return status;
    for (std::size_t i = 0; i < reports.size(); ++i) {
      if (reports[i].magic != kBatchMagic || reports[i].result != 0) {
        std::fprintf(stderr, "[KVAF_UBMEM] KERNEL_ERROR core=%zu code=%llu request=%llu\n", i,
            static_cast<unsigned long long>(reports[i].result),
            static_cast<unsigned long long>(reports[i].failed_request));
        return Status::InvalidArgument("UBmem kernel rejected a request; see KERNEL_ERROR");
      }
    }
    return Status::OK();
  }
};

AccessOperator::AccessOperator(OperatorOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
AccessOperator::~AccessOperator() {
  if (!Shutdown().ok()) {
    std::fprintf(stderr, "[KVAF_UBMEM] retaining mapping/resources after failed drain or cleanup\n");
    (void)impl_.release();
  }
}

Status AccessOperator::Initialize(const RemoteRegionDesc& region) {
  if (impl_->mapping || impl_->options.cores == 0 || impl_->options.cores > 32 ||
      !impl_->options.peer_link || region.device != DeviceKind::kCpu || region.base == 0 ||
      region.length > UINT64_MAX - region.base) {
    return Status::InvalidArgument("invalid UBmem configuration/remote region");
  }
  auto status = Acl("aclrtSetDevice", aclrtSetDevice(impl_->options.device_id));
  std::uint32_t logical_device = 0;
  if (status.ok()) status = DriverDevice(impl_->options.device_id, &logical_device);
  peer::PeerRegion peer;
  if (status.ok()) status = impl_->options.peer_link->LookupRegion(region.base, &peer);
  if (status.ok()) status = impl_->hal->Load(impl_->options.hal_library);
  if (status.ok()) status = impl_->options.peer_link->RetainUbmemMapping(region.base);
  if (status.ok()) {
    impl_->remote = region;
    impl_->retained = true;
    impl_->mapping = std::make_unique<Mapping>(impl_->hal);
    status = impl_->mapping->Import(peer.fabric_handle, peer.fabric_mapping_bytes, logical_device);
  }
  if (status.ok()) status = impl_->LoadKernel();
  if (!status.ok()) { (void)Shutdown(); return status; }
  impl_->remote = region;
  impl_->ready = true;
  impl_->poisoned = false;
  std::fprintf(stderr, "[KVAF_UBMEM] DEVICE_MAPPING_READY bytes=%llu cores=%u\n",
      static_cast<unsigned long long>(region.length), impl_->options.cores);
  return Status::OK();
}

Status AccessOperator::Shutdown() {
  impl_->ready = false;
  for (auto& entry : impl_->streams) {
    if (entry.second.needs_stream_drain) {
      auto status = Acl("aclrtSynchronizeStream(recovery)", aclrtSynchronizeStream(entry.first));
      if (!status.ok()) return status;
      entry.second.needs_stream_drain = false;
    }
    if (entry.second.recorded) {
      auto status = Acl("aclrtSynchronizeEvent(shutdown)", aclrtSynchronizeEvent(entry.second.event));
      if (!status.ok()) return status;
    }
  }
  for (auto& entry : impl_->streams) {
    auto& s = entry.second;
    if (s.metadata) aclrtFree(s.metadata);
    if (s.statuses) aclrtFree(s.statuses);
    if (s.event) aclrtDestroyEvent(s.event);
  }
  impl_->streams.clear();
  if (impl_->binary) {
    auto status = Acl("aclrtBinaryUnLoad", aclrtBinaryUnLoad(impl_->binary));
    if (!status.ok()) return status;
    impl_->binary = nullptr;
    impl_->function = nullptr;
  }
  if (impl_->mapping) {
    auto status = impl_->mapping->Close();
    if (!status.ok()) return status;
    impl_->mapping.reset();
  }
  if (impl_->retained) {
    impl_->options.peer_link->ReleaseUbmemMapping(impl_->remote.base);
    impl_->retained = false;
  }
  impl_->ready = false;
  return Status::OK();
}

Status AccessOperator::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  if (!exec.request) return Status::InvalidArgument("missing UBmem BatchGet request");
  const auto& r = *exec.request;
  if (!DeviceBuffer(r.indices, 4) || !DeviceBuffer(r.offsets, 4) ||
      !DeviceBuffer(r.output_indices, 4) || !DeviceBuffer(r.miss_count, 4) || r.miss_count.bytes != 4 ||
      r.indices.bytes != r.offsets.bytes || r.indices.bytes != r.output_indices.bytes ||
      !DeviceBuffer(r.device_access_table.addresses, 8) || !DeviceBuffer(r.device_access_table.valid, 1) ||
      r.device_access_table.valid.bytes != r.device_access_table.addresses.bytes / 8 ||
      !r.output.valid() || r.output.device != DeviceKind::kNpu || r.output_block_bytes == 0 ||
      r.output.bytes % r.output_block_bytes != 0 || r.tokens_per_block == 0 ||
      r.output_block_bytes > UINT32_MAX || r.tokens_per_block > UINT32_MAX ||
      r.indices.bytes / 4 > INT32_MAX || r.device_access_table.valid.bytes > UINT32_MAX) {
    return Status::InvalidArgument("invalid UBmem BatchGet device buffers/layout");
  }
  Impl::StreamState* state;
  auto stream = static_cast<aclrtStream>(exec.execution.stream);
  auto status = impl_->State(stream, &state);
  if (!status.ok()) return status;
  auto a = impl_->Args(Operation::kGet, r.output.data, r.output.bytes, *state);
  a.addresses = AddressOf(r.device_access_table.addresses.data);
  a.valid = AddressOf(r.device_access_table.valid.data);
  a.indices = AddressOf(r.indices.data);
  a.offsets = AddressOf(r.offsets.data);
  a.output_indices = AddressOf(r.output_indices.data);
  a.miss_count = AddressOf(r.miss_count.data);
  a.capacity = r.indices.bytes / 4;
  a.table_entries = r.device_access_table.valid.bytes;
  a.token_bytes = r.output_block_bytes;
  a.tokens_per_block = r.tokens_per_block;
  return impl_->Launch(a, stream, *state, exec.execution.synchronize);
}

Status AccessOperator::ExecuteBatchPut(const BatchPutExecRequest& exec) {
  if (!exec.request || !exec.put_desc) return Status::InvalidArgument("missing UBmem BatchPut request/allocations");
  const auto& r = *exec.request;
  const auto rows = r.keys.size();
  if (!DeviceBuffer(r.input, 1) || rows == 0 || rows > UINT32_MAX || r.input_offsets.size() != rows ||
      r.lengths.size() != rows || exec.put_desc->entries.size() != rows || rows > SIZE_MAX / 24) {
    return Status::InvalidArgument("invalid UBmem BatchPut metadata");
  }
  std::vector<std::uint64_t> metadata(rows * 3);
  for (std::size_t i = 0; i < rows; ++i) {
    const auto& dest = exec.put_desc->entries[i];
    if (r.lengths[i] == 0 || r.input_offsets[i] > r.input.bytes ||
        r.lengths[i] > r.input.bytes - r.input_offsets[i] || dest.key != r.keys[i] ||
        dest.reserved_length != r.lengths[i] || dest.address < impl_->remote.base ||
        r.lengths[i] > impl_->remote.length ||
        dest.address - impl_->remote.base > impl_->remote.length - r.lengths[i]) {
      return Status::InvalidArgument("UBmem BatchPut source/destination is outside its region");
    }
    metadata[i] = r.input_offsets[i]; metadata[rows + i] = r.lengths[i];
    metadata[2 * rows + i] = dest.address;
  }
  Impl::StreamState* state;
  auto stream = static_cast<aclrtStream>(exec.execution.stream);
  auto status = impl_->State(stream, &state);
  if (!status.ok()) return status;
  const auto bytes = metadata.size() * sizeof(std::uint64_t);
  if (state->capacity < bytes) {
    void* buffer = nullptr;
    status = Acl("aclrtMalloc(metadata)", aclrtMalloc(&buffer, bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    if (!status.ok()) return status;
    if (state->metadata) aclrtFree(state->metadata);
    state->metadata = buffer; state->capacity = bytes;
  }
  status = Acl("aclrtMemcpy(metadata)", aclrtMemcpy(state->metadata, bytes, metadata.data(), bytes,
                                                  ACL_MEMCPY_HOST_TO_DEVICE));
  if (!status.ok()) return status;
  auto a = impl_->Args(Operation::kPut, const_cast<void*>(r.input.data), r.input.bytes, *state);
  a.capacity = rows;
  a.put_offsets = AddressOf(state->metadata);
  a.put_lengths = a.put_offsets + rows * 8;
  a.put_destinations = a.put_lengths + rows * 8;
  return impl_->Launch(a, stream, *state, true);
}
}  // namespace kvaf::ubmem
