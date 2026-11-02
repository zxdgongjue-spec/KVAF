#include "ubmem/access_operator.h"
#include "ubmem/batch_abi.h"
#include "ubmem/device_mapping.h"
#include "peer_link.h"
#include "meta_manager.h"

#include <acl/acl.h>
#include <cstdio>
#include <fstream>
#include <unordered_map>

namespace kvaf::ubmem {
namespace {
std::uint64_t AddressOf(const void* ptr) { return reinterpret_cast<std::uintptr_t>(ptr); }
}  // namespace

struct AccessOperator::Impl {
  explicit Impl(OperatorOptions value) : options(std::move(value)) {}
  struct StreamState {
    enum class Pending { kNone, kEvent, kStream };
    aclrtEvent event{};
    void* statuses{};
    void* host_statuses{};
    void* metadata{};
    std::size_t capacity{};
    Pending pending{Pending::kNone};
  };
  OperatorOptions options;
  std::unique_ptr<DeviceMapping> mapping;
  RemoteRegionDesc remote{};
  aclrtBinHandle binaries[2]{};
  aclrtFuncHandle functions[2]{};
  std::vector<char> binary_bytes[2];
  std::unordered_map<aclrtStream, StreamState> streams;
  bool ready{};
  bool retained{};
  bool profiling{};
  bool profile_detail{};
  aclrtStream profile_stream{};

  Status LoadKernel(unsigned index, const std::string& path, const char* name) {
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

  Status State(aclrtStream stream, StreamState** result) {
    if (!ready) return Status::InvalidState("UBmem operator is uninitialized or failed; rebuild it");
    if (stream == nullptr) return Status::InvalidArgument("UBmem requires a caller-owned ACL stream");
    auto [it, inserted] = streams.try_emplace(stream);
    auto& state = it->second;
    if (inserted) {
      auto status = Acl("aclrtCreateEvent", aclrtCreateEvent(&state.event));
      if (status.ok()) status = Acl("aclrtMalloc(status)", aclrtMalloc(
          &state.statuses, options.cores * sizeof(CoreStatus), ACL_MEM_MALLOC_NORMAL_ONLY));
      if (status.ok()) status = Acl("aclrtMallocHost(status)", aclrtMallocHost(
          &state.host_statuses, options.cores * sizeof(CoreStatus)));
      if (!status.ok()) {
        if (state.host_statuses) aclrtFreeHost(state.host_statuses);
        if (state.statuses) aclrtFree(state.statuses);
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
    args.profile = profiling && args.operation == Operation::kGet ? (profile_detail ? 2 : 1) : 0;
    profile_stream = nullptr;
    aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI graph = nullptr;
    if (!synchronize) {
      auto status = Acl("aclmdlRICaptureGetInfo",
          aclmdlRICaptureGetInfo(stream, &capture, &graph));
      if (!status.ok()) return status;
    }
    const auto size = options.cores * sizeof(CoreStatus);
    if (synchronize) {
      // Same ordering as KvafExecuteBatchPutSync. An unwritten report must
      // not look like a successful zero-row kernel.
      auto status = Acl("aclrtMemsetAsync(status)", aclrtMemsetAsync(state.statuses, size, 0xFF, size, stream));
      if (!status.ok()) return status;
      state.pending = StreamState::Pending::kStream;
    }
    aclrtLaunchKernelAttr attrs[4]{};
    attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attrs[0].value.schemMode = 1;
    attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attrs[1].value.timeoutUs.timeoutLow = 180000000U;
    attrs[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attrs[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
    attrs[3].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
    attrs[3].value.dynUBufSize = kMteBufferBytes;
    aclrtLaunchKernelCfg cfg{}; cfg.numAttrs = 4; cfg.attrs = attrs;
    auto status = Acl("aclrtLaunchKernelWithHostArgs", aclrtLaunchKernelWithHostArgs(
        functions[static_cast<unsigned>(args.operation)], options.cores, stream, &cfg, &args, sizeof(args), nullptr, 0));
    if (!status.ok()) return status;
    state.pending = StreamState::Pending::kStream;
    if (args.profile) profile_stream = stream;
    if (!synchronize) {
      // Ordinary events cannot be recorded during ACL Graph capture. The
      // graph owner completes replay on this stream before releasing it;
      // Shutdown retains the stream drain for captured work.
      if (capture == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) return Status::OK();
      status = Acl("aclrtRecordEvent", aclrtRecordEvent(state.event, stream));
      if (!status.ok()) {
        ready = false;
        // No event: drain now or leave Shutdown responsible for retrying.
        if (aclrtSynchronizeStream(stream) == ACL_SUCCESS) state.pending = StreamState::Pending::kNone;
        return status;
      }
      state.pending = StreamState::Pending::kEvent;
      return Status::OK();
    }
    // Keep the kernel, status D2H and completion wait on the caller's stream,
    // exactly as the existing synchronous BatchPut path does.
    status = Acl("aclrtMemcpyAsync(status)", aclrtMemcpyAsync(state.host_statuses, size,
        state.statuses, size, ACL_MEMCPY_DEVICE_TO_HOST, stream));
    if (status.ok()) status = Acl("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
    if (!status.ok()) {
      std::fprintf(stderr, "[KVAF_UBMEM] FAILED_LAUNCH op=%u rows=%u local_base=0x%llx local_bytes=%llu "
          "mapped_base=0x%llx remote_bytes=%llu metadata=0x%llx\n",
          static_cast<unsigned>(args.operation), args.capacity,
          static_cast<unsigned long long>(args.local_base),
          static_cast<unsigned long long>(args.local_bytes),
          static_cast<unsigned long long>(args.mapped_base),
          static_cast<unsigned long long>(args.remote_bytes),
          static_cast<unsigned long long>(args.put_offsets));
      ready = false;
      return status;
    }
    state.pending = StreamState::Pending::kNone;
    const auto* reports = static_cast<const CoreStatus*>(state.host_statuses);
    for (std::size_t i = 0; i < options.cores; ++i) {
      if (reports[i].result == UINT64_MAX) {
        return Status::Internal("UBmem kernel status was not written (still 0xFF)");
      }
      if (reports[i].result != 0) {
        std::fprintf(stderr, "[KVAF_UBMEM] KERNEL_ERROR core=%zu code=%llu request=%llu\n", i,
            static_cast<unsigned long long>(reports[i].result),
            static_cast<unsigned long long>(reports[i].failed_request));
        return Status::InvalidArgument("UBmem kernel rejected a request; see KERNEL_ERROR");
      }
      if (args.operation == Operation::kPut &&
          reports[i].completed != static_cast<std::uint64_t>(args.capacity) * (i + 1) / args.blocks -
              static_cast<std::uint64_t>(args.capacity) * i / args.blocks) {
        return Status::Internal("UBmem Put completion report does not cover the submitted rows");
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
  if (impl_->mapping || impl_->options.cores == 0 || impl_->options.cores > 48 ||
      !impl_->options.peer_link || region.device != DeviceKind::kCpu || region.base == 0 ||
      region.length > UINT64_MAX - region.base) {
    return Status::InvalidArgument("invalid UBmem configuration/remote region");
  }
  auto status = Acl("aclrtSetDevice", aclrtSetDevice(impl_->options.device_id));
  peer::PeerRegion peer;
  if (status.ok()) status = impl_->options.peer_link->LookupRegion(region.base, &peer);
  if (status.ok()) status = impl_->options.peer_link->RetainUbmemMapping(region.base);
  if (status.ok()) {
    impl_->remote = region;
    impl_->retained = true;
    impl_->mapping = std::make_unique<DeviceMapping>();
    status = impl_->mapping->Import(peer.fabric_handle, peer.fabric_mapping_bytes);
  }
  if (status.ok()) status = impl_->LoadKernel(0, impl_->options.batch_get_kernel_object_path,
      "KvafExecuteBatchGetUbmemKernelV2");
  if (status.ok() && !impl_->options.batch_put_kernel_object_path.empty())
    status = impl_->LoadKernel(1, impl_->options.batch_put_kernel_object_path,
        "KvafExecuteBatchPutUbmemKernelV2");
  if (!status.ok()) { (void)Shutdown(); return status; }
  impl_->ready = true;
  std::fprintf(stderr, "[KVAF_UBMEM] DEVICE_MAPPING_READY transport=ubmem mapping=acl ub_bytes=%u bytes=%llu cores=%u acl_device=%u "
      "host_base=0x%llx mapped_base=0x%llx\n",
      kMteBufferBytes,
      static_cast<unsigned long long>(region.length), impl_->options.cores,
      impl_->options.device_id,
      static_cast<unsigned long long>(region.base),
      static_cast<unsigned long long>(impl_->mapping->address()));
  return Status::OK();
}

Status AccessOperator::Shutdown() {
  impl_->ready = false;
  impl_->profiling = false;
  impl_->profile_stream = nullptr;
  for (auto& entry : impl_->streams) {
    if (entry.second.pending == Impl::StreamState::Pending::kStream) {
      auto status = Acl("aclrtSynchronizeStream(recovery)", aclrtSynchronizeStream(entry.first));
      if (!status.ok()) return status;
    } else if (entry.second.pending == Impl::StreamState::Pending::kEvent) {
      auto status = Acl("aclrtSynchronizeEvent(shutdown)", aclrtSynchronizeEvent(entry.second.event));
      if (!status.ok()) return status;
    }
    entry.second.pending = Impl::StreamState::Pending::kNone;
  }
  for (auto& entry : impl_->streams) {
    auto& s = entry.second;
    if (s.metadata) aclrtFree(s.metadata);
    if (s.statuses) aclrtFree(s.statuses);
    if (s.host_statuses) aclrtFreeHost(s.host_statuses);
    if (s.event) aclrtDestroyEvent(s.event);
  }
  impl_->streams.clear();
  for (unsigned i = 0; i < 2; ++i) {
    if (impl_->binaries[i]) {
      auto status = Acl("aclrtBinaryUnLoad", aclrtBinaryUnLoad(impl_->binaries[i]));
      if (!status.ok()) return status;
      impl_->binaries[i] = nullptr;
      impl_->functions[i] = nullptr;
    }
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
  return Status::OK();
}

Status AccessOperator::SetBatchGetKernelProfiling(bool enabled, bool detailed) {
  if (!impl_->ready) return Status::NotInitialized("UBmem operator is not initialized");
  impl_->profiling = enabled;
  impl_->profile_detail = enabled && detailed;
  impl_->profile_stream = nullptr;
  return Status::OK();
}

Status AccessOperator::CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const {
  if (!profiles) return Status::InvalidArgument("profile output is null");
  if (!impl_->ready || !impl_->profiling || !impl_->profile_stream)
    return Status::InvalidState("no profiled UBmem BatchGet launch");
  const auto& state = impl_->streams.at(impl_->profile_stream);
  std::vector<CoreStatus> reports(impl_->options.cores);
  // As for URMA profiling, the caller completes the stream before this snapshot.
  auto status = Acl("aclrtMemcpy(profile)", aclrtMemcpy(reports.data(), reports.size() * sizeof(CoreStatus),
      state.statuses, reports.size() * sizeof(CoreStatus), ACL_MEMCPY_DEVICE_TO_HOST));
  if (!status.ok()) return status;
  profiles->clear();
  for (const auto& report : reports) {
    if (report.result != 0 || report.profile_magic != kProfileMagic || report.finish_cycle < report.start_cycle) {
      profiles->clear();
      return Status::Internal("invalid UBmem BatchGet profile");
    }
    BatchGetKernelProfile profile{};
    profile.kernel_start_cycle = report.start_cycle;
    profile.kernel_finish_cycle = report.finish_cycle;
    profile.active_requests = report.completed;
    profile.ubmem_profile_detail = report.profile_detail != 0;
    if (profile.ubmem_profile_detail) {
      profile.ubmem_setup_cycles = report.setup_cycles;
      profile.ubmem_address_cycles = report.address_cycles;
      profile.ubmem_submit_cycles = report.submit_cycles;
      profile.ubmem_drain_cycles = report.drain_cycles;
    }
    profiles->push_back(profile);
  }
  return Status::OK();
}

Status AccessOperator::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  // RuntimeCore owns buffer/layout validation. This adapter only restricts
  // the device and checks narrowing to its kernel ABI.
  const auto& r = *exec.request;
  if (r.output.device != DeviceKind::kNpu ||
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
  if (!impl_->functions[1]) return Status::NotImplemented("UBmem BatchPut kernel is not configured");
  const auto& r = *exec.request;
  const auto rows = r.keys.size();
  if (r.input.device != DeviceKind::kNpu || rows > UINT32_MAX) {
    return Status::InvalidArgument("invalid UBmem BatchPut metadata");
  }
  // RuntimeCore checked source ranges; MetaManager allocated these destination
  // ranges in the region supplied to Initialize, in request order.
  std::vector<std::uint64_t> metadata(rows * 3);
  for (std::size_t i = 0; i < rows; ++i) {
    metadata[i] = r.input_offsets[i]; metadata[rows + i] = r.lengths[i];
    metadata[2 * rows + i] = exec.put_desc->entries[i].address;
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
