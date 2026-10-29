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
  aclrtBinHandle binary{};
  aclrtFuncHandle function{};
  std::vector<char> binary_bytes;
  std::unordered_map<aclrtStream, StreamState> streams;
  bool ready{};
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
        aclrtBinaryGetFunction(binary, "KvafUbmemBatchMteKernel", &function));
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
    attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
    attrs[1].value.dynUBufSize = kMteBufferBytes;
    attrs[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attrs[2].value.timeoutUs.timeoutLow = 180000000U;
    attrs[3].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attrs[3].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
    aclrtLaunchKernelCfg cfg{}; cfg.numAttrs = 4; cfg.attrs = attrs;
    auto status = Acl("aclrtLaunchKernelWithHostArgs", aclrtLaunchKernelWithHostArgs(
        function, options.cores, stream, &cfg, &args, sizeof(args), nullptr, 0));
    if (!status.ok()) return status;
    state.pending = StreamState::Pending::kStream;
    if (!synchronize) {
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
    // Print every Put core before validating, including cores after a failure.
    for (std::size_t i = 0; i < options.cores; ++i) {
      if (args.operation == Operation::kPut) {
        const std::uint64_t expected = args.capacity / args.blocks + (i < args.capacity % args.blocks);
        std::fprintf(stderr, "[KVAF_UBMEM] PUT_CORE core=%zu result=%llu completed=%llu expected=%llu "
            "failed_request=%llu seen_capacity=%llu seen_blocks=%llu seen_op=%llu\n", i,
            static_cast<unsigned long long>(reports[i].result),
            static_cast<unsigned long long>(reports[i].completed),
            static_cast<unsigned long long>(expected),
            static_cast<unsigned long long>(reports[i].failed_request),
            static_cast<unsigned long long>(reports[i].capacity),
            static_cast<unsigned long long>(reports[i].blocks),
            static_cast<unsigned long long>(reports[i].operation));
      }
    }
    for (std::size_t i = 0; i < options.cores; ++i) {
      if (reports[i].result == UINT64_MAX) {
        return Status::Internal("UBmem kernel status was not written (still 0xFF)");
      }
      if (reports[i].capacity != args.capacity || reports[i].blocks != args.blocks ||
          reports[i].operation != static_cast<std::uint32_t>(args.operation)) {
        return Status::Internal("UBmem kernel launch fields differ from the host request");
      }
      if (reports[i].result != 0) {
        std::fprintf(stderr, "[KVAF_UBMEM] KERNEL_ERROR core=%zu code=%llu request=%llu\n", i,
            static_cast<unsigned long long>(reports[i].result),
            static_cast<unsigned long long>(reports[i].failed_request));
        return Status::InvalidArgument("UBmem kernel rejected a request; see KERNEL_ERROR");
      }
      if (args.operation == Operation::kPut &&
          reports[i].completed != args.capacity / args.blocks + (i < args.capacity % args.blocks)) {
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
  if (impl_->mapping || impl_->options.cores == 0 || impl_->options.cores > 32 ||
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
  if (status.ok()) status = impl_->LoadKernel();
  if (!status.ok()) { (void)Shutdown(); return status; }
  impl_->ready = true;
  std::fprintf(stderr, "[KVAF_UBMEM] DEVICE_MAPPING_READY engine=mte mapping=acl ub_bytes=%u bytes=%llu cores=%u acl_device=%u "
      "host_base=0x%llx mapped_base=0x%llx\n",
      kMteBufferBytes, static_cast<unsigned long long>(region.length), impl_->options.cores,
      impl_->options.device_id,
      static_cast<unsigned long long>(region.base),
      static_cast<unsigned long long>(impl_->mapping->address()));
  return Status::OK();
}

Status AccessOperator::Shutdown() {
  impl_->ready = false;
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

Status AccessOperator::ProbeLocalCopy(const ConstBufferView& input, const MutableBufferView& output,
                                     const ConstBufferView& metadata, void* caller_stream) {
  auto stream = static_cast<aclrtStream>(caller_stream);
  Impl::StreamState* state;
  auto status = impl_->State(stream, &state);
  if (!status.ok()) return status;
  // Test-only, read-only diagnosis: distinguish absent device permissions
  // from an access fault despite driver-reported RW permission. A query
  // unsupported by a deployed driver is logged, not a new transfer prerequisite.
  aclrtMemLocation location{};
  location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  location.id = impl_->options.device_id;
  std::uint64_t access = 0;
  const auto ret = aclrtMemGetAccess(reinterpret_cast<void*>(impl_->mapping->address()), &location, &access);
  if (ret == ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_UBMEM] MAPPING_ACCESS device=%u flags=0x%llx expected_rw=0x%x\n",
        impl_->options.device_id, static_cast<unsigned long long>(access), ACL_RT_MEM_ACCESS_FLAGS_READWRITE);
  } else {
    (void)Acl("aclrtMemGetAccess(diagnostic)", ret);
  }
  auto a = impl_->Args(Operation::kPut, const_cast<void*>(input.data), input.bytes, *state);
  // Keep the CPU base and absolute destinations: the control must exercise
  // the same 64-bit address subtraction as the mapped-DDR Put.
  a.remote_bytes = output.bytes;
  a.mapped_base = AddressOf(output.data);
  a.capacity = metadata.bytes / (3 * sizeof(std::uint64_t));
  a.put_offsets = AddressOf(metadata.data);
  a.put_lengths = a.put_offsets + a.capacity * sizeof(std::uint64_t);
  a.put_destinations = a.put_lengths + a.capacity * sizeof(std::uint64_t);
  return impl_->Launch(a, stream, *state, true);
}
}  // namespace kvaf::ubmem
