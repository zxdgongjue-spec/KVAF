#include "npu_python_runtime.h"
#include <unistd.h>

#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "access_table_python.h"
#include "access_operator.h"
#include "acl/acl.h"
#include "acl_python_utils.h"
#include "npu_runtime.h"

namespace py = pybind11;

namespace kvaf {
namespace {

void ThrowIfError(const Status& status) {
  if (status.ok()) return;
  const std::string message = status.message().empty()
                                  ? "KVAF AIV operation failed"
                                  : status.message();
  if (status.code() == StatusCode::kInvalidArgument ||
      status.code() == StatusCode::kInvalidState ||
      status.code() == StatusCode::kAlreadyExists ||
      status.code() == StatusCode::kBusy) {
    throw py::value_error(message);
  }
  if (status.code() == StatusCode::kNotFound) {
    throw py::key_error(message);
  }
  throw std::runtime_error(message);
}

void ThrowIfAclError(aclError error, const char* operation) {
  if (error == ACL_SUCCESS) return;
  throw std::runtime_error(std::string(operation) +
                           " failed with ACL error " +
                           std::to_string(static_cast<int>(error)));
}

void* PointerFromValue(std::uint64_t pointer) {
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer));
}

struct NpuTensorView {
  void* data{nullptr};
  std::size_t bytes{0};
  std::size_t element_bytes{0};
  std::vector<py::ssize_t> shape;
};

NpuTensorView GetNpuTensorView(const py::object& tensor, const char* name) {
  if (!py::hasattr(tensor, "data_ptr") || !py::hasattr(tensor, "numel") ||
      !py::hasattr(tensor, "element_size") ||
      !py::hasattr(tensor, "is_contiguous") ||
      !py::hasattr(tensor, "device") || !py::hasattr(tensor, "shape")) {
    throw py::type_error(std::string(name) + " must be a contiguous NPU Tensor");
  }
  if (!tensor.attr("is_contiguous")().cast<bool>()) {
    throw py::value_error(std::string(name) + " must be contiguous");
  }
  const std::string device = py::str(tensor.attr("device")).cast<std::string>();
  if (device.rfind("npu", 0) != 0) {
    throw py::value_error(std::string(name) + " must reside on an NPU");
  }
  const std::uint64_t pointer = tensor.attr("data_ptr")().cast<std::uint64_t>();
  const std::uint64_t elements = tensor.attr("numel")().cast<std::uint64_t>();
  const std::uint64_t element_bytes =
      tensor.attr("element_size")().cast<std::uint64_t>();
  if (pointer == 0 || elements == 0 || element_bytes == 0 ||
      elements > std::numeric_limits<std::size_t>::max() ||
      element_bytes > std::numeric_limits<std::size_t>::max() ||
      static_cast<std::size_t>(elements) >
          std::numeric_limits<std::size_t>::max() /
              static_cast<std::size_t>(element_bytes)) {
    throw py::value_error(std::string(name) + " must be a non-empty NPU Tensor");
  }
  return NpuTensorView{
      PointerFromValue(pointer),
      static_cast<std::size_t>(elements) * static_cast<std::size_t>(element_bytes),
      static_cast<std::size_t>(element_bytes),
      tensor.attr("shape").cast<std::vector<py::ssize_t>>()};
}

std::vector<KeyId> CopyPrepareKeys(const py::object& keys) {
  // A full AccessTable can contain millions of keys. Accepting a CPU int64
  // Tensor avoids Pybind iterating millions of Python int objects before the
  // Runtime control plane even starts. Runtime's C++ API still owns a vector
  // snapshot, so this performs one contiguous 8-byte copy per key.
  if (!py::hasattr(keys, "data_ptr")) {
    return keys.cast<std::vector<KeyId>>();
  }
  if (!py::hasattr(keys, "numel") || !py::hasattr(keys, "element_size") ||
      !py::hasattr(keys, "is_contiguous") || !py::hasattr(keys, "device") ||
      !py::hasattr(keys, "dim") || !py::hasattr(keys, "dtype")) {
    throw py::type_error("PrepareAccessTable keys must be a sequence or CPU int64 Tensor");
  }
  if (!keys.attr("is_contiguous")().cast<bool>() ||
      keys.attr("dim")().cast<int>() != 1 ||
      py::str(keys.attr("device")).cast<std::string>() != "cpu" ||
      py::str(keys.attr("dtype")).cast<std::string>() != "torch.int64" ||
      keys.attr("element_size")().cast<std::size_t>() != sizeof(KeyId)) {
    throw py::value_error(
        "PrepareAccessTable Tensor keys must be contiguous CPU torch.int64 with one dimension");
  }
  const std::uint64_t count = keys.attr("numel")().cast<std::uint64_t>();
  if (count == 0 || count > std::numeric_limits<std::size_t>::max()) {
    throw py::value_error("PrepareAccessTable Tensor keys must be non-empty");
  }
  const auto* data = static_cast<const KeyId*>(
      PointerFromValue(keys.attr("data_ptr")().cast<std::uint64_t>()));
  if (data == nullptr) {
    throw py::value_error("PrepareAccessTable Tensor keys must have storage");
  }
  return std::vector<KeyId>{data, data + static_cast<std::size_t>(count)};
}

class DeviceAccessTable final {
 public:
  DeviceAccessTable(AccessTableHandle source, py::object addresses,
                    py::object valid)
      : source_(source), addresses_(std::move(addresses)), valid_(std::move(valid)) {}

  const AccessTableHandle& source() const { return source_; }
  py::object addresses() const { return addresses_; }
  py::object valid() const { return valid_; }

 private:
  AccessTableHandle source_;
  py::object addresses_;
  py::object valid_;
};

aclrtStream ResolveStream(const py::object& stream) {
  if (!stream.is_none()) {
    return acl_python::StreamFromPython(stream);
  }
  py::module_ torch = py::module_::import("torch");
  if (!py::hasattr(torch, "npu")) {
    throw std::runtime_error("torch.npu is unavailable for the default KVAF stream");
  }
  return acl_python::StreamFromPython(torch.attr("npu").attr("current_stream")());
}

py::dict FusedContextDict(const FusedBatchGetContext& context) {
  py::dict result;
  result["lease_id"] = context.lease_id;
  result["device_channel_entity"] = context.device_channel_entity;
  result["device_queue_state"] = context.device_queue_state;
  result["device_core_status"] = context.device_core_status;
  result["output_address"] = context.output_address;
  result["output_bytes"] = context.output_bytes;
  result["worker_count"] = context.worker_count;
  result["jettys_per_core"] = context.jettys_per_core;
  result["token_bytes"] = context.token_bytes;
  result["tokens_per_block"] = context.tokens_per_block;
  return result;
}

class NpuPythonRuntime final {
 public:
  NpuPythonRuntime(std::string batch_get_kernel_object_path,
                   std::uint32_t batch_get_token_bytes,
                   std::uint32_t batch_get_tokens_per_block,
                   std::uint32_t device_id, std::uint32_t npu_core_count,
                   std::string batch_put_kernel_object_path,
                   std::uint32_t jettys_per_core, bool local_host_ddr,
                   std::string local_ddr_page_mode,
                   std::uint32_t batch_put_npu_core_count) {
    NpuRuntimeOptions options{std::move(batch_get_kernel_object_path),
                              batch_get_token_bytes,
                              batch_get_tokens_per_block};
    options.batch_put_kernel_object_path =
        std::move(batch_put_kernel_object_path);
    options.device_id = device_id;
    options.npu_core_count = npu_core_count;
    options.jettys_per_core = jettys_per_core;
    options.local_host_ddr = local_host_ddr;
    options.local_ddr_page_mode = std::move(local_ddr_page_mode);
    options.batch_put_npu_core_count = batch_put_npu_core_count;
    native_ = std::make_unique<::kvaf::NpuRuntime>(std::move(options));
  }

  ~NpuPythonRuntime() {
    if (fused_context_.lease_id != 0U) {
      // GC is not proof that caller-launched DMA finished. Keep both the
      // registered storage and its native owner until process exit. Native
      // teardown refusing a lease must not be followed by DECREF of output.
      std::fprintf(stderr,
          "[KVAF_AIV] FUSED_PYTHON_LEASE_RETAINED_ON_DESTRUCTION "
          "lease_id=%llu reason=explicit_release_required "
          "output_and_stream_retained=1 teardown_not_safe=1\n",
          static_cast<unsigned long long>(fused_context_.lease_id));
      (void)fused_output_.release();
      (void)fused_stream_.release();
      (void)native_.release();
    }
  }

  void LinkTo(std::string address) {
    ThrowIfError(native_->LinkTo(LinkTarget{std::move(address)}));
  }

  void Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
    ThrowIfError(native_->Initialize(ddr_bytes, slot_size));
  }

  bool Initialized() const { return native_->initialized(); }
  std::uint32_t DeviceId() const { return native_->device_id(); }
  std::uint32_t NpuCoreCount() const { return native_->npu_core_count(); }
  std::uint32_t BatchPutNpuCoreCount() const { return native_->batch_put_npu_core_count(); }
  std::uint32_t JettysPerCore() const { return native_->jettys_per_core(); }
  bool LocalHostDdr() const { return native_->local_host_ddr(); }
  std::uint32_t BatchGetTokenBytes() const {
    return native_->batch_get_token_bytes();
  }
  std::uint32_t BatchGetTokensPerBlock() const {
    return native_->batch_get_tokens_per_block();
  }

  py::dict AcquireFusedBatchGetContext(const py::object& output,
                                      const py::object& stream) {
    if (fused_context_.lease_id != 0U) {
      throw py::value_error("Runtime already owns a fused BatchGet lease");
    }
    if (!Initialized()) {
      throw std::runtime_error("Runtime has not been initialized");
    }
    const NpuTensorView view = GetNpuTensorView(output, "fused output");
    const std::string device = "npu:" + std::to_string(DeviceId());
    if (py::str(output.attr("device")).cast<std::string>() != device ||
        view.shape.size() < 2 || view.shape.front() <= 0 ||
        view.bytes % static_cast<std::size_t>(view.shape.front()) != 0 ||
        view.bytes / static_cast<std::size_t>(view.shape.front()) !=
            BatchGetTokenBytes()) {
      throw py::value_error(
          "fused output must be on the Runtime device with one token_bytes-sized row per slot");
    }
    py::object launch_stream = stream;
    if (launch_stream.is_none()) {
      py::module_ torch = py::module_::import("torch");
      if (!py::hasattr(torch, "npu")) {
        throw std::runtime_error("torch.npu is unavailable for the default KVAF stream");
      }
      launch_stream = torch.attr("npu").attr("current_stream")(DeviceId());
    }
    if (!py::hasattr(launch_stream, "device") ||
        py::str(launch_stream.attr("device")).cast<std::string>() != device) {
      throw py::value_error("fused stream must belong to the Runtime device");
    }
    const aclrtStream acl_stream = acl_python::StreamFromPython(launch_stream);
    ThrowIfAclError(aclrtSetDevice(static_cast<int32_t>(DeviceId())),
                    "aclrtSetDevice before fused acquire");
    FusedBatchGetContext context;
    ThrowIfError(native_->AcquireFusedBatchGetContext(view.data, view.bytes,
                                                      &context));
    // Track ownership before allocating the dict: even an allocation failure
    // must leave a retained native lease rather than unowned registered HBM.
    fused_output_ = output;
    fused_stream_ = std::move(launch_stream);
    fused_acl_stream_ = acl_stream;
    fused_context_ = context;
    return FusedContextDict(fused_context_);
  }

  void ReleaseFusedBatchGetContext(const py::dict& context,
                                   const py::object& stream) {
    if (fused_context_.lease_id == 0U) {
      throw py::value_error("Runtime has no active fused BatchGet lease");
    }
    // Never forward user-modified pointers or an old lease to native code.
    const py::dict expected = FusedContextDict(fused_context_);
    for (const auto& item : expected) {
      if (!context.contains(item.first) ||
          context[item.first].cast<std::uint64_t>() !=
              item.second.cast<std::uint64_t>()) {
        throw py::value_error("fused context does not match this Runtime's active lease");
      }
    }
    if (!stream.is_none() &&
        acl_python::StreamFromPython(stream) != fused_acl_stream_) {
      throw py::value_error("fused release must use the stream selected at acquire");
    }
    ThrowIfAclError(aclrtSetDevice(static_cast<int32_t>(DeviceId())),
                    "aclrtSetDevice before fused release");
    // Native release fences this exact stream, then checks queue statuses.
    // Failed completion quarantines the native lease. Keep Python references
    // and surface the error, without retrying teardown or freeing the buffer.
    ThrowIfError(native_->ReleaseFusedBatchGetContext(fused_context_,
                                                      fused_acl_stream_));
    fused_context_ = FusedBatchGetContext{};
    fused_acl_stream_ = nullptr;
    fused_output_ = py::object{};
    fused_stream_ = py::object{};
  }

  py::dict RemoteRegion() const {
    const RemoteRegionDesc& region = native_->core().remote_region();
    py::dict result;
    result["base"] = region.base;
    result["length"] = region.length;
    result["slot_size"] = region.slot_size;
    result["device"] = "cpu";
    if (native_->local_host_ddr()) {
      result["local_host_ddr"] = true;
      const LocalHostDdrStats stats = native_->local_ddr_page_stats();
      result["owner_pid"] = static_cast<int>(getpid());
      result["remote_service"] = false;
      result["ddr_page_mode"] = native_->local_ddr_page_mode();
      result["ddr_anon_huge_bytes"] = stats.anon_huge_bytes;
      result["ddr_smaps_covered_bytes"] = stats.smaps_covered_bytes;
    }
    return result;
  }

  // The caller owns one contiguous NPU input tensor; CPU offset/length
  // metadata describes its variable-size blocks for the duration of
  // Runtime::BatchPut. When batch_put_kernel_object_path was configured,
  // the Direct-HCCP implementation synchronously completes this stream.
  void BatchPut(const std::vector<KeyId>& keys, const py::object& source,
                const std::vector<std::uint64_t>& input_offsets,
                const std::vector<std::uint64_t>& lengths,
                const py::object& stream) {
    if (!Initialized()) {
      throw std::runtime_error("Runtime has not been initialized");
    }
    if (keys.empty() || keys.size() != input_offsets.size() ||
        keys.size() != lengths.size()) {
      throw py::value_error(
          "BatchPut keys, input_offsets, and lengths must be non-empty and equal in size");
    }
    const NpuTensorView source_view = GetNpuTensorView(source, "BatchPut source");
    for (std::size_t index = 0; index < keys.size(); ++index) {
      const std::uint64_t offset = input_offsets[index];
      const std::uint64_t length = lengths[index];
      if (length == 0 || offset > source_view.bytes ||
          length > source_view.bytes - offset) {
        throw py::value_error(
            "BatchPut input_offsets and lengths must describe bytes inside source");
      }
    }
    const aclrtStream acl_stream = ResolveStream(stream);
    ThrowIfError(native_->core().BatchPut(BatchPutRequest{
        keys,
        ConstBufferView{source_view.data, source_view.bytes, DeviceKind::kNpu},
        input_offsets,
        lengths}, BatchPutExecutionOptions{acl_stream}));
  }

  // Untimed initialization of owned Host DDR. The CPU buffer stays borrowed
  // and alive through synchronous metadata allocation, copy and commit.
  void BatchPutHost(const std::vector<KeyId>& keys, const py::buffer& source,
                    const std::vector<std::uint64_t>& input_offsets,
                    const std::vector<std::uint64_t>& lengths) {
    if (!Initialized()) throw std::runtime_error("Runtime has not been initialized");
    if (!native_->local_host_ddr()) throw py::value_error("batch_put_host requires local_host_ddr=True");
    if (fused_context_.lease_id != 0U) throw py::value_error("cannot seed Host DDR during a fused lease");
    if (keys.empty() || keys.size() != input_offsets.size() || keys.size() != lengths.size()) {
      throw py::value_error("batch_put_host requires equally sized nonempty keys, offsets and lengths");
    }
    const py::buffer_info view = source.request();
    if (view.itemsize != 1 || view.format != py::format_descriptor<std::uint8_t>::format() ||
        view.ndim < 1 || view.size <= 0 || view.ptr == nullptr) {
      throw py::value_error("batch_put_host source must be a nonempty unsigned uint8 CPU buffer");
    }
    py::ssize_t stride = 1;
    for (py::ssize_t dimension = view.ndim; dimension-- > 0;) {
      if (view.shape[dimension] <= 0 ||
          (view.shape[dimension] > 1 && view.strides[dimension] != stride) ||
          view.shape[dimension] > std::numeric_limits<py::ssize_t>::max() / stride) {
        throw py::value_error("batch_put_host source must be C-contiguous");
      }
      stride *= view.shape[dimension];
    }
    const auto bytes = static_cast<std::size_t>(view.size);
    for (std::size_t index = 0; index < keys.size(); ++index) {
      if (lengths[index] == 0U || input_offsets[index] > bytes || lengths[index] > bytes - input_offsets[index]) {
        throw py::value_error("batch_put_host source offsets/lengths escape the CPU buffer");
      }
    }
    ThrowIfError(native_->core().BatchPut(
        BatchPutRequest{keys, ConstBufferView{view.ptr, bytes, DeviceKind::kCpu}, input_offsets, lengths},
        BatchPutExecutionOptions{}));
  }

  AccessTableHandle PrepareAccessTable(const py::object& keys) {
    std::vector<KeyId> copied_keys = CopyPrepareKeys(keys);
    AccessTableHandle handle;
    ThrowIfError(native_->core().PrepareAccessTable(
        AccessTablePrepareRequest{std::move(copied_keys)}, &handle));
    return handle;
  }

  py::object AccessTableView(const AccessTableHandle& handle,
                             const std::string& device,
                             std::uint64_t capacity) const {
    kvaf::AccessTableView view;
    ThrowIfError(native_->core().GetAccessTableView(handle, &view));
    if (device == "cpu") {
      return python_binding::MakeAccessTableView(view);
    }
    if (device != "npu") {
      throw py::value_error("AccessTable device must be 'cpu' or 'npu'");
    }
    if (capacity == 0) {
      capacity = static_cast<std::uint64_t>(view.size);
    }
    if (capacity < view.size ||
        capacity > std::numeric_limits<std::uint32_t>::max()) {
      throw py::value_error(
          "NPU AccessTable capacity must cover all entries and fit the AIV ABI");
    }

    py::module_ torch = py::module_::import("torch");
    const std::string npu_device = "npu:" + std::to_string(native_->device_id());
    py::object addresses = torch.attr("empty")(
        py::make_tuple(capacity), py::arg("dtype") = torch.attr("int64"),
        py::arg("device") = npu_device);
    py::object valid = torch.attr("empty")(
        py::make_tuple(capacity), py::arg("dtype") = torch.attr("uint8"),
        py::arg("device") = npu_device);
    const NpuTensorView address_view = GetNpuTensorView(addresses, "addresses");
    const NpuTensorView valid_view = GetNpuTensorView(valid, "valid");
    if (address_view.element_bytes != sizeof(Address) ||
        valid_view.element_bytes != sizeof(std::uint8_t) ||
        address_view.bytes < capacity * sizeof(Address) ||
        valid_view.bytes < capacity) {
      throw std::runtime_error("torch created an invalid NPU AccessTable Tensor");
    }
    ThrowIfAclError(aclrtMemset(valid_view.data, valid_view.bytes, 0,
                                 valid_view.bytes),
                    "aclrtMemset NPU AccessTable valid");
    ThrowIfAclError(
        aclrtMemcpy(address_view.data, address_view.bytes,
                    const_cast<Address*>(view.addresses),
                    view.size * sizeof(Address), ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy NPU AccessTable addresses");
    ThrowIfAclError(
        aclrtMemcpy(valid_view.data, valid_view.bytes,
                    const_cast<std::uint8_t*>(view.valid), view.size,
                    ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy NPU AccessTable valid");
    return py::cast(DeviceAccessTable{handle, std::move(addresses),
                                      std::move(valid)});
  }

  void ReleaseAccessTable(const AccessTableHandle& handle) {
    ThrowIfError(native_->core().ReleaseAccessTable(handle));
  }

  // BatchFree is a control-plane operation.  It deliberately takes the same
  // Python key forms as PrepareAccessTable (a Python key sequence or a
  // contiguous one-dimensional CPU torch.int64 Tensor), rather than imposing
  // a second representation solely for the NPU Runtime binding.
  void BatchFree(const py::object& keys) {
    std::vector<KeyId> copied_keys = CopyPrepareKeys(keys);
    ThrowIfError(native_->core().BatchFree(BatchFreeRequest{std::move(copied_keys)}));
  }

  void StartBatchGetKernelProfile() {
    ThrowIfError(native_->SetBatchGetKernelProfiling(true));
  }

  py::list BatchGetCqDiagnostics() const {
    std::vector<kvaf::BatchGetCqDiagnostic> diagnostics;
    ThrowIfError(native_->CopyBatchGetCqDiagnostics(&diagnostics));
    py::list result;
    for (const auto& source : diagnostics) {
      py::dict item;
      item["core"] = source.core;
      item["jetty"] = source.jetty;
      item["last_error"] = source.last_error;
      item["sq_producer_index"] = source.sq_producer_index;
      item["cq_consumer_index"] = source.cq_consumer_index;
      item["completed_batch_count"] = source.completed_batch_count;
      item["trace_valid"] = source.trace_valid;
      if (source.trace_valid) {
        item["raw_cq_word"] = source.raw_cq_word;
        item["observed_owner"] = (source.raw_cq_word >> 2U) & 1U;
        item["expected_owner"] = source.expected_owner;
        item["cq_status"] = (source.raw_cq_word >> 24U) & 0xffU;
        item["cq_sub_status"] = (source.raw_cq_word >> 16U) & 0xffU;
        item["slow_elapsed_ticks"] = source.slow_elapsed_ticks;
        item["polls"] = source.polls;
        item["flags"] = source.flags;
        item["fast_budget_exceeded"] = source.fast_budget_exceeded;
        item["late_success"] = source.late_success;
        item["bypass_probe"] = source.bypass_probe;
        item["reason"] = source.reason;
        item["fast_poll_budget"] = 20000;
        item["slow_deadline_ticks"] = 100000000;
        item["clock_hz"] = 1000000000;
        item["trace_scope"] = "most_recent_slow_or_error_wait";
      }
      result.append(std::move(item));
    }
    return result;
  }

  py::list BatchGetKernelProfileSnapshot() const {
    std::vector<kvaf::BatchGetKernelProfile> profiles;
    ThrowIfError(native_->CopyBatchGetKernelProfiles(&profiles));
    py::list result;
    for (std::size_t core = 0; core < profiles.size(); ++core) {
      const kvaf::BatchGetKernelProfile& profile = profiles[core];
      py::dict item;
      item["core"] = core;
      item["kernel_start_cycle"] = profile.kernel_start_cycle;
      item["setup_done_cycle"] = profile.setup_done_cycle;
      item["compact_done_cycle"] = profile.compact_done_cycle;
      item["first_doorbell_cycle"] = profile.first_doorbell_cycle;
      item["post_done_cycle"] = profile.post_done_cycle;
      item["completion_done_cycle"] = profile.completion_done_cycle;
      item["kernel_finish_cycle"] = profile.kernel_finish_cycle;
      item["active_requests"] = profile.active_requests;
      item["total_posts"] = profile.total_posts;
      item["jetty_completion_wait_cycles"] =
          profile.jetty_completion_wait_cycles;
      item["jetty_completion_polls"] = profile.jetty_completion_polls;
      result.append(std::move(item));
    }
    return result;
  }

  py::list StopBatchGetKernelProfile() {
    py::list result = BatchGetKernelProfileSnapshot();
    ThrowIfError(native_->SetBatchGetKernelProfiling(false));
    return result;
  }

  // Tensor ownership remains with Python. KVAF creates no output Tensor and
  // only retains its internal Direct-HCCP execution state; callers must keep
  // input/output Tensors and either the DeviceAccessTable or the direct
  // addresses/valid Tensors alive until their stream completes BatchGet.
  void BatchGet(const DeviceAccessTable& access_table,
                const py::object& indices, const py::object& offsets,
                const py::object& miss_count,
                const py::object& output_indices, const py::object& output,
                const py::object& stream, bool synchronize) {
    BatchGetImpl(access_table.source(), access_table.addresses(),
                 access_table.valid(), indices, offsets, miss_count,
                 output_indices, output, stream, synchronize);
  }

  // This overload accepts a caller-owned device table directly. Since bare
  // Tensors carry no AccessTableHandle, Runtime validates their NPU layout but
  // cannot validate a corresponding Host AccessTable lifetime or entry length.
  void BatchGetFromTensors(const py::object& addresses,
                           const py::object& valid,
                           const py::object& indices,
                           const py::object& offsets,
                           const py::object& miss_count,
                           const py::object& output_indices,
                           const py::object& output,
                           const py::object& stream, bool synchronize) {
    BatchGetImpl(AccessTableHandle{}, addresses, valid, indices, offsets,
                 miss_count, output_indices, output, stream, synchronize);
  }

 private:
  void BatchGetImpl(const AccessTableHandle& access_table,
                    const py::object& addresses, const py::object& valid,
                    const py::object& indices, const py::object& offsets,
                    const py::object& miss_count,
                    const py::object& output_indices,
                    const py::object& output, const py::object& stream,
                    bool synchronize) {
    const NpuTensorView index_view = GetNpuTensorView(indices, "indices");
    const NpuTensorView offset_view = GetNpuTensorView(offsets, "offsets");
    const NpuTensorView miss_count_view =
        GetNpuTensorView(miss_count, "miss_count");
    const NpuTensorView output_index_view =
        GetNpuTensorView(output_indices, "output_indices");
    const NpuTensorView output_view = GetNpuTensorView(output, "output");
    const NpuTensorView address_view = GetNpuTensorView(addresses, "addresses");
    const NpuTensorView valid_view = GetNpuTensorView(valid, "valid");
    if (index_view.shape.size() != 1 || offset_view.shape.size() != 1 ||
        output_index_view.shape.size() != 1 || output_view.shape.size() < 2 ||
        index_view.element_bytes != sizeof(std::int32_t) ||
        offset_view.element_bytes != sizeof(std::int32_t) ||
        output_index_view.element_bytes != sizeof(std::int32_t) ||
        miss_count_view.element_bytes != sizeof(std::int32_t) ||
        miss_count_view.bytes != sizeof(std::int32_t) ||
        index_view.bytes != offset_view.bytes ||
        index_view.bytes != output_index_view.bytes ||
        address_view.element_bytes != sizeof(Address) ||
        valid_view.element_bytes != sizeof(std::uint8_t) ||
        valid_view.bytes != address_view.bytes / sizeof(Address) ||
        output_view.shape.front() <= 0 ||
        output_view.bytes % static_cast<std::size_t>(output_view.shape.front()) != 0) {
      throw py::value_error("BatchGet Tensors do not satisfy the NPU layout contract");
    }
    const std::uint64_t output_rows =
        static_cast<std::uint64_t>(output_view.shape.front());
    const std::uint64_t output_block_bytes = output_view.bytes / output_rows;
    if (output_block_bytes != native_->batch_get_token_bytes()) {
      throw py::value_error(
          "BatchGet output Tensor row size does not match batch_get_token_bytes");
    }
    const BatchGetRequest request{
        access_table,
        ConstBufferView{index_view.data, index_view.bytes, DeviceKind::kNpu},
        ConstBufferView{offset_view.data, offset_view.bytes, DeviceKind::kNpu},
        ConstBufferView{miss_count_view.data, miss_count_view.bytes,
                        DeviceKind::kNpu},
        ConstBufferView{output_index_view.data, output_index_view.bytes,
                        DeviceKind::kNpu},
        MutableBufferView{output_view.data, output_view.bytes, DeviceKind::kNpu},
        output_block_bytes,
        native_->batch_get_tokens_per_block(),
        AccessTableDeviceView{
            ConstBufferView{address_view.data, address_view.bytes, DeviceKind::kNpu},
            ConstBufferView{valid_view.data, valid_view.bytes, DeviceKind::kNpu}}};
    const aclrtStream acl_stream = ResolveStream(stream);
    ThrowIfError(native_->core().BatchGet(
        request, BatchGetExecutionOptions{acl_stream,
                                           synchronize}));
  }

 public:
  void Shutdown() {
    if (fused_context_.lease_id != 0U) {
      throw py::value_error(
          "release the fused BatchGet context successfully before shutdown; "
          "uncertain DMA completion requires retaining output and Runtime until process teardown");
    }
    ThrowIfError(native_->Shutdown());
  }

 private:
  std::unique_ptr<::kvaf::NpuRuntime> native_;
  FusedBatchGetContext fused_context_;
  py::object fused_output_;
  py::object fused_stream_;
  aclrtStream fused_acl_stream_{nullptr};
};

}  // namespace

void BindNpuPythonRuntime(py::module_& module) {
  py::class_<DeviceAccessTable>(module, "DeviceAccessTable")
      .def_property_readonly("addresses", &DeviceAccessTable::addresses)
      .def_property_readonly("valid", &DeviceAccessTable::valid)
      .def_property_readonly("source_id", [](const DeviceAccessTable& table) {
        return table.source().opaque;
      });

  py::class_<NpuPythonRuntime>(module, "NpuRuntime")
      .def(py::init<std::string, std::uint32_t, std::uint32_t, std::uint32_t,
                    std::uint32_t, std::string, std::uint32_t, bool, std::string,
                    std::uint32_t>(),
           py::arg("batch_get_kernel_object_path"),
           py::arg("batch_get_token_bytes"),
           py::arg("batch_get_tokens_per_block"), py::arg("device_id") = 0,
           py::arg("npu_core_count") = 1,
           py::arg("batch_put_kernel_object_path") = "",
           py::arg("jettys_per_core") = 2, py::arg("local_host_ddr") = false,
           py::arg("local_ddr_page_mode") = "thp",
           py::arg("batch_put_npu_core_count") = 0)
      .def("link_to", &NpuPythonRuntime::LinkTo, py::arg("address"),
           "Connect this NPU Runtime to a CPU Runtime service before initialize.")
      .def("initialize", &NpuPythonRuntime::Initialize, py::arg("ddr_bytes"),
           py::arg("slot_size"))
      .def("shutdown", &NpuPythonRuntime::Shutdown)
      .def("acquire_fused_batch_get_context",
           &NpuPythonRuntime::AcquireFusedBatchGetContext,
           py::arg("output"), py::arg("stream") = py::none(),
           "Exclusively lease native BatchGet queues and register output HBM. "
           "Returns integer fields matching FusedBatchGetContext, including lease_id. "
           "output and stream are retained until successful release. All fused launches "
           "must run sequentially on this stream; do not resize/rebind output storage, "
           "reset queue cursors, or free source DDR while the lease is in use. "
           "stream=None captures the Runtime device's current torch.npu stream.")
      .def("release_fused_batch_get_context",
           &NpuPythonRuntime::ReleaseFusedBatchGetContext,
           py::arg("context"), py::arg("stream") = py::none(),
           "Fence the acquire stream and check native queue completion before release. "
           "Pass the unmodified acquire dict. stream=None uses the retained acquire "
           "stream, not the current stream. Failure retains all resources and blocks "
           "shutdown; unknown DMA completion requires process/device teardown.")
      .def_property_readonly("initialized", &NpuPythonRuntime::Initialized)
      .def_property_readonly("device_id", &NpuPythonRuntime::DeviceId)
      .def_property_readonly("npu_core_count", &NpuPythonRuntime::NpuCoreCount)
      .def_property_readonly("batch_put_npu_core_count", &NpuPythonRuntime::BatchPutNpuCoreCount)
      .def_property_readonly("jettys_per_core", &NpuPythonRuntime::JettysPerCore)
      .def_property_readonly("local_host_ddr", &NpuPythonRuntime::LocalHostDdr)
      .def_property_readonly("batch_get_token_bytes",
                             &NpuPythonRuntime::BatchGetTokenBytes)
      .def_property_readonly("batch_get_tokens_per_block",
                             &NpuPythonRuntime::BatchGetTokensPerBlock)
      .def_property_readonly("remote_region", &NpuPythonRuntime::RemoteRegion)
      .def("batch_put", &NpuPythonRuntime::BatchPut, py::arg("keys"),
           py::arg("source"), py::arg("input_offsets"), py::arg("lengths"),
           py::arg("stream") = py::none(),
           "Synchronously write variable-size packed NPU blocks. input_offsets and "
           "lengths are CPU integer sequences with one entry per key; "
           "stream=None uses torch.npu.current_stream().")
      .def("batch_put_host", &NpuPythonRuntime::BatchPutHost, py::arg("keys"),
           py::arg("source"), py::arg("input_offsets"), py::arg("lengths"),
           "Synchronously seed owned local Host DDR from a C-contiguous unsigned uint8 CPU buffer. "
           "Uses Runtime metadata allocation/commit and no NPU Put kernel; call outside performance measurement.")
      .def("prepare_access_table", &NpuPythonRuntime::PrepareAccessTable,
           py::arg("keys"))
      .def("access_table_view", &NpuPythonRuntime::AccessTableView,
           py::arg("access_table"), py::arg("device") = "cpu",
           py::arg("capacity") = 0)
      .def("release_access_table", &NpuPythonRuntime::ReleaseAccessTable,
           py::arg("access_table"))
      .def("batch_free", &NpuPythonRuntime::BatchFree, py::arg("keys"),
           "Release the supplied keys from the Runtime metadata store. keys "
           "accepts a Python key sequence or a contiguous one-dimensional "
           "CPU torch.int64 Tensor.")
      .def("start_batch_get_kernel_profile",
           &NpuPythonRuntime::StartBatchGetKernelProfile,
           "Enable the per-core BatchGet kernel cycle timeline.")
      .def("batch_get_cq_diagnostics", &NpuPythonRuntime::BatchGetCqDiagnostics,
           "After caller stream completion, copy persistent slow/error CQ traces. "
           "Healthy queues without a trace are omitted; this never clears queue poison.")
      .def("batch_get_kernel_profile",
           &NpuPythonRuntime::BatchGetKernelProfileSnapshot,
           "Copy the latest completed per-core BatchGet kernel timeline.")
      .def("stop_batch_get_kernel_profile",
           &NpuPythonRuntime::StopBatchGetKernelProfile,
           "Copy the latest timeline and disable BatchGet kernel profiling.")
      .def("batch_get", &NpuPythonRuntime::BatchGet,
           py::arg("device_access_table"), py::arg("indices"), py::arg("offsets"),
           py::arg("miss_count"),
           py::arg("output_indices"), py::arg("output"),
           py::arg("stream") = py::none(), py::arg("synchronize") = false,
           "Submit BatchGet from NPU Tensors. stream=None uses torch.npu.current_stream(); "
           "the default returns after enqueueing work on that stream.")
      .def("batch_get", &NpuPythonRuntime::BatchGetFromTensors,
           py::arg("addresses"), py::arg("valid"), py::arg("indices"),
           py::arg("offsets"), py::arg("miss_count"), py::arg("output_indices"),
           py::arg("output"), py::arg("stream") = py::none(),
           py::arg("synchronize") = false,
           "Submit BatchGet from caller-owned NPU addresses and valid Tensors. "
           "stream=None uses torch.npu.current_stream(); the default returns "
           "after enqueueing work on that stream.");
}

}  // namespace kvaf
