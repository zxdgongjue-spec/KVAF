#include "npu_python_runtime.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "access_table_python.h"
#include "access_operator.h"
#include "acl/acl.h"
#include "acl_python_utils.h"
#include "npu_runtime.h"
#include "scatter_host_profile.h"

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
  std::size_t dimensions{0};
  py::ssize_t first_dimension{0};
};

NpuTensorView GetNpuTensorView(const py::object& tensor, const char* name) {
  // Read metadata once. A hasattr preflight repeats six Python attribute
  // lookups per Tensor; it neither validates their values nor freezes storage.
  try {
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
    const py::sequence shape = tensor.attr("shape");
    const auto dimensions = shape.size();
    return NpuTensorView{
        PointerFromValue(pointer),
        static_cast<std::size_t>(elements) * static_cast<std::size_t>(element_bytes),
        static_cast<std::size_t>(element_bytes),
        dimensions, dimensions ? shape[0].cast<py::ssize_t>() : 0};
  } catch (const py::error_already_set& error) {
    if (error.matches(PyExc_AttributeError))
      throw py::type_error(std::string(name) + " must be a contiguous NPU Tensor");
    throw;
  }
}

// Reuse method-name objects and use CPython's method vectorcall, avoiding an
// attribute accessor/bound-method/argument-tuple on every trusted tensor read.
// Cache names and the stream-query callable; pointers, sizes and stream handles
// are always read for this call.
struct TensorMetadataMethods {
  py::str data_ptr{"data_ptr"}, numel{"numel"}, size{"size"};
  py::str stream_handle{"npu_stream"};
  py::int_ zero{0};
  py::object current_stream;
  py::object Call(const py::object& tensor, const py::str& name) const {
#if PY_VERSION_HEX >= 0x03090000
    PyObject* result = PyObject_CallMethodNoArgs(tensor.ptr(), name.ptr());
    if (!result) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(result);
#else
    return tensor.attr(name)();
#endif
  }
  py::object Rows(const py::object& tensor) const {
#if PY_VERSION_HEX >= 0x03090000
    PyObject* result = PyObject_CallMethodOneArg(tensor.ptr(), size.ptr(), zero.ptr());
    if (!result) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(result);
#else
    return tensor.attr(size)(zero);
#endif
  }
  void* Pointer(const py::object& tensor) const {
    return PointerFromValue(Call(tensor, data_ptr).cast<std::uint64_t>());
  }
  aclrtStream Stream(const py::object& requested) {
    py::object selected = requested;
    if (selected.is_none()) {
      // Cache the Python callable, never the stream or handle. Resolve the
      // caller's current device/stream on every call, including stream switches.
      if (!current_stream)
        current_stream = py::module_::import("torch").attr("npu").attr("current_stream");
      selected = current_stream();
    }
    return reinterpret_cast<aclrtStream>(PointerFromValue(
        selected.attr(stream_handle).cast<std::uint64_t>()));
  }
};

BatchGetRequest TrustedScatterRequest(
    const AccessTableHandle& table, const py::object& addresses,
    const py::object& valid, const py::object& indices, const py::object& offsets,
    const py::object& miss_count, const py::object& output_indices,
    const py::object& output, std::uint32_t token_bytes, std::uint32_t tokens_per_block,
    const TensorMetadataMethods& methods) {
  // Tensor layout/lifetime is the caller's contract. Read only metadata needed
  // to launch: seven current pointers, two capacities and the output row count.
  // No dtype/device/contiguity/shape checks or cross-call pointer cache.
  const auto request_bytes = methods.Call(indices, methods.numel).cast<std::size_t>() * sizeof(std::int32_t);
  const auto entries = methods.Call(addresses, methods.numel).cast<std::size_t>();
  const auto output_rows = methods.Rows(output).cast<std::size_t>();
  return BatchGetRequest{
      table,
      {methods.Pointer(indices), request_bytes, DeviceKind::kNpu},
      {methods.Pointer(offsets), request_bytes, DeviceKind::kNpu},
      {methods.Pointer(miss_count), sizeof(std::int32_t), DeviceKind::kNpu},
      {methods.Pointer(output_indices), request_bytes, DeviceKind::kNpu},
      {methods.Pointer(output), output_rows * token_bytes, DeviceKind::kNpu},
      token_bytes, tokens_per_block,
      {{methods.Pointer(addresses), entries * sizeof(Address), DeviceKind::kNpu},
       {methods.Pointer(valid), entries, DeviceKind::kNpu}}};
}

void CopyPrepareKeysInto(const py::object& keys, std::vector<KeyId>* result) {
  if (result == nullptr) {
    throw std::invalid_argument("key output must not be null");
  }
  // A full AccessTable can contain millions of keys. Accepting a CPU int64
  // Tensor avoids Pybind iterating millions of Python int objects before the
  // Runtime control plane even starts. Runtime's C++ API still owns a vector
  // snapshot, so this performs one contiguous 8-byte copy per key.
  if (!py::hasattr(keys, "data_ptr")) {
    *result = keys.cast<std::vector<KeyId>>();
    return;
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
  const std::size_t size = static_cast<std::size_t>(count);
  result->resize(size);
  std::copy(data, data + size, result->begin());
}

std::vector<KeyId> CopyPrepareKeys(const py::object& keys) {
  std::vector<KeyId> result;
  CopyPrepareKeysInto(keys, &result);
  return result;
}

void CopyCpuUint64SequenceInto(const py::object& values, const char* name,
                               std::vector<std::uint64_t>* result) {
  if (result == nullptr) {
    throw std::invalid_argument("sequence output must not be null");
  }
  if (!py::hasattr(values, "data_ptr")) {
    *result = values.cast<std::vector<std::uint64_t>>();
    return;
  }
  if (!py::hasattr(values, "numel") ||
      !py::hasattr(values, "element_size") ||
      !py::hasattr(values, "is_contiguous") ||
      !py::hasattr(values, "device") ||
      !py::hasattr(values, "dim") || !py::hasattr(values, "dtype")) {
    throw py::type_error(std::string(name) +
                         " must be a sequence or CPU int64 Tensor");
  }
  if (!values.attr("is_contiguous")().cast<bool>() ||
      values.attr("dim")().cast<int>() != 1 ||
      py::str(values.attr("device")).cast<std::string>() != "cpu" ||
      py::str(values.attr("dtype")).cast<std::string>() != "torch.int64" ||
      values.attr("element_size")().cast<std::size_t>() != sizeof(std::int64_t)) {
    throw py::value_error(std::string(name) +
                          " Tensor must be contiguous CPU torch.int64 with one dimension");
  }
  const std::uint64_t count = values.attr("numel")().cast<std::uint64_t>();
  if (count == 0 || count > std::numeric_limits<std::size_t>::max()) {
    throw py::value_error(std::string(name) + " must be non-empty");
  }
  const auto* data = static_cast<const std::int64_t*>(
      PointerFromValue(values.attr("data_ptr")().cast<std::uint64_t>()));
  if (data == nullptr) {
    throw py::value_error(std::string(name) + " Tensor must have storage");
  }
  const std::size_t size = static_cast<std::size_t>(count);
  result->resize(size);
  for (std::size_t index = 0; index < size; ++index) {
    if (data[index] < 0) {
      throw py::value_error(std::string(name) +
                            " must not contain negative values");
    }
    (*result)[index] = static_cast<std::uint64_t>(data[index]);
  }
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

class NpuPythonRuntime final {
 public:
  NpuPythonRuntime(std::string batch_get_kernel_object_path,
                   std::uint32_t batch_get_token_bytes,
                   std::uint32_t batch_get_tokens_per_block,
                   std::uint32_t device_id, std::uint32_t npu_core_count,
                   std::string batch_put_kernel_object_path,
                   std::uint32_t jettys_per_core,
                   std::vector<std::string> npu_eids,
                   bool adaptive_ub_device,
                   std::uint32_t adaptive_ub_miss_count,
                   std::string peer_ub_device,
                   std::vector<std::string> adaptive_peer_ub_devices,
                   std::uint32_t batch_get_max_inflight_total,
                   std::string aiv_batch_get_mode, std::uint32_t aiv_gather_sgl_size) {
    NpuRuntimeOptions options{std::move(batch_get_kernel_object_path),
                              batch_get_token_bytes,
                              batch_get_tokens_per_block};
    options.batch_put_kernel_object_path =
        std::move(batch_put_kernel_object_path);
    options.device_id = device_id;
    options.npu_core_count = npu_core_count;
    options.jettys_per_core = jettys_per_core;
    options.batch_get_max_inflight_total = batch_get_max_inflight_total;
    trusted_scatter_inputs_ = aiv_batch_get_mode == "cpu_push_scatter";
    options.aiv_batch_get_mode = std::move(aiv_batch_get_mode);
    options.aiv_gather_sgl_size = aiv_gather_sgl_size;
    options.npu_eids = std::move(npu_eids);
    options.adaptive_ub_device = adaptive_ub_device;
    options.adaptive_ub_miss_count = adaptive_ub_miss_count;
    options.peer_ub_device = std::move(peer_ub_device);
    options.adaptive_peer_ub_devices = std::move(adaptive_peer_ub_devices);
    native_ = std::make_unique<::kvaf::NpuRuntime>(std::move(options));
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
  std::uint32_t JettysPerCore() const { return native_->jettys_per_core(); }
  const std::vector<std::string>& NpuEids() const { return native_->npu_eids(); }
  std::uint32_t BatchGetTokenBytes() const {
    return native_->batch_get_token_bytes();
  }
  std::uint32_t BatchGetTokensPerBlock() const {
    return native_->batch_get_tokens_per_block();
  }

  py::dict RemoteRegion() const {
    const RemoteRegionDesc& region = native_->core().remote_region();
    py::dict result;
    result["base"] = region.base;
    result["length"] = region.length;
    result["slot_size"] = region.slot_size;
    result["device"] = "cpu";
    return result;
  }

  // The caller owns one contiguous NPU input tensor; CPU offset/length
  // metadata describes its variable-size blocks for the duration of
  // Runtime::BatchPut. When batch_put_kernel_object_path was configured,
  // the Direct-HCCP implementation synchronously completes this stream.
  void BatchPut(const py::object& keys, const py::object& source,
                const py::object& input_offsets, const py::object& lengths,
                const py::object& stream) {
    if (!Initialized()) {
      throw std::runtime_error("Runtime has not been initialized");
    }
    CopyPrepareKeysInto(keys, &batch_put_request_.keys);
    CopyCpuUint64SequenceInto(input_offsets, "BatchPut input_offsets",
                              &batch_put_request_.input_offsets);
    CopyCpuUint64SequenceInto(lengths, "BatchPut lengths",
                              &batch_put_request_.lengths);
    if (batch_put_request_.keys.empty() ||
        batch_put_request_.keys.size() !=
            batch_put_request_.input_offsets.size() ||
        batch_put_request_.keys.size() != batch_put_request_.lengths.size()) {
      throw py::value_error(
          "BatchPut keys, input_offsets, and lengths must be non-empty and equal in size");
    }
    const NpuTensorView source_view = GetNpuTensorView(source, "BatchPut source");
    const aclrtStream acl_stream = ResolveStream(stream);
    batch_put_request_.input =
        ConstBufferView{source_view.data, source_view.bytes, DeviceKind::kNpu};
    ThrowIfError(native_->core().BatchPut(
        batch_put_request_, BatchPutExecutionOptions{acl_stream}));
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

  void StartScatterProfile() {
    const char* value = std::getenv("KVAF_SCATTER_HOST_PROFILE");
    if (value && std::string(value) != "0" && std::string(value) != "1")
      throw py::value_error("KVAF_SCATTER_HOST_PROFILE must be 0 or 1");
    ThrowIfError(native_->SetScatterProfiling(true));
    host_profile_enabled_ = value && std::string(value) == "1";
    host_profile_ready_ = false;
  }
  void StopScatterProfile() {
    ThrowIfError(native_->SetScatterProfiling(false));
    host_profile_enabled_ = host_profile_ready_ = false;
  }
  py::list ScatterProfileSnapshot() const {
    std::vector<kvaf::ScatterBatchGetCoreProfile> profiles;
    ThrowIfError(native_->CopyScatterProfiles(&profiles));
    py::list result;
    for (std::size_t core = 0; core < profiles.size(); ++core) {
      const auto& profile = profiles[core];
      py::dict item;
      item["core"] = core;
      item["magic"] = profile.magic;
      item["sequence"] = profile.sequence;
      item["start_cycle"] = profile.start_cycle;
      item["inputs_ready_cycle"] = profile.inputs_ready_cycle;
      item["notify_cycle"] = profile.notify_cycle;
      item["done_cycle"] = profile.done_cycle;
      item["finish_cycle"] = profile.finish_cycle;
      item["requests"] = profile.requests;
      item["bytes"] = profile.bytes;
      item["cpu_magic"] = profile.cpu_magic;
      item["cpu_header_ns"] = profile.cpu_header_ns;
      item["cpu_descriptor_ns"] = profile.cpu_descriptor_ns;
      item["cpu_write_ns"] = profile.cpu_write_ns;
      item["cpu_comm_ns"] = profile.cpu_comm_ns;
      item["cpu_post_ns"] = profile.cpu_post_ns;
      item["cpu_cq_wait_ns"] = profile.cpu_cq_wait_ns;
      item["pipeline"] = profile.pipeline;
      item["scatter_cycles"] = profile.scatter_cycles;
      item["ready_wait_cycles"] = profile.ready_wait_cycles;
      item["scatter_start_cycle"] = profile.scatter_start_cycle;
      item["scatter_finish_cycle"] = profile.scatter_finish_cycle;
      item["scatter_chunks"] = profile.scatter_chunks;
      if (!core && host_profile_enabled_ && host_profile_ready_) {
        py::dict host;
        for (std::size_t phase = 0; phase < scatter_host::kCount; ++phase)
          host[scatter_host::kNames[phase]] = host_profile_.ns[phase];
        item["host_profile"] = std::move(host);
      }
      result.append(std::move(item));
    }
    return result;
  }

  void StartBatchGetKernelProfile() {
    ThrowIfError(native_->SetBatchGetKernelProfiling(true));
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
      item["submitted_group_count"] = profile.submitted_group_count;
      item["completed_group_count"] = profile.completed_group_count;
      item["completion_count"] = profile.completion_count;
      item["post_active_cycles"] = profile.post_active_cycles;
      item["drain_wait_cycles"] = profile.drain_wait_cycles;
      item["peak_submitted_not_drained"] = profile.peak_submitted_not_drained;
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
  // Tensors carry no AccessTableHandle. Checked modes validate their NPU
  // layout; scatter uses the caller's trusted-input contract in either overload.
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
    scatter_host::Session session(host_profile_enabled_ ? &host_profile_ : nullptr);
    scatter_host::Timer binding_timer(scatter_host::kBinding);
    scatter_host::Timer tensor_timer(scatter_host::kTensorMetadata);
    host_profile_ready_ = false;
    BatchGetRequest request;
    if (trusted_scatter_inputs_) {
      request = TrustedScatterRequest(access_table, addresses, valid, indices, offsets,
          miss_count, output_indices, output, native_->batch_get_token_bytes(),
          native_->batch_get_tokens_per_block(), tensor_methods_);
    } else {
      const NpuTensorView index_view = GetNpuTensorView(indices, "indices");
      const NpuTensorView offset_view = GetNpuTensorView(offsets, "offsets");
      const NpuTensorView miss_count_view =
          GetNpuTensorView(miss_count, "miss_count");
      const NpuTensorView output_index_view =
          GetNpuTensorView(output_indices, "output_indices");
      const NpuTensorView output_view = GetNpuTensorView(output, "output");
      const NpuTensorView address_view = GetNpuTensorView(addresses, "addresses");
      const NpuTensorView valid_view = GetNpuTensorView(valid, "valid");
      if (index_view.dimensions != 1 || offset_view.dimensions != 1 ||
          output_index_view.dimensions != 1 || output_view.dimensions < 2 ||
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
          output_view.first_dimension <= 0 ||
          output_view.bytes % static_cast<std::size_t>(output_view.first_dimension) != 0) {
        throw py::value_error("BatchGet Tensors do not satisfy the NPU layout contract");
      }
      const std::uint64_t output_rows =
          static_cast<std::uint64_t>(output_view.first_dimension);
      const std::uint64_t output_block_bytes = output_view.bytes / output_rows;
      if (output_block_bytes != native_->batch_get_token_bytes()) {
        throw py::value_error(
            "BatchGet output Tensor row size does not match batch_get_token_bytes");
      }
      request = BatchGetRequest{
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
    }
    tensor_timer.Stop();
    scatter_host::Timer stream_timer(scatter_host::kStreamResolve);
    const aclrtStream acl_stream = trusted_scatter_inputs_ ?
        tensor_methods_.Stream(stream) : ResolveStream(stream);
    stream_timer.Stop();
    Status status;
    {
      scatter_host::Timer native_timer(scatter_host::kNative);
      status = native_->core().BatchGet(
          request, BatchGetExecutionOptions{acl_stream, synchronize, trusted_scatter_inputs_});
    }
    ThrowIfError(status);
    host_profile_ready_ = host_profile_enabled_;
  }

 public:
  void Shutdown() {
    ThrowIfError(native_->Shutdown());
    host_profile_enabled_ = host_profile_ready_ = false;
  }

 private:
  std::unique_ptr<::kvaf::NpuRuntime> native_;
  TensorMetadataMethods tensor_methods_;
  bool trusted_scatter_inputs_{false};
  bool host_profile_enabled_{false}, host_profile_ready_{false};
  scatter_host::Profile host_profile_{};
  // Runtime::BatchPut is synchronous, so these vectors can be reused after
  // each call and avoid allocator churn in the measured loop.
  BatchPutRequest batch_put_request_;
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
                    std::uint32_t, std::string, std::uint32_t,
                    std::vector<std::string>, bool, std::uint32_t,
                    std::string, std::vector<std::string>, std::uint32_t, std::string, std::uint32_t>(),
           py::arg("batch_get_kernel_object_path"),
           py::arg("batch_get_token_bytes"),
           py::arg("batch_get_tokens_per_block"), py::arg("device_id") = 0,
           py::arg("npu_core_count") = 2,
           py::arg("batch_put_kernel_object_path") = "",
           py::arg("jettys_per_core") = 2,
           py::arg("npu_eids") = std::vector<std::string>{},
           py::arg("adaptive_ub_device") = false,
           py::arg("adaptive_ub_miss_count") = 2048,
           py::arg("peer_ub_device") = "",
           py::arg("adaptive_peer_ub_devices") =
               std::vector<std::string>{},
           py::arg("batch_get_max_inflight_total") = 2048,
           py::arg("aiv_batch_get_mode") = "direct_read",
           py::arg("aiv_gather_sgl_size") = 6)
      .def("link_to", &NpuPythonRuntime::LinkTo, py::arg("address"),
           "Connect this NPU Runtime to a CPU Runtime service before initialize.")
      .def("initialize", &NpuPythonRuntime::Initialize, py::arg("ddr_bytes"),
           py::arg("slot_size"))
      .def("shutdown", &NpuPythonRuntime::Shutdown)
      .def_property_readonly("initialized", &NpuPythonRuntime::Initialized)
      .def_property_readonly("device_id", &NpuPythonRuntime::DeviceId)
      .def_property_readonly("npu_core_count", &NpuPythonRuntime::NpuCoreCount)
      .def_property_readonly("jettys_per_core", &NpuPythonRuntime::JettysPerCore)
      .def_property_readonly("npu_eids", &NpuPythonRuntime::NpuEids)
      .def_property_readonly("batch_get_token_bytes",
                             &NpuPythonRuntime::BatchGetTokenBytes)
      .def_property_readonly("batch_get_tokens_per_block",
                             &NpuPythonRuntime::BatchGetTokensPerBlock)
      .def_property_readonly("remote_region", &NpuPythonRuntime::RemoteRegion)
      .def("batch_put", &NpuPythonRuntime::BatchPut, py::arg("keys"),
           py::arg("source"), py::arg("input_offsets"), py::arg("lengths"),
           py::arg("stream") = py::none(),
           "Synchronously write variable-size packed NPU blocks. keys, input_offsets, and "
           "lengths are CPU integer sequences or contiguous CPU int64 tensors "
           "with one entry per key; "
           "stream=None uses torch.npu.current_stream().")
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
      .def("start_scatter_profile", &NpuPythonRuntime::StartScatterProfile,
           "Enable optional AIV/CPU scatter timing; call after initialize.")
      .def("scatter_profile", &NpuPythonRuntime::ScatterProfileSnapshot,
           "Drain and copy the latest scatter timing records outside the measured call.")
      .def("stop_scatter_profile", &NpuPythonRuntime::StopScatterProfile)
      .def("start_batch_get_kernel_profile",
           &NpuPythonRuntime::StartBatchGetKernelProfile,
           "Enable the per-core BatchGet kernel cycle timeline.")
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
           "the default returns after enqueueing work on that stream. CPU-push scatter trusts "
           "the caller's contiguous NPU layouts and eager-stream/lifetime contract.")
      .def("batch_get", &NpuPythonRuntime::BatchGetFromTensors,
           py::arg("addresses"), py::arg("valid"), py::arg("indices"),
           py::arg("offsets"), py::arg("miss_count"), py::arg("output_indices"),
           py::arg("output"), py::arg("stream") = py::none(),
           py::arg("synchronize") = false,
           "Submit BatchGet from caller-owned NPU addresses and valid Tensors. "
           "stream=None uses torch.npu.current_stream(); the default returns "
           "after enqueueing work on that stream. CPU-push scatter trusts the "
           "caller's contiguous NPU layouts and eager-stream/lifetime contract.");
}

}  // namespace kvaf
