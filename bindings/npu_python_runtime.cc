#include "npu_python_runtime.h"

#include <pybind11/stl.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "access_table_python.h"
#include "acl/acl.h"
#include "acl_python_utils.h"
#include "batch_get_abi.h"
#include "kvaf/access_operator/access_operator.h"
#include "kvaf/meta_manager/meta_manager.h"
#include "kvaf/runtime/runtime.h"

namespace py = pybind11;

namespace kvaf {
namespace {

Status InvalidArgument(std::string message) {
  return Status::InvalidArgument(std::move(message));
}

Status Internal(std::string message) {
  return Status::Internal(std::move(message));
}

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

class AclHostDdrAllocator final : public DdrMemoryAllocator {
 public:
  Status Allocate(const RuntimeInitRequest& request,
                  RemoteRegionDesc* region) override {
    if (region == nullptr || request.ddr_device != DeviceKind::kCpu ||
        request.ddr_bytes == 0 || request.slot_size == 0 ||
        request.ddr_bytes % request.slot_size != 0) {
      return InvalidArgument("invalid ACL Host DDR allocation request");
    }
    if (address_ != nullptr) {
      return Status::Error(StatusCode::kInvalidState,
                           "ACL Host DDR is already allocated");
    }
    if (aclrtMallocHost(&address_, request.ddr_bytes) != ACL_SUCCESS) {
      return Status::Error(StatusCode::kOutOfMemory,
                           "aclrtMallocHost failed for Runtime DDR");
    }
    *region = RemoteRegionDesc{reinterpret_cast<Address>(address_),
                               request.ddr_bytes, request.slot_size,
                               DeviceKind::kCpu};
    return Status::OK();
  }

  Status Release(const RemoteRegionDesc& region) override {
    if (address_ == nullptr || reinterpret_cast<Address>(address_) != region.base) {
      return Status::Error(StatusCode::kNotFound,
                           "Runtime DDR allocation was not found");
    }
    const aclError error = aclrtFreeHost(address_);
    address_ = nullptr;
    return error == ACL_SUCCESS
               ? Status::OK()
               : Internal("aclrtFreeHost failed for Runtime DDR");
  }

 private:
  void* address_{nullptr};
};

void* PointerFromValue(std::uint64_t pointer) {
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer));
}

struct NpuTensorView {
  void* data{nullptr};
  std::size_t bytes{0};
  std::size_t element_bytes{0};
  std::vector<py::ssize_t> shape;
};

using ProfileClock = std::chrono::steady_clock;

std::uint64_t ElapsedNs(ProfileClock::time_point begin,
                        ProfileClock::time_point end) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

std::uint64_t SaturatingSubtract(std::uint64_t total,
                                 std::uint64_t accounted) {
  return total > accounted ? total - accounted : 0;
}

struct BatchGetBindingCallProfile {
  std::uint64_t binding_total_ns{0};
  std::uint64_t indices_tensor_ns{0};
  std::uint64_t mask_tensor_ns{0};
  std::uint64_t output_indices_tensor_ns{0};
  std::uint64_t output_tensor_ns{0};
  std::uint64_t table_addresses_tensor_ns{0};
  std::uint64_t table_valid_tensor_ns{0};
  std::uint64_t validate_and_build_request_ns{0};
  std::uint64_t resolve_stream_ns{0};
  std::uint64_t runtime_call_ns{0};
  BatchGetExecutionProfile execution;
};

struct BatchGetBindingProfile {
  bool enabled{false};
  std::uint64_t samples{0};
  BatchGetBindingCallProfile totals;

  void Start() {
    *this = BatchGetBindingProfile{};
    enabled = true;
  }

  void Add(const BatchGetBindingCallProfile& sample) {
    ++samples;
    totals.binding_total_ns += sample.binding_total_ns;
    totals.indices_tensor_ns += sample.indices_tensor_ns;
    totals.mask_tensor_ns += sample.mask_tensor_ns;
    totals.output_indices_tensor_ns += sample.output_indices_tensor_ns;
    totals.output_tensor_ns += sample.output_tensor_ns;
    totals.table_addresses_tensor_ns += sample.table_addresses_tensor_ns;
    totals.table_valid_tensor_ns += sample.table_valid_tensor_ns;
    totals.validate_and_build_request_ns +=
        sample.validate_and_build_request_ns;
    totals.resolve_stream_ns += sample.resolve_stream_ns;
    totals.runtime_call_ns += sample.runtime_call_ns;
    totals.execution.runtime_total_ns += sample.execution.runtime_total_ns;
    totals.execution.runtime_validation_ns +=
        sample.execution.runtime_validation_ns;
    totals.execution.operator_total_ns += sample.execution.operator_total_ns;
    totals.execution.operator_build_args_ns +=
        sample.execution.operator_build_args_ns;
    totals.execution.operator_local_buffer_acquire_ns +=
        sample.execution.operator_local_buffer_acquire_ns;
    totals.execution.operator_kernel_launch_ns +=
        sample.execution.operator_kernel_launch_ns;
    totals.execution.operator_stream_synchronize_ns +=
        sample.execution.operator_stream_synchronize_ns;
    totals.execution.operator_status_copy_ns +=
        sample.execution.operator_status_copy_ns;
    totals.execution.operator_local_buffer_release_ns +=
        sample.execution.operator_local_buffer_release_ns;
  }

  py::dict ToPython() const {
    py::dict total_ns;
    py::dict mean_us;
    const auto add = [&](const char* name, std::uint64_t value) {
      total_ns[name] = value;
      mean_us[name] = samples == 0
                          ? 0.0
                          : static_cast<double>(value) /
                                static_cast<double>(samples) / 1000.0;
    };

    const std::uint64_t tensor_views_ns =
        totals.indices_tensor_ns + totals.mask_tensor_ns +
        totals.output_indices_tensor_ns + totals.output_tensor_ns +
        totals.table_addresses_tensor_ns + totals.table_valid_tensor_ns;
    const std::uint64_t binding_accounted_ns =
        tensor_views_ns + totals.validate_and_build_request_ns +
        totals.resolve_stream_ns + totals.runtime_call_ns;
    const std::uint64_t runtime_accounted_ns =
        totals.execution.runtime_validation_ns +
        totals.execution.operator_total_ns;
    const std::uint64_t operator_accounted_ns =
        totals.execution.operator_build_args_ns +
        totals.execution.operator_local_buffer_acquire_ns +
        totals.execution.operator_kernel_launch_ns +
        totals.execution.operator_stream_synchronize_ns +
        totals.execution.operator_status_copy_ns +
        totals.execution.operator_local_buffer_release_ns;

    add("binding_total", totals.binding_total_ns);
    add("binding_tensor_views", tensor_views_ns);
    add("binding_indices_tensor", totals.indices_tensor_ns);
    add("binding_mask_tensor", totals.mask_tensor_ns);
    add("binding_output_indices_tensor", totals.output_indices_tensor_ns);
    add("binding_output_tensor", totals.output_tensor_ns);
    add("binding_table_addresses_tensor",
        totals.table_addresses_tensor_ns);
    add("binding_table_valid_tensor", totals.table_valid_tensor_ns);
    add("binding_validate_and_build_request",
        totals.validate_and_build_request_ns);
    add("binding_resolve_stream", totals.resolve_stream_ns);
    add("binding_runtime_call", totals.runtime_call_ns);
    add("binding_unaccounted",
        SaturatingSubtract(totals.binding_total_ns, binding_accounted_ns));

    add("runtime_total", totals.execution.runtime_total_ns);
    add("runtime_validation", totals.execution.runtime_validation_ns);
    add("runtime_operator", totals.execution.operator_total_ns);
    add("runtime_unaccounted",
        SaturatingSubtract(totals.execution.runtime_total_ns,
                           runtime_accounted_ns));

    add("operator_total", totals.execution.operator_total_ns);
    add("operator_build_args", totals.execution.operator_build_args_ns);
    add("operator_local_buffer_acquire",
        totals.execution.operator_local_buffer_acquire_ns);
    add("operator_kernel_launch",
        totals.execution.operator_kernel_launch_ns);
    add("operator_stream_synchronize",
        totals.execution.operator_stream_synchronize_ns);
    add("operator_status_copy", totals.execution.operator_status_copy_ns);
    add("operator_local_buffer_release",
        totals.execution.operator_local_buffer_release_ns);
    add("operator_unaccounted",
        SaturatingSubtract(totals.execution.operator_total_ns,
                           operator_accounted_ns));

    py::dict result;
    result["enabled"] = enabled;
    result["samples"] = samples;
    result["total_ns"] = std::move(total_ns);
    result["mean_us"] = std::move(mean_us);
    return result;
  }
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

class NpuRuntime final {
 public:
  NpuRuntime(std::string batch_get_kernel_object_path, std::uint32_t device_id,
             std::uint32_t npu_core_count,
             std::string batch_put_kernel_object_path,
             std::uint32_t jettys_per_core)
      : device_id_(device_id), npu_core_count_(npu_core_count),
        jettys_per_core_(jettys_per_core),
        meta_manager_(CreateInMemoryMetaManager()) {
    if (batch_get_kernel_object_path.empty() || npu_core_count == 0 ||
        npu_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES ||
        jettys_per_core == 0 ||
        jettys_per_core > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE) {
      throw py::value_error("invalid NPU Runtime configuration");
    }

    // torch_npu owns process-wide ACL initialisation/finalisation. The binding
    // therefore only selects the requested device; it never calls aclFinalize
    // or aclrtResetDevice. The Python caller supplies an optional stream for
    // each BatchGet, so external framework graph capture stays outside KVAF.
    ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");

    allocator_ = std::make_unique<AclHostDdrAllocator>();
    access_operator_ = std::make_unique<DefaultAccessOperator>(
        AccessOperatorConfig{std::move(batch_get_kernel_object_path),
                             std::move(batch_put_kernel_object_path),
                             npu_core_count_, jettys_per_core_});
    runtime_ = std::make_unique<Runtime>(*meta_manager_, *access_operator_,
                                         *allocator_);
  }

  ~NpuRuntime() { CloseNoThrow(); }

  NpuRuntime(const NpuRuntime&) = delete;
  NpuRuntime& operator=(const NpuRuntime&) = delete;

  void Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
    ThrowIfError(runtime_->Initialize(
        RuntimeInitRequest{ddr_bytes, slot_size, DeviceKind::kCpu}));
  }

  bool Initialized() const {
    return runtime_ != nullptr && runtime_->initialized();
  }

  std::uint32_t DeviceId() const { return device_id_; }
  std::uint32_t NpuCoreCount() const { return npu_core_count_; }
  std::uint32_t JettysPerCore() const { return jettys_per_core_; }

  py::dict RemoteRegion() const {
    const RemoteRegionDesc& region = runtime_->remote_region();
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
    ThrowIfError(runtime_->BatchPut(BatchPutRequest{
        keys,
        ConstBufferView{source_view.data, source_view.bytes, DeviceKind::kNpu},
        input_offsets,
        lengths}, BatchPutExecutionOptions{acl_stream}));
  }

  AccessTableHandle PrepareAccessTable(const py::object& keys) {
    std::vector<KeyId> copied_keys = CopyPrepareKeys(keys);
    AccessTableHandle handle;
    ThrowIfError(runtime_->PrepareAccessTable(
        AccessTablePrepareRequest{std::move(copied_keys)}, &handle));
    return handle;
  }

  py::object AccessTableView(const AccessTableHandle& handle,
                             const std::string& device,
                             std::uint64_t capacity) const {
    kvaf::AccessTableView view;
    ThrowIfError(runtime_->GetAccessTableView(handle, &view));
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
    const std::string npu_device = "npu:" + std::to_string(device_id_);
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
    ThrowIfError(runtime_->ReleaseAccessTable(handle));
  }

  // BatchFree is a control-plane operation.  It deliberately takes the same
  // Python key forms as PrepareAccessTable (a Python key sequence or a
  // contiguous one-dimensional CPU torch.int64 Tensor), rather than imposing
  // a second representation solely for the NPU Runtime binding.
  void BatchFree(const py::object& keys) {
    std::vector<KeyId> copied_keys = CopyPrepareKeys(keys);
    ThrowIfError(runtime_->BatchFree(BatchFreeRequest{std::move(copied_keys)}));
  }

  void StartBatchGetProfile() { batch_get_profile_.Start(); }

  py::dict BatchGetProfile() const { return batch_get_profile_.ToPython(); }

  py::dict StopBatchGetProfile() {
    batch_get_profile_.enabled = false;
    return batch_get_profile_.ToPython();
  }

  // Tensor ownership remains with Python. KVAF creates no output Tensor and
  // only retains its internal Direct-HCCP execution state; callers must keep
  // input/output Tensors and the DeviceAccessTable alive until their stream
  // completes the BatchGet operation.
  void BatchGet(const DeviceAccessTable& access_table,
                const py::object& indices, const py::object& mask,
                const py::object& output_indices, const py::object& output,
                const py::object& stream, bool synchronize) {
    const bool profiling = batch_get_profile_.enabled;
    BatchGetBindingCallProfile call_profile;
    const auto binding_begin = profiling ? ProfileClock::now()
                                         : ProfileClock::time_point{};

    auto stage_begin = profiling ? ProfileClock::now()
                                 : ProfileClock::time_point{};
    const NpuTensorView index_view = GetNpuTensorView(indices, "indices");
    if (profiling) {
      call_profile.indices_tensor_ns =
          ElapsedNs(stage_begin, ProfileClock::now());
      stage_begin = ProfileClock::now();
    }
    const NpuTensorView mask_view = GetNpuTensorView(mask, "mask");
    if (profiling) {
      call_profile.mask_tensor_ns =
          ElapsedNs(stage_begin, ProfileClock::now());
      stage_begin = ProfileClock::now();
    }
    const NpuTensorView output_index_view =
        GetNpuTensorView(output_indices, "output_indices");
    if (profiling) {
      call_profile.output_indices_tensor_ns =
          ElapsedNs(stage_begin, ProfileClock::now());
      stage_begin = ProfileClock::now();
    }
    const NpuTensorView output_view = GetNpuTensorView(output, "output");
    if (profiling) {
      call_profile.output_tensor_ns =
          ElapsedNs(stage_begin, ProfileClock::now());
      stage_begin = ProfileClock::now();
    }
    const NpuTensorView address_view =
        GetNpuTensorView(access_table.addresses(), "device_access_table.addresses");
    if (profiling) {
      call_profile.table_addresses_tensor_ns =
          ElapsedNs(stage_begin, ProfileClock::now());
      stage_begin = ProfileClock::now();
    }
    const NpuTensorView valid_view =
        GetNpuTensorView(access_table.valid(), "device_access_table.valid");
    if (profiling) {
      call_profile.table_valid_tensor_ns =
          ElapsedNs(stage_begin, ProfileClock::now());
      stage_begin = ProfileClock::now();
    }
    if (index_view.shape.size() != 1 || mask_view.shape.size() != 1 ||
        output_index_view.shape.size() != 1 || output_view.shape.size() < 2 ||
        index_view.element_bytes != sizeof(std::int32_t) ||
        output_index_view.element_bytes != sizeof(std::int32_t) ||
        mask_view.element_bytes != sizeof(std::uint8_t) ||
        index_view.bytes != output_index_view.bytes ||
        mask_view.bytes != index_view.bytes / sizeof(std::int32_t) ||
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
    if (output_block_bytes == 0 ||
        output_block_bytes > std::numeric_limits<std::uint32_t>::max()) {
      throw py::value_error("BatchGet output Tensor has an invalid row size");
    }
    const BatchGetRequest request{
        access_table.source(),
        ConstBufferView{index_view.data, index_view.bytes, DeviceKind::kNpu},
        ConstBufferView{mask_view.data, mask_view.bytes, DeviceKind::kNpu},
        ConstBufferView{output_index_view.data, output_index_view.bytes,
                        DeviceKind::kNpu},
        MutableBufferView{output_view.data, output_view.bytes, DeviceKind::kNpu},
        output_block_bytes,
        AccessTableDeviceView{
            ConstBufferView{address_view.data, address_view.bytes, DeviceKind::kNpu},
            ConstBufferView{valid_view.data, valid_view.bytes, DeviceKind::kNpu}}};
    if (profiling) {
      call_profile.validate_and_build_request_ns =
          ElapsedNs(stage_begin, ProfileClock::now());
      stage_begin = ProfileClock::now();
    }
    const aclrtStream acl_stream = ResolveStream(stream);
    if (profiling) {
      call_profile.resolve_stream_ns =
          ElapsedNs(stage_begin, ProfileClock::now());
      stage_begin = ProfileClock::now();
    }
    ThrowIfError(runtime_->BatchGet(
        request, BatchGetExecutionOptions{acl_stream,
                                           synchronize,
                                           profiling ? &call_profile.execution
                                                     : nullptr}));
    if (profiling) {
      const auto binding_end = ProfileClock::now();
      call_profile.runtime_call_ns = ElapsedNs(stage_begin, binding_end);
      call_profile.binding_total_ns =
          ElapsedNs(binding_begin, binding_end);
      batch_get_profile_.Add(call_profile);
    }
  }

  void Shutdown() {
    if (runtime_ != nullptr && runtime_->initialized()) {
      ThrowIfError(runtime_->Shutdown());
    }
  }

 private:
  void CloseNoThrow() {
    if (runtime_ != nullptr && runtime_->initialized()) {
      (void)runtime_->Shutdown();
    }
    runtime_.reset();
    access_operator_.reset();
    allocator_.reset();
  }

  std::uint32_t device_id_{0};
  std::uint32_t npu_core_count_{0};
  std::uint32_t jettys_per_core_{2};
  std::unique_ptr<MetaManager> meta_manager_;
  std::unique_ptr<AclHostDdrAllocator> allocator_;
  std::unique_ptr<DefaultAccessOperator> access_operator_;
  std::unique_ptr<Runtime> runtime_;
  BatchGetBindingProfile batch_get_profile_;
};

}  // namespace

void BindNpuPythonRuntime(py::module_& module) {
  py::class_<DeviceAccessTable>(module, "DeviceAccessTable")
      .def_property_readonly("addresses", &DeviceAccessTable::addresses)
      .def_property_readonly("valid", &DeviceAccessTable::valid)
      .def_property_readonly("source_id", [](const DeviceAccessTable& table) {
        return table.source().opaque;
      });

  py::class_<NpuRuntime>(module, "NpuRuntime")
      .def(py::init<std::string, std::uint32_t, std::uint32_t, std::string,
                    std::uint32_t>(),
           py::arg("batch_get_kernel_object_path"), py::arg("device_id") = 0,
           py::arg("npu_core_count") = 1,
           py::arg("batch_put_kernel_object_path") = "",
           py::arg("jettys_per_core") = 2)
      .def("initialize", &NpuRuntime::Initialize, py::arg("ddr_bytes"),
           py::arg("slot_size"))
      .def("shutdown", &NpuRuntime::Shutdown)
      .def_property_readonly("initialized", &NpuRuntime::Initialized)
      .def_property_readonly("device_id", &NpuRuntime::DeviceId)
      .def_property_readonly("npu_core_count", &NpuRuntime::NpuCoreCount)
      .def_property_readonly("jettys_per_core", &NpuRuntime::JettysPerCore)
      .def_property_readonly("remote_region", &NpuRuntime::RemoteRegion)
      .def("batch_put", &NpuRuntime::BatchPut, py::arg("keys"),
           py::arg("source"), py::arg("input_offsets"), py::arg("lengths"),
           py::arg("stream") = py::none(),
           "Synchronously write variable-size packed NPU blocks. input_offsets and "
           "lengths are CPU integer sequences with one entry per key; "
           "stream=None uses torch.npu.current_stream().")
      .def("prepare_access_table", &NpuRuntime::PrepareAccessTable,
           py::arg("keys"))
      .def("access_table_view", &NpuRuntime::AccessTableView,
           py::arg("access_table"), py::arg("device") = "cpu",
           py::arg("capacity") = 0)
      .def("release_access_table", &NpuRuntime::ReleaseAccessTable,
           py::arg("access_table"))
      .def("batch_free", &NpuRuntime::BatchFree, py::arg("keys"),
           "Release the supplied keys from the Runtime metadata store. keys "
           "accepts a Python key sequence or a contiguous one-dimensional "
           "CPU torch.int64 Tensor.")
      .def("start_batch_get_profile", &NpuRuntime::StartBatchGetProfile,
           "Reset and enable cumulative BatchGet host timing.")
      .def("batch_get_profile", &NpuRuntime::BatchGetProfile,
           "Return the current cumulative BatchGet host timing snapshot.")
      .def("stop_batch_get_profile", &NpuRuntime::StopBatchGetProfile,
           "Disable and return cumulative BatchGet host timing.")
      .def("batch_get", &NpuRuntime::BatchGet,
           py::arg("device_access_table"), py::arg("indices"), py::arg("mask"),
           py::arg("output_indices"), py::arg("output"),
           py::arg("stream") = py::none(), py::arg("synchronize") = false,
           "Submit BatchGet from NPU Tensors. stream=None uses torch.npu.current_stream(); "
           "the default returns after enqueueing work on that stream.");
}

}  // namespace kvaf
