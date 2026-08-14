#include "npu_python_runtime.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
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

std::string ConfigString(const py::object& config,
                         const std::vector<const char*>& keys,
                         const std::vector<const char*>& environment_keys,
                         const std::string& default_value = "") {
  if (!config.is_none()) {
    if (!py::isinstance<py::dict>(config)) {
      throw py::type_error("npu_config must be a dict when supplied");
    }
    const py::dict values = config.cast<py::dict>();
    for (const char* key : keys) {
      const py::str name(key);
      if (values.contains(name)) {
        const py::object candidate = values[name];
        if (candidate.is_none()) continue;
        const std::string value = py::str(candidate).cast<std::string>();
        if (!value.empty()) return value;
      }
    }
  }
  for (const char* key : environment_keys) {
    const char* value = std::getenv(key);
    if (value != nullptr && value[0] != '\0') return std::string(value);
  }
  return default_value;
}

std::uint32_t ConfigU32(const py::object& config,
                        const std::vector<const char*>& keys,
                        const std::vector<const char*>& environment_keys,
                        std::uint32_t default_value) {
  const std::string value = ConfigString(config, keys, environment_keys);
  if (value.empty()) return default_value;
  try {
    const unsigned long long parsed = std::stoull(value);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
      throw py::value_error("NPU configuration value exceeds uint32");
    }
    return static_cast<std::uint32_t>(parsed);
  } catch (const std::invalid_argument&) {
    throw py::value_error("NPU configuration value must be an unsigned integer");
  } catch (const std::out_of_range&) {
    throw py::value_error("NPU configuration value exceeds uint32");
  }
}

std::uint64_t DictU64(const py::dict& values, const char* key,
                      std::uint64_t default_value = 0,
                      bool required = true) {
  const py::str name(key);
  if (!values.contains(name)) {
    if (required) {
      throw py::value_error(std::string("missing required DSA_ROW field: ") + key);
    }
    return default_value;
  }
  const py::object value = values[name];
  if (value.is_none()) {
    throw py::value_error(std::string("DSA_ROW field must not be None: ") + key);
  }
  try {
    return value.cast<std::uint64_t>();
  } catch (const py::cast_error&) {
    throw py::value_error(std::string("DSA_ROW field must be an integer: ") + key);
  }
}

bool FitsU32(std::uint64_t value) {
  return value <= std::numeric_limits<std::uint32_t>::max();
}

bool AddWouldOverflow(std::uint64_t left, std::uint64_t right) {
  return left > std::numeric_limits<std::uint64_t>::max() - right;
}

bool MultiplyWouldOverflow(std::uint64_t left, std::uint64_t right) {
  return left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left;
}

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
  if (value == 0 || alignment == 0 ||
      alignment > std::numeric_limits<std::uint64_t>::max() - value + 1) {
    return 0;
  }
  const std::uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
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

  // Tensor ownership remains with Python. KVAF creates no output Tensor and
  // only retains its internal Direct-HCCP execution state; callers must keep
  // input/output Tensors and the DeviceAccessTable alive until their stream
  // completes the BatchGet operation.
  void BatchGet(const DeviceAccessTable& access_table,
                const py::object& indices, const py::object& mask,
                const py::object& output_indices, const py::object& output,
                const py::object& stream, bool synchronize) {
    const NpuTensorView index_view = GetNpuTensorView(indices, "indices");
    const NpuTensorView mask_view = GetNpuTensorView(mask, "mask");
    const NpuTensorView output_index_view =
        GetNpuTensorView(output_indices, "output_indices");
    const NpuTensorView output_view = GetNpuTensorView(output, "output");
    const NpuTensorView address_view =
        GetNpuTensorView(access_table.addresses(), "device_access_table.addresses");
    const NpuTensorView valid_view =
        GetNpuTensorView(access_table.valid(), "device_access_table.valid");
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
    const aclrtStream acl_stream = ResolveStream(stream);
    ThrowIfError(runtime_->BatchGet(
        request, BatchGetExecutionOptions{acl_stream,
                                           synchronize}));
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
};

// Production DSA_ROW binding.  Unlike DsaRowMockRuntime this owns a real
// Direct-HCCP Host-DDR arena plus an NPU-resident row-major directory.  It
// deliberately does not construct KeyIds or an AccessTable: every data-plane
// address is `flat_index -> directory UBA`, and DirectGet additionally carries
// a packed-token offset to the AIV kernel.
class DsaRowRuntime final {
 public:
  DsaRowRuntime() = default;
  ~DsaRowRuntime() { CloseNoThrow(); }

  DsaRowRuntime(const DsaRowRuntime&) = delete;
  DsaRowRuntime& operator=(const DsaRowRuntime&) = delete;

  static const char* ProbeDsaRowPayloadAbi() {
    return "torch_npu_packed_v1";
  }

  void InitializeDsaRowIndex(const py::dict& layout, const py::dict& geometry,
                             std::uint32_t ddr_block_count,
                             const py::object& npu_config) {
    if (initialized_) {
      throw py::value_error("DSA_ROW index is already initialized");
    }
#if !defined(KVAF_ENABLE_AIV_BATCH_PUT)
    (void)layout;
    (void)geometry;
    (void)ddr_block_count;
    (void)npu_config;
    throw std::runtime_error(
        "kvaf.DsaRowRuntime requires KVAF_ENABLE_AIV_BATCH_PUT=ON; "
        "the installed binding cannot provide the DSA_ROW payload route");
#else
    InitializeLayout(layout, geometry, ddr_block_count);
    const std::string batch_get_kernel = ConfigString(
        npu_config,
        {"batch_get_kernel_object_path", "KVAF_AIV_BATCH_GET_KERNEL"},
        {"KVAF_AIV_BATCH_GET_KERNEL", "KVAF_DSA_BATCH_GET_KERNEL_OBJECT_PATH"});
    const std::string batch_put_kernel = ConfigString(
        npu_config,
        {"batch_put_kernel_object_path", "KVAF_AIV_BATCH_PUT_KERNEL"},
        {"KVAF_AIV_BATCH_PUT_KERNEL", "KVAF_DSA_BATCH_PUT_KERNEL_OBJECT_PATH"});
    device_id_ = ConfigU32(
        npu_config, {"device_id", "KVAF_NPU_DEVICE_ID"},
        {"KVAF_NPU_DEVICE_ID", "KVAF_DSA_DEVICE_ID"}, 0);
    npu_core_count_ = ConfigU32(
        npu_config, {"npu_core_count", "KVAF_NPU_CORE_COUNT"},
        {"KVAF_NPU_CORE_COUNT", "KVAF_DSA_NPU_CORE_COUNT"}, 1);
    jettys_per_core_ = ConfigU32(
        npu_config, {"jettys_per_core", "KVAF_JETTYS_PER_CORE"},
        {"KVAF_JETTYS_PER_CORE", "KVAF_DSA_JETTYS_PER_CORE"}, 2);
    if (batch_get_kernel.empty() || batch_put_kernel.empty() ||
        npu_core_count_ == 0 ||
        npu_core_count_ > KVAF_AIV_BATCH_GET_MAX_AIV_CORES ||
        jettys_per_core_ == 0 ||
        jettys_per_core_ > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE) {
      throw py::value_error(
          "invalid DSA_ROW npu_config: require non-empty "
          "KVAF_AIV_BATCH_GET_KERNEL/KVAF_AIV_BATCH_PUT_KERNEL and valid "
          "KVAF_NPU_CORE_COUNT/KVAF_JETTYS_PER_CORE");
    }
    ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");

    allocator_ = std::make_unique<AclHostDdrAllocator>();
    const std::uint64_t ddr_bytes =
        static_cast<std::uint64_t>(ddr_block_count_) * layout_.ddr_block_bytes;
    Status status = allocator_->Allocate(
        RuntimeInitRequest{ddr_bytes, layout_.ddr_block_bytes, DeviceKind::kCpu},
        &remote_region_);
    if (!status.ok()) {
      allocator_.reset();
      ThrowIfError(status);
    }
    access_operator_ = std::make_unique<DefaultAccessOperator>(
        AccessOperatorConfig{batch_get_kernel, batch_put_kernel, npu_core_count_,
                             jettys_per_core_});
    status = access_operator_->Initialize(remote_region_);
    if (!status.ok()) {
      (void)allocator_->Release(remote_region_);
      access_operator_.reset();
      allocator_.reset();
      remote_region_ = RemoteRegionDesc{};
      ThrowIfError(status);
    }

    try {
      device_directory_ = AllocateNpuTensor("int64", geometry_.directory_cells);
      device_valid_ = AllocateNpuTensor("uint8", geometry_.directory_cells);
      const NpuTensorView directory =
          GetNpuTensorView(device_directory_, "DSA_ROW device directory");
      const NpuTensorView valid =
          GetNpuTensorView(device_valid_, "DSA_ROW device valid directory");
      if (directory.element_bytes != sizeof(Address) ||
          valid.element_bytes != sizeof(std::uint8_t) ||
          directory.bytes != host_directory_.size() * sizeof(Address) ||
          valid.bytes != host_valid_.size()) {
        throw std::runtime_error("invalid DSA_ROW device directory Tensor layout");
      }
      ThrowIfAclError(aclrtMemset(directory.data, directory.bytes, 0,
                                   directory.bytes),
                      "aclrtMemset DSA_ROW device directory");
      ThrowIfAclError(aclrtMemset(valid.data, valid.bytes, 0, valid.bytes),
                      "aclrtMemset DSA_ROW device valid directory");
    } catch (...) {
      CloseNoThrow();
      throw;
    }
    initialized_ = true;
#endif
  }

  void BindResidentLayerCache(std::uint32_t layer_id, const py::object& cache,
                              std::uint64_t byte_capacity,
                              std::uint64_t row_capacity,
                              const py::dict& layout) {
    RequireInitialized();
    ValidateLayoutPayload(layout);
    if (layer_id >= geometry_.n_layers || byte_capacity == 0 ||
        row_capacity == 0 || bound_layers_.count(layer_id) != 0) {
      throw py::value_error("invalid or duplicate DSA_ROW resident layer binding");
    }
    const NpuTensorView view =
        GetNpuTensorView(cache, "DSA_ROW resident packed cache");
    if (view.bytes < byte_capacity ||
        byte_capacity % layout_.packed_token_bytes != 0 ||
        row_capacity > byte_capacity / layout_.packed_token_bytes) {
      throw py::value_error(
          "DSA_ROW resident packed cache capacity is incompatible with layout");
    }
    bound_layers_.emplace(layer_id,
                          LayerBinding{cache, byte_capacity, row_capacity});
  }

  py::none BindResidentPoolRow(std::uint32_t pool_row,
                               std::uint32_t max_sequence_tokens,
                               std::uint32_t resident_budget_tokens) {
    RequireInitialized();
    if (pool_row >= rows_.size() || max_sequence_tokens == 0 ||
        resident_budget_tokens > max_sequence_tokens) {
      throw py::value_error("invalid DSA_ROW resident-pool row binding");
    }
    RowRecord& row = rows_[pool_row];
    if (row.state != RowState::kFree) {
      throw py::value_error("DSA_ROW resident-pool row is not free");
    }
    const std::uint64_t max_blocks =
        (static_cast<std::uint64_t>(max_sequence_tokens) +
         layout_.tokens_per_logical_block - 1U) /
        layout_.tokens_per_logical_block;
    if (max_blocks == 0 || max_blocks > geometry_.q_capacity) {
      throw py::value_error("DSA_ROW max_sequence_tokens exceeds q_capacity");
    }
    row.state = RowState::kOpen;
    row.max_blocks = static_cast<std::uint32_t>(max_blocks);
    row.committed_blocks = 0;
    return py::none();
  }

  py::int_ PrepareFullBlockGroupPut(const py::iterable& all_targets,
                                    const py::iterable& row_commits) {
    RequireInitialized();
    std::vector<std::uint32_t> targets;
    for (const py::handle value : all_targets) {
      const std::uint64_t flat = value.cast<std::uint64_t>();
      if (!FitsU32(flat)) {
        throw py::value_error("DSA_ROW Put flat_index exceeds uint32");
      }
      targets.push_back(static_cast<std::uint32_t>(flat));
    }
    if (targets.empty()) {
      throw py::value_error("DSA_ROW group Put requires non-empty targets");
    }
    std::vector<RowCommit> commits;
    for (const py::handle value : row_commits) {
      if (!py::isinstance<py::dict>(value)) {
        throw py::type_error("DSA_ROW row_commits entries must be dicts");
      }
      const py::dict entry = py::reinterpret_borrow<py::dict>(value);
      const std::uint64_t base = DictU64(entry, "row_flat_base");
      const std::uint64_t expected =
          DictU64(entry, "expected_committed_block_count");
      const std::uint64_t appended = DictU64(entry, "appended_block_count");
      if (!FitsU32(base) || !FitsU32(expected) || !FitsU32(appended)) {
        throw py::value_error("DSA_ROW row commit exceeds uint32");
      }
      commits.push_back(RowCommit{static_cast<std::uint32_t>(base),
                                  static_cast<std::uint32_t>(expected),
                                  static_cast<std::uint32_t>(appended)});
    }
    if (commits.empty()) {
      throw py::value_error("DSA_ROW group Put requires row_commits");
    }
    return py::int_(PreparePut(std::move(targets), std::move(commits)));
  }

  py::none WriteFullBlockGroupLayer(const py::object& plan,
                                    const py::iterable& layer_blocks,
                                    const py::dict& layer_cache,
                                    const py::object& stream) {
    RequireInitialized();
    PutPlan& put = FindPutPlan(ParseHandle(plan, "Put plan"));
    if (put.state != PutState::kPrepared) {
      throw py::value_error("DSA_ROW Put plan is not writable");
    }
    const std::uint32_t layer = static_cast<std::uint32_t>(
        DictU64(layer_cache, "local_layer_ordinal"));
    if (layer >= geometry_.n_layers || put.written_layers.count(layer) != 0) {
      throw py::value_error("DSA_ROW layer is invalid or was already written");
    }
    const py::str cache_name("cache");
    if (!layer_cache.contains(cache_name)) {
      throw py::value_error("DSA_ROW layer cache is missing 'cache'");
    }
    const py::object cache = layer_cache[cache_name];
    const NpuTensorView source =
        GetNpuTensorView(cache, "DSA_ROW full-block Put source cache");
    const std::uint64_t byte_capacity = DictU64(layer_cache, "byte_capacity");
    const auto binding = bound_layers_.find(layer);
    if (binding == bound_layers_.end() ||
        binding->second.byte_capacity != byte_capacity ||
        binding->second.row_capacity != DictU64(layer_cache, "row_capacity") ||
        GetNpuTensorView(binding->second.cache,
                         "DSA_ROW bound resident cache").data != source.data ||
        source.bytes < byte_capacity) {
      throw py::value_error(
          "DSA_ROW layer cache does not match its bound packed-cache view");
    }

    std::vector<LayerBlock> blocks;
    std::set<std::uint32_t> supplied;
    for (const py::handle value : layer_blocks) {
      if (!py::isinstance<py::dict>(value)) {
        throw py::type_error("DSA_ROW layer_blocks entries must be dicts");
      }
      const py::dict entry = py::reinterpret_borrow<py::dict>(value);
      const std::uint64_t flat = DictU64(entry, "flat_index");
      const std::uint64_t offset = DictU64(entry, "source_byte_offset");
      const std::uint64_t length = DictU64(entry, "byte_length");
      if (!FitsU32(flat) || length != layout_.layer_payload_bytes ||
          offset > byte_capacity || length > byte_capacity - offset ||
          !supplied.insert(static_cast<std::uint32_t>(flat)).second) {
        throw py::value_error("invalid DSA_ROW layer Put block descriptor");
      }
      blocks.push_back(LayerBlock{static_cast<std::uint32_t>(flat), offset,
                                  static_cast<std::uint32_t>(length)});
    }
    ValidateLayerBlocks(put, layer, blocks);

    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> lengths;
    std::vector<Address> destinations;
    offsets.reserve(blocks.size());
    lengths.reserve(blocks.size());
    destinations.reserve(blocks.size());
    for (const LayerBlock& block : blocks) {
      PutTarget& target = FindPutTarget(put, block.flat_index);
      offsets.push_back(block.source_offset);
      lengths.push_back(block.byte_length);
      destinations.push_back(DdrBlockAddress(target.ddr_block));
    }
    const aclrtStream acl_stream = ResolveStream(stream);
    const Status status = access_operator_->ExecuteDsaFullBlockPut(
        DsaFullBlockPutRequest{ConstBufferView{source.data, source.bytes,
                                               DeviceKind::kNpu},
                               std::move(offsets), std::move(lengths),
                               std::move(destinations), acl_stream});
    ThrowIfError(status);
    for (const LayerBlock& block : blocks) {
      FindPutTarget(put, block.flat_index).written = true;
    }
    put.written_layers.insert(layer);
    return py::none();
  }

  py::dict SealFullBlockGroupPut(const py::object& plan) {
    RequireInitialized();
    PutPlan& put = FindPutPlan(ParseHandle(plan, "Put plan"));
    if (put.state != PutState::kPrepared) {
      throw py::value_error("DSA_ROW Put plan is not ready to seal");
    }
    if (put.written_layers.size() != geometry_.n_layers) {
      throw py::value_error(
          "DSA_ROW Put plan cannot seal until every local layer is written");
    }
    for (const PutTarget& target : put.targets) {
      if (!target.written) {
        throw py::value_error("DSA_ROW Put plan has an unwritten target");
      }
    }

    std::vector<Address> pending_directory = host_directory_;
    std::vector<std::uint8_t> pending_valid = host_valid_;
    for (const PutTarget& target : put.targets) {
      pending_directory[target.flat_index] = DdrBlockAddress(target.ddr_block);
      pending_valid[target.flat_index] = 1;
    }
    UploadDirectory(pending_directory, pending_valid);
    host_directory_.swap(pending_directory);
    host_valid_.swap(pending_valid);
    for (const PutTarget& target : put.targets) {
      block_states_[target.ddr_block] = BlockState::kOwned;
    }
    put.state = PutState::kSealed;
    py::dict result;
    result["state"] = "sealed_ready";
    result["published_block_count"] = put.targets.size();
    return result;
  }

  py::dict CommitFullBlockGroupPut(const py::object& plan) {
    RequireInitialized();
    const std::uint64_t handle = ParseHandle(plan, "Put plan");
    PutPlan& put = FindPutPlan(handle);
    if (put.state != PutState::kSealed) {
      throw py::value_error("DSA_ROW Put plan must be sealed before commit");
    }
    py::list row_results;
    for (const RowCommit& commit : put.commits) {
      const std::uint32_t pool_row = commit.row_flat_base / geometry_.row_stride;
      RowRecord& row = rows_[pool_row];
      if (row.state != RowState::kOpen || row.active_put_id != handle) {
        throw py::value_error("DSA_ROW row changed while Put plan was active");
      }
      row.committed_blocks += commit.appended_blocks;
      row.active_put_id = 0;
      py::dict row_result;
      row_result["row_flat_base"] = commit.row_flat_base;
      row_result["old_committed_block_count"] = commit.expected_blocks;
      row_result["new_committed_block_count"] = row.committed_blocks;
      row_results.append(std::move(row_result));
    }
    put_plans_.erase(handle);
    py::dict result;
    result["state"] = "committed";
    result["row_commits"] = std::move(row_results);
    return result;
  }

  py::dict AbortFullBlockGroupPut(const py::object& plan) {
    RequireInitialized();
    const std::uint64_t handle = ParseHandle(plan, "Put plan");
    PutPlan& put = FindPutPlan(handle);
    if (put.state == PutState::kSealed) {
      std::vector<Address> pending_directory = host_directory_;
      std::vector<std::uint8_t> pending_valid = host_valid_;
      for (const PutTarget& target : put.targets) {
        pending_directory[target.flat_index] = 0;
        pending_valid[target.flat_index] = 0;
      }
      UploadDirectory(pending_directory, pending_valid);
      host_directory_.swap(pending_directory);
      host_valid_.swap(pending_valid);
    }
    for (const PutTarget& target : put.targets) {
      cell_blocks_[target.flat_index] = kInvalidBlock;
      if (block_states_[target.ddr_block] != BlockState::kFree) {
        block_states_[target.ddr_block] = BlockState::kFree;
        free_blocks_.push_back(target.ddr_block);
      }
    }
    for (const RowCommit& commit : put.commits) {
      const std::uint32_t pool_row = commit.row_flat_base / geometry_.row_stride;
      if (rows_[pool_row].active_put_id == handle) {
        rows_[pool_row].active_put_id = 0;
      }
    }
    put_plans_.erase(handle);
    py::dict result;
    result["state"] = "aborted_safe";
    return result;
  }

  py::int_ PrepareUnifiedAccess(const py::iterable& rows) {
    RequireInitialized();
    AccessPlan access;
    std::set<std::uint32_t> seen_sparse_rows;
    for (const py::handle value : rows) {
      if (!py::isinstance<py::dict>(value)) {
        throw py::type_error("DSA_ROW unified-access entries must be dicts");
      }
      const py::dict entry = py::reinterpret_borrow<py::dict>(value);
      const py::object mode_value = entry["mode"];
      const py::object row_valid_value = entry["row_valid"];
      const std::string mode = py::str(mode_value).cast<std::string>();
      const bool row_valid = row_valid_value.cast<bool>();
      const std::uint64_t expected =
          DictU64(entry, "expected_committed_block_count");
      if (mode == "pad") {
        if (row_valid || expected != 0) {
          throw py::value_error("DSA_ROW PAD access must be invalid");
        }
        continue;
      }
      if (mode == "dense_local") {
        if (!row_valid || expected != 0) {
          throw py::value_error("DSA_ROW dense_local access is malformed");
        }
        continue;
      }
      if (mode != "sparse_remote" || !row_valid) {
        throw py::value_error("DSA_ROW sparse access is malformed");
      }
      const py::str pool_name("pool_row");
      if (!entry.contains(pool_name)) {
        throw py::value_error("DSA_ROW sparse access requires pool_row");
      }
      const py::object pool_value = entry[pool_name];
      if (pool_value.is_none()) {
        throw py::value_error("DSA_ROW sparse access requires pool_row");
      }
      const std::uint64_t raw_pool = pool_value.cast<std::uint64_t>();
      if (!FitsU32(raw_pool) || raw_pool >= rows_.size() ||
          !seen_sparse_rows.insert(static_cast<std::uint32_t>(raw_pool)).second) {
        throw py::value_error("DSA_ROW sparse access has an invalid pool_row");
      }
      RowRecord& row = rows_[static_cast<std::uint32_t>(raw_pool)];
      if (row.state != RowState::kOpen || expected != row.committed_blocks) {
        throw py::value_error(
            "DSA_ROW sparse access disagrees with current row visibility");
      }
      access.sparse_rows.insert(static_cast<std::uint32_t>(raw_pool));
    }
    const std::uint64_t handle = NextHandle();
    for (const std::uint32_t pool_row : access.sparse_rows) {
      ++rows_[pool_row].active_accesses;
    }
    accesses_.emplace(handle, std::move(access));
    return py::int_(handle);
  }

  py::int_ PrepareDirectBatchGet(const py::object& access,
                                 const py::iterable& items,
                                 const py::dict& output_cache,
                                 const py::object& stream) {
    RequireInitialized();
    const std::uint64_t access_handle = ParseHandle(access, "unified access");
    AccessPlan& access_plan = FindAccessPlan(access_handle);
    if (access_plan.released) {
      throw py::value_error("DSA_ROW unified access was already released");
    }
    const std::uint32_t output_layer = static_cast<std::uint32_t>(
        DictU64(output_cache, "local_layer_ordinal"));
    const std::uint64_t byte_capacity =
        DictU64(output_cache, "byte_capacity");
    const std::uint64_t row_capacity = DictU64(output_cache, "row_capacity");
    const py::str cache_name("cache");
    if (output_layer >= geometry_.n_layers || !output_cache.contains(cache_name)) {
      throw py::value_error("invalid DSA_ROW DirectGet output cache");
    }
    const py::object output = output_cache[cache_name];
    const NpuTensorView output_view =
        GetNpuTensorView(output, "DSA_ROW DirectGet output cache");
    const auto binding = bound_layers_.find(output_layer);
    if (binding == bound_layers_.end() ||
        binding->second.byte_capacity != byte_capacity ||
        binding->second.row_capacity != row_capacity ||
        GetNpuTensorView(binding->second.cache,
                         "DSA_ROW bound output cache").data != output_view.data ||
        byte_capacity == 0 || row_capacity == 0 || output_view.bytes < byte_capacity ||
        byte_capacity % layout_.packed_token_bytes != 0 ||
        row_capacity > byte_capacity / layout_.packed_token_bytes) {
      throw py::value_error("DSA_ROW DirectGet output cache capacity is invalid");
    }

    std::vector<std::int32_t> flat_indices;
    std::vector<std::int32_t> token_offsets;
    std::vector<std::int32_t> destinations;
    std::vector<std::uint8_t> mask;
    std::set<std::int32_t> unique_destinations;
    for (const py::handle value : items) {
      if (!py::isinstance<py::dict>(value)) {
        throw py::type_error("DSA_ROW DirectGet items must be dicts");
      }
      const py::dict entry = py::reinterpret_borrow<py::dict>(value);
      const std::uint64_t flat = DictU64(entry, "flat_index");
      const std::uint64_t offset = DictU64(entry, "offset");
      const std::uint64_t destination = DictU64(entry, "destination_index");
      if (!FitsU32(flat) || !FitsU32(offset) || !FitsU32(destination) ||
          flat >= geometry_.directory_cells ||
          offset >= layout_.tokens_per_logical_block ||
          destination >= row_capacity ||
          !unique_destinations.insert(static_cast<std::int32_t>(destination)).second) {
        throw py::value_error("DSA_ROW DirectGet item is invalid or duplicate");
      }
      const std::uint32_t flat_u32 = static_cast<std::uint32_t>(flat);
      const std::uint32_t pool_row = flat_u32 / geometry_.row_stride;
      const std::uint32_t in_row = flat_u32 % geometry_.row_stride;
      const std::uint32_t layer = in_row / geometry_.q_capacity;
      const std::uint32_t block_q = in_row % geometry_.q_capacity;
      if (layer != output_layer || access_plan.sparse_rows.count(pool_row) == 0 ||
          rows_[pool_row].state != RowState::kOpen ||
          block_q >= rows_[pool_row].committed_blocks ||
          host_valid_[flat_u32] == 0) {
        throw py::value_error(
            "DSA_ROW DirectGet source is outside the prepared sparse snapshot");
      }
      flat_indices.push_back(static_cast<std::int32_t>(flat));
      token_offsets.push_back(static_cast<std::int32_t>(offset));
      destinations.push_back(static_cast<std::int32_t>(destination));
      mask.push_back(1);
    }
    const std::uint64_t get_handle = NextHandle();
    GetPlan plan;
    plan.access_handle = access_handle;
    plan.item_count = static_cast<std::uint32_t>(flat_indices.size());
    plan.output = output;
    plan.stream = ResolveStream(stream);
    if (!flat_indices.empty()) {
      plan.flat_indices = MakeNpuTensor(flat_indices, "int32");
      plan.token_offsets = MakeNpuTensor(token_offsets, "int32");
      plan.mask = MakeNpuTensor(mask, "uint8");
      plan.destinations = MakeNpuTensor(destinations, "int32");
    }
    ++access_plan.active_gets;
    get_plans_.emplace(get_handle, std::move(plan));
    return py::int_(get_handle);
  }

  // Fast decode-facing variant: vLLM constructs these descriptors on NPU from
  // A5 LIDU metadata and resident-pool maps.  KVAF only validates the stable
  // tensor ABI; it never copies or inspects them on CPU.
  py::int_ PrepareDirectBatchGetDevice(
      const py::object& access, const py::object& flat_indices,
      const py::object& token_offsets, const py::object& mask,
      const py::object& destination_indices, const py::dict& output_cache,
      const py::object& stream) {
    RequireInitialized();
    const std::uint64_t access_handle = ParseHandle(access, "unified access");
    AccessPlan& access_plan = FindAccessPlan(access_handle);
    if (access_plan.released) {
      throw py::value_error("DSA_ROW unified access was already released");
    }
    const std::uint32_t output_layer = static_cast<std::uint32_t>(
        DictU64(output_cache, "local_layer_ordinal"));
    const std::uint64_t byte_capacity =
        DictU64(output_cache, "byte_capacity");
    const std::uint64_t row_capacity = DictU64(output_cache, "row_capacity");
    const py::str cache_name("cache");
    if (output_layer >= geometry_.n_layers || !output_cache.contains(cache_name)) {
      throw py::value_error("invalid DSA_ROW DirectGet output cache");
    }
    const py::object output = output_cache[cache_name];
    const NpuTensorView output_view =
        GetNpuTensorView(output, "DSA_ROW DirectGet output cache");
    const NpuTensorView index_view =
        GetNpuTensorView(flat_indices, "DSA_ROW DirectGet flat_indices");
    const NpuTensorView offset_view =
        GetNpuTensorView(token_offsets, "DSA_ROW DirectGet token_offsets");
    const NpuTensorView mask_view =
        GetNpuTensorView(mask, "DSA_ROW DirectGet mask");
    const NpuTensorView destination_view = GetNpuTensorView(
        destination_indices, "DSA_ROW DirectGet destination_indices");
    const auto binding = bound_layers_.find(output_layer);
    if (binding == bound_layers_.end() ||
        binding->second.byte_capacity != byte_capacity ||
        binding->second.row_capacity != row_capacity ||
        GetNpuTensorView(binding->second.cache,
                         "DSA_ROW bound output cache").data != output_view.data ||
        byte_capacity == 0 || row_capacity == 0 ||
        output_view.bytes < byte_capacity ||
        byte_capacity % layout_.packed_token_bytes != 0 ||
        row_capacity > byte_capacity / layout_.packed_token_bytes ||
        index_view.shape.size() != 1 || offset_view.shape.size() != 1 ||
        mask_view.shape.size() != 1 || destination_view.shape.size() != 1 ||
        index_view.element_bytes != sizeof(std::int32_t) ||
        offset_view.element_bytes != sizeof(std::int32_t) ||
        destination_view.element_bytes != sizeof(std::int32_t) ||
        mask_view.element_bytes != sizeof(std::uint8_t) ||
        index_view.bytes == 0 || offset_view.bytes != index_view.bytes ||
        destination_view.bytes != index_view.bytes ||
        mask_view.bytes != index_view.bytes / sizeof(std::int32_t)) {
      throw py::value_error(
          "DSA_ROW device DirectGet descriptors do not satisfy the NPU ABI");
    }
    const std::size_t item_count = index_view.bytes / sizeof(std::int32_t);
    if (item_count > std::numeric_limits<std::uint32_t>::max()) {
      throw py::value_error("DSA_ROW device DirectGet request count exceeds uint32");
    }
    const std::uint64_t get_handle = NextHandle();
    GetPlan plan;
    plan.access_handle = access_handle;
    plan.item_count = static_cast<std::uint32_t>(item_count);
    plan.flat_indices = flat_indices;
    plan.token_offsets = token_offsets;
    plan.mask = mask;
    plan.destinations = destination_indices;
    plan.output = output;
    plan.stream = ResolveStream(stream);
    ++access_plan.active_gets;
    get_plans_.emplace(get_handle, std::move(plan));
    return py::int_(get_handle);
  }

  py::int_ ExecutePreparedDirectBatchGet(const py::object& plan) {
    RequireInitialized();
    const std::uint64_t handle = ParseHandle(plan, "DirectGet plan");
    GetPlan& get = FindGetPlan(handle);
    if (get.executed) {
      throw py::value_error("DSA_ROW DirectGet plan was already executed");
    }
    if (get.item_count != 0) {
      const NpuTensorView directory =
          GetNpuTensorView(device_directory_, "DSA_ROW device directory");
      const NpuTensorView valid =
          GetNpuTensorView(device_valid_, "DSA_ROW device valid directory");
      const NpuTensorView indices =
          GetNpuTensorView(get.flat_indices, "DSA_ROW DirectGet flat indices");
      const NpuTensorView offsets =
          GetNpuTensorView(get.token_offsets, "DSA_ROW DirectGet offsets");
      const NpuTensorView mask =
          GetNpuTensorView(get.mask, "DSA_ROW DirectGet mask");
      const NpuTensorView destinations = GetNpuTensorView(
          get.destinations, "DSA_ROW DirectGet destination indices");
      const NpuTensorView output =
          GetNpuTensorView(get.output, "DSA_ROW DirectGet output");
      const Status status = access_operator_->ExecuteDsaDirectBatchGet(
          DsaDirectBatchGetRequest{
              ConstBufferView{directory.data, directory.bytes, DeviceKind::kNpu},
              ConstBufferView{valid.data, valid.bytes, DeviceKind::kNpu},
              ConstBufferView{indices.data, indices.bytes, DeviceKind::kNpu},
              ConstBufferView{offsets.data, offsets.bytes, DeviceKind::kNpu},
              ConstBufferView{mask.data, mask.bytes, DeviceKind::kNpu},
              ConstBufferView{destinations.data, destinations.bytes,
                              DeviceKind::kNpu},
              MutableBufferView{output.data, output.bytes, DeviceKind::kNpu},
              layout_.packed_token_bytes, layout_.tokens_per_logical_block,
              get.stream,
              // M0 is intentionally synchronous: a committed source remains
              // pinned until the direct read completes and the next C8 QSFA
              // may consume resident HBM immediately after this verb returns.
              true});
      if (!status.ok()) {
        FinishGetPlan(handle, false);
        ThrowIfError(status);
      }
    }
    get.executed = true;
    return py::int_(handle);
  }

  py::dict SettleDirectBatchGet(const py::object& plan,
                                const py::object& completion) {
    RequireInitialized();
    const std::uint64_t handle = ParseHandle(plan, "DirectGet plan");
    if (!completion.is_none() &&
        ParseHandle(completion, "DirectGet completion") != handle) {
      throw py::value_error("DSA_ROW DirectGet completion does not match plan");
    }
    GetPlan& get = FindGetPlan(handle);
    if (!get.executed) {
      throw py::value_error("DSA_ROW DirectGet plan has not executed");
    }
    const std::uint32_t count = get.item_count;
    FinishGetPlan(handle, true);
    py::dict result;
    result["state"] = "success";
    result["completed_item_count"] = count;
    return result;
  }

  py::none ReleaseUnifiedAccess(const py::object& access) {
    RequireInitialized();
    const std::uint64_t handle = ParseHandle(access, "unified access");
    AccessPlan& access_plan = FindAccessPlan(handle);
    if (access_plan.active_gets != 0) {
      throw py::value_error(
          "DSA_ROW unified access cannot release while DirectGet is active");
    }
    for (const std::uint32_t pool_row : access_plan.sparse_rows) {
      if (rows_[pool_row].active_accesses == 0) {
        throw std::runtime_error("DSA_ROW sparse access pin underflow");
      }
      --rows_[pool_row].active_accesses;
    }
    accesses_.erase(handle);
    return py::none();
  }

  py::dict CloseResidentPoolRow(std::uint32_t pool_row) {
    RequireInitialized();
    if (pool_row >= rows_.size()) {
      throw py::value_error("DSA_ROW pool_row is outside geometry");
    }
    RowRecord& row = rows_[pool_row];
    if (row.state == RowState::kFree) {
      py::dict already;
      already["state"] = "closed";
      already["released_block_count"] = 0;
      return already;
    }
    if (row.state != RowState::kOpen || row.active_put_id != 0 ||
        row.active_accesses != 0) {
      throw py::value_error("DSA_ROW pool row is busy and cannot close");
    }
    std::vector<Address> pending_directory = host_directory_;
    std::vector<std::uint8_t> pending_valid = host_valid_;
    std::vector<std::uint32_t> released;
    const std::uint32_t row_base = pool_row * geometry_.row_stride;
    for (std::uint32_t index = 0; index < geometry_.row_stride; ++index) {
      const std::uint32_t flat = row_base + index;
      if (cell_blocks_[flat] != kInvalidBlock) {
        released.push_back(cell_blocks_[flat]);
        pending_directory[flat] = 0;
        pending_valid[flat] = 0;
      }
    }
    UploadDirectory(pending_directory, pending_valid);
    host_directory_.swap(pending_directory);
    host_valid_.swap(pending_valid);
    for (const std::uint32_t block : released) {
      block_states_[block] = BlockState::kFree;
      free_blocks_.push_back(block);
    }
    // Clear every cell in the row only after the zeroed device directory has
    // been published, so a returned DDR block is never reused while its old
    // flat-index entry is still device-visible.
    for (std::uint32_t index = 0; index < geometry_.row_stride; ++index) {
      cell_blocks_[row_base + index] = kInvalidBlock;
    }
    row = RowRecord{};
    py::dict result;
    result["state"] = "closed";
    result["released_block_count"] = released.size();
    return result;
  }

  void Close() {
    if (!initialized_) return;
    for (const RowRecord& row : rows_) {
      if (row.state == RowState::kOpen &&
          (row.active_put_id != 0 || row.active_accesses != 0)) {
        throw py::value_error(
            "DSA_ROW close requires all rows/accesses to be released first");
      }
    }
    CloseNoThrow();
  }

  bool Initialized() const { return initialized_; }

  py::dict Geometry() const {
    RequireInitialized();
    py::dict result;
    result["raw_row_capacity"] = geometry_.raw_row_capacity;
    result["n_layers"] = geometry_.n_layers;
    result["q_capacity"] = geometry_.q_capacity;
    result["row_stride_cells"] = geometry_.row_stride;
    result["directory_cell_count"] = geometry_.directory_cells;
    result["ddr_block_count"] = ddr_block_count_;
    result["ddr_block_bytes"] = layout_.ddr_block_bytes;
    return result;
  }

 private:
  static constexpr std::uint32_t kInvalidBlock =
      std::numeric_limits<std::uint32_t>::max();

  enum class RowState : std::uint8_t { kFree = 0, kOpen };
  enum class BlockState : std::uint8_t { kFree = 0, kReserved, kOwned };
  enum class PutState : std::uint8_t { kPrepared = 0, kSealed };

  struct Layout {
    std::uint32_t tokens_per_logical_block{0};
    std::uint32_t packed_token_bytes{0};
    std::uint32_t ddr_block_alignment_bytes{0};
    std::uint64_t layer_payload_bytes{0};
    std::uint64_t ddr_block_bytes{0};
  };

  struct GeometryState {
    std::uint32_t raw_row_capacity{0};
    std::uint32_t n_layers{0};
    std::uint32_t q_capacity{0};
    std::uint32_t row_stride{0};
    std::uint32_t directory_cells{0};
  };

  struct RowRecord {
    RowState state{RowState::kFree};
    std::uint32_t max_blocks{0};
    std::uint32_t committed_blocks{0};
    std::uint64_t active_put_id{0};
    std::uint32_t active_accesses{0};
  };

  struct LayerBinding {
    py::object cache;
    std::uint64_t byte_capacity{0};
    std::uint64_t row_capacity{0};
  };

  struct RowCommit {
    std::uint32_t row_flat_base{0};
    std::uint32_t expected_blocks{0};
    std::uint32_t appended_blocks{0};
  };

  struct PutTarget {
    std::uint32_t flat_index{0};
    std::uint32_t ddr_block{kInvalidBlock};
    bool written{false};
  };

  struct LayerBlock {
    std::uint32_t flat_index{0};
    std::uint64_t source_offset{0};
    std::uint32_t byte_length{0};
  };

  struct PutPlan {
    PutState state{PutState::kPrepared};
    std::vector<RowCommit> commits;
    std::vector<PutTarget> targets;
    std::unordered_map<std::uint32_t, std::size_t> target_positions;
    std::set<std::uint32_t> written_layers;
  };

  struct AccessPlan {
    std::set<std::uint32_t> sparse_rows;
    std::uint32_t active_gets{0};
    bool released{false};
  };

  struct GetPlan {
    std::uint64_t access_handle{0};
    std::uint32_t item_count{0};
    py::object flat_indices;
    py::object token_offsets;
    py::object mask;
    py::object destinations;
    py::object output;
    aclrtStream stream{nullptr};
    bool executed{false};
  };

  void RequireInitialized() const {
    if (!initialized_ || access_operator_ == nullptr || allocator_ == nullptr) {
      throw std::runtime_error("DSA_ROW runtime has not been initialized");
    }
  }

  void InitializeLayout(const py::dict& layout, const py::dict& geometry,
                        std::uint32_t ddr_block_count) {
    const std::uint64_t tokens =
        DictU64(layout, "tokens_per_logical_block");
    const std::uint64_t token_bytes = DictU64(layout, "packed_token_bytes");
    const std::uint64_t alignment =
        DictU64(layout, "ddr_block_alignment_bytes");
    const std::uint64_t raw_rows =
        DictU64(geometry, "raw_row_capacity");
    const std::uint64_t layers = DictU64(geometry, "n_layers");
    const std::uint64_t q_capacity = DictU64(geometry, "q_capacity");
    if (ddr_block_count == 0 || !FitsU32(tokens) || !FitsU32(token_bytes) ||
        !FitsU32(alignment) || !FitsU32(raw_rows) || !FitsU32(layers) ||
        !FitsU32(q_capacity) ||
        MultiplyWouldOverflow(tokens, token_bytes)) {
      throw py::value_error("invalid DSA_ROW packed layout or geometry");
    }
    const std::uint64_t payload = tokens * token_bytes;
    const std::uint64_t block_bytes = AlignUp(payload, alignment);
    if (block_bytes == 0 || block_bytes > std::numeric_limits<std::uint32_t>::max() ||
        MultiplyWouldOverflow(layers, q_capacity)) {
      throw py::value_error("DSA_ROW packed block or row stride overflows");
    }
    const std::uint64_t row_stride = layers * q_capacity;
    if (row_stride == 0 || !FitsU32(row_stride) ||
        MultiplyWouldOverflow(raw_rows, row_stride) ||
        !FitsU32(raw_rows * row_stride) ||
        MultiplyWouldOverflow(ddr_block_count, block_bytes)) {
      throw py::value_error("DSA_ROW directory or DDR arena exceeds its ABI");
    }
    const std::uint64_t supplied_payload =
        DictU64(layout, "layer_payload_bytes", payload, false);
    const std::uint64_t supplied_block =
        DictU64(layout, "ddr_block_bytes", block_bytes, false);
    if (supplied_payload != payload || supplied_block != block_bytes) {
      throw py::value_error(
          "DSA_ROW layout derived sizes disagree with the supplied payload");
    }
    const std::uint64_t supplied_stride =
        DictU64(geometry, "row_stride_cells", row_stride, false);
    const std::uint64_t supplied_cells =
        DictU64(geometry, "directory_cell_count", raw_rows * row_stride,
                false);
    if (supplied_stride != row_stride ||
        supplied_cells != raw_rows * row_stride) {
      throw py::value_error("DSA_ROW redundant geometry disagrees with layout");
    }
    layout_ = Layout{static_cast<std::uint32_t>(tokens),
                     static_cast<std::uint32_t>(token_bytes),
                     static_cast<std::uint32_t>(alignment), payload,
                     block_bytes};
    geometry_ = GeometryState{static_cast<std::uint32_t>(raw_rows),
                              static_cast<std::uint32_t>(layers),
                              static_cast<std::uint32_t>(q_capacity),
                              static_cast<std::uint32_t>(row_stride),
                              static_cast<std::uint32_t>(raw_rows * row_stride)};
    ddr_block_count_ = ddr_block_count;
    rows_.assign(geometry_.raw_row_capacity, RowRecord{});
    host_directory_.assign(geometry_.directory_cells, 0);
    host_valid_.assign(geometry_.directory_cells, 0);
    cell_blocks_.assign(geometry_.directory_cells, kInvalidBlock);
    block_states_.assign(ddr_block_count_, BlockState::kFree);
    free_blocks_.clear();
    free_blocks_.reserve(ddr_block_count_);
    for (std::uint32_t block = 0; block < ddr_block_count_; ++block) {
      free_blocks_.push_back(block);
    }
  }

  void ValidateLayoutPayload(const py::dict& layout) const {
    const std::uint64_t tokens =
        DictU64(layout, "tokens_per_logical_block");
    const std::uint64_t token_bytes = DictU64(layout, "packed_token_bytes");
    const std::uint64_t alignment =
        DictU64(layout, "ddr_block_alignment_bytes");
    const std::uint64_t payload = DictU64(layout, "layer_payload_bytes");
    const std::uint64_t block = DictU64(layout, "ddr_block_bytes");
    if (tokens != layout_.tokens_per_logical_block ||
        token_bytes != layout_.packed_token_bytes ||
        alignment != layout_.ddr_block_alignment_bytes ||
        payload != layout_.layer_payload_bytes ||
        block != layout_.ddr_block_bytes) {
      throw py::value_error("DSA_ROW layer binding uses a mismatched packed layout");
    }
  }

  py::object AllocateNpuTensor(const char* dtype, std::uint64_t count) const {
    py::module_ torch = py::module_::import("torch");
    return torch.attr("empty")(
        py::make_tuple(count), py::arg("dtype") = torch.attr(dtype),
        py::arg("device") = "npu:" + std::to_string(device_id_));
  }

  template <typename T>
  py::object MakeNpuTensor(const std::vector<T>& host, const char* dtype) const {
    if (host.empty()) {
      throw py::value_error("DSA_ROW cannot materialize an empty NPU descriptor");
    }
    py::object tensor = AllocateNpuTensor(dtype, host.size());
    const NpuTensorView view = GetNpuTensorView(tensor, "DSA_ROW descriptor");
    if (view.bytes != host.size() * sizeof(T)) {
      throw std::runtime_error("torch created an invalid DSA_ROW descriptor Tensor");
    }
    ThrowIfAclError(aclrtMemcpy(view.data, view.bytes, host.data(), view.bytes,
                                 ACL_MEMCPY_HOST_TO_DEVICE),
                    "aclrtMemcpy DSA_ROW descriptor");
    return tensor;
  }

  void UploadDirectory(const std::vector<Address>& directory,
                       const std::vector<std::uint8_t>& valid) const {
    if (directory.size() != host_directory_.size() ||
        valid.size() != host_valid_.size()) {
      throw std::runtime_error("DSA_ROW directory publication size mismatch");
    }
    const NpuTensorView device_directory =
        GetNpuTensorView(device_directory_, "DSA_ROW device directory");
    const NpuTensorView device_valid =
        GetNpuTensorView(device_valid_, "DSA_ROW device valid directory");
    ThrowIfAclError(
        aclrtMemcpy(device_directory.data, device_directory.bytes,
                    directory.data(), directory.size() * sizeof(Address),
                    ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy DSA_ROW address directory publication");
    ThrowIfAclError(
        aclrtMemcpy(device_valid.data, device_valid.bytes, valid.data(), valid.size(),
                    ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy DSA_ROW valid directory publication");
  }

  std::uint64_t NextHandle() {
    if (next_handle_ == 0) {
      throw std::runtime_error("DSA_ROW handle space is exhausted");
    }
    return next_handle_++;
  }

  std::uint64_t ParseHandle(const py::object& value, const char* name) const {
    try {
      const std::uint64_t handle = value.cast<std::uint64_t>();
      if (handle == 0) {
        throw py::value_error(std::string(name) + " must be non-zero");
      }
      return handle;
    } catch (const py::cast_error&) {
      throw py::type_error(std::string(name) + " must be an integer handle");
    }
  }

  std::uint64_t PreparePut(std::vector<std::uint32_t> targets,
                           std::vector<RowCommit> commits) {
    std::set<std::uint32_t> expected;
    std::set<std::uint32_t> seen_rows;
    for (const RowCommit& commit : commits) {
      if (commit.appended_blocks == 0 ||
          commit.row_flat_base % geometry_.row_stride != 0) {
        throw py::value_error("DSA_ROW group Put row commit is malformed");
      }
      const std::uint32_t pool_row =
          commit.row_flat_base / geometry_.row_stride;
      if (pool_row >= rows_.size() || !seen_rows.insert(pool_row).second) {
        throw py::value_error("DSA_ROW group Put has duplicate or invalid row");
      }
      RowRecord& row = rows_[pool_row];
      if (row.state != RowState::kOpen || row.active_put_id != 0 ||
          commit.expected_blocks != row.committed_blocks ||
          commit.expected_blocks > row.max_blocks ||
          commit.appended_blocks > row.max_blocks - commit.expected_blocks) {
        throw py::value_error("DSA_ROW group Put disagrees with live row state");
      }
      for (std::uint32_t layer = 0; layer < geometry_.n_layers; ++layer) {
        for (std::uint32_t block = 0; block < commit.appended_blocks; ++block) {
          const std::uint32_t flat = commit.row_flat_base +
              layer * geometry_.q_capacity + commit.expected_blocks + block;
          if (cell_blocks_[flat] != kInvalidBlock) {
            throw py::value_error("DSA_ROW group Put would overwrite an owned cell");
          }
          expected.insert(flat);
        }
      }
    }
    const std::set<std::uint32_t> supplied(targets.begin(), targets.end());
    if (supplied.size() != targets.size() || supplied != expected) {
      throw py::value_error(
          "DSA_ROW group Put targets must exactly cover each layer/block cell");
    }
    if (free_blocks_.size() < targets.size()) {
      throw std::runtime_error("DSA_ROW DDR arena has insufficient free blocks");
    }
    const std::uint64_t handle = NextHandle();
    PutPlan plan;
    plan.commits = std::move(commits);
    plan.targets.reserve(targets.size());
    for (const std::uint32_t flat : targets) {
      const std::uint32_t block = free_blocks_.back();
      free_blocks_.pop_back();
      block_states_[block] = BlockState::kReserved;
      plan.target_positions.emplace(flat, plan.targets.size());
      plan.targets.push_back(PutTarget{flat, block, false});
      cell_blocks_[flat] = block;
    }
    for (const RowCommit& commit : plan.commits) {
      rows_[commit.row_flat_base / geometry_.row_stride].active_put_id = handle;
    }
    put_plans_.emplace(handle, std::move(plan));
    return handle;
  }

  void ValidateLayerBlocks(const PutPlan& plan, std::uint32_t layer,
                           const std::vector<LayerBlock>& blocks) const {
    std::set<std::uint32_t> expected;
    for (const PutTarget& target : plan.targets) {
      const std::uint32_t within_row = target.flat_index % geometry_.row_stride;
      if (within_row / geometry_.q_capacity == layer) {
        expected.insert(target.flat_index);
      }
    }
    std::set<std::uint32_t> supplied;
    for (const LayerBlock& block : blocks) {
      if (block.flat_index >= geometry_.directory_cells ||
          block.byte_length != layout_.layer_payload_bytes ||
          block.flat_index % geometry_.row_stride / geometry_.q_capacity != layer) {
        throw py::value_error("DSA_ROW layer Put block does not match its layer");
      }
      supplied.insert(block.flat_index);
    }
    if (supplied != expected) {
      throw py::value_error(
          "DSA_ROW layer Put must exactly cover every planned block for one layer");
    }
  }

  PutTarget& FindPutTarget(PutPlan& plan, std::uint32_t flat_index) {
    const auto it = plan.target_positions.find(flat_index);
    if (it == plan.target_positions.end()) {
      throw py::value_error("DSA_ROW Put target does not belong to the plan");
    }
    return plan.targets[it->second];
  }

  PutPlan& FindPutPlan(std::uint64_t handle) {
    const auto it = put_plans_.find(handle);
    if (it == put_plans_.end()) {
      throw py::value_error("DSA_ROW Put plan was not found");
    }
    return it->second;
  }

  AccessPlan& FindAccessPlan(std::uint64_t handle) {
    const auto it = accesses_.find(handle);
    if (it == accesses_.end()) {
      throw py::value_error("DSA_ROW unified access was not found");
    }
    return it->second;
  }

  GetPlan& FindGetPlan(std::uint64_t handle) {
    const auto it = get_plans_.find(handle);
    if (it == get_plans_.end()) {
      throw py::value_error("DSA_ROW DirectGet plan was not found");
    }
    return it->second;
  }

  void FinishGetPlan(std::uint64_t handle, bool /*success*/) {
    const auto get_it = get_plans_.find(handle);
    if (get_it == get_plans_.end()) return;
    const std::uint64_t access_handle = get_it->second.access_handle;
    const auto access_it = accesses_.find(access_handle);
    if (access_it != accesses_.end() && access_it->second.active_gets != 0) {
      --access_it->second.active_gets;
    }
    get_plans_.erase(get_it);
  }

  Address DdrBlockAddress(std::uint32_t block) const {
    if (block >= ddr_block_count_ ||
        MultiplyWouldOverflow(block, layout_.ddr_block_bytes) ||
        remote_region_.base > std::numeric_limits<Address>::max() -
                                  static_cast<Address>(block) *
                                      layout_.ddr_block_bytes) {
      throw std::runtime_error("DSA_ROW DDR block address overflow");
    }
    return remote_region_.base +
           static_cast<Address>(block) * layout_.ddr_block_bytes;
  }

  void CloseNoThrow() {
    // Shutdown must run before freeing Host DDR because Direct-HCCP owns
    // registrations against that exact region.  This path is intentionally
    // best-effort for destruction after an exception.
    if (access_operator_ != nullptr) {
      (void)access_operator_->Shutdown();
    }
    access_operator_.reset();
    if (allocator_ != nullptr && remote_region_.base != 0) {
      (void)allocator_->Release(remote_region_);
    }
    allocator_.reset();
    remote_region_ = RemoteRegionDesc{};
    device_directory_ = py::none();
    device_valid_ = py::none();
    bound_layers_.clear();
    put_plans_.clear();
    accesses_.clear();
    get_plans_.clear();
    free_blocks_.clear();
    block_states_.clear();
    cell_blocks_.clear();
    host_directory_.clear();
    host_valid_.clear();
    rows_.clear();
    initialized_ = false;
  }

  bool initialized_{false};
  std::uint32_t device_id_{0};
  std::uint32_t npu_core_count_{0};
  std::uint32_t jettys_per_core_{0};
  std::uint32_t ddr_block_count_{0};
  std::uint64_t next_handle_{1};
  Layout layout_;
  GeometryState geometry_;
  RemoteRegionDesc remote_region_;
  std::unique_ptr<AclHostDdrAllocator> allocator_;
  std::unique_ptr<DefaultAccessOperator> access_operator_;
  py::object device_directory_;
  py::object device_valid_;
  std::unordered_map<std::uint32_t, LayerBinding> bound_layers_;
  std::vector<RowRecord> rows_;
  std::vector<Address> host_directory_;
  std::vector<std::uint8_t> host_valid_;
  std::vector<std::uint32_t> cell_blocks_;
  std::vector<BlockState> block_states_;
  std::vector<std::uint32_t> free_blocks_;
  std::unordered_map<std::uint64_t, PutPlan> put_plans_;
  std::unordered_map<std::uint64_t, AccessPlan> accesses_;
  std::unordered_map<std::uint64_t, GetPlan> get_plans_;
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
      .def("batch_get", &NpuRuntime::BatchGet,
           py::arg("device_access_table"), py::arg("indices"), py::arg("mask"),
           py::arg("output_indices"), py::arg("output"),
           py::arg("stream") = py::none(), py::arg("synchronize") = false,
           "Submit BatchGet from NPU Tensors. stream=None uses torch.npu.current_stream(); "
           "the default returns after enqueueing work on that stream.");

#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  py::class_<DsaRowRuntime>(module, "DsaRowRuntime")
      .def(py::init<>())
      .def_static("probe_dsa_row_payload_abi",
                  &DsaRowRuntime::ProbeDsaRowPayloadAbi)
      .def("initialize_dsa_row_index", &DsaRowRuntime::InitializeDsaRowIndex,
           py::arg("layout"), py::arg("geometry"), py::arg("ddr_block_count"),
           py::arg("npu_config") = py::none(),
           "Initialize the production DSA_ROW Host-DDR arena and NPU directory.")
      .def("bind_resident_layer_cache", &DsaRowRuntime::BindResidentLayerCache,
           py::arg("layer_id"), py::arg("cache"), py::arg("byte_capacity"),
           py::arg("row_capacity"), py::arg("layout"))
      .def("bind_resident_pool_row", &DsaRowRuntime::BindResidentPoolRow,
           py::arg("pool_row"), py::arg("max_sequence_tokens"),
           py::arg("resident_budget_tokens"))
      .def("prepare_full_block_group_put",
           &DsaRowRuntime::PrepareFullBlockGroupPut, py::arg("all_targets"),
           py::arg("row_commits"))
      .def("write_full_block_group_layer",
           &DsaRowRuntime::WriteFullBlockGroupLayer, py::arg("plan"),
           py::arg("layer_blocks"), py::arg("layer_cache"),
           py::arg("stream") = py::none())
      .def("seal_full_block_group_put", &DsaRowRuntime::SealFullBlockGroupPut,
           py::arg("plan"))
      .def("commit_full_block_group_put", &DsaRowRuntime::CommitFullBlockGroupPut,
           py::arg("plan"))
      .def("abort_full_block_group_put", &DsaRowRuntime::AbortFullBlockGroupPut,
           py::arg("plan"))
      .def("prepare_unified_access", &DsaRowRuntime::PrepareUnifiedAccess,
           py::arg("rows"))
      .def("prepare_direct_batch_get", &DsaRowRuntime::PrepareDirectBatchGet,
           py::arg("access"), py::arg("items"), py::arg("output_cache"),
           py::arg("stream") = py::none())
      .def("prepare_direct_batch_get_device",
           &DsaRowRuntime::PrepareDirectBatchGetDevice, py::arg("access"),
           py::arg("flat_indices"), py::arg("token_offsets"), py::arg("mask"),
           py::arg("destination_indices"), py::arg("output_cache"),
           py::arg("stream") = py::none())
      .def("execute_prepared_direct_batch_get",
           &DsaRowRuntime::ExecutePreparedDirectBatchGet, py::arg("plan"))
      .def("settle_direct_batch_get", &DsaRowRuntime::SettleDirectBatchGet,
           py::arg("plan"), py::arg("completion"))
      .def("release_unified_access", &DsaRowRuntime::ReleaseUnifiedAccess,
           py::arg("access"))
      .def("close_resident_pool_row", &DsaRowRuntime::CloseResidentPoolRow,
           py::arg("pool_row"))
      .def("close", &DsaRowRuntime::Close)
      .def_property_readonly("initialized", &DsaRowRuntime::Initialized)
      .def_property_readonly("geometry", &DsaRowRuntime::Geometry);
#endif
}

}  // namespace kvaf
