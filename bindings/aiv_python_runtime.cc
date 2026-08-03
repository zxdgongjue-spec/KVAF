#include "aiv_python_runtime.h"

#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "batch_get_abi.h"
#include "kvaf/access_operator/aiv_batch_get_access_operator.h"
#include "kvaf/meta_manager/meta_manager.h"
#include "kvaf/runtime/runtime.h"

namespace py = pybind11;

namespace kvaf {
namespace {

Status InvalidArgument(std::string message) {
  return Status::Error(StatusCode::kInvalidArgument, std::move(message));
}

Status Internal(std::string message) {
  return Status::Error(StatusCode::kInternal, std::move(message));
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

std::vector<std::vector<std::uint8_t>> CopyPythonBlocks(
    const py::iterable& blocks) {
  std::vector<std::vector<std::uint8_t>> result;
  for (const py::handle& item : blocks) {
    if (!PyObject_CheckBuffer(item.ptr())) {
      throw py::type_error(
          "blocks must be bytes, bytearray, or another Python buffer object");
    }
    py::buffer buffer = py::reinterpret_borrow<py::buffer>(item);
    const py::buffer_info info = buffer.request();
    if (info.ptr == nullptr || info.size <= 0) {
      throw py::value_error("blocks must be non-empty");
    }
    const auto byte_count = static_cast<std::size_t>(info.size * info.itemsize);
    const auto* begin = static_cast<const std::uint8_t*>(info.ptr);
    result.emplace_back(begin, begin + byte_count);
  }
  return result;
}

void* PointerFromValue(std::uint64_t pointer) {
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer));
}

class AivPythonRuntime final {
 public:
  AivPythonRuntime(std::string kernel_object_path, std::uint32_t device_id,
                   std::uint32_t aiv_core_count)
      : device_id_(device_id), aiv_core_count_(aiv_core_count),
        meta_manager_(CreateInMemoryMetaManager()) {
    if (kernel_object_path.empty() || aiv_core_count == 0 ||
        aiv_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES) {
      throw py::value_error("invalid AIV BatchGet Runtime configuration");
    }

    // torch_npu owns process-wide ACL initialisation/finalisation.  The
    // binding therefore only selects the requested device and owns a private
    // stream; it never calls aclFinalize or aclrtResetDevice.
    ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");
    ThrowIfAclError(aclrtCreateStream(&stream_), "aclrtCreateStream");

    allocator_ = std::make_unique<AclHostDdrAllocator>();
    access_operator_ = std::make_unique<AivBatchGetAccessOperator>(
        AivBatchGetOperatorConfig{std::move(kernel_object_path), stream_,
                                  aiv_core_count_});
    runtime_ = std::make_unique<Runtime>(*meta_manager_, *access_operator_,
                                         *allocator_);
  }

  ~AivPythonRuntime() { CloseNoThrow(); }

  AivPythonRuntime(const AivPythonRuntime&) = delete;
  AivPythonRuntime& operator=(const AivPythonRuntime&) = delete;

  void Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
    ThrowIfError(runtime_->Initialize(
        RuntimeInitRequest{ddr_bytes, slot_size, DeviceKind::kCpu}));
  }

  bool Initialized() const {
    return runtime_ != nullptr && runtime_->initialized();
  }

  std::uint32_t DeviceId() const { return device_id_; }
  std::uint32_t AivCoreCount() const { return aiv_core_count_; }

  py::dict RemoteRegion() const {
    const RemoteRegionDesc& region = runtime_->remote_region();
    py::dict result;
    result["base"] = region.base;
    result["length"] = region.length;
    result["slot_size"] = region.slot_size;
    result["device"] = "cpu";
    return result;
  }

  // BatchPut is intentionally not exposed here: the current AIV demo has no
  // ExecuteBatchPut implementation.  This test-only seeding route writes the
  // Runtime-owned Host DDR after the MetaManager reservation and commit, so
  // BatchGet can be validated without pretending that BatchPut exists.
  void SeedFixedBlocks(const std::vector<KeyId>& keys, const py::iterable& blocks) {
    if (!Initialized()) {
      throw std::runtime_error("Runtime has not been initialized");
    }
    const std::vector<std::vector<std::uint8_t>> payloads =
        CopyPythonBlocks(blocks);
    if (keys.empty() || keys.size() != payloads.size()) {
      throw py::value_error("keys and blocks must be non-empty and equal in size");
    }

    std::vector<std::uint64_t> lengths;
    lengths.reserve(payloads.size());
    for (const auto& payload : payloads) {
      lengths.push_back(payload.size());
    }
    PutContext context;
    PutAccessDesc descriptor;
    ThrowIfError(meta_manager_->PreparePut(keys, lengths, &context, &descriptor));
    if (descriptor.entries.size() != payloads.size()) {
      (void)meta_manager_->AbortPut(context);
      throw std::runtime_error("MetaManager returned an invalid Put descriptor");
    }
    for (std::size_t index = 0; index < payloads.size(); ++index) {
      std::memcpy(reinterpret_cast<void*>(descriptor.entries[index].address),
                  payloads[index].data(), payloads[index].size());
    }
    const Status status = meta_manager_->CommitPut(context);
    if (!status.ok()) {
      (void)meta_manager_->AbortPut(context);
      ThrowIfError(status);
    }
  }

  AccessTableHandle PrepareAccessTable(const std::vector<KeyId>& keys) {
    AccessTableHandle handle;
    ThrowIfError(runtime_->PrepareAccessTable(AccessTablePrepareRequest{keys},
                                              &handle));
    return handle;
  }

  py::dict AccessTableView(const AccessTableHandle& handle) const {
    kvaf::AccessTableView view;
    ThrowIfError(runtime_->GetAccessTableView(handle, &view));
    py::list addresses;
    py::list lengths;
    py::list valid;
    for (std::size_t index = 0; index < view.size; ++index) {
      addresses.append(view.addresses[index]);
      lengths.append(view.lengths[index]);
      valid.append(view.valid[index] != 0);
    }
    py::dict result;
    result["addresses"] = std::move(addresses);
    result["lengths"] = std::move(lengths);
    result["valid"] = std::move(valid);
    return result;
  }

  void ReleaseAccessTable(const AccessTableHandle& handle) {
    ThrowIfError(runtime_->ReleaseAccessTable(handle));
  }

  void BindBatchGetOutput(std::uint64_t output_pointer,
                          std::size_t output_bytes,
                          std::uint64_t core_status_pointer,
                          std::size_t core_status_bytes,
                          std::uint64_t queue_state_pointer,
                          std::size_t queue_state_bytes) {
    ThrowIfError(access_operator_->BindBatchGetOutput(AivBatchGetOutputBinding{
        MutableBufferView{PointerFromValue(output_pointer), output_bytes,
                          DeviceKind::kNpu},
        MutableBufferView{PointerFromValue(core_status_pointer), core_status_bytes,
                          DeviceKind::kNpu},
        MutableBufferView{PointerFromValue(queue_state_pointer), queue_state_bytes,
                          DeviceKind::kNpu}}));
  }

  void UnbindBatchGetOutput() { ThrowIfError(access_operator_->UnbindBatchGetOutput()); }

  // All pointers here are supplied by the Python caller, normally as
  // ``int(tensor.data_ptr())`` from torch_npu tensors.  No tensor ownership is
  // transferred; the caller must keep every tensor alive until this method
  // returns and must keep output/status/queue alive until unbind/shutdown.
  double BatchGet(const AccessTableHandle& access_table,
                  std::uint64_t indices_pointer, std::size_t indices_bytes,
                  std::uint64_t mask_pointer, std::size_t mask_bytes,
                  std::uint64_t output_indices_pointer,
                  std::size_t output_indices_bytes,
                  std::uint64_t output_pointer, std::size_t output_bytes,
                  std::uint64_t output_block_bytes,
                  std::uint64_t address_table_pointer,
                  std::size_t address_table_bytes,
                  std::uint64_t valid_table_pointer,
                  std::size_t valid_table_bytes) {
    const BatchGetRequest request{
        access_table,
        ConstBufferView{PointerFromValue(indices_pointer), indices_bytes,
                        DeviceKind::kNpu},
        ConstBufferView{PointerFromValue(mask_pointer), mask_bytes,
                        DeviceKind::kNpu},
        ConstBufferView{PointerFromValue(output_indices_pointer),
                        output_indices_bytes, DeviceKind::kNpu},
        MutableBufferView{PointerFromValue(output_pointer), output_bytes,
                          DeviceKind::kNpu},
        output_block_bytes,
        AccessTableDeviceView{
            ConstBufferView{PointerFromValue(address_table_pointer),
                            address_table_bytes, DeviceKind::kNpu},
            ConstBufferView{PointerFromValue(valid_table_pointer),
                            valid_table_bytes, DeviceKind::kNpu}}};
    const auto begin = std::chrono::steady_clock::now();
    ThrowIfError(runtime_->BatchGet(request));
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
  }

  void Shutdown() {
    if (runtime_ != nullptr && runtime_->initialized()) {
      ThrowIfError(runtime_->Shutdown());
    }
    DestroyStream();
  }

 private:
  void DestroyStream() {
    if (stream_ != nullptr) {
      (void)aclrtDestroyStream(stream_);
      stream_ = nullptr;
    }
  }

  void CloseNoThrow() {
    if (runtime_ != nullptr && runtime_->initialized()) {
      (void)runtime_->Shutdown();
    }
    runtime_.reset();
    access_operator_.reset();
    allocator_.reset();
    DestroyStream();
  }

  std::uint32_t device_id_{0};
  std::uint32_t aiv_core_count_{0};
  aclrtStream stream_{nullptr};
  std::unique_ptr<MetaManager> meta_manager_;
  std::unique_ptr<AclHostDdrAllocator> allocator_;
  std::unique_ptr<AivBatchGetAccessOperator> access_operator_;
  std::unique_ptr<Runtime> runtime_;
};

}  // namespace

void BindAivBatchGetPythonRuntime(py::module_& module) {
  module.attr("AIV_BATCH_GET_STATUS_BYTES") = KVAF_AIV_BATCH_GET_STATUS_BYTES;
  module.attr("AIV_BATCH_GET_QUEUE_STATE_BYTES") =
      KVAF_AIV_BATCH_GET_QUEUE_STATE_BYTES;
  module.attr("AIV_BATCH_GET_STATUS_MAGIC") = KVAF_AIV_BATCH_GET_STATUS_MAGIC;

  py::class_<AivPythonRuntime>(module, "AivBatchGetRuntime")
      .def(py::init<std::string, std::uint32_t, std::uint32_t>(),
           py::arg("kernel_object_path"), py::arg("device_id") = 0,
           py::arg("aiv_core_count") = 1)
      .def("initialize", &AivPythonRuntime::Initialize, py::arg("ddr_bytes"),
           py::arg("slot_size"))
      .def("shutdown", &AivPythonRuntime::Shutdown)
      .def_property_readonly("initialized", &AivPythonRuntime::Initialized)
      .def_property_readonly("device_id", &AivPythonRuntime::DeviceId)
      .def_property_readonly("aiv_core_count", &AivPythonRuntime::AivCoreCount)
      .def_property_readonly("remote_region", &AivPythonRuntime::RemoteRegion)
      .def("seed_fixed_blocks", &AivPythonRuntime::SeedFixedBlocks,
           py::arg("keys"), py::arg("blocks"))
      .def("prepare_access_table", &AivPythonRuntime::PrepareAccessTable,
           py::arg("keys"))
      .def("access_table_view", &AivPythonRuntime::AccessTableView,
           py::arg("access_table"))
      .def("release_access_table", &AivPythonRuntime::ReleaseAccessTable,
           py::arg("access_table"))
      .def("bind_batch_get_output", &AivPythonRuntime::BindBatchGetOutput,
           py::arg("output_pointer"), py::arg("output_bytes"),
           py::arg("core_status_pointer"), py::arg("core_status_bytes"),
           py::arg("queue_state_pointer"), py::arg("queue_state_bytes"))
      .def("unbind_batch_get_output", &AivPythonRuntime::UnbindBatchGetOutput)
      .def("batch_get", &AivPythonRuntime::BatchGet,
           py::arg("access_table"), py::arg("indices_pointer"),
           py::arg("indices_bytes"), py::arg("mask_pointer"),
           py::arg("mask_bytes"), py::arg("output_indices_pointer"),
           py::arg("output_indices_bytes"), py::arg("output_pointer"),
           py::arg("output_bytes"), py::arg("output_block_bytes"),
           py::arg("address_table_pointer"), py::arg("address_table_bytes"),
           py::arg("valid_table_pointer"), py::arg("valid_table_bytes"),
           "Run synchronous Runtime.BatchGet and return call-to-stream-completion milliseconds.");
}

}  // namespace kvaf
