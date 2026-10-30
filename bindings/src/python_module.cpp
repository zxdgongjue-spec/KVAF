#include "hbm_read_python.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "access_table_python.h"
#include "access_operator.h"
#include "meta_manager.h"
#include "runtime.h"

#if defined(KVAF_ENABLE_AIV_BATCH_GET)
#include "acl_graph_python.h"
#endif
#if defined(KVAF_ENABLE_AIV_BATCH_GET) || defined(KVAF_ENABLE_CPU_OFFLOAD)
#include "npu_python_runtime.h"
#endif

namespace py = pybind11;

namespace kvaf {
namespace {

Status InvalidArgument(std::string message) {
  return Status::InvalidArgument(std::move(message));
}

Status Internal(std::string message) {
  return Status::Internal(std::move(message));
}

bool IsCContiguous(const py::buffer_info& info) {
  std::size_t expected_stride = static_cast<std::size_t>(info.itemsize);
  for (py::ssize_t dimension = info.ndim - 1; dimension >= 0; --dimension) {
    const py::ssize_t shape = info.shape[static_cast<std::size_t>(dimension)];
    if (shape < 0 ||
        info.strides[static_cast<std::size_t>(dimension)] !=
            static_cast<py::ssize_t>(expected_stride) ||
        static_cast<std::size_t>(shape) >
            std::numeric_limits<std::size_t>::max() / expected_stride) {
      return false;
    }
    expected_stride *= static_cast<std::size_t>(shape);
  }
  return true;
}

void ThrowIfError(const Status& status) {
  if (status.ok()) return;

  const std::string message = status.message().empty()
                                  ? "KVAF operation failed"
                                  : status.message();
  switch (status.code()) {
    case StatusCode::kInvalidArgument:
      throw py::value_error(message);
    case StatusCode::kNotFound:
      throw py::key_error(message);
    case StatusCode::kAlreadyExists:
    case StatusCode::kBusy:
    case StatusCode::kInvalidState:
      throw py::value_error(message);
    default:
      throw std::runtime_error(message);
  }
}

// This operator deliberately lives in the binding rather than the Runtime
// library: production callers inject a hardware operator, while Python unit
// tests need a deterministic, in-process CPU data plane.
class MockAccessOperator final : public AccessOperator {
 public:
  Status Initialize(const RemoteRegionDesc& remote_region) override {
    if (initialized_) {
      return Status::Error(StatusCode::kInvalidState,
                           "mock operator is already initialized");
    }
    if (remote_region.base == 0 || remote_region.length == 0) {
      return InvalidArgument("mock operator received an invalid remote region");
    }
    initialized_ = true;
    return Status::OK();
  }

  Status Shutdown() override {
    if (!initialized_) {
      return Status::Error(StatusCode::kNotInitialized,
                           "mock operator is not initialized");
    }
    initialized_ = false;
    return Status::OK();
  }

  Status ExecuteBatchGet(const BatchGetExecRequest& exec) override {
    if (fail_next_get_) {
      fail_next_get_ = false;
      return Internal("configured mock BatchGet failure");
    }
    if (!initialized_ || exec.request == nullptr ||
        exec.host_access_table.addresses == nullptr ||
        exec.host_access_table.lengths == nullptr ||
        exec.host_access_table.valid == nullptr) {
      return InvalidArgument("invalid mock BatchGet request");
    }

    const BatchGetRequest& request = *exec.request;
    if (request.indices.device != DeviceKind::kCpu ||
        request.offsets.device != DeviceKind::kCpu ||
        request.output_indices.device != DeviceKind::kCpu ||
        request.output.device != DeviceKind::kCpu) {
      return InvalidArgument("mock operator supports CPU buffers only");
    }

    const auto capacity = request.indices.bytes / sizeof(std::int32_t);
    const std::int32_t miss_count =
        *static_cast<const std::int32_t*>(request.miss_count.data);
    if (miss_count < 0 || static_cast<std::size_t>(miss_count) > capacity) {
      return InvalidArgument("BatchGet miss_count is outside the request capacity");
    }
    const auto count = static_cast<std::size_t>(miss_count);
    const auto* indices = static_cast<const std::int32_t*>(request.indices.data);
    const auto* offsets = static_cast<const std::int32_t*>(request.offsets.data);
    const auto* output_indices =
        static_cast<const std::int32_t*>(request.output_indices.data);
    auto* output = static_cast<std::uint8_t*>(request.output.data);
    const std::size_t output_rows =
        request.output.bytes / request.output_block_bytes;

    for (std::size_t i = 0; i < count; ++i) {
      if (indices[i] < 0 ||
          static_cast<std::size_t>(indices[i]) >= exec.host_access_table.size ||
          exec.host_access_table.valid[indices[i]] == 0) {
        return InvalidArgument("BatchGet index is outside the valid AccessTable");
      }
      if (output_indices[i] < 0 ||
          static_cast<std::size_t>(output_indices[i]) >= output_rows) {
        return InvalidArgument("BatchGet output index is outside the output buffer");
      }

      const std::size_t table_index = static_cast<std::size_t>(indices[i]);
      const std::uint64_t length = exec.host_access_table.lengths[table_index];
      if (offsets[i] < 0) {
        return InvalidArgument("BatchGet offset is outside the AccessTable entry");
      }
      const std::uint64_t offset_bytes =
          static_cast<std::uint64_t>(offsets[i]);
      if (offset_bytes >= length) {
        return InvalidArgument("BatchGet offset is outside source data");
      }
      const std::uint64_t copy_bytes =
          std::min(length - offset_bytes, request.output_block_bytes);
      std::uint8_t* destination =
          output + static_cast<std::size_t>(output_indices[i]) *
                       request.output_block_bytes;
      std::memcpy(destination,
                  reinterpret_cast<const void*>(
                      exec.host_access_table.addresses[table_index] +
                      offset_bytes),
                  static_cast<std::size_t>(copy_bytes));
    }
    return Status::OK();
  }

  Status ExecuteBatchPut(const BatchPutExecRequest& exec) override {
    if (fail_next_put_) {
      fail_next_put_ = false;
      return Internal("configured mock BatchPut failure");
    }
    if (!initialized_ || exec.request == nullptr || exec.put_desc == nullptr ||
        exec.request->input.data == nullptr || exec.request->input.bytes == 0 ||
        exec.request->keys.size() != exec.request->input_offsets.size() ||
        exec.request->keys.size() != exec.request->lengths.size() ||
        exec.request->keys.size() != exec.put_desc->entries.size()) {
      return InvalidArgument("invalid mock BatchPut request");
    }

    const auto* input = static_cast<const std::uint8_t*>(exec.request->input.data);
    for (std::size_t i = 0; i < exec.request->keys.size(); ++i) {
      const std::uint64_t offset = exec.request->input_offsets[i];
      const std::uint64_t length = exec.request->lengths[i];
      const PutAllocation& target = exec.put_desc->entries[i];
      if (exec.request->input.device != DeviceKind::kCpu || length == 0 ||
          offset > exec.request->input.bytes ||
          length > exec.request->input.bytes - offset ||
          length != target.reserved_length ||
          target.key != exec.request->keys[i]) {
        return InvalidArgument("invalid mock BatchPut block");
      }
      std::memcpy(reinterpret_cast<void*>(target.address),
                  input + static_cast<std::size_t>(offset),
                  static_cast<std::size_t>(length));
    }
    return Status::OK();
  }

  void FailNextPut() { fail_next_put_ = true; }
  void FailNextGet() { fail_next_get_ = true; }

 private:
  bool initialized_{false};
  bool fail_next_put_{false};
  bool fail_next_get_{false};
};

class PythonRuntime final {
 public:
  PythonRuntime()
      : meta_manager_(CreateInMemoryMetaManager()),
        allocator_(CreateHostDdrMemoryAllocator()),
        access_operator_(std::make_unique<MockAccessOperator>()),
        runtime_(std::make_unique<RuntimeCore>(*meta_manager_, *access_operator_,
                                               *allocator_)) {}

  void Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
    ThrowIfError(runtime_->Initialize(
        RuntimeInitRequest{ddr_bytes, slot_size, DeviceKind::kCpu}));
  }

  void Shutdown() { ThrowIfError(runtime_->Shutdown()); }

  bool Initialized() const { return runtime_->initialized(); }

  py::dict RemoteRegion() const {
    const RemoteRegionDesc& region = runtime_->remote_region();
    py::dict result;
    result["base"] = region.base;
    result["length"] = region.length;
    result["slot_size"] = region.slot_size;
    result["device"] = "cpu";
    return result;
  }

  // input_offsets and lengths are host metadata for the caller-owned packed
  // input buffer. Runtime uses lengths for PreparePut; hardware operators can
  // upload the same arrays as their per-request execution descriptors.
  void BatchPut(const std::vector<KeyId>& keys, const py::buffer& input,
                const std::vector<std::uint64_t>& input_offsets,
                const std::vector<std::uint64_t>& lengths) {
    const py::buffer_info info = input.request();
    if (info.ptr == nullptr || info.size <= 0 || info.itemsize <= 0 ||
        !IsCContiguous(info) ||
        static_cast<std::size_t>(info.size) >
            std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(info.itemsize)) {
      throw py::value_error("BatchPut input must be a non-empty contiguous buffer");
    }
    ThrowIfError(runtime_->BatchPut(BatchPutRequest{
        keys,
        ConstBufferView{info.ptr,
                        static_cast<std::size_t>(info.size) *
                            static_cast<std::size_t>(info.itemsize),
                        DeviceKind::kCpu},
        input_offsets,
        lengths}));
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
    return python_binding::MakeAccessTableView(view);
  }

  void ReleaseAccessTable(const AccessTableHandle& handle) {
    ThrowIfError(runtime_->ReleaseAccessTable(handle));
  }

  void BatchGet(const AccessTableHandle& handle,
                const std::vector<std::int32_t>& indices,
                const std::vector<std::int32_t>& offsets,
                std::int32_t miss_count,
                const std::vector<std::int32_t>& output_indices,
                py::bytearray output, std::uint64_t output_block_bytes) {
    if (indices.empty() || indices.size() != offsets.size() ||
        indices.size() != output_indices.size()) {
      throw py::value_error(
          "BatchGet indices, offsets, and output_indices must be non-empty and equal in size");
    }
    if (output_block_bytes == 0) {
      throw py::value_error("BatchGet output_block_bytes must be non-zero");
    }
    if (miss_count < 0 ||
        static_cast<std::size_t>(miss_count) > indices.size()) {
      throw py::value_error(
          "BatchGet miss_count must be between zero and the request capacity");
    }

    char* output_data = PyByteArray_AsString(output.ptr());
    const Py_ssize_t output_size = PyByteArray_Size(output.ptr());
    if (output_data == nullptr || output_size <= 0) {
      throw py::value_error("BatchGet output must be a non-empty bytearray");
    }
    const BatchGetRequest request{
        handle,
        ConstBufferView{indices.data(),
                        indices.size() * sizeof(indices.front()),
                        DeviceKind::kCpu},
        ConstBufferView{offsets.data(),
                        offsets.size() * sizeof(offsets.front()),
                        DeviceKind::kCpu},
        ConstBufferView{&miss_count, sizeof(miss_count), DeviceKind::kCpu},
        ConstBufferView{output_indices.data(),
                        output_indices.size() * sizeof(output_indices.front()),
                        DeviceKind::kCpu},
        MutableBufferView{output_data, static_cast<std::size_t>(output_size),
                          DeviceKind::kCpu},
        output_block_bytes,
        0,
        AccessTableDeviceView{}};
    ThrowIfError(runtime_->BatchGet(request));
  }

  void BatchFree(const std::vector<KeyId>& keys) {
    ThrowIfError(runtime_->BatchFree(BatchFreeRequest{keys}));
  }

  void FailNextPut() { access_operator_->FailNextPut(); }
  void FailNextGet() { access_operator_->FailNextGet(); }

 private:
  // Runtime must be destroyed first: its destructor can call Shutdown(), which
  // still needs all three injected dependencies to be alive.
  std::unique_ptr<MetaManager> meta_manager_;
  std::unique_ptr<DdrMemoryAllocator> allocator_;
  std::unique_ptr<MockAccessOperator> access_operator_;
  std::unique_ptr<RuntimeCore> runtime_;
};

}  // namespace
}  // namespace kvaf

#if defined(KVAF_ENABLE_CPU_OFFLOAD)
namespace kvaf { void BindOffloadRuntime(pybind11::module_&); void BindHbmSource(pybind11::module_&); }
#endif

PYBIND11_MODULE(_kvaf, m) {
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  kvaf::BindOffloadRuntime(m);
  kvaf::BindHbmSource(m);
#endif
  m.doc() = "Synchronous KVAF Runtime Python bindings";

  py::class_<kvaf::AccessTableHandle>(m, "AccessTable")
      .def_property_readonly("id", [](const kvaf::AccessTableHandle& handle) {
        return handle.opaque;
      })
      .def("__repr__", [](const kvaf::AccessTableHandle& handle) {
        return "AccessTable(id=" + std::to_string(handle.opaque) + ")";
      });

  py::class_<kvaf::PythonRuntime>(m, "Runtime")
      .def(py::init<>())
      .def("initialize", &kvaf::PythonRuntime::Initialize, py::arg("ddr_bytes"),
           py::arg("slot_size"),
           "Allocate Runtime-owned host DDR and initialize the mock operator.")
      .def("shutdown", &kvaf::PythonRuntime::Shutdown)
      .def_property_readonly("initialized", &kvaf::PythonRuntime::Initialized)
      .def_property_readonly("remote_region", &kvaf::PythonRuntime::RemoteRegion)
      .def("batch_put", &kvaf::PythonRuntime::BatchPut, py::arg("keys"),
           py::arg("input"), py::arg("input_offsets"), py::arg("lengths"))
      .def("prepare_access_table", &kvaf::PythonRuntime::PrepareAccessTable,
           py::arg("keys"))
      .def("access_table_view", &kvaf::PythonRuntime::AccessTableView,
           py::arg("access_table"))
      .def("release_access_table", &kvaf::PythonRuntime::ReleaseAccessTable,
           py::arg("access_table"))
      .def("batch_get", &kvaf::PythonRuntime::BatchGet,
           py::arg("access_table"), py::arg("indices"),
           py::arg("offsets"), py::arg("miss_count"),
           py::arg("output_indices"), py::arg("output"),
           py::arg("output_block_bytes"))
      .def("batch_free", &kvaf::PythonRuntime::BatchFree, py::arg("keys"))
      .def("_fail_next_put", &kvaf::PythonRuntime::FailNextPut,
           "Test hook: make the next BatchPut fail before Meta commit.")
      .def("_fail_next_get", &kvaf::PythonRuntime::FailNextGet,
           "Test hook: make the next BatchGet fail.");

  m.def("create_mock_runtime", [] { return kvaf::PythonRuntime(); },
        "Create an in-process Runtime with the deterministic CPU mock operator.");
  m.attr("MockRuntime") = m.attr("Runtime");

  py::module_ runtime = m.def_submodule("runtime");
  runtime.attr("Runtime") = m.attr("Runtime");
  runtime.attr("MockRuntime") = m.attr("Runtime");
  runtime.attr("create_mock_runtime") = m.attr("create_mock_runtime");
  m.def_submodule("meta_manager");
  m.def_submodule("access_operator");

#if defined(KVAF_ENABLE_AIV_BATCH_GET) || defined(KVAF_ENABLE_CPU_OFFLOAD)
  kvaf::BindNpuPythonRuntime(m);
  kvaf::BindHbmReadSource(m);
#endif
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  kvaf::BindAclGraph(m);
#endif
}
