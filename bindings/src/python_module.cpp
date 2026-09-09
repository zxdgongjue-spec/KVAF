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

#if defined(KVAF_BUILD_REMOTE_CPU_SERVICE)
#include "cpu_runtime.h"
#endif

#if defined(KVAF_ENABLE_AIV_BATCH_GET)
#include "acl_graph_python.h"
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

    const auto count = request.indices.bytes / sizeof(std::int32_t);
    const auto* indices = static_cast<const std::int32_t*>(request.indices.data);
    const auto* offsets = static_cast<const std::int32_t*>(request.offsets.data);
    const auto* output_indices =
        static_cast<const std::int32_t*>(request.output_indices.data);
    const auto* mask = request.mask.data == nullptr
                           ? nullptr
                           : static_cast<const std::uint8_t*>(request.mask.data);
    auto* output = static_cast<std::uint8_t*>(request.output.data);
    const std::size_t output_rows =
        request.output.bytes / request.output_block_bytes;

    for (std::size_t i = 0; i < count; ++i) {
      if (mask != nullptr && mask[i] == 0) continue;
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
    if (!initialized_ || exec.plan == nullptr || exec.put_desc == nullptr ||
        exec.plan->flat_keys.empty() ||
        exec.plan->flat_keys.size() != exec.plan->source_indices.size() ||
        exec.plan->flat_keys.size() != exec.plan->offsets.size() ||
        exec.plan->flat_keys.size() != exec.plan->lengths.size() ||
        exec.plan->flat_keys.size() != exec.put_desc->entries.size()) {
      return InvalidArgument("invalid mock BatchPut request");
    }

    for (std::size_t i = 0; i < exec.plan->flat_keys.size(); ++i) {
      const std::uint32_t source_index = exec.plan->source_indices[i];
      if (source_index >= exec.plan->sources.size()) {
        return InvalidArgument("invalid mock BatchPut source index");
      }
      const ConstBufferView& source = exec.plan->sources[source_index];
      const auto* input = static_cast<const std::uint8_t*>(source.data);
      const std::uint64_t offset = exec.plan->offsets[i];
      const std::uint64_t length = exec.plan->lengths[i];
      const PutAllocation& target = exec.put_desc->entries[i];
      if (source.device != DeviceKind::kCpu || length == 0 ||
          offset > source.bytes || length > source.bytes - offset ||
          length != target.reserved_length ||
          target.key != exec.plan->flat_keys[i]) {
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

  void BatchPut(std::vector<std::vector<KeyId>> keys,
                const py::list& inputs,
                std::vector<std::vector<std::uint64_t>> input_offsets,
                std::vector<std::vector<std::uint64_t>> lengths,
                py::object miss_count) {
    if (keys.empty() || keys.size() != static_cast<std::size_t>(inputs.size()) ||
        keys.size() != input_offsets.size() || keys.size() != lengths.size()) {
      throw py::value_error(
          "BatchPut keys, inputs, input_offsets, and lengths must have equal outer sizes");
    }

    BatchPutRequest request;
    request.groups.reserve(keys.size());
    for (std::size_t group_index = 0; group_index < keys.size(); ++group_index) {
      if (keys[group_index].empty() ||
          keys[group_index].size() != input_offsets[group_index].size() ||
          keys[group_index].size() != lengths[group_index].size()) {
        throw py::value_error(
            "BatchPut group keys, input_offsets, and lengths must be non-empty and equal in size");
      }
      py::buffer input =
          py::reinterpret_borrow<py::buffer>(inputs[group_index]);
      const py::buffer_info info = input.request();
      if (info.ptr == nullptr || info.size <= 0 || info.itemsize <= 0 ||
          !IsCContiguous(info) ||
          static_cast<std::size_t>(info.size) >
              std::numeric_limits<std::size_t>::max() /
                  static_cast<std::size_t>(info.itemsize)) {
        throw py::value_error(
            "BatchPut input must be a non-empty contiguous buffer");
      }
      request.groups.push_back(BatchPutGroup{
          std::move(keys[group_index]),
          ConstBufferView{info.ptr,
                          static_cast<std::size_t>(info.size) *
                              static_cast<std::size_t>(info.itemsize),
                          DeviceKind::kCpu},
          std::move(input_offsets[group_index]),
          std::move(lengths[group_index])});
    }
    if (!miss_count.is_none()) {
      const std::int64_t count = miss_count.cast<std::int64_t>();
      if (count < 0) {
        throw py::value_error("BatchPut miss_count must be non-negative");
      }
      request.miss_count = static_cast<std::size_t>(count);
    }
    ThrowIfError(runtime_->BatchPut(request));
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
                const std::vector<std::int32_t>& output_indices,
                py::bytearray output, std::uint64_t output_block_bytes,
                py::object mask) {
    if (indices.empty() || indices.size() != offsets.size() ||
        indices.size() != output_indices.size()) {
      throw py::value_error(
          "BatchGet indices, offsets, and output_indices must be non-empty and equal in size");
    }
    if (output_block_bytes == 0) {
      throw py::value_error("BatchGet output_block_bytes must be non-zero");
    }

    std::vector<std::uint8_t> mask_values;
    if (!mask.is_none()) {
      mask_values = mask.cast<std::vector<std::uint8_t>>();
      if (mask_values.size() != indices.size()) {
        throw py::value_error("BatchGet mask must have one value per index");
      }
    }

    char* output_data = PyByteArray_AsString(output.ptr());
    const Py_ssize_t output_size = PyByteArray_Size(output.ptr());
    if (output_data == nullptr || output_size <= 0) {
      throw py::value_error("BatchGet output must be a non-empty bytearray");
    }
    const ConstBufferView mask_view = mask_values.empty()
                                          ? ConstBufferView{}
                                          : ConstBufferView{mask_values.data(),
                                                            mask_values.size(),
                                                            DeviceKind::kCpu};
    const BatchGetRequest request{
        handle,
        ConstBufferView{indices.data(),
                        indices.size() * sizeof(indices.front()),
                        DeviceKind::kCpu},
        ConstBufferView{offsets.data(),
                        offsets.size() * sizeof(offsets.front()),
                        DeviceKind::kCpu},
        mask_view,
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

#if defined(KVAF_BUILD_REMOTE_CPU_SERVICE)
class CpuPythonRuntime final {
 public:
  void Initialize(const std::string& peer, std::uint64_t hbm_bytes,
                  std::uint64_t slot_size, std::uint32_t device_id,
                  const std::string& ub_device, py::object eid_index,
                  std::uint32_t jetty_count, std::uint32_t omp_thread_count,
                  std::uint32_t wqes_per_doorbell, std::uint32_t sgl_size,
                  std::uint32_t remote_sgl_size) {
    CpuRuntimeClientOptions options;
    options.peer_address = peer;
    options.hbm_bytes = hbm_bytes;
    options.slot_size = slot_size;
    options.device_id = device_id;
    options.ub_device = ub_device;
    options.jetty_count = jetty_count;
    options.omp_thread_count = omp_thread_count;
    options.wqes_per_doorbell = wqes_per_doorbell;
    options.sgl_size = sgl_size;
    options.remote_sgl_size = remote_sgl_size;
    if (!eid_index.is_none()) {
      options.eid_index = eid_index.cast<std::uint32_t>();
      options.has_eid_index = true;
    }
    ThrowIfError(native_.InitializeClient(options));
  }

  void Shutdown() { ThrowIfError(native_.Shutdown()); }
  bool Initialized() const { return native_.initialized(); }

  py::dict RemoteRegion() const {
    const RemoteRegionDesc& region = native_.core().remote_region();
    py::dict result;
    result["base"] = region.base;
    result["length"] = region.length;
    result["slot_size"] = region.slot_size;
    result["device"] = "npu";
    return result;
  }

  py::dict LastBatchTiming() const {
    const CpuRuntimeBatchTiming timing = native_.last_batch_timing();
    py::dict result;
    result["prepare_ns"] = timing.prepare_ns;
    result["post_ns"] = timing.post_ns;
    result["poll_ns"] = timing.poll_ns;
    result["total_ns"] = timing.total_ns;
    result["total_scope"] = "transport_call";
    result["wr_count"] = timing.wr_count;
    return result;
  }

  void BatchPut(std::vector<std::vector<KeyId>> keys, const py::list& inputs,
                std::vector<std::vector<std::uint64_t>> input_offsets,
                std::vector<std::vector<std::uint64_t>> lengths,
                py::object miss_count) {
    if (keys.empty() || keys.size() != static_cast<std::size_t>(inputs.size()) ||
        keys.size() != input_offsets.size() || keys.size() != lengths.size()) {
      throw py::value_error("BatchPut arguments must have equal outer sizes");
    }
    BatchPutRequest request;
    request.groups.reserve(keys.size());
    for (std::size_t group = 0; group < keys.size(); ++group) {
      if (keys[group].empty() || keys[group].size() != input_offsets[group].size() ||
          keys[group].size() != lengths[group].size()) {
        throw py::value_error("BatchPut group arguments must have equal non-empty sizes");
      }
      py::buffer input = py::reinterpret_borrow<py::buffer>(inputs[group]);
      const py::buffer_info info = input.request();
      if (info.ptr == nullptr || info.size <= 0 || info.itemsize <= 0 ||
          !IsCContiguous(info)) {
        throw py::value_error("BatchPut input must be a non-empty contiguous buffer");
      }
      const std::size_t bytes = static_cast<std::size_t>(info.size) *
                                static_cast<std::size_t>(info.itemsize);
      request.groups.push_back(BatchPutGroup{
          std::move(keys[group]),
          ConstBufferView{info.ptr, bytes, DeviceKind::kCpu},
          std::move(input_offsets[group]), std::move(lengths[group])});
    }
    if (!miss_count.is_none()) {
      const std::int64_t count = miss_count.cast<std::int64_t>();
      if (count < 0) {
        throw py::value_error("BatchPut miss_count must be non-negative");
      }
      request.miss_count = static_cast<std::size_t>(count);
    }
    ThrowIfError(native_.core().BatchPut(request));
  }

  AccessTableHandle PrepareAccessTable(const std::vector<KeyId>& keys) {
    AccessTableHandle handle;
    ThrowIfError(native_.core().PrepareAccessTable(
        AccessTablePrepareRequest{keys}, &handle));
    return handle;
  }

  py::dict AccessTableView(const AccessTableHandle& handle) const {
    kvaf::AccessTableView view;
    ThrowIfError(native_.core().GetAccessTableView(handle, &view));
    return python_binding::MakeAccessTableView(view);
  }

  void ReleaseAccessTable(const AccessTableHandle& handle) {
    ThrowIfError(native_.core().ReleaseAccessTable(handle));
  }

  void BatchGet(const AccessTableHandle& handle,
                const std::vector<std::int32_t>& indices,
                const std::vector<std::int32_t>& offsets,
                const std::vector<std::int32_t>& output_indices,
                py::bytearray output, std::uint64_t output_block_bytes,
                py::object mask) {
    if (indices.empty() || indices.size() != offsets.size() ||
        indices.size() != output_indices.size() || output_block_bytes == 0U) {
      throw py::value_error("BatchGet arguments are invalid or have mismatched sizes");
    }
    std::vector<std::uint8_t> mask_values;
    if (!mask.is_none()) {
      mask_values = mask.cast<std::vector<std::uint8_t>>();
      if (mask_values.size() != indices.size()) {
        throw py::value_error("BatchGet mask must match indices size");
      }
    }
    char* output_data = PyByteArray_AsString(output.ptr());
    const Py_ssize_t output_size = PyByteArray_Size(output.ptr());
    if (output_data == nullptr || output_size <= 0) {
      throw py::value_error("BatchGet output must be a non-empty bytearray");
    }
    const ConstBufferView mask_view = mask_values.empty()
        ? ConstBufferView{}
        : ConstBufferView{mask_values.data(), mask_values.size(), DeviceKind::kCpu};
    const BatchGetRequest request{
        handle,
        ConstBufferView{indices.data(), indices.size() * sizeof(std::int32_t), DeviceKind::kCpu},
        ConstBufferView{offsets.data(), offsets.size() * sizeof(std::int32_t), DeviceKind::kCpu},
        mask_view,
        ConstBufferView{output_indices.data(), output_indices.size() * sizeof(std::int32_t), DeviceKind::kCpu},
        MutableBufferView{output_data, static_cast<std::size_t>(output_size), DeviceKind::kCpu},
        output_block_bytes, 0, AccessTableDeviceView{}};
    ThrowIfError(native_.core().BatchGet(request));
  }

  void BatchFree(const std::vector<KeyId>& keys) {
    ThrowIfError(native_.core().BatchFree(BatchFreeRequest{keys}));
  }

 private:
  CpuRuntime native_;
};
#endif

}  // namespace
}  // namespace kvaf

PYBIND11_MODULE(_kvaf, m) {
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
           py::arg("input"), py::arg("input_offsets"), py::arg("lengths"),
           py::arg("miss_count") = py::none())
      .def("prepare_access_table", &kvaf::PythonRuntime::PrepareAccessTable,
           py::arg("keys"))
      .def("access_table_view", &kvaf::PythonRuntime::AccessTableView,
           py::arg("access_table"))
      .def("release_access_table", &kvaf::PythonRuntime::ReleaseAccessTable,
           py::arg("access_table"))
      .def("batch_get", &kvaf::PythonRuntime::BatchGet,
           py::arg("access_table"), py::arg("indices"),
           py::arg("offsets"),
           py::arg("output_indices"), py::arg("output"),
           py::arg("output_block_bytes"), py::arg("mask") = py::none())
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

#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  kvaf::BindNpuPythonRuntime(m);
  kvaf::BindAclGraph(m);
#endif
#if defined(KVAF_BUILD_REMOTE_CPU_SERVICE)
  py::class_<kvaf::CpuPythonRuntime>(m, "CpuRuntime")
      .def(py::init<>())
      .def("initialize", &kvaf::CpuPythonRuntime::Initialize,
           py::arg("peer"), py::arg("hbm_bytes"), py::arg("slot_size"),
           py::arg("device_id") = 0, py::arg("ub_device") = "",
           py::arg("eid_index") = py::none(), py::arg("jetty_count") = 1,
           py::arg("omp_thread_count") = 1,
           py::arg("wqes_per_doorbell") = 0, py::arg("sgl_size") = 1,
           py::arg("remote_sgl_size") = 1)
      .def("shutdown", &kvaf::CpuPythonRuntime::Shutdown)
      .def_property_readonly("initialized", &kvaf::CpuPythonRuntime::Initialized)
      .def_property_readonly("remote_region", &kvaf::CpuPythonRuntime::RemoteRegion)
      .def_property_readonly("last_batch_timing",
                             &kvaf::CpuPythonRuntime::LastBatchTiming)
      .def("batch_put", &kvaf::CpuPythonRuntime::BatchPut,
           py::arg("keys"), py::arg("input"), py::arg("input_offsets"),
           py::arg("lengths"), py::arg("miss_count") = py::none())
      .def("prepare_access_table", &kvaf::CpuPythonRuntime::PrepareAccessTable,
           py::arg("keys"))
      .def("access_table_view", &kvaf::CpuPythonRuntime::AccessTableView,
           py::arg("access_table"))
      .def("release_access_table", &kvaf::CpuPythonRuntime::ReleaseAccessTable,
           py::arg("access_table"))
      .def("batch_get", &kvaf::CpuPythonRuntime::BatchGet,
           py::arg("access_table"), py::arg("indices"), py::arg("offsets"),
           py::arg("output_indices"), py::arg("output"),
           py::arg("output_block_bytes"), py::arg("mask") = py::none())
      .def("batch_free", &kvaf::CpuPythonRuntime::BatchFree, py::arg("keys"));
#endif
}
