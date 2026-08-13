#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "access_table_python.h"
#include "kvaf/access_operator/access_operator.h"
#include "kvaf/meta_manager/meta_manager.h"
#include "kvaf/runtime/runtime.h"

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
        request.output_indices.device != DeviceKind::kCpu ||
        request.output.device != DeviceKind::kCpu) {
      return InvalidArgument("mock operator supports CPU buffers only");
    }

    const auto count = request.indices.bytes / sizeof(std::int32_t);
    const auto* indices = static_cast<const std::int32_t*>(request.indices.data);
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
      if (length > request.output_block_bytes) {
        return InvalidArgument("BatchGet output block is smaller than source data");
      }
      std::uint8_t* destination =
          output + static_cast<std::size_t>(output_indices[i]) *
                       request.output_block_bytes;
      std::memcpy(destination,
                  reinterpret_cast<const void*>(
                      exec.host_access_table.addresses[table_index]),
                  static_cast<std::size_t>(length));
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
        runtime_(std::make_unique<Runtime>(*meta_manager_, *access_operator_,
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
                const std::vector<std::int32_t>& output_indices,
                py::bytearray output, std::uint64_t output_block_bytes,
                py::object mask) {
    if (indices.empty() || indices.size() != output_indices.size()) {
      throw py::value_error(
          "BatchGet indices and output_indices must be non-empty and equal in size");
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
        mask_view,
        ConstBufferView{output_indices.data(),
                        output_indices.size() * sizeof(output_indices.front()),
                        DeviceKind::kCpu},
        MutableBufferView{output_data, static_cast<std::size_t>(output_size),
                          DeviceKind::kCpu},
        output_block_bytes,
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
  std::unique_ptr<Runtime> runtime_;
};

// A small CPU-only facade for exercising the public DSA_ROW Runtime API from
// Python.  It deliberately takes/returns simple Python primitives; production
// torch/NPU bindings can call the same C++ Runtime methods with device views.
class DsaRowMockPythonRuntime final {
 public:
  DsaRowMockPythonRuntime()
      : meta_manager_(CreateInMemoryMetaManager()),
        allocator_(CreateHostDdrMemoryAllocator()),
        access_operator_(std::make_unique<MockAccessOperator>()),
        runtime_(std::make_unique<Runtime>(*meta_manager_, *access_operator_,
                                           *allocator_)) {}

  void Initialize(std::uint64_t ddr_bytes, std::uint32_t raw_row_capacity,
                  std::uint32_t n_layers, std::uint32_t q_capacity,
                  std::uint32_t tokens_per_logical_block = 128,
                  std::uint32_t packed_token_bytes = 656,
                  std::uint32_t layout_id = 1,
                  std::uint32_t ddr_block_count = 0,
                  std::uint32_t ddr_block_alignment_bytes = 32) {
    if (tokens_per_logical_block == 0 || packed_token_bytes == 0 ||
        tokens_per_logical_block >
            std::numeric_limits<std::uint64_t>::max() / packed_token_bytes) {
      throw py::value_error("invalid DSA_ROW packed layout");
    }
    if (ddr_block_alignment_bytes == 0) {
      throw py::value_error("DSA_ROW DDR block alignment must be non-zero");
    }
    const std::uint64_t payload_bytes =
        static_cast<std::uint64_t>(tokens_per_logical_block) * packed_token_bytes;
    const std::uint64_t remainder = payload_bytes % ddr_block_alignment_bytes;
    const std::uint64_t block_bytes =
        remainder == 0 ? payload_bytes
                       : payload_bytes + (ddr_block_alignment_bytes - remainder);
    ThrowIfError(runtime_->Initialize(
        RuntimeInitRequest{ddr_bytes, block_bytes, DeviceKind::kCpu}));
    const DsaRowIndexGeometry geometry{raw_row_capacity, n_layers, q_capacity,
                                       0, 0};
    const DsaRowInitRequest request{
        PackedKvLayout{layout_id, tokens_per_logical_block, packed_token_bytes,
                       ddr_block_alignment_bytes},
        geometry, ddr_block_count};
    const Status row_status = runtime_->InitializeDsaRowIndex(request);
    if (!row_status.ok()) {
      (void)runtime_->Shutdown();
      ThrowIfError(row_status);
    }
    block_payload_bytes_ = payload_bytes;
    ddr_block_bytes_ = block_bytes;
    tokens_per_logical_block_ = tokens_per_logical_block;
    packed_token_bytes_ = packed_token_bytes;
    raw_row_capacity_ = raw_row_capacity;
    n_layers_ = n_layers;
    q_capacity_ = q_capacity;
  }

  void Shutdown() { ThrowIfError(runtime_->Shutdown()); }

  bool Initialized() const { return runtime_->initialized(); }

  bool IsCpuReference() const { return true; }

  static const char* ProbeDsaRowPayloadAbi() { return "cpu_reference_only"; }

  py::dict Geometry() const {
    DsaRowDirectoryView view;
    ThrowIfError(runtime_->GetDsaRowDirectoryView(&view));
    py::dict result;
    result["raw_row_capacity"] = view.geometry.raw_row_capacity;
    result["n_layers"] = view.geometry.n_layers;
    result["q_capacity"] = view.geometry.q_capacity;
    result["row_stride_cells"] = view.geometry.row_stride_cells;
    result["directory_cell_count"] = view.geometry.directory_cell_count;
    result["tokens_per_logical_block"] = view.layout.tokens_per_logical_block;
    result["packed_token_bytes"] = view.layout.packed_token_bytes;
    result["block_payload_bytes"] = block_payload_bytes_;
    result["ddr_block_bytes"] = ddr_block_bytes_;
    return result;
  }

  void BindRow(std::uint32_t pool_row, std::uint32_t max_sequence_tokens,
               std::uint32_t resident_budget_tokens = 0) {
    ThrowIfError(runtime_->BindResidentPoolRow(
        RowBindRequest{pool_row, max_sequence_tokens, resident_budget_tokens}));
  }

  void AbortBindRow(std::uint32_t pool_row) {
    ThrowIfError(runtime_->AbortBindResidentPoolRow(pool_row));
  }

  std::uint32_t FlatIndex(std::uint32_t pool_row, std::uint32_t local_layer,
                          std::uint32_t block_q) const {
    CheckFlatCoordinates(pool_row, local_layer, block_q);
    return pool_row * n_layers_ * q_capacity_ + local_layer * q_capacity_ +
           block_q;
  }

  // Input is one contiguous full-block sequence per local layer.  This helper
  // constructs the exact public Prepare/Write/Seal/Commit transaction for a
  // single row append; it is intentionally only a CPU mock convenience.
  py::dict PutBlocks(std::uint32_t pool_row, std::uint32_t expected_blocks,
                     py::list layer_payloads) {
    if (layer_payloads.size() != n_layers_ || expected_blocks == 0 ||
        expected_blocks > q_capacity_) {
      throw py::value_error("PutBlocks layer count or expected block count is invalid");
    }
    std::vector<py::buffer_info> inputs;
    inputs.reserve(static_cast<std::size_t>(n_layers_));
    for (const py::handle payload : layer_payloads) {
      if (!py::isinstance<py::buffer>(payload)) {
        throw py::value_error("PutBlocks layer payloads must be contiguous buffers");
      }
      py::buffer_info info = py::reinterpret_borrow<py::buffer>(payload).request();
      if (info.ptr == nullptr || info.itemsize <= 0 || info.size < 0 ||
          !IsCContiguous(info) ||
          static_cast<std::uint64_t>(info.size) >
              std::numeric_limits<std::uint64_t>::max() /
                  static_cast<std::size_t>(info.itemsize) ||
          static_cast<std::uint64_t>(info.size) * info.itemsize !=
          static_cast<std::uint64_t>(expected_blocks) * block_payload_bytes_) {
        throw py::value_error(
            "each PutBlocks layer payload must contain exactly expected_blocks full blocks");
      }
      inputs.push_back(std::move(info));
    }

    const std::uint32_t expected_committed = CommittedBlocks(pool_row);
    if (expected_blocks > q_capacity_ - expected_committed) {
      throw py::value_error("PutBlocks append exceeds the DSA_ROW q_capacity");
    }
    std::vector<KvafPutBlockTarget> targets;
    std::vector<KvafPutBlockItem> layer_items;
    targets.reserve(static_cast<std::size_t>(n_layers_) * expected_blocks);
    for (std::uint32_t layer = 0; layer < n_layers_; ++layer) {
      for (std::uint32_t block = 0; block < expected_blocks; ++block) {
        targets.push_back(KvafPutBlockTarget{FlatIndex(
            pool_row, layer, expected_committed + block)});
      }
    }
    const std::uint32_t row_flat_base = FlatIndex(pool_row, 0, 0);
    DsaPutPlanHandle plan;
    ThrowIfError(runtime_->PrepareFullBlockGroupPut(
        targets, {RowCountCommitSpec{row_flat_base, expected_committed,
                                     expected_blocks}},
        &plan));
    try {
      for (std::uint32_t layer = 0; layer < n_layers_; ++layer) {
        layer_items.clear();
        for (std::uint32_t block = 0; block < expected_blocks; ++block) {
          layer_items.push_back(KvafPutBlockItem{
              FlatIndex(pool_row, layer, expected_committed + block),
              PackedSourceBlockView{static_cast<std::uint64_t>(block) *
                                        block_payload_bytes_,
                                    static_cast<std::uint32_t>(block_payload_bytes_)}});
        }
        ThrowIfError(runtime_->WriteFullBlockGroupLayer(
            plan, layer_items,
            BoundLayerCacheView{ConstBufferView{
                                    inputs[layer].ptr,
                                    static_cast<std::size_t>(inputs[layer].size) *
                                        static_cast<std::size_t>(inputs[layer].itemsize),
                                    DeviceKind::kCpu},
                                1, layer}));
      }
      DsaSealResult seal;
      ThrowIfError(runtime_->SealFullBlockGroupPut(plan, &seal));
      PutLocalResult commit;
      ThrowIfError(runtime_->CommitFullBlockGroupPut(&plan, &commit));
      py::dict result;
      result["state"] = "committed";
      result["published_block_count"] = seal.published_block_count;
      result["committed_block_count"] = commit.rows.empty()
                                              ? 0
                                              : commit.rows.front().committed_block_count;
      return result;
    } catch (...) {
      if (plan.opaque != 0) {
        PutLocalResult ignored;
        (void)runtime_->AbortFullBlockGroupPut(&plan, &ignored);
      }
      throw;
    }
  }

  py::bytearray GetTokens(std::uint32_t pool_row, std::uint32_t local_layer,
                          const std::vector<std::uint32_t>& source_token_ids,
                          const std::vector<std::int32_t>& destination_indices,
                          py::bytearray output) {
    if (source_token_ids.size() != destination_indices.size() ||
        local_layer >= n_layers_) {
      throw py::value_error("GetTokens sources/destinations or layer is invalid");
    }
    char* output_data = PyByteArray_AsString(output.ptr());
    const Py_ssize_t output_size = PyByteArray_Size(output.ptr());
    if (output_data == nullptr || output_size < 0 ||
        static_cast<std::uint64_t>(output_size) % packed_token_bytes_ != 0) {
      throw py::value_error("GetTokens output must be token-sized bytearray storage");
    }
    PrepareUnifiedAccessResult access;
    ThrowIfError(runtime_->PrepareUnifiedAccess(
        PrepareUnifiedAccessRequest{{UnifiedAccessRow{
            DsaRowAccessMode::kSparse, true, pool_row, CommittedBlocks(pool_row)}}},
        &access));
    DsaDirectGetPlanHandle plan;
    try {
      std::vector<KvafGetItem> items;
      items.reserve(source_token_ids.size());
      for (std::size_t index = 0; index < source_token_ids.size(); ++index) {
        const std::uint32_t token = source_token_ids[index];
        const std::uint32_t block_q = token / tokens_per_logical_block_;
        items.push_back(KvafGetItem{
            KvafGetSource{FlatIndex(pool_row, local_layer, block_q),
                          token % tokens_per_logical_block_},
            destination_indices[index]});
      }
      ThrowIfError(runtime_->PrepareDirectBatchGet(
          access.handle,
          DirectGetRequest{items,
                           MutableBufferView{output_data,
                                             static_cast<std::size_t>(output_size),
                                             DeviceKind::kCpu},
                           local_layer, nullptr},
          &plan));
      DirectGetCompletion completion;
      ThrowIfError(runtime_->ExecutePreparedDirectBatchGet(plan, &completion));
      GetLocalResult settled;
      ThrowIfError(
          runtime_->SettleDirectBatchGet(&plan, completion, &settled));
      ThrowIfError(runtime_->ReleaseUnifiedAccess(&access.handle));
      return output;
    } catch (...) {
      if (plan.opaque != 0) {
        GetLocalResult ignored;
        // If an execution error is reported after dispatch, the Runtime
        // quarantines the source row before releasing this access handle.
        (void)runtime_->AbortPreparedDirectBatchGet(&plan, &ignored);
      }
      if (access.handle.opaque != 0) {
        (void)runtime_->ReleaseUnifiedAccess(&access.handle);
      }
      throw;
    }
  }

  py::dict CloseRow(std::uint32_t pool_row) {
    CloseLocalResult close;
    ThrowIfError(runtime_->CloseResidentPoolRow(pool_row, &close));
    py::dict result;
    result["state"] = "free";
    result["released_block_count"] = close.released_block_count;
    return result;
  }

 private:
  void CheckFlatCoordinates(std::uint32_t pool_row, std::uint32_t local_layer,
                            std::uint32_t block_q) const {
    if (!runtime_->initialized() || pool_row >= raw_row_capacity_ ||
        local_layer >= n_layers_ || block_q >= q_capacity_) {
      throw py::value_error("DSA_ROW flat-index coordinates are out of range");
    }
  }

  std::uint32_t CommittedBlocks(std::uint32_t pool_row) const {
    std::uint32_t count = 0;
    ThrowIfError(runtime_->GetDsaRowCommittedBlockCount(pool_row, &count));
    return count;
  }

  std::unique_ptr<MetaManager> meta_manager_;
  std::unique_ptr<DdrMemoryAllocator> allocator_;
  std::unique_ptr<MockAccessOperator> access_operator_;
  std::unique_ptr<Runtime> runtime_;
  std::uint64_t block_payload_bytes_{0};
  std::uint64_t ddr_block_bytes_{0};
  std::uint32_t tokens_per_logical_block_{0};
  std::uint32_t packed_token_bytes_{0};
  std::uint32_t raw_row_capacity_{0};
  std::uint32_t n_layers_{0};
  std::uint32_t q_capacity_{0};
};

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
           py::arg("input"), py::arg("input_offsets"), py::arg("lengths"))
      .def("prepare_access_table", &kvaf::PythonRuntime::PrepareAccessTable,
           py::arg("keys"))
      .def("access_table_view", &kvaf::PythonRuntime::AccessTableView,
           py::arg("access_table"))
      .def("release_access_table", &kvaf::PythonRuntime::ReleaseAccessTable,
           py::arg("access_table"))
      .def("batch_get", &kvaf::PythonRuntime::BatchGet,
           py::arg("access_table"), py::arg("indices"),
           py::arg("output_indices"), py::arg("output"),
           py::arg("output_block_bytes"), py::arg("mask") = py::none())
      .def("batch_free", &kvaf::PythonRuntime::BatchFree, py::arg("keys"))
      .def("_fail_next_put", &kvaf::PythonRuntime::FailNextPut,
           "Test hook: make the next BatchPut fail before Meta commit.")
      .def("_fail_next_get", &kvaf::PythonRuntime::FailNextGet,
           "Test hook: make the next BatchGet fail.");

  py::class_<kvaf::DsaRowMockPythonRuntime>(m, "DsaRowMockRuntime")
      .def(py::init<>())
      .def("initialize", &kvaf::DsaRowMockPythonRuntime::Initialize,
           py::arg("ddr_bytes"), py::arg("raw_row_capacity"),
           py::arg("n_layers"), py::arg("q_capacity"),
           py::arg("tokens_per_logical_block") = 128,
           py::arg("packed_token_bytes") = 656, py::arg("layout_id") = 1,
           py::arg("ddr_block_count") = 0,
           py::arg("ddr_block_alignment_bytes") = 32,
           "Initialize CPU DSA_ROW geometry and its fixed DDR block arena.")
      .def("shutdown", &kvaf::DsaRowMockPythonRuntime::Shutdown)
      .def_property_readonly("initialized", &kvaf::DsaRowMockPythonRuntime::Initialized)
      .def_property_readonly("geometry", &kvaf::DsaRowMockPythonRuntime::Geometry)
      .def_property_readonly("is_cpu_reference",
                             &kvaf::DsaRowMockPythonRuntime::IsCpuReference)
      .def_static("probe_dsa_row_payload_abi",
                  &kvaf::DsaRowMockPythonRuntime::ProbeDsaRowPayloadAbi)
      .def("bind_row", &kvaf::DsaRowMockPythonRuntime::BindRow,
           py::arg("pool_row"), py::arg("max_sequence_tokens"),
           py::arg("resident_budget_tokens") = 0)
      .def("abort_bind_row", &kvaf::DsaRowMockPythonRuntime::AbortBindRow,
           py::arg("pool_row"))
      .def("flat_index", &kvaf::DsaRowMockPythonRuntime::FlatIndex,
           py::arg("pool_row"), py::arg("local_layer"), py::arg("block_q"))
      .def("put_blocks", &kvaf::DsaRowMockPythonRuntime::PutBlocks,
           py::arg("pool_row"), py::arg("expected_blocks"),
           py::arg("layer_payloads"),
           "Append full logical blocks from one contiguous CPU buffer per layer.")
      .def("get_tokens", &kvaf::DsaRowMockPythonRuntime::GetTokens,
           py::arg("pool_row"), py::arg("local_layer"),
           py::arg("source_token_ids"), py::arg("destination_indices"),
           py::arg("output"),
           "Copy selected packed tokens through DSA flat-index DirectGet.")
      .def("close_row", &kvaf::DsaRowMockPythonRuntime::CloseRow,
           py::arg("pool_row"));

  m.def("create_mock_runtime", [] { return kvaf::PythonRuntime(); },
        "Create an in-process Runtime with the deterministic CPU mock operator.");
  m.attr("MockRuntime") = m.attr("Runtime");
  m.def("create_dsa_row_mock_runtime", [] {
    return kvaf::DsaRowMockPythonRuntime();
  });

  py::module_ runtime = m.def_submodule("runtime");
  runtime.attr("Runtime") = m.attr("Runtime");
  runtime.attr("MockRuntime") = m.attr("Runtime");
  runtime.attr("create_mock_runtime") = m.attr("create_mock_runtime");
  runtime.attr("DsaRowMockRuntime") = m.attr("DsaRowMockRuntime");
  runtime.attr("create_dsa_row_mock_runtime") =
      m.attr("create_dsa_row_mock_runtime");
  m.def_submodule("meta_manager");
  m.def_submodule("access_operator");

#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  kvaf::BindNpuPythonRuntime(m);
  kvaf::BindAclGraph(m);
#endif
}
