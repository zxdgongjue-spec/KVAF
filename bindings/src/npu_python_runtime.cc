#include "npu_python_runtime.h"

#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
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

class NpuPythonRuntime final {
 public:
  NpuPythonRuntime(std::string batch_get_kernel_object_path,
                   std::uint32_t batch_get_token_bytes,
                   std::uint32_t batch_get_tokens_per_block,
                   std::uint32_t device_id, std::uint32_t npu_core_count,
                   std::string batch_put_kernel_object_path,
                   std::uint32_t jettys_per_core) {
    NpuRuntimeOptions options{std::move(batch_get_kernel_object_path),
                              batch_get_token_bytes,
                              batch_get_tokens_per_block};
    options.batch_put_kernel_object_path =
        std::move(batch_put_kernel_object_path);
    options.device_id = device_id;
    options.npu_core_count = npu_core_count;
    options.jettys_per_core = jettys_per_core;
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

  // Cold-bind a forward-owned canonical resident payload.  This method does
  // not issue a Get: it publishes the MTP 112-byte BatchGet ABI state that a
  // graph-resident device executor later consumes with PostOnly/CQ drain.
  // Keeping this registration separate from BatchGet prevents Python/GIL
  // submission from entering the score-tile hot path.
  void BindLayerResidentBatchGetControl(
      std::uint32_t layer_id, const py::object& resident_payload,
      const py::object& control, std::uint32_t slot_base,
      std::uint32_t slot_count, const py::object& queue_leases,
      std::uint32_t queue_count, const py::object& producer_stream) {
    if (!Initialized()) {
      throw std::runtime_error("Runtime has not been initialized");
    }
    if (layer_resident_batch_get_control_bound_) {
      throw py::value_error(
          "layer-resident BatchGet control is already bound; release after "
          "the forward terminal before rebinding");
    }
    const NpuTensorView payload_view = GetNpuTensorView(
        resident_payload, "layer-resident resident_payload");
    const NpuTensorView control_view = GetNpuTensorView(
        control, "layer-resident BatchGet control");
    const NpuTensorView leases_view = GetNpuTensorView(
        queue_leases, "layer-resident BatchGet queue_leases");
    const std::uint32_t token_bytes = native_->batch_get_token_bytes();
    if (token_bytes == 0U || payload_view.shape.size() < 1U ||
        payload_view.shape.front() <= 0 ||
        payload_view.bytes % token_bytes != 0U ||
        payload_view.bytes / token_bytes !=
            static_cast<std::uint64_t>(payload_view.shape.front())) {
      throw py::value_error(
          "layer-resident resident_payload must have one full MTP token row "
          "per first-dimension slot");
    }
    const std::uint64_t payload_rows = payload_view.bytes / token_bytes;
    if (payload_rows > std::numeric_limits<std::uint32_t>::max() ||
        slot_count == 0U || slot_base >= payload_rows ||
        slot_count > payload_rows - slot_base) {
      throw py::value_error(
          "layer-resident slot_base/slot_count must select a non-empty fixed "
          "interval inside resident_payload");
    }
    if (control_view.shape.size() != 1U ||
        control_view.shape.front() != 16 ||
        control_view.element_bytes != sizeof(std::uint64_t) ||
        py::str(control.attr("dtype")).cast<std::string>() != "torch.int64") {
      throw py::value_error(
          "layer-resident control must be contiguous NPU torch.int64[16]");
    }
    if (leases_view.shape.size() != 1U ||
        leases_view.shape.front() != static_cast<py::ssize_t>(queue_count) ||
        leases_view.element_bytes != sizeof(std::int32_t) ||
        py::str(queue_leases.attr("dtype")).cast<std::string>() !=
            "torch.int32") {
      throw py::value_error(
          "layer-resident queue_leases must be contiguous NPU "
          "torch.int32[queue_count]");
    }

    // Inputs normally originate from torch.zeros/current-stream initialization.
    // The cold fence makes that producer visible before raw ACL registration
    // and control publication; it is never called from device tile execution.
    ThrowIfAclError(aclrtSynchronizeStream(ResolveStream(producer_stream)),
                    "aclrtSynchronizeStream before layer-resident BatchGet bind");
    ThrowIfError(native_->BindLayerResidentBatchGetControl(
        LayerResidentBatchGetBinding{
            layer_id,
            MutableBufferView{payload_view.data, payload_view.bytes,
                              DeviceKind::kNpu},
            MutableBufferView{control_view.data, control_view.bytes,
                              DeviceKind::kNpu},
            slot_base,
            slot_count},
        MutableBufferView{leases_view.data, leases_view.bytes,
                          DeviceKind::kNpu},
        queue_count));
    layer_resident_batch_get_control_bound_ = true;
    layer_resident_batch_get_payload_keepalive_ = resident_payload;
    layer_resident_batch_get_control_keepalive_ = control;
    layer_resident_batch_get_queue_leases_keepalive_ = queue_leases;
  }

  // Publish one immutable descriptor-template record per queue.  MTP's
  // ExecuteBatchGet ABI uses a compact prefix length (`miss_counts[q]` is
  // torch.int32[1]) rather than a per-request bitmap.
  void ConfigureLayerResidentBatchGetQueueTemplates(
      const py::sequence& queue_ids, const py::sequence& address_tables,
      const py::sequence& valid_tables, const py::sequence& indices,
      const py::sequence& offsets, const py::sequence& miss_counts,
      const py::sequence& output_indices, const py::object& producer_stream) {
    if (!Initialized() || !layer_resident_batch_get_control_bound_) {
      throw py::value_error(
          "layer-resident descriptor templates require a bound control");
    }
    const py::ssize_t count = py::len(queue_ids);
    if (count <= 0 || py::len(address_tables) != count ||
        py::len(valid_tables) != count || py::len(indices) != count ||
        py::len(offsets) != count || py::len(miss_counts) != count ||
        py::len(output_indices) != count) {
      throw py::value_error(
          "layer-resident template sequences must be non-empty and "
          "equal-length");
    }
    const auto checked_queue_id = [](const py::object& value) {
      const py::ssize_t converted = value.cast<py::ssize_t>();
      if (converted < 0 ||
          static_cast<std::uint64_t>(converted) >
              std::numeric_limits<std::uint32_t>::max()) {
        throw py::value_error("layer-resident queue_id must fit uint32");
      }
      return static_cast<std::uint32_t>(converted);
    };
    const auto validate_vector = [](const NpuTensorView& view,
                                    const py::object& tensor,
                                    std::size_t element_bytes,
                                    const char* dtype,
                                    const std::string& label) {
      if (view.shape.size() != 1U || view.element_bytes != element_bytes ||
          py::str(tensor.attr("dtype")).cast<std::string>() != dtype) {
        throw py::value_error(label + " must be a contiguous one-dimensional " +
                              std::string(dtype) + " NPU Tensor");
      }
    };

    std::vector<LayerResidentBatchGetQueueTemplate> templates;
    std::vector<py::object> keepalives;
    templates.reserve(static_cast<std::size_t>(count));
    keepalives.reserve(static_cast<std::size_t>(count) * 6U);
    for (py::ssize_t index = 0; index < count; ++index) {
      const std::uint32_t queue_id =
          checked_queue_id(queue_ids[index].cast<py::object>());
      py::object address_table = address_tables[index].cast<py::object>();
      py::object valid_table = valid_tables[index].cast<py::object>();
      py::object index_vector = indices[index].cast<py::object>();
      py::object offset_vector = offsets[index].cast<py::object>();
      py::object miss_count = miss_counts[index].cast<py::object>();
      py::object output_index_vector =
          output_indices[index].cast<py::object>();
      const std::string label =
          "layer-resident descriptor queue=" + std::to_string(queue_id);
      const NpuTensorView address_view =
          GetNpuTensorView(address_table, (label + " address_table").c_str());
      const NpuTensorView valid_view =
          GetNpuTensorView(valid_table, (label + " valid_table").c_str());
      const NpuTensorView index_view =
          GetNpuTensorView(index_vector, (label + " indices").c_str());
      const NpuTensorView offset_view =
          GetNpuTensorView(offset_vector, (label + " offsets").c_str());
      const NpuTensorView miss_count_view =
          GetNpuTensorView(miss_count, (label + " miss_count").c_str());
      const NpuTensorView output_index_view = GetNpuTensorView(
          output_index_vector, (label + " output_indices").c_str());
      validate_vector(address_view, address_table, sizeof(std::uint64_t),
                      "torch.int64", label + " address_table");
      validate_vector(valid_view, valid_table, sizeof(std::uint8_t),
                      "torch.uint8", label + " valid_table");
      validate_vector(index_view, index_vector, sizeof(std::int32_t),
                      "torch.int32", label + " indices");
      validate_vector(offset_view, offset_vector, sizeof(std::int32_t),
                      "torch.int32", label + " offsets");
      validate_vector(miss_count_view, miss_count, sizeof(std::int32_t),
                      "torch.int32", label + " miss_count");
      if (miss_count_view.bytes != sizeof(std::int32_t)) {
        throw py::value_error(label + " miss_count must be torch.int32[1]");
      }
      validate_vector(output_index_view, output_index_vector,
                      sizeof(std::int32_t), "torch.int32",
                      label + " output_indices");
      templates.push_back(LayerResidentBatchGetQueueTemplate{
          queue_id,
          MutableBufferView{address_view.data, address_view.bytes,
                            DeviceKind::kNpu},
          MutableBufferView{valid_view.data, valid_view.bytes,
                            DeviceKind::kNpu},
          MutableBufferView{index_view.data, index_view.bytes,
                            DeviceKind::kNpu},
          MutableBufferView{offset_view.data, offset_view.bytes,
                            DeviceKind::kNpu},
          MutableBufferView{miss_count_view.data, miss_count_view.bytes,
                            DeviceKind::kNpu},
          MutableBufferView{output_index_view.data, output_index_view.bytes,
                            DeviceKind::kNpu}});
      keepalives.push_back(std::move(address_table));
      keepalives.push_back(std::move(valid_table));
      keepalives.push_back(std::move(index_vector));
      keepalives.push_back(std::move(offset_vector));
      keepalives.push_back(std::move(miss_count));
      keepalives.push_back(std::move(output_index_vector));
    }
    ThrowIfAclError(
        aclrtSynchronizeStream(ResolveStream(producer_stream)),
        "aclrtSynchronizeStream before layer-resident template configure");
    ThrowIfError(native_->ConfigureLayerResidentBatchGetQueueTemplates(
        templates));
    layer_resident_batch_get_template_keepalives_ = std::move(keepalives);
  }

  void ReleaseLayerResidentBatchGetControl() {
    if (!layer_resident_batch_get_control_bound_) {
      return;
    }
    ThrowIfError(native_->ReleaseLayerResidentBatchGetControl());
    layer_resident_batch_get_control_bound_ = false;
    layer_resident_batch_get_payload_keepalive_ = py::none();
    layer_resident_batch_get_control_keepalive_ = py::none();
    layer_resident_batch_get_queue_leases_keepalive_ = py::none();
    layer_resident_batch_get_template_keepalives_.clear();
  }

  std::string LayerResidentBatchGetControlDiagnostic() const {
    std::string diagnostic;
    ThrowIfError(native_->DescribeLayerResidentBatchGetControl(&diagnostic));
    return diagnostic;
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
    ThrowIfError(native_->Shutdown());
  }

 private:
  std::unique_ptr<::kvaf::NpuRuntime> native_;
  bool layer_resident_batch_get_control_bound_{false};
  py::object layer_resident_batch_get_payload_keepalive_{py::none()};
  py::object layer_resident_batch_get_control_keepalive_{py::none()};
  py::object layer_resident_batch_get_queue_leases_keepalive_{py::none()};
  std::vector<py::object> layer_resident_batch_get_template_keepalives_;
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
                    std::uint32_t, std::string, std::uint32_t>(),
           py::arg("batch_get_kernel_object_path"),
           py::arg("batch_get_token_bytes"),
           py::arg("batch_get_tokens_per_block"), py::arg("device_id") = 0,
           py::arg("npu_core_count") = 1,
           py::arg("batch_put_kernel_object_path") = "",
           py::arg("jettys_per_core") = 2)
      .def("link_to", &NpuPythonRuntime::LinkTo, py::arg("address"),
           "Connect this NPU Runtime to a CPU Runtime service before initialize.")
      .def("initialize", &NpuPythonRuntime::Initialize, py::arg("ddr_bytes"),
           py::arg("slot_size"))
      .def("shutdown", &NpuPythonRuntime::Shutdown)
      .def_property_readonly("initialized", &NpuPythonRuntime::Initialized)
      .def_property_readonly("device_id", &NpuPythonRuntime::DeviceId)
      .def_property_readonly("npu_core_count", &NpuPythonRuntime::NpuCoreCount)
      .def_property_readonly("jettys_per_core", &NpuPythonRuntime::JettysPerCore)
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
      .def("batch_get_kernel_profile",
           &NpuPythonRuntime::BatchGetKernelProfileSnapshot,
           "Copy the latest completed per-core BatchGet kernel timeline.")
      .def("stop_batch_get_kernel_profile",
           &NpuPythonRuntime::StopBatchGetKernelProfile,
           "Copy the latest timeline and disable BatchGet kernel profiling.")
      .def("bind_layer_resident_batch_get_control",
           &NpuPythonRuntime::BindLayerResidentBatchGetControl,
           py::arg("layer_id"), py::arg("resident_payload"),
           py::arg("control"), py::arg("slot_base"), py::arg("slot_count"),
           py::arg("queue_leases"), py::arg("queue_count"),
           py::arg("stream") = py::none(),
           "Cold-bind one layer's fixed resident payload interval and a "
           "graph-visible NPU torch.int64[16] MTP control. resident_payload "
           "has one batch_get_token_bytes row per slot; slot_base/slot_count "
           "select rows the device executor may populate. queue_leases is "
           "NPU torch.int32[queue_count]. This method never posts a Get from "
           "Python; PostOnly/CQ drain remain in the device graph.")
      .def("configure_layer_resident_batch_get_queue_templates",
           &NpuPythonRuntime::ConfigureLayerResidentBatchGetQueueTemplates,
           py::arg("queue_ids"), py::arg("address_tables"),
           py::arg("valid_tables"), py::arg("indices"), py::arg("offsets"),
           py::arg("miss_counts"), py::arg("output_indices"),
           py::arg("stream") = py::none(),
           "Cold-publish one MTP 112-byte BatchGet descriptor template per "
           "queue. queue_ids cover the pool exactly once. addresses are NPU "
           "torch.int64 vectors, valid is torch.uint8, indices/offsets/"
           "output_indices are torch.int32 vectors, and miss_counts are "
           "torch.int32[1] compact-prefix counters (not masks).")
      .def("release_layer_resident_batch_get_control",
           &NpuPythonRuntime::ReleaseLayerResidentBatchGetControl,
           "Release the cold Direct-HCCP registration only after the forward "
           "has terminally drained every queue lease. Resident payload slots "
           "are caller-owned and never cleared or remapped by KVAF.")
      .def("layer_resident_batch_get_control_diagnostic",
           &NpuPythonRuntime::LayerResidentBatchGetControlDiagnostic,
           "Return the MTP layer-resident cold binding diagnostic.")
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
