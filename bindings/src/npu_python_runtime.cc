#include "hbm_read_python.h"
#include "npu_python_runtime.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
#include "remote_offload_python.h"
#include "connector_python.h"
#endif

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
                   bool enable_cpu_offload, std::int64_t cpu_candidate,
                   std::uint32_t offload_timeout_ms, std::string raw_kernel) {
    NpuRuntimeOptions options{std::move(batch_get_kernel_object_path),
                              batch_get_token_bytes,
                              batch_get_tokens_per_block};
    options.batch_put_kernel_object_path =
        std::move(batch_put_kernel_object_path);
    options.device_id = device_id;
    options.npu_core_count = npu_core_count;
    options.jettys_per_core = jettys_per_core;
    options.batch_get_max_inflight_total = batch_get_max_inflight_total;
    options.npu_eids = std::move(npu_eids);
    options.adaptive_ub_device = adaptive_ub_device;
    options.adaptive_ub_miss_count = adaptive_ub_miss_count;
    options.peer_ub_device = std::move(peer_ub_device);
    options.adaptive_peer_ub_devices = std::move(adaptive_peer_ub_devices);
    if (cpu_candidate < -1 || cpu_candidate >= UINT32_MAX) throw py::value_error("invalid CPU candidate");
    options.enable_cpu_offload = enable_cpu_offload;
    options.cpu_candidate = cpu_candidate < 0 ? UINT32_MAX : static_cast<std::uint32_t>(cpu_candidate);
    options.offload_timeout_ms = offload_timeout_ms;
    options.batch_get_raw_addr_kernel_object_path = std::move(raw_kernel);
    native_ = std::make_unique<::kvaf::NpuRuntime>(std::move(options));
  }

  ~NpuPythonRuntime() {
    if (!native_->Shutdown().ok()) {
      (void)raw_retained_.release();
      (void)raw_outputs_.release();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
      (void)local_sources_.release();
#endif
    }
  }

  void LinkTo(std::string address) {
    ThrowIfError(native_->LinkTo(LinkTarget{std::move(address)}));
  }

  void Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
    ThrowIfError(native_->Initialize(ddr_bytes, slot_size));
  }

#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  void BatchCpuPull(const py::object& keys, py::object source, const py::object& offsets,
                    const py::object& lengths, const py::object& stream) {
    if (native_->pending_offload()) throw py::value_error("finish the existing transaction or shutdown the unresolved batch_cpu_pull");
    const auto view = GetNpuTensorView(source, "source");
    if (source.attr("device").attr("index").cast<std::uint32_t>() != native_->device_id())
      throw py::value_error("batch_cpu_pull requires local HBM; use CpuPullHbmSource and submit_cpu_pull for another NPU");
    BatchPutRequest request; request.keys = CopyPrepareKeys(keys);
    CopyCpuUint64SequenceInto(offsets, "input_offsets", &request.input_offsets);
    CopyCpuUint64SequenceInto(lengths, "lengths", &request.lengths);
    request.input = {view.data, view.bytes, DeviceKind::kNpu};
    const auto producer = ResolveStream(stream);
    if (!stream.is_none() && py::hasattr(stream, "device") &&
        stream.attr("device").attr("index").cast<std::uint32_t>() != native_->device_id())
      throw py::value_error("producer stream belongs to another NPU");
    bool retained = false;
    for (auto item : local_sources_) if (item.is(source)) retained = true;
    if (!retained) local_sources_.append(source);
    const auto status = native_->BatchCpuPull(request, {producer});
    if (status.ok()) { local_sources_ = py::list(); local_sources_.append(source); }
    ThrowIfError(status);
  }
  py::dict OffloadRegion() const {
    OffloadRegionTicket ticket; ThrowIfError(native_->GetOffloadRegion(&ticket)); return remote_python::Pack(ticket);
  }
  CpuPullSourceHandle AttachCpuSource(const CpuPullSourceDesc& source) {
    CpuPullSourceHandle result; ThrowIfError(native_->AttachCpuSource(source,&result)); return result;
  }
  PutHandle PreparePut(const py::object& keys,const py::object& lengths) {
    std::vector<std::uint64_t> sizes;
    CopyCpuUint64SequenceInto(lengths,"lengths",&sizes);
    PutHandle put; ThrowIfError(native_->PreparePut(CopyPrepareKeys(keys),sizes,&put)); return put;
  }
  CpuPullTask SubmitCpuPull(const CpuPullSourceHandle& source,const ReadyToken& ready,const PutHandle& put,
                           const py::object& src,const py::object& dst,const py::object& lengths) {
    std::vector<std::uint64_t> sources, destinations, sizes;
    CopyCpuUint64SequenceInto(src, "src_offsets", &sources);
    CopyCpuUint64SequenceInto(dst, "dst_offsets", &destinations);
    CopyCpuUint64SequenceInto(lengths, "lengths", &sizes);
    CpuPullTask task;
    const auto status=native_->SubmitCpuPull(source,ready,put,sources,destinations,sizes,&task);
    if(!status.ok()) throw std::runtime_error(status.message()+"; recover the original task via put.task");
    return task;
  }
  CpuPullState QueryCpuPull(const CpuPullTask& task) {
    CpuPullState state; ThrowIfError(native_->QueryCpuPull(task,&state)); return state;
  }
  CpuPullState WaitCpuPull(const CpuPullTask& task,std::uint32_t timeout) {
    CpuPullState state; Status status;
    {py::gil_scoped_release release;status=native_->WaitCpuPull(task,timeout,&state);}
    ThrowIfError(status); return state;
  }
  CpuPullState CommitPut(const PutHandle& put) { ThrowIfError(native_->CommitPut(put)); return put.state(); }
  CpuPullState AbortPut(const PutHandle& put) { ThrowIfError(native_->AbortPut(put)); return put.state(); }
  bool PendingOffload() const { return native_->pending_offload(); }
  std::uint64_t LastChecksum() const { std::uint64_t hash{}; ThrowIfError(native_->LastOffloadChecksum(&hash)); return hash; }
#endif

  std::uint64_t AttachHbm(const std::string& address, const py::dict& ticket, const std::string& eid) {
    LinkId region{};
    auto status = native_->AttachHbm(address, hbm_read_python::Ticket(ticket), &region, eid);
    if (!status.ok()) throw std::runtime_error(status.message() + "; region=" + std::to_string(region));
    return region;
  }
  void ResolveRawAddr(LinkId region) {
    const auto status = native_->ResolveRawAddr(region);
    if (!native_->pending_raw_addr()) raw_retained_ = py::none();
    ThrowIfError(status);
  }
  py::dict BatchGetRawArrays(LinkId region,const ReadyToken& ready,const py::object& src,const py::object& dst,
                         const py::object& lengths,py::object output,py::object stream,bool profiling,bool validate_ranges) {
    auto array = [&](const py::object& tensor, const char* name) {
      const auto view = GetNpuTensorView(tensor, name);
      if (view.shape.size() != 1 || view.element_bytes != sizeof(std::int64_t) ||
          !py::hasattr(tensor, "dtype") || py::str(tensor.attr("dtype")).cast<std::string>() != "torch.int64" ||
          reinterpret_cast<std::uintptr_t>(view.data) % alignof(std::int64_t) ||
          tensor.attr("device").attr("index").cast<std::uint32_t>() != native_->device_id())
        throw py::value_error(std::string(name) + " must be a one-dimensional torch.int64 Tensor on the Runtime NPU");
      return NpuInt64ArrayView{static_cast<const std::int64_t*>(view.data), view.bytes / sizeof(std::int64_t)};
    };
    const auto sources = array(src, "src_offsets");
    const auto destinations = array(dst, "dst_offsets");
    const auto sizes = array(lengths, "lengths");
    if (sources.size != destinations.size || sources.size != sizes.size || sources.size > 4096)
      throw py::value_error("raw READ metadata must have equal lengths in 1..4096");
    if(native_->pending_raw_addr())throw py::value_error("resolve the pending raw READ first");
    const auto view=GetNpuTensorView(output,"output");
    if(output.attr("device").attr("index").cast<std::uint32_t>()!=native_->device_id())
      throw py::value_error("raw READ output belongs to another NPU");
    if(stream.is_none())stream=py::module_::import("torch").attr("npu").attr("current_stream")(native_->device_id());
    if(py::hasattr(stream,"device") && stream.attr("device").attr("index").cast<std::uint32_t>()!=native_->device_id())
      throw py::value_error("raw READ stream belongs to another NPU");
    const auto acl_stream=acl_python::StreamFromPython(stream);
    // Transport caches (address, size) registrations until detach. Pin their
    // Python owners for the same lifetime so allocator reuse cannot alias an MR.
    const py::int_ region_key(region);
    if(!raw_outputs_.contains(region_key))raw_outputs_[region_key]=py::dict();
    auto outputs=raw_outputs_[region_key].cast<py::dict>();
    outputs[py::make_tuple(reinterpret_cast<std::uintptr_t>(view.data),view.bytes)]=output;
    raw_retained_=py::make_tuple(src,dst,lengths,output,stream);
    HbmReadProfile profile;
    const auto status=native_->BatchGetRawAddr(region,ready,sources,destinations,sizes,
        {const_cast<void*>(view.data),view.bytes,DeviceKind::kNpu},acl_stream,profiling?&profile:nullptr,validate_ranges);
    if(!native_->pending_raw_addr())raw_retained_=py::none();
    ThrowIfError(status);
    py::dict result; result["state"]="done"; result["sequence"]=ready.sequence;
    if(profiling) {
      py::dict timing;
      timing["host_total_us"]=profile.total_us;
      double phases=0;
#define RAW_PHASE(name) timing[#name]=profile.name; phases+=profile.name
      RAW_PHASE(validation_us); RAW_PHASE(acquire_us); RAW_PHASE(buffers_us); RAW_PHASE(register_output_us);
      RAW_PHASE(event_record_us); RAW_PHASE(launch_us); RAW_PHASE(synchronize_us); RAW_PHASE(event_query_us);
      RAW_PHASE(status_us); RAW_PHASE(cleanup_us); RAW_PHASE(complete_us);
#undef RAW_PHASE
      timing["host_other_us"]=std::max(0.0,profile.total_us-phases);
      timing["kernel_event_us"]=profile.kernel_event_valid?py::cast(profile.kernel_event_us):py::none();
      timing["kernel_event_error"]=profile.event_error;
      result["profile"]=std::move(timing);
    }
    return result;
  }
  void DetachHbm(LinkId region) {
    ThrowIfError(native_->DetachHbm(region));
    raw_outputs_.attr("pop")(py::int_(region),py::none());
  }
  bool PendingRawAddr() const { return native_->pending_raw_addr(); }

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
    ThrowIfError(native_->Shutdown());
    raw_outputs_=py::dict();
    raw_retained_=py::none();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
    local_sources_ = py::list();
#endif
  }

 private:
  py::object raw_retained_{py::none()};
  py::dict raw_outputs_;
  std::unique_ptr<::kvaf::NpuRuntime> native_;
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  py::list local_sources_;
#endif
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
                    std::string, std::vector<std::string>, std::uint32_t, bool, std::int64_t, std::uint32_t, std::string>(),
           py::arg("batch_get_kernel_object_path") = "",
           py::arg("batch_get_token_bytes") = 0,
           py::arg("batch_get_tokens_per_block") = 0, py::arg("device_id") = 0,
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
           py::arg("enable_cpu_offload") = false, py::arg("cpu_candidate") = -1,
           py::arg("offload_timeout_ms") = 30000,
           py::arg("batch_get_raw_addr_kernel_object_path") = "")
      .def("attach_hbm", &NpuPythonRuntime::AttachHbm, py::arg("address"), py::arg("export_ticket"), py::arg("local_eid") = "")
      .def("batch_get_raw_addr", &NpuPythonRuntime::BatchGetRawArrays,
           py::arg("hbm_link"),py::arg("ready"),py::arg("src_offsets"),py::arg("dst_offsets"),py::arg("lengths"),
           py::arg("output"),py::arg("stream")=py::none(),py::arg("profile")=false,py::arg("validate_ranges")=false,
           "Synchronously read peer HBM. profile=True adds Host stage timings and a device event "
           "interval around the kernel to the returned profile dictionary; profiling is off by default. "
           "Output registrations and their Tensors are retained until detach/shutdown; do not resize their storage. "
           "Callers guarantee valid, disjoint ranges. validate_ranges=True enables bounds diagnostics only.")
      .def("resolve_raw_addr", &NpuPythonRuntime::ResolveRawAddr, py::arg("region"))
      .def("detach_hbm", &NpuPythonRuntime::DetachHbm, py::arg("region"))
      .def_property_readonly("pending_raw_addr", &NpuPythonRuntime::PendingRawAddr)
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
      .def("batch_cpu_pull", &NpuPythonRuntime::BatchCpuPull, py::arg("keys"), py::arg("source"),
           py::arg("input_offsets"), py::arg("lengths"), py::arg("stream") = py::none(),
           "Synchronously pull local NPU HBM into MetaManager-allocated DDR slots and commit. "
           "Returns None on success. On unresolved failure retain source data and call shutdown(); "
           "calling batch_cpu_pull again never resumes or resubmits a pending operation.")
      .def("offload_region", &NpuPythonRuntime::OffloadRegion)
      .def("attach_cpu_source",&NpuPythonRuntime::AttachCpuSource,py::arg("source_desc"))
      .def("prepare_put",&NpuPythonRuntime::PreparePut,py::arg("keys"),py::arg("lengths"))
      .def("submit_cpu_pull",&NpuPythonRuntime::SubmitCpuPull,py::arg("source_handle"),py::arg("ready"),py::arg("put"),
           py::arg("src_offsets"),py::arg("dst_offsets"),py::arg("lengths"))
      .def("query_cpu_pull",&NpuPythonRuntime::QueryCpuPull,py::arg("task"))
      .def("wait_cpu_pull",&NpuPythonRuntime::WaitCpuPull,py::arg("task"),py::arg("timeout")=30000,
           "Wait for DMA only; timeout is in milliseconds. Commit explicitly afterward.")
      .def("commit_put",&NpuPythonRuntime::CommitPut,py::arg("put"))
      .def("abort_put",&NpuPythonRuntime::AbortPut,py::arg("put"))
      .def("last_offload_checksum", &NpuPythonRuntime::LastChecksum,
           "Hash current CPU DDR via a separate RPC after offload. Call before freeing or "
           "overwriting the destination slots, the source's next ready(), or source shutdown.")
      .def_property_readonly("pending_offload", &NpuPythonRuntime::PendingOffload)
#endif
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
