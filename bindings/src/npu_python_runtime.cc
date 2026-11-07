#include "npu_python_runtime.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "access_table_python.h"
#include "batch_get_profile.h"
#include "access_operator.h"
#include "acl/acl.h"
#include "acl_python_utils.h"
#include "npu_runtime.h"
#include "npu_runtime_constructor.h"

namespace py = pybind11;

namespace kvaf {
namespace {

void ThrowIfError(const Status& status) {
  if (status.ok()) return;
  const std::string message = status.message().empty()
                                  ? "KVAF NPU operation failed"
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

// Shared identity has no Runtime back-reference. A surviving table keeps the
// cookie alive, so allocator address reuse cannot impersonate another Runtime.
struct AccessTableOwner {};

class DeviceAccessTable final {
 public:
  DeviceAccessTable(AccessTableHandle source, py::object addresses,
                    py::object valid, std::shared_ptr<const AccessTableOwner> owner)
      : source_(source), addresses_(std::move(addresses)), valid_(std::move(valid)),
        owner_(std::move(owner)) {}

  const AccessTableHandle& source() const { return source_; }
  py::object addresses() const { return addresses_; }
  py::object valid() const { return valid_; }
  bool OwnedBy(const std::shared_ptr<const AccessTableOwner>& owner) const {
    return owner_ == owner;
  }

 private:
  AccessTableHandle source_;
  py::object addresses_;
  py::object valid_;
  std::shared_ptr<const AccessTableOwner> owner_;
};

py::object ResolveStreamOwner(const py::object& stream) {
  if (!stream.is_none()) {
    return stream;
  }
  py::module_ torch = py::module_::import("torch");
  if (!py::hasattr(torch, "npu")) {
    throw std::runtime_error("torch.npu is unavailable for the default KVAF stream");
  }
  return torch.attr("npu").attr("current_stream")();
}


py::dict UbmemFusedContextDict(const UbmemFusedContext& context) {
  py::dict result;
  result["abi_version"] = context.abi_version;
  result["lease_id"] = context.lease_id;
  result["device_id"] = context.device_id;
  result["region_base"] = context.region_base;
  result["region_bytes"] = context.region_bytes;
  result["slot_bytes"] = context.slot_bytes;
  result["output_address"] = context.output_address;
  result["output_bytes"] = context.output_bytes;
  result["token_bytes"] = context.token_bytes;
  result["tokens_per_block"] = context.tokens_per_block;
  return result;
}

py::dict BatchGetProfileDict(const BatchGetProfile& profile) {
  py::dict result, host;
#define HOST_TIME(name) host[#name "_us"] = profile.name##_ns / 1000.0
  HOST_TIME(prepare);
  HOST_TIME(launch);
  HOST_TIME(status_enqueue);
  HOST_TIME(wait);
  HOST_TIME(finish);
  HOST_TIME(total);
#undef HOST_TIME
  result["host"] = host;
  py::list workers;
  std::uint64_t first = UINT64_MAX;
  std::uint64_t last = 0;
  for (unsigned index = 0; index < profile.worker_count; ++index) {
    first = std::min(first, profile.workers[index].enter_ns);
    last = std::max(last, profile.workers[index].exit_ns);
  }
  for (unsigned index = 0; index < profile.worker_count; ++index) {
    const auto& sample = profile.workers[index];
    py::dict worker;
    worker["worker"] = index;
    worker["entry_us"] = (sample.enter_ns - first) / 1000.0;
    worker["exit_us"] = (sample.exit_ns - first) / 1000.0;
    worker["wall_us"] = (sample.exit_ns - sample.enter_ns) / 1000.0;
#define WORKER_TIME(name) worker[#name "_us"] = sample.name##_ns / 1000.0
    WORKER_TIME(request);
    WORKER_TIME(input_cache);
    WORKER_TIME(descriptor_cache);
    WORKER_TIME(source_write);
    WORKER_TIME(notify);
    WORKER_TIME(read);
    WORKER_TIME(ack);
    WORKER_TIME(read_setup);
    WORKER_TIME(wr_build);
    WORKER_TIME(ready_wait);
    WORKER_TIME(post);
    WORKER_TIME(cq_wait);
    WORKER_TIME(scatter);
    WORKER_TIME(output_cache);
#undef WORKER_TIME
    worker["bytes"] = sample.bytes;
    worker["spans"] = sample.spans;
    worker["read_wqes"] = sample.read_wqes;
    worker["post_calls"] = sample.post_calls;
    worker["ready_queries"] = sample.ready_queries;
    worker["scatter_copies"] = sample.scatter_copies;
    workers.append(worker);
  }
  result["workers"] = workers;
  result["device_span_us"] = profile.worker_count ? (last - first) / 1000.0 : 0.0;
  return result;
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
                   std::uint32_t jettys_per_core,
                   std::vector<std::string> npu_eids,
                   bool adaptive_ub_device,
                   std::uint32_t adaptive_ub_miss_count,
                   std::string peer_ub_device,
                   std::vector<std::string> adaptive_peer_ub_devices,
                   std::uint32_t batch_get_max_inflight_total,
                   const std::string& transport, std::string aicpu_batch_get_mode,
                   bool reuse_device_ra, const std::string& ubmem_page_mode, int ubmem_numa_node) {
    NpuRuntimeOptions options{std::move(batch_get_kernel_object_path),
                              batch_get_token_bytes,
                              batch_get_tokens_per_block};
    if (transport == "urma") options.transport = Transport::kUrma;
    else if (transport == "ubmem") options.transport = Transport::kUbmem;
    else throw py::value_error("transport must be urma or ubmem");
    if (ubmem_page_mode == "thp") options.ubmem_page_mode = UbmemPageMode::kThp;
    else if (ubmem_page_mode == "base") options.ubmem_page_mode = UbmemPageMode::kBase;
    else throw py::value_error("ubmem_page_mode must be thp or base");
    options.ubmem_numa_node = ubmem_numa_node;
    options.batch_put_kernel_object_path =
        std::move(batch_put_kernel_object_path);
    options.device_id = device_id;
    options.npu_core_count = npu_core_count;
    options.jettys_per_core = jettys_per_core;
    options.aicpu_batch_get_mode = std::move(aicpu_batch_get_mode);
    options.batch_get_max_inflight_total = batch_get_max_inflight_total;
    options.reuse_device_ra = reuse_device_ra;
    options.npu_eids = std::move(npu_eids);
    options.adaptive_ub_device = adaptive_ub_device;
    options.adaptive_ub_miss_count = adaptive_ub_miss_count;
    options.peer_ub_device = std::move(peer_ub_device);
    options.adaptive_peer_ub_devices = std::move(adaptive_peer_ub_devices);
    native_ = std::make_unique<::kvaf::NpuRuntime>(std::move(options));
    aicpu_backend_ = native_->uses_aicpu();
  }

  ~NpuPythonRuntime() {
    if (completion_unknown_ || ubmem_fused_context_.lease_id != 0) {
      // A graph can still reference these storages after Python GC. Only
      // explicit successful release establishes that this lease is finished.
      for (auto& owner : ubmem_fused_owners_) (void)owner.release();
      (void)ubmem_fused_stream_.release();
      (void)native_.release();
    }
    if (completion_unknown_) {
      // Caller-owned Tensor/stream references were quarantined before the
      // error escaped. Keep their native arena and queued work alive as well.
      (void)native_.release();
    }
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
      for (auto& output : fused_output_views_) (void)output.release();
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
  const char* TransportName() const {
    return native_->transport() == Transport::kUbmem ? "ubmem" : "urma";
  }
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

  py::dict RegisterFusedBatchGetOutput(const py::dict& context,
                                      const py::object& output) {
    if (fused_context_.lease_id == 0U) {
      throw py::value_error("output registration requires a live fused lease");
    }
    const py::dict expected = FusedContextDict(fused_context_);
    for (const auto& item : expected) {
      if (!context.contains(item.first) ||
          context[item.first].cast<std::uint64_t>() !=
              item.second.cast<std::uint64_t>()) {
        throw py::value_error("pass the unmodified original fused lease");
      }
    }
    const auto tensor = GetNpuTensorView(output, "fused output view");
    const std::string device = "npu:" + std::to_string(DeviceId());
    if (py::str(output.attr("device")).cast<std::string>() != device ||
        tensor.shape.size() < 2 || tensor.shape.front() <= 0 ||
        tensor.bytes % static_cast<std::size_t>(tensor.shape.front()) != 0 ||
        tensor.bytes / static_cast<std::size_t>(tensor.shape.front()) != BatchGetTokenBytes()) {
      throw py::value_error("fused output view must have one token_bytes-sized row per slot on the Runtime device");
    }
    ThrowIfAclError(aclrtSetDevice(static_cast<int32_t>(DeviceId())),
                    "aclrtSetDevice before fused output registration");
    // Retain before native registration; failures cannot free registered HBM.
    fused_output_views_.push_back(output);
    FusedBatchGetContext view;
    ThrowIfError(native_->RegisterFusedBatchGetOutput(
        fused_context_, tensor.data, tensor.bytes, &view));
    return FusedContextDict(view);
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
    fused_output_views_.clear();
    fused_stream_ = py::object{};
  }

  py::dict AcquireUbmemFusedContext(const py::object& output,
                                    const py::object& stream) {
    if (native_->transport() != Transport::kUbmem) {
      throw py::value_error("UBmem fused context requires transport=ubmem");
    }
    if (completion_unknown_) throw std::runtime_error("Runtime completion is unknown");
    if (ubmem_fused_context_.lease_id != 0) {
      throw py::value_error("Runtime already owns a UBmem fused lease");
    }
    const auto tensor = CheckUbmemFusedOutput(output);
    py::object owner = stream;
    if (owner.is_none()) {
      owner = py::module_::import("torch").attr("npu").attr("current_stream")(DeviceId());
    }
    if (!py::hasattr(owner, "device")) {
      throw py::value_error("UBmem fused acquire requires a stream object with a device");
    }
    CheckUbmemDevice(owner, "UBmem fused stream");
    const aclrtStream acl_stream = acl_python::StreamFromPython(owner);
    ThrowIfAclError(aclrtSetDevice(static_cast<int32_t>(DeviceId())),
                    "aclrtSetDevice before UBmem fused acquire");
    // Retain before the native fence: unknown completion must not DECREF them.
    ubmem_fused_owners_.push_back(output);
    ubmem_fused_stream_ = std::move(owner);
    ubmem_fused_acl_stream_ = acl_stream;
    UbmemFusedContext context;
    const Status status = native_->AcquireUbmemFusedContext(
        tensor.data, tensor.bytes, acl_stream, &context);
    if (status.code() == StatusCode::kCompletionUnknown) completion_unknown_ = true;
    if (!status.ok()) {
      if (!completion_unknown_) {
        ubmem_fused_owners_.clear();
        ubmem_fused_stream_ = py::object{};
        ubmem_fused_acl_stream_ = nullptr;
      }
      ThrowIfError(status);
    }
    ubmem_fused_context_ = context;
    return UbmemFusedContextDict(context);
  }

  py::dict RegisterUbmemFusedOutput(const py::dict& context,
                                    const py::object& output) {
    CheckUbmemFusedOwner(context);
    const auto tensor = CheckUbmemFusedOutput(output);
    ThrowIfAclError(aclrtSetDevice(static_cast<int32_t>(DeviceId())),
                    "aclrtSetDevice before UBmem fused output registration");
    ubmem_fused_owners_.push_back(output);
    UbmemFusedContext view;
    const Status status = native_->RegisterUbmemFusedOutput(
        ubmem_fused_context_, tensor.data, tensor.bytes, &view);
    if (!status.ok()) {
      ubmem_fused_owners_.pop_back();
      ThrowIfError(status);
    }
    return UbmemFusedContextDict(view);
  }

  void RegisterUbmemFusedAccessTable(const py::dict& context,
                                     const py::object& table_owner) {
    CheckUbmemFusedOwner(context);
    const auto& table = table_owner.cast<const DeviceAccessTable&>();
    CheckAccessTableOwner(table);
    CheckUbmemDevice(table.addresses(), "UBmem fused table addresses");
    CheckUbmemDevice(table.valid(), "UBmem fused table valid");
    ThrowIfAclError(aclrtSetDevice(static_cast<int32_t>(DeviceId())),
                    "aclrtSetDevice before UBmem fused table registration");
    ubmem_fused_owners_.push_back(table_owner);
    const Status status = native_->RegisterUbmemFusedAccessTable(
        ubmem_fused_context_, table.source());
    if (!status.ok()) {
      ubmem_fused_owners_.pop_back();
      ThrowIfError(status);
    }
  }

  void RetainUbmemFusedGraph(const py::dict& context, const py::object& graph) {
    CheckUbmemFusedOwner(context);
    CheckUbmemDevice(graph, "UBmem fused graph");
    if (!py::hasattr(graph, "replay") || !PyCallable_Check(graph.attr("replay").ptr())) {
      throw py::type_error("UBmem fused graph must expose replay()");
    }
    ubmem_fused_owners_.push_back(graph);
  }

  void ReleaseUbmemFusedContext(const py::dict& context,
                                 const py::object& stream) {
    CheckUbmemFusedOwner(context);
    if (!stream.is_none()) {
      if (!py::hasattr(stream, "device")) {
        throw py::value_error("UBmem fused release requires the acquire stream object");
      }
      CheckUbmemDevice(stream, "UBmem fused release stream");
      if (acl_python::StreamFromPython(stream) != ubmem_fused_acl_stream_) {
        throw py::value_error("UBmem fused release must use its acquire stream");
      }
    }
    ThrowIfAclError(aclrtSetDevice(static_cast<int32_t>(DeviceId())),
                    "aclrtSetDevice before UBmem fused release");
    const Status status = native_->ReleaseUbmemFusedContext(
        ubmem_fused_context_, ubmem_fused_acl_stream_);
    if (status.code() == StatusCode::kCompletionUnknown) completion_unknown_ = true;
    ThrowIfError(status);
    ubmem_fused_context_ = {};
    ubmem_fused_acl_stream_ = nullptr;
    ubmem_fused_owners_.clear();
    ubmem_fused_stream_ = py::object{};
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
    if (put_in_flight_) throw py::value_error("another BatchPut is in flight");
    struct PutGuard {
      bool& active;
      explicit PutGuard(bool& value) : active(value) { active = true; }
      ~PutGuard() { active = false; }
    } guard(put_in_flight_);
    CheckUbmemDevice(source, "BatchPut source");
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
    py::object stream_owner = ResolveStreamOwner(stream);
    CheckUbmemDevice(stream_owner, "BatchPut stream");
    const aclrtStream acl_stream = acl_python::StreamFromPython(stream_owner);
    std::array<py::object, 8> owners{source, stream_owner};
    batch_put_request_.input =
        ConstBufferView{source_view.data, source_view.bytes, DeviceKind::kNpu};
    Status status;
    {
      py::gil_scoped_release release;
      status = native_->core().BatchPut(batch_put_request_, BatchPutExecutionOptions{acl_stream});
    }
    QuarantineUnknownCompletion(status, &owners);
    ThrowIfError(status);
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
                                      std::move(valid), access_table_owner_});
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
    const Status status = native_->core().BatchFree(BatchFreeRequest{std::move(copied_keys)});
    if (status.code() == StatusCode::kCompletionUnknown) completion_unknown_ = true;
    ThrowIfError(status);
  }

  void StartBatchGetKernelProfile(bool detailed) {
    ThrowIfError(native_->SetBatchGetKernelProfiling(true, detailed));
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
      item["ubmem_profile_detail"] = profile.ubmem_profile_detail;
      item["ubmem_setup_cycles"] = profile.ubmem_setup_cycles;
      item["ubmem_address_cycles"] = profile.ubmem_address_cycles;
      item["ubmem_submit_cycles"] = profile.ubmem_submit_cycles;
      item["ubmem_drain_cycles"] = profile.ubmem_drain_cycles;
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
  py::object BatchGet(const DeviceAccessTable& access_table,
                const py::object& indices, const py::object& offsets,
                const py::object& miss_count,
                const py::object& output_indices, const py::object& output,
                const py::object& stream, bool synchronize, bool profile) {
    CheckAccessTableOwner(access_table);
    return BatchGetImpl(access_table.source(), access_table.addresses(),
                 access_table.valid(), indices, offsets, miss_count, output_indices,
                 output, stream, synchronize, profile);
  }

  // This overload accepts a caller-owned device table directly. Since bare
  // Tensors carry no AccessTableHandle, Runtime validates their NPU layout but
  // cannot validate a corresponding Host AccessTable lifetime or entry length.
  py::object BatchGetFromTensors(const py::object& addresses,
                           const py::object& valid,
                           const py::object& indices,
                           const py::object& offsets,
                           const py::object& miss_count,
                           const py::object& output_indices,
                           const py::object& output,
                           const py::object& stream, bool synchronize, bool profile) {
    return BatchGetImpl(AccessTableHandle{}, addresses, valid, indices, offsets, miss_count,
                 output_indices, output, stream, synchronize, profile);
  }

 private:
  void CheckAccessTableOwner(const DeviceAccessTable& table) const {
    if (!table.OwnedBy(access_table_owner_)) {
      throw py::value_error("DeviceAccessTable belongs to another Runtime");
    }
  }

  NpuTensorView CheckUbmemFusedOutput(const py::object& output) const {
    CheckUbmemDevice(output, "UBmem fused output");
    const auto view = GetNpuTensorView(output, "UBmem fused output");
    if (view.shape.size() < 2 || view.shape.front() <= 0 ||
        view.bytes % static_cast<std::size_t>(view.shape.front()) != 0 ||
        view.bytes / static_cast<std::size_t>(view.shape.front()) != BatchGetTokenBytes()) {
      throw py::value_error("UBmem fused output requires one token_bytes-sized row per slot");
    }
    return view;
  }

  void CheckUbmemFusedOwner(const py::dict& context) const {
    if (completion_unknown_) throw std::runtime_error("Runtime completion is unknown; fused owners are retained");
    if (ubmem_fused_context_.lease_id == 0) throw py::value_error("Runtime has no active UBmem fused lease");
    ThrowIfError(native_->core().CheckFusedReadLease());
    const auto expected = UbmemFusedContextDict(ubmem_fused_context_);
    for (const auto& item : expected) {
      if (!context.contains(item.first) ||
          context[item.first].cast<std::uint64_t>() != item.second.cast<std::uint64_t>()) {
        throw py::value_error("pass the original unmodified UBmem fused owner context");
      }
    }
  }

  void CheckUbmemDevice(const py::object& object, const char* name) const {
    if (native_->transport() != Transport::kUbmem ||
        !py::hasattr(object, "device")) return;
    const std::string expected = "npu:" + std::to_string(DeviceId());
    if (py::str(object.attr("device")).cast<std::string>() != expected) {
      throw py::value_error(std::string(name) + " must belong to Runtime device " + expected);
    }
    // Raw ACL stream pointers expose no device. Their owner must enforce the
    // same-device contract; do not infer an ACL device from the pointer value.
  }

  void QuarantineUnknownCompletion(const Status& status,
                                   std::array<py::object, 8>* owners) noexcept {
    if (status.code() != StatusCode::kCompletionUnknown) return;
    // Allocate/copy these owners before submitting work. release() is then
    // allocation-free even when an uncertain launch coincides with host OOM.
    // No normal Python GC or Runtime destruction can prove DMA completion.
    for (auto& owner : *owners) (void)owner.release();
    completion_unknown_ = true;
  }

  py::object BatchGetImpl(const AccessTableHandle& access_table,
                    const py::object& addresses, const py::object& valid,
                    const py::object& indices, const py::object& offsets,
                    const py::object& miss_count,
                    const py::object& output_indices,
                    const py::object& output, const py::object& stream,
                    bool synchronize, bool profile) {
    if (profile && (!synchronize || !aicpu_backend_))
      throw py::value_error("profile=True requires the AICPU backend and synchronize=True");
    const auto begin = profile ? BatchGetProfileNowNs() : 0U;
    CheckUbmemDevice(addresses, "BatchGet addresses");
    CheckUbmemDevice(valid, "BatchGet valid");
    CheckUbmemDevice(indices, "BatchGet indices");
    CheckUbmemDevice(offsets, "BatchGet offsets");
    CheckUbmemDevice(miss_count, "BatchGet miss_count");
    CheckUbmemDevice(output_indices, "BatchGet output_indices");
    CheckUbmemDevice(output, "BatchGet output");
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
        ConstBufferView{miss_count_view.data, miss_count_view.bytes, DeviceKind::kNpu},
        ConstBufferView{output_index_view.data, output_index_view.bytes,
                        DeviceKind::kNpu},
        MutableBufferView{output_view.data, output_view.bytes, DeviceKind::kNpu},
        output_block_bytes,
        native_->batch_get_tokens_per_block(),
        AccessTableDeviceView{
            ConstBufferView{address_view.data, address_view.bytes, DeviceKind::kNpu},
            ConstBufferView{valid_view.data, valid_view.bytes, DeviceKind::kNpu}}};
    py::object stream_owner = ResolveStreamOwner(stream);
    CheckUbmemDevice(stream_owner, "BatchGet stream");
    const aclrtStream acl_stream = acl_python::StreamFromPython(stream_owner);
    std::array<py::object, 8> owners{
        addresses, valid, indices, offsets, miss_count, output_indices, output,
        stream_owner};
    auto timings = profile ? std::make_unique<BatchGetProfile>() : nullptr;
    const auto native_begin = profile ? BatchGetProfileNowNs() : 0U;
    const Status status = native_->core().BatchGet(
        request, BatchGetExecutionOptions{acl_stream, synchronize, timings.get()});
    QuarantineUnknownCompletion(status, &owners);
    ThrowIfError(status);
    if (!profile) return py::none();
    const auto end = BatchGetProfileNowNs();
    auto result = BatchGetProfileDict(*timings);
    result["binding_prepare_us"] = (native_begin - begin) / 1000.0;
    result["native_call_us"] = (end - native_begin) / 1000.0;
    result["binding_total_us"] = (end - begin) / 1000.0;
    return result;
  }

 public:
  void Shutdown() {
    if (put_in_flight_) throw py::value_error("finish BatchPut before shutdown");
    if (completion_unknown_) {
      throw std::runtime_error(
          "work completion is unknown; retain this Runtime and its Tensor storage until process teardown");
    }
    if (fused_context_.lease_id != 0U) {
      throw py::value_error(
          "release the fused BatchGet context successfully before shutdown; "
          "uncertain DMA completion requires retaining output and Runtime until process teardown");
    }
    if (ubmem_fused_context_.lease_id != 0) {
      throw py::value_error("release the UBmem fused lease before shutdown");
    }
    ThrowIfError(native_->Shutdown());
  }

 private:
  std::unique_ptr<::kvaf::NpuRuntime> native_;
  std::shared_ptr<const AccessTableOwner> access_table_owner_{
      std::make_shared<const AccessTableOwner>()};
  bool completion_unknown_{false};
  bool aicpu_backend_{false};
  bool put_in_flight_{false};
  // Runtime::BatchPut is synchronous, so these vectors can be reused after
  // each call and avoid allocator churn in the measured loop.
  BatchPutRequest batch_put_request_;
  FusedBatchGetContext fused_context_;
  py::object fused_output_;
  std::vector<py::object> fused_output_views_;
  py::object fused_stream_;
  aclrtStream fused_acl_stream_{nullptr};
  UbmemFusedContext ubmem_fused_context_;
  std::vector<py::object> ubmem_fused_owners_;
  py::object ubmem_fused_stream_;
  aclrtStream ubmem_fused_acl_stream_{nullptr};
};

}  // namespace

void BindNpuPythonRuntime(py::module_& module) {
  py::class_<DeviceAccessTable>(module, "DeviceAccessTable")
      .def_property_readonly("addresses", &DeviceAccessTable::addresses)
      .def_property_readonly("valid", &DeviceAccessTable::valid)
      .def_property_readonly("source_id", [](const DeviceAccessTable& table) {
        return table.source().opaque;
      });

  auto runtime = py::class_<NpuPythonRuntime>(module, "NpuRuntime");
  python_binding::BindNpuRuntimeConstructor(runtime);
  runtime.def("link_to", &NpuPythonRuntime::LinkTo, py::arg("address"),

           "Connect this NPU Runtime to a CPU Runtime service before initialize.")
      .def("initialize", &NpuPythonRuntime::Initialize, py::arg("ddr_bytes"),
           py::arg("slot_size"))
      .def("shutdown", &NpuPythonRuntime::Shutdown)
      .def("acquire_ubmem_fused_context",
           &NpuPythonRuntime::AcquireUbmemFusedContext,
           py::arg("output"), py::arg("stream") = py::none(),
           "Lease the worker-local HostDDR arena and an output tensor for UBmem fusion. "
           "Returns an independent mapping/geometry ABI with no URMA queues. "
           "Keep all launches ordered on the acquire stream. Do not resize retained storage. "
           "BatchFree fences the device at request retirement: first stop submitting that "
           "request and invalidate its addresses; later graph replays must not use freed keys.")
      .def("register_ubmem_fused_output", &NpuPythonRuntime::RegisterUbmemFusedOutput,
           py::arg("context"), py::arg("output"),
           "Retain another output under the original UBmem owner lease before capture. "
           "Returns a view sharing lease and arena, with this output's address and size.")
      .def("register_ubmem_fused_access_table", &NpuPythonRuntime::RegisterUbmemFusedAccessTable,
           py::arg("context"), py::arg("device_access_table"),
           "Optionally retain a KVAF DeviceAccessTable and its metadata pin until lease "
           "release. Its keys cannot be freed and its handle cannot be released meanwhile. "
           "Dynamic request address tables may instead use the request-retirement contract.")
      .def("retain_ubmem_fused_graph", &NpuPythonRuntime::RetainUbmemFusedGraph,
           py::arg("context"), py::arg("graph"),
           "Retain a captured graph until explicit UBmem lease release. The graph and its "
           "remaining inputs must belong to the same Runtime device. Never replay it after release.")
      .def("release_ubmem_fused_context", &NpuPythonRuntime::ReleaseUbmemFusedContext,
           py::arg("context"), py::arg("stream") = py::none(),
           "Pass the original owner dict. Drain device work, remove optional table reader "
           "pins, and release retained output/table/graph/stream owners. The caller must "
           "stop all future replays before release. Unknown completion retains all resources.")
      .def("acquire_fused_batch_get_context",
           &NpuPythonRuntime::AcquireFusedBatchGetContext,
           py::arg("output"), py::arg("stream") = py::none(),
           "Exclusively lease native BatchGet queues and register output HBM. "
           "Returns integer fields matching FusedBatchGetContext, including lease_id. "
           "output and stream are retained until successful release. All fused launches "
           "must run sequentially on this stream; do not resize/rebind output storage, "
           "reset queue cursors, or free source DDR while the lease is in use. "
           "stream=None captures the Runtime device's current torch.npu stream.")
      .def("register_fused_batch_get_output",
           &NpuPythonRuntime::RegisterFusedBatchGetOutput,
           py::arg("context"), py::arg("output"),
           "Setup-only output registration under the original fused lease. "
           "Returns a view with shared persistent queues; all launches must use "
           "the acquire stream sequentially. Register before capture. Release "
           "the original lease to release all views; do not resize their tensors.")
      .def("release_fused_batch_get_context",
           &NpuPythonRuntime::ReleaseFusedBatchGetContext,
           py::arg("context"), py::arg("stream") = py::none(),
           "Fence the acquire stream and check native queue completion before release. "
           "Pass the unmodified acquire dict. stream=None uses the retained acquire "
           "stream, not the current stream. Failure retains all resources and blocks "
           "shutdown; unknown DMA completion requires process/device teardown.")
      .def_property_readonly("initialized", &NpuPythonRuntime::Initialized)
      .def_property_readonly("transport", &NpuPythonRuntime::TransportName)
      .def_property_readonly("device_id", &NpuPythonRuntime::DeviceId)
      .def_property_readonly("npu_core_count", &NpuPythonRuntime::NpuCoreCount)
      .def_property_readonly("jettys_per_core", &NpuPythonRuntime::JettysPerCore)
      .def_property_readonly("npu_eids", &NpuPythonRuntime::NpuEids)
      .def_property_readonly("batch_get_token_bytes",
                             &NpuPythonRuntime::BatchGetTokenBytes)
      .def_property_readonly("batch_get_tokens_per_block",
                             &NpuPythonRuntime::BatchGetTokensPerBlock)
      .def_property_readonly("remote_region", &NpuPythonRuntime::RemoteRegion)
      .def_property_readonly("supports_background_put", [](const NpuPythonRuntime&) { return true; })
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
           py::arg("detailed") = false,
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
           py::arg("profile") = false,
           "Submit BatchGet from NPU Tensors. stream=None uses torch.npu.current_stream(); "
           "the default returns after enqueueing work on that stream. AICPU profile=True "
           "requires synchronize=True and returns a stage timing dictionary.")
      .def("batch_get", &NpuPythonRuntime::BatchGetFromTensors,
           py::arg("addresses"), py::arg("valid"), py::arg("indices"),
           py::arg("offsets"), py::arg("miss_count"), py::arg("output_indices"),
           py::arg("output"), py::arg("stream") = py::none(),
           py::arg("synchronize") = false,
           py::arg("profile") = false,
           "Submit BatchGet from caller-owned NPU addresses and valid Tensors. "
           "stream=None uses torch.npu.current_stream(); the default returns "
           "after enqueueing work on that stream. AICPU profile=True requires "
           "synchronize=True and returns a stage timing dictionary.");
}

}  // namespace kvaf
