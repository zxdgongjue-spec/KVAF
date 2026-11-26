#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <thread>
#include "npu_python_runtime.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
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
#include "npu_tensor_python.h"
#include "hbm_read_python.h"
#include "remote_offload_python.h"

namespace py = pybind11;

namespace kvaf {
namespace {

class GetCallTrace {
 public:
  struct Row {
    GetExecutionConfig config;
    const char* path;
    BatchGetHostTiming timing;
  };
  explicit GetCallTrace(std::string path) : path_(std::move(path)) { rows_.reserve(256); }
  ~GetCallTrace() { Flush(); }
  void Record(GetExecutionConfig config, const char* path, BatchGetHostTiming timing) {
    rows_.push_back({config, path, timing});
  }
  void Flush() {
    if (rows_.empty()) return;
    std::ofstream output(path_, std::ios::app);
    if (!output) {
      std::fprintf(stderr, "[KVAF] cannot write Get call trace: %s\n", path_.c_str());
      return;
    }
    const std::pair<const char*, std::uint64_t BatchGetHostTiming::*> fields[] = {
#define HOST_FIELD(name) {#name, &BatchGetHostTiming::name}
        HOST_FIELD(tensor_views_ns), HOST_FIELD(request_validation_ns), HOST_FIELD(stream_ns),
        HOST_FIELD(refs_ns), HOST_FIELD(binding_lock_ns), HOST_FIELD(set_device_ns),
        HOST_FIELD(core_call_ns), HOST_FIELD(owner_update_ns), HOST_FIELD(total_ns),
        HOST_FIELD(core_lifecycle_ns), HOST_FIELD(core_table_ns), HOST_FIELD(core_direction_ns),
        HOST_FIELD(core_previous_drain_ns), HOST_FIELD(core_validate_ns), HOST_FIELD(core_dispatch_ns),
        HOST_FIELD(core_finish_ns), HOST_FIELD(core_total_ns),
#undef HOST_FIELD
    };
    output.seekp(0, std::ios::end);
    if (output.tellp() == 0) {
      output << "event,index,algorithm,path";
      for (const auto& field : fields) output << ',' << field.first;
      output << '\n';
    }
    for (const auto& row : rows_) {
      const char* algorithm = row.config.transport == Transport::Ubmem ? "ubmem" :
          row.config.executor == Executor::Aiv ? "aiv_direct" :
          row.config.algorithm == GetAlgorithm::Scatter ? "scatter" : "gather";
      output << "submit," << ++next_index_ << ',' << algorithm << ',' << row.path;
      for (const auto& field : fields) output << ',' << row.timing.*(field.second);
      output << '\n';
    }
    rows_.clear();
  }
 private:
  std::string path_;
  std::vector<Row> rows_;
  std::uint64_t next_index_{};
};

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

using python_binding::NpuTensorView;
using python_binding::GetNpuTensorView;

using python_binding::TensorStorageOwner;

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

class PythonGetGraphResources;

class NpuPythonRuntime final {
 public:
  NpuPythonRuntime(py::object batch_get_token_bytes,
                   py::object batch_get_tokens_per_block, std::uint32_t device_id,
                   const std::string& transport_mode, py::object default_put,
                   py::object default_get, py::object put_resources, py::object get_resources,
                   std::vector<std::string> npu_eids, std::string peer_ub_device,
                   py::object cpu_pull_options, py::object npu_pull_options,
                   std::vector<std::uint32_t> get_core_fe_map) {
    NpuRuntimeOptions options;
    if (batch_get_token_bytes.is_none() != batch_get_tokens_per_block.is_none())
      throw py::value_error("cache layout requires both token_bytes and tokens_per_block");
    if (!batch_get_token_bytes.is_none()) {
      options.batch_get_token_bytes = batch_get_token_bytes.cast<std::uint32_t>();
      options.batch_get_tokens_per_block = batch_get_tokens_per_block.cast<std::uint32_t>();
      if (!options.batch_get_token_bytes || !options.batch_get_tokens_per_block)
        throw py::value_error("explicit cache layout dimensions must be positive");
    }
    if (!cpu_pull_options.is_none()) options.cpu_pull_options = cpu_pull_options.cast<CpuPullOptions>();
    if (!npu_pull_options.is_none()) options.npu_pull_options = npu_pull_options.cast<NpuPullOptions>();
    if (transport_mode == "urma") options.transport_mode = TransportMode::Urma;
    else if (transport_mode == "ubmem") options.transport_mode = TransportMode::Ubmem;
    else if (transport_mode == "urma+ubmem") options.transport_mode = TransportMode::UrmaAndUbmem;
    else throw py::value_error("transport_mode must be urma, ubmem or urma+ubmem");
    if (!put_resources.is_none())
      options.put_resources = put_resources.cast<decltype(options.put_resources)>();
    if (!get_resources.is_none())
      options.get_resources = get_resources.cast<decltype(options.get_resources)>();
    if (!default_put.is_none()) options.default_put = default_put.cast<PutExecutionConfig>();
    if (!default_get.is_none()) options.default_get = default_get.cast<GetExecutionConfig>();
    options.device_id = device_id;
    options.npu_eids = std::move(npu_eids);
    options.peer_ub_device = std::move(peer_ub_device);
    options.get_core_fe_map = std::move(get_core_fe_map);
    native_ = std::make_unique<::kvaf::NpuRuntime>(std::move(options));
    if (const char* path = std::getenv("KVAF_GET_DISPATCH_TRACE"); path && *path)
      get_call_trace_ = std::make_unique<GetCallTrace>(std::string(path) + ".host.csv");
  }

  void LinkTo(std::string address) {
    ThrowIfError(native_->LinkTo(LinkTarget{std::move(address)}));
  }

  void Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
    ThrowIfError(native_->Initialize(ddr_bytes, slot_size));
  }

  void InitializeBase() { ThrowIfError(native_->Initialize()); }
  void InitializeDdr(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
    ThrowIfError(native_->InitializeDdr(ddr_bytes, slot_size));
  }
  void ShutdownDdr() {
    Status status;
    std::unique_lock<std::mutex> get(get_mutex_, std::defer_lock), put(put_mutex_, std::defer_lock);
    { py::gil_scoped_release release; std::lock(get, put); status = native_->ShutdownDdr(); }
    if (status.ok()) {
      pending_get_refs_.clear(); output_refs_.clear(); pending_put_ref_ = py::object{};
      put_input_refs_.clear(); pending_put_input_refs_.clear();
    }
    ThrowIfError(status);
  }

#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  py::object BatchCpuPull(const py::object& keys, const ReadyToken& ready, const py::object& offsets,
                         const py::object& lengths, const py::object& timeout, bool profiling) {
    using Clock = std::chrono::steady_clock;
    const auto begin = profiling ? Clock::now() : Clock::time_point{};
    const auto key_ids = CopyPrepareKeys(keys);
    std::vector<std::uint64_t> sources, sizes;
    CopyCpuUint64SequenceInto(offsets, "src_offsets", &sources);
    CopyCpuUint64SequenceInto(lengths, "lengths", &sizes);
    const auto timeout_ms = timeout.is_none() ? UINT32_MAX : timeout.cast<std::uint32_t>();
    const double binding_prepare_us = profiling
        ? std::chrono::duration<double, std::micro>(Clock::now() - begin).count() : 0;
    CpuPullProfile profile;
    Status status;
    { py::gil_scoped_release release;
      status = native_->BatchCpuPull(key_ids, ready, sources, sizes, timeout_ms, profiling ? &profile : nullptr); }
    ThrowIfError(status);
    if (!profiling) return py::none();
    py::dict timing;
    timing["binding_prepare_us"] = binding_prepare_us;
    timing["native_total_us"] = profile.total_us;
    timing["validation_us"] = profile.validation_us;
    timing["prepare_put_us"] = profile.prepare_put_us;
    timing["dst_offsets_us"] = profile.dst_offsets_us;
    timing["submit_wait_cpu_pull_us"] = profile.submit_wait_cpu_pull_us;
    timing["wait_cpu_pull_us"] = profile.wait_cpu_pull_us;
    timing["commit_put_us"] = profile.commit_put_us;
    timing["native_other_us"] = std::max(0.0, profile.total_us - profile.validation_us
        - profile.prepare_put_us - profile.dst_offsets_us - profile.submit_wait_cpu_pull_us
        - profile.wait_cpu_pull_us - profile.commit_put_us);
    const auto& detail = profile.submit;
    py::dict submit;
    submit["snapshot_us"] = detail.snapshot_us;
    submit["cover_check_us"] = detail.cover_check_us;
    submit["span_prepare_us"] = detail.span_prepare_us;
    submit["request_save_us"] = detail.request_save_us;
    submit["channel_setup_us"] = detail.channel_setup_us;
    submit["send_us"] = detail.send_us;
    submit["receive_us"] = detail.receive_us;
    submit["response_check_us"] = detail.response_check_us;
    submit["other_us"] = profile.submit_wait_cpu_pull_us - detail.snapshot_us - detail.cover_check_us
        - detail.span_prepare_us - detail.request_save_us - detail.channel_setup_us
        - detail.send_us - detail.receive_us - detail.response_check_us;
    timing["submit_detail"] = std::move(submit);
    timing["server_detail"] = py::none();
    timing["rpc_gap_us"] = py::none();
    if (detail.server_available) {
      const auto& cpu = detail.server;
      py::dict server;
      server["span_receive_us"] = cpu.receive_ns / 1000.0;
      server["control_lock_us"] = cpu.control_lock_ns / 1000.0;
      server["validation_us"] = cpu.validation_ns / 1000.0;
      server["worker_start_us"] = cpu.worker_start_ns / 1000.0;
      server["backend_read_us"] = cpu.backend_read_ns / 1000.0;
      server["completion_us"] = cpu.completion_ns / 1000.0;
      server["reply_prepare_us"] = cpu.reply_prepare_ns / 1000.0;
      server["total_us"] = cpu.total_ns / 1000.0;
      timing["server_detail"] = std::move(server);
      timing["rpc_gap_us"] = detail.send_us + detail.receive_us - cpu.total_ns / 1000.0;
    }
    py::dict result;
    result["profile"] = std::move(timing);
    return result;
  }
  void CancelCpuPull(const ReadyToken& ready) {
    Status status;
    { py::gil_scoped_release release; status = native_->CancelCpuPull(ready); }
    ThrowIfError(status);
  }
  void ResolveCpuPull(const py::object& timeout) {
    const auto timeout_ms = timeout.is_none() ? UINT32_MAX : timeout.cast<std::uint32_t>();
    Status status;
    { py::gil_scoped_release release; status = native_->ResolveCpuPull(timeout_ms); }
    ThrowIfError(status);
  }
  py::dict OffloadRegion(std::optional<std::uint32_t> source_physical_device_id) {
    OffloadRegionTicket ticket; Status status;
    { py::gil_scoped_release release; status = native_->GetOffloadRegion(&ticket, source_physical_device_id); }
    ThrowIfError(status); return remote_python::Pack(ticket);
  }
  PutHandle PreparePut(const py::object& keys,const py::object& lengths) {
    std::vector<std::uint64_t> sizes;
    CopyCpuUint64SequenceInto(lengths,"lengths",&sizes);
    const auto copied_keys = CopyPrepareKeys(keys);
    PutHandle put; Status status;
    { py::gil_scoped_release release; status = native_->PreparePut(copied_keys,sizes,&put); }
    ThrowIfError(status); return put;
  }
  void SubmitCpuPull(const ReadyToken& ready,const PutHandle& put,
                           const py::object& src,const py::object& dst_offsets,const py::object& lengths) {
    std::vector<std::uint64_t> sources, destinations, sizes;
    CopyCpuUint64SequenceInto(src, "src_offsets", &sources);
    CopyCpuUint64SequenceInto(dst_offsets, "dst_offsets", &destinations);
    CopyCpuUint64SequenceInto(lengths, "lengths", &sizes);
    Status status;
    { py::gil_scoped_release release; status = native_->SubmitCpuPull(ready,put,sources,destinations,sizes); }
    ThrowIfError(status);
  }
  CpuPullState QueryCpuPull(const PutHandle& put) {
    CpuPullState state; Status status;
    { py::gil_scoped_release release; status = native_->QueryCpuPull(put,&state); }
    ThrowIfError(status); return state;
  }
  CpuPullState WaitCpuPull(const PutHandle& put,std::uint32_t timeout) {
    CpuPullState state; Status status;
    {py::gil_scoped_release release;status=native_->WaitCpuPull(put,timeout,&state);}
    ThrowIfError(status); return state;
  }
  CpuPullState CommitPut(const PutHandle& put) {
    Status status; { py::gil_scoped_release release; status = native_->CommitPut(put); }
    ThrowIfError(status); return put.state();
  }
  CpuPullState AbortPut(const PutHandle& put) {
    Status status; { py::gil_scoped_release release; status = native_->AbortPut(put); }
    ThrowIfError(status); return put.state();
  }
  bool PendingOffload() const { py::gil_scoped_release release; return native_->pending_offload(); }
  std::uint64_t LastChecksum() const {
    std::uint64_t hash{}; Status status;
    { py::gil_scoped_release release; status = native_->LastOffloadChecksum(&hash); }
    ThrowIfError(status); return hash;
  }
#endif

  py::dict ListenHbm(const std::string& address,const std::string& advertised,const std::string& eid) {
    HbmReceiverTicket ticket;Status status;
    {py::gil_scoped_release release;status=native_->ListenHbm(address,advertised,eid,&ticket);}
    ThrowIfError(status);return hbm_read_python::Pack(ticket);
  }
  void CancelRawAddr(const ReadyToken& ready) {
    Status status;{py::gil_scoped_release release;status=native_->CancelRawAddr(ready);}
    ThrowIfError(status);
  }
  void ResolveRawAddr(const ReadyToken& ready) {
    Status status;{py::gil_scoped_release release;status=native_->ResolveRawAddr(ready);}
    ThrowIfError(status);
  }
  py::dict BatchGetRawArrays(const ReadyToken& ready,const py::object& src,const py::object& dst,
                         const py::object& lengths,py::object output,py::object stream,bool profiling,bool validate_ranges,
                         bool synchronize) {
    using Clock = std::chrono::steady_clock;
    const auto begin=profiling?Clock::now():Clock::time_point{};
    auto previous=begin;
    struct {
      double checks{},src{},dst{},lengths{},output{},stream{},retain{},native{},release{},result{},profile{},report{};
    } binding;
    // Adjacent boundaries keep stages disjoint. No clock reads with profile off.
    auto mark=[&](double& stage) {
      if(profiling) {
        const auto now=Clock::now();
        stage+=std::chrono::duration<double,std::micro>(now-previous).count();previous=now;
      }
    };
    const auto* source=std::get_if<NpuPullSourceDesc>(&ready.source);
    if(!source)throw py::value_error("raw READ requires an NPU Pull ReadyToken");
    mark(binding.checks);
    auto array = [&](const py::object& tensor, const char* name) {
      const auto view = python_binding::GetNpuInt64TensorView(tensor, name, native_->device_id());
      return NpuInt64ArrayView{static_cast<const std::int64_t*>(view.data), view.bytes / sizeof(std::int64_t)};
    };
    const auto sources = array(src, "src_offsets");
    mark(binding.src);
    const auto destinations = array(dst, "dst_offsets");
    mark(binding.dst);
    const auto sizes = array(lengths, "lengths");
    mark(binding.lengths);
    if (sources.size != destinations.size || sources.size != sizes.size || sources.size > 4096)
      throw py::value_error("raw READ metadata must have equal lengths in 1..4096");
    mark(binding.checks);
    const auto view=GetNpuTensorView(output,"output",false);
    if(view.device.attr("index").cast<std::uint32_t>()!=native_->device_id())
      throw py::value_error("raw READ output belongs to another NPU");
    mark(binding.output);
    if(stream.is_none())stream=py::module_::import("torch").attr("npu").attr("current_stream")(native_->device_id());
    const auto stream_device=py::getattr(stream,"device",py::object{});
    if(stream_device && stream_device.attr("index").cast<std::uint32_t>()!=native_->device_id())
      throw py::value_error("raw READ stream belongs to another NPU");
    const auto acl_stream=acl_python::StreamFromPython(stream);
    mark(binding.stream);
    // Keep registered output storage until successful Runtime shutdown. Normal
    // asynchronous metadata/stream ownership remains with the application.
    // This Host lock only orders submission/owner publication against shutdown;
    // it does not wait for device completion or extend captured graph lifetime.
    std::shared_lock<std::shared_mutex> raw(raw_lifecycle_mutex_, std::defer_lock);
    { py::gil_scoped_release release; raw.lock(); }
    const auto source_key = py::make_tuple(source->source_id, source->source_epoch);
    if (!raw_outputs_.contains(source_key)) raw_outputs_[source_key] = py::dict();
    auto outputs = raw_outputs_[source_key].cast<py::dict>();
    // Retain before native setup: even a failed diagnostic step can leave a
    // cached output registration. Match opt's conservative shutdown lifetime.
    outputs[py::make_tuple(reinterpret_cast<std::uintptr_t>(view.data), view.bytes)] =
        TensorStorageOwner(output);
    HbmReadProfile profile;
    mark(binding.retain);
    Status status;
    {
      py::gil_scoped_release release;
      status = native_->BatchGetRawAddr(ready, sources, destinations, sizes,
          {view.data, view.bytes, DeviceKind::kNpu}, acl_stream,
          profiling ? &profile : nullptr, validate_ranges, synchronize);
    }
    mark(binding.native);
    // Only this call's CompletionUnknown means its metadata may still be in
    // use. A concurrent Busy rejection must not claim this source's owners.
    if (status.code() == StatusCode::kCompletionUnknown) {
      if (!raw_failed_inputs_.contains(source_key)) raw_failed_inputs_[source_key] = py::list();
      raw_failed_inputs_[source_key].cast<py::list>().append(py::make_tuple(TensorStorageOwner(src),
          TensorStorageOwner(dst), TensorStorageOwner(lengths), TensorStorageOwner(output), stream));
    }
    ThrowIfError(status);
    mark(binding.release);
    py::dict result; result["state"]=synchronize?"done":"submitted"; result["sequence"]=ready.sequence;
    mark(binding.result);
    if(profiling) {
      py::dict timing;
      timing["host_total_us"]=profile.total_us;
      double phases=0;
#define RAW_PHASE(name) timing[#name]=profile.name; phases+=profile.name
      RAW_PHASE(validation_us); RAW_PHASE(buffers_us); RAW_PHASE(register_output_us);
      RAW_PHASE(event_record_us); RAW_PHASE(launch_us); RAW_PHASE(synchronize_us); RAW_PHASE(event_query_us);
      RAW_PHASE(status_us); RAW_PHASE(cleanup_us);
#undef RAW_PHASE
      timing["host_other_us"]=std::max(0.0,profile.total_us-phases);
      timing["kernel_event_us"]=profile.kernel_event_valid?py::cast(profile.kernel_event_us):py::none();
      timing["kernel_event_error"]=profile.event_error;
      result["profile"]=timing;
      mark(binding.profile);
      // Account for most of the instrumentation's own dictionary work. Only
      // the final report/total fields and function return remain outside total.
#define RAW_BINDING_PHASE(name) timing["binding_" #name "_us"]=binding.name
      RAW_BINDING_PHASE(checks); RAW_BINDING_PHASE(src); RAW_BINDING_PHASE(dst);
      RAW_BINDING_PHASE(lengths); RAW_BINDING_PHASE(output); RAW_BINDING_PHASE(stream);
      RAW_BINDING_PHASE(retain); RAW_BINDING_PHASE(native); RAW_BINDING_PHASE(release);
      RAW_BINDING_PHASE(result); RAW_BINDING_PHASE(profile);
#undef RAW_BINDING_PHASE
      mark(binding.report);
      timing["binding_report_us"]=binding.report;
      timing["binding_total_us"]=std::chrono::duration<double,std::micro>(previous-begin).count();
    }
    return result;
  }
  bool PendingRawAddr() const { py::gil_scoped_release release; return native_->pending_raw_addr(); }

  bool DdrInitialized() const { return native_->ddr_initialized(); }
  bool Initialized() const { return native_->initialized(); }
  std::uint32_t DeviceId() const { return native_->device_id(); }
  ExecutionResources GetResources(GetExecutionConfig config) const { return native_->GetResources(config); }
  const std::vector<std::string>& NpuEids() const { return native_->npu_eids(); }
  std::uint32_t BatchGetTokenBytes() const {
    return native_->batch_get_token_bytes();
  }
  std::uint32_t BatchGetTokensPerBlock() const {
    return native_->batch_get_tokens_per_block();
  }

  py::dict RemoteRegion() const {
    if (!DdrInitialized()) throw std::runtime_error("DDR is not initialized");
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
                const py::object& stream, std::optional<PutExecutionConfig> execution,
                std::uint64_t input_lease) {
    BatchPutRequest batch_put_request;
    if (!DdrInitialized()) {
      throw std::runtime_error("DDR is not initialized");
    }
    CopyPrepareKeysInto(keys, &batch_put_request.keys);
    CopyCpuUint64SequenceInto(input_offsets, "BatchPut input_offsets",
                              &batch_put_request.input_offsets);
    CopyCpuUint64SequenceInto(lengths, "BatchPut lengths",
                              &batch_put_request.lengths);
    if (batch_put_request.keys.empty() ||
        batch_put_request.keys.size() !=
            batch_put_request.input_offsets.size() ||
        batch_put_request.keys.size() != batch_put_request.lengths.size()) {
      throw py::value_error(
          "BatchPut keys, input_offsets, and lengths must be non-empty and equal in size");
    }
    const NpuTensorView source_view = GetNpuTensorView(source, "BatchPut source");
    const aclrtStream acl_stream = ResolveStream(stream);
    batch_put_request.input =
        ConstBufferView{source_view.data, source_view.bytes, DeviceKind::kNpu};
    // Allocate Python ownership before submission. Moving this reference after
    // an uncertain DMA result must not need another allocation.
    py::object refs = py::make_tuple(source, stream);
    Status status;
    std::exception_ptr failure;
    bool pending_before = false, pending_after = false;
    std::unique_lock<std::mutex> direction(put_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      direction.lock();
      pending_before = native_->core().HasPendingPut();
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS) status = Status::Internal("aclrtSetDevice failed");
      else {
        try {
          status = native_->core().BatchPut(
              batch_put_request, BatchPutExecutionOptions{acl_stream, execution, input_lease});
        } catch (...) {
          failure = std::current_exception();
        }
      }
      pending_after = native_->core().HasPendingPut();
    }
    if (!pending_before && pending_after)
      pending_put_ref_ = std::move(refs);
    if (failure) std::rethrow_exception(failure);
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

  void ConfigureGetKernelProfile(bool enabled, bool detailed,
                                 std::optional<GetExecutionConfig> config) {
    Status status;
    std::unique_lock<std::mutex> lock(get_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      lock.lock();
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS)
        status = Status::Internal("aclrtSetDevice failed");
      else status = native_->SetBatchGetKernelProfiling(enabled, detailed, config);
    }
    ThrowIfError(status);
  }
  void StartBatchGetKernelProfile(bool detailed, std::optional<GetExecutionConfig> config) {
    ConfigureGetKernelProfile(true, detailed, config);
  }

  py::list BatchGetKernelProfileSnapshot(std::optional<GetExecutionConfig> config,
                                         std::uint64_t graph_lease = 0) {
    std::vector<kvaf::BatchGetKernelProfile> profiles;
    Status status;
    std::unique_lock<std::mutex> lock(get_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      lock.lock();
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS)
        status = Status::Internal("aclrtSetDevice failed");
      else status = native_->CopyBatchGetKernelProfiles(&profiles, config, graph_lease);
    }
    ThrowIfError(status);
    py::list result;
    for (std::size_t core = 0; core < profiles.size(); ++core) {
      const kvaf::BatchGetKernelProfile& profile = profiles[core];
      py::dict item;
      item["core"] = core;
      if (profile.device_start_ns) {
        item["device_start_ns"] = profile.device_start_ns;
        item["device_finish_ns"] = profile.device_finish_ns;
        BatchGetProfile detailed{};
        detailed.worker_count = 1;
        detailed.workers[0] = profile.aicpu_profile;
        auto fields = BatchGetProfileDict(detailed);
        auto workers = fields["workers"].cast<py::list>();
        auto stages = workers[0].cast<py::dict>();
        stages["worker"] = core;
        item["stages"] = stages;
      }
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

  py::list StopBatchGetKernelProfile(std::optional<GetExecutionConfig> config) {
    py::list result = BatchGetKernelProfileSnapshot(config);
    ConfigureGetKernelProfile(false, false, config);
    return result;
  }

  // KVAF creates no output Tensor. Managed calls retain request storage through
  // completion; explicit output bindings retain output until successful release.
  // Legacy calls require caller-owned tensors to outlive stream completion.
  py::object BatchGet(const DeviceAccessTable& access_table, const py::object& indices,
                      const py::object& offsets, const py::object& miss_count,
                      const py::object& output_indices, const py::object& output,
                      const py::object& stream, bool synchronize, bool profile,
                      std::optional<GetExecutionConfig> execution, std::uint64_t graph_lease,
                      std::uint64_t output_lease) {
    return BatchGetImpl(access_table.source(), access_table.addresses(), access_table.valid(),
                        indices, offsets, miss_count, output_indices, output, stream, synchronize,
                        profile, execution, graph_lease, output_lease);
  }

  // This overload accepts a caller-owned device table directly. Since bare
  // Tensors carry no AccessTableHandle, Runtime validates their NPU layout but
  // cannot validate a corresponding Host AccessTable lifetime or entry length.
  py::object BatchGetFromTensors(const py::object& addresses, const py::object& valid,
                                 const py::object& indices, const py::object& offsets,
                                 const py::object& miss_count, const py::object& output_indices,
                                 const py::object& output, const py::object& stream,
                                 bool synchronize, bool profile,
                                 std::optional<GetExecutionConfig> execution,
                                 std::uint64_t graph_lease, std::uint64_t output_lease) {
    return BatchGetImpl(AccessTableHandle{}, addresses, valid, indices, offsets, miss_count,
                        output_indices, output, stream, synchronize, profile, execution,
                        graph_lease, output_lease);
  }

 private:
  py::object BatchGetImpl(const AccessTableHandle& access_table, const py::object& addresses,
                          const py::object& valid, const py::object& indices,
                          const py::object& offsets, const py::object& miss_count,
                          const py::object& output_indices, const py::object& output,
                          const py::object& stream, bool synchronize, bool profile,
                          std::optional<GetExecutionConfig> execution, std::uint64_t graph_lease,
                          std::uint64_t output_lease) {
    BatchGetHostTiming host{};
    const auto host_begin = get_call_trace_ ? BatchGetProfileNowNs() : 0;
    auto phase = host_begin;
    const auto mark = [&](std::uint64_t BatchGetHostTiming::*field) {
      if (!get_call_trace_) return;
      const auto now = BatchGetProfileNowNs();
      host.*field += now - phase;
      phase = now;
    };
    if (profile) {
      const auto selected = execution ? execution : native_->default_get();
      if (!synchronize || !selected || selected->executor != Executor::Aicpu)
        throw py::value_error("profile=True requires the AICPU backend and synchronize=True");
    }
    const auto begin = profile ? BatchGetProfileNowNs() : 0U;
    const NpuTensorView index_view = GetNpuTensorView(indices, "indices");
    const NpuTensorView offset_view = GetNpuTensorView(offsets, "offsets");
    const NpuTensorView miss_count_view =
        GetNpuTensorView(miss_count, "miss_count");
    const NpuTensorView output_index_view =
        GetNpuTensorView(output_indices, "output_indices");
    const NpuTensorView output_view = GetNpuTensorView(output, "output");
    const NpuTensorView address_view = GetNpuTensorView(addresses, "addresses");
    const NpuTensorView valid_view = GetNpuTensorView(valid, "valid");
    mark(&BatchGetHostTiming::tensor_views_ns);
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
    mark(&BatchGetHostTiming::request_validation_ns);
    const aclrtStream acl_stream = ResolveStream(stream);
    mark(&BatchGetHostTiming::stream_ns);
    auto timings = profile ? std::make_unique<BatchGetProfile>() : nullptr;
    const auto native_begin = profile ? BatchGetProfileNowNs() : 0U;
    auto refs = py::make_tuple(addresses, valid, indices, offsets, miss_count, output_indices, output, stream);
    mark(&BatchGetHostTiming::refs_ns);
    Status status;
    std::unique_lock<std::mutex> direction(get_mutex_, std::defer_lock);
    decltype(graph_refs_)::iterator graph_ref;
    {
      py::gil_scoped_release release;
      direction.lock();
    }
    mark(&BatchGetHostTiming::binding_lock_ns);
    ApplyGraphScope(output, output_view, index_view.bytes / sizeof(std::int32_t),
                    acl_stream, synchronize, profile, execution, graph_lease, output_lease);
    if (graph_lease) {
      graph_ref = graph_refs_.find(graph_lease);
      if (graph_ref != graph_refs_.end())
        graph_ref->second.reserve(graph_ref->second.size() + 1);
    }
    {
      py::gil_scoped_release release;
      aclError device_status;
      {
        BatchGetProfileScope device(get_call_trace_ ? &host.set_device_ns : nullptr);
        device_status = aclrtSetDevice(native_->device_id());
      }
      if (device_status != ACL_SUCCESS) status = Status::Internal("aclrtSetDevice failed");
      else {
        BatchGetProfileScope call(get_call_trace_ ? &host.core_call_ns : nullptr);
        status = native_->core().BatchGet(
            request, BatchGetExecutionOptions{acl_stream, synchronize, timings.get(), execution,
                                              graph_lease, output_lease}, get_call_trace_ ? &host : nullptr);
      }
    }
    const auto owner_begin = get_call_trace_ ? BatchGetProfileNowNs() : 0;
    if (graph_lease) {
      if (graph_ref != graph_refs_.end()) graph_ref->second.push_back(std::move(refs));
    } else {
      if (status.ok()) pending_get_refs_.clear();  // Native completed the preceding Get.
      if (!status.ok() || !synchronize) pending_get_refs_.push_back(std::move(refs));
    }
    FinishGraphScopeGet(status.ok());
    ThrowIfError(status);
    if (get_call_trace_) {
      host.owner_update_ns = BatchGetProfileNowNs() - owner_begin;
      host.total_ns = BatchGetProfileNowNs() - host_begin;
      const auto selected = execution ? execution : native_->default_get();
      if (selected) get_call_trace_->Record(*selected, graph_lease ? "graph" : output_lease ? "output" : "ordinary", host);
    }
    if (!profile) return py::none();
    const auto end = BatchGetProfileNowNs();
    auto result = BatchGetProfileDict(*timings);
    result["binding_prepare_us"] = (native_begin - begin) / 1000.0;
    result["native_call_us"] = (end - native_begin) / 1000.0;
    result["binding_total_us"] = (end - begin) / 1000.0;
    return result;
  }

 public:
  std::uint64_t BeginPutInput(PutExecutionConfig config, const py::object& input,
                              std::uint32_t capacity, const py::object& stream) {
    const auto view = GetNpuTensorView(input, "Put input");
    const auto native_stream = ResolveStream(stream);
    std::unique_lock<std::mutex> lock(put_mutex_, std::defer_lock);
    { py::gil_scoped_release release; lock.lock(); }
    // Preallocate the Python owner before native registration, including the
    // map node needed to publish a successfully created lease without allocating.
    put_input_refs_.reserve(put_input_refs_.size() + 1);
    auto reserved = put_input_refs_.emplace(0, py::make_tuple(input, stream));
    auto node = put_input_refs_.extract(reserved.first);
    pending_put_input_refs_.push_back(node.mapped());
    std::uint64_t id = 0;
    Status status;
    {
      py::gil_scoped_release release;
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS)
        status = Status::Internal("aclrtSetDevice failed");
      else status = native_->core().BeginPutInput(
          config, {{view.data, view.bytes, DeviceKind::kNpu}, native_stream, capacity}, &id);
    }
    if (status.ok()) {
      node.key() = id;
      put_input_refs_.insert(std::move(node));
      pending_put_input_refs_.pop_back();
    }
    // A failed setup may leave native registrations awaiting cleanup. Keep its
    // source alive until Shutdown; do not replace an unresolved Put's owners.
    ThrowIfError(status);
    return id;
  }
  void EndPutInput(std::uint64_t id) {
    Status status;
    std::unique_lock<std::mutex> lock(put_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      lock.lock();
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS)
        status = Status::Internal("aclrtSetDevice failed");
      else status = native_->core().EndPutInput(id);
    }
    if (status.ok()) put_input_refs_.erase(id);
    ThrowIfError(status);
  }
  std::uint64_t BeginGetOutput(GetExecutionConfig config, const py::object& output,
                              std::uint32_t capacity, const py::object& stream) {
    const auto view = GetNpuTensorView(output, "output");
    const auto native_stream = ResolveStream(stream);
    std::unique_lock<std::mutex> lock(get_mutex_, std::defer_lock);
    { py::gil_scoped_release release; lock.lock(); }
    // Allocate Python ownership before native registration. Failed setup keeps
    // this reference until native cleanup is positively confirmed.
    output_refs_.reserve(output_refs_.size() + 1);
    auto reserved = output_refs_.emplace(0, py::make_tuple(output, stream));
    auto node = output_refs_.extract(reserved.first);
    pending_get_refs_.push_back(node.mapped());
    std::uint64_t id = 0;
    Status status;
    {
      py::gil_scoped_release release;
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS)
        status = Status::Internal("aclrtSetDevice failed");
      else status = native_->core().BeginGetOutput(
          config, {{view.data, view.bytes, DeviceKind::kNpu}, native_stream, capacity}, &id);
    }
    if (status.ok()) {
      node.key() = id;
      output_refs_.insert(std::move(node));
      pending_get_refs_.clear();
    }
    ThrowIfError(status);
    return id;
  }
  void EndGetOutput(std::uint64_t id) {
    Status status;
    bool pending = true;
    std::unique_lock<std::mutex> lock(get_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      lock.lock();
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS)
        status = Status::Internal("aclrtSetDevice failed");
      else {
        status = native_->core().EndGetOutput(id);
        pending = native_->core().HasPendingGet();
      }
    }
    // Another algorithm may still be using pending tensors on the Get direction.
    if (status.ok()) output_refs_.erase(id);
    if (!pending) pending_get_refs_.clear();
    ThrowIfError(status);
  }
  void DrainGet() {
    Status status;
    std::unique_lock<std::mutex> direction(get_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      direction.lock();
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS) status = Status::Internal("aclrtSetDevice failed");
      else status = native_->DrainGet();
    }
    if (status.ok()) pending_get_refs_.clear();
    ThrowIfError(status);
  }
  std::uint64_t BeginGetGraph(GetExecutionConfig config, const py::object& output,
                              std::uint32_t capacity, const py::object& stream) {
    const auto view = GetNpuTensorView(output, "output");
    const auto native_stream = ResolveStream(stream);
    auto refs = py::make_tuple(output, stream);
    std::uint64_t id = 0;
    Status status;
    std::unique_lock<std::mutex> lock(get_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      lock.lock();
    }
    // Keep the output alive before registration, including a failed graph
    // preparation whose cleanup must be retried by DrainGet/Shutdown. Reserve
    // the graph map node now as well: publishing a native lease must not be
    // followed by an allocation that could lose its Python ownership entry.
    graph_refs_.reserve(graph_refs_.size() + 1);
    std::vector<py::object> graph_refs;
    graph_refs.push_back(refs);
    auto reserved = graph_refs_.emplace(0, std::move(graph_refs));
    auto graph_node = graph_refs_.extract(reserved.first);
    pending_get_refs_.push_back(std::move(refs));
    {
      py::gil_scoped_release release;
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS)
        status = Status::Internal("aclrtSetDevice failed");
      else
        status = native_->core().BeginGetGraph(
            config, {{view.data, view.bytes, DeviceKind::kNpu}, native_stream, capacity}, &id);
    }
    if (status.ok()) {
      graph_node.key() = id;
      graph_refs_.insert(std::move(graph_node));
      pending_get_refs_.clear();
    }
    ThrowIfError(status);
    return id;
  }
  void EndGetGraph(std::uint64_t id, const py::object& stream, bool discard) {
    const auto native_stream = ResolveStream(stream);
    Status status;
    std::unique_lock<std::mutex> lock(get_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      lock.lock();
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS)
        status = Status::Internal("aclrtSetDevice failed");
      else
        status = native_->core().EndGetGraph(id, native_stream, discard);
    }
    if (status.ok()) graph_refs_.erase(id);
    ThrowIfError(status);
  }
  void RecoverPut(PutExecutionConfig c) {
    Status status;
    bool pending = true;
    std::unique_lock<std::mutex> lock(put_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      lock.lock();
      status = native_->Recover(c);
      pending = native_->core().HasPendingPut();
    }
    // A safely aborted Put returns its original error, but no DMA owns the
    // Tensor or stream anymore. Unknown completion must retain both objects.
    if (!pending) pending_put_ref_ = py::object{};
    ThrowIfError(status);
  }
  void RecoverGet(GetExecutionConfig c) {
    Status status;
    bool pending = true;
    std::unique_lock<std::mutex> lock(get_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      lock.lock();
      status = native_->Recover(c);
      pending = native_->core().HasPendingGet();
    }
    // Safe retirement of the old request is independent of rebuilding a new
    // provider. Never retain stale tensors after a closed-domain init failure,
    // or drop them while a failed close still owns DMA resources.
    if (!pending) pending_get_refs_.clear();
    ThrowIfError(status);
  }
  void WarmupPut(PutExecutionConfig c, std::uint32_t capacity, const py::object& stream) {
    const auto native_stream = ResolveStream(stream);
    py::gil_scoped_release release;
    if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS) throw std::runtime_error("aclrtSetDevice failed");
    ThrowIfError(native_->Warmup(c, {capacity, native_stream}));
  }
  void WarmupGet(GetExecutionConfig c, std::uint32_t capacity, const py::object& stream) {
    const auto native_stream = ResolveStream(stream);
    py::gil_scoped_release release;
    if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS) throw std::runtime_error("aclrtSetDevice failed");
    ThrowIfError(native_->Warmup(c, {capacity, native_stream}));
  }
  void Shutdown() {
    Status status;
    std::unique_lock<std::mutex> get(get_mutex_, std::defer_lock), put(put_mutex_, std::defer_lock);
    std::unique_lock<std::shared_mutex> raw(raw_lifecycle_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      std::lock(get, put, raw);
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS) status = Status::Internal("aclrtSetDevice failed");
      else status = native_->Shutdown();
    }
    if (status.ok()) {
      pending_get_refs_.clear(); output_refs_.clear(); pending_put_ref_ = py::object{};
      put_input_refs_.clear(); pending_put_input_refs_.clear();
      raw_outputs_ = py::dict(); raw_failed_inputs_ = py::dict();
    }
    if (get_call_trace_) get_call_trace_->Flush();
    ThrowIfError(status);
  }
  ~NpuPythonRuntime() {
    if (!native_) return;
    Status close_status;
    { py::gil_scoped_release release;
      (void)aclrtSetDevice(native_->device_id());
      close_status = native_->Shutdown(); }
    if (!close_status.ok()) {
      // Preserve Python storage as well as native DMA owners on failed cleanup.
      for (auto& ref : pending_get_refs_) (void)ref.release();
      (void)pending_put_ref_.release();
      for (auto& entry : graph_refs_)
        for (auto& ref : entry.second) (void)ref.release();
      for (auto& entry : output_refs_) (void)entry.second.release();
      for (auto& entry : put_input_refs_) (void)entry.second.release();
      for (auto& ref : pending_put_input_refs_) (void)ref.release();
      (void)raw_outputs_.release(); (void)raw_failed_inputs_.release();
      (void)native_.release();
    }
  }

  void CheckBatchGetStatus(const py::object& stream, std::optional<GetExecutionConfig> config,
                           std::uint64_t graph_lease) {
    const auto native_stream = ResolveStream(stream);
    Status status;
    std::unique_lock<std::mutex> lock(get_mutex_, std::defer_lock);
    {
      py::gil_scoped_release release;
      lock.lock();
      if (aclrtSetDevice(native_->device_id()) != ACL_SUCCESS)
        status = Status::Internal("aclrtSetDevice failed");
      else status = native_->CheckBatchGetStatus(native_stream, config, graph_lease);
    }
    ThrowIfError(status);
  }

 private:
  friend class PythonGetGraphResources;
  void ApplyGraphScope(const py::object&, const NpuTensorView&, std::size_t, aclrtStream,
                       bool, bool, std::optional<GetExecutionConfig>&, std::uint64_t&,
                       std::uint64_t);
  void FinishGraphScopeGet(bool success);
  PythonGetGraphResources* active_graph_scope_{nullptr};
  std::unique_ptr<::kvaf::NpuRuntime> native_;
  std::unique_ptr<GetCallTrace> get_call_trace_;
  std::mutex get_mutex_, put_mutex_;
  std::shared_mutex raw_lifecycle_mutex_;
  py::dict raw_outputs_, raw_failed_inputs_;
  std::vector<py::object> pending_get_refs_;
  py::object pending_put_ref_;
  std::unordered_map<std::uint64_t, std::vector<py::object>> graph_refs_;
  std::unordered_map<std::uint64_t, py::object> output_refs_;
  std::unordered_map<std::uint64_t, py::object> put_input_refs_;
  std::vector<py::object> pending_put_input_refs_;
};

// One owner per external model graph. It records a real eager forward, then
// routes ordinary Python Get calls to independently prepared native nodes.
// Scheduling and graph destruction remain the caller's responsibility.
class PythonGetGraphResources : public std::enable_shared_from_this<PythonGetGraphResources> {
 public:
  explicit PythonGetGraphResources(py::object runtime)
      : runtime_owner_(std::move(runtime)), runtime_(runtime_owner_.cast<NpuPythonRuntime*>()) {}
  ~PythonGetGraphResources() {
    // No implicit EndGraph: the external graph might still replay. Native graph
    // refs retain allocations until explicit close or safe Runtime cleanup.
    if (runtime_->active_graph_scope_ == this) runtime_->active_graph_scope_ = nullptr;
  }

  void Enter(bool capture, py::object stream) {
    if (runtime_->active_graph_scope_)
      throw py::value_error("a Get graph preparation/capture scope is already active on this Runtime");
    if (capture ? state_ != State::Prepared : state_ != State::New)
      throw py::value_error(capture ? "capture requires one successfully prepared forward"
                                   : "graph resources may prepare only once; create a new owner");
    if (stream.is_none()) stream = py::module_::import("torch").attr("npu").attr("current_stream")();
    const auto native_stream = ResolveStream(stream);
    RequireOutsideCapture(native_stream);
    if (capture && native_stream != native_stream_)
      throw py::value_error("one graph must prepare and capture on the same stream");
    if (!capture) {
      stream_ = std::move(stream);
      native_stream_ = native_stream;
    }
    thread_ = std::this_thread::get_id();
    cursor_ = 0;
    state_ = capture ? State::Capturing : State::Preparing;
    runtime_->active_graph_scope_ = this;
  }

  void Exit(bool capture, bool exception) {
    if (runtime_->active_graph_scope_ != this || thread_ != std::this_thread::get_id())
      throw py::value_error("Get graph scope must exit on its entering thread");
    runtime_->active_graph_scope_ = nullptr;
    const bool failed = exception || call_failed_ || pending_call_;
    state_ = State::Failed;  // Every exception below keeps the owner retryable by close().
    if (capture) {
      capture_failed_ = true;
      if (exception) return;
      RequireOutsideCapture(native_stream_);
      if (failed || cursor_ != nodes_.size())
        throw py::value_error("Get capture did not match its prepared forward; destroy the graph and close(discard_result=True)");
      state_ = State::Captured;
      capture_failed_ = false;
      return;
    }
    // A real warm-up may submit asynchronously. Complete it before registering
    // graph nodes; never reinterpret a dry record as a completed warm-up.
    try {
      runtime_->DrainGet();
      {
        py::gil_scoped_release release;
        ThrowIfAclError(aclrtSynchronizeStream(native_stream_), "complete graph warm-up forward");
      }
    } catch (...) {
      if (!exception) throw;
      return;
    }
    if (exception) return;
    if (failed) throw py::value_error("a Get failed during preparation; close these resources and prepare a new owner");
    if (nodes_.empty()) throw py::value_error("prepare must execute at least one BatchGet");
    for (auto& node : nodes_) {
      ValidateStorage(node.output, node);
      node.lease = runtime_->BeginGetGraph(node.config, node.output, node.capacity, stream_);
      // Retain the allocation itself as well as Tensor wrappers. A caller must
      // still not resize/change storage through any alias while graph is live.
      runtime_->graph_refs_.at(node.lease).push_back(node.storage);
    }
    state_ = State::Prepared;
  }

  void BeforeGet(const py::object& output, const NpuTensorView& view, std::size_t capacity,
                 aclrtStream stream, bool synchronize, bool profile,
                 std::optional<GetExecutionConfig>& execution, std::uint64_t& lease,
                 std::uint64_t output_lease) {
    try {
      if (thread_ != std::this_thread::get_id())
        throw py::value_error("Get graph scope may only submit from its entering thread");
      if (call_failed_ || pending_call_)
        throw py::value_error("Get graph scope already contains a failed call");
      if (stream != native_stream_)
        throw py::value_error("Get stream differs from the graph preparation/capture stream");
      if (lease || output_lease)
        throw py::value_error("automatic graph scopes cannot mix explicit graph/output leases");
      const auto selected = execution ? execution : runtime_->native_->default_get();
      if (!selected) throw py::value_error("graph Get requires an execution config or Runtime default");
      if (!capacity || capacity > UINT32_MAX)
        throw py::value_error("graph Get request capacity must fit uint32");
      if (state_ == State::Preparing) {
        RequireOutsideCapture(stream);
        auto storage = Storage(output);
        nodes_.push_back(Node{*selected, output, storage, StorageIdentity(storage),
            StoragePointer(storage), StorageBytes(storage), view.data, view.bytes,
            static_cast<std::uint32_t>(capacity), 0});
      } else if (state_ == State::Capturing) {
        if (synchronize || profile)
          throw py::value_error("captured BatchGet requires synchronize=False and profile=False");
        if (cursor_ == nodes_.size()) throw py::value_error("capture contains more Get nodes than prepare");
        auto& node = nodes_[cursor_];
        if (!(*selected == node.config) || capacity != node.capacity ||
            view.data != node.data || view.bytes != node.bytes)
          throw py::value_error("Get capture algorithm/output/capacity or call order differs from prepare");
        ValidateStorage(output, node);
        aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
        aclmdlRI graph = nullptr;
        ThrowIfAclError(aclmdlRICaptureGetInfo(stream, &status, &graph), "query Get capture");
        if (status != ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)
          throw py::value_error("resources.capture must enclose an actual external graph capture");
        if (captured_graph_ && captured_graph_ != graph)
          throw py::value_error("one graph_resources owner cannot capture multiple external graphs");
        captured_graph_ = graph;
        execution = node.config;
        lease = node.lease;
        ++cursor_;
      } else {
        throw py::value_error("Get graph scope is not active");
      }
      pending_call_ = true;
    } catch (...) {
      call_failed_ = true;
      throw;
    }
  }
  void AfterGet(bool success) {
    pending_call_ = false;
    call_failed_ = call_failed_ || !success;
  }

  py::tuple Leases() const {
    const auto count = std::count_if(nodes_.begin(), nodes_.end(), [](const Node& n) { return n.lease != 0; });
    py::tuple ids(count);
    std::size_t i = 0;
    for (const auto& node : nodes_) if (node.lease) ids[i++] = node.lease;
    return ids;
  }

  void Close(bool discard) {
    if (state_ == State::Closed) return;
    if (runtime_->active_graph_scope_)
      throw py::value_error("exit the active Get graph scope before closing resources");
    if (native_stream_) RequireOutsideCapture(native_stream_);
    if (capture_failed_ && !discard)
      throw py::value_error("failed capture requires graph destruction and close(discard_result=True)");
    if (std::none_of(nodes_.begin(), nodes_.end(), [](const Node& n) { return n.lease != 0; }))
      runtime_->DrainGet();
    std::exception_ptr error;
    // On a shared poisoned domain all nodes must first enter discard before
    // any one may finish. Visit every node, then retry survivors once.
    for (int pass = 0; pass < 2; ++pass) {
      error = nullptr;
      for (auto& node : nodes_) {
        if (!node.lease) continue;
        try {
          runtime_->EndGetGraph(node.lease, stream_, discard || state_ == State::Prepared || !captured_graph_);
          node.lease = 0;
        } catch (...) {
          if (!error) error = std::current_exception();
        }
      }
      if (!error) break;
    }
    if (error) std::rethrow_exception(error);
    nodes_.clear();
    stream_ = py::object{};
    state_ = State::Closed;
  }

 private:
  enum class State { New, Preparing, Prepared, Capturing, Captured, Failed, Closed };
  struct Node {
    GetExecutionConfig config;
    py::object output, storage;
    std::uint64_t storage_id, storage_pointer, storage_bytes;
    void* data;
    std::size_t bytes;
    std::uint32_t capacity;
    std::uint64_t lease;
  };
  static py::object Storage(const py::object& output) {
    if (!py::hasattr(output, "untyped_storage"))
      throw py::type_error("automatic Get graph preparation requires a Tensor with untyped_storage");
    return output.attr("untyped_storage")();
  }
  static std::uint64_t StorageIdentity(const py::object& storage) {
    return storage.attr("_cdata").cast<std::uint64_t>();
  }
  static std::uint64_t StoragePointer(const py::object& storage) {
    return storage.attr("data_ptr")().cast<std::uint64_t>();
  }
  static std::uint64_t StorageBytes(const py::object& storage) {
    return storage.attr("nbytes")().cast<std::uint64_t>();
  }
  static void ValidateStorage(const py::object& output, const Node& node) {
    auto storage = Storage(output);
    const auto view = GetNpuTensorView(output, "graph output");
    if (StorageIdentity(storage) != node.storage_id ||
        StoragePointer(storage) != node.storage_pointer || StorageBytes(storage) != node.storage_bytes ||
        view.data != node.data || view.bytes != node.bytes)
      throw py::value_error("Get graph output allocation/range changed after preparation");
  }
  void RequireOutsideCapture(aclrtStream stream) const {
    ThrowIfAclError(aclrtSetDevice(runtime_->DeviceId()), "select Get graph device");
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI graph = nullptr;
    ThrowIfAclError(aclmdlRICaptureGetInfo(stream, &status, &graph), "query Get graph scope");
    if (status != ACL_MODEL_RI_CAPTURE_STATUS_NONE)
      throw py::value_error("prepare/close must run outside graph capture; enter resources.capture before torch.npu.graph");
  }
  py::object runtime_owner_, stream_;
  NpuPythonRuntime* runtime_;
  State state_{State::New};
  std::thread::id thread_;
  aclrtStream native_stream_{nullptr};
  aclmdlRI captured_graph_{nullptr};
  std::vector<Node> nodes_;
  std::size_t cursor_{0};
  bool call_failed_{false}, pending_call_{false}, capture_failed_{false};
};

class PythonGetGraphScope {
 public:
  PythonGetGraphScope(std::shared_ptr<PythonGetGraphResources> owner, py::object stream, bool capture)
      : owner_(std::move(owner)), stream_(std::move(stream)), capture_(capture) {}
  std::shared_ptr<PythonGetGraphResources> Enter() {
    if (entered_) throw py::value_error("Get graph scope may only be entered once");
    owner_->Enter(capture_, stream_);
    entered_ = true;
    return owner_;
  }
  bool Exit(const py::object& type, const py::object&, const py::object&) {
    if (!entered_ || exited_) throw py::value_error("Get graph scope is not active");
    exited_ = true;
    owner_->Exit(capture_, !type.is_none());
    return false;
  }
 private:
  std::shared_ptr<PythonGetGraphResources> owner_;
  py::object stream_;
  bool capture_, entered_{false}, exited_{false};
};

void NpuPythonRuntime::ApplyGraphScope(const py::object& output, const NpuTensorView& view,
                                     std::size_t capacity, aclrtStream stream, bool synchronize,
                                     bool profile, std::optional<GetExecutionConfig>& config,
                                     std::uint64_t& lease, std::uint64_t output_lease) {
  if (active_graph_scope_)
    active_graph_scope_->BeforeGet(output, view, capacity, stream, synchronize, profile,
                                  config, lease, output_lease);
}
void NpuPythonRuntime::FinishGraphScopeGet(bool success) {
  if (active_graph_scope_) active_graph_scope_->AfterGet(success);
}

}  // namespace

void BindNpuPythonRuntime(py::module_& module) {
  py::class_<DeviceAccessTable>(module, "DeviceAccessTable")
      .def_property_readonly("addresses", &DeviceAccessTable::addresses)
      .def_property_readonly("valid", &DeviceAccessTable::valid)
      .def_property_readonly("source_id", [](const DeviceAccessTable& table) {
        return table.source().opaque;
      });

  py::class_<PythonGetGraphScope, std::shared_ptr<PythonGetGraphScope>>(module, "_GetGraphScope")
      .def("__enter__", &PythonGetGraphScope::Enter)
      .def("__exit__", &PythonGetGraphScope::Exit);
  py::class_<PythonGetGraphResources, std::shared_ptr<PythonGetGraphResources>>(module, "GetGraphResources")
      .def("prepare", [](std::shared_ptr<PythonGetGraphResources> self, py::object stream) {
        return std::make_shared<PythonGetGraphScope>(std::move(self), std::move(stream), false);
      }, py::arg("stream") = py::none(), "Run one real eager forward and prepare its Get nodes on scope exit.")
      .def("capture", [](std::shared_ptr<PythonGetGraphResources> self, py::object stream) {
        return std::make_shared<PythonGetGraphScope>(std::move(self), std::move(stream), true);
      }, py::arg("stream") = py::none(), "Enclose external graph capture; ordinary Gets consume prepared nodes in order.")
      .def_property_readonly("leases", &PythonGetGraphResources::Leases,
          "Native node leases in forward order, for per-node status/profile queries.")
      .def("close", &PythonGetGraphResources::Close, py::kw_only(), py::arg("discard_result") = false,
          "Stop replay, synchronize and destroy the external graph before close. Failure retains resources for retry.");

  py::class_<CpuPullOptions>(module, "CpuPullOptions")
      .def(py::init([](std::uint32_t timeout_ms) {
        CpuPullOptions value; value.timeout_ms = timeout_ms; return value;
      }), py::kw_only(), py::arg("timeout_ms") = 30000)
      .def_readonly("timeout_ms", &CpuPullOptions::timeout_ms);
  py::class_<NpuPullOptions>(module, "NpuPullOptions")
      .def(py::init([](std::string kernel_path, std::uint32_t cores, std::uint32_t jettys,
                       std::uint32_t inflight, std::uint32_t output_bindings, bool diagnostics) {
        NpuPullOptions value; value.kernel_path = std::move(kernel_path);
        value.cores = cores; value.jettys = jettys; value.max_inflight_total = inflight;
        value.max_output_bindings = output_bindings;
        value.diagnostics = diagnostics; return value;
      }), py::kw_only(), py::arg("kernel_path") = "", py::arg("cores") = 2,
          py::arg("jettys") = 2, py::arg("max_inflight_total") = 0,
          py::arg("max_output_bindings") = 64,
          py::arg("diagnostics") = false)
      .def_readonly("kernel_path", &NpuPullOptions::kernel_path)
      .def_readonly("cores", &NpuPullOptions::cores).def_readonly("jettys", &NpuPullOptions::jettys)
      .def_readonly("max_inflight_total", &NpuPullOptions::max_inflight_total)
      .def_readonly("max_output_bindings", &NpuPullOptions::max_output_bindings)
      .def_readonly("diagnostics", &NpuPullOptions::diagnostics);

  auto runtime = py::class_<NpuPythonRuntime>(module, "NpuRuntime");
  python_binding::BindNpuRuntimeConstructor(runtime);
  runtime
      .def("graph_resources", [](py::object self) {
        return std::make_shared<PythonGetGraphResources>(std::move(self));
      }, "Create resources for one model graph. Caller orders graph/eager execution and destroys the graph before close.")
      .def("listen_hbm", &NpuPythonRuntime::ListenHbm, py::arg("listen_address"),
           py::arg("advertise_address")="", py::arg("local_eid")="")
      .def("batch_get_raw_addr", &NpuPythonRuntime::BatchGetRawArrays,
           py::arg("ready"),py::arg("src_offsets"),py::arg("dst_offsets"),py::arg("lengths"),
           py::arg("output"),py::arg("stream")=py::none(),py::arg("profile")=false,py::arg("validate_ranges")=false,
           py::arg("synchronize")=false,
           "Enqueue peer HBM READ on the caller's stream, like batch_get. Default success means submitted. "
           "Warm up registrations before graph capture; synchronize=False supports capture and replay. "
           "Caller keeps metadata/storage and stream alive through completion and graph destruction, "
           "and owns stream ordering, graph lifetime and producer reuse. "
           "synchronize=True waits for completion. profile=True adds Host timings; device event timing "
           "is available only with synchronize=True. Device status is not read back. "
           "Output storage is retained until successful Runtime shutdown; do not resize storage while registered. "
           "One WQE per request, without splitting. Callers guarantee valid, disjoint ranges and lengths in "
           "1..UINT32_MAX within device READ limits. validate_ranges=True enables length/bounds diagnostics.")
      .def("resolve_raw_addr", &NpuPythonRuntime::ResolveRawAddr, py::arg("ready"))
      .def("cancel_raw_addr", &NpuPythonRuntime::CancelRawAddr, py::arg("ready"),
           "Fence future Host submissions for this batch. Does not cancel queued work or captured graphs. "
           "Application waits for execution and retires graphs before producer reuse; no notification is sent.")
      .def_property_readonly("pending_raw_addr", &NpuPythonRuntime::PendingRawAddr)
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
      .def("batch_cpu_pull", &NpuPythonRuntime::BatchCpuPull, py::arg("keys"), py::arg("ready"),
           py::arg("src_offsets"), py::arg("lengths"), py::kw_only(), py::arg("timeout") = py::none(),
           py::arg("profile") = false,
           "Prepare DDR slots, submit CPU READ, wait and commit using an attached CPU ReadyToken. "
           "One source span per key; Host metadata matches batch_put. Returns None on success. "
           "timeout is in milliseconds (None uses cpu_pull_options.timeout_ms). On unresolved failure "
           "retain producer data and call resolve_cpu_pull; never repeat the batch as recovery. "
           "Producer synchronization and release_batch remain the application's responsibility. "
           "profile=True returns a profile dictionary with Host stage timings in microseconds instead of None; "
           "profiling is disabled by default and does not include verification or source release.")
      .def("cancel_cpu_pull", &NpuPythonRuntime::CancelCpuPull,
           py::arg("ready"), "Fence this batch and abort its Put before notifying the producer. Retry on an unresolved result.")
      .def("resolve_cpu_pull", &NpuPythonRuntime::ResolveCpuPull, py::kw_only(), py::arg("timeout") = py::none(),
           "Finish the internally retained batch_cpu_pull, without replaying READ. Returns None "
           "only after successful commit; failure raises. Does not recover explicit PutHandle transactions.")
      .def("offload_region", &NpuPythonRuntime::OffloadRegion,
           py::arg("source_physical_device_id") = py::none())
      .def("prepare_put",&NpuPythonRuntime::PreparePut,py::arg("keys"),py::arg("lengths"))
      .def("submit_cpu_pull",&NpuPythonRuntime::SubmitCpuPull,py::arg("ready"),py::arg("put"),
           py::arg("src_offsets"),py::arg("dst_offsets"),py::arg("lengths"),
           "Submit CPU READ using an attached ReadyToken source and DDR-relative destination "
           "offsets: put.entries[i].address minus runtime.remote_region['base'], plus any in-slot offset. "
           "The three Host arrays must be equally sized. "
           "Returns None; query/wait using the same put, including after a lost response.")
      .def("query_cpu_pull",&NpuPythonRuntime::QueryCpuPull,py::arg("put"))
      .def("wait_cpu_pull",&NpuPythonRuntime::WaitCpuPull,py::arg("put"),py::arg("timeout")=30000,
           "Wait for DMA only; timeout is in milliseconds. Commit explicitly afterward.")
      .def("commit_put",&NpuPythonRuntime::CommitPut,py::arg("put"))
      .def("abort_put",&NpuPythonRuntime::AbortPut,py::arg("put"))
      .def("last_offload_checksum", &NpuPythonRuntime::LastChecksum,
           "Hash current CPU DDR via a separate RPC after offload. Call before freeing or "
           "overwriting the destination slots, the source's next ready(), or source shutdown.")
      .def_property_readonly("pending_offload", &NpuPythonRuntime::PendingOffload)
#endif
      .def("initialize", &NpuPythonRuntime::InitializeBase)
      .def("initialize_ddr", &NpuPythonRuntime::InitializeDdr, py::arg("ddr_bytes"), py::arg("slot_size"))
      .def("shutdown_ddr", &NpuPythonRuntime::ShutdownDdr)
      .def_property_readonly("ddr_initialized", &NpuPythonRuntime::DdrInitialized)
      .def("get_execution_resources", &NpuPythonRuntime::GetResources, py::arg("execution"),
           "Return the effective Get resource budget after DDR initialization.")
      .def("link_to", &NpuPythonRuntime::LinkTo, py::arg("address"),
           "Set the CPU Runtime target before initialize_ddr or initialize(ddr_bytes, slot_size).")
      .def("initialize", &NpuPythonRuntime::Initialize, py::arg("ddr_bytes"), py::arg("slot_size"))
      .def("shutdown", &NpuPythonRuntime::Shutdown)
      .def("check_batch_get_status", &NpuPythonRuntime::CheckBatchGetStatus,
           py::arg("stream") = py::none(), py::kw_only(), py::arg("execution") = py::none(),
           py::arg("graph_lease") = 0,
           "Synchronize and check AICPU BatchGet results outside capture. Retains the first failed replay.")
      .def_property_readonly("initialized", &NpuPythonRuntime::Initialized)
      .def_property_readonly("device_id", &NpuPythonRuntime::DeviceId)
      .def_property_readonly("npu_eids", &NpuPythonRuntime::NpuEids)
      .def_property_readonly("batch_get_token_bytes", &NpuPythonRuntime::BatchGetTokenBytes)
      .def_property_readonly("batch_get_tokens_per_block",
                             &NpuPythonRuntime::BatchGetTokensPerBlock)
      .def_property_readonly("remote_region", &NpuPythonRuntime::RemoteRegion)
      .def("batch_put", &NpuPythonRuntime::BatchPut, py::arg("keys"), py::arg("source"),
           py::arg("input_offsets"), py::arg("lengths"), py::arg("stream") = py::none(),
           py::kw_only(), py::arg("execution") = py::none(), py::arg("input_lease") = 0,
           "Synchronously write variable-size packed NPU blocks. keys, input_offsets, and "
           "lengths are CPU integer sequences or contiguous CPU int64 tensors "
           "with one entry per key; "
           "stream=None uses torch.npu.current_stream().")
      .def("recover", &NpuPythonRuntime::RecoverPut, py::arg("execution"))
      .def("recover", &NpuPythonRuntime::RecoverGet, py::arg("execution"))
      .def("drain_get", &NpuPythonRuntime::DrainGet)
      .def("begin_put_input", &NpuPythonRuntime::BeginPutInput, py::arg("execution"),
           py::arg("input"), py::arg("request_capacity"), py::arg("stream") = py::none(),
           "Bind URMA AIV or AICPU input for repeated synchronous Puts. Keep its storage and stream "
           "alive and do not resize or replace storage until end_put_input succeeds.")
      .def("end_put_input", &NpuPythonRuntime::EndPutInput, py::arg("lease"),
           "Release reusable Put input registration after its transaction completes; failure retains owners.")
      .def("begin_get_output", &NpuPythonRuntime::BeginGetOutput, py::arg("execution"),
           py::arg("output"), py::arg("request_capacity"), py::arg("stream") = py::none(),
           "Bind URMA AIV Direct or AICPU Gather output for repeated eager Gets. "
           "Do not resize, replace or free its storage (including aliases) until end_get_output succeeds.")
      .def("end_get_output", &NpuPythonRuntime::EndGetOutput, py::arg("lease"),
           "Drain pending Get and unregister its reusable output; failure retains the binding.")
      .def("begin_get_graph", &NpuPythonRuntime::BeginGetGraph, py::arg("execution"),
           py::arg("output"), py::arg("request_capacity"), py::arg("stream") = py::none())
      .def("end_get_graph", &NpuPythonRuntime::EndGetGraph, py::arg("lease"),
           py::arg("stream") = py::none(), py::kw_only(), py::arg("discard_result") = false)
      .def("warmup", &NpuPythonRuntime::WarmupPut, py::arg("execution"), py::kw_only(),
           py::arg("request_capacity") = 0, py::arg("stream") = py::none())
      .def("warmup", &NpuPythonRuntime::WarmupGet, py::arg("execution"), py::kw_only(),
           py::arg("request_capacity") = 0, py::arg("stream") = py::none())
      .def("prepare_access_table", &NpuPythonRuntime::PrepareAccessTable, py::arg("keys"))
      .def("access_table_view", &NpuPythonRuntime::AccessTableView, py::arg("access_table"),
           py::arg("device") = "cpu", py::arg("capacity") = 0)
      .def("release_access_table", &NpuPythonRuntime::ReleaseAccessTable, py::arg("access_table"))
      .def("batch_free", &NpuPythonRuntime::BatchFree, py::arg("keys"),
           "Release the supplied keys from the Runtime metadata store. keys "
           "accepts a Python key sequence or a contiguous one-dimensional "
           "CPU torch.int64 Tensor.")
      .def("start_batch_get_kernel_profile", &NpuPythonRuntime::StartBatchGetKernelProfile,
           py::arg("detailed") = false, py::kw_only(), py::arg("execution") = py::none(),
           "Enable the selected Get instance's per-core kernel timeline.")
      .def("batch_get_kernel_profile", &NpuPythonRuntime::BatchGetKernelProfileSnapshot,
           py::kw_only(), py::arg("execution") = py::none(), py::arg("graph_lease") = 0,
           "Copy the latest completed per-core timeline; select a node with graph_lease.")
      .def("stop_batch_get_kernel_profile", &NpuPythonRuntime::StopBatchGetKernelProfile,
           py::kw_only(), py::arg("execution") = py::none(), "Copy the latest timeline and disable BatchGet kernel profiling.")
      .def("batch_get", &NpuPythonRuntime::BatchGet, py::arg("device_access_table"),
           py::arg("indices"), py::arg("offsets"), py::arg("miss_count"), py::arg("output_indices"),
           py::arg("output"), py::arg("stream") = py::none(), py::arg("synchronize") = false,
           py::arg("profile") = false, py::kw_only(), py::arg("execution") = py::none(),
           py::arg("graph_lease") = 0, py::arg("output_lease") = 0,
           "Submit BatchGet from NPU Tensors. stream=None uses torch.npu.current_stream(); "
           "the default returns after enqueueing work on that stream. AICPU profile=True "
           "requires synchronize=True and returns a stage timing dictionary.")
      .def("batch_get", &NpuPythonRuntime::BatchGetFromTensors, py::arg("addresses"),
           py::arg("valid"), py::arg("indices"), py::arg("offsets"), py::arg("miss_count"),
           py::arg("output_indices"), py::arg("output"), py::arg("stream") = py::none(),
           py::arg("synchronize") = false, py::arg("profile") = false, py::kw_only(),
           py::arg("execution") = py::none(), py::arg("graph_lease") = 0, py::arg("output_lease") = 0,
           "Submit BatchGet from caller-owned NPU addresses and valid Tensors. "
           "stream=None uses torch.npu.current_stream(); the default returns "
           "after enqueueing work on that stream. AICPU profile=True requires "
           "synchronize=True and returns a stage timing dictionary.");
}

}  // namespace kvaf
