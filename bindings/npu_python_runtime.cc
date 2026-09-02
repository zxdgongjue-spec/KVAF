#include "npu_python_runtime.h"

#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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

void ValidateFusedBatchGetOutputTensor(const NpuTensorView& output,
                                       const std::string& label) {
  if (output.shape.size() < 2 || output.shape.front() <= 0 ||
      output.bytes % static_cast<std::size_t>(output.shape.front()) != 0 ||
      output.bytes / static_cast<std::size_t>(output.shape.front()) !=
          kBatchGetTransferBytes) {
    throw py::value_error(label +
                          " Tensor rows must each be 656 bytes");
  }
}

void ValidateFusedBatchGetControlTensor(const NpuTensorView& control,
                                        const py::object& tensor,
                                        const std::string& label) {
  if (control.shape.size() != 1 ||
      control.shape.front() !=
          static_cast<py::ssize_t>(KVAF_FUSED_BATCH_GET_CONTROL_WORDS) ||
      control.element_bytes != sizeof(std::uint64_t) ||
      py::str(tensor.attr("dtype")).cast<std::string>() != "torch.int64") {
    throw py::value_error(label +
                          " must be contiguous NPU torch.int64[16]");
  }
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

std::string StatusMessage(const Status& status, const char* operation) {
  const std::string detail = status.message().empty()
                                 ? "unknown KVAF error"
                                 : status.message();
  return std::string(operation) + ": " + detail;
}

// The lane normally sleeps while idle.  The explicit spin mode is only a
// performance probe: it keeps one host CPU busy so that a Post() does not pay
// the scheduler wake-up tail of condition_variable.  It must remain opt-in
// until an end-to-end runtime can reserve an appropriate CPU for the worker.
enum class NativeGetWorkerWaitMode : std::uint8_t {
  kCv,
  kDedicatedSpin,
};

struct NativeGetReapTiming {
  double terminal_sync_us{0.0};
  double reclaim_wait_us{0.0};
  double total_us{0.0};
};

// A deliberately narrow eager-only submission lane.  It has one worker and
// one caller-owned copy stream because Direct-HCCP's queue/status state is
// shared inside DefaultAccessOperator; it is not a general multi-stream Get
// scheduler.  Python owns every Tensor reference, while this class owns only
// POD BatchGetRequest fields and ACL events.
class NativeGetSubmitLane final {
 public:
  NativeGetSubmitLane(DefaultAccessOperator& access_operator,
                      std::uint32_t device_id, aclrtStream copy_stream,
                      bool bind_stream_resource,
                      std::uint32_t expected_get_vector_cores,
                      NativeGetWorkerWaitMode worker_wait_mode)
      : access_operator_(access_operator),
        device_id_(device_id),
        copy_stream_(copy_stream),
        bind_stream_resource_(bind_stream_resource),
        expected_get_vector_cores_(expected_get_vector_cores),
        worker_wait_mode_(worker_wait_mode) {}

  ~NativeGetSubmitLane() { CloseNoThrow(); }

  NativeGetSubmitLane(const NativeGetSubmitLane&) = delete;
  NativeGetSubmitLane& operator=(const NativeGetSubmitLane&) = delete;

  Status Initialize() {
    if (copy_stream_ == nullptr) {
      return InvalidArgument("native Get lane requires a copy stream");
    }
    if (expected_get_vector_cores_ != 0 && !bind_stream_resource_) {
      return InvalidArgument(
          "native Get worker vector verification requires a stream resource binding");
    }
    if (aclrtCreateEvent(&descriptor_ready_) != ACL_SUCCESS ||
        aclrtCreateEvent(&get_done_) != ACL_SUCCESS ||
        aclrtCreateEvent(&terminal_done_) != ACL_SUCCESS ||
        aclrtCreateEvent(&worker_ready_) != ACL_SUCCESS) {
      DestroyEventsNoThrow();
      return Internal("failed to create native Get lane ACL events");
    }
    if (bind_stream_resource_) {
      stream_resource_library_ = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
      if (stream_resource_library_ == nullptr) {
        DestroyEventsNoThrow();
        return Internal("failed to load libascendcl.so for native Get quota");
      }
      use_stream_resource_ = reinterpret_cast<UseStreamResourceFn>(
          dlsym(stream_resource_library_, "aclrtUseStreamResInCurrentThread"));
      unuse_stream_resource_ = reinterpret_cast<UseStreamResourceFn>(
          dlsym(stream_resource_library_, "aclrtUnuseStreamResInCurrentThread"));
      if (expected_get_vector_cores_ != 0) {
        get_current_thread_resource_ =
            reinterpret_cast<GetCurrentThreadResourceFn>(
                dlsym(stream_resource_library_, "aclrtGetResInCurrentThread"));
      }
      if (use_stream_resource_ == nullptr || unuse_stream_resource_ == nullptr) {
        DestroyEventsNoThrow();
        CloseStreamResourceLibraryNoThrow();
        return Internal("CANN has no native Get stream-resource TLS API");
      }
      if (expected_get_vector_cores_ != 0 &&
          get_current_thread_resource_ == nullptr) {
        DestroyEventsNoThrow();
        CloseStreamResourceLibraryNoThrow();
        return Internal("CANN cannot verify native Get worker vector quota");
      }
    }

    worker_ = std::thread(&NativeGetSubmitLane::WorkerMain, this);
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this] { return worker_initialized_; });
    if (!worker_error_.empty()) {
      const std::string error = worker_error_;
      lock.unlock();
      StopWorkerNoThrow();
      DestroyEventsNoThrow();
      CloseStreamResourceLibraryNoThrow();
      return Internal(error);
    }
    lock.unlock();
    if (aclrtSynchronizeEvent(worker_ready_) != ACL_SUCCESS) {
      StopWorkerNoThrow();
      DestroyEventsNoThrow();
      CloseStreamResourceLibraryNoThrow();
      return Internal("native Get worker cannot submit to the copy stream");
    }
    return Status::OK();
  }

  Status BindOutput(const MutableBufferView& output) {
    std::lock_guard<std::mutex> lock(mu_);
    if (state_ != State::kFree) {
      return Status::Error(StatusCode::kBusy,
                           "native Get lane has an in-flight job");
    }
    if (output_binding_id_ != 0) {
      return Status::Error(StatusCode::kAlreadyExists,
                           "native Get lane output is already bound");
    }
    if (!output.valid() || output.device != DeviceKind::kNpu ||
        output.bytes % kBatchGetTransferBytes != 0) {
      return InvalidArgument(
          "native Get lane output must be an NPU buffer with 656-byte rows");
    }
    BatchGetOutputBindingId binding_id = 0;
    Status status = access_operator_.BindBatchGetOutput(output, &binding_id);
    if (!status.ok()) {
      return status;
    }
    output_ = output;
    output_binding_id_ = binding_id;
    return Status::OK();
  }

  // Descriptor Tensor identities are fixed for the lane lifetime. Their
  // contents may be updated in-place by the producer stream before Post(),
  // but changing a pointer requires an idle-lane Prepare() first.
  Status Prepare(const BatchGetRequest& request) {
    std::lock_guard<std::mutex> lock(mu_);
    if (state_ != State::kFree) {
      return Status::Error(StatusCode::kBusy,
                           "native Get lane has an in-flight job");
    }
    if (output_binding_id_ == 0 || request.output.data != output_.data ||
        request.output.bytes != output_.bytes) {
      return InvalidArgument(
          "native Get request output does not match the bound HBM slot");
    }
    prepared_request_ = request;
    has_prepared_request_ = true;
    return Status::OK();
  }

  // This is the hot host path: it only orders already-prepared descriptor
  // buffers, records one event, and wakes the worker.  It must not validate
  // Python Tensors or acquire a Direct-HCCP output registration.
  Status Post(aclrtStream producer_stream) {
    if (producer_stream == nullptr) {
      return InvalidArgument("native Get lane requires a producer stream");
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (state_ != State::kFree) {
      return Status::Error(StatusCode::kBusy,
                           "native Get lane slot is not free");
    }
    if (!has_prepared_request_) {
      return Status::Error(StatusCode::kInvalidState,
                           "native Get lane requires prepare_native_get first");
    }
    if (aclrtRecordEvent(descriptor_ready_, producer_stream) != ACL_SUCCESS) {
      return Internal("failed to record native Get descriptor-ready event");
    }
    job_ = prepared_request_;
    worker_error_.clear();
    consumer_may_read_output_ = false;
    post_time_ = std::chrono::steady_clock::now();
    worker_start_time_ = std::chrono::steady_clock::time_point{};
    worker_issue_time_ = std::chrono::steady_clock::time_point{};
    state_ = State::kPosted;
    worker_wake_epoch_.fetch_add(1, std::memory_order_release);
    cv_.notify_all();
    return Status::OK();
  }

  Status Arm(aclrtStream consumer_stream, double* post_to_worker_start_us,
             double* worker_issue_us, double* post_to_issue_us,
             double* arm_wait_us) {
    if (consumer_stream == nullptr) {
      return InvalidArgument("native Get lane requires a consumer stream");
    }
    const auto arm_enter = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this] {
      return state_ == State::kIssued || state_ == State::kFailed || stopping_;
    });
    if (state_ == State::kFailed) {
      return Internal(worker_error_.empty()
                          ? "native Get worker failed before event record"
                          : worker_error_);
    }
    if (state_ != State::kIssued) {
      return Status::Error(StatusCode::kInvalidState,
                           "native Get lane stopped before event record");
    }
    const auto worker_start_time = worker_start_time_;
    const auto issue_time = worker_issue_time_;
    lock.unlock();
    if (aclrtStreamWaitEvent(consumer_stream, get_done_) != ACL_SUCCESS) {
      lock.lock();
      if (state_ == State::kIssued) {
        FailLocked("failed to arm native Get completion event on consumer stream");
      }
      return Internal("failed to arm native Get completion event on consumer stream");
    }
    const auto arm_leave = std::chrono::steady_clock::now();
    lock.lock();
    if (state_ != State::kIssued) {
      // The dependency has already been accepted by consumer_stream.  Do not
      // release the HBM slot unless a terminal event later proves that every
      // consumer has completed.
      consumer_may_read_output_ = true;
      FailLocked("native Get lane state changed while arming completion");
      return Status::Error(StatusCode::kInvalidState,
                           "native Get lane state changed while arming completion");
    }
    consumer_may_read_output_ = true;
    state_ = State::kArmed;
    if (post_to_worker_start_us != nullptr) {
      *post_to_worker_start_us = DurationUs(post_time_, worker_start_time);
    }
    if (worker_issue_us != nullptr) {
      *worker_issue_us = DurationUs(worker_start_time, issue_time);
    }
    if (post_to_issue_us != nullptr) {
      *post_to_issue_us = DurationUs(post_time_, issue_time);
    }
    if (arm_wait_us != nullptr) {
      *arm_wait_us = DurationUs(arm_enter, arm_leave);
    }
    return Status::OK();
  }

  Status RecordTerminal(aclrtStream terminal_stream) {
    if (terminal_stream == nullptr) {
      return InvalidArgument("native Get lane requires a terminal stream");
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (state_ != State::kArmed) {
      return Status::Error(StatusCode::kInvalidState,
                           "native Get completion must be armed before terminal record");
    }
    if (aclrtRecordEvent(terminal_done_, terminal_stream) != ACL_SUCCESS) {
      // A consumer dependency has already been queued.  Without a terminal
      // event there is no proof that it stopped reading this output slot, so
      // the lane enters quarantine rather than being made reusable.
      FailLocked("failed to record native Get terminal event");
      return Internal("failed to record native Get terminal event");
    }
    state_ = State::kTerminal;
    return Status::OK();
  }

  Status Reap(NativeGetReapTiming* timing) {
    const auto reap_enter = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mu_);
    if (state_ == State::kFailed) {
      return Internal(worker_error_.empty() ? "native Get lane failed"
                                             : worker_error_);
    }
    if (state_ != State::kTerminal) {
      return Status::Error(StatusCode::kInvalidState,
                           "native Get lane requires a recorded terminal event before reap");
    }
    lock.unlock();
    const auto terminal_sync_enter = std::chrono::steady_clock::now();
    if (aclrtSynchronizeEvent(terminal_done_) != ACL_SUCCESS) {
      lock.lock();
      if (state_ == State::kTerminal) {
        FailLocked("native Get terminal event synchronization failed");
      }
      return Internal("native Get terminal event synchronization failed");
    }
    const auto terminal_sync_leave = std::chrono::steady_clock::now();
    lock.lock();
    if (state_ != State::kTerminal) {
      return Status::Error(StatusCode::kInvalidState,
                           "native Get lane state changed while reaping terminal");
    }
    // The final reader is now known to have completed.  A later failure while
    // resetting worker-owned events can still be explicitly closed safely.
    consumer_may_read_output_ = false;
    state_ = State::kReclaim;
    const auto reclaim_wait_enter = std::chrono::steady_clock::now();
    worker_wake_epoch_.fetch_add(1, std::memory_order_release);
    cv_.notify_all();
    cv_.wait(lock, [this] {
      return state_ == State::kFree || state_ == State::kFailed || stopping_;
    });
    if (state_ == State::kFailed) {
      return Internal(worker_error_.empty()
                          ? "native Get lane failed while reclaiming events"
                          : worker_error_);
    }
    if (state_ != State::kFree) {
      return Status::Error(StatusCode::kInvalidState,
                           "native Get lane stopped while reclaiming events");
    }
    const auto reap_leave = std::chrono::steady_clock::now();
    if (timing != nullptr) {
      timing->terminal_sync_us =
          DurationUs(terminal_sync_enter, terminal_sync_leave);
      timing->reclaim_wait_us =
          DurationUs(reclaim_wait_enter, reap_leave);
      timing->total_us = DurationUs(reap_enter, reap_leave);
    }
    return Status::OK();
  }

  bool Idle() const {
    std::lock_guard<std::mutex> lock(mu_);
    return state_ == State::kFree;
  }

  Status Close() {
    State state;
    bool consumer_may_read_output = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      state = state_;
      consumer_may_read_output = consumer_may_read_output_;
    }
    if (state != State::kFree && state != State::kFailed) {
      return Status::Error(StatusCode::kBusy,
                           "native Get lane still owns an in-flight slot");
    }
    if (state == State::kFailed && consumer_may_read_output) {
      return Status::Error(
          StatusCode::kInvalidState,
          "native Get lane failed after a consumer dependency was armed; "
          "terminal completion is unknown, so the HBM binding is quarantined");
    }
    StopWorkerNoThrow();
    if (aclrtSynchronizeStream(copy_stream_) != ACL_SUCCESS) {
      std::lock_guard<std::mutex> lock(mu_);
      FailLocked("native Get lane close could not synchronize its copy stream");
      return Internal("native Get lane close could not synchronize its copy stream");
    }
    if (output_binding_id_ != 0) {
      const Status release = access_operator_.ReleaseBatchGetOutput(output_binding_id_);
      if (!release.ok()) {
        std::lock_guard<std::mutex> lock(mu_);
        FailLocked(StatusMessage(release, "native Get lane close release failed"));
        return release;
      }
      output_binding_id_ = 0;
      output_ = MutableBufferView{};
    }
    prepared_request_ = BatchGetRequest{};
    has_prepared_request_ = false;
    DestroyEventsNoThrow();
    CloseStreamResourceLibraryNoThrow();
    {
      std::lock_guard<std::mutex> lock(mu_);
      state_ = State::kFree;
      consumer_may_read_output_ = false;
    }
    return Status::OK();
  }

 private:
  using UseStreamResourceFn = aclError (*)(aclrtStream);
  using GetCurrentThreadResourceFn =
      aclError (*)(aclrtDevResLimitType, std::uint32_t*);

  enum class State : std::uint8_t {
    kFree,
    kPosted,
    kIssued,
    kArmed,
    kTerminal,
    kReclaim,
    kFailed,
  };

  static double DurationUs(std::chrono::steady_clock::time_point begin,
                           std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
  }

  static void CpuRelax() {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    std::this_thread::yield();
#endif
  }

  void FailLocked(std::string error) {
    worker_error_ = std::move(error);
    state_ = State::kFailed;
    cv_.notify_all();
  }

  void WorkerMain() {
    std::string init_error;
    bool stream_resource_bound = false;
    if (aclrtSetDevice(device_id_) != ACL_SUCCESS) {
      init_error = "native Get worker aclrtSetDevice failed";
    } else if (bind_stream_resource_) {
      if (use_stream_resource_(copy_stream_) != ACL_SUCCESS) {
        init_error = "native Get worker failed to bind copy-stream resource quota";
      } else {
        stream_resource_bound = true;
      }
    }
    if (init_error.empty() && expected_get_vector_cores_ != 0) {
      std::uint32_t observed_vector_cores = 0;
      if (get_current_thread_resource_(ACL_RT_DEV_RES_VECTOR_CORE,
                                       &observed_vector_cores) != ACL_SUCCESS) {
        init_error = "native Get worker could not read its vector quota";
      } else if (observed_vector_cores != expected_get_vector_cores_) {
        init_error = "native Get worker vector quota differs from the requested value";
      }
    }
    if (init_error.empty() &&
        aclrtRecordEvent(worker_ready_, copy_stream_) != ACL_SUCCESS) {
      init_error = "native Get worker failed to record copy-stream readiness";
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      worker_initialized_ = true;
      if (!init_error.empty()) {
        worker_error_ = init_error;
      }
      cv_.notify_all();
    }
    if (!init_error.empty()) {
      if (stream_resource_bound && unuse_stream_resource_ != nullptr) {
        (void)unuse_stream_resource_(copy_stream_);
      }
      return;
    }

    while (true) {
      BatchGetRequest job;
      BatchGetOutputBindingId output_binding_id = 0;
      {
        std::unique_lock<std::mutex> lock(mu_);
        if (worker_wait_mode_ == NativeGetWorkerWaitMode::kCv) {
          cv_.wait(lock, [this] {
            return stopping_ || state_ == State::kPosted ||
                   state_ == State::kReclaim;
          });
        } else if (!stopping_ && state_ != State::kPosted &&
                   state_ != State::kReclaim) {
          // Snapshot while holding mu_: Post()/Reap() must acquire the same
          // mutex before changing state and incrementing the epoch, so this
          // cannot lose a notification between the state check and unlock.
          const std::uint64_t observed_epoch =
              worker_wake_epoch_.load(std::memory_order_acquire);
          lock.unlock();
          while (worker_wake_epoch_.load(std::memory_order_acquire) ==
                 observed_epoch) {
            CpuRelax();
          }
          lock.lock();
        }
        if (stopping_) {
          break;
        }
        if (state_ == State::kReclaim) {
          lock.unlock();
          const aclError descriptor_reset =
              aclrtResetEvent(descriptor_ready_, copy_stream_);
          const aclError get_reset =
              descriptor_reset == ACL_SUCCESS
                  ? aclrtResetEvent(get_done_, copy_stream_)
                  : descriptor_reset;
          const aclError terminal_reset =
              get_reset == ACL_SUCCESS
                  ? aclrtResetEvent(terminal_done_, copy_stream_)
                  : get_reset;
          const aclError reset_sync =
              terminal_reset == ACL_SUCCESS
                  ? aclrtSynchronizeStream(copy_stream_)
                  : terminal_reset;
          lock.lock();
          if (reset_sync != ACL_SUCCESS) {
            FailLocked("native Get worker failed to reclaim ACL events");
          } else {
            job_ = BatchGetRequest{};
            state_ = State::kFree;
            cv_.notify_all();
          }
          continue;
        }
        if (state_ != State::kPosted) {
          // A CV/spin wake can race a caller finishing Arm().  That state is
          // not worker work; take another idle wait without treating it as a
          // submission error.
          continue;
        }
        job = job_;
        output_binding_id = output_binding_id_;
      }

      const auto worker_start_time = std::chrono::steady_clock::now();
      std::string launch_error;
      if (aclrtStreamWaitEvent(copy_stream_, descriptor_ready_) != ACL_SUCCESS) {
        launch_error = "native Get worker failed to wait for descriptor-ready event";
      } else {
        const Status launch = access_operator_.ExecuteBoundBatchGet(
            BatchGetExecRequest{AccessTableView{}, &job,
                                BatchGetExecutionOptions{copy_stream_, false}},
            output_binding_id);
        if (!launch.ok()) {
          launch_error = StatusMessage(launch, "native Get worker launch failed");
        } else if (aclrtRecordEvent(get_done_, copy_stream_) != ACL_SUCCESS) {
          launch_error = "native Get worker failed to record completion event";
        }
      }

      std::lock_guard<std::mutex> lock(mu_);
      worker_start_time_ = worker_start_time;
      worker_issue_time_ = std::chrono::steady_clock::now();
      if (!launch_error.empty()) {
        FailLocked(std::move(launch_error));
      } else if (state_ == State::kPosted) {
        state_ = State::kIssued;
        cv_.notify_all();
      } else {
        FailLocked("native Get worker observed an invalid slot state");
      }
    }

    if (stream_resource_bound && unuse_stream_resource_ != nullptr) {
      (void)unuse_stream_resource_(copy_stream_);
    }
  }

  void StopWorkerNoThrow() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
      worker_wake_epoch_.fetch_add(1, std::memory_order_release);
      cv_.notify_all();
    }
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void DestroyEventsNoThrow() {
    if (worker_ready_ != nullptr) {
      (void)aclrtDestroyEvent(worker_ready_);
      worker_ready_ = nullptr;
    }
    if (terminal_done_ != nullptr) {
      (void)aclrtDestroyEvent(terminal_done_);
      terminal_done_ = nullptr;
    }
    if (get_done_ != nullptr) {
      (void)aclrtDestroyEvent(get_done_);
      get_done_ = nullptr;
    }
    if (descriptor_ready_ != nullptr) {
      (void)aclrtDestroyEvent(descriptor_ready_);
      descriptor_ready_ = nullptr;
    }
  }

  void CloseStreamResourceLibraryNoThrow() {
    if (stream_resource_library_ != nullptr) {
      (void)dlclose(stream_resource_library_);
      stream_resource_library_ = nullptr;
    }
    use_stream_resource_ = nullptr;
    unuse_stream_resource_ = nullptr;
    get_current_thread_resource_ = nullptr;
  }

  void CloseNoThrow() {
    // A destructor can run during Python exception/interpreter teardown, when
    // the caller may not have supplied the final-reader event.  Releasing the
    // HBM registration in that state would be worse than a bounded process
    // teardown leak, so only the explicit Close() path destroys an idle lane.
    if (!Idle()) {
      // The worker may be blocked in a device operation or its consumer may
      // still read the fixed HBM output.  A destructor cannot prove either
      // completion, so it intentionally does not join or release here.
      return;
    }
    StopWorkerNoThrow();
    if (output_binding_id_ != 0) {
      (void)access_operator_.ReleaseBatchGetOutput(output_binding_id_);
      output_binding_id_ = 0;
    }
    DestroyEventsNoThrow();
    CloseStreamResourceLibraryNoThrow();
  }

  DefaultAccessOperator& access_operator_;
  std::uint32_t device_id_{0};
  aclrtStream copy_stream_{nullptr};
  bool bind_stream_resource_{false};
  std::uint32_t expected_get_vector_cores_{0};
  NativeGetWorkerWaitMode worker_wait_mode_{NativeGetWorkerWaitMode::kCv};
  void* stream_resource_library_{nullptr};
  UseStreamResourceFn use_stream_resource_{nullptr};
  UseStreamResourceFn unuse_stream_resource_{nullptr};
  GetCurrentThreadResourceFn get_current_thread_resource_{nullptr};
  aclrtEvent descriptor_ready_{nullptr};
  aclrtEvent get_done_{nullptr};
  aclrtEvent terminal_done_{nullptr};
  aclrtEvent worker_ready_{nullptr};
  MutableBufferView output_;
  BatchGetOutputBindingId output_binding_id_{0};
  BatchGetRequest prepared_request_;
  bool has_prepared_request_{false};
  BatchGetRequest job_;
  std::chrono::steady_clock::time_point post_time_;
  std::chrono::steady_clock::time_point worker_start_time_;
  std::chrono::steady_clock::time_point worker_issue_time_;
  std::atomic<std::uint64_t> worker_wake_epoch_{0};
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::thread worker_;
  State state_{State::kFree};
  bool consumer_may_read_output_{false};
  bool stopping_{false};
  bool worker_initialized_{false};
  std::string worker_error_;
};

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
    EnsureNoNativeGetLane("initialize");
    EnsureNoFusedBatchGetControl("initialize");
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
    EnsureNoNativeGetLane("batch_put");
    // A fused Get reads the Runtime-owned DDR region directly from AIV.  A
    // concurrent Put could alter the same payload or metadata lifetime while
    // the graph is still using its fixed DeviceAccessTable.
    EnsureNoFusedBatchGetControl("batch_put");
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
    EnsureNoNativeGetLane("prepare_access_table");
    // This may allocate/replace Runtime metadata that a captured fused graph
    // still treats as stable for its current layer.
    EnsureNoFusedBatchGetControl("prepare_access_table");
    std::vector<KeyId> copied_keys = CopyPrepareKeys(keys);
    AccessTableHandle handle;
    ThrowIfError(runtime_->PrepareAccessTable(
        AccessTablePrepareRequest{std::move(copied_keys)}, &handle));
    return handle;
  }

  py::object AccessTableView(const AccessTableHandle& handle,
                             const std::string& device,
                             std::uint64_t capacity) const {
    EnsureNoNativeGetLane("access_table_view");
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
    EnsureNoNativeGetLane("release_access_table");
    // Releasing an AccessTable can return remote slots to Runtime metadata
    // even though a fused ExactScan + Get graph still owns its device view.
    EnsureNoFusedBatchGetControl("release_access_table");
    ThrowIfError(runtime_->ReleaseAccessTable(handle));
  }

  // BatchFree is a control-plane operation.  It deliberately takes the same
  // Python key forms as PrepareAccessTable (a Python key sequence or a
  // contiguous one-dimensional CPU torch.int64 Tensor), rather than imposing
  // a second representation solely for the NPU Runtime binding.
  void BatchFree(const py::object& keys) {
    EnsureNoNativeGetLane("batch_free");
    EnsureNoFusedBatchGetControl("batch_free");
    std::vector<KeyId> copied_keys = CopyPrepareKeys(keys);
    ThrowIfError(runtime_->BatchFree(BatchFreeRequest{std::move(copied_keys)}));
  }

  void StartBatchGetKernelProfile() {
    EnsureNoNativeGetLane("start_batch_get_kernel_profile");
    EnsureNoFusedBatchGetControl("start_batch_get_kernel_profile");
    ThrowIfError(access_operator_->SetBatchGetKernelProfiling(true));
  }

  py::list BatchGetKernelProfileSnapshot() const {
    EnsureNoNativeGetLane("batch_get_kernel_profile");
    EnsureNoFusedBatchGetControl("batch_get_kernel_profile");
    std::vector<kvaf::BatchGetKernelProfile> profiles;
    ThrowIfError(access_operator_->CopyBatchGetKernelProfiles(&profiles));
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
    EnsureNoNativeGetLane("stop_batch_get_kernel_profile");
    EnsureNoFusedBatchGetControl("stop_batch_get_kernel_profile");
    py::list result = BatchGetKernelProfileSnapshot();
    ThrowIfError(access_operator_->SetBatchGetKernelProfiling(false));
    return result;
  }

  // Tensor ownership remains with Python. KVAF creates no output Tensor and
  // only retains its internal Direct-HCCP execution state; callers must keep
  // input/output Tensors and either the DeviceAccessTable or the direct
  // addresses/valid Tensors alive until their stream completes BatchGet.
  void BatchGet(const DeviceAccessTable& access_table,
                const py::object& indices, const py::object& offsets,
                const py::object& mask,
                const py::object& output_indices, const py::object& output,
                const py::object& stream, bool synchronize) {
    EnsureNoNativeGetLane("batch_get");
    EnsureNoFusedBatchGetControl("batch_get");
    BatchGetImpl(access_table.source(), access_table.addresses(),
                 access_table.valid(), indices, offsets, mask, output_indices,
                 output, stream, synchronize);
  }

  // This overload accepts a caller-owned device table directly. Since bare
  // Tensors carry no AccessTableHandle, Runtime validates their NPU layout but
  // cannot validate a corresponding Host AccessTable lifetime or entry length.
  void BatchGetFromTensors(const py::object& addresses,
                           const py::object& valid,
                           const py::object& indices,
                           const py::object& offsets,
                           const py::object& mask,
                           const py::object& output_indices,
                           const py::object& output,
                           const py::object& stream, bool synchronize) {
    EnsureNoNativeGetLane("batch_get");
    EnsureNoFusedBatchGetControl("batch_get");
    BatchGetImpl(AccessTableHandle{}, addresses, valid, indices, offsets, mask,
                 output_indices, output, stream, synchronize);
  }

  // Eager-only host-submission escape hatch for the LI/Get overlap probe.
  // It deliberately exposes no ticket, retry, or general scheduler: one
  // Runtime owns one fixed HBM output, one prepared descriptor view, and one
  // worker-owned copy stream. Before creating it, the caller must complete
  // every earlier eager Get issued by this Runtime; Direct-HCCP queue/status
  // state is shared and a host mutex cannot isolate an older device launch.
  // All native-lane calls must remain on the creating host thread. This keeps
  // the single-worker probe from racing Python-side destroy/shutdown while
  // Arm/Reap intentionally release the GIL around their blocking portions.
  // The public event methods only express device ordering and final-reader
  // lifetime; callers must use record_native_get_terminal after their last
  // consumer (QSFA in the real path) has been queued.
  void CreateNativeGetSubmitLane(const py::object& copy_stream,
                                 bool bind_stream_resource,
                                 std::uint32_t expected_get_vector_cores,
                                 const std::string& worker_wait) {
    if (!Initialized()) {
      throw std::runtime_error("Runtime has not been initialized");
    }
    EnsureNoFusedBatchGetControl("create_native_get_submit_lane");
    if (native_get_lane_ != nullptr) {
      EnsureNativeGetLaneOwner("create_native_get_submit_lane");
      throw py::value_error("native Get submit lane already exists");
    }
    NativeGetWorkerWaitMode worker_wait_mode;
    if (worker_wait == "cv") {
      worker_wait_mode = NativeGetWorkerWaitMode::kCv;
    } else if (worker_wait == "dedicated_spin") {
      worker_wait_mode = NativeGetWorkerWaitMode::kDedicatedSpin;
    } else {
      throw py::value_error(
          "native Get worker_wait must be 'cv' or 'dedicated_spin'");
    }
    const aclrtStream raw_copy_stream = ResolveStream(copy_stream);
    auto lane = std::make_unique<NativeGetSubmitLane>(
        *access_operator_, device_id_, raw_copy_stream, bind_stream_resource,
        expected_get_vector_cores, worker_wait_mode);
    ThrowIfError(lane->Initialize());
    native_get_lane_copy_stream_keepalive_ = copy_stream;
    native_get_lane_ = std::move(lane);
    native_get_lane_owner_thread_ = std::this_thread::get_id();
  }

  void BindNativeGetOutput(const py::object& output) {
    EnsureNoFusedBatchGetControl("bind_native_get_output");
    NativeGetSubmitLane& lane = RequireNativeGetLane("bind_native_get_output");
    const NpuTensorView output_view = GetNpuTensorView(output, "native Get output");
    if (output_view.shape.size() < 2 || output_view.shape.front() <= 0 ||
        output_view.bytes % static_cast<std::size_t>(output_view.shape.front()) != 0 ||
        output_view.bytes / static_cast<std::size_t>(output_view.shape.front()) !=
            kBatchGetTransferBytes) {
      throw py::value_error("native Get output Tensor rows must each be 656 bytes");
    }
    ThrowIfError(lane.BindOutput(
        MutableBufferView{output_view.data, output_view.bytes, DeviceKind::kNpu}));
    native_get_lane_output_keepalive_ = output;
  }

  void PrepareNativeGet(const py::object& addresses, const py::object& valid,
                        const py::object& indices, const py::object& offsets,
                        const py::object& mask,
                        const py::object& output_indices,
                        const py::object& output) {
    NativeGetSubmitLane& lane = RequireNativeGetLane("prepare_native_get");
    const BatchGetRequest request = BuildBatchGetRequest(
        AccessTableHandle{}, addresses, valid, indices, offsets, mask,
        output_indices, output);
    // The worker sees raw pointers only. Store the Python references only
    // after C++ accepted the prepared POD request, so a rejected re-prepare
    // cannot drop the references of a job that is still in flight.
    py::tuple pending_keepalive = py::make_tuple(
        addresses, valid, indices, offsets, mask, output_indices, output);
    ThrowIfError(lane.Prepare(request));
    native_get_lane_request_keepalive_ = std::move(pending_keepalive);
  }

  void PostNativeGet(const py::object& producer_stream) {
    NativeGetSubmitLane& lane = RequireNativeGetLane("post_native_get");
    const aclrtStream raw_producer_stream = ResolveStream(producer_stream);
    Status status;
    {
      py::gil_scoped_release release;
      status = lane.Post(raw_producer_stream);
    }
    ThrowIfError(status);
  }

  py::dict ArmNativeGet(const py::object& consumer_stream) {
    NativeGetSubmitLane& lane = RequireNativeGetLane("arm_native_get");
    double post_to_worker_start_us = 0.0;
    double worker_issue_us = 0.0;
    double post_to_issue_us = 0.0;
    double arm_wait_us = 0.0;
    const aclrtStream raw_consumer_stream = ResolveStream(consumer_stream);
    Status status;
    {
      py::gil_scoped_release release;
      status = lane.Arm(raw_consumer_stream, &post_to_worker_start_us,
                        &worker_issue_us, &post_to_issue_us, &arm_wait_us);
    }
    ThrowIfError(status);
    py::dict result;
    result["post_to_worker_start_us"] = post_to_worker_start_us;
    result["worker_issue_us"] = worker_issue_us;
    result["post_to_worker_issue_us"] = post_to_issue_us;
    result["arm_wait_us"] = arm_wait_us;
    return result;
  }

  void RecordNativeGetTerminal(const py::object& terminal_stream) {
    NativeGetSubmitLane& lane = RequireNativeGetLane("record_native_get_terminal");
    ThrowIfError(lane.RecordTerminal(ResolveStream(terminal_stream)));
  }

  py::dict ReapNativeGet() {
    NativeGetSubmitLane& lane = RequireNativeGetLane("reap_native_get");
    NativeGetReapTiming timing;
    Status status;
    {
      py::gil_scoped_release release;
      status = lane.Reap(&timing);
    }
    ThrowIfError(status);
    py::dict result;
    result["terminal_sync_wait_us"] = timing.terminal_sync_us;
    result["event_reclaim_wait_us"] = timing.reclaim_wait_us;
    result["total_us"] = timing.total_us;
    return result;
  }

  void DestroyNativeGetSubmitLane() {
    if (native_get_lane_ == nullptr) {
      return;
    }
    EnsureNativeGetLaneOwner("destroy_native_get_submit_lane");
    Status status;
    {
      py::gil_scoped_release release;
      status = native_get_lane_->Close();
    }
    ThrowIfError(status);
    native_get_lane_.reset();
    native_get_lane_owner_thread_ = std::thread::id{};
    native_get_lane_request_keepalive_ = py::none();
    native_get_lane_output_keepalive_ = py::none();
    native_get_lane_copy_stream_keepalive_ = py::none();
  }

  // Bind the fixed output registration and Direct-HCCP SQ/CQ state that a
  // device-side fused ExactScan + Get operator consumes. The control tensor
  // is intentionally plain NPU torch.int64[16], not a Python object with
  // hidden host state: the fused kernel can load every pointer it needs
  // directly from HBM during graph replay.
  void BindFusedBatchGetControl(const py::object& output,
                                const py::object& control,
                                const py::object& producer_stream,
                                std::uint32_t queue_count) {
    EnsureNoNativeGetLane("bind_fused_batch_get_control");
    if (!Initialized()) {
      throw std::runtime_error("Runtime has not been initialized");
    }
    if (HasFusedOrStreamingBatchGetControl()) {
      throw py::value_error("a fused or streaming BatchGet control is already bound");
    }
    // The control and output tensors are normally allocated or prepared on a
    // PyTorch stream.  BindFusedBatchGetControl publishes the cold control
    // block with raw ACL calls, which do not inherit that stream's ordering.
    // Drain the explicit producer stream first so an earlier torch.zeros (or
    // any caller-side initialization) cannot later overwrite the KVAF copy.
    // This is a one-time lifecycle fence before graph capture, never a replay
    // path synchronization.
    const aclrtStream raw_producer_stream = ResolveStream(producer_stream);
    ThrowIfAclError(aclrtSynchronizeStream(raw_producer_stream),
                    "aclrtSynchronizeStream before fused BatchGet bind");
    const NpuTensorView output_view =
        GetNpuTensorView(output, "fused BatchGet output");
    const NpuTensorView control_view =
        GetNpuTensorView(control, "fused BatchGet control");
    ValidateFusedBatchGetOutputTensor(output_view, "fused BatchGet output");
    ValidateFusedBatchGetControlTensor(control_view, control,
                                       "fused BatchGet control");
    ThrowIfError(access_operator_->BindFusedBatchGetControl(
        MutableBufferView{output_view.data, output_view.bytes, DeviceKind::kNpu},
        MutableBufferView{control_view.data, control_view.bytes,
                          DeviceKind::kNpu},
        queue_count));
    fused_batch_get_control_bound_ = true;
    fused_batch_get_output_keepalive_ = output;
    fused_batch_get_control_keepalive_ = control;
  }

  void ReleaseFusedBatchGetControl() {
    if (!fused_batch_get_control_bound_) {
      return;
    }
    EnsureNoNativeGetLane("release_fused_batch_get_control");
    ThrowIfError(access_operator_->ReleaseFusedBatchGetControl());
    fused_batch_get_control_bound_ = false;
    fused_batch_get_output_keepalive_ = py::none();
    fused_batch_get_control_keepalive_ = py::none();
  }

  // Cold-bind a layer-keyed table for the streaming producer-owned queue
  // pool. All controls share one HBM int32[queue_count] lease table, while
  // each layer retains its own registered output/entity table. There is no
  // hot per-layer update: callers release the entire table after the final
  // graph reader, then bind a replacement at a cold graph lifecycle point.
  void BindStreamingBatchGetControl(const py::sequence& layer_ids,
                                    const py::sequence& outputs,
                                    const py::sequence& controls,
                                    const py::object& queue_leases,
                                    std::uint32_t queue_count,
                                    const py::object& producer_stream) {
    EnsureNoNativeGetLane("bind_streaming_batch_get_control");
    if (!Initialized()) {
      throw std::runtime_error("Runtime has not been initialized");
    }
    if (HasFusedOrStreamingBatchGetControl()) {
      throw py::value_error("a fused or streaming BatchGet control is already bound");
    }
    const py::ssize_t count = py::len(layer_ids);
    if (count <= 0 || py::len(outputs) != count || py::len(controls) != count) {
      throw py::value_error(
          "streaming BatchGet layer_ids, outputs, and controls must be "
          "non-empty sequences of equal length");
    }
    const NpuTensorView queue_leases_view =
        GetNpuTensorView(queue_leases, "streaming BatchGet queue_leases");
    if (queue_leases_view.shape.size() != 1 ||
        queue_leases_view.element_bytes != sizeof(std::int32_t) ||
        queue_leases_view.shape.front() != static_cast<std::int64_t>(queue_count) ||
        py::str(queue_leases.attr("dtype")).cast<std::string>() != "torch.int32") {
      throw py::value_error(
          "streaming BatchGet queue_leases must be a contiguous NPU torch.int32[queue_count]");
    }

    std::vector<StreamingBatchGetLayerBinding> bindings;
    std::vector<py::object> output_keepalives;
    std::vector<py::object> control_keepalives;
    bindings.reserve(static_cast<std::size_t>(count));
    output_keepalives.reserve(static_cast<std::size_t>(count));
    control_keepalives.reserve(static_cast<std::size_t>(count));
    for (py::ssize_t index = 0; index < count; ++index) {
      const py::ssize_t layer_id_value = layer_ids[index].cast<py::ssize_t>();
      if (layer_id_value < 0 ||
          static_cast<std::uint64_t>(layer_id_value) >
              std::numeric_limits<std::uint32_t>::max()) {
        throw py::value_error("streaming BatchGet layer_ids must fit uint32");
      }
      py::object output = outputs[index].cast<py::object>();
      py::object control = controls[index].cast<py::object>();
      const NpuTensorView output_view =
          GetNpuTensorView(output, "streaming BatchGet output");
      const NpuTensorView control_view =
          GetNpuTensorView(control, "streaming BatchGet control");
      const std::string label =
          "streaming BatchGet layer " + std::to_string(layer_id_value);
      ValidateFusedBatchGetOutputTensor(output_view, label + " output");
      ValidateFusedBatchGetControlTensor(control_view, control,
                                         label + " control");
      bindings.push_back(StreamingBatchGetLayerBinding{
          static_cast<std::uint32_t>(layer_id_value),
          MutableBufferView{output_view.data, output_view.bytes, DeviceKind::kNpu},
          MutableBufferView{control_view.data, control_view.bytes,
                            DeviceKind::kNpu}});
      output_keepalives.push_back(std::move(output));
      control_keepalives.push_back(std::move(control));
    }

    const aclrtStream raw_producer_stream = ResolveStream(producer_stream);
    ThrowIfAclError(aclrtSynchronizeStream(raw_producer_stream),
                    "aclrtSynchronizeStream before streaming BatchGet bind");
    ThrowIfError(access_operator_->BindStreamingBatchGetControl(
        bindings,
        MutableBufferView{queue_leases_view.data, queue_leases_view.bytes,
                          DeviceKind::kNpu},
        queue_count));
    streaming_batch_get_control_bound_ = true;
    streaming_batch_get_output_keepalives_ = std::move(output_keepalives);
    streaming_batch_get_control_keepalives_ = std::move(control_keepalives);
    streaming_batch_get_queue_leases_keepalive_ = queue_leases;
  }

  // Publish the graph-stable descriptor-pointer table for every queue of
  // every already-bound streaming layer.  This is intentionally a cold
  // lifecycle operation: the LI AIV code later copies the selected queue's
  // template, fills only that queue's candidate vectors/count, and rings its
  // own doorbell.  There is no host-side descriptor rewrite during capture or
  // replay.
  void ConfigureStreamingBatchGetQueueTemplates(
      const py::sequence& layer_ids, const py::sequence& queue_ids,
      const py::sequence& address_tables, const py::sequence& valid_tables,
      const py::sequence& indices, const py::sequence& offsets,
      const py::sequence& masks, const py::sequence& output_indices,
      const py::object& producer_stream) {
    EnsureNoNativeGetLane("configure_streaming_batch_get_queue_templates");
    if (!Initialized()) {
      throw std::runtime_error("Runtime has not been initialized");
    }
    if (!streaming_batch_get_control_bound_) {
      throw py::value_error(
          "configure_streaming_batch_get_queue_templates requires a bound "
          "streaming BatchGet control table");
    }
    const py::ssize_t count = py::len(layer_ids);
    if (count <= 0 || py::len(queue_ids) != count ||
        py::len(address_tables) != count || py::len(valid_tables) != count ||
        py::len(indices) != count || py::len(offsets) != count ||
        py::len(masks) != count || py::len(output_indices) != count) {
      throw py::value_error(
          "streaming descriptor templates require non-empty equal-length "
          "layer_ids, queue_ids, address_tables, valid_tables, indices, "
          "offsets, masks, and output_indices sequences");
    }

    const auto checked_layer_id = [](const py::object& value,
                                     const char* name) -> std::uint32_t {
      const py::ssize_t converted = value.cast<py::ssize_t>();
      if (converted < 0 ||
          static_cast<std::uint64_t>(converted) >
              std::numeric_limits<std::uint32_t>::max()) {
        throw py::value_error(std::string(name) + " must fit uint32");
      }
      return static_cast<std::uint32_t>(converted);
    };
    const auto validate_vector = [](const NpuTensorView& view,
                                    const py::object& tensor,
                                    std::size_t element_bytes,
                                    const char* dtype,
                                    const std::string& label) {
      if (view.shape.size() != 1 || view.element_bytes != element_bytes ||
          py::str(tensor.attr("dtype")).cast<std::string>() != dtype) {
        throw py::value_error(label + " must be a contiguous one-dimensional " +
                              std::string(dtype) + " NPU Tensor");
      }
    };

    std::vector<StreamingBatchGetLayerTemplates> templates;
    std::vector<py::object> keepalives;
    templates.reserve(static_cast<std::size_t>(count));
    keepalives.reserve(static_cast<std::size_t>(count) * 6U);
    for (py::ssize_t index = 0; index < count; ++index) {
      py::object layer_id_object = layer_ids[index].cast<py::object>();
      py::object queue_id_object = queue_ids[index].cast<py::object>();
      const std::uint32_t layer_id =
          checked_layer_id(layer_id_object, "streaming descriptor layer_id");
      const std::uint32_t queue_id =
          checked_layer_id(queue_id_object, "streaming descriptor queue_id");
      py::object address_table = address_tables[index].cast<py::object>();
      py::object valid_table = valid_tables[index].cast<py::object>();
      py::object index_vector = indices[index].cast<py::object>();
      py::object offset_vector = offsets[index].cast<py::object>();
      py::object mask = masks[index].cast<py::object>();
      py::object output_index_vector = output_indices[index].cast<py::object>();
      const std::string label =
          "streaming descriptor layer=" + std::to_string(layer_id) +
          " queue=" + std::to_string(queue_id);
      const NpuTensorView address_view =
          GetNpuTensorView(address_table, (label + " address_table").c_str());
      const NpuTensorView valid_view =
          GetNpuTensorView(valid_table, (label + " valid_table").c_str());
      const NpuTensorView index_view =
          GetNpuTensorView(index_vector, (label + " indices").c_str());
      const NpuTensorView offset_view =
          GetNpuTensorView(offset_vector, (label + " offsets").c_str());
      const NpuTensorView mask_view =
          GetNpuTensorView(mask, (label + " mask").c_str());
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
      validate_vector(mask_view, mask, sizeof(std::uint8_t), "torch.uint8",
                      label + " mask");
      validate_vector(output_index_view, output_index_vector,
                      sizeof(std::int32_t), "torch.int32",
                      label + " output_indices");

      StreamingBatchGetLayerTemplates* layer_templates = nullptr;
      for (StreamingBatchGetLayerTemplates& candidate : templates) {
        if (candidate.layer_id == layer_id) {
          layer_templates = &candidate;
          break;
        }
      }
      if (layer_templates == nullptr) {
        templates.push_back(StreamingBatchGetLayerTemplates{});
        templates.back().layer_id = layer_id;
        layer_templates = &templates.back();
      }
      layer_templates->queues.push_back(StreamingBatchGetQueueTemplate{
          queue_id,
          MutableBufferView{address_view.data, address_view.bytes,
                            DeviceKind::kNpu},
          MutableBufferView{valid_view.data, valid_view.bytes, DeviceKind::kNpu},
          MutableBufferView{index_view.data, index_view.bytes, DeviceKind::kNpu},
          MutableBufferView{offset_view.data, offset_view.bytes,
                            DeviceKind::kNpu},
          MutableBufferView{mask_view.data, mask_view.bytes, DeviceKind::kNpu},
          MutableBufferView{output_index_view.data, output_index_view.bytes,
                            DeviceKind::kNpu}});
      keepalives.push_back(std::move(address_table));
      keepalives.push_back(std::move(valid_table));
      keepalives.push_back(std::move(index_vector));
      keepalives.push_back(std::move(offset_vector));
      keepalives.push_back(std::move(mask));
      keepalives.push_back(std::move(output_index_vector));
    }

    const aclrtStream raw_producer_stream = ResolveStream(producer_stream);
    ThrowIfAclError(
        aclrtSynchronizeStream(raw_producer_stream),
        "aclrtSynchronizeStream before streaming descriptor template configure");
    ThrowIfError(
        access_operator_->ConfigureStreamingBatchGetQueueTemplates(templates));
    streaming_batch_get_template_keepalives_ = std::move(keepalives);
  }

  void ReleaseStreamingBatchGetControl() {
    if (!streaming_batch_get_control_bound_) {
      return;
    }
    EnsureNoNativeGetLane("release_streaming_batch_get_control");
    ThrowIfError(access_operator_->ReleaseStreamingBatchGetControl());
    streaming_batch_get_control_bound_ = false;
    streaming_batch_get_output_keepalives_.clear();
    streaming_batch_get_control_keepalives_.clear();
    streaming_batch_get_template_keepalives_.clear();
    streaming_batch_get_queue_leases_keepalive_ = py::none();
  }

  // P9 binds one already-reserved canonical resident interval for one fused
  // layer invocation.  It is deliberately separate from the P8/private and
  // multi-layer streaming APIs: after the fused op joins every CQ, QSFA reads
  // the same resident rows directly.
  void BindLayerResidentBatchGetControl(
      std::uint32_t layer_id, const py::object& output,
      const py::object& control, std::uint32_t resident_row_base,
      std::uint32_t resident_row_count, const py::object& queue_leases,
      std::uint32_t queue_count, const py::object& producer_stream) {
    EnsureNoNativeGetLane("bind_layer_resident_batch_get_control");
    if (!Initialized()) {
      throw std::runtime_error("Runtime has not been initialized");
    }
    if (HasFusedOrStreamingBatchGetControl()) {
      throw py::value_error(
          "a fused, streaming, or layer-resident BatchGet control is already bound");
    }
    const NpuTensorView output_view =
        GetNpuTensorView(output, "layer-resident BatchGet output");
    const NpuTensorView control_view =
        GetNpuTensorView(control, "layer-resident BatchGet control");
    const NpuTensorView leases_view =
        GetNpuTensorView(queue_leases, "layer-resident BatchGet queue_leases");
    ValidateFusedBatchGetOutputTensor(output_view,
                                      "layer-resident BatchGet output");
    ValidateFusedBatchGetControlTensor(control_view, control,
                                       "layer-resident BatchGet control");
    const std::uint64_t output_rows = output_view.bytes / 656U;
    if (resident_row_count == 0U || resident_row_base >= output_rows ||
        resident_row_count > output_rows - resident_row_base) {
      throw py::value_error(
          "layer-resident interval must be a non-empty range within output rows");
    }
    if (leases_view.shape.size() != 1 ||
        leases_view.element_bytes != sizeof(std::int32_t) ||
        leases_view.shape.front() != static_cast<std::int64_t>(queue_count) ||
        py::str(queue_leases.attr("dtype")).cast<std::string>() != "torch.int32") {
      throw py::value_error(
          "layer-resident queue_leases must be NPU torch.int32[queue_count]");
    }
    const aclrtStream raw_producer_stream = ResolveStream(producer_stream);
    ThrowIfAclError(aclrtSynchronizeStream(raw_producer_stream),
                    "aclrtSynchronizeStream before layer-resident BatchGet bind");
    ThrowIfError(access_operator_->BindLayerResidentBatchGetControl(
        LayerResidentBatchGetBinding{
            layer_id,
            MutableBufferView{output_view.data, output_view.bytes,
                              DeviceKind::kNpu},
            MutableBufferView{control_view.data, control_view.bytes,
                              DeviceKind::kNpu},
            resident_row_base,
            resident_row_count},
        MutableBufferView{leases_view.data, leases_view.bytes, DeviceKind::kNpu},
        queue_count));
    layer_resident_batch_get_control_bound_ = true;
    layer_resident_batch_get_output_keepalive_ = output;
    layer_resident_batch_get_control_keepalive_ = control;
    layer_resident_batch_get_queue_leases_keepalive_ = queue_leases;
  }

  void ConfigureLayerResidentBatchGetQueueTemplates(
      const py::sequence& queue_ids, const py::sequence& address_tables,
      const py::sequence& valid_tables, const py::sequence& indices,
      const py::sequence& offsets, const py::sequence& masks,
      const py::sequence& output_indices, const py::object& producer_stream) {
    EnsureNoNativeGetLane("configure_layer_resident_batch_get_queue_templates");
    if (!Initialized() || !layer_resident_batch_get_control_bound_) {
      throw py::value_error(
          "layer-resident descriptor templates require a bound control");
    }
    const py::ssize_t count = py::len(queue_ids);
    if (count <= 0 || py::len(address_tables) != count ||
        py::len(valid_tables) != count || py::len(indices) != count ||
        py::len(offsets) != count || py::len(masks) != count ||
        py::len(output_indices) != count) {
      throw py::value_error(
          "layer-resident descriptor template sequences must be non-empty and equal-length");
    }
    const auto checked_queue_id = [](const py::object& value) {
      const py::ssize_t converted = value.cast<py::ssize_t>();
      if (converted < 0 ||
          static_cast<std::uint64_t>(converted) >
              std::numeric_limits<std::uint32_t>::max()) {
        throw py::value_error("layer-resident descriptor queue_id must fit uint32");
      }
      return static_cast<std::uint32_t>(converted);
    };
    const auto validate_vector = [](const NpuTensorView& view,
                                    const py::object& tensor,
                                    std::size_t element_bytes,
                                    const char* dtype,
                                    const std::string& label) {
      if (view.shape.size() != 1 || view.element_bytes != element_bytes ||
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
      py::object mask = masks[index].cast<py::object>();
      py::object output_index_vector = output_indices[index].cast<py::object>();
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
      const NpuTensorView mask_view =
          GetNpuTensorView(mask, (label + " mask").c_str());
      const NpuTensorView output_index_view =
          GetNpuTensorView(output_index_vector, (label + " output_indices").c_str());
      validate_vector(address_view, address_table, sizeof(std::uint64_t),
                      "torch.int64", label + " address_table");
      validate_vector(valid_view, valid_table, sizeof(std::uint8_t),
                      "torch.uint8", label + " valid_table");
      validate_vector(index_view, index_vector, sizeof(std::int32_t),
                      "torch.int32", label + " indices");
      validate_vector(offset_view, offset_vector, sizeof(std::int32_t),
                      "torch.int32", label + " offsets");
      validate_vector(mask_view, mask, sizeof(std::uint8_t), "torch.uint8",
                      label + " mask");
      validate_vector(output_index_view, output_index_vector,
                      sizeof(std::int32_t), "torch.int32",
                      label + " output_indices");
      templates.push_back(LayerResidentBatchGetQueueTemplate{
          queue_id,
          MutableBufferView{address_view.data, address_view.bytes, DeviceKind::kNpu},
          MutableBufferView{valid_view.data, valid_view.bytes, DeviceKind::kNpu},
          MutableBufferView{index_view.data, index_view.bytes, DeviceKind::kNpu},
          MutableBufferView{offset_view.data, offset_view.bytes, DeviceKind::kNpu},
          MutableBufferView{mask_view.data, mask_view.bytes, DeviceKind::kNpu},
          MutableBufferView{output_index_view.data, output_index_view.bytes,
                            DeviceKind::kNpu}});
      keepalives.push_back(std::move(address_table));
      keepalives.push_back(std::move(valid_table));
      keepalives.push_back(std::move(index_vector));
      keepalives.push_back(std::move(offset_vector));
      keepalives.push_back(std::move(mask));
      keepalives.push_back(std::move(output_index_vector));
    }
    const aclrtStream raw_producer_stream = ResolveStream(producer_stream);
    ThrowIfAclError(
        aclrtSynchronizeStream(raw_producer_stream),
        "aclrtSynchronizeStream before layer-resident descriptor configure");
    ThrowIfError(
        access_operator_->ConfigureLayerResidentBatchGetQueueTemplates(templates));
    layer_resident_batch_get_template_keepalives_ = std::move(keepalives);
  }

  void ReleaseLayerResidentBatchGetControl() {
    if (!layer_resident_batch_get_control_bound_) {
      return;
    }
    EnsureNoNativeGetLane("release_layer_resident_batch_get_control");
    ThrowIfError(access_operator_->ReleaseLayerResidentBatchGetControl());
    layer_resident_batch_get_control_bound_ = false;
    layer_resident_batch_get_output_keepalive_ = py::none();
    layer_resident_batch_get_control_keepalive_ = py::none();
    layer_resident_batch_get_queue_leases_keepalive_ = py::none();
    layer_resident_batch_get_template_keepalives_.clear();
  }

  std::string LayerResidentBatchGetControlDiagnostic() const {
    std::string diagnostic;
    ThrowIfError(access_operator_->DescribeLayerResidentBatchGetControl(
        &diagnostic));
    return diagnostic;
  }

  std::string StreamingBatchGetControlDiagnostic() const {
    std::string diagnostic;
    ThrowIfError(access_operator_->DescribeStreamingBatchGetControl(
        &diagnostic));
    return diagnostic;
  }

  py::dict StreamingBatchGetPoolSnapshot() const {
    StreamingBatchGetControlSnapshot snapshot;
    ThrowIfError(access_operator_->GetStreamingBatchGetControlSnapshot(
        &snapshot));
    py::dict result;
    result["magic"] = snapshot.magic;
    result["version"] = snapshot.version;
    result["active"] = snapshot.active;
    result["bind_epoch"] = snapshot.bind_epoch;
    result["bind_count"] = snapshot.bind_count;
    result["release_count"] = snapshot.release_count;
    result["template_configure_count"] = snapshot.template_configure_count;
    result["queue_count"] = snapshot.queue_count;
    result["queue_lease_address"] = snapshot.queue_lease_address;
    result["queue_lease_bytes"] = snapshot.queue_lease_bytes;
    result["descriptor_scratch_address"] = snapshot.descriptor_scratch_address;
    result["descriptor_scratch_bytes"] = snapshot.descriptor_scratch_bytes;
    py::list layers;
    for (const StreamingBatchGetLayerSnapshot& layer : snapshot.layers) {
      py::dict entry;
      entry["layer_id"] = layer.layer_id;
      entry["output_address"] = layer.output_address;
      entry["control_address"] = layer.control_address;
      entry["channel_entity_address"] = layer.channel_entity_address;
      entry["descriptor_template_address"] =
          layer.descriptor_template_address;
      entry["descriptor_template_bytes"] = layer.descriptor_template_bytes;
      entry["templates_configured"] = layer.templates_configured;
      entry["private_output_rows"] = layer.private_output_rows;
      entry["private_output_stride_bytes"] =
          layer.private_output_stride_bytes;
      layers.append(std::move(entry));
    }
    result["layers"] = std::move(layers);
    return result;
  }

 private:
  BatchGetRequest BuildBatchGetRequest(
      const AccessTableHandle& access_table, const py::object& addresses,
      const py::object& valid, const py::object& indices,
      const py::object& offsets, const py::object& mask,
      const py::object& output_indices, const py::object& output) const {
    const NpuTensorView index_view = GetNpuTensorView(indices, "indices");
    const NpuTensorView offset_view = GetNpuTensorView(offsets, "offsets");
    const NpuTensorView mask_view = GetNpuTensorView(mask, "mask");
    const NpuTensorView output_index_view =
        GetNpuTensorView(output_indices, "output_indices");
    const NpuTensorView output_view = GetNpuTensorView(output, "output");
    const NpuTensorView address_view = GetNpuTensorView(addresses, "addresses");
    const NpuTensorView valid_view = GetNpuTensorView(valid, "valid");
    if (index_view.shape.size() != 1 || offset_view.shape.size() != 1 ||
        mask_view.shape.size() != 1 ||
        output_index_view.shape.size() != 1 || output_view.shape.size() < 2 ||
        index_view.element_bytes != sizeof(std::int32_t) ||
        offset_view.element_bytes != sizeof(std::int32_t) ||
        output_index_view.element_bytes != sizeof(std::int32_t) ||
        mask_view.element_bytes != sizeof(std::uint8_t) ||
        index_view.bytes != offset_view.bytes ||
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
    if (output_block_bytes != kBatchGetTransferBytes) {
      throw py::value_error("BatchGet output Tensor rows must each be 656 bytes");
    }
    return BatchGetRequest{
        access_table,
        ConstBufferView{index_view.data, index_view.bytes, DeviceKind::kNpu},
        ConstBufferView{offset_view.data, offset_view.bytes, DeviceKind::kNpu},
        ConstBufferView{mask_view.data, mask_view.bytes, DeviceKind::kNpu},
        ConstBufferView{output_index_view.data, output_index_view.bytes,
                        DeviceKind::kNpu},
        MutableBufferView{output_view.data, output_view.bytes, DeviceKind::kNpu},
        output_block_bytes,
        AccessTableDeviceView{
            ConstBufferView{address_view.data, address_view.bytes, DeviceKind::kNpu},
            ConstBufferView{valid_view.data, valid_view.bytes, DeviceKind::kNpu}}};
  }

  void BatchGetImpl(const AccessTableHandle& access_table,
                    const py::object& addresses, const py::object& valid,
                    const py::object& indices, const py::object& offsets,
                    const py::object& mask,
                    const py::object& output_indices,
                    const py::object& output, const py::object& stream,
                    bool synchronize) {
    const BatchGetRequest request = BuildBatchGetRequest(
        access_table, addresses, valid, indices, offsets, mask, output_indices,
        output);
    const aclrtStream acl_stream = ResolveStream(stream);
    ThrowIfError(runtime_->BatchGet(
        request, BatchGetExecutionOptions{acl_stream,
                                           synchronize}));
  }

 public:
  void Shutdown() {
    // Explicitly close the worker before Runtime tears down its transport and
    // the cold HBM registration.  Close rejects an unretired slot rather than
    // guessing whether a later QSFA reader is still using the payload.
    DestroyNativeGetSubmitLane();
    if (HasFusedOrStreamingBatchGetControl()) {
      throw py::value_error(
          "shutdown is forbidden while fused or streaming BatchGet control "
          "is bound; wait for the final fused reader and release the "
          "control first");
    }
    if (runtime_ != nullptr && runtime_->initialized()) {
      ThrowIfError(runtime_->Shutdown());
    }
  }

 private:
  NativeGetSubmitLane& RequireNativeGetLane(const char* operation) {
    if (native_get_lane_ == nullptr) {
      throw py::value_error(std::string(operation) +
                            " requires create_native_get_submit_lane");
    }
    EnsureNativeGetLaneOwner(operation);
    return *native_get_lane_;
  }

  void EnsureNativeGetLaneOwner(const char* operation) const {
    if (native_get_lane_owner_thread_ != std::thread::id{} &&
        native_get_lane_owner_thread_ != std::this_thread::get_id()) {
      throw py::value_error(
          std::string(operation) +
          " must run on the host thread that created this native Get lane");
    }
  }

  void EnsureNoNativeGetLane(const char* operation) const {
    if (native_get_lane_ != nullptr) {
      throw py::value_error(
          std::string(operation) +
          " is forbidden while native Get submit lane owns this Runtime; "
          "reap and destroy the lane first");
    }
  }

  bool HasFusedOrStreamingBatchGetControl() const {
    return fused_batch_get_control_bound_ || streaming_batch_get_control_bound_ ||
           layer_resident_batch_get_control_bound_;
  }

  void EnsureNoFusedBatchGetControl(const char* operation) const {
    if (HasFusedOrStreamingBatchGetControl()) {
      throw py::value_error(
          std::string(operation) +
          " is forbidden while fused, streaming, or layer-resident BatchGet "
          "control owns this Runtime; "
          "wait for the final fused reader and release the control first");
    }
  }

  static void LeakPyObjects(std::vector<py::object>* objects) {
    if (objects == nullptr) {
      return;
    }
    for (py::object& object : *objects) {
      (void)object.release();
    }
    objects->clear();
  }

  void CloseNoThrow() {
    // Destruction is normally preceded by the Python-visible shutdown(),
    // which enforces terminal completion.  Do not call Runtime::Shutdown
    // behind a live lane during exceptional teardown: DefaultAccessOperator
    // would release its transport while an HCCP Get might still be in flight.
    if (native_get_lane_ != nullptr && !native_get_lane_->Idle()) {
      // Keep Tensor/stream references alive together with the deliberately
      // leaked emergency objects.  This branch is only an exception-safety
      // backstop; normal callers must reap+destroy explicitly.
      (void)native_get_lane_copy_stream_keepalive_.release();
      (void)native_get_lane_output_keepalive_.release();
      (void)native_get_lane_request_keepalive_.release();
      (void)native_get_lane_.release();
      (void)runtime_.release();
      (void)access_operator_.release();
      (void)allocator_.release();
      (void)meta_manager_.release();
      return;
    }
    if (HasFusedOrStreamingBatchGetControl()) {
      // The Python-visible release method requires the caller to establish
      // the terminal reader.  At exception-time destruction we cannot prove
      // that a captured fused graph stopped reading the output, so preserve
      // the Runtime, transport and both Tensor references rather than freeing
      // Direct-HCCP state underneath an in-flight AIV.
      (void)fused_batch_get_output_keepalive_.release();
      (void)fused_batch_get_control_keepalive_.release();
      LeakPyObjects(&streaming_batch_get_output_keepalives_);
      LeakPyObjects(&streaming_batch_get_control_keepalives_);
      LeakPyObjects(&streaming_batch_get_template_keepalives_);
      (void)streaming_batch_get_queue_leases_keepalive_.release();
      (void)layer_resident_batch_get_output_keepalive_.release();
      (void)layer_resident_batch_get_control_keepalive_.release();
      (void)layer_resident_batch_get_queue_leases_keepalive_.release();
      LeakPyObjects(&layer_resident_batch_get_template_keepalives_);
      (void)runtime_.release();
      (void)access_operator_.release();
      (void)allocator_.release();
      (void)meta_manager_.release();
      return;
    }
    if (native_get_lane_ != nullptr) {
      native_get_lane_.reset();
    }
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
  std::unique_ptr<NativeGetSubmitLane> native_get_lane_;
  std::thread::id native_get_lane_owner_thread_;
  py::object native_get_lane_copy_stream_keepalive_;
  py::object native_get_lane_output_keepalive_;
  py::object native_get_lane_request_keepalive_;
  bool fused_batch_get_control_bound_{false};
  py::object fused_batch_get_output_keepalive_;
  py::object fused_batch_get_control_keepalive_;
  bool streaming_batch_get_control_bound_{false};
  std::vector<py::object> streaming_batch_get_output_keepalives_;
  std::vector<py::object> streaming_batch_get_control_keepalives_;
  std::vector<py::object> streaming_batch_get_template_keepalives_;
  py::object streaming_batch_get_queue_leases_keepalive_;
  bool layer_resident_batch_get_control_bound_{false};
  py::object layer_resident_batch_get_output_keepalive_;
  py::object layer_resident_batch_get_control_keepalive_;
  py::object layer_resident_batch_get_queue_leases_keepalive_;
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
      .def("start_batch_get_kernel_profile",
           &NpuRuntime::StartBatchGetKernelProfile,
           "Enable the per-core BatchGet kernel cycle timeline.")
      .def("batch_get_kernel_profile",
           &NpuRuntime::BatchGetKernelProfileSnapshot,
           "Copy the latest completed per-core BatchGet kernel timeline.")
      .def("stop_batch_get_kernel_profile",
           &NpuRuntime::StopBatchGetKernelProfile,
           "Copy the latest timeline and disable BatchGet kernel profiling.")
      .def("create_native_get_submit_lane",
           &NpuRuntime::CreateNativeGetSubmitLane,
           py::arg("copy_stream"),
           py::arg("bind_stream_resource") = false,
           py::arg("expected_get_vector_cores") = 0,
           py::arg("worker_wait") = "cv",
           "Create the eager-only, single-worker Get submission lane. "
           "The worker exclusively owns copy_stream submissions until "
           "destroy_native_get_submit_lane(). The caller must first complete "
           "all earlier eager Get work from this Runtime and use the creating "
           "host thread for every native-lane call. worker_wait='dedicated_spin' "
           "is an opt-in host-CPU latency probe, not the default runtime policy.")
      .def("bind_native_get_output", &NpuRuntime::BindNativeGetOutput,
           py::arg("output"),
           "Cold-bind one fixed HBM output Tensor for native Get. The binding "
           "remains live until the lane is destroyed after terminal completion.")
      .def("prepare_native_get", &NpuRuntime::PrepareNativeGet,
           py::arg("addresses"), py::arg("valid"), py::arg("indices"),
           py::arg("offsets"), py::arg("mask"), py::arg("output_indices"),
           py::arg("output"),
           "Cold-validate and retain stable descriptor/output Tensor identities "
           "for the native lane. Descriptor contents may be updated in-place.")
      .def("post_native_get", &NpuRuntime::PostNativeGet,
           py::arg("producer_stream"),
           "Hot-post the prepared request: record descriptor readiness and wake "
           "the C++ worker without Python Tensor validation or HBM registration.")
      .def("arm_native_get", &NpuRuntime::ArmNativeGet,
           py::arg("consumer_stream"),
           "After submitting LI/QSFA, wait until the worker has recorded the "
           "Get completion event and enqueue its dependency on consumer_stream.")
      .def("record_native_get_terminal", &NpuRuntime::RecordNativeGetTerminal,
           py::arg("terminal_stream"),
           "Record the final-reader terminal on the already-armed consumer stream.")
      .def("reap_native_get", &NpuRuntime::ReapNativeGet,
           "Wait until the terminal event has completed and make the fixed slot reusable. "
           "Returns terminal-sync, event-reclaim, and total host timings in microseconds.")
      .def("destroy_native_get_submit_lane",
           &NpuRuntime::DestroyNativeGetSubmitLane,
           "Stop the worker and release the fixed HBM binding. The slot must be reaped.")
      .def("bind_fused_batch_get_control",
           &NpuRuntime::BindFusedBatchGetControl, py::arg("output"),
           py::arg("control"), py::arg("producer_stream") = py::none(),
           py::arg("queue_count") = 2,
           "Cold-bind fixed output HBM and write a graph-stable uint64[16] "
           "Direct-HCCP control block for the legacy fused Get path. "
           "queue_count selects [1, npu_core_count] queues and requires "
           "jettys_per_core=1; ordinary "
           "and native-lane BatchGet remain unavailable until release. The "
           "caller must first complete earlier ordinary Get work from this Runtime. "
           "producer_stream orders any earlier PyTorch writes to output/control; "
           "None selects torch.npu.current_stream().")
      .def("release_fused_batch_get_control",
           &NpuRuntime::ReleaseFusedBatchGetControl,
           "Release the fused Get HBM registration after the final fused "
           "consumer has completed. This method does not synchronize a stream.")
      .def("bind_streaming_batch_get_control",
           &NpuRuntime::BindStreamingBatchGetControl, py::arg("layer_ids"),
           py::arg("outputs"), py::arg("controls"), py::arg("queue_leases"),
           py::arg("queue_count"),
           py::arg("producer_stream") = py::none(),
           "Cold-bind a complete layer-keyed streaming Direct-HCCP table. "
           "layer_ids, outputs, and controls are equal-length non-empty sequences; "
           "each control is a graph-stable NPU torch.int64[16] "
           "KvafStreamingBatchGetControl. queue_leases is the graph-visible "
           "NPU torch.int32[queue_count] table written by LI atomics and used "
           "verbatim by the control; any LI producer may lease a queue. "
           "No layer can be hot-rebound: release the entire table only after all "
           "terminal readers complete. ordinary and native-lane BatchGet remain "
           "unavailable while the table is live.")
      .def("configure_streaming_batch_get_queue_templates",
           &NpuRuntime::ConfigureStreamingBatchGetQueueTemplates,
           py::arg("layer_ids"), py::arg("queue_ids"),
           py::arg("address_tables"), py::arg("valid_tables"),
           py::arg("indices"), py::arg("offsets"), py::arg("masks"),
           py::arg("output_indices"),
           py::arg("producer_stream") = py::none(),
           "Cold-publish one graph-stable descriptor template for every "
           "(layer_id, queue_id) in the already-bound streaming table. Each "
           "argument is a same-length sequence. address_tables are NPU int64 "
           "vectors; valid_tables and masks are NPU uint8 vectors; indices, "
           "offsets, and output_indices are NPU int32 vectors. A layer must "
           "supply every queue exactly once. Address/valid tables may be "
           "identically shared; all mutable descriptor vectors must be "
           "non-overlapping. This can run once before graph capture only; LI "
           "later copies the template and overwrites candidate data/count "
           "without a host hot-submit.")
      .def("release_streaming_batch_get_control",
           &NpuRuntime::ReleaseStreamingBatchGetControl,
           "Release every streaming layer registration and its shared queue-pool "
           "lease table after all terminal graph readers have completed. This "
           "method does not synchronize a stream.")
      .def("streaming_batch_get_control_diagnostic",
           &NpuRuntime::StreamingBatchGetControlDiagnostic,
           "Return a compact cold-path diagnostic with queue-pool, descriptor "
           "scratch, and per-layer private-output ABI addresses/counts.")
      .def("streaming_pool_snapshot", &NpuRuntime::StreamingBatchGetPoolSnapshot,
           "Return a structured read-only streaming-pool snapshot: control "
           "magic/version, queue count, shared lease/scratch addresses, per-layer "
           "private arenas/templates, and host lifecycle counters.")
      .def("bind_layer_resident_batch_get_control",
           &NpuRuntime::BindLayerResidentBatchGetControl,
           py::arg("layer_id"), py::arg("output"), py::arg("control"),
           py::arg("resident_row_base"), py::arg("resident_row_count"),
           py::arg("queue_leases"), py::arg("queue_count"),
           py::arg("producer_stream") = py::none(),
           "Cold-bind one layer's already-reserved canonical resident output "
           "interval and graph-stable NPU torch.int64[16] "
           "KvafLayerResidentBatchGetControl. queue_leases is an NPU "
           "torch.int32[queue_count] temporary Direct-HCCP queue pool. The "
           "device operator writes only resident_row_base .. "
           "resident_row_base + resident_row_count; rows are never privately "
           "released or rebound by this API.")
      .def("configure_layer_resident_batch_get_queue_templates",
           &NpuRuntime::ConfigureLayerResidentBatchGetQueueTemplates,
           py::arg("queue_ids"), py::arg("address_tables"),
           py::arg("valid_tables"), py::arg("indices"), py::arg("offsets"),
           py::arg("masks"), py::arg("output_indices"),
           py::arg("producer_stream") = py::none(),
           "Cold-publish one descriptor template for every layer-resident "
           "Direct-HCCP queue. queue_ids must cover the bound queue pool "
           "exactly once. The device operator copies a template into scratch, "
           "fills candidate data and canonical output indices, publishes the "
           "descriptor, then Posts without a host submit.")
      .def("release_layer_resident_batch_get_control",
           &NpuRuntime::ReleaseLayerResidentBatchGetControl,
           "Release only the cold Direct-HCCP registration and descriptor "
           "resources after the enclosing layer operator has drained all CQ "
           "terminals. This never clears or releases canonical resident rows.")
      .def("layer_resident_batch_get_control_diagnostic",
           &NpuRuntime::LayerResidentBatchGetControlDiagnostic,
           "Return a compact cold-path diagnostic for the layer-resident "
           "control, canonical resident range, queue pool, and descriptor ABI.")
      .def("batch_get", &NpuRuntime::BatchGet,
           py::arg("device_access_table"), py::arg("indices"), py::arg("offsets"),
           py::arg("mask"),
           py::arg("output_indices"), py::arg("output"),
           py::arg("stream") = py::none(), py::arg("synchronize") = false,
           "Submit BatchGet from NPU Tensors. stream=None uses torch.npu.current_stream(); "
           "the default returns after enqueueing work on that stream.")
      .def("batch_get", &NpuRuntime::BatchGetFromTensors,
           py::arg("addresses"), py::arg("valid"), py::arg("indices"),
           py::arg("offsets"), py::arg("mask"), py::arg("output_indices"),
           py::arg("output"), py::arg("stream") = py::none(),
           py::arg("synchronize") = false,
           "Submit BatchGet from caller-owned NPU addresses and valid Tensors. "
           "stream=None uses torch.npu.current_stream(); the default returns "
           "after enqueueing work on that stream.");
}

}  // namespace kvaf
