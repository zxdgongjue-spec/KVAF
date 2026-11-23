#include "acl_graph_python.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "acl/acl.h"
#include "acl_python_utils.h"

namespace py = pybind11;

namespace kvaf {
namespace {

void ThrowIfAclError(aclError error, const char* operation) {
  if (error == ACL_SUCCESS) {
    return;
  }
  throw std::runtime_error(std::string(operation) +
                           " failed with ACL error " +
                           std::to_string(static_cast<int>(error)));
}

std::size_t BufferBytes(const py::buffer_info& info) {
  if (info.ptr == nullptr || info.size <= 0 || info.itemsize <= 0) {
    throw py::value_error("ACL async copy source must be a non-empty buffer");
  }
  if (static_cast<std::uint64_t>(info.size) >
      std::numeric_limits<std::size_t>::max() /
          static_cast<std::size_t>(info.itemsize)) {
    throw py::value_error("ACL async copy source is too large");
  }
  return static_cast<std::size_t>(info.size) *
         static_cast<std::size_t>(info.itemsize);
}

bool IsCContiguous(const py::buffer_info& info) {
  std::size_t expected_stride = static_cast<std::size_t>(info.itemsize);
  for (py::ssize_t dimension = info.ndim - 1; dimension >= 0; --dimension) {
    if (info.shape[dimension] > 1 &&
        info.strides[dimension] !=
            static_cast<py::ssize_t>(expected_stride)) {
      return false;
    }
    if (info.shape[dimension] > 0 &&
        expected_stride > std::numeric_limits<std::size_t>::max() /
                              static_cast<std::size_t>(info.shape[dimension])) {
      return false;
    }
    expected_stride *= static_cast<std::size_t>(info.shape[dimension]);
  }
  return true;
}

class AclGraph final {
 public:
  AclGraph() = default;
  ~AclGraph() { ResetNoThrow(); }

  AclGraph(const AclGraph&) = delete;
  AclGraph& operator=(const AclGraph&) = delete;

  void CaptureBegin(const py::object& stream) {
    Use use(in_use_);
    if (capturing_) {
      throw py::value_error("ACLGraph capture is already active");
    }
    if (graph_ != nullptr) {
      throw py::value_error(
          "ACLGraph already owns a captured graph; call reset before recapturing");
    }
    stream_ = acl_python::StreamFromPython(stream);
    ThrowIfAclError(
        aclmdlRICaptureBegin(stream_, ACL_MODEL_RI_CAPTURE_MODE_RELAXED),
        "aclmdlRICaptureBegin");
    capturing_ = true;
  }

  void CaptureEnd() {
    Use use(in_use_);
    if (!capturing_) {
      throw py::value_error("ACLGraph capture is not active");
    }
    aclmdlRI captured_graph = nullptr;
    const aclError error = aclmdlRICaptureEnd(stream_, &captured_graph);
    capturing_ = false;
    if (error != ACL_SUCCESS) {
      stream_ = nullptr;
      ThrowIfAclError(error, "aclmdlRICaptureEnd");
    }
    if (captured_graph == nullptr) {
      stream_ = nullptr;
      throw std::runtime_error("aclmdlRICaptureEnd returned a null graph");
    }
    graph_ = captured_graph;
  }

  void Replay(const py::object& stream) const {
    Use use(in_use_);
    const aclrtStream replay_stream = ReplayStream(stream);
    ThrowIfAclError(aclmdlRIExecuteAsync(graph_, replay_stream),
                    "aclmdlRIExecuteAsync");
  }

  py::dict ReplayTimed(const py::object& stream, bool profile) {
    Use use(in_use_);
    const auto stamp = [profile]() -> std::uint64_t {
      return profile ? std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count() : 0;
    };
    const auto begin = stamp();
    const aclrtStream replay_stream = ReplayStream(stream);
    const auto resolved = stamp();
    if (start_event_ == nullptr) {
      ThrowIfAclError(aclrtCreateEventWithFlag(&start_event_, ACL_EVENT_TIME_LINE),
                      "aclrtCreateEventWithFlag start");
    }
    if (end_event_ == nullptr) {
      ThrowIfAclError(aclrtCreateEventWithFlag(&end_event_, ACL_EVENT_TIME_LINE),
                      "aclrtCreateEventWithFlag end");
    }
    const auto prepared = stamp();
    ThrowIfAclError(aclrtRecordEvent(start_event_, replay_stream), "aclrtRecordEvent start");
    const auto recorded = stamp();
    ThrowIfAclError(aclmdlRIExecuteAsync(graph_, replay_stream), "aclmdlRIExecuteAsync");
    const auto replayed = stamp();
    ThrowIfAclError(aclrtRecordEvent(end_event_, replay_stream), "aclrtRecordEvent end");
    const auto end_recorded = stamp();
    std::uint64_t synced = 0, measured = 0;
    float milliseconds = 0;
    {
      py::gil_scoped_release release;
      ThrowIfAclError(aclrtSynchronizeEvent(end_event_), "aclrtSynchronizeEvent");
      synced = stamp();
      ThrowIfAclError(aclrtEventElapsedTime(&milliseconds, start_event_, end_event_),
                      "aclrtEventElapsedTime");
      measured = stamp();
    }
    const auto returned = stamp();
    py::dict host;
    if (profile) {
      host["stream_resolve"] = resolved - begin;
      host["timing_prepare"] = prepared - resolved;
      host["start_event_record"] = recorded - prepared;
      host["replay_submit"] = replayed - recorded;
      host["end_event_record"] = end_recorded - replayed;
      host["event_sync"] = synced - end_recorded;
      host["elapsed_time"] = measured - synced;
      host["gil_reacquire"] = returned - measured;
      host["native_total"] = returned - begin;
    }
    py::dict result;
    result["elapsed_ms"] = milliseconds;
    result["host_ns"] = host;
    return result;
  }

  void Synchronize(const py::object& stream) const {
    Use use(in_use_);
    const aclrtStream target_stream = acl_python::StreamFromPython(stream);
    if (capturing_ || graph_ == nullptr || target_stream != stream_) {
      throw py::value_error(
          "ACLGraph synchronize requires a completed graph and its capture stream");
    }
    ThrowIfAclError(aclrtSynchronizeStream(target_stream),
                    "aclrtSynchronizeStream");
  }

  void CopyHostToDeviceAsync(std::uint64_t destination,
                             std::size_t destination_bytes,
                             const py::buffer& source,
                             const py::object& stream) const {
    Use use(in_use_);
    if (capturing_ || graph_ == nullptr) {
      throw py::value_error(
          "ACLGraph async input copy requires a completed graph capture");
    }
    if (destination == 0 || destination_bytes == 0) {
      throw py::value_error("ACL async copy destination must be non-empty");
    }
    const aclrtStream copy_stream = acl_python::StreamFromPython(stream);
    if (copy_stream != stream_) {
      throw py::value_error(
          "ACLGraph async input copy must use the capture stream");
    }

    const py::buffer_info source_info = source.request();
    if (!IsCContiguous(source_info)) {
      throw py::value_error("ACL async copy source must be C-contiguous");
    }
    const std::size_t source_bytes = BufferBytes(source_info);
    if (source_bytes != destination_bytes) {
      throw py::value_error(
          "ACL async copy source and destination byte sizes must match");
    }
    ThrowIfAclError(
        aclrtMemcpyAsync(
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(destination)),
            destination_bytes, source_info.ptr, source_bytes,
            ACL_MEMCPY_HOST_TO_DEVICE, copy_stream),
        "aclrtMemcpyAsync host-to-device");
  }

  void Reset() {
    Use use(in_use_);
    if (capturing_) {
      throw py::value_error(
          "ACLGraph capture is active; call capture_end before reset");
    }
    ThrowIfAclError(DestroyTimingEvents(), "aclrtDestroyEvent");
    if (graph_ != nullptr) {
      const aclError error = aclmdlRIDestroy(graph_);
      graph_ = nullptr;
      stream_ = nullptr;
      ThrowIfAclError(error, "aclmdlRIDestroy");
    }
  }

  bool Capturing() const { return capturing_; }
  bool Ready() const { return graph_ != nullptr; }

 private:
  // A timed replay releases the GIL while waiting; reject concurrent reset/reuse.
  struct Use {
    explicit Use(bool& active) : active_(active) {
      if (active_) throw py::value_error("ACLGraph is already in use");
      active_ = true;
    }
    ~Use() { active_ = false; }
    bool& active_;
  };

  aclrtStream ReplayStream(const py::object& stream) const {
    if (capturing_ || graph_ == nullptr) {
      throw py::value_error("ACLGraph has no completed capture to replay");
    }
    const aclrtStream target = acl_python::StreamFromPython(stream);
    if (target != stream_) {
      throw py::value_error("ACLGraph replay must use the same ACL stream as capture");
    }
    return target;
  }

  aclError DestroyTimingEvents() {
    const auto start = start_event_ ? aclrtDestroyEvent(start_event_) : ACL_SUCCESS;
    const auto end = end_event_ ? aclrtDestroyEvent(end_event_) : ACL_SUCCESS;
    start_event_ = end_event_ = nullptr;
    return start != ACL_SUCCESS ? start : end;
  }

  void ResetNoThrow() {
    (void)DestroyTimingEvents();
    if (capturing_) {
      aclmdlRI ignored_graph = nullptr;
      (void)aclmdlRICaptureEnd(stream_, &ignored_graph);
      if (ignored_graph != nullptr) {
        (void)aclmdlRIDestroy(ignored_graph);
      }
    }
    if (graph_ != nullptr) {
      (void)aclmdlRIDestroy(graph_);
    }
    graph_ = nullptr;
    stream_ = nullptr;
    capturing_ = false;
  }

  aclmdlRI graph_{nullptr};
  aclrtStream stream_{nullptr};
  aclrtEvent start_event_{nullptr}, end_event_{nullptr};
  mutable bool in_use_{false};
  bool capturing_{false};
};

}  // namespace

void BindAclGraph(py::module_& module) {
  py::class_<AclGraph>(module, "AclGraph")
      .def(py::init<>())
      .def("capture_begin", &AclGraph::CaptureBegin, py::arg("stream"),
           "Begin raw ACL Runtime Graph capture on a caller-owned ACL stream.")
      .def("capture_end", &AclGraph::CaptureEnd,
           "Finish capture and retain the resulting ACL graph.")
      .def("replay", &AclGraph::Replay, py::arg("stream"),
           "Submit the captured ACL graph to its capture stream.")
      .def("replay_timed", &AclGraph::ReplayTimed, py::arg("stream"), py::arg("profile") = false,
           "Record/replay/record/wait in one native call; return elapsed_ms and optional Host ns. "
           "The first call creates reusable timing events. Use the capture device and stream.")
      .def("synchronize", &AclGraph::Synchronize, py::arg("stream"),
           "Wait for graph work on its capture stream.")
      .def("copy_host_to_device_async", &AclGraph::CopyHostToDeviceAsync,
           py::arg("destination"), py::arg("destination_bytes"),
           py::arg("source"), py::arg("stream"),
           "Queue a raw ACL H2D copy on the capture stream. The caller owns "
           "source and must keep it alive until synchronize() returns.")
      .def("reset", &AclGraph::Reset,
           "Destroy the captured graph before releasing caller buffers or Runtime.")
      .def_property_readonly("capturing", &AclGraph::Capturing)
      .def_property_readonly("ready", &AclGraph::Ready);
}

}  // namespace kvaf
