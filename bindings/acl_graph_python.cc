#include "acl_graph_python.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "acl/acl.h"

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

aclrtStream StreamFromPython(const py::object& stream) {
  if (!py::hasattr(stream, "npu_stream")) {
    throw py::type_error("ACLGraph stream must be a torch.npu.Stream");
  }
  const std::uint64_t pointer =
      stream.attr("npu_stream").cast<std::uint64_t>();
  if (pointer == 0) {
    throw py::value_error("ACLGraph stream must not be null");
  }
  return reinterpret_cast<aclrtStream>(
      static_cast<std::uintptr_t>(pointer));
}

class AclGraph final {
 public:
  AclGraph() = default;
  ~AclGraph() { ResetNoThrow(); }

  AclGraph(const AclGraph&) = delete;
  AclGraph& operator=(const AclGraph&) = delete;

  void CaptureBegin(const py::object& stream) {
    if (capturing_) {
      throw py::value_error("ACLGraph capture is already active");
    }
    if (graph_ != nullptr) {
      throw py::value_error(
          "ACLGraph already owns a captured graph; call reset before recapturing");
    }
    stream_ = StreamFromPython(stream);
    ThrowIfAclError(
        aclmdlRICaptureBegin(stream_, ACL_MODEL_RI_CAPTURE_MODE_RELAXED),
        "aclmdlRICaptureBegin");
    capturing_ = true;
  }

  void CaptureEnd() {
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
    if (capturing_ || graph_ == nullptr) {
      throw py::value_error("ACLGraph has no completed capture to replay");
    }
    const aclrtStream replay_stream = StreamFromPython(stream);
    if (replay_stream != stream_) {
      throw py::value_error(
          "ACLGraph replay must use the same torch.npu.Stream as capture");
    }
    ThrowIfAclError(aclmdlRIExecuteAsync(graph_, replay_stream),
                    "aclmdlRIExecuteAsync");
  }

  void Synchronize(const py::object& stream) const {
    const aclrtStream target_stream = StreamFromPython(stream);
    if (capturing_ || graph_ == nullptr || target_stream != stream_) {
      throw py::value_error(
          "ACLGraph synchronize requires a completed graph and its capture stream");
    }
    ThrowIfAclError(aclrtSynchronizeStream(target_stream),
                    "aclrtSynchronizeStream");
  }

  void Reset() {
    if (capturing_) {
      throw py::value_error(
          "ACLGraph capture is active; call capture_end before reset");
    }
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
  void ResetNoThrow() {
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
  bool capturing_{false};
};

}  // namespace

void BindAclGraph(py::module_& module) {
  py::class_<AclGraph>(module, "AclGraph")
      .def(py::init<>())
      .def("capture_begin", &AclGraph::CaptureBegin, py::arg("stream"),
           "Begin raw ACL Runtime Graph capture on a caller-owned torch.npu.Stream.")
      .def("capture_end", &AclGraph::CaptureEnd,
           "Finish capture and retain the resulting ACL graph.")
      .def("replay", &AclGraph::Replay, py::arg("stream"),
           "Submit the captured ACL graph to its capture stream.")
      .def("synchronize", &AclGraph::Synchronize, py::arg("stream"),
           "Wait for graph work on its capture stream.")
      .def("reset", &AclGraph::Reset,
           "Destroy the captured graph before releasing caller buffers or Runtime.")
      .def_property_readonly("capturing", &AclGraph::Capturing)
      .def_property_readonly("ready", &AclGraph::Ready);
}

}  // namespace kvaf
