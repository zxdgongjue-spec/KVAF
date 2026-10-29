#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include "offload_runtime.h"

namespace py = pybind11;
namespace kvaf {
namespace {
void Check(const Status& s) {
  if (!s.ok()) throw std::runtime_error(s.message());
}
std::vector<std::uint64_t> Integers(py::object value) {
  if (py::hasattr(value, "data_ptr")) {
    if (py::str(value.attr("device")).cast<std::string>() != "cpu" ||
        py::str(value.attr("dtype")).cast<std::string>() != "torch.int64" ||
        value.attr("dim")().cast<int>() != 1 || !value.attr("is_contiguous")().cast<bool>())
      throw py::value_error("metadata must be a sequence or contiguous 1-D CPU int64 Tensor");
    value = value.attr("tolist")();
  }
  return value.cast<std::vector<std::uint64_t>>();
}
class PythonOffload {
 public:
  PythonOffload(std::string eid, std::uint32_t device, std::uint32_t candidate, std::uint32_t timeout)
      : runtime_({device, std::move(eid), candidate, timeout}) {}
  ~PythonOffload() {
    if (!runtime_.Shutdown().ok() && !source_.is_none())
      (void)source_.release(); // Leak the strong reference with quarantined DMA, never free its HBM.
  }
  void Offload(py::object keys, py::object source, py::object offsets, py::object lengths, py::object stream) {
    if (runtime_.pending_offload()) throw py::value_error("resolve the previous Offload first");
    if (!py::hasattr(source, "device") || !py::hasattr(source, "data_ptr") ||
        !source.attr("is_contiguous")().cast<bool>() ||
        py::str(source.attr("device").attr("type")).cast<std::string>() != "npu" ||
        source.attr("device").attr("index").cast<std::uint32_t>() != runtime_.device_id())
      throw py::value_error("source must be contiguous on the configured NPU");
    const auto count = source.attr("numel")().cast<std::uint64_t>();
    const auto element = source.attr("element_size")().cast<std::uint64_t>();
    if (!count || !element || count > SIZE_MAX / element) throw py::value_error("invalid source size");
    BatchPutRequest request;
    request.keys = Integers(keys); request.input_offsets = Integers(offsets); request.lengths = Integers(lengths);
    request.input = {reinterpret_cast<const void*>(source.attr("data_ptr")().cast<std::uintptr_t>()),
                     static_cast<std::size_t>(count*element), DeviceKind::kNpu};
    if (stream.is_none()) stream = py::module_::import("torch").attr("npu").attr("current_stream")();
    const auto pointer = stream.attr("npu_stream").cast<std::uintptr_t>();
    if (!pointer) throw py::value_error("producer stream must be non-null");
    if (py::hasattr(stream, "device") &&
        stream.attr("device").attr("index").cast<std::uint32_t>() != runtime_.device_id())
      throw py::value_error("producer stream is on a different NPU");
    source_ = source;
    Status status;
    try { status = runtime_.Offload(request, {reinterpret_cast<void*>(pointer)}); }
    catch (...) { if (!runtime_.pending_offload()) source_ = py::none(); throw; }
    if (!runtime_.pending_offload()) source_ = py::none();
    if (!status.ok()) throw std::runtime_error(status.message() + " (task_id=" + std::to_string(runtime_.task_id()) + ")");
  }
  void Resolve() {
    auto status = runtime_.ResolveOffload();
    if (!runtime_.pending_offload()) source_ = py::none();
    Check(status);
  }
  OffloadRuntime runtime_;
  py::object source_{py::none()};
};
}
void BindOffloadRuntime(py::module_& module) {
  py::class_<PythonOffload>(module, "OffloadRuntime")
    .def(py::init<std::string, std::uint32_t, std::uint32_t, std::uint32_t>(),
         py::arg("npu_eid"), py::arg("device_id") = 0, py::arg("cpu_candidate") = 0, py::arg("timeout_ms") = 30000)
    .def("link_to", [](PythonOffload& r, const std::string& address) { Check(r.runtime_.LinkTo({address})); })
    .def("initialize", [](PythonOffload& r, std::uint64_t bytes, std::uint64_t slot) { Check(r.runtime_.Initialize(bytes, slot)); },
         py::arg("ddr_bytes"), py::arg("slot_size"))
    .def("offload", &PythonOffload::Offload, py::arg("keys"), py::arg("source"), py::arg("input_offsets"),
         py::arg("lengths"), py::arg("stream") = py::none())
    .def("resolve_offload", &PythonOffload::Resolve)
    .def("batch_free", [](PythonOffload& r, py::object keys) { Check(r.runtime_.core().BatchFree({Integers(keys)})); })
    .def("last_checksum", [](PythonOffload& r) { std::uint64_t hash{}; Check(r.runtime_.LastChecksum(&hash)); return hash; })
    .def("shutdown", [](PythonOffload& r) { Check(r.runtime_.Shutdown()); })
    .def_property_readonly("initialized", [](PythonOffload& r) { return r.runtime_.initialized(); })
    .def_property_readonly("pending_offload", [](PythonOffload& r) { return r.runtime_.pending_offload(); })
    .def_property_readonly("task_id", [](PythonOffload& r) { return r.runtime_.task_id(); });
}
}
