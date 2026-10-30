#include "remote_offload_python.h"
#include "acl_python_utils.h"
#include <limits>

namespace kvaf {
namespace py = pybind11;
namespace {
void Check(const Status& s) { if (!s.ok()) throw std::runtime_error(s.message()); }
class PythonSource {
 public:
  PythonSource(std::uint32_t device, std::string eid) : source_(device, std::move(eid)) {}
  ~PythonSource() {
    if (!source_.Shutdown().ok()) (void)tensor_.release(); // Retain HBM with quarantined DMA/registration.
  }
  void Register(const py::dict& ticket, py::object tensor, py::object stream) {
    if (!tensor_.is_none()) throw py::value_error("shutdown the previous source before registering again");
    if (!py::hasattr(tensor, "data_ptr") || !py::hasattr(tensor, "device") ||
        py::str(tensor.attr("device").attr("type")).cast<std::string>() != "npu" ||
        tensor.attr("device").attr("index").cast<std::uint32_t>() != source_.device_id() ||
        !tensor.attr("is_contiguous")().cast<bool>())
      throw py::value_error("source must be contiguous on this HbmSource's NPU");
    const auto count = tensor.attr("numel")().cast<std::uint64_t>();
    const auto element = tensor.attr("element_size")().cast<std::uint64_t>();
    if (!count || !element || count > SIZE_MAX/element) throw py::value_error("invalid HBM source size");
    const auto t = remote_python::Ticket(ticket);
    const auto producer = Stream(stream);
    tensor_ = tensor; // Covers registration/probe failures and lost control replies.
    Check(source_.Register(t, {reinterpret_cast<void*>(tensor.attr("data_ptr")().cast<std::uintptr_t>()),
                              static_cast<std::size_t>(count*element), DeviceKind::kNpu}, producer));
  }
  void* Stream(py::object stream) const {
    if (stream.is_none()) stream = py::module_::import("torch").attr("npu").attr("current_stream")(source_.device_id());
    if (py::hasattr(stream, "device") && stream.attr("device").attr("index").cast<std::uint32_t>() != source_.device_id())
      throw py::value_error("producer stream belongs to another NPU");
    return acl_python::StreamFromPython(stream);
  }
  py::dict Ready(const std::vector<std::uint64_t>& offsets, const std::vector<std::uint64_t>& lengths, py::object stream) {
    HbmReadyBatch batch; Check(source_.Ready(offsets, lengths, Stream(stream), &batch));
    return remote_python::Pack(batch);
  }
  void Shutdown() { Check(source_.Shutdown()); tensor_ = py::none(); }
  HbmSource source_;
  py::object tensor_{py::none()};
};
}
void BindHbmSource(py::module_& module) {
  py::class_<PythonSource>(module, "HbmSource")
    .def(py::init<std::uint32_t, std::string>(), py::arg("device_id"), py::arg("npu_eid") = "")
    .def("register", &PythonSource::Register, py::arg("region_ticket"), py::arg("source"), py::arg("stream") = py::none())
    .def("ready", &PythonSource::Ready, py::arg("input_offsets"), py::arg("lengths"), py::arg("stream") = py::none())
    .def("release_batch", [](PythonSource& s) { Check(s.source_.ReleaseBatch()); })
    .def("cancel_batch", [](PythonSource& s) { Check(s.source_.CancelBatch()); })
    .def("shutdown", &PythonSource::Shutdown)
    .def_property_readonly("frozen", [](PythonSource& s) { return s.source_.frozen(); })
    .def_property_readonly("registered", [](PythonSource& s) { return s.source_.registered(); })
    .def_property_readonly("selected_npu_eid", [](PythonSource& s) { return s.source_.selected_npu_eid(); });
}
}
