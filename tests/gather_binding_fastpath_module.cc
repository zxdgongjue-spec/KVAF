// Keep the production metadata helpers in this TU. Dead-strip the hardware
// runtime registration so this test needs Python/pybind, not a CANN runtime.
#include "../bindings/src/npu_python_runtime.cc"

namespace {
class MetadataHarness {
 public:
  py::dict Request(const py::object& addresses, const py::object& valid,
                   const py::object& indices, const py::object& offsets,
                   const py::object& count, const py::object& rows,
                   const py::object& output, const py::object& stream) {
    const auto r = kvaf::TrustedGatherRequest({}, addresses, valid, indices,
        offsets, count, rows, output, 656, 128, methods_);
    py::dict result;
    result["pointers"] = py::make_tuple(
        reinterpret_cast<uintptr_t>(r.device_access_table.addresses.data),
        reinterpret_cast<uintptr_t>(r.device_access_table.valid.data),
        reinterpret_cast<uintptr_t>(r.indices.data), reinterpret_cast<uintptr_t>(r.offsets.data),
        reinterpret_cast<uintptr_t>(r.miss_count.data), reinterpret_cast<uintptr_t>(r.output_indices.data),
        reinterpret_cast<uintptr_t>(r.output.data));
    result["bytes"] = py::make_tuple(r.device_access_table.addresses.bytes,
        r.device_access_table.valid.bytes, r.indices.bytes, r.offsets.bytes,
        r.miss_count.bytes, r.output_indices.bytes, r.output.bytes);
    result["stream"] = reinterpret_cast<uintptr_t>(methods_.Stream(stream));
    return result;
  }
 private:
  kvaf::TensorMetadataMethods methods_;
};
}
PYBIND11_MODULE(_kvaf_gather_metadata_test, m) {
  py::class_<MetadataHarness>(m, "Harness")
      .def(py::init<>()).def("request", &MetadataHarness::Request);
  m.def("checked", [](const py::object& tensor) {
    const auto view = kvaf::GetNpuTensorView(tensor, "input");
    return py::make_tuple(reinterpret_cast<uintptr_t>(view.data), view.bytes,
                         view.dimensions, view.first_dimension);
  });
}
