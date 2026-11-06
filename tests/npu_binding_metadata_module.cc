#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <chrono>
#include <cstdint>

#include "acl_python_utils.h"
#include "npu_tensor_python.h"

namespace py = pybind11;
namespace binding = kvaf::python_binding;

namespace {
py::dict Describe(const binding::NpuTensorView& view) {
  py::dict result;
  result["pointer"] = reinterpret_cast<std::uintptr_t>(view.data);
  result["bytes"] = view.bytes;
  result["element_bytes"] = view.element_bytes;
  result["shape"] = view.shape;
  result["dimensions"] = view.dimensions;
  result["device"] = view.device;
  return result;
}
}  // namespace

// Test-only module: the production metadata helpers run without ACL or DMA.
PYBIND11_MODULE(_kvaf_binding_metadata_test, module) {
  module.def("view", [](const py::object& tensor, bool keep_shape) {
    return Describe(binding::GetNpuTensorView(tensor, "tensor", keep_shape));
  }, py::arg("tensor"), py::arg("keep_shape") = true);
  module.def("int64_view", [](const py::object& tensor, std::uint32_t device) {
    return Describe(binding::GetNpuInt64TensorView(tensor, "metadata", device));
  }, py::arg("tensor"), py::arg("device") = 0);
  module.def("stream", [](const py::object& stream) {
    return reinterpret_cast<std::uintptr_t>(kvaf::acl_python::StreamFromPython(stream));
  });
  module.def("profile_arrays", [](const py::object& src, const py::object& dst,
                                  const py::object& lengths, std::uint32_t device,
                                  unsigned first) {
    if (first > 2) throw py::value_error("first must be 0, 1 or 2");
    const std::array<py::object, 3> tensors{src, dst, lengths};
    const std::array<const char*, 3> names{"src", "dst", "lengths"};
    std::array<double, 3> elapsed{};
    for (unsigned i = 0; i < 3; ++i) {
      const auto index = (first + i) % 3;
      const auto begin = std::chrono::steady_clock::now();
      {
        const auto view = binding::GetNpuInt64TensorView(tensors[index], names[index], device);
      }
      elapsed[index] = std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - begin).count();
    }
    return elapsed;
  }, py::arg("src"), py::arg("dst"), py::arg("lengths"),
     py::arg("device") = 0, py::arg("first") = 0);
}
