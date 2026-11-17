#include "npu_runtime_constructor.h"

namespace py = pybind11;

// Exercise the production pybind argument registration without CANN or a device.
struct ConstructorProbe {
  py::tuple values;
  ConstructorProbe(std::uint32_t bytes, std::uint32_t tokens, std::uint32_t device,
                   const std::string& mode, py::object put, py::object get,
                   py::object put_resources, py::object get_resources,
                   std::vector<std::string> eids, std::string peer)
      : values(py::make_tuple(bytes, tokens, device, mode, put, get, put_resources,
                              get_resources, eids, peer)) {}
};

PYBIND11_MODULE(_kvaf_constructor_test, module) {
  auto runtime = py::class_<ConstructorProbe>(module, "NpuRuntime");
  kvaf::python_binding::BindNpuRuntimeConstructor(runtime);
  runtime.def_readonly("values", &ConstructorProbe::values);
}
