#include "npu_runtime_constructor.h"

namespace py = pybind11;

// Exercise the production pybind registration without CANN or a device.
struct ConstructorProbe {
  py::tuple values;
  ConstructorProbe(py::object bytes, py::object tokens, std::uint32_t device,
                   const std::string& transport, py::object default_put, py::object default_get,
                   py::object put_resources, py::object get_resources,
                   std::vector<std::string> eids, std::string peer, py::object cpu_pull, py::object npu_pull)
      : values(py::make_tuple(bytes, tokens, device, transport, default_put, default_get,
                              put_resources, get_resources, eids, peer, cpu_pull, npu_pull)) {}
};

PYBIND11_MODULE(_kvaf_constructor_test, module) {
  auto runtime = py::class_<ConstructorProbe>(module, "NpuRuntime");
  kvaf::python_binding::BindNpuRuntimeConstructor(runtime);
  runtime.def_readonly("values", &ConstructorProbe::values);
}
