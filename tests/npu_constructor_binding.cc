#include "npu_runtime_constructor.h"

namespace py = pybind11;

// Exercise the production pybind argument registration without CANN or a device.
struct ConstructorProbe {
  py::tuple values;
  bool reuse_device_ra;
  ConstructorProbe(std::string get, std::uint32_t bytes, std::uint32_t tokens,
                   std::uint32_t device, std::uint32_t cores, std::string put,
                   std::uint32_t jettys, std::vector<std::string> eids, bool adaptive,
                   std::uint32_t misses, std::string peer, std::vector<std::string> peers,
                   std::uint32_t inflight, const std::string& transport, std::string mode, bool reuse)
      : values(py::make_tuple(get, bytes, tokens, device, cores, put, jettys, eids,
                              adaptive, misses, peer, peers, inflight, transport, mode)), reuse_device_ra(reuse) {}
};

PYBIND11_MODULE(_kvaf_constructor_test, module) {
  auto runtime = py::class_<ConstructorProbe>(module, "NpuRuntime");
  kvaf::python_binding::BindNpuRuntimeConstructor(runtime);
  runtime.def_readonly("values", &ConstructorProbe::values);
  runtime.def_readonly("reuse_device_ra", &ConstructorProbe::reuse_device_ra);
}
