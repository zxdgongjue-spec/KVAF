#pragma once
#include <pybind11/pybind11.h>
#include "execution_config.h"
namespace kvaf::python_binding {
inline Transport ParseTransport(const std::string& s) {
  if (s == "urma") return Transport::Urma;
  if (s == "ubmem") return Transport::Ubmem;
  throw pybind11::value_error("transport must be urma or ubmem");
}
inline Executor ParseExecutor(const std::string& s) {
  if (s == "aiv") return Executor::Aiv;
  if (s == "aicpu") return Executor::Aicpu;
  throw pybind11::value_error("executor must be aiv or aicpu");
}
inline GetAlgorithm ParseAlgorithm(const std::string& s) {
  if (s == "direct") return GetAlgorithm::Direct;
  if (s == "gather") return GetAlgorithm::Gather;
  if (s == "scatter") return GetAlgorithm::Scatter;
  throw pybind11::value_error("algorithm must be direct, gather or scatter");
}
inline void BindExecutionConfigs(pybind11::module_& m) {
  namespace py = pybind11;
  py::class_<ExecutionResources>(m, "ExecutionResources")
      .def(py::init([](std::uint32_t cores, std::uint32_t jettys, std::uint32_t capacity,
                       std::optional<std::uint32_t> inflight, std::string kernel,
                       std::uint32_t cpu_threads) {
             ExecutionResources r;
             r.cores = cores;
             r.jettys = jettys;
             r.request_capacity = capacity;
             r.max_inflight_total = inflight;
             r.kernel_path = std::move(kernel);
             r.cpu_threads = cpu_threads;
             return r;
           }),
           py::kw_only(), py::arg("cores") = 0, py::arg("jettys") = 0,
           py::arg("request_capacity") = 65536, py::arg("max_inflight_total") = py::none(),
           py::arg("kernel_path") = "", py::arg("cpu_threads") = 0)
      .def_readonly("cpu_threads", &ExecutionResources::cpu_threads)
      .def_readonly("cores", &ExecutionResources::cores)
      .def_readonly("jettys", &ExecutionResources::jettys)
      .def_readonly("request_capacity", &ExecutionResources::request_capacity)
      .def_readonly("max_inflight_total", &ExecutionResources::max_inflight_total)
      .def_readonly("kernel_path", &ExecutionResources::kernel_path);
  py::class_<PutExecutionConfig>(m, "PutExecutionConfig")
      .def(py::init([](const std::string& t, const std::string& e) {
        return PutExecutionConfig{ParseTransport(t), ParseExecutor(e)};
      }), py::arg("transport"), py::arg("executor"))
      .def_property_readonly("transport", [](PutExecutionConfig c) { return c.transport == Transport::Urma ? "urma" : "ubmem"; })
      .def_property_readonly("executor", [](PutExecutionConfig c) { return c.executor == Executor::Aiv ? "aiv" : "aicpu"; });
  py::class_<GetExecutionConfig>(m, "GetExecutionConfig")
      .def(py::init([](const std::string& t, const std::string& e, const std::string& a) {
        return GetExecutionConfig{ParseTransport(t), ParseExecutor(e), ParseAlgorithm(a)};
      }), py::arg("transport"), py::arg("executor"), py::arg("algorithm"))
      .def_property_readonly("transport", [](GetExecutionConfig c) { return c.transport == Transport::Urma ? "urma" : "ubmem"; })
      .def_property_readonly("executor", [](GetExecutionConfig c) { return c.executor == Executor::Aiv ? "aiv" : "aicpu"; })
      .def_property_readonly("algorithm", [](GetExecutionConfig c) {
        return c.algorithm == GetAlgorithm::Direct ? "direct" : c.algorithm == GetAlgorithm::Gather ? "gather" : "scatter";
      });
}
}  // namespace kvaf::python_binding
