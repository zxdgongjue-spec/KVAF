#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace kvaf::python_binding {

// Layout is positional; runtime policy and provider budgets are keyword-only.
template <class Runtime>
void BindNpuRuntimeConstructor(pybind11::class_<Runtime>& runtime) {
  namespace py = pybind11;
  runtime.def(
      py::init<std::uint32_t, std::uint32_t, std::uint32_t, const std::string&,
               py::object, py::object, py::object, py::object,
               std::vector<std::string>, std::string>(),
      py::arg("batch_get_token_bytes"), py::arg("batch_get_tokens_per_block"), py::kw_only(),
      py::arg("device_id") = 0, py::arg("transport_mode") = "urma",
      py::arg("default_put") = py::none(), py::arg("default_get") = py::none(),
      py::arg("put_resources") = py::none(), py::arg("get_resources") = py::none(),
      py::arg("npu_eids") = std::vector<std::string>{}, py::arg("peer_ub_device") = "");
}

}  // namespace kvaf::python_binding
