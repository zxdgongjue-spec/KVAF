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
      py::init<py::object, py::object, std::uint32_t, const std::string&,
               py::object, py::object, py::object, py::object,
               std::vector<std::string>, std::string, py::object, py::object>(),
      py::arg("batch_get_token_bytes") = py::none(), py::arg("batch_get_tokens_per_block") = py::none(), py::kw_only(),
      py::arg("device_id") = 0, py::arg("transport_mode") = "urma",
      py::arg("default_put") = py::none(), py::arg("default_get") = py::none(),
      py::arg("put_resources") = py::none(), py::arg("get_resources") = py::none(),
      py::arg("npu_eids") = std::vector<std::string>{}, py::arg("peer_ub_device") = "",
      py::arg("cpu_pull_options") = py::none(), py::arg("npu_pull_options") = py::none());
}

template <class Runtime>
void BindLegacyNpuRuntimeConstructor(pybind11::class_<Runtime>& runtime) {
  namespace py = pybind11;
  runtime.def(py::init<std::string, std::uint32_t, std::uint32_t, std::uint32_t,
                       std::uint32_t, std::string, std::uint32_t,
                       std::vector<std::string>, bool, std::uint32_t,
                       std::string, std::vector<std::string>, std::uint32_t,
                       const std::string&, std::string, std::string, std::uint32_t, bool>(),
              py::arg("batch_get_kernel_object_path"),
              py::arg("batch_get_token_bytes"),
              py::arg("batch_get_tokens_per_block"), py::arg("device_id") = 0,
              py::arg("npu_core_count") = 2,
              py::arg("batch_put_kernel_object_path") = "",
              py::arg("jettys_per_core") = 2,
              py::arg("npu_eids") = std::vector<std::string>{},
              py::arg("adaptive_ub_device") = false,
              py::arg("adaptive_ub_miss_count") = 2048,
              py::arg("peer_ub_device") = "",
              py::arg("adaptive_peer_ub_devices") = std::vector<std::string>{},
              py::arg("batch_get_max_inflight_total") = 2048,
              py::arg("transport") = "urma", py::kw_only(),
              py::arg("aicpu_batch_get_mode") = "",
              py::arg("aiv_batch_get_mode") = "direct_read",
              py::arg("aiv_gather_sgl_size") = 6,
              py::arg("trusted_gather_inputs") = true);
}

}  // namespace kvaf::python_binding
