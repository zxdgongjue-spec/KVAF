#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace kvaf::python_binding {

// Keep main's positional constructor contract through transport. New backend
// options are keyword-only so they cannot shift existing positional arguments.
template <class Runtime>
void BindNpuRuntimeConstructor(pybind11::class_<Runtime>& runtime) {
  namespace py = pybind11;
  runtime.def(py::init<std::string, std::uint32_t, std::uint32_t, std::uint32_t,
                       std::uint32_t, std::string, std::uint32_t,
                       std::vector<std::string>, bool, std::uint32_t,
                       std::string, std::vector<std::string>, std::uint32_t,
                       const std::string&, std::string, bool, const std::string&, int>(),
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
              py::arg("reuse_device_ra") = false,
              py::arg("ubmem_page_mode") = "thp",
              py::arg("ubmem_numa_node") = -1);
}

}  // namespace kvaf::python_binding
