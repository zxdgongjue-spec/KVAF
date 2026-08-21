#pragma once

#include <cstddef>
#include <cstdint>

#include <pybind11/pybind11.h>

#include "requests.h"

namespace kvaf::python_binding {

// Export the Runtime-owned contiguous Host arrays without materializing Python
// integers. The caller must retain the associated AccessTableHandle until it
// has finished consuming these views.
inline pybind11::dict MakeAccessTableView(const AccessTableView& view) {
  namespace py = pybind11;
  const py::ssize_t size = static_cast<py::ssize_t>(view.size);

  py::dict result;
  result["addresses"] = py::memoryview::from_buffer(
      const_cast<Address*>(view.addresses), static_cast<py::ssize_t>(sizeof(Address)),
      "Q", {size}, {static_cast<py::ssize_t>(sizeof(Address))}, true);
  result["lengths"] = py::memoryview::from_buffer(
      const_cast<std::uint64_t*>(view.lengths),
      static_cast<py::ssize_t>(sizeof(std::uint64_t)), "Q", {size},
      {static_cast<py::ssize_t>(sizeof(std::uint64_t))}, true);
  result["valid"] = py::memoryview::from_buffer(
      const_cast<std::uint8_t*>(view.valid),
      static_cast<py::ssize_t>(sizeof(std::uint8_t)), "B", {size},
      {static_cast<py::ssize_t>(sizeof(std::uint8_t))}, true);
  result["size"] = view.size;
  return result;
}

}  // namespace kvaf::python_binding
