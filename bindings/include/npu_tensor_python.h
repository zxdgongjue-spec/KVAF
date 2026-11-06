#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace kvaf::python_binding {

struct NpuTensorView {
  void* data{nullptr};
  std::size_t bytes{0};
  std::size_t element_bytes{0};
  std::vector<pybind11::ssize_t> shape;
  std::size_t dimensions{0};
  pybind11::object device;
};

inline NpuTensorView GetNpuTensorView(const pybind11::object& tensor,
                                     const char* name, bool keep_shape = true) {
  namespace py = pybind11;
  // Fetch each attribute once, rather than looking it up for both hasattr and
  // attr. Keep duck-typed Tensor layouts usable without importing PyTorch.
  const auto data_ptr = py::getattr(tensor, "data_ptr", py::object{});
  const auto numel = py::getattr(tensor, "numel", py::object{});
  const auto element_size = py::getattr(tensor, "element_size", py::object{});
  const auto is_contiguous = py::getattr(tensor, "is_contiguous", py::object{});
  auto device = py::getattr(tensor, "device", py::object{});
  const auto shape = py::getattr(tensor, "shape", py::object{});
  if (!data_ptr || !numel || !element_size || !is_contiguous || !device || !shape) {
    throw py::type_error(std::string(name) + " must be a contiguous NPU Tensor");
  }
  if (!is_contiguous().cast<bool>()) {
    throw py::value_error(std::string(name) + " must be contiguous");
  }
  if (py::str(device).cast<std::string>().rfind("npu", 0) != 0) {
    throw py::value_error(std::string(name) + " must reside on an NPU");
  }
  const auto pointer = data_ptr().cast<std::uint64_t>();
  const auto elements = numel().cast<std::uint64_t>();
  const auto element_bytes = element_size().cast<std::uint64_t>();
  if (pointer == 0 || elements == 0 || element_bytes == 0 ||
      elements > std::numeric_limits<std::size_t>::max() ||
      element_bytes > std::numeric_limits<std::size_t>::max() ||
      static_cast<std::size_t>(elements) >
          std::numeric_limits<std::size_t>::max() /
              static_cast<std::size_t>(element_bytes)) {
    throw py::value_error(std::string(name) + " must be a non-empty NPU Tensor");
  }
  NpuTensorView view;
  view.data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer));
  view.bytes = static_cast<std::size_t>(elements) * static_cast<std::size_t>(element_bytes);
  view.element_bytes = static_cast<std::size_t>(element_bytes);
  view.device = std::move(device);
  if (!keep_shape && PyTuple_Check(shape.ptr())) {
    // torch.Size is a tuple. Raw READ needs rank only; validate its dimensions
    // without allocating a C++ vector. Other callers still receive full shape.
    for (const auto& dimension : shape) {
      (void)py::cast<py::ssize_t>(dimension);
      ++view.dimensions;
    }
  } else {
    view.shape = shape.cast<std::vector<py::ssize_t>>();
    view.dimensions = view.shape.size();
  }
  return view;
}

inline NpuTensorView GetNpuInt64TensorView(const pybind11::object& tensor,
                                          const char* name, std::uint32_t device_id) {
  namespace py = pybind11;
  auto view = GetNpuTensorView(tensor, name, false);
  const auto dtype = py::getattr(tensor, "dtype", py::object{});
  if (view.dimensions != 1 || view.element_bytes != sizeof(std::int64_t) ||
      !dtype || py::str(dtype).cast<std::string>() != "torch.int64" ||
      reinterpret_cast<std::uintptr_t>(view.data) % alignof(std::int64_t) ||
      view.device.attr("index").cast<std::uint32_t>() != device_id) {
    throw py::value_error(std::string(name) +
                         " must be a one-dimensional torch.int64 Tensor on the Runtime NPU");
  }
  return view;
}

}  // namespace kvaf::python_binding
