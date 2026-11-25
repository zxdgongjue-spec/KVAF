#include "acl_python_utils.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace kvaf::acl_python {

aclrtStream StreamFromPython(const py::object& stream) {
  // The getter may flush Torch's Host queue; evaluate it exactly once.
  const auto handle = py::getattr(stream, "npu_stream", py::object{});
  if (!handle) {
    throw py::type_error("ACL stream must be a torch.npu.Stream");
  }
  const std::uint64_t pointer = handle.cast<std::uint64_t>();
  if (pointer == 0) {
    throw py::value_error("ACL stream must not be null");
  }
  return reinterpret_cast<aclrtStream>(static_cast<std::uintptr_t>(pointer));
}

}  // namespace kvaf::acl_python
