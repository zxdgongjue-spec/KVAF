#include "acl_python_utils.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace kvaf::acl_python {

aclrtStream StreamFromPython(const py::object& stream) {
  py::object handle;
  try {
    // This property fences the torch_npu Host task queue. Read it once.
    handle = py::getattr(stream, "npu_stream");
  } catch (const py::error_already_set& error) {
    if (error.matches(PyExc_AttributeError)) {
      throw py::type_error("ACL stream must be a torch.npu.Stream");
    }
    throw;
  }
  const std::uint64_t pointer = handle.cast<std::uint64_t>();
  if (pointer == 0) {
    throw py::value_error("ACL stream must not be null");
  }
  return reinterpret_cast<aclrtStream>(static_cast<std::uintptr_t>(pointer));
}

}  // namespace kvaf::acl_python
