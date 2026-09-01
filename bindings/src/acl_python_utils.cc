#include "acl_python_utils.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace kvaf::acl_python {

aclrtStream StreamFromPython(const py::object& stream) {
  if (!py::hasattr(stream, "npu_stream")) {
    throw py::type_error("ACL stream must be a torch.npu.Stream");
  }
  const std::uint64_t pointer = stream.attr("npu_stream").cast<std::uint64_t>();
  if (pointer == 0) {
    throw py::value_error("ACL stream must not be null");
  }
  return reinterpret_cast<aclrtStream>(static_cast<std::uintptr_t>(pointer));
}

}  // namespace kvaf::acl_python
