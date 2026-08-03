#include "acl_python_utils.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace kvaf::acl_python {

aclrtStream StreamFromPython(const py::object& stream) {
  const char* const attributes[] = {"npu_stream", "acl_stream"};
  for (const char* attribute : attributes) {
    if (!py::hasattr(stream, attribute)) {
      continue;
    }
    const std::uint64_t pointer =
        stream.attr(attribute).cast<std::uint64_t>();
    if (pointer == 0) {
      throw py::value_error("ACL stream must not be null");
    }
    return reinterpret_cast<aclrtStream>(
        static_cast<std::uintptr_t>(pointer));
  }
  throw py::type_error(
      "ACL stream must be a torch.npu.Stream or kvaf.AclStream");
}

}  // namespace kvaf::acl_python
