#pragma once

#include <pybind11/pybind11.h>

#include "acl/acl.h"

namespace kvaf::acl_python {

// Extract the raw ACL stream handle from a caller-owned torch.npu.Stream.
aclrtStream StreamFromPython(const pybind11::object& stream);

}  // namespace kvaf::acl_python
