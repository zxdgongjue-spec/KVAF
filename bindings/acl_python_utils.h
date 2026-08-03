#pragma once

#include <pybind11/pybind11.h>

#include "acl/acl.h"

namespace kvaf::acl_python {

// Accept either a framework-owned torch.npu.Stream or a Python-owned
// kvaf.AclStream. Both expose a raw aclrtStream handle to KVAF.
aclrtStream StreamFromPython(const pybind11::object& stream);

}  // namespace kvaf::acl_python
