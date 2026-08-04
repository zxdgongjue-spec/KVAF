#pragma once

#include <pybind11/pybind11.h>

namespace kvaf {

// Registers a thin Python-owned wrapper over ACL Runtime Graph. It is kept
// outside Runtime and AccessOperator so the caller controls graph lifetime,
// capture scope, and replay policy.
void BindAclGraph(pybind11::module_& module);

}  // namespace kvaf
