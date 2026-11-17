#pragma once

#include <pybind11/pybind11.h>

namespace kvaf {

// Registered only in a CANN build. Keep this separate from the portable
// binding so a normal Mac/CPU build never includes ACL or Direct-HCCP headers.
// Fused context methods retain output and the selected stream until native
// fenced release succeeds; the dict is not a version-specific device ABI.
void BindNpuPythonRuntime(pybind11::module_& module);

}  // namespace kvaf
