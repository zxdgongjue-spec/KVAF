#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "runtime.h"

namespace kvaf {
void BindConnectorTypes(pybind11::module_&);
}
