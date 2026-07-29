#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_kvaf, m) {
  m.doc() = "KVAF initial placeholder bindings";
  m.def("build_stage", []() { return "placeholder"; });

  // 仅固定 Python 侧命名空间；具体类和函数在对应模块接口稳定后补充。
  m.def_submodule("runtime");
  m.def_submodule("meta_manager");
  m.def_submodule("access_operator");
}
