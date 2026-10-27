#pragma once

#include <pybind11/pybind11.h>
#include "runtime.h"

namespace kvaf::python_binding {
inline void PoolError(const Status& status) {
  if (status.ok()) return;
  if (status.code() == StatusCode::kNotFound) throw pybind11::key_error(status.message());
  if (status.code() == StatusCode::kOutOfMemory) throw std::runtime_error(status.message());
  throw pybind11::value_error(status.message());
}
inline void BindPoolTypes(pybind11::module_& m) {
  namespace py = pybind11;
  py::class_<PoolHandle>(m, "PoolHandle")
      .def_readonly("owner", &PoolHandle::owner).def_readonly("id", &PoolHandle::id);
  py::class_<LayerHandle>(m, "LayerHandle")
      .def_readonly("owner", &LayerHandle::owner).def_readonly("id", &LayerHandle::id);
}
template <class Wrapper>
void BindPoolMethods(pybind11::class_<Wrapper>& cls) {
  namespace py = pybind11;
  cls.def("create_pool", [](Wrapper& self, std::string name, std::uint64_t slot_size,
                            std::uint64_t slot_count, std::uint32_t token_bytes,
                            std::uint32_t tokens_per_block) {
        PoolHandle handle;
        PoolError(self.Core().CreatePool(PoolSpec{std::move(name), slot_size, slot_count,
                                                  token_bytes, tokens_per_block}, &handle));
        return handle;
      }, py::arg("name"), py::arg("slot_size"), py::arg("slot_count"),
         py::arg("token_bytes"), py::arg("tokens_per_block"))
    .def("bind_layer", [](Wrapper& self, std::uint64_t layer_id, PoolHandle pool,
                          std::uint64_t namespace_id) {
        LayerHandle layer;
        PoolError(self.Core().BindLayer(namespace_id, layer_id, pool, &layer));
        return layer;
      }, py::arg("layer_id"), py::arg("pool"), py::arg("namespace_id") = 0)
    .def("get_layer", [](Wrapper& self, std::uint64_t layer_id, std::uint64_t namespace_id) {
        LayerHandle layer;
        PoolError(self.Core().GetLayer(namespace_id, layer_id, &layer));
        return layer;
      }, py::arg("layer_id"), py::arg("namespace_id") = 0)
    .def("unbind_layer", [](Wrapper& self, LayerHandle layer) {
        PoolError(self.Core().UnbindLayer(layer));
      })
    .def("destroy_pool", [](Wrapper& self, PoolHandle pool) {
        PoolError(self.Core().DestroyPool(pool));
      })
    .def("pool_info", [](Wrapper& self, PoolHandle pool) {
        PoolInfo info;
        PoolError(self.Core().GetPoolInfo(pool, &info));
        py::dict result;
        result["name"] = info.spec.name; result["base"] = info.region.base;
        result["length"] = info.region.length; result["slot_size"] = info.spec.slot_size;
        result["slot_count"] = info.spec.slot_count;
        result["token_bytes"] = info.spec.token_bytes;
        result["tokens_per_block"] = info.spec.tokens_per_block;
        result["free_slots"] = info.free_slots; result["reserved_slots"] = info.reserved_slots;
        result["committed_slots"] = info.committed_slots;
        result["quarantined_slots"] = info.quarantined_slots;
        result["quarantined"] = info.quarantined; result["bound_layers"] = info.bound_layers;
        return result;
      });
}
}  // namespace kvaf::python_binding
