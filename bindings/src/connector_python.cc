#include "connector_python.h"
namespace kvaf {
namespace py=pybind11;
void BindConnectorTypes(py::module_& m) {
  py::class_<CpuPullSourceDesc>(m,"CpuPullSourceDesc")
    .def_readonly("source_id",&CpuPullSourceDesc::source_id)
    .def_readonly("region_id",&CpuPullSourceDesc::region_id)
    .def_readonly("region_epoch",&CpuPullSourceDesc::region_epoch)
    .def_readonly("mr_length",&CpuPullSourceDesc::mr_length)
    .def(py::pickle([](const CpuPullSourceDesc& s){return py::make_tuple("kvaf.cpu-source.v2",s.source_id,s.region_id,s.region_epoch,s.mr_length);},
      [](py::tuple t){if(t.size()!=5 || !py::isinstance<py::str>(t[0]) || t[0].cast<std::string>()!="kvaf.cpu-source.v2")throw py::value_error("unsupported CPU source descriptor version");
        return CpuPullSourceDesc{t[1].cast<std::uint64_t>(),t[2].cast<std::uint64_t>(),t[3].cast<std::uint64_t>(),t[4].cast<std::uint64_t>()};}));
  py::class_<NpuPullSourceDesc>(m,"NpuPullSourceDesc")
    .def_readonly("source_id",&NpuPullSourceDesc::source_id)
    .def_readonly("source_epoch",&NpuPullSourceDesc::source_epoch)
    .def(py::pickle([](const NpuPullSourceDesc& s){return py::make_tuple("kvaf.npu-source.v2",s.source_id,s.source_epoch);},
      [](py::tuple t){if(t.size()!=3 || !py::isinstance<py::str>(t[0]) || t[0].cast<std::string>()!="kvaf.npu-source.v2")throw py::value_error("unsupported NPU source descriptor version");
        return NpuPullSourceDesc{t[1].cast<std::uint64_t>(),t[2].cast<std::uint64_t>()};}));
  py::class_<ReadyToken>(m,"ReadyToken")
    .def_readonly("source",&ReadyToken::source).def_readonly("sequence",&ReadyToken::sequence)
    .def_readonly("mr_id",&ReadyToken::mr_id)
    .def(py::pickle([](const ReadyToken& t) -> py::tuple {
        return py::make_tuple("kvaf.ready.v2",t.source,t.sequence,t.mr_id);},
      [](py::tuple t){if(t.size()!=4 || !py::isinstance<py::str>(t[0]) || t[0].cast<std::string>()!="kvaf.ready.v2")throw py::value_error("unsupported ReadyToken version");
        return ReadyToken{t[1].cast<std::variant<CpuPullSourceDesc,NpuPullSourceDesc>>(),t[2].cast<std::uint64_t>(),t[3].cast<std::uint64_t>()};}));
  py::enum_<CpuPullState>(m,"CpuPullState")
    .value("PREPARED",CpuPullState::Prepared).value("SUBMITTED",CpuPullState::Submitted)
    .value("RUNNING",CpuPullState::Running).value("DONE",CpuPullState::Done)
    .value("FAILED",CpuPullState::Failed).value("IN_DOUBT",CpuPullState::InDoubt)
    .value("COMMITTED",CpuPullState::Committed).value("ABORTED",CpuPullState::Aborted);
  py::class_<PutAllocation>(m,"PutEntry")
    .def_readonly("key",&PutAllocation::key).def_readonly("address",&PutAllocation::address)
    .def_readonly("length",&PutAllocation::reserved_length);
  py::class_<PutHandle>(m,"PutHandle")
    .def_property_readonly("id",&PutHandle::id).def_property_readonly("entries",&PutHandle::entries)
    .def_property_readonly("state",&PutHandle::state).def_property_readonly("submitted",&PutHandle::submitted);
}
}
