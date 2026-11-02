#include "connector_python.h"
namespace kvaf {
namespace py=pybind11;
void BindConnectorTypes(py::module_& m) {
  py::class_<CpuPullSourceDesc>(m,"CpuPullSourceDesc")
    .def_readonly("source_id",&CpuPullSourceDesc::source_id)
    .def_readonly("source_bytes",&CpuPullSourceDesc::source_bytes)
    .def_readonly("region_id",&CpuPullSourceDesc::region_id)
    .def_readonly("region_epoch",&CpuPullSourceDesc::region_epoch)
    .def(py::pickle([](const CpuPullSourceDesc& s){return py::make_tuple(s.source_id,s.source_bytes,s.region_id,s.region_epoch);},
      [](py::tuple t){if(t.size()!=4)throw py::value_error("invalid CPU source descriptor");
        return CpuPullSourceDesc{t[0].cast<std::uint64_t>(),t[1].cast<std::uint64_t>(),t[2].cast<std::uint64_t>(),t[3].cast<std::uint64_t>()};}));
  py::class_<NpuPullSourceDesc>(m,"NpuPullSourceDesc")
    .def_readonly("source_id",&NpuPullSourceDesc::source_id)
    .def_readonly("source_bytes",&NpuPullSourceDesc::source_bytes)
    .def_readonly("source_epoch",&NpuPullSourceDesc::source_epoch)
    .def(py::pickle([](const NpuPullSourceDesc& s){return py::make_tuple(s.source_id,s.source_bytes,s.source_epoch);},
      [](py::tuple t){if(t.size()!=3)throw py::value_error("invalid NPU source descriptor");
        return NpuPullSourceDesc{t[0].cast<std::uint64_t>(),t[1].cast<std::uint64_t>(),t[2].cast<std::uint64_t>()};}));
  py::class_<ReadyToken>(m,"ReadyToken")
    .def_readonly("source",&ReadyToken::source).def_readonly("sequence",&ReadyToken::sequence)
    .def(py::pickle([](const ReadyToken& t){return py::make_tuple(t.source,t.sequence);},
      [](py::tuple t){if(t.size()!=2)throw py::value_error("invalid ReadyToken");
        return ReadyToken{t[0].cast<std::variant<CpuPullSourceDesc,NpuPullSourceDesc>>(),t[1].cast<std::uint64_t>()};}));
  py::enum_<CpuPullState>(m,"CpuPullState")
    .value("PREPARED",CpuPullState::Prepared).value("SUBMITTED",CpuPullState::Submitted)
    .value("RUNNING",CpuPullState::Running).value("DONE",CpuPullState::Done)
    .value("FAILED",CpuPullState::Failed).value("IN_DOUBT",CpuPullState::InDoubt)
    .value("COMMITTED",CpuPullState::Committed).value("ABORTED",CpuPullState::Aborted);
  py::class_<PutAllocation>(m,"PutEntry")
    .def_readonly("key",&PutAllocation::key).def_readonly("address",&PutAllocation::address)
    .def_readonly("length",&PutAllocation::reserved_length);
  py::class_<CpuPullTask>(m,"CpuPullTask")
    .def_property_readonly("id",&CpuPullTask::id).def_property_readonly("valid",&CpuPullTask::valid);
  py::class_<PutHandle>(m,"PutHandle")
    .def_property_readonly("id",&PutHandle::id).def_property_readonly("entries",&PutHandle::entries)
    .def_property_readonly("state",&PutHandle::state).def_property_readonly("task",&PutHandle::task);
}
}
