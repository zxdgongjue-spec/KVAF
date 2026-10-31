#include "hbm_read_python.h"
#include "acl_python_utils.h"
namespace kvaf {
namespace py=pybind11;
namespace {
void Check(const Status& s){if(!s.ok())throw std::runtime_error(s.message());}
class PythonReadSource {
 public:
  PythonReadSource(std::uint32_t device,std::string eid):source(device,std::move(eid)){}
  ~PythonReadSource(){if(!source.Shutdown().ok())(void)tensor.release();}
  void* Stream(py::object stream){
    if(stream.is_none())stream=py::module_::import("torch").attr("npu").attr("current_stream")(source.device_id());
    if(py::hasattr(stream,"device") && stream.attr("device").attr("index").cast<std::uint32_t>()!=source.device_id())
      throw py::value_error("producer stream belongs to another NPU");
    return acl_python::StreamFromPython(stream);
  }
  py::dict Register(py::object value,py::object stream){
    if(!tensor.is_none())throw py::value_error("shutdown the previous HBM read source first");
    if(!py::hasattr(value,"data_ptr") || !py::hasattr(value,"device") ||
       py::str(value.attr("device").attr("type")).cast<std::string>()!="npu" ||
       value.attr("device").attr("index").cast<std::uint32_t>()!=source.device_id() || !value.attr("is_contiguous")().cast<bool>())
      throw py::value_error("source must be contiguous on this NpuPullHbmSource's NPU");
    auto n=value.attr("numel")().cast<std::size_t>(),size=value.attr("element_size")().cast<std::size_t>();
    if(!n || !size || n>SIZE_MAX/size)throw py::value_error("invalid HBM source size");
    auto producer=Stream(stream);
    tensor=value;HbmReadTicket ticket;
    Check(source.Register({reinterpret_cast<void*>(value.attr("data_ptr")().cast<std::uintptr_t>()),n*size,DeviceKind::kNpu},producer,&ticket));
    return hbm_read_python::Pack(ticket);
  }
  void Shutdown(){Check(source.Shutdown());tensor=py::none();}
  NpuPullHbmSource source;
  py::object tensor{py::none()};
};
}
void BindNpuPullHbmSource(py::module_& m){
  py::class_<PythonReadSource>(m,"NpuPullHbmSource")
    .def(py::init<std::uint32_t,std::string>(),py::arg("device_id"),py::arg("npu_eid")="")
    .def("register",&PythonReadSource::Register,py::arg("source"),py::arg("stream")=py::none())
    .def("listen",[](PythonReadSource& p,const std::string& address){Check(p.source.Listen(address));return p.source.port();},py::arg("address"))
    .def("ready",[](PythonReadSource& p){ReadyToken t;Check(p.source.Ready(&t));return t;})
    .def("release_batch",[](PythonReadSource& p,const ReadyToken& t){Check(p.source.ReleaseBatch(t));},py::arg("ready_token"))
    .def("cancel_batch",[](PythonReadSource& p,const ReadyToken& t){Check(p.source.CancelBatch(t));},py::arg("ready_token"))
    .def("shutdown",&PythonReadSource::Shutdown)
    .def_property_readonly("connected",[](PythonReadSource& p){return p.source.connected();})
    .def_property_readonly("frozen",[](PythonReadSource& p){return p.source.frozen();})
    .def_property_readonly("selected_npu_eid",[](PythonReadSource& p){return p.source.selected_npu_eid();});
}
}
