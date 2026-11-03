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
  ConstBufferView View(py::handle value){
    if(!py::hasattr(value,"data_ptr") || !py::hasattr(value,"device") ||
       py::str(value.attr("device").attr("type")).cast<std::string>()!="npu" ||
       value.attr("device").attr("index").cast<std::uint32_t>()!=source.device_id() || !value.attr("is_contiguous")().cast<bool>())
      throw py::value_error("source must be contiguous on this NpuPullHbmSource's NPU");
    auto n=value.attr("numel")().cast<std::size_t>(),size=value.attr("element_size")().cast<std::size_t>();
    if(!n || !size || n>SIZE_MAX/size)throw py::value_error("invalid HBM source size");
    return {reinterpret_cast<void*>(value.attr("data_ptr")().cast<std::uintptr_t>()),n*size,DeviceKind::kNpu};
  }
  void Register(py::object value,const py::dict& receiver,py::object stream){
    if(!tensor.is_none())throw py::value_error("shutdown the previous HBM read source first");
    py::list owners;
    if(py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
      for(auto item:value) owners.append(item);
    } else owners.append(value);
    std::vector<ConstBufferView> views;
    for(auto item:owners) views.push_back(View(item));
    auto producer=Stream(stream);auto target=hbm_read_python::Receiver(receiver);
    // A failed handshake can already own MRs or have uncertain peer state.
    // Pin all storage before entering native initialization, including failures.
    tensor=std::move(owners);
    Status status;
    {py::gil_scoped_release release;status=source.Register(views,target,producer);}
    Check(status);
  }
  ReadyToken Ready(py::object value){
    ReadyToken ready;Check(value.is_none()?source.Ready(&ready):source.Ready(View(value),&ready));return ready;
  }
  void Shutdown(){Status status;{py::gil_scoped_release release;status=source.Shutdown();}Check(status);tensor=py::none();}
  NpuPullHbmSource source;
  py::object tensor{py::none()};
};
}
void BindNpuPullHbmSource(py::module_& m){
  py::class_<PythonReadSource>(m,"NpuPullHbmSource")
    .def(py::init<std::uint32_t,std::string>(),py::arg("device_id"),py::arg("npu_eid")="")
    .def("register",&PythonReadSource::Register,py::arg("source"),py::kw_only(),py::arg("receiver"),py::arg("stream")=py::none(),
         "Register and connect all MRs to the receiver; returns only after READ probes and confirmation. "
         "On failure retain storage and retry shutdown. Ready cannot be cancelled unilaterally.")
    .def("ready",&PythonReadSource::Ready,py::arg("tensor")=py::none())
    .def("release_batch",[](PythonReadSource& p,const ReadyToken& t){Check(p.source.ReleaseBatch(t));},py::arg("ready_token"))
    .def("cancel_batch",[](PythonReadSource& p,const ReadyToken& t){Check(p.source.CancelBatch(t));},py::arg("ready_token"))
    .def("shutdown",&PythonReadSource::Shutdown)
    .def_property_readonly("connected",[](PythonReadSource& p){return p.source.connected();})
    .def_property_readonly("frozen",[](PythonReadSource& p){return p.source.frozen();})
    .def_property_readonly("selected_npu_eid",[](PythonReadSource& p){return p.source.selected_npu_eid();});
}
}
