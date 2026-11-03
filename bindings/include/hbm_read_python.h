#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hbm_read.h"
namespace kvaf {
void BindNpuPullHbmSource(pybind11::module_&);
namespace hbm_read_python {
namespace py=pybind11;
inline py::dict Pack(const HbmReceiverTicket& t) {
  py::dict d;d["version"]=1;d["control_address"]=t.control_address;
  d["receiver_id"]=t.receiver_id;d["receiver_epoch"]=t.receiver_epoch;return d;
}
inline HbmReceiverTicket Receiver(const py::dict& d) {
  if(d["version"].cast<int>()!=1)throw py::value_error("unsupported HBM receiver ticket");
  HbmReceiverTicket t{d["control_address"].cast<std::string>(),d["receiver_id"].cast<std::uint64_t>(),d["receiver_epoch"].cast<std::uint64_t>()};
  if(!t.valid())throw py::value_error("invalid HBM receiver ticket");return t;
}
}
}
