#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hbm_read.h"
namespace kvaf {
void BindNpuPullHbmSource(pybind11::module_&);
namespace hbm_read_python {
namespace py=pybind11;
inline py::dict Pack(const HbmReadTicket& t){py::dict d;d["version"]=1;d["source_id"]=t.source_id;d["epoch"]=t.epoch;d["bytes"]=t.bytes;d["token"]=t.token;return d;}
inline HbmReadTicket Ticket(const py::dict& d){
  if(d["version"].cast<int>()!=1)throw py::value_error("unsupported HBM READ ticket");
  return {d["source_id"].cast<std::uint64_t>(),d["epoch"].cast<std::uint64_t>(),d["bytes"].cast<std::uint64_t>(),d["token"].cast<std::uint32_t>()};
}
}
}
