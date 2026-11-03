#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hbm_read.h"
namespace kvaf {
void BindNpuPullHbmSource(pybind11::module_&);
namespace hbm_read_python {
namespace py=pybind11;
inline py::dict Pack(const HbmReadTicket& t){py::dict d;d["version"]=2;d["source_id"]=t.source_id;d["epoch"]=t.epoch;d["bytes"]=t.bytes;d["token"]=t.token;d["mr_bytes"]=t.mr_bytes;return d;}
inline HbmReadTicket Ticket(const py::dict& d){
  const auto version=d["version"].cast<int>();
  if(version!=1 && version!=2)throw py::value_error("unsupported HBM READ ticket");
  HbmReadTicket t{d["source_id"].cast<std::uint64_t>(),d["epoch"].cast<std::uint64_t>(),d["bytes"].cast<std::uint64_t>(),d["token"].cast<std::uint32_t>()};
  if(version==2)t.mr_bytes=d["mr_bytes"].cast<std::vector<std::uint64_t>>();
  if(!t.valid())throw py::value_error("invalid HBM READ ticket");
  return t;
}
}
}
