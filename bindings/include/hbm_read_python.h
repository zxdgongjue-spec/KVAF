#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hbm_read.h"
namespace kvaf {
void BindHbmReadSource(pybind11::module_&);
namespace hbm_read_python {
namespace py=pybind11;
inline py::dict Pack(const HbmReadTicket& t){py::dict d;d["version"]=1;d["source_id"]=t.source_id;d["epoch"]=t.epoch;d["bytes"]=t.bytes;d["token"]=t.token;return d;}
inline HbmReadTicket Ticket(const py::dict& d){
  if(d["version"].cast<int>()!=1)throw py::value_error("unsupported HBM READ ticket");
  return {d["source_id"].cast<std::uint64_t>(),d["epoch"].cast<std::uint64_t>(),d["bytes"].cast<std::uint64_t>(),d["token"].cast<std::uint32_t>()};
}
inline py::dict Pack(const HbmReadBatch& b){py::dict d;d["version"]=1;d["source_id"]=b.source_id;d["epoch"]=b.epoch;d["sequence"]=b.sequence;d["source_addresses"]=b.source_addresses;d["lengths"]=b.lengths;return d;}
inline HbmReadBatch Batch(const py::dict& d){
  if(d["version"].cast<int>()!=1)throw py::value_error("unsupported raw READ batch");
  return {d["source_id"].cast<std::uint64_t>(),d["epoch"].cast<std::uint64_t>(),d["sequence"].cast<std::uint64_t>(),d["source_addresses"].cast<std::vector<std::uint64_t>>(),d["lengths"].cast<std::vector<std::uint64_t>>()};
}
}
}
