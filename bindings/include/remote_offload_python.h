#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "remote_offload.h"

namespace kvaf::remote_python {
namespace py = pybind11;
inline py::dict Pack(const OffloadRegionTicket& t) {
  py::dict d;
  d["version"] = 2; d["address"] = t.address; d["region_id"] = t.region_id;
  d["epoch"] = t.epoch; d["token"] = t.token; d["eid"] = t.eid;
  return d;
}
inline OffloadRegionTicket Ticket(const py::dict& d) {
  if (d["version"].cast<int>() != 2) throw py::value_error("unsupported DDR ticket version");
  OffloadRegionTicket t;
  t.address = d["address"].cast<std::string>(); t.region_id = d["region_id"].cast<std::uint64_t>();
  t.epoch = d["epoch"].cast<std::uint64_t>(); t.token = d["token"].cast<std::uint32_t>();
  auto eid = d["eid"].cast<std::vector<std::uint8_t>>();
  if (eid.size() != t.eid.size()) throw py::value_error("DDR ticket EID must contain 16 bytes");
  std::copy(eid.begin(), eid.end(), t.eid.begin()); return t;
}
}
