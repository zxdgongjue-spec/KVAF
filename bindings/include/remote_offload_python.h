#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "remote_offload.h"

namespace kvaf::remote_python {
namespace py = pybind11;
inline py::dict Pack(const OffloadRegionTicket& t) {
  py::dict d;
  d["version"] = 1; d["address"] = t.address; d["epoch"] = t.region.epoch;
  d["region_id"] = t.region.id; d["base"] = t.region.base; d["bytes"] = t.region.bytes;
  d["slot_size"] = t.region.slot_size; d["cpu_candidate"] = t.region.candidate;
  d["token"] = t.region.token;
  d["eid"] = std::vector<std::uint8_t>(t.region.eid, t.region.eid+16);
  return d;
}
inline OffloadRegionTicket Ticket(const py::dict& d) {
  if (d["version"].cast<int>() != 1) throw py::value_error("unsupported DDR ticket version");
  OffloadRegionTicket t;
  t.address = d["address"].cast<std::string>(); t.region.epoch = d["epoch"].cast<std::uint64_t>();
  t.region.id = d["region_id"].cast<std::uint64_t>(); t.region.base = d["base"].cast<std::uint64_t>();
  t.region.bytes = d["bytes"].cast<std::uint64_t>(); t.region.slot_size = d["slot_size"].cast<std::uint64_t>();
  t.region.candidate = d["cpu_candidate"].cast<std::uint32_t>(); t.region.token = d["token"].cast<std::uint32_t>();
  auto eid = d["eid"].cast<std::vector<std::uint8_t>>();
  if (eid.size() != 16) throw py::value_error("DDR ticket EID must contain 16 bytes");
  std::copy(eid.begin(), eid.end(), t.region.eid); return t;
}
inline py::dict Pack(const HbmReadyBatch& b) {
  py::dict d;
  d["version"] = 1; d["epoch"] = b.source.region_epoch; d["region_id"] = b.source.region_id;
  d["source_id"] = b.source.source_id; d["sequence"] = b.source.sequence;
  d["source_bytes"] = b.source.source_bytes;
  d["input_offsets"] = b.input_offsets; d["lengths"] = b.lengths; return d;
}
inline HbmReadyBatch Batch(const py::dict& d) {
  if (d["version"].cast<int>() != 1) throw py::value_error("unsupported HBM batch version");
  return {{d["epoch"].cast<std::uint64_t>(), d["region_id"].cast<std::uint64_t>(),
           d["source_id"].cast<std::uint64_t>(), d["sequence"].cast<std::uint64_t>(),
           d["source_bytes"].cast<std::uint64_t>()},
          d["input_offsets"].cast<std::vector<std::uint64_t>>(), d["lengths"].cast<std::vector<std::uint64_t>>()};
}
}
