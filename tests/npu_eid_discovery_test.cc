#include "npu_eid_fixture.h"
#include <iostream>

void Require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() { try {
  using namespace kvaf::npu_eid;
  EidFixture fixture;
  const std::string clos = "00000000003f030000100000df00eb00";
  const std::string pair = "000000000000020000100000df00e100";
  const auto& paths = fixture.paths;
  EidFixture::Write(paths.rootinfo, EidFixture::Clos(0, pair) + EidFixture::Clos(7, pair, 0) +
      EidFixture::Clos(7, clos) + EidFixture::Clos(7, clos));
  EidFixture::Write(paths.pair_info, EidFixture::Pair(7, "0000:0000:0000:0200:0010:0000:df00:e100"));
  std::vector<NpuLocalEidCandidate> candidates;
  std::string error;
  Require(DiscoverPhysicalNpuEids(7, &candidates, &error, paths), "discover physical device");
  Require(candidates.size() == 2 && EidToHex(candidates[0].eid) == clos && EidToHex(candidates[1].eid) == pair,
          "physical device filter, CLOS-first order, dedup and pair fallback");
  Require(candidates[1].source == "asdrv_pair_info_device_eid", "candidate provenance");
  unsigned selected = 0; std::ifstream(paths.selector) >> selected;
  Require(selected == 7, "write physical ID before reading pair_info");
  EidFixture::Write(paths.pair_info, EidFixture::Pair(7, clos));
  Require(DiscoverPhysicalNpuEids(7, &candidates, &error, paths) && candidates.size() == 1,
          "deduplicate across discovery sources");
  unlink(paths.rootinfo.c_str());
  Require(DiscoverPhysicalNpuEids(7, &candidates, &error, paths) && candidates.size() == 1,
          "missing rootinfo still uses pair_info");
  EidFixture::Write(paths.pair_info, EidFixture::Pair(0, pair));
  Require(!DiscoverPhysicalNpuEids(7, &candidates, &error, paths) && candidates.empty(), "reject another physical device");
  EidFixture::Write(paths.pair_info, EidFixture::Pair(7, pair) + "chan_id=1, flag=1\nremote_eid:" + clos + "\n");
  Require(!DiscoverPhysicalNpuEids(7, &candidates, &error, paths), "reject ambiguous active channels");
  EidFixture::Write(paths.pair_info, EidFixture::Pair(7, std::string(32, '0')));
  Require(!DiscoverPhysicalNpuEids(7, &candidates, &error, paths), "reject zero EID");
  EidFixture::Write(paths.pair_info, EidFixture::Pair(7, pair));
  auto unwritable = paths; unwritable.selector = fixture.directory + "/missing/dev_id";
  Require(!DiscoverPhysicalNpuEids(7, &candidates, &error, unwritable), "never use stale snapshot after failed selection");
  std::cout << "NPU EID discovery tests passed\n";
  return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
