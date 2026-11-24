#include "hbm_read_route.h"
#include "npu_eid_fixture.h"
#include <iostream>

namespace hr = kvaf::hbm_read;
void Check(bool ok, const char* detail) { if (!ok) throw std::runtime_error(detail); }

int main() { try {
  EidFixture files;
  setenv("KVAF_HCCL_ROOTINFO_PATH", files.paths.rootinfo.c_str(), 1);
  // The fake selector file is available as an optional topology file in this fixture.
  EidFixture::Write(files.paths.selector, R"({"edge_list":[
    {"local_a":3,"local_b":2,"net_layer":0,"link_type":"PEER2PEER","local_a_ports":["1/0"],"local_b_ports":["1/3"]},
    {"local_a":4,"local_b":3,"net_layer":0,"link_type":"PEER2PEER","local_a_ports":["1/5"],"local_b_ports":["1/6"]}]})");
  EidFixture::Write(files.paths.rootinfo, R"({"topo_file_path":"dev_id","rank_list":[
    {"device_id":"6","local_id":3,"level_list":[
      {"net_layer":1,"net_type":"CLOS","rank_addr_list":[
        {"addr":"00000000000000000000000000000003","addr_type":"EID","plane_id":"blue"},
        {"addr":"00000000000000000000000000000004","addr_type":"EID","plane_id":"green"}]},
      {"rank_addr_list":[
        {"ports":["1/0"],"addr":"00000000000000000000000000000001","addr_type":"EID"},
        {"ports":["1/6"],"addr":"00000000000000000000000000000002","addr_type":"EID"},
        {"ports":["0/4"],"addr":"00000000000000000000000000000009","addr_type":"EID"}],
       "net_type":"TOPO_FILE_DESC","net_layer":0}]},
    {"device_id":7,"local_id":4,"level_list":[
      {"net_layer":0,"net_type":"MESH","rank_addr_list":[
        {"addr_type":"EID","addr":"00000000000000000000000000000005","ports":["1/5"]},
        {"addr_type":"EID","addr":"00000000000000000000000000000005","ports":["1/5"]}]},
      {"net_layer":1,"net_type":"CLOS","rank_addr_list":[
        {"addr_type":"EID","addr":"00000000000000000000000000000006","plane_id":"green"}]}]}]})");
  hr::RoutePeer a{{1}, 6}, b{{1}, 7};
  std::vector<hr::RouteCandidate> left, right;
  Check(hr::RouteCandidates(a,b,"",&left).ok() && hr::RouteCandidates(b,a,"",&right).ok(), "read local ranks");
  Check(left.size()==2 && left[0].eid.back()==1 && right.size()==1 && right[0].eid.back()==5, "physical rank, layer, mesh ports and dedup");
  auto order = hr::RoutePairs(hr::ClassifyRoute(a,b), left, right);
  Check(order.size()==2 && order.front().first==1 && order.back().first==0, "direct edge first across reversed local IDs");
  unlink(files.paths.selector.c_str());
  Check(hr::RouteCandidates(a,b,"",&left).ok() && !left.front().direct_edge, "file-order fallback without port map");
  b.boot_id[0] = 2;
  Check(hr::RouteCandidates(a,b,"",&left).ok() && hr::RouteCandidates(b,a,"",&right).ok(), "read remote ranks");
  Check(left.size()==2 && left[0].eid.back()==3 && right.size()==1 && right[0].eid.back()==6, "remote CLOS only");
  order = hr::RoutePairs(hr::ClassifyRoute(a,b), left, right);
  Check(order.size()==2 && order.front().first==1 && order.back().first==0, "same plane first then other pairs");
  Check(hr::RouteCandidates(b,a,"00000000000000000000000000000006",&right).ok() && right.front().pinned &&
        hr::RoutePairs(hr::RouteScope::Remote,left,right).front().first==1, "match known plane of pinned peer");
  unlink(files.paths.rootinfo.c_str());
  Check(!hr::RouteCandidates(a,b,"",&left).ok(), "missing rootinfo cannot select a different network");
  Check(hr::ClassifyRoute({},b)==hr::RouteScope::Unknown && !hr::RouteCandidates({},b,"",&left).ok(), "unknown host is not remote");
  Check(hr::RouteCandidates({}, {}, "0000:0000:0000:0000:0000:0000:0000:0009", &right).ok() &&
        right.size()==1 && right[0].eid.back()==9, "pinned endpoint requires neither topology nor host identity");
  std::cout << "NPU Pull route selection passed\n";
  return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
