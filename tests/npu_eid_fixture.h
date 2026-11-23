#pragma once
#include "npu_eid_discovery.h"
#include <stdexcept>

struct EidFixture {
  std::string directory;
  kvaf::npu_eid::DiscoveryPaths paths;
  EidFixture() {
    char pattern[] = "/tmp/kvaf-eids-XXXXXX";
    const auto* dir = mkdtemp(pattern);
    if (!dir) throw std::runtime_error("create EID fixture directory");
    directory = dir;
    paths = {directory + "/rootinfo.json", directory + "/dev_id",
             directory + "/pair_info", directory + "/selector.lock"};
  }
  ~EidFixture() {
    for (const auto& path : {paths.rootinfo, paths.selector, paths.pair_info, paths.lock}) unlink(path.c_str());
    rmdir(directory.c_str());
  }
  static void Write(const std::string& path, const std::string& text) { std::ofstream(path) << text; }
  static std::string Clos(unsigned device, const std::string& eid, unsigned layer = 1) {
    return "{\n\"device_id\": " + std::to_string(device) + ",\n\"net_layer\": " + std::to_string(layer) +
        ",\n\"net_type\": \"CLOS\",\n\"addr_type\": \"EID\",\n\"addr\": \"" + eid + "\"\n}\n";
  }
  static std::string Pair(unsigned device, const std::string& eid) {
    return "Dev id info. dev_id=" + std::to_string(device) + ";\nchan_id=0, flag=1\nremote_eid:" + eid + "\n";
  }
};
