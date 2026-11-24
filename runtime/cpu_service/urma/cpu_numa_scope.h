#pragma once

#include <map>
#include <set>
#include <string>
#include "status.h"

namespace kvaf {
class CpuNumaScope {
 public:
  Status Capture(std::string sysfs = "/sys");
  bool Allows(const std::string& device) const;

 private:
  bool unrestricted_{false};
  std::string sysfs_;
  std::set<int> nodes_;
  std::set<int> packages_;
  std::map<unsigned, std::set<int>> domain_nodes_;
};
}  // namespace kvaf
