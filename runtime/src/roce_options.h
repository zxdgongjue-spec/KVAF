#pragma once

#include <cstdint>
#include <string>

namespace kvaf {

struct RoceOptions {
  std::string device;
  std::uint8_t port{1};
  int gid_index{-1};
};

}  // namespace kvaf
