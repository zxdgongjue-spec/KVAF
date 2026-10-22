#pragma once

#include "abi.h"
#include <cstring>
#include <vector>

namespace kvaf::aicpu_test {
// Precompute expected payload; update round tags before checking output.
class OutputReadbackVerifier {
 public:
  OutputReadbackVerifier(const Config& c, const Layout& l) : config_(c), layout_(l), expected_(l.output_bytes, 0xA5) {
    for (std::uint64_t slot = 0; slot < c.rows; ++slot)
      for (std::uint64_t b = 8; b < c.row_bytes; ++b)
        expected_[slot * l.stride + b] = Pattern(LogicalRowForOutputSlot(c, slot), b);
  }
  bool Check(std::uint64_t sequence, const std::uint8_t* data, std::uint64_t* mismatch_offset) {
    for (std::uint64_t slot = 0; slot < config_.rows; ++slot) {
      auto* row = expected_.data() + slot * layout_.stride;
      const auto tag = RowTag(sequence, LogicalRowForOutputSlot(config_, slot));
      std::memcpy(row, &tag, sizeof(tag));
    }
    if (std::memcmp(data, expected_.data(), expected_.size()) == 0) return true;
    for (std::uint64_t i = 0; i < layout_.output_bytes; ++i)
      if (data[i] != expected_[i]) {
        if (mismatch_offset) *mismatch_offset = i;
        return false;
      }
    return false;
  }
 private:
  Config config_;
  Layout layout_;
  std::vector<std::uint8_t> expected_;
};
}  // namespace kvaf::aicpu_test
