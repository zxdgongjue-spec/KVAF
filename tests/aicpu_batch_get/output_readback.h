#pragma once

#include "output_verify.h"
#include <vector>

namespace kvaf::aicpu_test {
// CPU-only expected image. Pattern generation happens once before launch;
// successful checks use libc memcmp rather than per-byte pattern arithmetic.
class OutputReadbackVerifier {
 public:
  OutputReadbackVerifier(const Config& c, const Layout& l) : config_(c), layout_(l), expected_(l.output_bytes, 0xA5) {
    for (std::uint64_t slot = 0; slot < c.rows; ++slot)
      for (std::uint64_t b = 8; b < c.row_bytes; ++b)
        expected_[slot * l.stride + b] = Pattern(LogicalRowForOutputSlot(c, slot), b);
  }
  bool Check(std::uint64_t sequence, const std::uint8_t* data, OutputMismatch* mismatch) {
    const auto mask = config_.verify_payload ?
        static_cast<std::uint8_t>(sequence * 37 + (sequence >> 8) * 29) : std::uint8_t{0};
    for (std::uint64_t slot = 0; slot < config_.rows; ++slot) {
      auto* row = expected_.data() + slot * layout_.stride;
      const auto tag = RowTag(sequence, LogicalRowForOutputSlot(config_, slot));
      std::memcpy(row, &tag, sizeof(tag));
      if (mask != mask_)
        for (std::uint64_t b = 8; b < config_.row_bytes; ++b) row[b] ^= mask ^ mask_;
    }
    mask_ = mask;
    if (std::memcmp(data, expected_.data(), expected_.size()) == 0) return true;
    // Detailed location is needed only on failure, never on the fast path.
    for (std::uint64_t i = 0; i < layout_.output_bytes; ++i)
      if (data[i] != expected_[i]) {
        if (mismatch) *mismatch = {i, expected_[i], data[i]};
        return false;
      }
    return false;
  }
 private:
  Config config_;
  Layout layout_;
  std::vector<std::uint8_t> expected_;
  std::uint8_t mask_{};
};
}  // namespace kvaf::aicpu_test
