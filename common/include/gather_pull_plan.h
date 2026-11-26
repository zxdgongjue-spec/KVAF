#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace kvaf {

inline constexpr std::uint64_t kGatherPullPlanMagic = 0x4b56414650554c31ULL;
inline constexpr std::uint32_t kGatherPullEnabled = 0x50554c31U;
inline constexpr std::uint64_t kGatherChunkedPullMagic = 0x4b56414650554c32ULL;
inline constexpr std::uint32_t kGatherChunkedPullEnabled = 0x50554c32U;
inline constexpr std::uint32_t kGatherMaxPullGroups = 8;

// Same group boundaries as the SEND experiment. Each completed group is
// published as a cumulative byte prefix in the CPU descriptor.
struct GatherPullPlan {
  std::uint32_t head_rows{}, round_rows{}, total_rows{}, token_bytes{};

  constexpr std::uint32_t GroupCount() const {
    if (!head_rows || !round_rows || head_rows > total_rows) return 0;
    return 1U + static_cast<std::uint32_t>(
        (std::uint64_t{total_rows - head_rows} + round_rows - 1) / round_rows);
  }
  constexpr std::uint64_t EndBytes(std::uint32_t index) const {
    return std::min<std::uint64_t>(total_rows, head_rows + std::uint64_t{index} * round_rows) * token_bytes;
  }
  constexpr bool Valid() const {
    return ValidLayout() && GroupCount() <= kGatherMaxPullGroups;
  }
  constexpr bool ValidChunked() const {
    return ValidLayout() && head_rows == round_rows;
  }
  constexpr bool IsReadyBoundary(std::uint64_t bytes) const {
    if (bytes == std::uint64_t{total_rows} * token_bytes) return true;
    const auto first = EndBytes(0);
    return bytes >= first && (bytes - first) % (std::uint64_t{round_rows} * token_bytes) == 0;
  }
  static constexpr GatherPullPlan MakeChunked(std::uint32_t rows, std::uint32_t token, std::uint32_t sgl,
                                              std::uint32_t batch_wrs) {
    if (!rows || !token || !sgl || !batch_wrs || std::uint64_t{rows} * token > UINT32_MAX)
      return {};
    const auto batch = std::min<std::uint64_t>(rows, std::uint64_t{sgl} * batch_wrs);
    const auto aligned = static_cast<std::uint32_t>(std::min<std::uint64_t>(rows, std::lcm(batch, std::uint64_t{4})));
    return {aligned, aligned, rows, token};
  }
  static constexpr GatherPullPlan Make(std::uint32_t rows, std::uint32_t token, std::uint32_t sgl,
                                        std::uint32_t batch_wrs, std::uint32_t jettys) {
    if (!rows || !token || !sgl || !batch_wrs || !jettys || std::uint64_t{rows} * token > UINT32_MAX)
      return {};
    const auto batch = std::min<std::uint64_t>(rows, std::uint64_t{sgl} * batch_wrs);
    // A split starts on both a full READ batch and a descriptor cache line.
    const auto head = std::min<std::uint64_t>(rows, std::lcm(batch, std::uint64_t{4}));
    const auto rotation = std::min<std::uint64_t>(rows, batch * jettys);
    const auto min_round = (rows - head + kGatherMaxPullGroups - 2) / (kGatherMaxPullGroups - 1);
    const auto rotations = std::max<std::uint64_t>(1, (min_round + rotation - 1) / rotation);
    return {static_cast<std::uint32_t>(head),
            static_cast<std::uint32_t>(std::min<std::uint64_t>(rows, rotation * rotations)),
            rows, token};
  }

 private:
  constexpr bool ValidLayout() const {
    return token_bytes && std::uint64_t{total_rows} * token_bytes <= UINT32_MAX &&
           head_rows && head_rows <= total_rows && (head_rows == total_rows || head_rows % 4 == 0) && round_rows;
  }
};
static_assert(sizeof(GatherPullPlan) == 16);
static_assert(GatherPullPlan::Make(1366, 656, 6, 32, 4).EndBytes(1) == 960U * 656);
static_assert(GatherPullPlan::MakeChunked(1366, 656, 6, 32).GroupCount() == 8);
}  // namespace kvaf
