#pragma once

#include <cstdint>

namespace kvaf {

// The caller owns this optional Host timing storage. Providers write it only
// before ExecuteBatchGet returns.
struct GetProviderTiming {
  std::uint64_t prepare_ns{}, submit_ns{}, completion_ns{}, cleanup_ns{};
  std::uint64_t aiv_capture_prepare_ns{}, aiv_capture_submit_ns{};
  std::uint64_t aiv_status_reset_ns{}, aiv_kernel_launch_ns{};
  std::uint64_t status_copy_ns{}, stream_sync_ns{}, status_validate_ns{}, total_ns{};
};

}  // namespace kvaf
