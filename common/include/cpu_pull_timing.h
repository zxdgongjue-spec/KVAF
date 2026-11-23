#pragma once
#include <chrono>
#include <cstdint>

namespace kvaf {
// CPU-local durations, never timestamps to subtract from the reader's clock.
struct CpuPullServerTiming {
  std::uint64_t magic{}, receive_ns{}, control_lock_ns{}, validation_ns{};
  std::uint64_t worker_start_ns{}, backend_read_ns{}, completion_ns{};
  std::uint64_t reply_prepare_ns{}, dispatch_ns{}, total_ns{};
};
struct CpuPullSubmitTiming {
  double snapshot_us{}, cover_check_us{}, span_prepare_us{}, request_save_us{};
  double channel_setup_us{}, send_us{}, receive_us{}, response_check_us{};
  bool server_available{};
  CpuPullServerTiming server;
};
class CpuPullStageTimer {
 public:
  explicit CpuPullStageTimer(double* output) : output_(output) {
    if (output_) begin_ = std::chrono::steady_clock::now();
  }
  ~CpuPullStageTimer() {
    if (output_) *output_ += std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - begin_).count();
  }
 private:
  double* output_;
  std::chrono::steady_clock::time_point begin_;
};
}
