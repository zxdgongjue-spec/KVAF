#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace kvaf::scatter_host {
// Host-only diagnostics. No device ABI fields, clock-domain subtraction or
// baseline clock reads. The synchronous call stays on its submitting thread.
enum Phase : std::size_t {
  kBinding, kTensorMetadata, kStreamResolve, kNative, kRuntimeValidation,
  kOperator, kMutexWait, kCaptureQuery, kPriorDrain, kInput,
  kLaunch, kStatusCopy, kStreamSync, kStatusValidate, kCount
};
inline constexpr const char* kNames[kCount] = {
    "host_binding_ns", "host_tensor_metadata_ns", "host_stream_resolve_ns",
    "host_native_ns", "host_runtime_validation_ns", "host_operator_ns",
    "host_mutex_wait_ns", "host_capture_query_ns", "host_prior_drain_ns",
    "host_input_ns", "host_launch_ns", "host_status_copy_enqueue_ns",
    "host_stream_sync_ns", "host_status_validate_ns"};
struct Profile { std::array<std::uint64_t, kCount> ns{}; };
// Pybind modules use hidden visibility. Export this inline TLS symbol so the
// binding, Runtime and the separately linked Host .so share one call context.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
inline thread_local Profile* current = nullptr;
inline std::uint64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
class Session {
 public:
  explicit Session(Profile* profile) : previous_(current) {
    if (profile) *profile = {};
    current = profile;
  }
  ~Session() { current = previous_; }
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
 private:
  Profile* previous_;
};
class Timer {
 public:
  explicit Timer(Phase phase) : profile_(current), phase_(phase),
      begin_(profile_ ? NowNs() : 0) {}
  ~Timer() { Stop(); }
  void Stop() {
    if (profile_) {
      profile_->ns[phase_] += NowNs() - begin_;
      profile_ = nullptr;
    }
  }
  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;
 private:
  Profile* profile_;
  Phase phase_;
  std::uint64_t begin_;
};
}  // namespace kvaf::scatter_host
