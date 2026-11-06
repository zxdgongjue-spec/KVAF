#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace kvaf::gather_call {
// Host-only diagnostics. No device ABI fields, clock-domain subtraction or
// baseline clock reads. The synchronous call stays on its submitting thread.
enum Phase : std::size_t {
  kBinding, kTensorMetadata, kStreamResolve, kNative, kRuntimeValidation,
  kOperator, kMutexWait, kCaptureQuery, kPriorDrain, kInput,
  kLaunch, kStatusCopy, kStreamSync, kStatusValidate, kOutputAcquire, kCount
};
inline constexpr const char* kNames[kCount] = {
    "call_binding_ns", "call_tensor_metadata_ns", "call_stream_resolve_ns",
    "call_native_ns", "call_runtime_validation_ns", "call_operator_ns",
    "call_mutex_wait_ns", "call_capture_query_ns", "call_prior_drain_ns",
    "call_input_ns", "call_launch_ns", "call_status_copy_enqueue_ns",
    "call_stream_sync_ns", "call_status_validate_ns", "call_output_acquire_ns"};
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
}  // namespace kvaf::gather_call
