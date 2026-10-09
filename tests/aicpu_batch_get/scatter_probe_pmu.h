#pragma once

#include <cerrno>
#include <cstdint>

#if defined(__linux__) && __has_include(<linux/perf_event.h>) && __has_include(<sys/syscall.h>)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#if defined(SYS_perf_event_open)
#define KVAF_SCATTER_PROBE_PERF 1
#else
#define KVAF_SCATTER_PROBE_PERF 0
#endif
#else
#define KVAF_SCATTER_PROBE_PERF 0
#endif

namespace kvaf::aicpu_test {
constexpr unsigned kScatterProbePmuEvents = 5;
constexpr std::uint64_t kScatterProbePmuMask = (1ULL << kScatterProbePmuEvents) - 1;
inline const char* ScatterProbePmuName(unsigned event) {
  constexpr const char* names[] = {"cycles", "instructions", "cache_references", "cache_misses", "backend_stalls"};
  return event < kScatterProbePmuEvents ? names[event] : "invalid";
}
struct ScatterProbePmuSample {
  std::uint64_t available_mask{}, read_errno{}, enabled_ns{}, running_ns{};
  std::uint64_t value[kScatterProbePmuEvents]{};
  std::uint64_t event_errno[kScatterProbePmuEvents]{};
};

// Optional Linux generic hardware events, grouped per calling AICPU thread.
// No guessed vendor raw event numbers, no privileged register access. Open,
// reset/enable, disable/read are all outside the operation wall-time interval.
class ScatterProbePmu {
 public:
  ScatterProbePmu() = default;
  ScatterProbePmu(const ScatterProbePmu&) = delete;
  ScatterProbePmu& operator=(const ScatterProbePmu&) = delete;
  ~ScatterProbePmu() {
#if KVAF_SCATTER_PROBE_PERF
    for (auto fd : fds_) if (fd >= 0) close(fd);
#endif
  }
  void Open(std::uint64_t requested) {
    if (!requested) return;
#if KVAF_SCATTER_PROBE_PERF
    constexpr std::uint64_t events[] = {PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS,
        PERF_COUNT_HW_CACHE_REFERENCES, PERF_COUNT_HW_CACHE_MISSES, PERF_COUNT_HW_STALLED_CYCLES_BACKEND};
    for (unsigned i = 0; i < kScatterProbePmuEvents; ++i) {
      if (!(requested & (1ULL << i))) continue;
      perf_event_attr attr{};
      attr.type = PERF_TYPE_HARDWARE; attr.size = sizeof(attr); attr.config = events[i];
      attr.disabled = leader_ < 0; attr.exclude_kernel = 1; attr.exclude_hv = 1;
      attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
      const auto fd = static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, leader_, 0));
      if (fd < 0) { state_.event_errno[i] = errno; continue; }
      fds_[i] = fd;
      if (leader_ < 0) leader_ = fd;
      order_[count_++] = i;
      state_.available_mask |= std::uint64_t{1} << i;
    }
#else
    for (unsigned i = 0; i < kScatterProbePmuEvents; ++i)
      if (requested & (1ULL << i)) state_.event_errno[i] = ENOSYS;
#endif
  }
  void Begin(ScatterProbePmuSample& sample) {
    sample = state_; active_ = false;
#if KVAF_SCATTER_PROBE_PERF
    if (leader_ < 0) return;
    if (ioctl(leader_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0 ||
        ioctl(leader_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
      sample.read_errno = errno; return;
    }
    active_ = true;
#endif
  }
  void End(ScatterProbePmuSample& sample) {
#if KVAF_SCATTER_PROBE_PERF
    if (!active_) return;
    active_ = false;
    if (ioctl(leader_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
      sample.read_errno = errno; return;
    }
    struct Readout { std::uint64_t count, enabled, running, values[kScatterProbePmuEvents]; } data{};
    const auto bytes = (3 + count_) * sizeof(std::uint64_t);
    const auto read_bytes = read(leader_, &data, bytes);
    if (read_bytes < 0 || static_cast<std::uint64_t>(read_bytes) != bytes || data.count != count_) {
      sample.read_errno = read_bytes < 0 ? errno : EIO; return;
    }
    sample.enabled_ns = data.enabled - previous_enabled_;
    sample.running_ns = data.running - previous_running_;
    previous_enabled_ = data.enabled; previous_running_ = data.running;
    for (unsigned i = 0; i < count_; ++i) sample.value[order_[i]] = data.values[i];
#else
    (void)sample;
#endif
  }
 private:
  ScatterProbePmuSample state_{};
  bool active_{};
#if KVAF_SCATTER_PROBE_PERF
  int fds_[kScatterProbePmuEvents]{-1, -1, -1, -1, -1};
  int leader_{-1};
  unsigned order_[kScatterProbePmuEvents]{}, count_{};
  std::uint64_t previous_enabled_{}, previous_running_{};
#endif
};
}  // namespace kvaf::aicpu_test
