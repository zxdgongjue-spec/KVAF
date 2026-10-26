#pragma once

#include <sys/resource.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace kvaf {
namespace detail {

// Absolute wall time is for aligning records only. Phase durations exclusively
// use steady_clock. Do not derive queue/network durations by subtracting these
// timestamps across hosts, whose system clocks can differ.
inline std::int64_t InitProfileTimestampUnixNs() noexcept {
  return static_cast<std::int64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count());
}

// Write one JSON string without allocating. Callers serialize complete records
// with flockfile so independently timed phases cannot interleave their fields.
inline void WriteInitProfileString(const char* value) {
  std::fputc('"', stderr);
  for (const unsigned char* cursor = reinterpret_cast<const unsigned char*>(
           value == nullptr ? "na" : value); *cursor != 0U; ++cursor) {
    switch (*cursor) {
      case '"': std::fputs("\\\"", stderr); break;
      case '\\': std::fputs("\\\\", stderr); break;
      case '\n': std::fputs("\\n", stderr); break;
      case '\r': std::fputs("\\r", stderr); break;
      case '\t': std::fputs("\\t", stderr); break;
      default:
        if (*cursor < 0x20U) {
          std::fprintf(stderr, "\\u%04x", static_cast<unsigned int>(*cursor));
        } else {
          std::fputc(*cursor, stderr);
        }
    }
  }
  std::fputc('"', stderr);
}

// An end record means the scope exited, including early returns and exceptions.
// It only reports success when the caller explicitly marks the outcome.
class InitPhaseTimer final {
 public:
  InitPhaseTimer(bool enabled, const char* role, const char* phase,
                 std::uint64_t attempt, std::uint64_t bytes,
                 std::uint32_t candidate, const char* page_mode = "na") noexcept
      : enabled_(enabled), attempt_(attempt), bytes_(bytes), candidate_(candidate) {
    if (!enabled_) return;
    const int saved_errno = errno;
    CopyLabel(role_, role);
    CopyLabel(phase_, phase);
    CopyLabel(page_mode_, page_mode);
    flockfile(stderr);
    WriteRecordStart("begin", InitProfileTimestampUnixNs());
    std::fputs("}\n", stderr);
    std::fflush(stderr);
    funlockfile(stderr);
    usage_available_ = ReadUsage(&usage_start_, &usage_who_, &usage_scope_);
    // Exclude the begin log, flushing, and resource snapshot from elapsed time.
    started_ = Clock::now();
    errno = saved_errno;
  }

  ~InitPhaseTimer() noexcept {
    if (!enabled_) return;
    const int saved_errno = errno;
    const auto finished = Clock::now();
    const auto finished_unix_ns = InitProfileTimestampUnixNs();
    struct rusage usage_end {};
    const bool have_delta = usage_available_ &&
                            getrusage(usage_who_, &usage_end) == 0;
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(finished - started_).count();
    flockfile(stderr);
    WriteRecordStart("end", finished_unix_ns);
    std::fprintf(stderr, ",\"elapsed_ms\":%.6f,\"outcome\":", elapsed_ms);
    WriteInitProfileString(outcome_);
    std::fputs(",\"usage_scope\":", stderr);
    WriteInitProfileString(have_delta ? usage_scope_ : "unavailable");
    if (have_delta) {
      std::fprintf(stderr,
                   ",\"user_cpu_ms\":%.6f,\"system_cpu_ms\":%.6f"
                   ",\"minor_faults\":%" PRId64 ",\"major_faults\":%" PRId64
                   ",\"voluntary_context_switches\":%" PRId64
                   ",\"involuntary_context_switches\":%" PRId64,
                   TimevalMillis(usage_end.ru_utime) - TimevalMillis(usage_start_.ru_utime),
                   TimevalMillis(usage_end.ru_stime) - TimevalMillis(usage_start_.ru_stime),
                   Delta(usage_end.ru_minflt, usage_start_.ru_minflt),
                   Delta(usage_end.ru_majflt, usage_start_.ru_majflt),
                   Delta(usage_end.ru_nvcsw, usage_start_.ru_nvcsw),
                   Delta(usage_end.ru_nivcsw, usage_start_.ru_nivcsw));
    }
    std::fputs("}\n", stderr);
    std::fflush(stderr);
    funlockfile(stderr);
    errno = saved_errno;
  }

  InitPhaseTimer(const InitPhaseTimer&) = delete;
  InitPhaseTimer& operator=(const InitPhaseTimer&) = delete;
  void MarkFailed() noexcept { outcome_ = "failed"; }
  void MarkSucceeded() noexcept { outcome_ = "success"; }

 private:
  using Clock = std::chrono::steady_clock;

  template <std::size_t Size>
  static void CopyLabel(char (&destination)[Size], const char* source) noexcept {
    if (source == nullptr) source = "na";
    std::size_t index = 0U;
    while (index + 1U < Size && source[index] != '\0') {
      destination[index] = source[index];
      ++index;
    }
    destination[index] = '\0';
  }

  static bool ReadUsage(struct rusage* usage, int* who,
                         const char** scope) noexcept {
#if defined(RUSAGE_THREAD)
    if (getrusage(RUSAGE_THREAD, usage) == 0) {
      *who = RUSAGE_THREAD;
      *scope = "thread";
      return true;
    }
#endif
    *who = RUSAGE_SELF;
    *scope = "process";
    return getrusage(RUSAGE_SELF, usage) == 0;
  }

  static double TimevalMillis(const struct timeval& value) noexcept {
    return static_cast<double>(value.tv_sec) * 1000.0 +
           static_cast<double>(value.tv_usec) / 1000.0;
  }

  static std::int64_t Delta(long after, long before) noexcept {
    return static_cast<std::int64_t>(after) - static_cast<std::int64_t>(before);
  }

  void WriteRecordStart(const char* event, std::int64_t timestamp_unix_ns) const noexcept {
    std::fputs("KVAF_INIT_PROFILE {\"event\":", stderr);
    WriteInitProfileString(event);
    std::fputs(",\"role\":", stderr);
    WriteInitProfileString(role_);
    std::fputs(",\"phase\":", stderr);
    WriteInitProfileString(phase_);
    std::fprintf(stderr, ",\"timestamp_unix_ns\":%" PRId64 ",\"pid\":%ld,\"attempt\":%" PRIu64
                 ",\"bytes\":%" PRIu64 ",\"candidate\":",
                 timestamp_unix_ns, static_cast<long>(getpid()), attempt_, bytes_);
    if (candidate_ == std::numeric_limits<std::uint32_t>::max()) {
      std::fputs("null", stderr);
    } else {
      std::fprintf(stderr, "%" PRIu32, candidate_);
    }
    std::fputs(",\"page_mode\":", stderr);
    WriteInitProfileString(page_mode_);
  }

  const bool enabled_;
  const std::uint64_t attempt_;
  const std::uint64_t bytes_;
  const std::uint32_t candidate_;
  char role_[32]{};
  char phase_[96]{};
  char page_mode_[32]{};
  const char* outcome_{"unknown"};
  Clock::time_point started_{};
  struct rusage usage_start_ {};
  int usage_who_{RUSAGE_SELF};
  const char* usage_scope_{"unavailable"};
  bool usage_available_{false};
};

}  // namespace detail
}  // namespace kvaf
