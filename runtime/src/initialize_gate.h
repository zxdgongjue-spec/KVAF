#pragma once

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "status.h"

namespace kvaf {
namespace detail {

// A host-local advisory lock. Every participating process must enable the gate;
// distinct physical hosts require an external coordination mechanism.
class InitializeGate final {
 public:
  InitializeGate() = default;
  ~InitializeGate() noexcept {
    if (fd_ < 0) return;
    const int saved_errno = errno;
    const double held = Seconds(Clock::now() - acquired_);
    (void)flock(fd_, LOCK_UN);
    (void)close(fd_);
    fd_ = -1;
    Log("released", wait_seconds_, held);
    errno = saved_errno;
  }

  InitializeGate(const InitializeGate&) = delete;
  InitializeGate& operator=(const InitializeGate&) = delete;

  Status AcquireFromEnvironment(const std::string& endpoint,
                                std::uint32_t device_id,
                                const std::string& excluded_lock_path = {}) {
    const int saved_errno = errno;
    if (fd_ >= 0) return Status::InvalidState("InitializeGate is already acquired");
    const char* enabled = std::getenv("KVAF_SERIALIZE_INITIALIZE");
    if (enabled == nullptr || std::strcmp(enabled, "0") == 0) {
      errno = saved_errno;
      return Status::OK();
    }
    if (std::strcmp(enabled, "1") != 0) {
      return Status::InvalidArgument("KVAF_SERIALIZE_INITIALIZE must be 0 or 1");
    }
    double timeout_seconds = 1800.0;
    if (const char* configured = std::getenv("KVAF_INITIALIZE_LOCK_TIMEOUT_SECONDS")) {
      char* end = nullptr;
      errno = 0;
      timeout_seconds = std::strtod(configured, &end);
      if (configured == end || end == nullptr || *end != '\0' || errno == ERANGE ||
          !std::isfinite(timeout_seconds) || timeout_seconds <= 0.0) {
        return Status::InvalidArgument(
            "KVAF_INITIALIZE_LOCK_TIMEOUT_SECONDS must be a positive finite number");
      }
    }
    endpoint_ = endpoint;
    device_id_ = device_id;
    const char* configured_path = std::getenv("KVAF_INITIALIZE_LOCK_PATH");
    if (configured_path != nullptr) {
      path_ = configured_path;
      if (path_.empty() || path_.front() != '/') {
        return Status::InvalidArgument("KVAF_INITIALIZE_LOCK_PATH must be an absolute path");
      }
    } else {
      path_ = DefaultPath(endpoint);
    }
    int flags = O_RDWR | O_CREAT | O_CLOEXEC;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    Descriptor candidate(open(path_.c_str(), flags, 0600));
    if (candidate.fd < 0) return SystemError("open", errno);
    // Only a regular persistent file can provide the intended lock identity.
    struct stat info {};
    if (fstat(candidate.fd, &info) != 0) return SystemError("fstat", errno);
    if (!S_ISREG(info.st_mode)) {
      return Status::InvalidArgument("InitializeGate lock path is not a regular file: " + path_);
    }
    if (!excluded_lock_path.empty()) {
      struct stat excluded {};
      if (stat(excluded_lock_path.c_str(), &excluded) == 0) {
        if (info.st_dev == excluded.st_dev && info.st_ino == excluded.st_ino) {
          return Status::InvalidArgument(
              "InitializeGate lock path refers to the same file as the excluded selector lock " +
              excluded_lock_path + "; choose another KVAF_INITIALIZE_LOCK_PATH to avoid self-deadlock");
        }
      } else if (errno != ENOENT) {
        const int error = errno;
        const auto result = Status::Internal(
            "InitializeGate stat failed for excluded selector lock " + excluded_lock_path +
            ": " + std::strerror(error));
        errno = error;
        return result;
      }
    }
    const auto started = Clock::now();
    bool waiting_logged = false;
    while (true) {
      const double before_attempt = Seconds(Clock::now() - started);
      if (waiting_logged && before_attempt >= timeout_seconds) {
        return TimeoutStatus(before_attempt, saved_errno);
      }
      if (flock(candidate.fd, LOCK_EX | LOCK_NB) == 0) {
        // Assign the descriptor before logging so stack unwinding always owns it.
        fd_ = candidate.Release();
        acquired_ = Clock::now();
        wait_seconds_ = Seconds(acquired_ - started);
        Log("acquired", wait_seconds_, 0.0);
        errno = saved_errno;
        return Status::OK();
      }
      const int error = errno;
      if (error != EWOULDBLOCK && error != EAGAIN && error != EINTR) {
        return SystemError("flock", error);
      }
      const double elapsed = Seconds(Clock::now() - started);
      if (!waiting_logged) {
        Log("wait", elapsed, 0.0);
        waiting_logged = true;
      }
      if (elapsed >= timeout_seconds) {
        return TimeoutStatus(elapsed, saved_errno);
      }
      // Cap each sleep; a very large finite timeout cannot overflow a clock duration.
      const double delay = std::min(0.01, timeout_seconds - elapsed);
      std::this_thread::sleep_for(std::chrono::duration<double>(delay));
    }
  }

 private:
  using Clock = std::chrono::steady_clock;

  struct Descriptor {
    explicit Descriptor(int value) : fd(value) {}
    ~Descriptor() noexcept {
      const int saved_errno = errno;
      if (fd >= 0) (void)close(fd);
      errno = saved_errno;
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    int Release() noexcept { const int result = fd; fd = -1; return result; }
    int fd;
  };

  static double Seconds(Clock::duration duration) noexcept {
    return std::chrono::duration<double>(duration).count();
  }

  static std::string DefaultPath(const std::string& endpoint) {
    // FNV-1a is specified here rather than std::hash, which need not be stable
    // between processes or library versions. ASCII host case is immaterial.
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned char byte : endpoint) {
      if (byte >= 'A' && byte <= 'Z') byte = static_cast<unsigned char>(byte - 'A' + 'a');
      hash ^= byte;
      hash *= UINT64_C(1099511628211);
    }
    char encoded[17]{};
    std::snprintf(encoded, sizeof(encoded), "%016llx", static_cast<unsigned long long>(hash));
    return "/tmp/kvaf-init-" + std::to_string(static_cast<unsigned long long>(getuid())) +
           "-" + encoded + ".lock";
  }

  Status SystemError(const char* operation, int error) const {
    const std::string message = std::string("InitializeGate ") + operation + " failed for " +
                                path_ + ": " + std::strerror(error);
    errno = error;
    return Status::Internal(message);
  }

  Status TimeoutStatus(double elapsed, int saved_errno) const {
    Log("timeout", elapsed, 0.0);
    const auto result = Status::Busy(
        "InitializeGate timed out after " + std::to_string(elapsed) +
        " seconds waiting for endpoint " + endpoint_ + " lock " + path_ +
        "; initialization was not started");
    errno = saved_errno;
    return result;
  }

  static void WriteQuoted(const std::string& value) noexcept {
    std::fputc('"', stderr);
    for (const unsigned char byte : value) {
      if (byte == '"' || byte == '\\') {
        std::fputc('\\', stderr);
        std::fputc(byte, stderr);
      } else if (byte < 0x20U) {
        std::fprintf(stderr, "\\u%04x", static_cast<unsigned int>(byte));
      } else {
        std::fputc(byte, stderr);
      }
    }
    std::fputc('"', stderr);
  }

  void Log(const char* event, double waited, double held) const noexcept {
    const int saved_errno = errno;
    flockfile(stderr);
    std::fprintf(stderr, "[KVAF_INIT_GATE] event=%s pid=%ld device=%u endpoint=",
                 event, static_cast<long>(getpid()), static_cast<unsigned int>(device_id_));
    WriteQuoted(endpoint_);
    std::fputs(" path=", stderr);
    WriteQuoted(path_);
    std::fprintf(stderr, " wait_s=%.6f held_s=%.6f\n", waited, held);
    (void)std::fflush(stderr);
    funlockfile(stderr);
    errno = saved_errno;
  }

  int fd_{-1};
  std::uint32_t device_id_{0U};
  std::string endpoint_;
  std::string path_;
  Clock::time_point acquired_{};
  double wait_seconds_{0.0};
};

}  // namespace detail
}  // namespace kvaf
