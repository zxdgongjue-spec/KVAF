#pragma once

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <string>

#include "umdk_control.h"
#if defined(__linux__)
#include <link.h>
#endif

namespace kvaf::aicpu_urma::umdk {

class DiagnosticLog {
 public:
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    used_ = 0;
    truncated_ = false;
    text_[0] = '\0';
    last_error_[0] = '\0';
  }
  void NativeLog(int level, const char* message) {
    Append("UMDK_LOG level=%d %s\n", level, message == nullptr ? "" : message);
    if (level <= 3 && message != nullptr) {
      const int saved_errno = errno;
      std::lock_guard<std::mutex> lock(mutex_);
      std::snprintf(last_error_, sizeof(last_error_), "%s", message);
      errno = saved_errno;
    }
  }
  void LastError(char* output, std::size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (last_error_[0] != '\0') std::snprintf(output, capacity, "%s", last_error_);
  }
  void Append(const char* format, ...) {
    const int saved_errno = errno;
    std::lock_guard<std::mutex> lock(mutex_);
    va_list arguments;
    va_start(arguments, format);
    const auto remaining = sizeof(text_) - used_;
    const int length = std::vsnprintf(text_ + used_, remaining, format, arguments);
    va_end(arguments);
    if (length > 0) {
      truncated_ = truncated_ || static_cast<std::size_t>(length) >= remaining;
      used_ += std::min<std::size_t>(length, remaining - 1);
    }
    errno = saved_errno;
  }
  void Publish(const UmdkControlArgs& args, UmdkControlStatus& status) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (args.diagnostics == 0 || args.diagnostic_capacity == 0) return;
    const auto bytes = std::min<std::size_t>(used_, args.diagnostic_capacity - 1);
    auto* output = reinterpret_cast<char*>(args.diagnostics);
    std::memcpy(output, text_, bytes);
    output[bytes] = '\0';
    status.diagnostic_bytes = bytes;
    status.diagnostic_truncated = truncated_ || bytes != used_;
  }

 private:
  std::mutex mutex_;
  char text_[kUmdkControlDiagnosticBytes]{};
  char last_error_[256]{};
  std::size_t used_ = 0;
  bool truncated_ = false;
};

}  // namespace kvaf::aicpu_urma::umdk
