// Optional Linux LD_PRELOAD diagnostic. Redirect this process's libc syslog
// entry points to stderr; no dependency on UMDK or a running syslog daemon.
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace {
void WriteAll(const char* data, std::size_t bytes) {
  while (bytes) {
    const auto written = write(STDERR_FILENO, data, bytes);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) return;
    data += written; bytes -= static_cast<std::size_t>(written);
  }
}

void Capture(int priority, const char* format, va_list args) {
  const int saved_errno = errno;
  char line[4096];
  const int prefix = std::snprintf(line, sizeof(line), "PROBE_SYSLOG pid=%ld priority=%d ",
    static_cast<long>(getpid()), priority);
  if (prefix < 0 || static_cast<std::size_t>(prefix) >= sizeof(line) - 1) {
    errno = saved_errno; return;
  }
  const auto available = sizeof(line) - static_cast<std::size_t>(prefix) - 1;
  va_list copy;
  va_copy(copy, args);
  errno = saved_errno;  // Preserve libc's %m formatting as well as caller errno.
  const int count = std::vsnprintf(line + prefix, available, format ? format : "(null)", copy);
  va_end(copy);
  std::size_t bytes = static_cast<std::size_t>(prefix);
  if (count >= 0) {
    if (static_cast<std::size_t>(count) >= available) {
      bytes += available - 1;
      constexpr char marker[] = " [truncated]";
      std::memcpy(line + bytes - (sizeof(marker) - 1), marker, sizeof(marker) - 1);
    } else {
      bytes += static_cast<std::size_t>(count);
    }
  }
  if (!bytes || line[bytes - 1] != '\n') line[bytes++] = '\n';
  WriteAll(line, bytes);
  errno = saved_errno;
}

__attribute__((constructor)) void Ready() {
  const int saved_errno = errno;
  constexpr char message[] = "PROBE_SYSLOG_CAPTURE_READY destination=stderr process_scope=1\n";
  WriteAll(message, sizeof(message) - 1);
  errno = saved_errno;
}
}  // namespace

extern "C" void syslog(int priority, const char* format, ...) {
  va_list args; va_start(args, format); Capture(priority, format, args); va_end(args);
}
extern "C" void vsyslog(int priority, const char* format, va_list args) {
  Capture(priority, format, args);
}
// glibc fortified callers may use these instead of the public entry points.
extern "C" void __syslog_chk(int priority, int, const char* format, ...) {
  va_list args; va_start(args, format); Capture(priority, format, args); va_end(args);
}
extern "C" void __vsyslog_chk(int priority, int, const char* format, va_list args) {
  Capture(priority, format, args);
}
