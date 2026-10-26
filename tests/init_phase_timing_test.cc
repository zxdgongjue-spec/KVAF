#include "init_phase_timing.h"

#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <thread>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stdout, "line %d: %s\n", __LINE__, #condition); std::exit(1); \
} } while (false)

class StderrCapture {
 public:
  StderrCapture() {
    std::fflush(stderr);
    saved_ = dup(STDERR_FILENO);
    file_ = std::tmpfile();
    CHECK(saved_ >= 0 && file_ != nullptr);
    CHECK(dup2(fileno(file_), STDERR_FILENO) >= 0);
  }

  ~StderrCapture() {
    std::fflush(stderr);
    (void)dup2(saved_, STDERR_FILENO);
    close(saved_);
    std::fclose(file_);
  }

  std::string Read() {
    std::fflush(stderr);
    CHECK(std::fseek(file_, 0L, SEEK_SET) == 0);
    std::string result;
    char buffer[1024];
    while (const std::size_t count = std::fread(buffer, 1U, sizeof(buffer), file_)) {
      result.append(buffer, count);
    }
    return result;
  }

 private:
  int saved_{-1};
  FILE* file_{nullptr};
};

void ReturnEarly() {
  kvaf::detail::InitPhaseTimer timer(true, "cpu", "early_return", 8U, 1024U, 2U);
  timer.MarkFailed();
  errno = EACCES;
  return;
}

int main() {
  static_assert(std::chrono::steady_clock::is_steady, "phase time must be monotonic");
  {
    StderrCapture capture;
    errno = EINTR;
    {
      kvaf::detail::InitPhaseTimer timer(false, "cpu", "disabled", 1U, 1U, 0U);
      CHECK(errno == EINTR);
      timer.MarkSucceeded();
    }
    CHECK(errno == EINTR);
    CHECK(capture.Read().empty());
  }
  {
    StderrCapture capture;
    char phase[] = "phase\"x\n";
    errno = ENOENT;
    {
      kvaf::detail::InitPhaseTimer timer(true, "cpu", phase, 7U, 4096U,
                                        std::numeric_limits<std::uint32_t>::max(), "no-thp");
      CHECK(errno == ENOENT);
      phase[0] = 'X';  // Timer metadata must remain stable until its end record.
      std::this_thread::sleep_for(std::chrono::milliseconds(15));
      timer.MarkSucceeded();
      errno = ERANGE;
    }
    CHECK(errno == ERANGE);
    const std::string records = capture.Read();
    CHECK(records.find("KVAF_INIT_PROFILE {\"event\":\"begin\"") == 0U);
    const auto second = records.find("KVAF_INIT_PROFILE {\"event\":\"end\"");
    CHECK(second != std::string::npos);
    CHECK(records.find("\"phase\":\"phase\\\"x\\n\"", second) != std::string::npos);
    CHECK(records.find("\"candidate\":null", second) != std::string::npos);
    CHECK(records.find("\"page_mode\":\"no-thp\"", second) != std::string::npos);
    CHECK(records.find("\"attempt\":7", second) != std::string::npos);
    CHECK(records.find("\"bytes\":4096", second) != std::string::npos);
    CHECK(records.find("\"outcome\":\"success\"", second) != std::string::npos);
    const std::string timestamp_key = "\"timestamp_unix_ns\":";
    for (const auto position : {std::size_t{0}, second}) {
      const auto timestamp_position = records.find(timestamp_key, position);
      CHECK(timestamp_position != std::string::npos);
      const char* number = records.c_str() + timestamp_position + timestamp_key.size();
      char* end = nullptr;
      CHECK(std::strtoll(number, &end, 10) > 0);
      CHECK(end != number && *end == ',');  // Integer JSON value, not a float/string.
    }
    const std::string elapsed_key = "\"elapsed_ms\":";
    const auto elapsed_position = records.find(elapsed_key, second);
    CHECK(elapsed_position != std::string::npos);
    const double elapsed = std::strtod(records.c_str() + elapsed_position + elapsed_key.size(), nullptr);
    CHECK(std::isfinite(elapsed) && elapsed >= 5.0);
    CHECK(records.find("\"usage_scope\":", second) != std::string::npos);
  }
  {
    StderrCapture capture;
    ReturnEarly();
    CHECK(errno == EACCES);
    const std::string records = capture.Read();
    CHECK(records.find("\"event\":\"end\"") != std::string::npos);
    CHECK(records.find("\"outcome\":\"failed\"") != std::string::npos);
  }
  {
    StderrCapture capture;
    {
      kvaf::detail::InitPhaseTimer timer(true, "npu", "unmarked", 3U, 1U, 0U);
    }
    CHECK(capture.Read().find("\"outcome\":\"unknown\"") != std::string::npos);
  }
  std::puts("init phase timing tests passed");
  return 0;
}
