#include "startup_selection.h"

#include <cstdlib>
#include <iostream>
#include <string>

using kvaf::Status;
using kvaf::StatusCode;
using kvaf::detail::StartupCandidate;

namespace {
void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::abort();
  }
}

void CheckFailure(const Status& status, bool cleanup_fails) {
  Check(status.code() == StatusCode::kDeviceError, "device error classification lost");
  Check(status.message().find("ACL 507034") != std::string::npos,
        "original device failure lost");
  Check((status.message().find("StopCpuPushGet") != std::string::npos) == cleanup_fails,
        "cleanup failure must accompany original failure");
}
}  // namespace

int main() {
  const std::vector<StartupCandidate> candidates{{0, 0}, {0, 1}, {1, 0}};
  // A failed EID is still a connectivity fallback; the successful candidate stays live.
  int attempts = 0, cleanups = 0;
  auto status = kvaf::detail::SelectFirstStartupCandidate(candidates,
      [&](const StartupCandidate&, bool measure, double*) {
        Check(!measure, "unexpected bandwidth measurement");
        return ++attempts == 1 ? Status::Internal("CQ timeout") : Status::OK();
      }, [&] { ++cleanups; return Status::OK(); });
  Check(status.ok() && attempts == 2 && cleanups == 1, "connectivity retry changed");

  // A device failure after a normal connectivity miss must not try a third EID,
  // including when teardown returns the previously misleading result=3.
  for (bool cleanup_fails : {false, true}) {
    attempts = cleanups = 0;
    status = kvaf::detail::SelectFirstStartupCandidate(candidates,
        [&](const StartupCandidate&, bool, double*) {
          return ++attempts == 1 ? Status::Internal("CQ timeout") :
                                  Status::DeviceError("BatchPut ACL 507034");
        }, [&] {
          return ++cleanups == 2 && cleanup_fails
              ? Status::Internal("StopCpuPushGet result=3") : Status::OK();
        });
    Check(attempts == 2 && cleanups == 2, "retried after device failure");
    CheckFailure(status, cleanup_fails);
  }

  const std::vector<std::vector<StartupCandidate>> devices{{{0, 0}, {0, 1}}, {{1, 0}}};
  // Both adaptive measurement and final activation obey the same device-failure rule.
  for (bool fail_measurement : {false, true}) {
    for (bool cleanup_fails : {false, true}) {
      attempts = cleanups = 0;
      bool device_failed = false;
      status = kvaf::detail::SelectStartupDeviceCandidate(devices,
          [&](const StartupCandidate& candidate, bool measure, double* cycles) {
            Check(!device_failed, "adaptive selection retried after device failure");
            ++attempts;
            if (measure && !fail_measurement) {
              *cycles = candidate.peer + 1.0;
              return Status::OK();
            }
            device_failed = true;
            return Status::DeviceError("BatchPut ACL 507034");
          }, [&] {
            ++cleanups;
            return device_failed && cleanup_fails
                ? Status::Internal("StopCpuPushGet result=3") : Status::OK();
          });
      Check(attempts == (fail_measurement ? 1 : 3) && cleanups == attempts,
            "unexpected adaptive cleanup count");
      CheckFailure(status, cleanup_fails);
    }
  }

  // Teardown failure blocks all retries, even if the original error was connectivity.
  attempts = cleanups = 0;
  status = kvaf::detail::SelectFirstStartupCandidate(candidates,
      [&](const StartupCandidate&, bool, double*) {
        ++attempts;
        return Status::Internal("first CQ error");
      }, [&] { ++cleanups; return Status::Internal("queues still live"); });
  Check(attempts == 1 && cleanups == 1, "retried with live queues");
  Check(status.message().find("first CQ error") != std::string::npos &&
        status.message().find("queues still live") != std::string::npos,
        "cleanup overwrote first error");
  std::cout << "startup selection tests passed\n";
}
