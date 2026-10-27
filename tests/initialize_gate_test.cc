#include "initialize_gate.h"

#include <poll.h>
#include <sys/wait.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); std::exit(1); \
} } while (false)

using kvaf::detail::InitializeGate;

struct Pipe {
  Pipe() { CHECK(pipe(fds) == 0); }
  ~Pipe() { close(fds[0]); close(fds[1]); }
  void Send(char value) const { CHECK(write(fds[1], &value, 1U) == 1); }
  char Receive() const {
    pollfd descriptor{fds[0], POLLIN, 0};
    CHECK(poll(&descriptor, 1U, 5000) > 0);
    char value = 0;
    CHECK(read(fds[0], &value, 1U) == 1);
    return value;
  }
  bool Ready() const {
    pollfd descriptor{fds[0], POLLIN, 0};
    CHECK(poll(&descriptor, 1U, 0) >= 0);
    return (descriptor.revents & POLLIN) != 0;
  }
  std::string Line() const {
    std::string result;
    while (true) {
      const char value = Receive();
      result += value;
      if (value == '\n') return result;
    }
  }
  int fds[2]{};
};

class Environment {
 public:
  Environment(const char* name, const char* value) : name_(name) {
    if (const char* previous = std::getenv(name)) {
      had_previous_ = true;
      previous_ = previous;
    }
    CHECK((value == nullptr ? unsetenv(name) : setenv(name, value, 1)) == 0);
  }
  ~Environment() {
    if (had_previous_) (void)setenv(name_.c_str(), previous_.c_str(), 1);
    else (void)unsetenv(name_.c_str());
  }
 private:
  std::string name_;
  std::string previous_;
  bool had_previous_{false};
};

void WaitSuccess(pid_t pid) {
  int status = 0;
  CHECK(waitpid(pid, &status, 0) == pid);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

void DisabledAndInvalid(const std::string& path) {
  Environment enabled("KVAF_SERIALIZE_INITIALIZE", nullptr);
  Environment configured_path("KVAF_INITIALIZE_LOCK_PATH", "relative/must-not-create");
  Environment timeout("KVAF_INITIALIZE_LOCK_TIMEOUT_SECONDS", "bad");
  errno = ENOENT;
  { InitializeGate gate; CHECK(gate.AcquireFromEnvironment("host:1", 0U).ok()); }
  CHECK(errno == ENOENT);
  CHECK(setenv("KVAF_SERIALIZE_INITIALIZE", "0", 1) == 0);
  { InitializeGate gate; CHECK(gate.AcquireFromEnvironment("host:1", 0U).ok()); }
  for (const char* value : {"", "true", "2", " 1", "1x"}) {
    CHECK(setenv("KVAF_SERIALIZE_INITIALIZE", value, 1) == 0);
    InitializeGate gate;
    CHECK(gate.AcquireFromEnvironment("host:1", 0U).code() == kvaf::StatusCode::kInvalidArgument);
  }
  CHECK(setenv("KVAF_SERIALIZE_INITIALIZE", "1", 1) == 0);
  CHECK(setenv("KVAF_INITIALIZE_LOCK_TIMEOUT_SECONDS", "1", 1) == 0);
  for (const char* value : {"", "relative"}) {
    CHECK(setenv("KVAF_INITIALIZE_LOCK_PATH", value, 1) == 0);
    InitializeGate gate;
    CHECK(gate.AcquireFromEnvironment("host:1", 0U).code() == kvaf::StatusCode::kInvalidArgument);
  }
  CHECK(setenv("KVAF_INITIALIZE_LOCK_PATH", path.c_str(), 1) == 0);
  for (const char* value : {"", "0", "-1", "inf", "nan", "1x", "1e999"}) {
    CHECK(setenv("KVAF_INITIALIZE_LOCK_TIMEOUT_SECONDS", value, 1) == 0);
    InitializeGate gate;
    CHECK(gate.AcquireFromEnvironment("host:1", 0U).code() == kvaf::StatusCode::kInvalidArgument);
  }
  CHECK(access(path.c_str(), F_OK) != 0);  // Invalid settings never open the file.
  CHECK(setenv("KVAF_INITIALIZE_LOCK_TIMEOUT_SECONDS", "1", 1) == 0);
  CHECK(setenv("KVAF_INITIALIZE_LOCK_PATH", (path + "/missing/lock").c_str(), 1) == 0);
  InitializeGate gate;
  const auto error = gate.AcquireFromEnvironment("host:1", 0U);
  CHECK(error.code() == kvaf::StatusCode::kInternal);
  CHECK(error.message().find("open failed") != std::string::npos);
}

void CrossProcessExclusion(const std::string& path) {
  Environment configured_path("KVAF_INITIALIZE_LOCK_PATH", path.c_str());
  Pipe events;
  Pipe release;
  Pipe logs;
  pid_t waiter = -1;
  {
    InitializeGate holder;
    CHECK(holder.AcquireFromEnvironment("same:1", 2U).ok());
    CHECK(holder.AcquireFromEnvironment("same:1", 2U).code() == kvaf::StatusCode::kInvalidState);
    waiter = fork();
    CHECK(waiter >= 0);
    if (waiter == 0) {
      CHECK(dup2(logs.fds[1], STDERR_FILENO) >= 0);
      {
        InitializeGate gate;
        CHECK(gate.AcquireFromEnvironment("same:1", 3U).ok());
        events.Send('A');
        CHECK(release.Receive() == 'R');
      }
      std::_Exit(0);
    }
    // The waiter itself proves it reached a failed LOCK_NB before release.
    const auto line = logs.Line();
    CHECK(line.find("[KVAF_INIT_GATE] event=wait") == 0U);
    CHECK(line.find("device=3") != std::string::npos);
    CHECK(!events.Ready());
  }
  CHECK(events.Receive() == 'A');
  const auto acquired = logs.Line();
  CHECK(acquired.find("event=acquired") != std::string::npos);
  CHECK(acquired.find("wait_s=") != std::string::npos);
  release.Send('R');
  WaitSuccess(waiter);
  CHECK(logs.Line().find("event=released") != std::string::npos);
  CHECK(access(path.c_str(), F_OK) == 0);  // Release never unlinks the identity.
}

void DifferentEndpointAndStableHash() {
  Environment configured_path("KVAF_INITIALIZE_LOCK_PATH", nullptr);
  const std::string first = "KVAF-test-" + std::to_string(getpid()) + ":29001";
  const std::string second = "KVAF-test-" + std::to_string(getpid()) + ":29002";
  InitializeGate holder;
  CHECK(holder.AcquireFromEnvironment(first, 0U).ok());
  Pipe result;
  const pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    {
      InitializeGate independent;
      CHECK(independent.AcquireFromEnvironment(second, 1U).ok());
      result.Send('I');
    }
    CHECK(setenv("KVAF_INITIALIZE_LOCK_TIMEOUT_SECONDS", "0.04", 1) == 0);
    InitializeGate same;
    std::string lowercase = first;
    for (char& byte : lowercase) if (byte >= 'A' && byte <= 'Z') byte = byte - 'A' + 'a';
    CHECK(same.AcquireFromEnvironment(lowercase, 2U).code() == kvaf::StatusCode::kBusy);
    result.Send('B');
    std::_Exit(0);
  }
  // Parent still owns its gate throughout both checks.
  CHECK(result.Receive() == 'I');
  CHECK(result.Receive() == 'B');
  WaitSuccess(child);
}

void TimeoutAndException(const std::string& path) {
  Environment configured_path("KVAF_INITIALIZE_LOCK_PATH", path.c_str());
  Environment timeout("KVAF_INITIALIZE_LOCK_TIMEOUT_SECONDS", "0.04");
  {
    InitializeGate holder;
    CHECK(holder.AcquireFromEnvironment("host:1", 0U).ok());
    InitializeGate waiter;
    const auto begin = std::chrono::steady_clock::now();
    const auto status = waiter.AcquireFromEnvironment("host:1", 1U);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    CHECK(status.code() == kvaf::StatusCode::kBusy);
    CHECK(status.message().find("initialization was not started") != std::string::npos);
    CHECK(elapsed >= 0.035 && elapsed < 2.0);
  }
  try {
    InitializeGate gate;
    CHECK(gate.AcquireFromEnvironment("host:1", 0U).ok());
    throw std::runtime_error("test unwinding");
  } catch (const std::runtime_error&) {}
  errno = EACCES;
  { InitializeGate gate; CHECK(gate.AcquireFromEnvironment("host:1", 0U).ok()); }
  CHECK(errno == EACCES);
}

void ProcessExitReleases(const std::string& path) {
  Environment configured_path("KVAF_INITIALIZE_LOCK_PATH", path.c_str());
  Pipe ready;
  Pipe exit_now;
  const pid_t holder = fork();
  CHECK(holder >= 0);
  if (holder == 0) {
    InitializeGate gate;
    CHECK(gate.AcquireFromEnvironment("exit:1", 0U).ok());
    ready.Send('H');
    CHECK(exit_now.Receive() == 'X');
    std::_Exit(0);  // Skip the destructor: the OS must close the flock descriptor.
  }
  CHECK(ready.Receive() == 'H');
  exit_now.Send('X');
  WaitSuccess(holder);
  InitializeGate successor;
  CHECK(successor.AcquireFromEnvironment("exit:1", 1U).ok());
}

void ExecClosesDescriptor(const std::string& path, const char* executable) {
  Environment configured_path("KVAF_INITIALIZE_LOCK_PATH", path.c_str());
  Pipe ready;
  Pipe exit_now;
  const pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    InitializeGate gate;
    CHECK(gate.AcquireFromEnvironment("exec:1", 0U).ok());
    const auto output = std::to_string(ready.fds[1]);
    const auto input = std::to_string(exit_now.fds[0]);
    execl(executable, executable, "--exec-child", output.c_str(), input.c_str(), nullptr);
    std::_Exit(99);
  }
  CHECK(ready.Receive() == 'E');  // The exec'd process is alive and parked.
  {
    InitializeGate gate;
    CHECK(gate.AcquireFromEnvironment("exec:1", 1U).ok());
  }
  exit_now.Send('X');
  WaitSuccess(child);
}

void ReturnWithGate() {
  InitializeGate gate;
  CHECK(gate.AcquireFromEnvironment("return:1", 0U).ok());
  return;
}

void LoggingFailureStillReleases(const std::string& path) {
  Environment configured_path("KVAF_INITIALIZE_LOCK_PATH", path.c_str());
  ReturnWithGate();
  Pipe result;
  const pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    const int read_only = open("/dev/null", O_RDONLY);
    CHECK(read_only >= 0 && dup2(read_only, STDERR_FILENO) >= 0);
    close(read_only);
    errno = ERANGE;
    {
      InitializeGate gate;
      CHECK(gate.AcquireFromEnvironment("return:1", 0U).ok());
    }
    CHECK(errno == ERANGE);
    // Reacquire before exiting, so OS cleanup cannot hide a destructor leak.
    InitializeGate successor;
    CHECK(successor.AcquireFromEnvironment("return:1", 1U).ok());
    result.Send('S');
    std::_Exit(0);
  }
  CHECK(result.Receive() == 'S');
  WaitSuccess(child);
}

void ExcludedSelectorLock(const std::string& path) {
  Environment configured_path("KVAF_INITIALIZE_LOCK_PATH", path.c_str());
  {
    InitializeGate gate;
    const auto status = gate.AcquireFromEnvironment("selector:1", 0U, path);
    CHECK(status.code() == kvaf::StatusCode::kInvalidArgument);
    CHECK(status.message().find("self-deadlock") != std::string::npos);
  }
  const std::string hardlink_path = path + ".hardlink";
  CHECK(link(path.c_str(), hardlink_path.c_str()) == 0);
  {
    InitializeGate gate;
    CHECK(gate.AcquireFromEnvironment("selector:1", 0U, hardlink_path).code() ==
          kvaf::StatusCode::kInvalidArgument);
  }
  // Failed checks close their descriptors; the normal gate remains usable.
  InitializeGate gate;
  CHECK(gate.AcquireFromEnvironment("selector:1", 0U, path + ".absent").ok());
}

int main(int argc, char** argv) {
  if (argc == 4 && std::string(argv[1]) == "--exec-child") {
    const int output = std::stoi(argv[2]);
    const int input = std::stoi(argv[3]);
    CHECK(write(output, "E", 1U) == 1);
    char value = 0;
    CHECK(read(input, &value, 1U) == 1 && value == 'X');
    return 0;
  }
  static_assert(!std::is_copy_constructible<InitializeGate>::value, "gate must not copy");
  static_assert(!std::is_copy_assignable<InitializeGate>::value, "gate must not copy");
  char directory[] = "/tmp/kvaf-init-gate-test-XXXXXX";
  CHECK(mkdtemp(directory) != nullptr);
  const std::string path = std::string(directory) + "/gate.lock";
  Environment enabled("KVAF_SERIALIZE_INITIALIZE", "1");
  Environment timeout("KVAF_INITIALIZE_LOCK_TIMEOUT_SECONDS", "2");
  DisabledAndInvalid(path);
  CrossProcessExclusion(path);
  DifferentEndpointAndStableHash();
  TimeoutAndException(path);
  ProcessExitReleases(path);
  ExecClosesDescriptor(path, argv[0]);
  LoggingFailureStillReleases(path);
  ExcludedSelectorLock(path);
  std::puts("initialize gate tests passed");
  return 0;
}
