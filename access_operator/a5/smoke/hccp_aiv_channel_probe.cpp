// A two-process, zero-WQE proof for the public HCOMM/HCCP bootstrap path.
//
// This is deliberately not an H2D data-plane test.  It establishes the
// product-owned Host CPU / Device AIV UBC-CTP channel, imports the Host DDR
// MR on the Device, and snapshots the SQ/CQ/token metadata that AIV will
// consume later.  It never creates an ACL stream, launches a kernel, writes a
// WQE, rings a doorbell, or waits for a CQE.

#include "kvaf/access_operator/a5/hcomm_urma_resources.h"
#include "kvaf/access_operator/a5/host_ddr_mapping.h"

#include <acl/acl.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace {

using kvaf::Status;
using kvaf::access::AivDirectUrmaChannel;
using kvaf::access::a5::ChannelEngine;
using kvaf::access::a5::ChannelRole;
using kvaf::access::a5::EndpointLocation;
using kvaf::access::a5::EndpointSpec;
using kvaf::access::a5::HcommUrmaResources;
using kvaf::access::a5::HostDdrDeviceMapping;
using kvaf::access::a5::ImportedMemoryView;
using kvaf::access::a5::MemoryLocation;

constexpr const char* kMarkerSchema = "kvaf-hccp-aiv-channel-probe-v1";
constexpr std::uint32_t kDefaultTeardownRetries = 3;
constexpr std::uint32_t kTeardownRetryDelayMs = 50;
constexpr std::uint32_t kPollIntervalMs = 20;

enum class WaitResult {
  kReady,
  kPeerFailed,
  kPeerUnsafe,
  kInvalidMarker,
  kTimeout,
};

struct Options {
  std::string role;
  std::string run_id;
  std::string exchange_dir;
  std::array<std::uint8_t, 16> host_eid{};
  std::array<std::uint8_t, 16> device_eid{};
  std::uint32_t logical_device_id{0};
  std::uint32_t physical_device_id{0};
  std::uint32_t super_device_id{0};
  std::uint32_t server_index{0};
  std::uint32_t super_pod_index{0};
  std::uint32_t timeout_seconds{30};
  std::uint32_t teardown_retries{kDefaultTeardownRetries};
  std::uint64_t bytes{4096};
  bool allow_anonymous_channel_fallback{false};
  int resource_guard_fd{-1};
};

struct RunResult {
  int code{1};
  // False means that a resource or memory backing remains intentionally owned
  // by this still-running process.  The main function then refuses context
  // finalization and holds the process; the supervisor must never kill it.
  bool safe_to_finalize{true};
  std::string stage;
};

// Both workers are started by one terminal-facing supervisor.  Once a worker
// has begun to own HCOMM/MR/HBM/Host-DDR state, terminal-group signals must
// not abruptly terminate it and let C++ unwinding perform a single unchecked
// destructor Close.  SIGKILL is unmaskable, which is why the supervisor never
// sends signals after worker launch.
class ResourceSignalGuard {
 public:
  bool Install() {
    struct sigaction ignored {};
    ignored.sa_handler = SIG_IGN;
    if (sigemptyset(&ignored.sa_mask) != 0) {
      return false;
    }
    ignored.sa_flags = 0;
    for (std::size_t index = 0; index < signals_.size(); ++index) {
      if (::sigaction(signals_[index], &ignored, &previous_[index]) != 0) {
        Restore();
        return false;
      }
      installed_ = index + 1;
    }
    return true;
  }

  void Restore() {
    while (installed_ != 0) {
      --installed_;
      (void)::sigaction(signals_[installed_], &previous_[installed_], nullptr);
    }
  }

  ~ResourceSignalGuard() { Restore(); }

 private:
  const std::array<int, 4> signals_{{SIGINT, SIGTERM, SIGHUP, SIGQUIT}};
  std::array<struct sigaction, 4> previous_{};
  std::size_t installed_{0};
};

bool ParseUnsigned(const std::string& text, std::uint64_t* value) {
  if (text.empty() || value == nullptr) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const auto parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') {
    return false;
  }
  *value = static_cast<std::uint64_t>(parsed);
  return true;
}

int HexNibble(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return 10 + value - 'a';
  }
  if (value >= 'A' && value <= 'F') {
    return 10 + value - 'A';
  }
  return -1;
}

bool ParseEid(const std::string& text, std::array<std::uint8_t, 16>* eid) {
  if (eid == nullptr) {
    return false;
  }
  std::string compact;
  compact.reserve(32);
  for (const char value : text) {
    if (value != ':' && value != '-') {
      compact.push_back(value);
    }
  }
  if (compact.size() != 32) {
    return false;
  }
  bool nonzero = false;
  for (std::size_t index = 0; index < eid->size(); ++index) {
    const int high = HexNibble(compact[index * 2]);
    const int low = HexNibble(compact[index * 2 + 1]);
    if (high < 0 || low < 0) {
      return false;
    }
    (*eid)[index] = static_cast<std::uint8_t>((high << 4) | low);
    nonzero = nonzero || (*eid)[index] != 0;
  }
  return nonzero;
}

bool ValidRunId(const std::string& run_id) {
  if (run_id.empty() || run_id.size() > 128) {
    return false;
  }
  for (const char value : run_id) {
    const auto byte = static_cast<unsigned char>(value);
    if (!std::isalnum(byte) && value != '-' && value != '_') {
      return false;
    }
  }
  return true;
}

void PrintUsage(const char* program) {
  std::cerr
      << "usage: " << program << " --print-capabilities\n"
      << "       " << program
      << " --role host|device --run-id TOKEN --exchange-dir PATH"
         " --host-eid HEX32 --device-eid HEX32"
         " --logical-device-id N --physical-device-id N"
         " --super-device-id N --server-index N --super-pod-index N"
         " [--bytes N] [--timeout-seconds N] [--teardown-retries N]"
         " [--channel-name-mode named-required|legacy-anonymous-risk-ack]"
         " [--resource-guard-fd N]\n";
}

bool ParseOptions(int argc, char** argv, Options* options) {
  if (options == nullptr) {
    return false;
  }
  std::string host_eid;
  std::string device_eid;
  bool logical_device_id_seen = false;
  bool physical_device_id_seen = false;
  bool super_device_id_seen = false;
  bool server_index_seen = false;
  bool super_pod_index_seen = false;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      return false;
    }
    const std::string key = argv[index];
    const std::string value = argv[index + 1];
    std::uint64_t parsed = 0;
    if (key == "--role") {
      options->role = value;
    } else if (key == "--run-id") {
      options->run_id = value;
    } else if (key == "--exchange-dir") {
      options->exchange_dir = value;
    } else if (key == "--host-eid") {
      host_eid = value;
    } else if (key == "--device-eid") {
      device_eid = value;
    } else if (key == "--channel-name-mode") {
      if (value == "named-required") {
        options->allow_anonymous_channel_fallback = false;
      } else if (value == "legacy-anonymous-risk-ack") {
        options->allow_anonymous_channel_fallback = true;
      } else {
        return false;
      }
    } else if (!ParseUnsigned(value, &parsed)) {
      return false;
    } else if (key == "--logical-device-id" &&
               parsed <= std::numeric_limits<std::uint32_t>::max()) {
      options->logical_device_id = static_cast<std::uint32_t>(parsed);
      logical_device_id_seen = true;
    } else if (key == "--physical-device-id" &&
               parsed <= std::numeric_limits<std::uint32_t>::max()) {
      options->physical_device_id = static_cast<std::uint32_t>(parsed);
      physical_device_id_seen = true;
    } else if (key == "--super-device-id" &&
               parsed <= std::numeric_limits<std::uint32_t>::max()) {
      options->super_device_id = static_cast<std::uint32_t>(parsed);
      super_device_id_seen = true;
    } else if (key == "--server-index" &&
               parsed <= std::numeric_limits<std::uint32_t>::max()) {
      options->server_index = static_cast<std::uint32_t>(parsed);
      server_index_seen = true;
    } else if (key == "--super-pod-index" &&
               parsed <= std::numeric_limits<std::uint32_t>::max()) {
      options->super_pod_index = static_cast<std::uint32_t>(parsed);
      super_pod_index_seen = true;
    } else if (key == "--timeout-seconds" && parsed != 0 && parsed <= 300) {
      options->timeout_seconds = static_cast<std::uint32_t>(parsed);
    } else if (key == "--teardown-retries" && parsed != 0 && parsed <= 100) {
      options->teardown_retries = static_cast<std::uint32_t>(parsed);
    } else if (key == "--bytes" && parsed != 0 &&
               parsed <= (1ULL << 30U)) {
      options->bytes = parsed;
    } else if (key == "--resource-guard-fd" &&
               parsed <= static_cast<std::uint64_t>(
                             std::numeric_limits<int>::max())) {
      options->resource_guard_fd = static_cast<int>(parsed);
    } else {
      return false;
    }
  }
  return (options->role == "host" || options->role == "device") &&
         ValidRunId(options->run_id) && !options->exchange_dir.empty() &&
         logical_device_id_seen && physical_device_id_seen &&
         super_device_id_seen && server_index_seen && super_pod_index_seen &&
         ParseEid(host_eid, &options->host_eid) &&
         ParseEid(device_eid, &options->device_eid) &&
         options->host_eid != options->device_eid;
}

bool HasOpenResourceGuardFd(const Options& options) {
  return options.resource_guard_fd >= 0 &&
         ::fcntl(options.resource_guard_fd, F_GETFD) != -1;
}

bool CheckAcl(aclError result, const char* operation) {
  if (result == ACL_SUCCESS) {
    return true;
  }
  std::cerr << operation << " failed, ACL ret=" << result << '\n';
  return false;
}

bool CheckStatus(const Status& status, const char* operation) {
  if (status.ok()) {
    return true;
  }
  std::cerr << operation << " failed: " << status.message() << '\n';
  return false;
}

EndpointSpec MakeHostEndpoint(const Options& options) {
  EndpointSpec endpoint{};
  endpoint.eid = options.host_eid;
  endpoint.location = EndpointLocation::kHost;
  endpoint.physical_device_id = options.physical_device_id;
  return endpoint;
}

EndpointSpec MakeDeviceEndpoint(const Options& options) {
  EndpointSpec endpoint{};
  endpoint.eid = options.device_eid;
  endpoint.location = EndpointLocation::kDevice;
  endpoint.physical_device_id = options.physical_device_id;
  endpoint.super_device_id = options.super_device_id;
  endpoint.server_index = options.server_index;
  endpoint.super_pod_index = options.super_pod_index;
  return endpoint;
}

std::string JoinPath(const std::string& directory, const char* name) {
  return directory + "/" + name;
}

bool FileExists(const std::string& path) {
  struct stat info {};
  return ::stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

bool AtomicWrite(const std::string& path, const std::vector<std::uint8_t>& bytes) {
  const std::string temporary =
      path + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
  const int descriptor =
      ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
             S_IRUSR | S_IWUSR);
  if (descriptor < 0) {
    return false;
  }
  std::size_t offset = 0;
  bool written = true;
  while (offset < bytes.size()) {
    const auto result = ::write(
        descriptor, bytes.data() + offset, bytes.size() - offset);
    if (result > 0) {
      offset += static_cast<std::size_t>(result);
      continue;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    written = false;
    break;
  }
  const bool flushed = written && ::fsync(descriptor) == 0;
  const bool closed = ::close(descriptor) == 0;
  if (!written || !flushed || !closed ||
      std::rename(temporary.c_str(), path.c_str()) != 0) {
    (void)::unlink(temporary.c_str());
    return false;
  }
  const std::size_t slash = path.rfind('/');
  const std::string directory =
      slash == std::string::npos ? "." : path.substr(0, slash);
  const int directory_fd =
      ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory_fd < 0) {
    return false;
  }
  const bool directory_flushed = ::fsync(directory_fd) == 0;
  const bool directory_closed = ::close(directory_fd) == 0;
  if (!directory_flushed || !directory_closed) {
    return false;
  }
  return true;
}

bool AtomicWriteText(const std::string& path, const std::string& text) {
  return AtomicWrite(path, std::vector<std::uint8_t>(text.begin(), text.end()));
}

bool ReadFile(const std::string& path, std::vector<std::uint8_t>* bytes) {
  if (bytes == nullptr) {
    return false;
  }
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return false;
  }
  const std::streamsize size = input.tellg();
  if (size <= 0 || size > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  input.seekg(0);
  bytes->resize(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(bytes->data()), size);
  return static_cast<bool>(input);
}

bool ReadText(const std::string& path, std::string* text) {
  std::vector<std::uint8_t> bytes;
  if (!ReadFile(path, &bytes) || text == nullptr) {
    return false;
  }
  text->assign(bytes.begin(), bytes.end());
  return true;
}

std::string MarkerText(const Options& options, const char* stage,
                       const std::string& detail = {}) {
  std::string text = "schema=";
  text += kMarkerSchema;
  text += "\nrun_id=" + options.run_id;
  text += "\nstage=";
  text += stage;
  if (!detail.empty()) {
    text += "\ndetail=" + detail;
  }
  text += '\n';
  return text;
}

bool MarkerMatches(const Options& options, const std::string& path,
                   const char* stage) {
  try {
    std::string text;
    if (!ReadText(path, &text)) {
      return false;
    }
    const std::array<std::string, 3> expected{{
        "schema=" + std::string(kMarkerSchema),
        "run_id=" + options.run_id,
        std::string("stage=") + stage,
    }};
    std::size_t offset = 0;
    for (const auto& expected_line : expected) {
      const std::size_t newline = text.find('\n', offset);
      if (newline == std::string::npos ||
          text.compare(offset, newline - offset, expected_line) != 0) {
        return false;
      }
      offset = newline + 1;
    }
    if (offset == text.size()) {
      return true;
    }
    const std::size_t newline = text.find('\n', offset);
    if (newline == std::string::npos || newline + 1 != text.size() ||
        text.compare(offset, sizeof("detail=") - 1, "detail=") != 0) {
      return false;
    }
    // Exactly one optional detail line is allowed.  It is diagnostic only;
    // the schema/run-id/stage triplet above is the complete marker identity.
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool WriteMarker(const Options& options, const char* name, const char* stage,
                 const std::string& detail = {}) {
  try {
    const bool ok = AtomicWriteText(JoinPath(options.exchange_dir, name),
                                    MarkerText(options, stage, detail));
    if (!ok) {
      std::cerr << "failed to atomically publish marker " << name << '\n';
    }
    return ok;
  } catch (const std::exception& error) {
    std::cerr << "exception while publishing marker " << name << ": "
              << error.what() << '\n';
    return false;
  } catch (...) {
    std::cerr << "non-standard exception while publishing marker " << name
              << '\n';
    return false;
  }
}

WaitResult PeerTerminal(const Options& options, const char* peer_role) {
  const std::string fatal_name = std::string("fatal.") + peer_role;
  const std::string fatal =
      JoinPath(options.exchange_dir, fatal_name.c_str());
  if (FileExists(fatal)) {
    return MarkerMatches(options, fatal, "fatal") ? WaitResult::kPeerFailed
                                                     : WaitResult::kInvalidMarker;
  }
  const std::string unsafe_name = std::string("unsafe.") + peer_role;
  const std::string unsafe =
      JoinPath(options.exchange_dir, unsafe_name.c_str());
  if (FileExists(unsafe)) {
    return MarkerMatches(options, unsafe, "unsafe") ? WaitResult::kPeerUnsafe
                                                       : WaitResult::kInvalidMarker;
  }
  return WaitResult::kTimeout;
}

WaitResult WaitForMarker(const Options& options, const char* name,
                         const char* stage, const char* peer_role,
                         bool watch_peer_terminal = true) {
  const std::string path = JoinPath(options.exchange_dir, name);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(options.timeout_seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (FileExists(path)) {
      return MarkerMatches(options, path, stage) ? WaitResult::kReady
                                                 : WaitResult::kInvalidMarker;
    }
    if (watch_peer_terminal) {
      const WaitResult terminal = PeerTerminal(options, peer_role);
      if (terminal != WaitResult::kTimeout) {
        return terminal;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
  }
  if (FileExists(path)) {
    return MarkerMatches(options, path, stage) ? WaitResult::kReady
                                               : WaitResult::kInvalidMarker;
  }
  if (watch_peer_terminal) {
    const WaitResult terminal = PeerTerminal(options, peer_role);
    if (terminal != WaitResult::kTimeout) {
      return terminal;
    }
  }
  return WaitResult::kTimeout;
}

const char* WaitResultName(WaitResult result) {
  switch (result) {
    case WaitResult::kReady:
      return "ready";
    case WaitResult::kPeerFailed:
      return "peer-failed";
    case WaitResult::kPeerUnsafe:
      return "peer-unsafe";
    case WaitResult::kInvalidMarker:
      return "invalid-marker";
    case WaitResult::kTimeout:
      return "timeout";
  }
  return "unknown";
}

template <typename Close>
bool CloseWithRetry(const char* subject, std::uint32_t retries, Close&& close) {
  for (std::uint32_t attempt = 1; attempt <= retries; ++attempt) {
    const Status status = close();
    if (status.ok()) {
      std::cout << "KVAF_A5_HCCP_CHANNEL_TEARDOWN_OK subject=" << subject
                << " attempts=" << attempt << '\n';
      return true;
    }
    std::cerr << "KVAF_A5_HCCP_CHANNEL_TEARDOWN_RETRY subject=" << subject
              << " attempt=" << attempt << " error=" << status.message()
              << '\n';
    if (attempt != retries) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kTeardownRetryDelayMs));
    }
  }
  return false;
}

RunResult RetainUnsafeHost(const Options& options,
                           std::unique_ptr<HcommUrmaResources>* resources,
                           std::unique_ptr<HostDdrDeviceMapping>* mapping,
                           const std::string& stage) {
  (void)WriteMarker(options, "unsafe.host", "unsafe", stage);
  std::cerr << "KVAF_A5_HCCP_CHANNEL_TEARDOWN_UNPROVEN role=host stage="
            << stage << " action=retain-process-owned-resources\n";
  if (resources != nullptr) {
    (void)resources->release();
  }
  if (mapping != nullptr) {
    (void)mapping->release();
  }
  return {1, false, stage};
}

RunResult RetainUnsafeDevice(const Options& options,
                             std::unique_ptr<HcommUrmaResources>* resources,
                             std::uint8_t** hbm,
                             const std::string& stage) {
  (void)WriteMarker(options, "unsafe.device", "unsafe", stage);
  std::cerr << "KVAF_A5_HCCP_CHANNEL_TEARDOWN_UNPROVEN role=device stage="
            << stage << " action=retain-process-owned-resources\n";
  if (resources != nullptr) {
    (void)resources->release();
  }
  // Do not aclrtFree HBM after an unproven HCOMM close.  The retained process
  // keeps its owning allocation alive until an operator chooses recovery.
  if (hbm != nullptr) {
    *hbm = nullptr;
  }
  return {1, false, stage};
}

RunResult CloseHost(const Options& options,
                    std::unique_ptr<HcommUrmaResources>* resources,
                    std::unique_ptr<HostDdrDeviceMapping>* mapping,
                    const std::string& stage) {
  if (resources != nullptr && resources->get() != nullptr &&
      !CloseWithRetry("hcomm-resources", options.teardown_retries,
                      [&resources] { return (*resources)->Close(); })) {
    return RetainUnsafeHost(options, resources, mapping, stage);
  }
  if (mapping != nullptr && mapping->get() != nullptr &&
      !CloseWithRetry("host-ddr-mapping", options.teardown_retries,
                      [&mapping] { return (*mapping)->Close(); })) {
    return RetainUnsafeHost(options, resources, mapping, stage);
  }
  return {0, true, stage};
}

RunResult CloseDevice(const Options& options,
                      std::unique_ptr<HcommUrmaResources>* resources,
                      std::uint8_t** hbm, const std::string& stage) {
  if (resources != nullptr && resources->get() != nullptr &&
      !CloseWithRetry("hcomm-resources", options.teardown_retries,
                      [&resources] { return (*resources)->Close(); })) {
    return RetainUnsafeDevice(options, resources, hbm, stage);
  }
  if (hbm != nullptr && *hbm != nullptr) {
    if (!CheckAcl(aclrtFree(*hbm), "aclrtFree(Device HBM)")) {
      return RetainUnsafeDevice(options, resources, hbm, stage);
    }
    *hbm = nullptr;
  }
  std::cout << "KVAF_A5_HCCP_CHANNEL_TEARDOWN_OK subject=device-hbm"
               " attempts=1\n";
  return {0, true, stage};
}

RunResult FailHost(const Options& options,
                   std::unique_ptr<HcommUrmaResources>* resources,
                   std::unique_ptr<HostDdrDeviceMapping>* mapping,
                   bool must_wait_for_device_teardown,
                   const std::string& stage) {
  (void)WriteMarker(options, "fatal.host", "fatal", stage);
  if (must_wait_for_device_teardown) {
    // From the instant host.ddr_exported becomes visible, the Device is
    // allowed to consume the exported descriptor.  This includes a failed
    // Host HcommChannelCreate: HCOMM can return a partial channel handle on
    // error, and the Device may already own a server/import.  Therefore the
    // Host never guesses based on a local channel result.  It waits for the
    // Device's exact teardown proof before unregistering the Host DDR MR or
    // releasing its HAL mapping.
    (void)WriteMarker(options, "host.detach_requested", "detach-requested");
    const WaitResult detached =
        WaitForMarker(options, "device.teardown_ok", "teardown-ok", "device",
                      false);
    if (detached != WaitResult::kReady) {
      return RetainUnsafeHost(options, resources, mapping,
                              stage + ":device-" + WaitResultName(detached));
    }
  }
  RunResult result = CloseHost(options, resources, mapping, stage);
  if (result.safe_to_finalize) {
    (void)WriteMarker(options, "host.teardown_ok", "teardown-ok");
  }
  result.code = 1;
  return result;
}

RunResult FailDevice(const Options& options,
                     std::unique_ptr<HcommUrmaResources>* resources,
                     std::uint8_t** hbm, const std::string& stage) {
  (void)WriteMarker(options, "fatal.device", "fatal", stage);
  RunResult result = CloseDevice(options, resources, hbm, stage);
  if (result.safe_to_finalize) {
    (void)WriteMarker(options, "device.teardown_ok", "teardown-ok");
  }
  result.code = 1;
  return result;
}

RunResult RunHost(const Options& options) {
  auto mapping = std::make_unique<HostDdrDeviceMapping>();
  auto resources = std::make_unique<HcommUrmaResources>();
  // This is deliberately stronger than "channel was created".  Once the
  // descriptor marker is published, the Device is a potential user of the
  // Host DDR backing even when a later local API reports an error.
  bool host_ddr_exported = false;

  try {

  const bool supports_channel_name = resources->SupportsChannelName();
  if (!supports_channel_name && !options.allow_anonymous_channel_fallback) {
    return FailHost(options, &resources, &mapping, false,
                    "channel-name-unsupported");
  }
  if (!supports_channel_name && !HasOpenResourceGuardFd(options)) {
    return FailHost(options, &resources, &mapping, false,
                    "legacy-anonymous-guard-fd-absent");
  }
  const std::string channel_name = supports_channel_name ? options.run_id : "";
  std::cout << "KVAF_A5_HCCP_CHANNEL_MATCHING mode="
            << (supports_channel_name ? "named-run-id"
                                      : "legacy-anonymous-risk-ack")
            << " resource_guard_fd=" << options.resource_guard_fd
            << '\n';

  if (!CheckStatus(mapping->AllocateAndRegister(options.bytes,
                                                 options.logical_device_id),
                   "allocate and map A5 Host DDR")) {
    return FailHost(options, &resources, &mapping, false,
                    "host-ddr-map-failed");
  }
  std::cout << "KVAF_A5_HCCP_CHANNEL_DDR_REGISTERED bytes="
            << mapping->bytes() << '\n';

  std::uint32_t registration = 0;
  std::vector<std::uint8_t> descriptor;
  if (!CheckStatus(resources->OpenEndpoint(MakeHostEndpoint(options)),
                   "open HCCP Host endpoint") ||
      !CheckStatus(resources->RegisterMemory(
                       "kvaf_hccp_channel_host_ddr", mapping->host_address(),
                       mapping->bytes(), MemoryLocation::kHostDdr,
                       &registration),
                   "register HCCP Host DDR MR") ||
      !CheckStatus(resources->ExportMemory(registration, &descriptor),
                   "export HCCP Host DDR MR") ||
      descriptor.empty() ||
      !AtomicWrite(JoinPath(options.exchange_dir, "host_ddr.mr"), descriptor) ||
      !WriteMarker(options, "host.ddr_exported", "ddr-exported")) {
    return FailHost(options, &resources, &mapping, false,
                    "host-ddr-export-failed");
  }
  host_ddr_exported = true;
  std::cout << "KVAF_A5_HCCP_CHANNEL_MR_EXPORTED descriptor_bytes="
            << descriptor.size() << '\n';

  const WaitResult server_created =
      WaitForMarker(options, "device.channel_armed", "channel-armed", "device");
  if (server_created != WaitResult::kReady) {
    return FailHost(options, &resources, &mapping, host_ddr_exported,
                    "wait-device-server-" +
                        std::string(WaitResultName(server_created)));
  }

  if (!CheckStatus(resources->CreateChannel(
                       MakeDeviceEndpoint(options), ChannelEngine::kCpu,
                       ChannelRole::kClient, channel_name),
                   "create Host CPU channel")) {
    return FailHost(options, &resources, &mapping, host_ddr_exported,
                    "host-channel-create-failed");
  }
  if (!CheckStatus(resources->WaitChannelReady(options.timeout_seconds * 1000U),
                   "wait Host CPU channel")) {
    return FailHost(options, &resources, &mapping, host_ddr_exported,
                    "host-channel-ready-failed");
  }
  if (!WriteMarker(options, "host.channel_ready", "channel-ready")) {
    return FailHost(options, &resources, &mapping, host_ddr_exported,
                    "publish-host-ready-failed");
  }
  std::cout << "KVAF_A5_HCCP_CHANNEL_READY role=host engine=CPU\n";

  const WaitResult snapshot =
      WaitForMarker(options, "device.snapshot_ready", "snapshot-ready", "device");
  if (snapshot != WaitResult::kReady) {
    return FailHost(options, &resources, &mapping, host_ddr_exported,
                    "wait-device-snapshot-" +
                        std::string(WaitResultName(snapshot)));
  }
  if (!WriteMarker(options, "host.detach_requested", "detach-requested")) {
    return FailHost(options, &resources, &mapping, host_ddr_exported,
                    "publish-detach-request-failed");
  }

  // Device closes first, so its HCOMM import has been destroyed before Host
  // unregisters the exported DDR MR or tears down the HAL mapping.
  const WaitResult device_closed =
      WaitForMarker(options, "device.teardown_ok", "teardown-ok", "device",
                    false);
  if (device_closed != WaitResult::kReady) {
    return RetainUnsafeHost(options, &resources, &mapping,
                            "wait-device-teardown-" +
                                std::string(WaitResultName(device_closed)));
  }
  RunResult result = CloseHost(options, &resources, &mapping,
                               "host-normal-teardown");
  if (!result.safe_to_finalize) {
    return result;
  }
  if (!WriteMarker(options, "host.teardown_ok", "teardown-ok")) {
    return {1, true, "publish-host-teardown-failed"};
  }
  std::cout << "KVAF_A5_HCCP_AIV_CHANNEL_ZERO_WQE_OK role=host\n";
  return {0, true, "host-success"};
  } catch (const std::exception& error) {
    std::cerr << "unexpected Host channel-probe exception: " << error.what()
              << '\n';
    return FailHost(options, &resources, &mapping, host_ddr_exported,
                    "host-exception");
  } catch (...) {
    std::cerr << "unexpected non-standard Host channel-probe exception\n";
    return FailHost(options, &resources, &mapping, host_ddr_exported,
                    "host-unknown-exception");
  }
}

RunResult RunDevice(const Options& options) {
  auto resources = std::make_unique<HcommUrmaResources>();
  std::uint8_t* hbm = nullptr;

  try {

  const bool supports_channel_name = resources->SupportsChannelName();
  if (!supports_channel_name && !options.allow_anonymous_channel_fallback) {
    return FailDevice(options, &resources, &hbm,
                      "channel-name-unsupported");
  }
  if (!supports_channel_name && !HasOpenResourceGuardFd(options)) {
    return FailDevice(options, &resources, &hbm,
                      "legacy-anonymous-guard-fd-absent");
  }
  const std::string channel_name = supports_channel_name ? options.run_id : "";
  std::cout << "KVAF_A5_HCCP_CHANNEL_MATCHING mode="
            << (supports_channel_name ? "named-run-id"
                                      : "legacy-anonymous-risk-ack")
            << " resource_guard_fd=" << options.resource_guard_fd
            << '\n';

  const WaitResult exported =
      WaitForMarker(options, "host.ddr_exported", "ddr-exported", "host");
  if (exported != WaitResult::kReady) {
    return FailDevice(options, &resources, &hbm,
                      "wait-host-ddr-" +
                          std::string(WaitResultName(exported)));
  }
  std::vector<std::uint8_t> descriptor;
  if (!ReadFile(JoinPath(options.exchange_dir, "host_ddr.mr"), &descriptor) ||
      descriptor.empty()) {
    return FailDevice(options, &resources, &hbm, "read-host-ddr-failed");
  }

  if (!CheckAcl(aclrtMalloc(reinterpret_cast<void**>(&hbm), options.bytes,
                            ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(Device HBM)")) {
    return FailDevice(options, &resources, &hbm, "device-hbm-alloc-failed");
  }
  std::uint32_t registration = 0;
  if (!CheckStatus(resources->OpenEndpoint(MakeDeviceEndpoint(options)),
                   "open HCCP Device endpoint") ||
      !CheckStatus(resources->RegisterMemory(
                       "kvaf_hccp_channel_device_hbm",
                       reinterpret_cast<std::uint64_t>(hbm), options.bytes,
                       MemoryLocation::kDeviceHbm, &registration),
                   "register HCCP Device HBM MR")) {
    return FailDevice(options, &resources, &hbm, "device-mr-register-failed");
  }
  if (!CheckStatus(resources->CreateChannel(
                       MakeHostEndpoint(options), ChannelEngine::kAiv,
                       ChannelRole::kServer, channel_name),
                   "create Device AIV server channel")) {
    return FailDevice(options, &resources, &hbm, "device-channel-create-failed");
  }
  bool server_ready = false;
  if (!CheckStatus(resources->PollChannelReady(&server_ready),
                   "arm Device AIV server channel")) {
    return FailDevice(options, &resources, &hbm, "device-channel-arm-failed");
  }
  if (!WriteMarker(options, "device.channel_armed", "channel-armed")) {
    return FailDevice(options, &resources, &hbm, "publish-device-arm-failed");
  }
  std::cout << "KVAF_A5_HCCP_CHANNEL_ARMED role=device engine=AIV state="
            << (server_ready ? "READY" : "CONNECTING") << '\n';

  if (!CheckStatus(resources->WaitChannelReady(options.timeout_seconds * 1000U),
                   "wait Device AIV channel")) {
    return FailDevice(options, &resources, &hbm, "device-channel-ready-failed");
  }
  const WaitResult host_ready =
      WaitForMarker(options, "host.channel_ready", "channel-ready", "host");
  if (host_ready != WaitResult::kReady) {
    return FailDevice(options, &resources, &hbm,
                      "wait-host-ready-" +
                          std::string(WaitResultName(host_ready)));
  }

  ImportedMemoryView imported{};
  if (!CheckStatus(resources->ImportMemory(descriptor, MemoryLocation::kHostDdr,
                                           &imported),
                   "import Host DDR MR") ||
      imported.location != MemoryLocation::kHostDdr ||
      imported.address == 0 || imported.bytes < options.bytes) {
    return FailDevice(options, &resources, &hbm, "host-ddr-import-failed");
  }
  AivDirectUrmaChannel snapshot{};
  if (!CheckStatus(resources->SnapshotDirectChannel(
                       reinterpret_cast<std::uint64_t>(hbm), options.bytes,
                       imported.address, options.bytes, &snapshot),
                   "snapshot AIV UBC-CTP SQ/CQ")) {
    return FailDevice(options, &resources, &hbm, "channel-snapshot-failed");
  }
  if (snapshot.sq_depth == 0 || snapshot.cq_depth == 0 ||
      snapshot.wqe_size != 64 || snapshot.cqe_size != 64 ||
      snapshot.sq_head != 0 || snapshot.cq_tail != 0 ||
      snapshot.wqe_count != 0) {
    return FailDevice(options, &resources, &hbm,
                      "channel-snapshot-not-zero-wqe");
  }
  if (!WriteMarker(options, "device.snapshot_ready", "snapshot-ready")) {
    return FailDevice(options, &resources, &hbm, "publish-snapshot-failed");
  }
  // Deliberately do not log queue addresses or token values.  The public
  // result proves only compatible shape and zero outstanding work.
  std::cout << "KVAF_A5_HCCP_AIV_CHANNEL_SNAPSHOT_OK sq_depth="
            << snapshot.sq_depth << " cq_depth=" << snapshot.cq_depth
            << " wqe_size=" << snapshot.wqe_size
            << " cqe_size=" << snapshot.cqe_size
            << " sq_head=" << snapshot.sq_head
            << " cq_tail=" << snapshot.cq_tail
            << " wqe_count=" << snapshot.wqe_count << '\n';

  const WaitResult detach = WaitForMarker(
      options, "host.detach_requested", "detach-requested", "host");
  if (detach != WaitResult::kReady) {
    return FailDevice(options, &resources, &hbm,
                      "wait-host-detach-" +
                          std::string(WaitResultName(detach)));
  }
  RunResult result = CloseDevice(options, &resources, &hbm,
                                 "device-normal-teardown");
  if (!result.safe_to_finalize) {
    return result;
  }
  if (!WriteMarker(options, "device.teardown_ok", "teardown-ok")) {
    return {1, true, "publish-device-teardown-failed"};
  }
  std::cout << "KVAF_A5_HCCP_AIV_CHANNEL_ZERO_WQE_OK role=device\n";
  return {0, true, "device-success"};
  } catch (const std::exception& error) {
    std::cerr << "unexpected Device channel-probe exception: " << error.what()
              << '\n';
    return FailDevice(options, &resources, &hbm, "device-exception");
  } catch (...) {
    std::cerr << "unexpected non-standard Device channel-probe exception\n";
    return FailDevice(options, &resources, &hbm, "device-unknown-exception");
  }
}

bool WriteFinal(const Options& options, const char* outcome,
                const std::string& stage) {
  const std::string name = options.role == "host" ? "host.final" : "device.final";
  return WriteMarker(options, name.c_str(), outcome, stage);
}

// These failures happen before a worker can own HCOMM, MR, HBM, or Host-DDR
// state.  Publishing the role's teardown proof prevents its peer from
// conservatively retaining unrelated backing merely because ACL setup failed
// on one side before the two-process protocol began.
void WriteEarlySafeFailure(const Options& options, const std::string& stage) {
  const std::string fatal_name = "fatal." + options.role;
  const std::string teardown_name = options.role == "host"
                                        ? "host.teardown_ok"
                                        : "device.teardown_ok";
  (void)WriteMarker(options, fatal_name.c_str(), "fatal", stage);
  (void)WriteMarker(options, teardown_name.c_str(), "teardown-ok", stage);
  (void)WriteFinal(options, "FAIL_SAFE", stage);
}

[[noreturn]] void RetainUnsafeProcess(const Options& options,
                                      const std::string& stage) {
  std::cerr << "KVAF_A5_HCCP_AIV_CHANNEL_FINALIZE_SKIPPED role="
            << options.role << " reason=teardown-unproven stage=" << stage
            << "\nThe process intentionally remains alive; do not SIGTERM or "
               "SIGKILL it and do not start another probe on this NPU.\n";
  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

}  // namespace

int main(int argc, char** argv) {
  // This query deliberately constructs no endpoint, memory, channel, ACL
  // context, or device allocation.  The runner uses it to reject an old
  // public header before it starts either resource-owning worker.
  if (argc == 2 && std::string(argv[1]) == "--print-capabilities") {
    HcommUrmaResources capability_probe;
    std::cout << "KVAF_HCCP_CHANNEL_NAME_SUPPORT="
              << (capability_probe.SupportsChannelName() ? 1 : 0) << '\n';
    return 0;
  }

  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }

  // HCOMM v3's run-id channelName is the production matching mechanism.  An
  // old public header can only request HCOMM's documented anonymous mode.
  // Reject that downgrade before ACL initialization unless the supervisor has
  // explicitly selected the compatibility probe and passed an inherited
  // resource-guard descriptor.  This check intentionally precedes every ACL,
  // HCOMM, memory, and device operation.
  HcommUrmaResources capability_probe;
  const bool supports_channel_name = capability_probe.SupportsChannelName();
  if (!supports_channel_name && !options.allow_anonymous_channel_fallback) {
    WriteEarlySafeFailure(options, "channel-name-unsupported");
    return 1;
  }
  if (!supports_channel_name && !HasOpenResourceGuardFd(options)) {
    WriteEarlySafeFailure(options, "legacy-anonymous-guard-fd-absent");
    return 1;
  }

  // Validate the exact HCOMM descriptor used below before either worker owns
  // ACL, Host DDR, HBM, an endpoint, or an MR. In particular, old HCOMM
  // headers share HCCS QoS and UBC queue fields in a union; a compatibility
  // bug must be a preflight FAIL_SAFE result, never a retained-resource run.
  const Status descriptor_status = options.role == "host"
      ? HcommUrmaResources::ValidateChannelDescriptor(
            MakeDeviceEndpoint(options), ChannelRole::kClient)
      : HcommUrmaResources::ValidateChannelDescriptor(
            MakeHostEndpoint(options), ChannelRole::kServer);
  if (!CheckStatus(descriptor_status,
                   "validate HCOMM UBC channel descriptor")) {
    WriteEarlySafeFailure(options, "channel-descriptor-preflight-failed");
    return 1;
  }

  if (!CheckAcl(aclInit(nullptr), "aclInit")) {
    WriteEarlySafeFailure(options, "acl-init-failed");
    return 1;
  }
  ResourceSignalGuard signal_guard;
  if (!signal_guard.Install()) {
    std::cerr << "failed to install resource-owner signal guard\n";
    WriteEarlySafeFailure(options, "signal-guard-install-failed");
    (void)aclFinalize();
    return 1;
  }
  if (!CheckAcl(aclrtSetDevice(options.logical_device_id), "aclrtSetDevice")) {
    WriteEarlySafeFailure(options, "aclrt-set-device-failed");
    (void)aclFinalize();
    return 1;
  }

  std::cout << "KVAF_A5_HCCP_AIV_CHANNEL_PROBE_BEGIN role=" << options.role
            << " run_id=" << options.run_id
            << " logical_device_id=" << options.logical_device_id
            << " physical_device_id=" << options.physical_device_id
            << " bytes=" << options.bytes << " WQE_SUBMISSIONS=0\n";
  const RunResult result =
      options.role == "host" ? RunHost(options) : RunDevice(options);
  if (!result.safe_to_finalize) {
    RetainUnsafeProcess(options, result.stage);
  }

  // Close* has either published the role's teardown proof or returned an
  // unsafe retained-owner result above.  Runtime-only teardown therefore no
  // longer protects a DMA backing allocation; restore normal terminal signal
  // handling before potentially slow ACL context cleanup.
  signal_guard.Restore();

  if (!CheckAcl(aclrtResetDevice(options.logical_device_id),
                "aclrtResetDevice(clean process context)")) {
    (void)WriteFinal(options, "FAIL_SAFE", "aclrt-reset-failed");
    (void)aclFinalize();
    return 1;
  }
  if (!CheckAcl(aclFinalize(), "aclFinalize")) {
    (void)WriteFinal(options, "FAIL_SAFE", "acl-finalize-failed");
    return 1;
  }
  if (!WriteFinal(options, result.code == 0 ? "PASS" : "FAIL_SAFE",
                  result.stage)) {
    return 1;
  }
  return result.code;
}
