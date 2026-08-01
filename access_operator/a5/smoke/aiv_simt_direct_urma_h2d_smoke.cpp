#include "kvaf/access_operator/a5/aiv_simt_urma_h2d_launcher.h"
#include "kvaf/access_operator/a5/direct_urma_resources.h"
#include "kvaf/access_operator/a5/host_ddr_mapping.h"

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using kvaf::Status;
using kvaf::access::AivDirectUrmaChannel;
using kvaf::access::DeviceBatchGetParams;
using kvaf::access::DeviceBatchState;
using kvaf::access::DeviceBatchStatus;
using kvaf::access::DeviceCompletion;
using kvaf::access::TransferState;
using kvaf::access::UrmaH2DDescriptor;
using kvaf::access::a5::DirectUrmaExecutionDomain;
using kvaf::access::a5::DirectUrmaOpenSpec;
using kvaf::access::a5::DirectUrmaResources;
using kvaf::access::a5::DirectUrmaRole;
using kvaf::access::a5::HostDdrDeviceMapping;

constexpr std::uint8_t kCanary = 0xa5;
constexpr std::uint64_t kCanaryBytes = 64;
constexpr std::uint64_t kProviderPageBytes = 4096;
constexpr std::uint64_t kSmokeMaxTransferBytes = 64ULL * 1024ULL * 1024ULL;
constexpr auto kTerminalMarkerRetryInterval =
    std::chrono::milliseconds(100);
constexpr auto kTerminalMarkerWarningInterval =
    std::chrono::seconds(10);
constexpr const char* kFinalMarkerSchema =
    "kvaf-a5-direct-final-v1";

enum class DeviceTransferOutcome {
  kPassed,
  kFailedQuiesced,
  kUnsafeUnknown,
};

enum class EndpointOutcome {
  kPassed,
  kFailedSafe,
  kFailedUnsafe,
};

struct EndpointResult {
  EndpointOutcome outcome{EndpointOutcome::kFailedSafe};
  std::string reason{"unspecified"};
  bool resource_armed{false};
  bool stream_quiesced{true};
  bool queue_drained{true};
  bool provider_closed{true};
  bool backing_released{true};
  bool host_mapping_unregistered{true};
  bool allocations_released{true};
};

struct Options {
  std::string role;
  std::array<std::uint8_t, 16> host_eid{};
  std::array<std::uint8_t, 16> device_eid{};
  std::string host_urma_device;
  std::string device_urma_device;
  std::string provider_library;
  std::string provider_soname{"libkvaf_urma_provider_device.so"};
  std::string exchange_dir;
  std::string run_id;
  std::uint32_t host_eid_index{0};
  std::uint32_t device_eid_index{0};
  std::uint32_t logical_device_id{0};
  std::uint32_t control_port{21115};
  std::uint32_t timeout_seconds{120};
  std::uint64_t bytes{4096};
};

std::uint8_t Pattern(std::uint64_t index) {
  return static_cast<std::uint8_t>((index * 131U + 17U) & 0xffU);
}

bool ParseUnsigned(const std::string& text, std::uint64_t* value) {
  if (text.empty() || value == nullptr) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed =
      std::strtoull(text.c_str(), &end, 10);
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

bool ParseEid(const std::string& text,
              std::array<std::uint8_t, 16>* eid) {
  std::string compact;
  compact.reserve(32);
  for (const char value : text) {
    if (value != ':' && value != '-') {
      compact.push_back(value);
    }
  }
  if (compact.size() != 32 || eid == nullptr) {
    return false;
  }
  bool any_nonzero = false;
  for (std::size_t index = 0; index < eid->size(); ++index) {
    const int high = HexNibble(compact[index * 2]);
    const int low = HexNibble(compact[index * 2 + 1]);
    if (high < 0 || low < 0) {
      return false;
    }
    (*eid)[index] =
        static_cast<std::uint8_t>((high << 4) | low);
    any_nonzero = any_nonzero || (*eid)[index] != 0;
  }
  return any_nonzero;
}

std::string FormatIpv6Eid(
    const std::array<std::uint8_t, 16>& eid) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::size_t group = 0; group < 8; ++group) {
    if (group != 0) {
      output << ':';
    }
    const std::uint32_t word =
        (static_cast<std::uint32_t>(eid[group * 2]) << 8U) |
        eid[group * 2 + 1];
    output << std::setw(4) << word;
  }
  return output.str();
}

std::string FormatWqeEid(const AivDirectUrmaChannel& channel) {
  std::array<std::uint8_t, 16> bytes{};
  std::memcpy(bytes.data(), &channel.remote_eid_low,
              sizeof(channel.remote_eid_low));
  std::memcpy(bytes.data() + sizeof(channel.remote_eid_low),
              &channel.remote_eid_high,
              sizeof(channel.remote_eid_high));
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto value : bytes) {
    output << std::setw(2) << static_cast<unsigned>(value);
  }
  return output.str();
}

bool RegistrationBytes(const Options& options,
                       std::uint64_t* payload_bytes,
                       std::uint64_t* registration_bytes) {
  if (payload_bytes == nullptr || registration_bytes == nullptr ||
      options.bytes >
          std::numeric_limits<std::uint64_t>::max() - kCanaryBytes) {
    return false;
  }
  *payload_bytes = options.bytes + kCanaryBytes;
  std::uint64_t half =
      std::max(*payload_bytes, kProviderPageBytes);
  const std::uint64_t remainder = half % kProviderPageBytes;
  if (remainder != 0) {
    if (half >
        std::numeric_limits<std::uint64_t>::max() -
            (kProviderPageBytes - remainder)) {
      return false;
    }
    half += kProviderPageBytes - remainder;
  }
  if (half > std::numeric_limits<std::uint64_t>::max() / 2U) {
    return false;
  }
  *registration_bytes = half * 2U;
  return true;
}

void PrintUsage(const char* program) {
  std::cerr
      << "usage: " << program
      << " --role host|device --host-eid HEX32 --device-eid HEX32"
         " --host-urma-device NAME --device-urma-device NAME"
         " --host-eid-index N --device-eid-index N"
         " --provider-library ABS_PATH --provider-soname NAME"
         " --exchange-dir PATH --run-id HEX32_OR_LONGER"
         " [--logical-device-id N]"
         " [--control-port N] [--bytes N]"
         " [--timeout-seconds N]\n";
}

bool ParseOptions(int argc, char** argv, Options* options) {
  if (options == nullptr) {
    return false;
  }
  std::string host_eid;
  std::string device_eid;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      return false;
    }
    const std::string key = argv[index];
    const std::string value = argv[index + 1];
    std::uint64_t parsed = 0;
    if (key == "--role") {
      options->role = value;
    } else if (key == "--host-eid") {
      host_eid = value;
    } else if (key == "--device-eid") {
      device_eid = value;
    } else if (key == "--host-urma-device") {
      options->host_urma_device = value;
    } else if (key == "--device-urma-device") {
      options->device_urma_device = value;
    } else if (key == "--provider-library") {
      options->provider_library = value;
    } else if (key == "--provider-soname") {
      options->provider_soname = value;
    } else if (key == "--exchange-dir") {
      options->exchange_dir = value;
    } else if (key == "--run-id") {
      options->run_id = value;
    } else if (!ParseUnsigned(value, &parsed)) {
      return false;
    } else if (key == "--host-eid-index" &&
               parsed <=
                   std::numeric_limits<std::uint32_t>::max()) {
      options->host_eid_index =
          static_cast<std::uint32_t>(parsed);
    } else if (key == "--device-eid-index" &&
               parsed <=
                   std::numeric_limits<std::uint32_t>::max()) {
      options->device_eid_index =
          static_cast<std::uint32_t>(parsed);
    } else if (key == "--logical-device-id" &&
               parsed <=
                   std::numeric_limits<std::uint32_t>::max()) {
      options->logical_device_id =
          static_cast<std::uint32_t>(parsed);
    } else if (key == "--control-port" && parsed >= 1024 &&
               parsed <= 65535) {
      options->control_port = static_cast<std::uint32_t>(parsed);
    } else if (key == "--timeout-seconds" && parsed != 0 &&
               parsed <=
                   std::numeric_limits<std::uint32_t>::max() / 1000U) {
      options->timeout_seconds =
          static_cast<std::uint32_t>(parsed);
    } else if (key == "--bytes" && parsed != 0 &&
               parsed <= kSmokeMaxTransferBytes) {
      options->bytes = parsed;
    } else {
      return false;
    }
  }
  const bool valid_run_id =
      options->run_id.size() >= 32 &&
      options->run_id.size() <= 128 &&
      std::all_of(options->run_id.begin(), options->run_id.end(),
                  [](unsigned char value) {
                    return std::isxdigit(value) != 0;
                  });
  return (options->role == "host" ||
          options->role == "device") &&
         !options->host_urma_device.empty() &&
         !options->device_urma_device.empty() &&
         !options->provider_library.empty() &&
         options->provider_library.front() == '/' &&
         !options->provider_soname.empty() &&
         options->provider_soname.find('/') == std::string::npos &&
         !options->exchange_dir.empty() &&
         valid_run_id &&
         ParseEid(host_eid, &options->host_eid) &&
         ParseEid(device_eid, &options->device_eid) &&
         options->host_eid != options->device_eid;
}

std::string JoinPath(const std::string& directory,
                     const char* name) {
  return directory + "/" + name;
}

bool AtomicWriteText(const std::string& path,
                     const std::string& text) {
  const std::string temporary =
      path + ".tmp." +
      std::to_string(static_cast<long long>(::getpid()));
  const int descriptor = ::open(
      temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
      S_IRUSR | S_IWUSR);
  if (descriptor < 0) {
    return false;
  }
  std::size_t written = 0;
  bool ok = true;
  while (written < text.size()) {
    const auto result = ::write(
        descriptor, text.data() + written, text.size() - written);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      ok = false;
      break;
    }
    written += static_cast<std::size_t>(result);
  }
  if (ok && ::fsync(descriptor) != 0) {
    ok = false;
  }
  if (::close(descriptor) != 0) {
    ok = false;
  }
  if (!ok) {
    (void)::unlink(temporary.c_str());
    return false;
  }
  if (std::rename(temporary.c_str(), path.c_str()) != 0) {
    (void)::unlink(temporary.c_str());
    return false;
  }

  const auto separator = path.find_last_of('/');
  const std::string directory =
      separator == std::string::npos
          ? "."
          : (separator == 0 ? "/" : path.substr(0, separator));
  const int directory_descriptor =
      ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory_descriptor < 0) {
    return false;
  }
  const bool directory_synced = ::fsync(directory_descriptor) == 0;
  const bool directory_closed = ::close(directory_descriptor) == 0;
  if (!directory_synced || !directory_closed) {
    return false;
  }
  return true;
}

const char* OutcomeName(EndpointOutcome outcome) {
  switch (outcome) {
    case EndpointOutcome::kPassed:
      return "PASS";
    case EndpointOutcome::kFailedSafe:
      return "FAIL_SAFE";
    case EndpointOutcome::kFailedUnsafe:
      return "FAIL_UNSAFE";
  }
  return "FAIL_UNSAFE";
}

EndpointResult PassedEndpoint() {
  EndpointResult result;
  result.outcome = EndpointOutcome::kPassed;
  result.reason = "ok";
  result.resource_armed = true;
  return result;
}

EndpointResult SafeEndpoint(std::string reason,
                            bool resource_armed = false,
                            bool backing_released = true,
                            bool host_mapping_unregistered = true,
                            bool allocations_released = true) {
  EndpointResult result;
  result.outcome = EndpointOutcome::kFailedSafe;
  result.reason = std::move(reason);
  result.resource_armed = resource_armed;
  result.backing_released = backing_released;
  result.host_mapping_unregistered = host_mapping_unregistered;
  result.allocations_released = allocations_released;
  return result;
}

EndpointResult UnsafeEndpoint(std::string reason,
                              bool resource_armed = true,
                              bool stream_quiesced = false,
                              bool queue_drained = false,
                              bool provider_closed = false,
                              bool backing_released = false) {
  EndpointResult result;
  result.outcome = EndpointOutcome::kFailedUnsafe;
  result.reason = std::move(reason);
  result.resource_armed = resource_armed;
  result.stream_quiesced = stream_quiesced;
  result.queue_drained = queue_drained;
  result.provider_closed = provider_closed;
  result.backing_released = backing_released;
  result.host_mapping_unregistered = backing_released;
  result.allocations_released = backing_released;
  return result;
}

std::string ControlMarker(const Options& options, const char* key,
                          const char* value) {
  return "schema=kvaf-a5-direct-control-v1\nrun_id=" +
         options.run_id + "\n" + key + "=" + value + "\n";
}

bool WriteTerminalMarker(const Options& options,
                         const EndpointResult& result) {
  std::ostringstream marker;
  marker << "schema=" << kFinalMarkerSchema << '\n'
         << "run_id=" << options.run_id << '\n'
         << "role=" << options.role << '\n'
         << "pid=" << static_cast<long long>(::getpid()) << '\n'
         << "outcome=" << OutcomeName(result.outcome) << '\n'
         << "reason=" << result.reason << '\n'
         << "resource_armed=" << (result.resource_armed ? 1 : 0) << '\n'
         << "stream_quiesced=" << (result.stream_quiesced ? 1 : 0) << '\n'
         << "queue_drained=" << (result.queue_drained ? 1 : 0) << '\n'
         << "provider_closed=" << (result.provider_closed ? 1 : 0) << '\n'
         << "backing_released=" << (result.backing_released ? 1 : 0) << '\n'
         << "host_mapping_unregistered="
         << (options.role != "host" ||
                     result.host_mapping_unregistered
                 ? 1
                 : 0)
         << '\n'
         << "allocations_released="
         << (options.role != "device" || result.allocations_released ? 1 : 0)
         << '\n';
  return AtomicWriteText(
      JoinPath(options.exchange_dir,
               options.role == "host" ? "host.final" : "device.final"),
      marker.str());
}

bool PublishDeviceSafeAbortWakeEdge(const Options& options,
                                    const EndpointResult& result) {
  if (options.role != "device" ||
      result.outcome != EndpointOutcome::kFailedSafe) {
    return false;
  }
  // The caller must persist device.final before publishing this optional
  // wake-up edge. Host validates the full nonce-bound final marker and never
  // trusts this short marker as teardown proof.
  return AtomicWriteText(
      JoinPath(options.exchange_dir, "device.safe_abort"),
      ControlMarker(options, "endpoint", "FAIL_SAFE"));
}

void IgnoreLifecycleSignals() {
  // Once this process can own registered DDR/HBM, asynchronous termination
  // must never bypass the checked provider teardown. The runner remains
  // interruptible and records quarantine; these workers keep ownership until
  // they reach a terminal marker or explicit unsafe quarantine.
  (void)std::signal(SIGHUP, SIG_IGN);
  (void)std::signal(SIGINT, SIG_IGN);
  (void)std::signal(SIGTERM, SIG_IGN);
  (void)std::signal(SIGQUIT, SIG_IGN);
  (void)std::signal(SIGPIPE, SIG_IGN);
}

void PersistTerminalMarkerUntilDurable(
    const Options& options, const EndpointResult& result,
    const char* description) {
  // Once transport safety is known, a temporary /data failure must not turn
  // that fact into a permanently UNPROVEN run by letting the worker exit.
  // Keep the inherited active-session lock and retry at a bounded cadence;
  // after directory fsync succeeds, the worker may finish and a later runner
  // invocation can recover the nonce-bound terminal state automatically.
  IgnoreLifecycleSignals();
  bool warning_written = false;
  auto next_warning = std::chrono::steady_clock::now();
  for (;;) {
    if (WriteTerminalMarker(options, result)) {
      if (warning_written) {
        std::cerr << "durable terminal marker publication recovered"
                  << " role=" << options.role
                  << " stage=" << description << '\n';
      }
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!warning_written || now >= next_warning) {
      std::cerr
          << "failed to durably publish terminal marker"
          << " role=" << options.role
          << " stage=" << description
          << "; transport is safe, worker remains live and will retry\n";
      warning_written = true;
      next_warning = now + kTerminalMarkerWarningInterval;
    }
    std::this_thread::sleep_for(kTerminalMarkerRetryInterval);
  }
}

void PublishDeviceSafeAbortWakeEdgeBestEffort(
    const Options& options, const EndpointResult& result) {
  if (options.role == "device" &&
      result.outcome == EndpointOutcome::kFailedSafe &&
      !PublishDeviceSafeAbortWakeEdge(options, result)) {
    std::cerr
        << "failed to publish optional Device safe-abort wake edge; "
           "durable device.final remains authoritative\n";
  }
}

bool ReadText(const std::string& path, std::string* text) {
  if (text == nullptr) {
    return false;
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  *text = contents.str();
  return static_cast<bool>(input) || input.eof();
}

bool ReadExactMarker(const std::string& path,
                     const char* expected) {
  std::string text;
  return expected != nullptr && ReadText(path, &text) &&
         text == expected;
}

bool UniqueMarkerField(const std::string& text, const char* key,
                       std::string* value) {
  if (key == nullptr || value == nullptr) {
    return false;
  }
  std::size_t count = 0;
  std::size_t cursor = 0;
  const std::string prefix = std::string(key) + "=";
  while (cursor < text.size()) {
    const auto end = text.find('\n', cursor);
    const auto length =
        (end == std::string::npos ? text.size() : end) - cursor;
    if (length > prefix.size() &&
        text.compare(cursor, prefix.size(), prefix) == 0) {
      *value = text.substr(cursor + prefix.size(),
                           length - prefix.size());
      ++count;
    }
    if (end == std::string::npos) {
      break;
    }
    cursor = end + 1;
  }
  return count == 1 && !value->empty();
}

bool MarkerFieldEquals(const std::string& text, const char* key,
                       const char* expected) {
  std::string value;
  return expected != nullptr && UniqueMarkerField(text, key, &value) &&
         value == expected;
}

bool IsCanonicalPositivePid(const std::string& value) {
  if (value.empty() || value.front() == '0') {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](const char character) {
    return character >= '0' && character <= '9';
  });
}

bool ValidPeerFatal(const Options& options, const std::string& path) {
  std::string marker;
  std::string pid;
  std::string reason;
  return ReadText(path, &marker) &&
         MarkerFieldEquals(marker, "schema",
                           "kvaf-a5-direct-fatal-v1") &&
         MarkerFieldEquals(marker, "run_id", options.run_id.c_str()) &&
         UniqueMarkerField(marker, "pid", &pid) &&
         IsCanonicalPositivePid(pid) &&
         UniqueMarkerField(marker, "reason", &reason);
}

bool ValidDeviceSafeAbortProof(const Options& options) {
  std::string marker;
  std::string resource_armed;
  std::string backing_released;
  std::string allocations_released;
  const auto final_path = JoinPath(options.exchange_dir, "device.final");
  return ReadText(final_path, &marker) &&
         MarkerFieldEquals(marker, "schema", kFinalMarkerSchema) &&
         MarkerFieldEquals(marker, "run_id", options.run_id.c_str()) &&
         MarkerFieldEquals(marker, "role", "device") &&
         MarkerFieldEquals(marker, "outcome", "FAIL_SAFE") &&
         MarkerFieldEquals(marker, "stream_quiesced", "1") &&
         MarkerFieldEquals(marker, "queue_drained", "1") &&
         MarkerFieldEquals(marker, "provider_closed", "1") &&
         MarkerFieldEquals(marker, "host_mapping_unregistered", "1") &&
         UniqueMarkerField(marker, "resource_armed", &resource_armed) &&
         (resource_armed == "0" || resource_armed == "1") &&
         UniqueMarkerField(marker, "backing_released", &backing_released) &&
         (backing_released == "0" || backing_released == "1") &&
         UniqueMarkerField(marker, "allocations_released",
                           &allocations_released) &&
         (allocations_released == "0" || allocations_released == "1");
}

enum class PeerMarkerWait {
  kFound,
  kPeerFatal,
  kPeerSafeAbort,
};

PeerMarkerWait WaitForPeerEvidence(
    const Options& options, const std::string& path,
    const char* expected_first, const char* expected_second,
    const std::string& peer_fatal, const char* description) {
  auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::seconds(options.timeout_seconds);
  for (;;) {
    if (ReadExactMarker(path, expected_first) ||
        (expected_second != nullptr &&
         ReadExactMarker(path, expected_second))) {
      return PeerMarkerWait::kFound;
    }
    if (ValidPeerFatal(options, peer_fatal)) {
      return PeerMarkerWait::kPeerFatal;
    }
    // The full nonce-bound final marker is the safety proof.  The short
    // device.safe_abort file is only a wake-up edge and is deliberately not
    // trusted on its own (nor required if writing that edge failed).
    if (ValidDeviceSafeAbortProof(options)) {
      return PeerMarkerWait::kPeerSafeAbort;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      std::cerr << "still waiting for valid " << description
                << "; ownership remains live and late evidence will be "
                   "accepted\n";
      deadline = std::chrono::steady_clock::now() +
                 std::chrono::seconds(options.timeout_seconds);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

[[noreturn]] void QuarantineUnsafeTeardown(
    const Options& options, const EndpointResult& result) {
  IgnoreLifecycleSignals();

  const auto pid = static_cast<long long>(::getpid());
  const std::string marker = JoinPath(
      options.exchange_dir,
      options.role == "host"
          ? "fatal.host"
          : "fatal.device");
  const bool marker_written = AtomicWriteText(
      marker, "schema=kvaf-a5-direct-fatal-v1\nrun_id=" +
                  options.run_id + "\npid=" + std::to_string(pid) +
                  "\nreason=" + result.reason + "\n");
  const bool final_written = WriteTerminalMarker(options, result);
  std::cerr
      << "KVAF_A5_DIRECT_QUARANTINED"
      << " role=" << options.role
      << " pid=" << pid
      << " marker_written=" << (marker_written ? 1 : 0)
      << " final_written=" << (final_written ? 1 : 0)
      << "; backing memory remains owned by this live process. "
         "Do not kill it until the selected NPU has been reset or "
         "otherwise hardware-isolated.\n";
  std::cout.flush();
  std::cerr.flush();
  std::fflush(nullptr);
  for (;;) {
    (void)::pause();
  }
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
  std::cerr << operation << " failed: "
            << status.message() << '\n';
  return false;
}

bool CloseProviderWithRetry(DirectUrmaResources* resources,
                            const char* operation,
                            bool* local_cleanup_complete = nullptr) {
  if (local_cleanup_complete != nullptr) {
    *local_cleanup_complete = false;
  }
  auto status = resources->Close();
  if (status.ok()) {
    if (local_cleanup_complete != nullptr) {
      *local_cleanup_complete = true;
    }
    return true;
  }
  std::cerr << operation << " failed: " << status.message()
            << "; retrying once without releasing backing memory\n";
  status = resources->Close();
  if (status.ok()) {
    if (local_cleanup_complete != nullptr) {
      *local_cleanup_complete = true;
    }
    return true;
  }
  if (resources->transport_teardown_complete()) {
    std::cerr << operation << " transport is closed, but local control "
              << "cleanup remains incomplete: " << status.message() << '\n';
    return true;
  }
  std::cerr << operation << " retry failed: " << status.message()
            << "; backing memory will be retained until process exit\n";
  return false;
}

DirectUrmaOpenSpec MakeHostOpenSpec(
    const Options& options, std::uint64_t mapped_device_address,
    std::uint64_t registration_bytes,
    std::uint64_t payload_bytes) {
  DirectUrmaOpenSpec spec{};
  spec.execution_domain =
      DirectUrmaExecutionDomain::kHostProcess;
  spec.role = DirectUrmaRole::kServer;
  spec.provider_library_path = options.provider_library;
  spec.provider_soname = options.provider_soname;
  spec.urma_device = options.host_urma_device;
  spec.local_eid = FormatIpv6Eid(options.host_eid);
  spec.peer_eid = FormatIpv6Eid(options.device_eid);
  spec.eid_index = options.host_eid_index;
  spec.control_port = options.control_port;
  spec.control_timeout_seconds = options.timeout_seconds;
  // URMA must register the NPU-visible DVA. The Host fills the same pages
  // through the HVA retained by HostDdrDeviceMapping.
  spec.local_address = mapped_device_address;
  spec.local_allocation_bytes = registration_bytes;
  spec.segment_payload_bytes = payload_bytes;
  return spec;
}

DirectUrmaOpenSpec MakeDeviceOpenSpec(
    const Options& options, std::uint64_t output_address,
    std::uint64_t registration_bytes,
    std::uint64_t payload_bytes) {
  DirectUrmaOpenSpec spec{};
  spec.execution_domain = DirectUrmaExecutionDomain::kAicpu;
  spec.role = DirectUrmaRole::kClient;
  spec.provider_library_path = options.provider_library;
  spec.provider_soname = options.provider_soname;
  spec.urma_device = options.device_urma_device;
  spec.local_eid = FormatIpv6Eid(options.device_eid);
  spec.peer_eid = FormatIpv6Eid(options.host_eid);
  spec.eid_index = options.device_eid_index;
  spec.control_port = options.control_port;
  spec.control_timeout_seconds = options.timeout_seconds;
  spec.local_address = output_address;
  spec.local_allocation_bytes = registration_bytes;
  spec.segment_payload_bytes = payload_bytes;
  return spec;
}

template <typename T>
bool AllocateDevice(std::size_t count, T** pointer) {
  return CheckAcl(
      aclrtMalloc(reinterpret_cast<void**>(pointer),
                  count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST),
      "aclrtMalloc");
}

template <typename T>
bool CopyHostToDevice(T* destination, const T* source,
                      std::size_t count) {
  return CheckAcl(
      aclrtMemcpy(destination, count * sizeof(T), source,
                  count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE),
      "aclrtMemcpy(H2D)");
}

template <typename T>
bool CopyDeviceToHost(T* destination, const T* source,
                      std::size_t count) {
  return CheckAcl(
      aclrtMemcpy(destination, count * sizeof(T), source,
                  count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST),
      "aclrtMemcpy(D2H)");
}

struct DeviceAllocations {
  std::uint8_t* output{nullptr};
  std::uint64_t* remote_addresses{nullptr};
  std::uint64_t* lengths{nullptr};
  std::uint8_t* valid{nullptr};
  std::uint32_t* indices{nullptr};
  std::uint8_t* mask{nullptr};
  std::uint64_t* output_offsets{nullptr};
  UrmaH2DDescriptor* descriptors{nullptr};
  DeviceBatchGetParams* params{nullptr};
  DeviceCompletion* completion{nullptr};
  AivDirectUrmaChannel* direct_channel{nullptr};

  ~DeviceAllocations() {
    (void)Close();
  }

  bool Close() {
    bool ok = true;
    auto release = [&ok](auto*& pointer) {
      if (pointer == nullptr) {
        return;
      }
      if (aclrtFree(pointer) != ACL_SUCCESS) {
        ok = false;
        return;
      }
      pointer = nullptr;
    };
    release(direct_channel);
    release(completion);
    release(params);
    release(descriptors);
    release(output_offsets);
    release(mask);
    release(indices);
    release(valid);
    release(lengths);
    release(remote_addresses);
    release(output);
    return ok;
  }
};

struct RetainedEndpointState {
  std::unique_ptr<DirectUrmaResources> resources;
  std::unique_ptr<HostDdrDeviceMapping> mapping;
  std::unique_ptr<DeviceAllocations> allocations;
  aclrtStream transfer_stream{nullptr};
  bool transfer_stream_quiesced{true};
};

RetainedEndpointState g_retained_endpoint;
std::atomic<bool> g_endpoint_may_own_resources{false};

[[noreturn]] void FailClosedTerminate() noexcept {
  static constexpr char kMessage[] =
      "KVAF_A5_DIRECT_TERMINATE_WITH_OWNERSHIP; retaining process and "
      "backing memory\n";
  if (g_endpoint_may_own_resources.load(std::memory_order_relaxed)) {
    (void)::write(STDERR_FILENO, kMessage, sizeof(kMessage) - 1);
    (void)::signal(SIGHUP, SIG_IGN);
    (void)::signal(SIGINT, SIG_IGN);
    (void)::signal(SIGTERM, SIG_IGN);
    (void)::signal(SIGQUIT, SIG_IGN);
    for (;;) {
      (void)::pause();
    }
  }
  _exit(134);
}

template <typename T>
void RetainUniqueOnFailure(std::unique_ptr<T>* local,
                           std::unique_ptr<T>* retained) noexcept {
  if (local == nullptr || retained == nullptr || *local == nullptr) {
    return;
  }
  if (*retained == nullptr) {
    retained->swap(*local);
    return;
  }
  // This should be unreachable (one endpoint per worker).  Leaking is still
  // safer than invoking an independent destructor over DMA-reachable state.
  (void)local->release();
}

void RetainEndpointOwnership(
    std::unique_ptr<DirectUrmaResources>* resources,
    std::unique_ptr<HostDdrDeviceMapping>* mapping,
    std::unique_ptr<DeviceAllocations>* allocations,
    aclrtStream* transfer_stream = nullptr,
    const bool* transfer_stream_quiesced = nullptr) noexcept {
  RetainUniqueOnFailure(resources, &g_retained_endpoint.resources);
  RetainUniqueOnFailure(mapping, &g_retained_endpoint.mapping);
  RetainUniqueOnFailure(allocations, &g_retained_endpoint.allocations);
  if (transfer_stream != nullptr && *transfer_stream != nullptr) {
    if (g_retained_endpoint.transfer_stream == nullptr) {
      g_retained_endpoint.transfer_stream = *transfer_stream;
      g_retained_endpoint.transfer_stream_quiesced =
          transfer_stream_quiesced != nullptr &&
          *transfer_stream_quiesced;
    } else if (transfer_stream_quiesced != nullptr) {
      g_retained_endpoint.transfer_stream_quiesced =
          g_retained_endpoint.transfer_stream_quiesced &&
          *transfer_stream_quiesced;
    }
    *transfer_stream = nullptr;
  }
}

class RetainEndpointOnException {
 public:
  RetainEndpointOnException(
      std::unique_ptr<DirectUrmaResources>* resources,
      std::unique_ptr<HostDdrDeviceMapping>* mapping,
      std::unique_ptr<DeviceAllocations>* allocations,
      aclrtStream* transfer_stream = nullptr,
      const bool* transfer_stream_quiesced = nullptr) noexcept
      : resources_(resources),
        mapping_(mapping),
        allocations_(allocations),
        transfer_stream_(transfer_stream),
        transfer_stream_quiesced_(transfer_stream_quiesced),
        uncaught_exceptions_(std::uncaught_exceptions()) {}

  ~RetainEndpointOnException() noexcept {
    if (std::uncaught_exceptions() > uncaught_exceptions_) {
      RetainEndpointOwnership(resources_, mapping_, allocations_,
                              transfer_stream_,
                              transfer_stream_quiesced_);
    }
  }

 private:
  std::unique_ptr<DirectUrmaResources>* resources_;
  std::unique_ptr<HostDdrDeviceMapping>* mapping_;
  std::unique_ptr<DeviceAllocations>* allocations_;
  aclrtStream* transfer_stream_;
  const bool* transfer_stream_quiesced_;
  int uncaught_exceptions_;
};

bool CloseMappingWithRetry(HostDdrDeviceMapping* mapping,
                           const char* operation,
                           bool* local_cleanup_complete = nullptr) {
  if (local_cleanup_complete != nullptr) {
    *local_cleanup_complete = false;
  }
  auto status = mapping->Close();
  if (status.ok()) {
    if (local_cleanup_complete != nullptr) {
      *local_cleanup_complete = true;
    }
    return true;
  }
  std::cerr << operation << " failed: " << status.message()
            << "; retrying once without releasing ownership\n";
  status = mapping->Close();
  if (status.ok()) {
    if (local_cleanup_complete != nullptr) {
      *local_cleanup_complete = true;
    }
    return true;
  }
  if (mapping->device_mapping_removed()) {
    std::cerr << operation << " removed the NPU mapping, but local cleanup "
              << "remains incomplete: " << status.message() << '\n';
    return true;
  }
  std::cerr << operation << " retry failed: " << status.message()
            << "; mapping ownership will be retained\n";
  return false;
}

bool CloseAllocationsWithRetry(DeviceAllocations* allocations) {
  if (allocations->Close()) {
    return true;
  }
  std::cerr << "one or more aclrtFree calls failed; retrying once\n";
  if (allocations->Close()) {
    return true;
  }
  std::cerr << "aclrtFree retry failed; allocation handles retained\n";
  return false;
}

bool AllocateDeviceInputs(
    std::uint64_t registration_bytes, std::size_t table_entries,
    std::size_t request_entries,
    DeviceAllocations* allocations) {
  return AllocateDevice(registration_bytes, &allocations->output) &&
         AllocateDevice(table_entries,
                        &allocations->remote_addresses) &&
         AllocateDevice(table_entries, &allocations->lengths) &&
         AllocateDevice(table_entries, &allocations->valid) &&
         AllocateDevice(request_entries, &allocations->indices) &&
         AllocateDevice(request_entries, &allocations->mask) &&
         AllocateDevice(request_entries,
                        &allocations->output_offsets) &&
         AllocateDevice(request_entries,
                        &allocations->descriptors) &&
         AllocateDevice(1, &allocations->params) &&
         AllocateDevice(1, &allocations->completion) &&
         AllocateDevice(1, &allocations->direct_channel);
}

DeviceTransferOutcome RunDeviceTransfer(
    const Options& options, std::uint64_t payload_bytes,
    DirectUrmaResources* resources,
    DeviceAllocations* allocations,
    aclrtStream* retained_stream, bool* stream_quiesced) {
  if (retained_stream != nullptr) {
    *retained_stream = nullptr;
  }
  if (stream_quiesced != nullptr) {
    *stream_quiesced = true;
  }
  const auto local = resources->local_memory();
  const auto remote = resources->remote_memory();
  if (local.address !=
          reinterpret_cast<std::uint64_t>(allocations->output) ||
      local.bytes < payload_bytes || remote.address == 0 ||
      remote.bytes < options.bytes) {
    std::cerr << "provider returned an unexpected local/remote MR\n";
    return DeviceTransferOutcome::kFailedQuiesced;
  }

  AivDirectUrmaChannel direct_channel{};
  if (!CheckStatus(
          resources->SnapshotDirectChannel(
              reinterpret_cast<std::uint64_t>(allocations->output),
              payload_bytes, remote.address, options.bytes,
              &direct_channel),
          "snapshot direct liburma AIV channel")) {
    return DeviceTransferOutcome::kFailedQuiesced;
  }
  std::cout
      << "KVAF_A5_DIRECT_URMA_CHANNEL"
      << " sq_depth=" << direct_channel.sq_depth
      << " cq_depth=" << direct_channel.cq_depth
      << " tp_id=" << direct_channel.tp_id
      << " token_policy="
      << ((direct_channel.flags &
           kvaf::access::kAivUrmaChannelTokenEnabled) != 0
              ? "enabled"
              : "none")
      << " remote_eid_wqe=" << FormatWqeEid(direct_channel)
      << '\n';

  constexpr std::size_t kActiveDescriptorTarget = 4;
  const std::size_t active_descriptor_count =
      static_cast<std::size_t>(std::min<std::uint64_t>(
          options.bytes, kActiveDescriptorTarget));
  const std::size_t request_count =
      active_descriptor_count + 1;

  std::vector<std::uint64_t> remote_addresses(
      active_descriptor_count);
  std::vector<std::uint64_t> lengths(
      active_descriptor_count);
  std::vector<std::uint64_t> table_output_offsets(
      active_descriptor_count);
  std::vector<std::uint8_t> valid(
      active_descriptor_count, 1);
  std::uint64_t consumed = 0;
  for (std::size_t table_index = 0;
       table_index < active_descriptor_count; ++table_index) {
    const std::uint64_t entries_left =
        active_descriptor_count - table_index;
    const std::uint64_t segment_bytes =
        (options.bytes - consumed) / entries_left;
    remote_addresses[table_index] =
        remote.address + consumed;
    lengths[table_index] = segment_bytes;
    table_output_offsets[table_index] = consumed;
    consumed += segment_bytes;
  }

  std::vector<std::uint32_t> indices(request_count);
  std::vector<std::uint8_t> mask(request_count, 1);
  std::vector<std::uint64_t> output_offsets(request_count);
  for (std::size_t request = 0;
       request < active_descriptor_count; ++request) {
    const std::size_t table_index =
        active_descriptor_count - request - 1;
    indices[request] = static_cast<std::uint32_t>(table_index);
    output_offsets[request] =
        table_output_offsets[table_index];
  }
  indices.back() = 0;
  mask.back() = 0;
  output_offsets.back() =
      std::numeric_limits<std::uint64_t>::max();

  const std::vector<UrmaH2DDescriptor> empty_descriptors(
      request_count);
  const DeviceCompletion empty_completion{};
  DeviceBatchGetParams params{};
  params.table.remote_addresses =
      reinterpret_cast<std::uint64_t>(
          allocations->remote_addresses);
  params.table.lengths =
      reinterpret_cast<std::uint64_t>(allocations->lengths);
  params.table.valid =
      reinterpret_cast<std::uint64_t>(allocations->valid);
  params.table.capacity =
      static_cast<std::uint32_t>(active_descriptor_count);
  params.table.generation = 1;
  params.indices =
      reinterpret_cast<std::uint64_t>(allocations->indices);
  params.mask =
      reinterpret_cast<std::uint64_t>(allocations->mask);
  params.output_offsets =
      reinterpret_cast<std::uint64_t>(
          allocations->output_offsets);
  params.descriptors =
      reinterpret_cast<std::uint64_t>(
          allocations->descriptors);
  params.output_base =
      reinterpret_cast<std::uint64_t>(allocations->output);
  params.output_bytes = options.bytes;
  params.entry_count =
      static_cast<std::uint32_t>(request_count);
  params.descriptor_capacity =
      static_cast<std::uint32_t>(request_count);
  params.generation = 1;

  if (!CopyHostToDevice(allocations->remote_addresses,
                        remote_addresses.data(),
                        active_descriptor_count) ||
      !CopyHostToDevice(allocations->lengths, lengths.data(),
                        active_descriptor_count) ||
      !CopyHostToDevice(allocations->valid, valid.data(),
                        active_descriptor_count) ||
      !CopyHostToDevice(allocations->indices, indices.data(),
                        request_count) ||
      !CopyHostToDevice(allocations->mask, mask.data(),
                        request_count) ||
      !CopyHostToDevice(allocations->output_offsets,
                        output_offsets.data(), request_count) ||
      !CopyHostToDevice(allocations->descriptors,
                        empty_descriptors.data(),
                        request_count) ||
      !CopyHostToDevice(allocations->completion,
                        &empty_completion, 1) ||
      !CopyHostToDevice(allocations->params, &params, 1) ||
      !CopyHostToDevice(allocations->direct_channel,
                        &direct_channel, 1)) {
    return DeviceTransferOutcome::kFailedQuiesced;
  }

  aclrtStream stream = nullptr;
  if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) {
    return DeviceTransferOutcome::kFailedQuiesced;
  }
  if (retained_stream != nullptr) {
    *retained_stream = stream;
  }
  if (stream_quiesced != nullptr) {
    *stream_quiesced = false;
  }
  kvaf::access::a5::LaunchAivSimtUrmaH2D(
      stream, allocations->params, allocations->completion,
      allocations->direct_channel);
  const bool synchronized =
      CheckAcl(aclrtSynchronizeStream(stream),
               "aclrtSynchronizeStream");
  if (!synchronized) {
    std::cerr
        << "AIV stream quiescence is unknown; retaining stream, "
           "provider, and backing allocations for device reset\n";
    return DeviceTransferOutcome::kUnsafeUnknown;
  }
  if (stream_quiesced != nullptr) {
    *stream_quiesced = true;
  }
  const bool stream_destroyed =
      CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");
  if (!stream_destroyed && retained_stream != nullptr) {
    *retained_stream = stream;
  } else if (stream_destroyed && retained_stream != nullptr) {
    *retained_stream = nullptr;
  }

  DeviceCompletion completion{};
  AivDirectUrmaChannel completed_channel{};
  AivDirectUrmaChannel provider_channel_after{};
  std::vector<UrmaH2DDescriptor> completed_descriptors(
      request_count);
  std::vector<std::uint8_t> output(payload_bytes);
  if (!CopyDeviceToHost(&completion, allocations->completion, 1) ||
      !CopyDeviceToHost(&completed_channel,
                        allocations->direct_channel, 1) ||
      !CopyDeviceToHost(completed_descriptors.data(),
                        allocations->descriptors, request_count) ||
      !CheckAcl(
          aclrtMemcpy(output.data(), payload_bytes,
                      allocations->output, payload_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "copy direct URMA output to Host") ||
      !CheckStatus(
          resources->SnapshotDirectChannel(
              reinterpret_cast<std::uint64_t>(
                  allocations->output),
              payload_bytes, remote.address, options.bytes,
              &provider_channel_after),
          "resnapshot direct liburma channel")) {
    return DeviceTransferOutcome::kFailedQuiesced;
  }

  bool valid_result =
      completion.state ==
          static_cast<std::uint32_t>(
              DeviceBatchState::kComplete) &&
      completion.status ==
          static_cast<std::uint32_t>(
              DeviceBatchStatus::kOk) &&
      completion.planned_count == active_descriptor_count &&
      completion.submitted_count == active_descriptor_count &&
      completion.completed_count == active_descriptor_count &&
      completion.first_failed_index ==
          kvaf::access::kNoFailedIndex &&
      completed_channel.sq_head - direct_channel.sq_head ==
          active_descriptor_count &&
      completed_channel.wqe_count -
              direct_channel.wqe_count ==
          active_descriptor_count &&
      completed_channel.cq_tail - direct_channel.cq_tail ==
          active_descriptor_count &&
      completed_channel.wqe_count ==
          completed_channel.cq_tail &&
      provider_channel_after.sq_head ==
          completed_channel.sq_head &&
      provider_channel_after.wqe_count ==
          completed_channel.wqe_count &&
      provider_channel_after.cq_tail ==
          completed_channel.cq_tail;
  for (std::size_t request = 0;
       request < active_descriptor_count && valid_result;
       ++request) {
    const std::size_t table_index = indices[request];
    const auto& descriptor = completed_descriptors[request];
    valid_result =
        descriptor.state ==
            static_cast<std::uint32_t>(
                TransferState::kComplete) &&
        descriptor.logical_index == request &&
        descriptor.remote_addr ==
            remote_addresses[table_index] &&
        descriptor.local_addr ==
            reinterpret_cast<std::uint64_t>(
                allocations->output) +
                output_offsets[request] &&
        descriptor.bytes == lengths[table_index];
  }
  valid_result =
      valid_result &&
      completed_descriptors.back().state ==
          static_cast<std::uint32_t>(
              TransferState::kSkipped);
  for (std::uint64_t byte = 0;
       byte < options.bytes && valid_result; ++byte) {
    valid_result = output[byte] == Pattern(byte);
  }
  for (std::uint64_t byte = options.bytes;
       byte < payload_bytes && valid_result; ++byte) {
    valid_result = output[byte] == kCanary;
  }
  if (!valid_result) {
    std::cerr
        << "direct H2D data/completion/canary validation failed"
        << " state=" << completion.state
        << " status=" << completion.status
        << " planned=" << completion.planned_count
        << " submitted=" << completion.submitted_count
        << " completed=" << completion.completed_count
        << " first_failed=" << completion.first_failed_index
        << '\n';
    return DeviceTransferOutcome::kFailedQuiesced;
  }
  std::cout
      << "KVAF_A5_DIRECT_H2D_SMOKE_OK"
      << " bytes=" << options.bytes
      << " planned=" << active_descriptor_count
      << " submitted=" << active_descriptor_count
      << " completed=" << active_descriptor_count
      << " masked=1 canary=PASS\n";
  return stream_destroyed
             ? DeviceTransferOutcome::kPassed
             : DeviceTransferOutcome::kFailedQuiesced;
}

EndpointResult RunHost(const Options& options) {
  std::uint64_t payload_bytes = 0;
  std::uint64_t registration_bytes = 0;
  if (!RegistrationBytes(
          options, &payload_bytes, &registration_bytes)) {
    std::cerr << "invalid direct provider allocation size\n";
    return SafeEndpoint("invalid_registration_size");
  }

  g_endpoint_may_own_resources.store(true, std::memory_order_relaxed);
  auto mapping = std::make_unique<HostDdrDeviceMapping>();
  std::unique_ptr<DirectUrmaResources> resources;
  RetainEndpointOnException exception_guard(
      &resources, &mapping, nullptr);
  const auto mapping_status = mapping->AllocateAndRegister(
      registration_bytes, options.logical_device_id);
  if (!CheckStatus(mapping_status,
                   "allocate/map direct URMA Host DDR")) {
    bool mapping_cleanup_complete = false;
    if (CloseMappingWithRetry(
            mapping.get(), "clean failed Host DDR allocation",
            &mapping_cleanup_complete)) {
      return SafeEndpoint(
          mapping_cleanup_complete
              ? "host_mapping_open_failed"
              : "host_mapping_open_local_cleanup_incomplete",
          false, mapping_cleanup_complete, true, true);
    }
    g_retained_endpoint.mapping = std::move(mapping);
    return UnsafeEndpoint("host_mapping_open_cleanup_failed", true,
                          true, true, true, false);
  }
  auto* source = reinterpret_cast<std::uint8_t*>(
      mapping->host_address());
  std::memset(source, kCanary,
              static_cast<std::size_t>(registration_bytes));
  for (std::uint64_t index = 0; index < options.bytes; ++index) {
    source[index] = Pattern(index);
  }
  std::cout
      << "KVAF_A5_DIRECT_DDR_REGISTERED"
      << " host_hva=0x" << std::hex << mapping->host_address()
      << " mapped_dva=0x" << mapping->device_address()
      << std::dec << " bytes=" << registration_bytes << '\n';

  resources = std::make_unique<DirectUrmaResources>();
  const auto spec = MakeHostOpenSpec(
      options, mapping->device_address(), registration_bytes,
      payload_bytes);
  const auto open_status = resources->Open(spec);
  if (!CheckStatus(open_status, "open Host direct liburma provider")) {
    bool provider_local_cleanup_complete = false;
    bool mapping_cleanup_complete = false;
    const bool provider_closed = CloseProviderWithRetry(
        resources.get(), "clean failed Host direct provider open",
        &provider_local_cleanup_complete);
    const bool mapping_closed =
        provider_closed &&
        CloseMappingWithRetry(
            mapping.get(), "clean Host DDR after provider open failure",
            &mapping_cleanup_complete);
    if (provider_closed && mapping_closed) {
      return SafeEndpoint(
          provider_local_cleanup_complete && mapping_cleanup_complete
              ? "host_provider_open_failed"
              : "host_provider_open_local_cleanup_incomplete",
          true, mapping_cleanup_complete, true, true);
    }
    g_retained_endpoint.resources = std::move(resources);
    g_retained_endpoint.mapping = std::move(mapping);
    return UnsafeEndpoint("host_provider_open_cleanup_failed", true,
                          true, true, provider_closed, mapping_closed);
  }
  std::cout
      << "KVAF_A5_DIRECT_URMA_READY"
      << " role=host control=liburma"
      << " device=" << options.host_urma_device
      << " eid_index=" << options.host_eid_index << '\n';

  const std::string close_ready =
      JoinPath(options.exchange_dir, "device.close_ready");
  const std::string device_fatal =
      JoinPath(options.exchange_dir, "fatal.device");
  const auto transfer_pass =
      ControlMarker(options, "transfer", "PASS");
  const auto transfer_fail =
      ControlMarker(options, "transfer", "FAIL");
  const auto close_ready_wait = WaitForPeerEvidence(
      options, close_ready, transfer_pass.c_str(), transfer_fail.c_str(),
      device_fatal, "Device close-ready proof");
  if (close_ready_wait == PeerMarkerWait::kPeerSafeAbort) {
    bool provider_local_cleanup_complete = false;
    bool mapping_cleanup_complete = false;
    const bool provider_closed = CloseProviderWithRetry(
        resources.get(),
        "close Host provider after safe Device abort",
        &provider_local_cleanup_complete);
    const bool mapping_closed =
        provider_closed &&
        CloseMappingWithRetry(
            mapping.get(), "unmap Host DDR after safe Device abort",
            &mapping_cleanup_complete);
    if (!provider_closed || !mapping_closed) {
      g_retained_endpoint.resources = std::move(resources);
      g_retained_endpoint.mapping = std::move(mapping);
      return UnsafeEndpoint("host_cleanup_after_device_abort_failed", true,
                            true, true, provider_closed, mapping_closed);
    }
    return SafeEndpoint(
        provider_local_cleanup_complete && mapping_cleanup_complete
            ? "device_failed_safe"
            : "device_failed_safe_local_cleanup_incomplete",
        true, mapping_cleanup_complete, true, true);
  }
  if (close_ready_wait == PeerMarkerWait::kPeerFatal) {
    std::cerr << "Device reported unsafe transfer quiescence; retaining "
                 "both provider and Host DDR mapping\n";
    g_retained_endpoint.resources = std::move(resources);
    g_retained_endpoint.mapping = std::move(mapping);
    return UnsafeEndpoint("device_reported_unsafe");
  }
  bool transfer_ok = false;
  if (ReadExactMarker(close_ready, transfer_pass.c_str())) {
    transfer_ok = true;
  } else if (!ReadExactMarker(close_ready, transfer_fail.c_str())) {
    std::cerr
        << "Device close-ready marker has an unknown state; "
           "retaining both provider and Host DDR mapping\n";
    g_retained_endpoint.resources = std::move(resources);
    g_retained_endpoint.mapping = std::move(mapping);
    return UnsafeEndpoint("device_close_ready_invalid");
  }

  bool provider_local_cleanup_complete = false;
  const bool provider_closed = CloseProviderWithRetry(
      resources.get(), "close Host direct liburma provider",
      &provider_local_cleanup_complete);
  if (!provider_closed) {
    g_retained_endpoint.resources = std::move(resources);
    g_retained_endpoint.mapping = std::move(mapping);
    return UnsafeEndpoint("host_provider_close_failed", true,
                          true, true, false, false);
  }
  const std::string device_done =
      JoinPath(options.exchange_dir, "device.done");
  const auto provider_closed_pass =
      ControlMarker(options, "provider_closed", "PASS");
  const auto device_done_wait = WaitForPeerEvidence(
      options, device_done, provider_closed_pass.c_str(), nullptr,
      device_fatal, "Device provider-close proof");
  const bool device_teardown_seen =
      device_done_wait == PeerMarkerWait::kFound;
  const bool device_aborted_safe =
      device_done_wait == PeerMarkerWait::kPeerSafeAbort;
  const bool device_teardown_safe =
      (device_teardown_seen &&
       ReadExactMarker(
           device_done, provider_closed_pass.c_str())) ||
      device_aborted_safe;
  if (device_done_wait == PeerMarkerWait::kPeerFatal) {
    std::cerr << "Device reported unsafe provider teardown\n";
  }
  if (device_teardown_seen && !device_teardown_safe) {
    std::cerr
        << "Device teardown marker is not exactly "
           "provider_closed=PASS; "
           "retaining Host DDR mapping\n";
  }
  if (!device_teardown_safe) {
    g_retained_endpoint.mapping = std::move(mapping);
    return UnsafeEndpoint("device_provider_close_unproven", true,
                          false, false, true, false);
  }
  bool mapping_cleanup_complete = false;
  const bool mapping_closed =
      CloseMappingWithRetry(
          mapping.get(), "unmap/free direct URMA Host DDR",
          &mapping_cleanup_complete);
  if (!mapping_closed) {
    g_retained_endpoint.mapping = std::move(mapping);
    return UnsafeEndpoint("host_mapping_close_failed", true,
                          true, true, true, false);
  }
  if (transfer_ok && !device_aborted_safe &&
      provider_local_cleanup_complete && mapping_cleanup_complete) {
    std::cout
        << "KVAF_A5_DIRECT_TEARDOWN_OK role=host\n";
    return PassedEndpoint();
  }
  const char* reason = nullptr;
  if (!provider_local_cleanup_complete || !mapping_cleanup_complete) {
    reason = "host_local_cleanup_incomplete";
  } else if (device_aborted_safe) {
    reason = "device_failed_safe_after_close_ready";
  } else {
    reason = "transfer_validation_failed";
  }
  return SafeEndpoint(reason, true, mapping_cleanup_complete, true, true);
}

EndpointResult RunDevice(const Options& options) {
  std::uint64_t payload_bytes = 0;
  std::uint64_t registration_bytes = 0;
  if (!RegistrationBytes(
          options, &payload_bytes, &registration_bytes)) {
    std::cerr << "invalid direct provider allocation size\n";
    return SafeEndpoint("invalid_registration_size");
  }

  g_endpoint_may_own_resources.store(true, std::memory_order_relaxed);
  constexpr std::size_t kActiveDescriptorTarget = 4;
  const std::size_t active_descriptor_count =
      static_cast<std::size_t>(std::min<std::uint64_t>(
          options.bytes, kActiveDescriptorTarget));
  const std::size_t request_count =
      active_descriptor_count + 1;
  auto allocations = std::make_unique<DeviceAllocations>();
  std::unique_ptr<DirectUrmaResources> resources;
  aclrtStream retained_stream = nullptr;
  bool transfer_stream_quiesced = true;
  RetainEndpointOnException exception_guard(
      &resources, nullptr, &allocations, &retained_stream,
      &transfer_stream_quiesced);
  if (!AllocateDeviceInputs(
          registration_bytes, active_descriptor_count,
          request_count, allocations.get())) {
    const bool allocations_closed =
        CloseAllocationsWithRetry(allocations.get());
    return SafeEndpoint(
        allocations_closed
            ? "device_allocation_failed"
            : "device_allocation_local_cleanup_incomplete",
        false, allocations_closed, true, allocations_closed);
  }
  std::vector<std::uint8_t> initial_output(
      payload_bytes, kCanary);
  if (!CheckAcl(
          aclrtMemcpy(allocations->output, payload_bytes,
                      initial_output.data(), payload_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE),
          "initialize direct output canary")) {
    const bool allocations_closed =
        CloseAllocationsWithRetry(allocations.get());
    return SafeEndpoint(
        allocations_closed
            ? "device_canary_initialize_failed"
            : "device_canary_local_cleanup_incomplete",
        false, allocations_closed, true, allocations_closed);
  }

  resources = std::make_unique<DirectUrmaResources>();
  const auto spec = MakeDeviceOpenSpec(
      options,
      reinterpret_cast<std::uint64_t>(allocations->output),
      registration_bytes, payload_bytes);
  const auto open_status = resources->Open(spec);
  if (!CheckStatus(open_status,
                   "open AICPU direct liburma provider")) {
    bool provider_local_cleanup_complete = false;
    const bool provider_closed = CloseProviderWithRetry(
        resources.get(), "clean failed AICPU direct provider open",
        &provider_local_cleanup_complete);
    const bool allocations_closed =
        provider_closed && CloseAllocationsWithRetry(allocations.get());
    if (provider_closed) {
      return SafeEndpoint(
          provider_local_cleanup_complete && allocations_closed
              ? "device_provider_open_failed"
              : "device_provider_open_local_cleanup_incomplete",
          true, allocations_closed, true, allocations_closed);
    }
    g_retained_endpoint.resources = std::move(resources);
    g_retained_endpoint.allocations = std::move(allocations);
    return UnsafeEndpoint("device_provider_open_cleanup_failed", true,
                          true, true, provider_closed,
                          allocations_closed);
  }
  std::cout
      << "KVAF_A5_DIRECT_URMA_READY"
      << " role=device control=AICPU+liburma"
      << " device=" << options.device_urma_device
      << " eid_index=" << options.device_eid_index << '\n';

  const auto transfer_outcome = RunDeviceTransfer(
      options, payload_bytes, resources.get(), allocations.get(),
      &retained_stream, &transfer_stream_quiesced);
  if (transfer_outcome == DeviceTransferOutcome::kUnsafeUnknown) {
    RetainEndpointOwnership(&resources, nullptr, &allocations,
                            &retained_stream,
                            &transfer_stream_quiesced);
    return UnsafeEndpoint("aiv_stream_quiescence_unknown");
  }
  bool transfer_stream_local_cleanup_complete = true;
  if (retained_stream != nullptr) {
    std::cerr
        << "quiesced AIV transfer stream destroy failed; retrying once\n";
    if (aclrtDestroyStream(retained_stream) == ACL_SUCCESS) {
      retained_stream = nullptr;
    } else {
      transfer_stream_local_cleanup_complete = false;
      // The stream was synchronized successfully, so this is a local handle
      // leak rather than an in-flight DMA risk. Keep the handle discoverable
      // until aclrtResetDevice/process exit instead of silently dropping it.
      RetainEndpointOwnership(nullptr, nullptr, nullptr,
                              &retained_stream,
                              &transfer_stream_quiesced);
    }
  }
  const bool transfer_ok =
      transfer_outcome == DeviceTransferOutcome::kPassed;
  const std::string close_ready =
      JoinPath(options.exchange_dir, "device.close_ready");
  const auto close_ready_marker =
      ControlMarker(options, "transfer", transfer_ok ? "PASS" : "FAIL");
  const auto host_fatal = JoinPath(options.exchange_dir, "fatal.host");
  bool close_ready_warning_written = false;
  while (!AtomicWriteText(close_ready, close_ready_marker)) {
    // rename may have succeeded even when the following directory fsync
    // failed. If the exact marker is visible, Host can safely participate in
    // the provider teardown handshake and Device must do so as well.
    if (ReadExactMarker(close_ready, close_ready_marker.c_str())) {
      break;
    }
    if (ValidPeerFatal(options, host_fatal)) {
      RetainEndpointOwnership(&resources, nullptr, &allocations,
                              &retained_stream,
                              &transfer_stream_quiesced);
      return UnsafeEndpoint("host_failed_before_close_ready_publish");
    }
    if (!close_ready_warning_written) {
      std::cerr << "failed to publish Device close-ready marker; retaining "
                   "ownership and retrying until the evidence path recovers\n";
      close_ready_warning_written = true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  bool provider_local_cleanup_complete = false;
  const bool provider_closed = CloseProviderWithRetry(
      resources.get(), "close AICPU direct liburma provider",
      &provider_local_cleanup_complete);
  if (!provider_closed) {
    const std::string device_done =
        JoinPath(options.exchange_dir, "device.done");
    (void)AtomicWriteText(
        device_done,
        ControlMarker(options, "provider_closed", "FAIL"));
    g_retained_endpoint.resources = std::move(resources);
    g_retained_endpoint.allocations = std::move(allocations);
    return UnsafeEndpoint("device_provider_close_failed", true,
                          true, false, false, false);
  }
  const bool allocations_closed =
      CloseAllocationsWithRetry(allocations.get());
  const std::string device_done =
      JoinPath(options.exchange_dir, "device.done");
  if (!AtomicWriteText(
          device_done,
          ControlMarker(options, "provider_closed", "PASS"))) {
    std::cerr
        << "failed to publish Device done marker\n";
    const bool local_cleanup_complete =
        provider_local_cleanup_complete && allocations_closed &&
        transfer_stream_local_cleanup_complete;
    return SafeEndpoint(
        local_cleanup_complete
            ? "device_done_publish_failed"
            : "device_done_publish_local_cleanup_incomplete",
        true, local_cleanup_complete, true, allocations_closed);
  }
  if (!allocations_closed) {
    return SafeEndpoint("device_allocations_local_cleanup_incomplete", true,
                        false, true, false);
  }
  if (transfer_ok && provider_local_cleanup_complete &&
      transfer_stream_local_cleanup_complete) {
    std::cout
        << "KVAF_A5_DIRECT_TEARDOWN_OK role=device\n";
    return PassedEndpoint();
  }
  const bool local_cleanup_complete =
      provider_local_cleanup_complete &&
      transfer_stream_local_cleanup_complete;
  return SafeEndpoint(
      !transfer_stream_local_cleanup_complete
          ? "device_stream_local_cleanup_incomplete"
          : (provider_local_cleanup_complete
                 ? "transfer_validation_failed"
                 : "device_provider_local_cleanup_incomplete"),
      true, local_cleanup_complete, true, true);
}

bool TryRecoverRetainedAfterException(
    EndpointResult* recovered_result) {
  if (recovered_result == nullptr) {
    return false;
  }
  const bool resource_armed =
      g_retained_endpoint.resources != nullptr ||
      g_retained_endpoint.mapping != nullptr;
  if (g_retained_endpoint.transfer_stream != nullptr &&
      !g_retained_endpoint.transfer_stream_quiesced) {
    return false;
  }

  bool provider_local_cleanup_complete = true;
  const bool provider_closed =
      g_retained_endpoint.resources == nullptr ||
      CloseProviderWithRetry(
          g_retained_endpoint.resources.get(),
          "close provider after unexpected C++ exception",
          &provider_local_cleanup_complete);
  if (!provider_closed) {
    return false;
  }

  bool mapping_local_cleanup_complete = true;
  const bool mapping_closed =
      g_retained_endpoint.mapping == nullptr ||
      CloseMappingWithRetry(
          g_retained_endpoint.mapping.get(),
          "unmap Host DDR after unexpected C++ exception",
          &mapping_local_cleanup_complete);
  if (!mapping_closed) {
    return false;
  }
  const bool allocations_closed =
      g_retained_endpoint.allocations == nullptr ||
      CloseAllocationsWithRetry(g_retained_endpoint.allocations.get());

  bool stream_local_cleanup_complete = true;
  if (g_retained_endpoint.transfer_stream != nullptr) {
    if (aclrtDestroyStream(g_retained_endpoint.transfer_stream) ==
        ACL_SUCCESS) {
      g_retained_endpoint.transfer_stream = nullptr;
    } else if (aclrtDestroyStream(
                   g_retained_endpoint.transfer_stream) == ACL_SUCCESS) {
      g_retained_endpoint.transfer_stream = nullptr;
    } else {
      stream_local_cleanup_complete = false;
    }
  }

  recovered_result->outcome = EndpointOutcome::kFailedSafe;
  recovered_result->resource_armed = resource_armed;
  recovered_result->stream_quiesced = true;
  recovered_result->queue_drained = true;
  recovered_result->provider_closed = true;
  recovered_result->backing_released =
      mapping_local_cleanup_complete && allocations_closed;
  recovered_result->host_mapping_unregistered = true;
  recovered_result->allocations_released = allocations_closed;
  if (!provider_local_cleanup_complete) {
    recovered_result->reason =
        "unexpected_cpp_exception_provider_local_cleanup_incomplete";
  } else if (!mapping_local_cleanup_complete) {
    recovered_result->reason =
        "unexpected_cpp_exception_mapping_local_cleanup_incomplete";
  } else if (!allocations_closed) {
    recovered_result->reason =
        "unexpected_cpp_exception_allocations_local_cleanup_incomplete";
  } else if (!stream_local_cleanup_complete) {
    recovered_result->reason =
        "unexpected_cpp_exception_stream_local_cleanup_incomplete";
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::set_terminate(FailClosedTerminate);
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }

  if (!CheckAcl(aclInit(nullptr), "aclInit")) {
    const auto result = SafeEndpoint("acl_init_failed");
    PersistTerminalMarkerUntilDurable(
        options, result, "acl_init_failure");
    PublishDeviceSafeAbortWakeEdgeBestEffort(options, result);
    return 1;
  }
  if (!CheckAcl(aclrtSetDevice(options.logical_device_id),
                "aclrtSetDevice")) {
    const auto result = SafeEndpoint("acl_set_device_failed");
    PersistTerminalMarkerUntilDurable(
        options, result, "acl_set_device_failure");
    PublishDeviceSafeAbortWakeEdgeBestEffort(options, result);
    (void)aclFinalize();
    return 1;
  }

  // Initialization is still pre-resource and remains externally
  // interruptible. Ignore lifecycle signals only after the runtime context is
  // ready and immediately before Host DDR/URMA/HBM ownership can begin.
  IgnoreLifecycleSignals();

  EndpointResult result = UnsafeEndpoint("unexpected_cpp_exception");
  EndpointResult recovered_exception =
      SafeEndpoint("unexpected_cpp_exception_cleaned");
  try {
    result = options.role == "host"
                 ? RunHost(options)
                 : RunDevice(options);
  } catch (const std::exception& error) {
    std::cerr << "unexpected C++ exception while endpoint owns resources: "
              << error.what() << '\n';
    try {
      if (TryRecoverRetainedAfterException(&recovered_exception)) {
        result = std::move(recovered_exception);
      }
    } catch (...) {
      std::cerr << "checked cleanup after exception also threw; retaining "
                   "all endpoint ownership\n";
    }
  } catch (...) {
    std::cerr << "unknown exception while endpoint owns resources\n";
    try {
      if (TryRecoverRetainedAfterException(&recovered_exception)) {
        result = std::move(recovered_exception);
      }
    } catch (...) {
      std::cerr << "checked cleanup after unknown exception also threw; "
                   "retaining all endpoint ownership\n";
    }
  }
  if (result.outcome == EndpointOutcome::kFailedUnsafe) {
    QuarantineUnsafeTeardown(options, result);
  }
  g_endpoint_may_own_resources.store(false, std::memory_order_relaxed);

  // Persist transport safety before aclrtResetDevice/aclFinalize. Runtime
  // detach is local cleanup and can itself fail or hang; it must not erase the
  // already-proven fact that streams/queues/providers/mappings are safe. A
  // successful PASS atomically replaces this provisional FAIL_SAFE marker
  // after runtime detach completes.
  EndpointResult pre_runtime_cleanup = result;
  if (pre_runtime_cleanup.outcome == EndpointOutcome::kPassed) {
    pre_runtime_cleanup.outcome = EndpointOutcome::kFailedSafe;
    pre_runtime_cleanup.reason = "transport_safe_runtime_cleanup_pending";
  }
  PersistTerminalMarkerUntilDurable(
      options, pre_runtime_cleanup, "before_runtime_cleanup");
  if (result.outcome == EndpointOutcome::kFailedSafe) {
    PublishDeviceSafeAbortWakeEdgeBestEffort(
        options, pre_runtime_cleanup);
  }

  const bool reset =
      CheckAcl(aclrtResetDevice(options.logical_device_id),
               "aclrtResetDevice");
  if (reset) {
    g_retained_endpoint.transfer_stream = nullptr;
  }
  const bool finalized =
      CheckAcl(aclFinalize(), "aclFinalize");
  if ((!reset || !finalized) &&
      result.outcome == EndpointOutcome::kPassed) {
    result.outcome = EndpointOutcome::kFailedSafe;
    result.reason = !reset ? "acl_reset_failed_after_safe_teardown"
                           : "acl_finalize_failed_after_safe_teardown";
  }
  // Publish the post-runtime result persistently as well. The provisional
  // proof above already prevents UNPROVEN classification if runtime cleanup
  // hangs; this durable replacement records PASS or the precise downgrade.
  PersistTerminalMarkerUntilDurable(
      options, result, "after_runtime_cleanup");
  PublishDeviceSafeAbortWakeEdgeBestEffort(options, result);
  return result.outcome == EndpointOutcome::kPassed && reset && finalized
             ? 0
             : 1;
}
