#include "kvaf/access_operator/a5/aiv_simt_urma_h2d_launcher.h"
#include "kvaf/access_operator/a5/direct_urma_resources.h"
#include "kvaf/access_operator/a5/host_ddr_mapping.h"

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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
constexpr int kFatalProviderTeardown = 3;

enum class DeviceTransferOutcome {
  kPassed,
  kFailedQuiesced,
  kUnsafeUnknown,
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
         " --exchange-dir PATH [--logical-device-id N]"
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
               parsed <=
                   kvaf::access::kAivUrmaMaxTransferBytes -
                       kCanaryBytes) {
      options->bytes = parsed;
    } else {
      return false;
    }
  }
  return (options->role == "host" ||
          options->role == "device") &&
         !options->host_urma_device.empty() &&
         !options->device_urma_device.empty() &&
         !options->provider_library.empty() &&
         options->provider_library.front() == '/' &&
         !options->provider_soname.empty() &&
         options->provider_soname.find('/') == std::string::npos &&
         !options->exchange_dir.empty() &&
         ParseEid(host_eid, &options->host_eid) &&
         ParseEid(device_eid, &options->device_eid) &&
         options->host_eid != options->device_eid;
}

std::string JoinPath(const std::string& directory,
                     const char* name) {
  return directory + "/" + name;
}

bool FileExists(const std::string& path) {
  struct stat info {};
  return ::stat(path.c_str(), &info) == 0 &&
         S_ISREG(info.st_mode);
}

bool WaitForFile(const std::string& path,
                 std::uint32_t timeout_seconds) {
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::seconds(timeout_seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (FileExists(path)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return FileExists(path);
}

enum class PeerMarkerWait {
  kFound,
  kPeerFatal,
  kTimedOut,
};

PeerMarkerWait WaitForFileOrPeerFatal(
    const std::string& path, const std::string& peer_fatal,
    std::uint32_t timeout_seconds) {
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::seconds(timeout_seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (FileExists(path)) {
      return PeerMarkerWait::kFound;
    }
    if (FileExists(peer_fatal)) {
      return PeerMarkerWait::kPeerFatal;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (FileExists(path)) {
    return PeerMarkerWait::kFound;
  }
  return FileExists(peer_fatal)
             ? PeerMarkerWait::kPeerFatal
             : PeerMarkerWait::kTimedOut;
}

bool AtomicWriteText(const std::string& path,
                     const std::string& text) {
  const std::string temporary =
      path + ".tmp." +
      std::to_string(static_cast<long long>(::getpid()));
  {
    std::ofstream output(
        temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      return false;
    }
    output.write(text.data(),
                 static_cast<std::streamsize>(text.size()));
    if (!output) {
      return false;
    }
  }
  if (std::rename(temporary.c_str(), path.c_str()) != 0) {
    std::remove(temporary.c_str());
    return false;
  }
  return true;
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

[[noreturn]] void QuarantineUnsafeTeardown(
    const Options& options) {
  (void)std::signal(SIGHUP, SIG_IGN);
  (void)std::signal(SIGINT, SIG_IGN);
  (void)std::signal(SIGTERM, SIG_IGN);
  (void)std::signal(SIGQUIT, SIG_IGN);

  const auto pid = static_cast<long long>(::getpid());
  const std::string marker = JoinPath(
      options.exchange_dir,
      options.role == "host"
          ? "fatal.host"
          : "fatal.device");
  const bool marker_written = AtomicWriteText(
      marker, "pid=" + std::to_string(pid) +
                  "\nreason=unsafe_provider_teardown\n");
  std::cerr
      << "KVAF_A5_DIRECT_QUARANTINED"
      << " role=" << options.role
      << " pid=" << pid
      << " marker_written=" << (marker_written ? 1 : 0)
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
                            const char* operation) {
  auto status = resources->Close();
  if (status.ok()) {
    return true;
  }
  std::cerr << operation << " failed: " << status.message()
            << "; retrying once without releasing backing memory\n";
  status = resources->Close();
  if (status.ok()) {
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
    DeviceAllocations* allocations) {
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
  const bool stream_destroyed =
      CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");

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

int RunHost(const Options& options) {
  std::uint64_t payload_bytes = 0;
  std::uint64_t registration_bytes = 0;
  if (!RegistrationBytes(
          options, &payload_bytes, &registration_bytes)) {
    std::cerr << "invalid direct provider allocation size\n";
    return 1;
  }

  auto mapping = std::make_unique<HostDdrDeviceMapping>();
  if (!CheckStatus(
          mapping->AllocateAndRegister(
              registration_bytes, options.logical_device_id),
          "allocate/map direct URMA Host DDR")) {
    return 1;
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

  auto resources = std::make_unique<DirectUrmaResources>();
  const auto spec = MakeHostOpenSpec(
      options, mapping->device_address(), registration_bytes,
      payload_bytes);
  const auto open_status = resources->Open(spec);
  if (!CheckStatus(open_status, "open Host direct liburma provider")) {
    if (!CloseProviderWithRetry(
            resources.get(), "clean failed Host direct provider open")) {
      (void)resources.release();
      (void)mapping.release();
      return kFatalProviderTeardown;
    }
    (void)mapping->Close();
    return 1;
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
  const auto close_ready_wait = WaitForFileOrPeerFatal(
      close_ready, device_fatal, options.timeout_seconds);
  if (close_ready_wait != PeerMarkerWait::kFound) {
    std::cerr
        << (close_ready_wait == PeerMarkerWait::kPeerFatal
                ? "Device reported unsafe transfer quiescence"
                : "timed out before Device proved transfer quiescence")
        << "; retaining both provider and Host DDR mapping\n";
    (void)resources.release();
    (void)mapping.release();
    return kFatalProviderTeardown;
  }
  bool transfer_ok = false;
  if (ReadExactMarker(close_ready, "transfer=PASS\n")) {
    transfer_ok = true;
  } else if (!ReadExactMarker(close_ready, "transfer=FAIL\n")) {
    std::cerr
        << "Device close-ready marker has an unknown state; "
           "retaining both provider and Host DDR mapping\n";
    (void)resources.release();
    (void)mapping.release();
    return kFatalProviderTeardown;
  }

  const bool provider_closed = CloseProviderWithRetry(
      resources.get(), "close Host direct liburma provider");
  if (!provider_closed) {
    (void)resources.release();
    (void)mapping.release();
    return kFatalProviderTeardown;
  }
  const std::string device_done =
      JoinPath(options.exchange_dir, "device.done");
  const bool device_teardown_seen =
      WaitForFile(device_done, options.timeout_seconds);
  if (!device_teardown_seen) {
    std::cerr
        << "timed out waiting for Device provider teardown\n";
  }
  const bool device_teardown_safe =
      device_teardown_seen &&
      ReadExactMarker(device_done, "provider_closed=PASS\n");
  if (device_teardown_seen && !device_teardown_safe) {
    std::cerr
        << "Device teardown marker is not exactly "
           "provider_closed=PASS; "
           "retaining Host DDR mapping\n";
  }
  if (!device_teardown_safe) {
    (void)mapping.release();
    return kFatalProviderTeardown;
  }
  const bool mapping_closed =
      CheckStatus(mapping->Close(),
                  "unmap/free direct URMA Host DDR");
  if (transfer_ok && provider_closed && device_teardown_safe &&
      mapping_closed) {
    std::cout
        << "KVAF_A5_DIRECT_TEARDOWN_OK role=host\n";
    return 0;
  }
  return 1;
}

int RunDevice(const Options& options) {
  std::uint64_t payload_bytes = 0;
  std::uint64_t registration_bytes = 0;
  if (!RegistrationBytes(
          options, &payload_bytes, &registration_bytes)) {
    std::cerr << "invalid direct provider allocation size\n";
    return 1;
  }

  constexpr std::size_t kActiveDescriptorTarget = 4;
  const std::size_t active_descriptor_count =
      static_cast<std::size_t>(std::min<std::uint64_t>(
          options.bytes, kActiveDescriptorTarget));
  const std::size_t request_count =
      active_descriptor_count + 1;
  auto allocations = std::make_unique<DeviceAllocations>();
  if (!AllocateDeviceInputs(
          registration_bytes, active_descriptor_count,
          request_count, allocations.get())) {
    return 1;
  }
  std::vector<std::uint8_t> initial_output(
      payload_bytes, kCanary);
  if (!CheckAcl(
          aclrtMemcpy(allocations->output, payload_bytes,
                      initial_output.data(), payload_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE),
          "initialize direct output canary")) {
    return 1;
  }

  auto resources = std::make_unique<DirectUrmaResources>();
  const auto spec = MakeDeviceOpenSpec(
      options,
      reinterpret_cast<std::uint64_t>(allocations->output),
      registration_bytes, payload_bytes);
  const auto open_status = resources->Open(spec);
  if (!CheckStatus(open_status,
                   "open AICPU direct liburma provider")) {
    if (!CloseProviderWithRetry(
            resources.get(), "clean failed AICPU direct provider open")) {
      (void)resources.release();
      (void)allocations.release();
      return kFatalProviderTeardown;
    }
    return 1;
  }
  std::cout
      << "KVAF_A5_DIRECT_URMA_READY"
      << " role=device control=AICPU+liburma"
      << " device=" << options.device_urma_device
      << " eid_index=" << options.device_eid_index << '\n';

  const auto transfer_outcome = RunDeviceTransfer(
      options, payload_bytes, resources.get(), allocations.get());
  if (transfer_outcome == DeviceTransferOutcome::kUnsafeUnknown) {
    (void)resources.release();
    (void)allocations.release();
    return kFatalProviderTeardown;
  }
  const bool transfer_ok =
      transfer_outcome == DeviceTransferOutcome::kPassed;
  const std::string close_ready =
      JoinPath(options.exchange_dir, "device.close_ready");
  if (!AtomicWriteText(
          close_ready,
          transfer_ok ? "transfer=PASS\n" : "transfer=FAIL\n")) {
    std::cerr
        << "failed to publish Device close-ready marker\n";
    // Enter Close here as well so a concurrently waiting Host can complete
    // the two-sided destroy_ctx socket handshake. If the marker never becomes
    // visible, this call may block; the runner records both live processes as
    // quarantined instead of killing either side.
    if (!CloseProviderWithRetry(
            resources.get(), "close AICPU provider after marker failure")) {
      (void)resources.release();
      (void)allocations.release();
      return kFatalProviderTeardown;
    }
    (void)allocations->Close();
    return 1;
  }

  const bool provider_closed = CloseProviderWithRetry(
      resources.get(), "close AICPU direct liburma provider");
  if (!provider_closed) {
    const std::string device_done =
        JoinPath(options.exchange_dir, "device.done");
    (void)AtomicWriteText(
        device_done, "provider_closed=FAIL\n");
    (void)resources.release();
    (void)allocations.release();
    return kFatalProviderTeardown;
  }
  const bool allocations_closed = allocations->Close();
  if (!allocations_closed) {
    std::cerr << "one or more aclrtFree calls failed\n";
  }
  const bool final_ok =
      transfer_ok && provider_closed && allocations_closed;
  const std::string device_done =
      JoinPath(options.exchange_dir, "device.done");
  if (!AtomicWriteText(
          device_done, "provider_closed=PASS\n")) {
    std::cerr
        << "failed to publish Device done marker\n";
    return 1;
  }
  if (final_ok) {
    std::cout
        << "KVAF_A5_DIRECT_TEARDOWN_OK role=device\n";
    return 0;
  }
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }

  if (!CheckAcl(aclInit(nullptr), "aclInit")) {
    return 1;
  }
  if (!CheckAcl(aclrtSetDevice(options.logical_device_id),
                "aclrtSetDevice")) {
    (void)aclFinalize();
    return 1;
  }

  const int result =
      options.role == "host"
          ? RunHost(options)
          : RunDevice(options);
  if (result == kFatalProviderTeardown) {
    QuarantineUnsafeTeardown(options);
  }
  const bool reset =
      CheckAcl(aclrtResetDevice(options.logical_device_id),
               "aclrtResetDevice");
  const bool finalized =
      CheckAcl(aclFinalize(), "aclFinalize");
  return result == 0 && reset && finalized ? 0 : 1;
}
