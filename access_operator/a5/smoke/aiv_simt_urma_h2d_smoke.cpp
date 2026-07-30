#include "kvaf/access_operator/a5/aiv_simt_urma_h2d_launcher.h"
#include "kvaf/access_operator/a5/hcomm_urma_resources.h"

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
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
using kvaf::access::a5::ChannelEngine;
using kvaf::access::a5::ChannelRole;
using kvaf::access::a5::EndpointLocation;
using kvaf::access::a5::EndpointSpec;
using kvaf::access::a5::HcommUrmaResources;
using kvaf::access::a5::HostDdrDeviceMapping;
using kvaf::access::a5::ImportedMemoryView;
using kvaf::access::a5::MemoryLocation;

constexpr std::uint8_t kCanary = 0xa5;
constexpr const char* kDeviceChannelCreatedMarker =
    "device.channel_created";

struct Options {
  std::string role;
  std::array<std::uint8_t, 16> host_eid{};
  std::array<std::uint8_t, 16> device_eid{};
  std::uint32_t logical_device_id{0};
  std::uint32_t physical_device_id{0};
  std::uint32_t super_device_id{0};
  std::uint32_t server_index{0};
  std::uint32_t super_pod_index{0};
  std::uint32_t timeout_seconds{120};
  std::uint64_t bytes{4096};
  std::string exchange_dir;
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
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
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
  std::string compact;
  compact.reserve(32);
  for (char value : text) {
    if (value == ':' || value == '-') {
      continue;
    }
    compact.push_back(value);
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
    (*eid)[index] = static_cast<std::uint8_t>((high << 4) | low);
    any_nonzero = any_nonzero || (*eid)[index] != 0;
  }
  return any_nonzero;
}

void PrintUsage(const char* program) {
  std::cerr
      << "usage: " << program
      << " --role host|device --host-eid HEX32 --device-eid HEX32"
         " --exchange-dir PATH [--logical-device-id N]"
         " [--physical-device-id N] [--super-device-id N]"
         " [--server-index N] [--super-pod-index N]"
         " [--bytes N] [--timeout-seconds N]\n";
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
    } else if (key == "--exchange-dir") {
      options->exchange_dir = value;
    } else if (!ParseUnsigned(value, &parsed)) {
      return false;
    } else if (key == "--logical-device-id" &&
               parsed <= std::numeric_limits<std::uint32_t>::max()) {
      options->logical_device_id = static_cast<std::uint32_t>(parsed);
    } else if (key == "--physical-device-id" &&
               parsed <= std::numeric_limits<std::uint32_t>::max()) {
      options->physical_device_id = static_cast<std::uint32_t>(parsed);
    } else if (key == "--super-device-id" &&
               parsed <= std::numeric_limits<std::uint32_t>::max()) {
      options->super_device_id = static_cast<std::uint32_t>(parsed);
    } else if (key == "--server-index" &&
               parsed <= std::numeric_limits<std::uint32_t>::max()) {
      options->server_index = static_cast<std::uint32_t>(parsed);
    } else if (key == "--super-pod-index" &&
               parsed <= std::numeric_limits<std::uint32_t>::max()) {
      options->super_pod_index = static_cast<std::uint32_t>(parsed);
    } else if (key == "--timeout-seconds" && parsed != 0 &&
               parsed <=
                   std::numeric_limits<std::uint32_t>::max() / 1000U) {
      options->timeout_seconds = static_cast<std::uint32_t>(parsed);
    } else if (key == "--bytes" && parsed != 0 &&
               parsed <= kvaf::access::kAivUrmaMaxTransferBytes) {
      options->bytes = parsed;
    } else {
      return false;
    }
  }
  return (options->role == "host" || options->role == "device") &&
         !options->exchange_dir.empty() &&
         ParseEid(host_eid, &options->host_eid) &&
         ParseEid(device_eid, &options->device_eid) &&
         options->host_eid != options->device_eid;
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

bool WaitForFile(const std::string& path, std::uint32_t timeout_seconds) {
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

bool AtomicWrite(const std::string& path,
                 const std::vector<std::uint8_t>& bytes) {
  const std::string temporary =
      path + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      return false;
    }
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
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

bool AtomicWriteText(const std::string& path, const std::string& text) {
  return AtomicWrite(path, std::vector<std::uint8_t>(text.begin(), text.end()));
}

bool ReadFile(const std::string& path, std::vector<std::uint8_t>* bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || bytes == nullptr) {
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

template <typename T>
bool AllocateDevice(std::size_t count, T** pointer) {
  return CheckAcl(
      aclrtMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T),
                  ACL_MEM_MALLOC_HUGE_FIRST),
      "aclrtMalloc");
}

template <typename T>
bool CopyHostToDevice(T* destination, const T* source, std::size_t count) {
  return CheckAcl(
      aclrtMemcpy(destination, count * sizeof(T), source, count * sizeof(T),
                  ACL_MEMCPY_HOST_TO_DEVICE),
      "aclrtMemcpy(H2D)");
}

template <typename T>
bool CopyDeviceToHost(T* destination, const T* source, std::size_t count) {
  return CheckAcl(
      aclrtMemcpy(destination, count * sizeof(T), source, count * sizeof(T),
                  ACL_MEMCPY_DEVICE_TO_HOST),
      "aclrtMemcpy(D2H)");
}

int RunHost(const Options& options) {
  HostDdrDeviceMapping device_mapping;
  if (!CheckStatus(
          device_mapping.AllocateAndRegister(
              options.bytes, options.logical_device_id),
          "allocate and map A5 Host DDR")) {
    return 1;
  }
  void* host_ddr =
      reinterpret_cast<void*>(device_mapping.host_address());
  auto* source = static_cast<std::uint8_t*>(host_ddr);
  for (std::uint64_t index = 0; index < options.bytes; ++index) {
    source[index] = Pattern(index);
  }

  bool transfer_and_hcomm_ok = false;
  {
    HcommUrmaResources resources;
    std::uint32_t registration = 0;
    std::vector<std::uint8_t> descriptor;
    if (!CheckStatus(resources.OpenEndpoint(MakeHostEndpoint(options)),
                     "open Host endpoint") ||
        !CheckStatus(resources.RegisterMemory(
                         "kvaf_host_ddr",
                         reinterpret_cast<std::uint64_t>(host_ddr),
                         options.bytes, MemoryLocation::kHostDdr,
                         &registration),
                     "register Host DDR") ||
        !CheckStatus(resources.ExportMemory(registration, &descriptor),
                     "export Host DDR MR") ||
        !AtomicWrite(JoinPath(options.exchange_dir, "host_ddr.mr"),
                     descriptor)) {
      std::cerr << "failed to publish Host DDR MR descriptor\n";
    } else {
      std::cout << "KVAF_A5_DDR_REGISTERED"
                << " host_hva=0x" << std::hex
                << reinterpret_cast<std::uint64_t>(host_ddr)
                << " mapped_dva=0x" << device_mapping.device_address()
                << std::dec
                << " bytes=" << options.bytes << '\n';
      const std::string device_channel_created =
          JoinPath(options.exchange_dir, kDeviceChannelCreatedMarker);
      if (!WaitForFile(device_channel_created, options.timeout_seconds)) {
        std::cerr
            << "timed out waiting for Device server channel creation\n";
      } else if (
          !CheckStatus(resources.CreateChannel(
                           MakeDeviceEndpoint(options), ChannelEngine::kCpu,
                           ChannelRole::kClient),
                       "create Host CPU channel") ||
          !CheckStatus(
              resources.WaitChannelReady(options.timeout_seconds * 1000U),
              "wait Host CPU channel")) {
        std::cerr << "Host channel did not become ready\n";
      } else {
        std::cout << "KVAF_A5_CHANNEL_READY role=host engine=CPU\n";
        const std::string done =
            JoinPath(options.exchange_dir, "device.done");
        if (!WaitForFile(done, options.timeout_seconds)) {
          std::cerr << "timed out waiting for Device completion marker\n";
        } else {
          const bool hcomm_closed =
              CheckStatus(resources.Close(), "close Host HCOMM resources");
          transfer_and_hcomm_ok = hcomm_closed;
        }
      }
    }
  }
  const bool mapping_closed =
      CheckStatus(device_mapping.Close(), "unmap and free A5 Host DDR");
  if (transfer_and_hcomm_ok && mapping_closed) {
    std::cout << "KVAF_A5_TEARDOWN_OK role=host\n";
    return 0;
  }
  return 1;
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

bool AllocateDeviceInputs(std::uint64_t output_bytes,
                          std::size_t table_entries,
                          std::size_t request_entries,
                          DeviceAllocations* allocations) {
  return AllocateDevice(output_bytes, &allocations->output) &&
         AllocateDevice(table_entries, &allocations->remote_addresses) &&
         AllocateDevice(table_entries, &allocations->lengths) &&
         AllocateDevice(table_entries, &allocations->valid) &&
         AllocateDevice(request_entries, &allocations->indices) &&
         AllocateDevice(request_entries, &allocations->mask) &&
         AllocateDevice(request_entries, &allocations->output_offsets) &&
         AllocateDevice(request_entries, &allocations->descriptors) &&
         AllocateDevice(1, &allocations->params) &&
         AllocateDevice(1, &allocations->completion) &&
         AllocateDevice(1, &allocations->direct_channel);
}

int RunDevice(const Options& options) {
  const std::string mr_path =
      JoinPath(options.exchange_dir, "host_ddr.mr");
  if (!WaitForFile(mr_path, options.timeout_seconds)) {
    std::cerr << "timed out waiting for Host DDR MR descriptor\n";
    return 1;
  }
  std::vector<std::uint8_t> descriptor;
  if (!ReadFile(mr_path, &descriptor)) {
    std::cerr << "failed to read Host DDR MR descriptor\n";
    return 1;
  }

  constexpr std::uint64_t kCanaryBytes = 64;
  if (options.bytes >
      std::numeric_limits<std::uint64_t>::max() - kCanaryBytes) {
    std::cerr << "transfer byte count overflows output allocation\n";
    return 1;
  }
  const std::uint64_t allocation_bytes = options.bytes + kCanaryBytes;
  constexpr std::size_t kActiveDescriptorTarget = 4;
  const std::size_t active_descriptor_count =
      static_cast<std::size_t>(std::min<std::uint64_t>(
          options.bytes, kActiveDescriptorTarget));
  const std::size_t request_count = active_descriptor_count + 1;
  DeviceAllocations allocations;
  if (!AllocateDeviceInputs(allocation_bytes, active_descriptor_count,
                            request_count, &allocations)) {
    return 1;
  }

  std::vector<std::uint8_t> initial_output(allocation_bytes, kCanary);
  if (!CheckAcl(
          aclrtMemcpy(allocations.output, allocation_bytes,
                      initial_output.data(), allocation_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE),
          "initialize output canary")) {
    return 1;
  }

  int result = 1;
  {
    HcommUrmaResources resources;
    ImportedMemoryView imported{};
    AivDirectUrmaChannel direct_channel{};
    std::uint32_t output_registration = 0;
    if (!CheckStatus(resources.OpenEndpoint(MakeDeviceEndpoint(options)),
                     "open Device endpoint") ||
        !CheckStatus(resources.RegisterMemory(
                         "kvaf_device_hbm",
                         reinterpret_cast<std::uint64_t>(allocations.output),
                         allocation_bytes, MemoryLocation::kDeviceHbm,
                         &output_registration),
                     "register Device HBM")) {
      return 1;
    }

    if (!CheckStatus(resources.CreateChannel(
                         MakeHostEndpoint(options), ChannelEngine::kAiv,
                         ChannelRole::kServer),
                     "create Device AIV channel")) {
      return 1;
    }
    bool server_ready = false;
    if (!CheckStatus(resources.PollChannelReady(&server_ready),
                     "arm Device AIV server channel")) {
      return 1;
    }
    const std::string server_state =
        server_ready ? "READY" : "CONNECTING";
    if (!AtomicWriteText(
            JoinPath(options.exchange_dir, kDeviceChannelCreatedMarker),
            "KVAF_A5_CHANNEL_CREATED role=device engine=AIV "
            "socket_role=SERVER state=" +
                server_state + "\n")) {
      std::cerr << "failed to publish Device server channel marker\n";
      return 1;
    }
    std::cout << "KVAF_A5_CHANNEL_CREATED role=device engine=AIV "
                 "socket_role=SERVER state="
              << server_state << '\n';
    if (!CheckStatus(
            resources.WaitChannelReady(options.timeout_seconds * 1000U),
            "wait Device AIV channel")) {
      return 1;
    }
    std::cout << "KVAF_A5_CHANNEL_READY role=device engine=AIV\n";
    if (!CheckStatus(
            resources.ImportMemory(
                descriptor, MemoryLocation::kHostDdr, &imported),
            "import Host DDR MR")) {
      return 1;
    }
    if (imported.location != MemoryLocation::kHostDdr ||
        imported.bytes < options.bytes || imported.address == 0) {
      std::cerr << "imported Host DDR view is invalid\n";
      return 1;
    }
    if (imported.address >
        std::numeric_limits<std::uint64_t>::max() - options.bytes) {
      std::cerr << "imported Host DDR range wraps the address space\n";
      return 1;
    }
    std::cout << "KVAF_A5_DDR_IMPORTED"
              << " npu_view=0x" << std::hex << imported.address << std::dec
              << " bytes=" << imported.bytes << " type=HOST\n";
    if (!CheckStatus(
            resources.SnapshotDirectChannel(
                reinterpret_cast<std::uint64_t>(allocations.output),
                allocation_bytes, imported.address, options.bytes,
                &direct_channel),
            "snapshot direct AIV URMA channel")) {
      return 1;
    }
    std::cout << "KVAF_A5_DIRECT_URMA_READY"
              << " sq_depth=" << direct_channel.sq_depth
              << " cq_depth=" << direct_channel.cq_depth
              << " wqe_size=" << direct_channel.wqe_size
              << " cqe_size=" << direct_channel.cqe_size
              << " tp_id=" << direct_channel.tp_id
              << " token_id=" << direct_channel.remote_token_id << '\n';

    std::vector<std::uint64_t> remote_addresses(active_descriptor_count);
    std::vector<std::uint64_t> lengths(active_descriptor_count);
    std::vector<std::uint64_t> table_output_offsets(
        active_descriptor_count);
    std::vector<std::uint8_t> valid(active_descriptor_count, 1);
    std::uint64_t consumed = 0;
    for (std::size_t table_index = 0;
         table_index < active_descriptor_count; ++table_index) {
      const std::uint64_t entries_left =
          active_descriptor_count - table_index;
      const std::uint64_t segment_bytes =
          (options.bytes - consumed) / entries_left;
      remote_addresses[table_index] = imported.address + consumed;
      lengths[table_index] = segment_bytes;
      table_output_offsets[table_index] = consumed;
      consumed += segment_bytes;
    }

    std::vector<std::uint32_t> indices(request_count);
    std::vector<std::uint8_t> mask(request_count, 1);
    std::vector<std::uint64_t> output_offsets(request_count);
    for (std::size_t request = 0; request < active_descriptor_count;
         ++request) {
      const std::size_t table_index =
          active_descriptor_count - request - 1;
      indices[request] = static_cast<std::uint32_t>(table_index);
      output_offsets[request] = table_output_offsets[table_index];
    }
    indices.back() = 0;
    mask.back() = 0;
    output_offsets.back() = std::numeric_limits<std::uint64_t>::max();

    const std::vector<UrmaH2DDescriptor> empty_descriptors(request_count);
    const DeviceCompletion empty_completion{};

    DeviceBatchGetParams params{};
    params.table.remote_addresses =
        reinterpret_cast<std::uint64_t>(allocations.remote_addresses);
    params.table.lengths =
        reinterpret_cast<std::uint64_t>(allocations.lengths);
    params.table.valid = reinterpret_cast<std::uint64_t>(allocations.valid);
    params.table.capacity =
        static_cast<std::uint32_t>(active_descriptor_count);
    params.table.generation = 1;
    params.indices = reinterpret_cast<std::uint64_t>(allocations.indices);
    params.mask = reinterpret_cast<std::uint64_t>(allocations.mask);
    params.output_offsets =
        reinterpret_cast<std::uint64_t>(allocations.output_offsets);
    params.descriptors =
        reinterpret_cast<std::uint64_t>(allocations.descriptors);
    params.output_base =
        reinterpret_cast<std::uint64_t>(allocations.output);
    params.output_bytes = options.bytes;
    params.entry_count = static_cast<std::uint32_t>(request_count);
    params.descriptor_capacity =
        static_cast<std::uint32_t>(request_count);
    params.generation = 1;

    if (!CopyHostToDevice(allocations.remote_addresses,
                          remote_addresses.data(),
                          active_descriptor_count) ||
        !CopyHostToDevice(allocations.lengths, lengths.data(),
                          active_descriptor_count) ||
        !CopyHostToDevice(allocations.valid, valid.data(),
                          active_descriptor_count) ||
        !CopyHostToDevice(allocations.indices, indices.data(),
                          request_count) ||
        !CopyHostToDevice(allocations.mask, mask.data(), request_count) ||
        !CopyHostToDevice(allocations.output_offsets,
                          output_offsets.data(), request_count) ||
        !CopyHostToDevice(allocations.descriptors,
                          empty_descriptors.data(), request_count) ||
        !CopyHostToDevice(allocations.completion, &empty_completion, 1) ||
        !CopyHostToDevice(allocations.params, &params, 1) ||
        !CopyHostToDevice(allocations.direct_channel, &direct_channel, 1)) {
      return 1;
    }

    aclrtStream stream = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) {
      return 1;
    }
    kvaf::access::a5::LaunchAivSimtUrmaH2D(
        stream, allocations.params, allocations.completion,
        allocations.direct_channel);
    const bool synchronized =
        CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    const bool stream_destroyed =
        CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");
    if (!synchronized || !stream_destroyed) {
      return 1;
    }

    DeviceCompletion completion{};
    AivDirectUrmaChannel completed_channel{};
    AivDirectUrmaChannel hcomm_channel_after{};
    std::vector<UrmaH2DDescriptor> completed_descriptors(request_count);
    std::vector<std::uint8_t> output(allocation_bytes);
    if (!CopyDeviceToHost(&completion, allocations.completion, 1) ||
        !CopyDeviceToHost(&completed_channel,
                          allocations.direct_channel, 1) ||
        !CopyDeviceToHost(completed_descriptors.data(),
                          allocations.descriptors, request_count) ||
        !CheckAcl(
            aclrtMemcpy(output.data(), allocation_bytes, allocations.output,
                        allocation_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
            "copy output to Host")) {
      return 1;
    }
    if (!CheckStatus(
            resources.SnapshotDirectChannel(
                reinterpret_cast<std::uint64_t>(allocations.output),
                allocation_bytes, imported.address, options.bytes,
                &hcomm_channel_after),
            "resnapshot direct AIV URMA channel after completion")) {
      return 1;
    }

    bool valid_result =
        completion.state ==
            static_cast<std::uint32_t>(DeviceBatchState::kComplete) &&
        completion.status ==
            static_cast<std::uint32_t>(DeviceBatchStatus::kOk) &&
        completion.planned_count == active_descriptor_count &&
        completion.submitted_count == active_descriptor_count &&
        completion.completed_count == active_descriptor_count &&
        completion.first_failed_index == kvaf::access::kNoFailedIndex &&
        completed_channel.sq_head - direct_channel.sq_head ==
            active_descriptor_count &&
        completed_channel.wqe_count - direct_channel.wqe_count ==
            active_descriptor_count &&
        completed_channel.cq_tail - direct_channel.cq_tail ==
            active_descriptor_count &&
        completed_channel.wqe_count == completed_channel.cq_tail &&
        hcomm_channel_after.sq_head == completed_channel.sq_head &&
        hcomm_channel_after.wqe_count == completed_channel.wqe_count &&
        hcomm_channel_after.cq_tail == completed_channel.cq_tail;
    for (std::size_t request = 0;
         request < active_descriptor_count && valid_result; ++request) {
      const std::size_t table_index = indices[request];
      const auto& completed_descriptor = completed_descriptors[request];
      valid_result =
          completed_descriptor.state ==
              static_cast<std::uint32_t>(TransferState::kComplete) &&
          completed_descriptor.logical_index == request &&
          completed_descriptor.remote_addr ==
              remote_addresses[table_index] &&
          completed_descriptor.local_addr ==
              reinterpret_cast<std::uint64_t>(allocations.output) +
                  output_offsets[request] &&
          completed_descriptor.bytes == lengths[table_index];
    }
    valid_result =
        valid_result &&
        completed_descriptors.back().state ==
            static_cast<std::uint32_t>(TransferState::kSkipped);
    for (std::uint64_t byte = 0; byte < options.bytes && valid_result;
         ++byte) {
      valid_result = output[byte] == Pattern(byte);
    }
    for (std::uint64_t byte = options.bytes;
         byte < allocation_bytes && valid_result; ++byte) {
      valid_result = output[byte] == kCanary;
    }
    if (!valid_result) {
      std::cerr << "H2D data, completion, descriptor, or canary validation "
                   "failed: state="
                << completion.state << " status=" << completion.status
                << " planned=" << completion.planned_count
                << " submitted=" << completion.submitted_count
                << " completed=" << completion.completed_count
                << " first_failed=" << completion.first_failed_index << '\n';
      return 1;
    }

    if (!CheckStatus(resources.Close(), "close Device HCOMM resources")) {
      return 1;
    }
    if (!AtomicWriteText(JoinPath(options.exchange_dir, "device.done"),
                         "KVAF_A5_H2D_SMOKE_OK\n")) {
      std::cerr << "failed to publish Device completion marker\n";
      return 1;
    }
    std::cout << "KVAF_A5_H2D_SMOKE_OK"
              << " bytes=" << options.bytes
              << " planned=" << active_descriptor_count
              << " submitted=" << active_descriptor_count
              << " completed=" << active_descriptor_count
              << " masked=1 canary=PASS\n";
    result = 0;
  }
  if (!allocations.Close()) {
    std::cerr << "one or more aclrtFree calls failed\n";
    result = 1;
  }
  if (result == 0) {
    std::cout << "KVAF_A5_TEARDOWN_OK role=device\n";
  }
  return result;
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
    aclFinalize();
    return 1;
  }

  const int result =
      options.role == "host" ? RunHost(options) : RunDevice(options);
  const bool reset =
      CheckAcl(aclrtResetDevice(options.logical_device_id),
               "aclrtResetDevice");
  const bool finalized = CheckAcl(aclFinalize(), "aclFinalize");
  return result == 0 && reset && finalized ? 0 : 1;
}
