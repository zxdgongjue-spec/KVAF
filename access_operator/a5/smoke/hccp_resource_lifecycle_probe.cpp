#include "kvaf/access_operator/a5/hcomm_urma_resources.h"
#include "kvaf/access_operator/a5/host_ddr_mapping.h"

#include <acl/acl.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using kvaf::Status;
using kvaf::access::a5::EndpointLocation;
using kvaf::access::a5::EndpointSpec;
using kvaf::access::a5::HcommUrmaResources;
using kvaf::access::a5::HostDdrDeviceMapping;
using kvaf::access::a5::MemoryLocation;

constexpr std::uint32_t kDefaultTeardownRetries = 3;
constexpr std::uint32_t kTeardownRetryDelayMs = 50;

enum class ProbePhase {
  kHostDdr,
  kDeviceHbm,
};

struct Options {
  ProbePhase phase{ProbePhase::kHostDdr};
  std::array<std::uint8_t, 16> eid{};
  std::uint32_t logical_device_id{0};
  std::uint32_t physical_device_id{0};
  std::uint32_t super_device_id{0};
  std::uint32_t server_index{0};
  std::uint32_t super_pod_index{0};
  std::uint32_t teardown_retries{kDefaultTeardownRetries};
  std::uint64_t bytes{4096};
};

struct RunResult {
  int code{1};
  // False means that the process deliberately retained a potentially live
  // mapping/resource rather than freeing memory underneath an unknown DMA
  // reference.  The caller must not treat this as a normal retryable result.
  bool safe_to_finalize{true};
};

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

void PrintUsage(const char* program) {
  std::cerr
      << "usage: " << program
      << " --phase host-ddr|device-hbm --eid HEX32"
         " --logical-device-id N --physical-device-id N"
         " [--super-device-id N] [--server-index N]"
         " [--super-pod-index N] [--bytes N]"
         " [--teardown-retries N]\n";
}

bool ParseOptions(int argc, char** argv, Options* options) {
  if (options == nullptr) {
    return false;
  }
  std::string eid;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      return false;
    }
    const std::string key = argv[index];
    const std::string value = argv[index + 1];
    std::uint64_t parsed = 0;
    if (key == "--phase") {
      if (value == "host-ddr") {
        options->phase = ProbePhase::kHostDdr;
      } else if (value == "device-hbm") {
        options->phase = ProbePhase::kDeviceHbm;
      } else {
        return false;
      }
    } else if (key == "--eid") {
      eid = value;
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
    } else if (key == "--bytes" && parsed != 0) {
      options->bytes = parsed;
    } else if (key == "--teardown-retries" && parsed != 0 &&
               parsed <= std::numeric_limits<std::uint32_t>::max()) {
      options->teardown_retries = static_cast<std::uint32_t>(parsed);
    } else {
      return false;
    }
  }
  return ParseEid(eid, &options->eid);
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

EndpointSpec MakeEndpoint(const Options& options) {
  EndpointSpec endpoint{};
  endpoint.eid = options.eid;
  endpoint.physical_device_id = options.physical_device_id;
  if (options.phase == ProbePhase::kHostDdr) {
    endpoint.location = EndpointLocation::kHost;
  } else {
    endpoint.location = EndpointLocation::kDevice;
    endpoint.super_device_id = options.super_device_id;
    endpoint.server_index = options.server_index;
    endpoint.super_pod_index = options.super_pod_index;
  }
  return endpoint;
}

template <typename Close>
bool CloseWithRetry(const char* subject, std::uint32_t retries,
                    Close&& close) {
  for (std::uint32_t attempt = 1; attempt <= retries; ++attempt) {
    const Status status = close();
    if (status.ok()) {
      std::cout << "KVAF_A5_HCCP_RESOURCE_TEARDOWN_OK subject=" << subject
                << " attempts=" << attempt << '\n';
      return true;
    }
    std::cerr << "KVAF_A5_HCCP_RESOURCE_TEARDOWN_RETRY subject=" << subject
              << " attempt=" << attempt << " error=" << status.message()
              << '\n';
    if (attempt != retries) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kTeardownRetryDelayMs));
    }
  }
  return false;
}

RunResult FailWithSafeCleanup(std::unique_ptr<HcommUrmaResources>* resources,
                              std::unique_ptr<HostDdrDeviceMapping>* mapping,
                              std::uint32_t retries) {
  bool resources_closed = true;
  if (resources != nullptr && resources->get() != nullptr) {
    resources_closed = CloseWithRetry(
        "hcomm-resources", retries, [&resources] { return (*resources)->Close(); });
  }
  if (!resources_closed) {
    // Do not unmap/free Host DDR while HCOMM might still own a DMA reference.
    // release() intentionally prevents a destructor from violating the
    // resource-before-memory teardown order during error unwinding.
    std::cerr << "KVAF_A5_HCCP_RESOURCE_TEARDOWN_UNPROVEN "
                 "action=retain-process-owned-resources\n";
    if (resources != nullptr) {
      (void)resources->release();
    }
    if (mapping != nullptr) {
      (void)mapping->release();
    }
    return {1, false};
  }
  if (mapping != nullptr && mapping->get() != nullptr &&
      !CloseWithRetry("host-ddr-mapping", retries,
                      [&mapping] { return (*mapping)->Close(); })) {
    std::cerr << "KVAF_A5_HCCP_RESOURCE_TEARDOWN_UNPROVEN "
                 "action=retain-host-ddr-mapping\n";
    (void)mapping->release();
    return {1, false};
  }
  return {1, true};
}

RunResult RunHostDdr(const Options& options) {
  auto mapping = std::make_unique<HostDdrDeviceMapping>();
  auto resources = std::make_unique<HcommUrmaResources>();
  if (!CheckStatus(mapping->AllocateAndRegister(options.bytes,
                                                 options.logical_device_id),
                   "allocate and map A5 Host DDR")) {
    return FailWithSafeCleanup(&resources, &mapping, options.teardown_retries);
  }
  std::cout << "KVAF_A5_HCCP_RESOURCE_DDR_REGISTERED host_hva=0x" << std::hex
            << mapping->host_address() << " mapped_dva=0x"
            << mapping->device_address() << std::dec
            << " bytes=" << mapping->bytes() << '\n';

  if (!CheckStatus(resources->OpenEndpoint(MakeEndpoint(options)),
                   "open HCCP Host endpoint")) {
    return FailWithSafeCleanup(&resources, &mapping, options.teardown_retries);
  }
  std::cout << "KVAF_A5_HCCP_RESOURCE_ENDPOINT_OPEN role=host\n";

  std::uint32_t registration = 0;
  if (!CheckStatus(resources->RegisterMemory(
                       "kvaf_hccp_probe_host_ddr", mapping->host_address(),
                       mapping->bytes(), MemoryLocation::kHostDdr,
                       &registration),
                   "register HCCP Host DDR MR")) {
    return FailWithSafeCleanup(&resources, &mapping, options.teardown_retries);
  }
  std::vector<std::uint8_t> descriptor;
  if (!CheckStatus(resources->ExportMemory(registration, &descriptor),
                   "export HCCP Host DDR MR") ||
      descriptor.empty()) {
    if (descriptor.empty()) {
      std::cerr << "export HCCP Host DDR MR returned an empty descriptor\n";
    }
    return FailWithSafeCleanup(&resources, &mapping, options.teardown_retries);
  }
  std::cout << "KVAF_A5_HCCP_RESOURCE_MR_EXPORTED role=host descriptor_bytes="
            << descriptor.size() << '\n';

  if (!CloseWithRetry("hcomm-resources", options.teardown_retries,
                      [&resources] { return resources->Close(); })) {
    std::cerr << "KVAF_A5_HCCP_RESOURCE_TEARDOWN_UNPROVEN "
                 "action=retain-process-owned-resources\n";
    (void)resources.release();
    (void)mapping.release();
    return {1, false};
  }
  if (!CloseWithRetry("host-ddr-mapping", options.teardown_retries,
                      [&mapping] { return mapping->Close(); })) {
    std::cerr << "KVAF_A5_HCCP_RESOURCE_TEARDOWN_UNPROVEN "
                 "action=retain-host-ddr-mapping\n";
    (void)mapping.release();
    return {1, false};
  }
  std::cout << "KVAF_A5_HCCP_RESOURCE_PROBE_OK phase=host-ddr\n";
  return {0, true};
}

RunResult RunDeviceHbm(const Options& options) {
  std::uint8_t* hbm = nullptr;
  if (!CheckAcl(aclrtMalloc(reinterpret_cast<void**>(&hbm), options.bytes,
                            ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(Device HBM)")) {
    return {1, true};
  }
  auto resources = std::make_unique<HcommUrmaResources>();
  if (!CheckStatus(resources->OpenEndpoint(MakeEndpoint(options)),
                   "open HCCP Device endpoint")) {
    if (aclrtFree(hbm) != ACL_SUCCESS) {
      std::cerr << "aclrtFree(Device HBM) failed after endpoint error\n";
      return {1, false};
    }
    return {1, true};
  }
  std::cout << "KVAF_A5_HCCP_RESOURCE_ENDPOINT_OPEN role=device\n";

  std::uint32_t registration = 0;
  if (!CheckStatus(resources->RegisterMemory(
                       "kvaf_hccp_probe_device_hbm",
                       reinterpret_cast<std::uint64_t>(hbm), options.bytes,
                       MemoryLocation::kDeviceHbm, &registration),
                   "register HCCP Device HBM MR")) {
    if (!CloseWithRetry("hcomm-resources", options.teardown_retries,
                        [&resources] { return resources->Close(); })) {
      std::cerr << "KVAF_A5_HCCP_RESOURCE_TEARDOWN_UNPROVEN "
                   "action=retain-device-hbm\n";
      (void)resources.release();
      return {1, false};
    }
    if (aclrtFree(hbm) != ACL_SUCCESS) {
      std::cerr << "aclrtFree(Device HBM) failed after registration error\n";
      return {1, false};
    }
    return {1, true};
  }
  std::cout << "KVAF_A5_HCCP_RESOURCE_MR_REGISTERED role=device bytes="
            << options.bytes << '\n';

  if (!CloseWithRetry("hcomm-resources", options.teardown_retries,
                      [&resources] { return resources->Close(); })) {
    std::cerr << "KVAF_A5_HCCP_RESOURCE_TEARDOWN_UNPROVEN "
                 "action=retain-device-hbm\n";
    (void)resources.release();
    return {1, false};
  }
  if (!CheckAcl(aclrtFree(hbm), "aclrtFree(Device HBM)")) {
    return {1, false};
  }
  std::cout << "KVAF_A5_HCCP_RESOURCE_TEARDOWN_OK subject=device-hbm"
               " attempts=1\n";
  std::cout << "KVAF_A5_HCCP_RESOURCE_PROBE_OK phase=device-hbm\n";
  return {0, true};
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }
  if (!CheckAcl(aclInit(nullptr), "aclInit") ||
      !CheckAcl(aclrtSetDevice(options.logical_device_id), "aclrtSetDevice")) {
    (void)aclFinalize();
    return 1;
  }

  std::cout << "KVAF_A5_HCCP_RESOURCE_PROBE_BEGIN phase="
            << (options.phase == ProbePhase::kHostDdr ? "host-ddr"
                                                       : "device-hbm")
            << " logical_device_id=" << options.logical_device_id
            << " physical_device_id=" << options.physical_device_id
            << " bytes=" << options.bytes
            << " teardown_retries=" << options.teardown_retries << '\n';

  const RunResult result = options.phase == ProbePhase::kHostDdr
                               ? RunHostDdr(options)
                               : RunDeviceHbm(options);
  if (!result.safe_to_finalize) {
    std::cerr << "KVAF_A5_HCCP_RESOURCE_PROBE_FINALIZE_SKIPPED "
                 "reason=teardown-unproven\n";
    return result.code;
  }
  // This is the ACL process-context release documented by CANN examples, not
  // an `npu-smi set -t reset` physical-card reset.  It is intentionally
  // reached only after HCOMM/HCCP and Host-DDR/HBM teardown is proven.
  if (!CheckAcl(aclrtResetDevice(options.logical_device_id),
                "aclrtResetDevice(clean process context)")) {
    (void)aclFinalize();
    return 1;
  }
  if (!CheckAcl(aclFinalize(), "aclFinalize")) {
    return 1;
  }
  return result.code;
}
