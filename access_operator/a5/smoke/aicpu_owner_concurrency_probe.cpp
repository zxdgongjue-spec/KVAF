#include <acl/acl.h>
#include <runtime/rt_external_kernel.h>

#include <dlfcn.h>

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

#include <unistd.h>

#include "kvaf/access_operator/a5/aiv_simt_urma_h2d_launcher.h"
#include "kvaf_urma_provider_abi.h"

namespace {

using kvaf::access::a5::AicpuOwnerProbeAivResult;
using kvaf::access::a5::AicpuOwnerProbeAivStatus;

constexpr std::uint32_t kAicpuKfcKernelType = 5U;
constexpr std::uint32_t kDefaultTimeoutSeconds = 5U;
constexpr std::uint32_t kMaximumTimeoutSeconds = 10U;
constexpr std::uint16_t kRuntimeTimeoutGraceSeconds = 5U;
constexpr std::uint32_t kOwnerPollIntervalMicroseconds = 1000U;
// READY is published before the AICPU owner starts its bounded wait.  This
// generous finite budget avoids depending on an AIV clock-frequency contract.
constexpr std::uint32_t kAivReadyPollBudget = 20000000U;
constexpr char kDefaultProviderSoname[] =
    "libkvaf_urma_provider_device.so";
constexpr char kOwnerEntry[] = "aicpu_kvaf_urma_owner_probe_v1";

static_assert(kMaximumTimeoutSeconds + kRuntimeTimeoutGraceSeconds <=
                  std::numeric_limits<std::uint16_t>::max(),
              "owner probe runtime timeout must fit the CANN ABI");

static_assert(sizeof(rtAicpuArgsEx_t) == 48,
              "unexpected CANN rtAicpuArgsEx ABI");
static_assert(offsetof(rtAicpuArgsEx_t, args) == 0 &&
                  offsetof(rtAicpuArgsEx_t, argsSize) == 24 &&
                  offsetof(rtAicpuArgsEx_t, soNameAddrOffset) == 32 &&
                  offsetof(rtAicpuArgsEx_t, kernelNameAddrOffset) == 36,
              "unexpected CANN rtAicpuArgsEx field layout");

struct Options {
  std::int32_t logical_device_id{-1};
  std::uint32_t timeout_seconds{kDefaultTimeoutSeconds};
  std::string provider_soname{kDefaultProviderSoname};
};

bool IsProviderBasename(std::string_view name) {
  return !name.empty() && name.size() < 32U &&
         name.find('/') == std::string_view::npos &&
         name.find('\\') == std::string_view::npos &&
         name != "." && name != "..";
}

bool ParseUnsigned(std::string_view text, std::uint32_t* value) {
  if (text.empty() || value == nullptr) {
    return false;
  }
  std::uint32_t parsed = 0;
  const auto result = std::from_chars(
      text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} ||
      result.ptr != text.data() + text.size()) {
    return false;
  }
  *value = parsed;
  return true;
}

bool ParseOptions(int argc, char** argv, Options* options) {
  if (options == nullptr) {
    return false;
  }
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--logical-device-id" && index + 1 < argc) {
      std::uint32_t device = 0;
      if (!ParseUnsigned(argv[++index], &device) ||
          device > static_cast<std::uint32_t>(
                       std::numeric_limits<std::int32_t>::max())) {
        return false;
      }
      options->logical_device_id = static_cast<std::int32_t>(device);
    } else if (argument == "--timeout-seconds" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], &options->timeout_seconds)) {
        return false;
      }
    } else if (argument == "--provider-soname" && index + 1 < argc) {
      options->provider_soname = argv[++index];
      if (!IsProviderBasename(options->provider_soname)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return options->logical_device_id >= 0 &&
         options->timeout_seconds >= 1U &&
         options->timeout_seconds <= kMaximumTimeoutSeconds &&
         IsProviderBasename(options->provider_soname);
}

void PrintUsage(const char* program) {
  std::cerr << "usage: " << program
            << " --logical-device-id N [--timeout-seconds 1..10]"
               " [--provider-soname BASENAME]\n";
}

std::uint64_t NewSessionId() {
  const auto ticks = static_cast<std::uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  std::uint64_t session =
      ticks ^ (static_cast<std::uint64_t>(::getpid()) << 32U) ^
      UINT64_C(0x9e3779b97f4a7c15);
  return session == 0 ? 1 : session;
}

bool CopyFixed(const char* source, char* destination, std::size_t capacity) {
  const std::size_t length = std::strlen(source);
  if (length == 0 || length >= capacity) {
    return false;
  }
  std::memcpy(destination, source, length + 1U);
  return true;
}

void ReportAclFailure(const char* phase, const char* operation,
                      aclError result) {
  std::cerr << "KVAF_A5_AICPU_OWNER_PROBE_FAILED phase=" << phase
            << " operation=" << operation << " acl_ret=" << result << '\n';
}

class ProbeResources {
 public:
  using AicpuLaunch = decltype(&rtAicpuKernelLaunchExWithArgs);

  ~ProbeResources() {
    // A failure before the explicit joins is still bounded by the AICPU-side
    // timeout.  Try the AIV stream first because it is the producer that
    // releases the owner, then join the owner before freeing shared GM.
    if (aiv_submitted && !aiv_quiesced && aiv_stream != nullptr) {
      aiv_quiesced = aclrtSynchronizeStream(aiv_stream) == ACL_SUCCESS;
    }
    if (owner_submitted && !owner_quiesced && owner_stream != nullptr) {
      owner_quiesced =
          aclrtSynchronizeStream(owner_stream) == ACL_SUCCESS;
    }

    const bool shared_memory_safe =
        (!aiv_submitted || aiv_quiesced) &&
        (!owner_submitted || owner_quiesced);
    if (shared_memory_safe) {
      Free(result);
      Free(status);
      Free(command);
    } else {
      std::cerr
          << "KVAF_A5_AICPU_OWNER_PROBE_CLEANUP_DEFERRED "
             "reason=stream_quiescence_unknown\n";
    }

    if ((!aiv_submitted || aiv_quiesced) && aiv_stream != nullptr) {
      (void)aclrtDestroyStream(aiv_stream);
      aiv_stream = nullptr;
    }
    if ((!owner_submitted || owner_quiesced) && owner_stream != nullptr) {
      (void)aclrtDestroyStream(owner_stream);
      owner_stream = nullptr;
    }
    if (runtime_library != nullptr && shared_memory_safe) {
      (void)dlclose(runtime_library);
      runtime_library = nullptr;
      aicpu_launch = nullptr;
    }
    // Never reset/detach the ACL context while a bounded owner invocation is
    // still potentially executing against these GM records.  Process exit is
    // the final fallback for this resource-free probe; it does not own URMA.
    if (device_set && shared_memory_safe) {
      (void)aclrtResetDevice(device_id);
      device_set = false;
    }
    if (acl_initialized && shared_memory_safe) {
      (void)aclFinalize();
      acl_initialized = false;
    }
  }

  template <typename T>
  static void Free(T*& pointer) {
    if (pointer != nullptr && aclrtFree(pointer) == ACL_SUCCESS) {
      pointer = nullptr;
    }
  }

  bool acl_initialized{false};
  bool device_set{false};
  std::int32_t device_id{-1};
  void* runtime_library{nullptr};
  AicpuLaunch aicpu_launch{nullptr};
  aclrtStream owner_stream{nullptr};
  aclrtStream aiv_stream{nullptr};
  kvaf_urma_owner_probe_command_v1* command{nullptr};
  kvaf_urma_owner_probe_status_v1* status{nullptr};
  AicpuOwnerProbeAivResult* result{nullptr};
  bool owner_submitted{false};
  bool owner_quiesced{false};
  bool aiv_submitted{false};
  bool aiv_quiesced{false};
};

template <typename T>
bool AllocateAndZero(T** destination, const char* label) {
  const auto allocation = aclrtMalloc(
      reinterpret_cast<void**>(destination), sizeof(T),
      ACL_MEM_MALLOC_HUGE_FIRST);
  if (allocation != ACL_SUCCESS) {
    ReportAclFailure("allocate", label, allocation);
    return false;
  }
  const T empty{};
  const auto copy = aclrtMemcpy(
      *destination, sizeof(T), &empty, sizeof(T),
      ACL_MEMCPY_HOST_TO_DEVICE);
  if (copy != ACL_SUCCESS) {
    ReportAclFailure("initialize", label, copy);
    return false;
  }
  return true;
}

int RunProbe(const Options& options) {
  ProbeResources resources;
  resources.device_id = options.logical_device_id;

  aclError acl_result = aclInit(nullptr);
  if (acl_result != ACL_SUCCESS) {
    ReportAclFailure("acl_init", "aclInit", acl_result);
    return 1;
  }
  resources.acl_initialized = true;
  acl_result = aclrtSetDevice(options.logical_device_id);
  if (acl_result != ACL_SUCCESS) {
    ReportAclFailure("set_device", "aclrtSetDevice", acl_result);
    return 1;
  }
  resources.device_set = true;

  resources.runtime_library = dlopen(
      "libascendcl.so", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (resources.runtime_library == nullptr) {
    const char* error = dlerror();
    std::cerr << "KVAF_A5_AICPU_OWNER_PROBE_FAILED phase=load_runtime "
              << "error=" << (error == nullptr ? "unknown" : error)
              << '\n';
    return 1;
  }
  resources.aicpu_launch =
      reinterpret_cast<ProbeResources::AicpuLaunch>(dlsym(
          resources.runtime_library, "rtAicpuKernelLaunchExWithArgs"));
  if (resources.aicpu_launch == nullptr) {
    std::cerr << "KVAF_A5_AICPU_OWNER_PROBE_FAILED phase=load_symbol\n";
    return 1;
  }

  acl_result = aclrtCreateStream(&resources.owner_stream);
  if (acl_result != ACL_SUCCESS) {
    ReportAclFailure("create_owner_stream", "aclrtCreateStream", acl_result);
    return 1;
  }
  acl_result = aclrtCreateStream(&resources.aiv_stream);
  if (acl_result != ACL_SUCCESS) {
    ReportAclFailure("create_aiv_stream", "aclrtCreateStream", acl_result);
    return 1;
  }
  if (resources.owner_stream == resources.aiv_stream) {
    std::cerr << "KVAF_A5_AICPU_OWNER_PROBE_FAILED "
                 "phase=distinct_stream_check\n";
    return 1;
  }

  if (!AllocateAndZero(&resources.command, "owner_command") ||
      !AllocateAndZero(&resources.status, "owner_status") ||
      !AllocateAndZero(&resources.result, "aiv_result")) {
    return 1;
  }
  constexpr std::uintptr_t kCacheLineMask = 63U;
  const auto command_address =
      reinterpret_cast<std::uintptr_t>(resources.command);
  const auto status_address =
      reinterpret_cast<std::uintptr_t>(resources.status);
  const auto result_address =
      reinterpret_cast<std::uintptr_t>(resources.result);
  if ((command_address & kCacheLineMask) != 0 ||
      (status_address & kCacheLineMask) != 0 ||
      (result_address & kCacheLineMask) != 0 ||
      command_address == status_address ||
      command_address == result_address ||
      status_address == result_address) {
    std::cerr << "KVAF_A5_AICPU_OWNER_PROBE_FAILED "
                 "phase=cache_line_ownership\n";
    return 1;
  }

  const std::uint64_t session_id = NewSessionId();
  kvaf_urma_aicpu_owner_probe_args_v1 owner_args{};
  if (!CopyFixed(kOwnerEntry, owner_args.kernel_name,
                 sizeof(owner_args.kernel_name)) ||
      !CopyFixed(options.provider_soname.c_str(), owner_args.so_name,
                 sizeof(owner_args.so_name))) {
    std::cerr << "KVAF_A5_AICPU_OWNER_PROBE_FAILED phase=build_args\n";
    return 1;
  }
  owner_args.magic = KVAF_URMA_OWNER_PROBE_MAGIC;
  owner_args.abi_version = KVAF_URMA_OWNER_PROBE_ABI_VERSION;
  owner_args.struct_size = sizeof(owner_args);
  owner_args.timeout_milliseconds = options.timeout_seconds * 1000U;
  owner_args.session_id = session_id;
  owner_args.command_address = command_address;
  owner_args.status_address = status_address;
  owner_args.poll_interval_microseconds =
      kOwnerPollIntervalMicroseconds;

  rtAicpuArgsEx_t launch_args{};
  launch_args.args = &owner_args;
  launch_args.argsSize = sizeof(owner_args);
  launch_args.soNameAddrOffset =
      offsetof(kvaf_urma_aicpu_owner_probe_args_v1, so_name);
  launch_args.kernelNameAddrOffset =
      offsetof(kvaf_urma_aicpu_owner_probe_args_v1, kernel_name);
  // The provider's monotonic deadline is the normal exit path.  The runtime
  // timeout is a second, independent bound for a stuck AICPU task.  This
  // probe owns no URMA/UMMU state, so forced task termination cannot orphan
  // transport handles.
  launch_args.timeout = static_cast<std::uint16_t>(
      options.timeout_seconds + kRuntimeTimeoutGraceSeconds);

  // A nonzero runtime return does not prove that the request was never
  // observed.  Arm cleanup before crossing the launch ABI so the destructor
  // always joins this stream before releasing the mailbox.
  resources.owner_submitted = true;
  const auto launch_result = resources.aicpu_launch(
      kAicpuKfcKernelType, "kvaf_aicpu_owner_probe", 1,
      &launch_args, nullptr,
      reinterpret_cast<rtStream_t>(resources.owner_stream), 0);
  if (launch_result != 0) {
    std::cerr << "KVAF_A5_AICPU_OWNER_PROBE_FAILED phase=launch_owner "
              << "runtime_ret=" << launch_result << '\n';
    return 1;
  }

  // Do not synchronize owner_stream here: the owner waits for this AIV task.
  // Arm the same conservative cleanup rule before the device launch.
  resources.aiv_submitted = true;
  kvaf::access::a5::LaunchAicpuOwnerConcurrencyProbe(
      resources.aiv_stream, resources.command, resources.status,
      resources.result, session_id, kAivReadyPollBudget);
  acl_result = aclrtSynchronizeStream(resources.aiv_stream);
  if (acl_result != ACL_SUCCESS) {
    ReportAclFailure("sync_aiv", "aclrtSynchronizeStream", acl_result);
    return 1;
  }
  resources.aiv_quiesced = true;

  acl_result = aclrtSynchronizeStream(resources.owner_stream);
  if (acl_result != ACL_SUCCESS) {
    ReportAclFailure("sync_owner", "aclrtSynchronizeStream", acl_result);
    return 1;
  }
  resources.owner_quiesced = true;

  kvaf_urma_owner_probe_command_v1 command{};
  kvaf_urma_owner_probe_status_v1 status{};
  AicpuOwnerProbeAivResult result{};
  if ((acl_result = aclrtMemcpy(
           &command, sizeof(command), resources.command, sizeof(command),
           ACL_MEMCPY_DEVICE_TO_HOST)) != ACL_SUCCESS) {
    ReportAclFailure("read_command", "aclrtMemcpy", acl_result);
    return 1;
  }
  if ((acl_result = aclrtMemcpy(
           &status, sizeof(status), resources.status, sizeof(status),
           ACL_MEMCPY_DEVICE_TO_HOST)) != ACL_SUCCESS) {
    ReportAclFailure("read_status", "aclrtMemcpy", acl_result);
    return 1;
  }
  if ((acl_result = aclrtMemcpy(
           &result, sizeof(result), resources.result, sizeof(result),
           ACL_MEMCPY_DEVICE_TO_HOST)) != ACL_SUCCESS) {
    ReportAclFailure("read_aiv_result", "aclrtMemcpy", acl_result);
    return 1;
  }

  const bool command_ok =
      command.magic == KVAF_URMA_OWNER_PROBE_MAGIC &&
      command.abi_version == KVAF_URMA_OWNER_PROBE_ABI_VERSION &&
      command.struct_size == sizeof(command) &&
      command.opcode == KVAF_URMA_OWNER_PROBE_COMMAND_RELEASE &&
      command.session_id == session_id &&
      command.sequence == KVAF_URMA_OWNER_PROBE_RELEASE_SEQUENCE;
  const bool status_ok =
      status.magic == KVAF_URMA_OWNER_PROBE_MAGIC &&
      status.abi_version == KVAF_URMA_OWNER_PROBE_ABI_VERSION &&
      status.struct_size == sizeof(status) &&
      status.state == KVAF_URMA_OWNER_PROBE_STATE_RELEASED &&
      status.session_id == session_id &&
      status.sequence == KVAF_URMA_OWNER_PROBE_RELEASE_SEQUENCE &&
      status.result_code == KVAF_URMA_OWNER_PROBE_RESULT_OK;
  const bool aiv_result_ok =
      result.magic == kvaf::access::a5::kAicpuOwnerProbeAivResultMagic &&
      result.abi_version ==
          kvaf::access::a5::kAicpuOwnerProbeAivResultAbiVersion &&
      result.struct_size == sizeof(result) &&
      result.status == static_cast<std::uint32_t>(
          AicpuOwnerProbeAivStatus::kReleaseSubmitted) &&
      result.session_id == session_id &&
      result.observed_sequence == 0 &&
      result.observed_owner_state == KVAF_URMA_OWNER_PROBE_STATE_READY &&
      result.poll_count != 0 && result.poll_count <= kAivReadyPollBudget;
  if (!command_ok || !status_ok || !aiv_result_ok) {
    std::cerr
        << "KVAF_A5_AICPU_OWNER_PROBE_FAILED phase=validate_terminal"
        << " command_opcode=" << command.opcode
        << " command_sequence=" << command.sequence
        << " owner_state=" << status.state
        << " owner_sequence=" << status.sequence
        << " owner_result=" << status.result_code
        << " aiv_status=" << result.status
        << " aiv_observed_state=" << result.observed_owner_state
        << " aiv_polls=" << result.poll_count << '\n';
    return 1;
  }

  std::cout << "KVAF_A5_AICPU_OWNER_PROBE_OK"
            << " logical_device_id=" << options.logical_device_id
            << " timeout_seconds=" << options.timeout_seconds
            << " session_id=" << session_id
            << " aiv_polls=" << result.poll_count
            << " owner_elapsed_ns=" << status.elapsed_nanoseconds << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }
  return RunProbe(options);
}
