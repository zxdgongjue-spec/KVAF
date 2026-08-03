#include <acl/acl.h>
#include <runtime/rt_external_kernel.h>

#include <dlfcn.h>
#include <fcntl.h>

#include <cerrno>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <unistd.h>

#include "kvaf_urma_provider_abi.h"

namespace {

constexpr std::uint32_t kAicpuKfcKernelType = 5U;
constexpr char kDefaultProviderSoname[] =
    "libkvaf_urma_provider_device.so";
constexpr char kRuntimeProbeEntry[] =
    "aicpu_kvaf_urma_rt_probe_v2";
static_assert(sizeof(kRuntimeProbeEntry) <= KVAF_URMA_PROVIDER_NAME_BYTES,
              "AICPU runtime probe entry must include a terminating NUL");
static_assert(sizeof(kDefaultProviderSoname) <=
                  KVAF_URMA_PROVIDER_NAME_BYTES,
              "AICPU provider SONAME must include a terminating NUL");

static_assert(sizeof(rtAicpuArgsEx_t) == 48,
              "unexpected CANN rtAicpuArgsEx ABI");
static_assert(offsetof(rtAicpuArgsEx_t, args) == 0 &&
                  offsetof(rtAicpuArgsEx_t, hostInputInfoPtr) == 8 &&
                  offsetof(rtAicpuArgsEx_t, kernelOffsetInfoPtr) == 16 &&
                  offsetof(rtAicpuArgsEx_t, argsSize) == 24 &&
                  offsetof(rtAicpuArgsEx_t, hostInputInfoNum) == 28 &&
                  offsetof(rtAicpuArgsEx_t, kernelOffsetInfoNum) == 30 &&
                  offsetof(rtAicpuArgsEx_t, soNameAddrOffset) == 32 &&
                  offsetof(rtAicpuArgsEx_t, kernelNameAddrOffset) == 36 &&
                  offsetof(rtAicpuArgsEx_t, isNoNeedH2DCopy) == 40 &&
                  offsetof(rtAicpuArgsEx_t, timeout) == 42 &&
                  offsetof(rtAicpuArgsEx_t, reserved) == 44,
              "unexpected CANN rtAicpuArgsEx field layout");

struct Options {
  std::int32_t logical_device_id{-1};
  std::uint32_t eid_index{0};
  std::uint32_t queue_depth{
      KVAF_URMA_RUNTIME_PROBE_DEFAULT_QUEUE_DEPTH};
  std::uint32_t mode{KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER};
  std::string urma_device;
  std::string provider_soname{kDefaultProviderSoname};
  std::array<std::uint8_t, KVAF_URMA_RUNTIME_PROBE_BUILD_ID_BYTES>
      provider_build_id_sha256{};
};

int HexNibble(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  return -1;
}

bool ParseSha256(std::string_view text,
                 std::array<std::uint8_t,
                     KVAF_URMA_RUNTIME_PROBE_BUILD_ID_BYTES>* output) {
  if (output == nullptr || text.size() != output->size() * 2U) {
    return false;
  }
  bool any_nonzero = false;
  for (std::size_t index = 0; index < output->size(); ++index) {
    const int upper = HexNibble(text[index * 2U]);
    const int lower = HexNibble(text[index * 2U + 1U]);
    if (upper < 0 || lower < 0) {
      return false;
    }
    (*output)[index] = static_cast<std::uint8_t>(
        static_cast<unsigned int>(upper) << 4U |
        static_cast<unsigned int>(lower));
    any_nonzero = any_nonzero || (*output)[index] != 0U;
  }
  return any_nonzero;
}

std::string HexSha256(const std::uint8_t* digest) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(KVAF_URMA_RUNTIME_PROBE_BUILD_ID_BYTES * 2U, '0');
  for (std::size_t index = 0;
       index < KVAF_URMA_RUNTIME_PROBE_BUILD_ID_BYTES; ++index) {
    result[index * 2U] = kHex[digest[index] >> 4U];
    result[index * 2U + 1U] = kHex[digest[index] & 0x0fU];
  }
  return result;
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

bool IsBasename(std::string_view name) {
  return !name.empty() && name.size() < KVAF_URMA_PROVIDER_NAME_BYTES &&
         name.find('/') == std::string_view::npos &&
         name.find('\\') == std::string_view::npos &&
         name != "." && name != "..";
}

bool ParseOptions(int argc, char** argv, Options* options) {
  if (options == nullptr) {
    return false;
  }
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--logical-device-id" && index + 1 < argc) {
      std::uint32_t value = 0;
      if (!ParseUnsigned(argv[++index], &value) ||
          value > static_cast<std::uint32_t>(
                      std::numeric_limits<std::int32_t>::max())) {
        return false;
      }
      options->logical_device_id = static_cast<std::int32_t>(value);
    } else if (argument == "--urma-device" && index + 1 < argc) {
      options->urma_device = argv[++index];
    } else if (argument == "--eid-index" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], &options->eid_index)) {
        return false;
      }
    } else if (argument == "--queue-depth" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], &options->queue_depth)) {
        return false;
      }
    } else if (argument == "--provider-soname" && index + 1 < argc) {
      options->provider_soname = argv[++index];
    } else if (argument == "--provider-build-id" && index + 1 < argc) {
      if (!ParseSha256(argv[++index],
                       &options->provider_build_id_sha256)) {
        return false;
      }
    } else if (argument == "--mode" && index + 1 < argc) {
      const std::string_view mode(argv[++index]);
      if (mode == "discover") {
        options->mode = KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER;
      } else if (mode == "init-device") {
        options->mode = KVAF_URMA_RUNTIME_PROBE_MODE_INIT_DEVICE;
      } else if (mode == "create-delete") {
        options->mode = KVAF_URMA_RUNTIME_PROBE_MODE_CREATE_DELETE;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  return options->logical_device_id >= 0 &&
         IsBasename(options->provider_soname) &&
         options->provider_build_id_sha256 !=
             decltype(options->provider_build_id_sha256){} &&
         options->queue_depth != 0 &&
         options->queue_depth <= KVAF_URMA_RUNTIME_PROBE_MAX_QUEUE_DEPTH &&
         (options->queue_depth & (options->queue_depth - 1U)) == 0 &&
         (options->mode == KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER ||
          IsBasename(options->urma_device));
}

void PrintUsage(const char* program) {
  std::cerr
      << "usage: " << program
      << " --logical-device-id N --mode discover|init-device|create-delete"
         " [--urma-device NAME --eid-index N]"
         " [--queue-depth POWER_OF_TWO]"
         " --provider-soname BASENAME --provider-build-id SHA256\n";
}

bool CopyFixed(const std::string& source, char* destination,
               std::size_t capacity) {
  if (source.empty() || source.size() >= capacity) {
    return false;
  }
  std::memcpy(destination, source.c_str(), source.size() + 1U);
  return true;
}

bool FillRequestNonce(std::uint64_t nonce[2]) {
  if (nonce == nullptr) {
    return false;
  }
  const int descriptor = ::open("/dev/urandom", O_RDONLY);
  if (descriptor < 0) {
    return false;
  }
  auto* cursor = reinterpret_cast<unsigned char*>(nonce);
  std::size_t remaining = sizeof(std::uint64_t) * 2U;
  while (remaining != 0U) {
    const ssize_t count = ::read(descriptor, cursor, remaining);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      (void)::close(descriptor);
      return false;
    }
    cursor += static_cast<std::size_t>(count);
    remaining -= static_cast<std::size_t>(count);
  }
  if (::close(descriptor) != 0) {
    return false;
  }
  return nonce[0] != 0U || nonce[1] != 0U;
}

std::uint64_t ExpectedRequiredSymbolMask(std::uint32_t mode) {
  return mode == KVAF_URMA_RUNTIME_PROBE_MODE_CREATE_DELETE
      ? KVAF_URMA_RUNTIME_PROBE_LIFECYCLE_SYMBOLS
      : KVAF_URMA_RUNTIME_PROBE_BASE_SYMBOLS;
}

bool FixedStringHasTerminator(const char* text, std::size_t capacity) {
  return text != nullptr && std::memchr(text, '\0', capacity) != nullptr;
}

bool FixedBytesAreZero(const std::uint8_t* bytes, std::size_t size) {
  if (bytes == nullptr) {
    return false;
  }
  for (std::size_t index = 0; index < size; ++index) {
    if (bytes[index] != 0U) {
      return false;
    }
  }
  return true;
}

bool ValidateResultAgainstRequest(
    const Options& options,
    const kvaf_urma_aicpu_runtime_probe_args_v2& args,
    const kvaf_urma_runtime_probe_result_v2& result,
    const char** failure_reason) {
  auto fail = [&](const char* reason) {
    if (failure_reason != nullptr) {
      *failure_reason = reason;
    }
    return false;
  };
  if (result.magic != KVAF_URMA_RUNTIME_PROBE_MAGIC ||
      result.abi_version != KVAF_URMA_RUNTIME_PROBE_ABI_VERSION ||
      result.struct_size != sizeof(result)) {
    return fail("header");
  }
  if (result.request_nonce[0] != args.request_nonce[0] ||
      result.request_nonce[1] != args.request_nonce[1]) {
    return fail("nonce");
  }
  if (result.provider_abi_token !=
          KVAF_URMA_RUNTIME_PROBE_PROVIDER_ABI_TOKEN ||
      result.provider_abi_token != args.expected_provider_abi_token) {
    return fail("provider_abi_token");
  }
  if (result.mode != options.mode || result.eid_index != options.eid_index ||
      result.logical_device_id != options.logical_device_id ||
      result.queue_depth != options.queue_depth ||
      result.args_size != sizeof(args)) {
    return fail("request_echo");
  }
  if (result.outcome >
          KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH ||
      result.primary_outcome >
          KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH ||
      result.stage > KVAF_URMA_RUNTIME_PROBE_STAGE_COMPLETE) {
    return fail("enum_range");
  }
  if (result.outcome != KVAF_URMA_RUNTIME_PROBE_OUTCOME_CLEANUP_UNSAFE &&
      result.primary_outcome != result.outcome) {
    return fail("primary_outcome");
  }
  if ((result.flags & ~KVAF_URMA_RUNTIME_PROBE_ALL_FLAG_BITS) != 0U ||
      (result.cleanup_required &
       ~KVAF_URMA_RUNTIME_PROBE_ALL_CLEANUP_BITS) != 0U ||
      (result.cleanup_completed &
       ~KVAF_URMA_RUNTIME_PROBE_ALL_CLEANUP_BITS) != 0U ||
      (result.cleanup_completed & ~result.cleanup_required) != 0U) {
    return fail("bit_range");
  }
  if (result.operation_errno_valid > 1U ||
      result.compiled_urma_api_version == 0U) {
    return fail("scalar_range");
  }
  const std::uint64_t expected_required =
      ExpectedRequiredSymbolMask(options.mode);
  if (result.required_symbol_mask != expected_required ||
      (result.resolved_symbol_mask &
       ~KVAF_URMA_RUNTIME_PROBE_ALL_SYMBOLS) != 0U) {
    return fail("symbol_mask");
  }
  if (result.outcome == KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK &&
      (result.resolved_symbol_mask & expected_required) !=
          expected_required) {
    return fail("required_symbol_missing");
  }
  if (!FixedStringHasTerminator(result.runtime_library_path,
                                sizeof(result.runtime_library_path)) ||
      !FixedStringHasTerminator(result.loader_error,
                                sizeof(result.loader_error)) ||
      !FixedBytesAreZero(result.reserved, sizeof(result.reserved))) {
    return fail("string_or_reserved_bytes");
  }
  if (std::memcmp(args.expected_provider_build_id_sha256,
                  options.provider_build_id_sha256.data(),
                  sizeof(args.expected_provider_build_id_sha256)) != 0) {
    return fail("provider_build_id_request");
  }
  const bool provider_identity_matches =
      std::memcmp(result.provider_build_id_sha256,
                  args.expected_provider_build_id_sha256,
                  sizeof(result.provider_build_id_sha256)) == 0;
  if (result.outcome ==
          KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH) {
    if (provider_identity_matches ||
        FixedBytesAreZero(result.provider_build_id_sha256,
                          sizeof(result.provider_build_id_sha256)) ||
        result.stage != KVAF_URMA_RUNTIME_PROBE_STAGE_NONE ||
        result.cleanup_required != 0U || result.cleanup_completed != 0U ||
        result.resolved_symbol_mask != 0U ||
        (result.flags &
         KVAF_URMA_RUNTIME_PROBE_RESOURCE_OPERATION_ATTEMPTED) != 0U) {
      return fail("provider_build_id_mismatch_contract");
    }
  } else if (!provider_identity_matches) {
    return fail("provider_build_id");
  }
  if (result.outcome == KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK) {
    const std::uint32_t expected_cleanup =
        options.mode == KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER
            ? KVAF_URMA_RUNTIME_PROBE_CLEANUP_LIBRARY
            : (options.mode == KVAF_URMA_RUNTIME_PROBE_MODE_INIT_DEVICE
                   ? KVAF_URMA_RUNTIME_PROBE_CLEANUP_RUNTIME |
                         KVAF_URMA_RUNTIME_PROBE_CLEANUP_LIBRARY
                   : KVAF_URMA_RUNTIME_PROBE_ALL_CLEANUP_BITS);
    const bool resource_attempted =
        (result.flags &
         KVAF_URMA_RUNTIME_PROBE_RESOURCE_OPERATION_ATTEMPTED) != 0U;
    if (result.stage != KVAF_URMA_RUNTIME_PROBE_STAGE_COMPLETE ||
        result.cleanup_required != expected_cleanup ||
        result.cleanup_completed != expected_cleanup ||
        resource_attempted !=
            (options.mode != KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER) ||
        result.runtime_library_path[0] == '\0') {
      return fail("successful_lifecycle");
    }
  }
  return true;
}

const char* OutcomeName(std::uint32_t outcome) {
  switch (outcome) {
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK:
      return "ok";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_INVALID_ARGUMENT:
      return "invalid_argument";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_LIBRARY_UNAVAILABLE:
      return "library_unavailable";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_SYMBOL_UNAVAILABLE:
      return "symbol_unavailable";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_INIT_FAILED:
      return "init_failed";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_RUNTIME_ALREADY_INITIALIZED:
      return "runtime_already_initialized";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_DEVICE_NOT_FOUND:
      return "device_not_found";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_CONTEXT_CREATE_FAILED_UNPROVEN:
      return "context_create_failed_unproven";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_JFC_CREATE_FAILED_UNPROVEN:
      return "jfc_create_failed_unproven";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_JFR_CREATE_FAILED_UNPROVEN:
      return "jfr_create_failed_unproven";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_CLEANUP_UNSAFE:
      return "cleanup_unsafe";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_CONCURRENT_USE:
      return "concurrent_use";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROCESS_POISONED:
      return "process_poisoned";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH:
      return "provider_identity_mismatch";
    default:
      return "unknown";
  }
}

const char* ModeName(std::uint32_t mode) {
  switch (mode) {
    case KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER:
      return "discover";
    case KVAF_URMA_RUNTIME_PROBE_MODE_INIT_DEVICE:
      return "init-device";
    case KVAF_URMA_RUNTIME_PROBE_MODE_CREATE_DELETE:
      return "create-delete";
    default:
      return "invalid";
  }
}

bool HasSymbolGroup(std::uint64_t resolved, std::uint64_t expected) {
  return (resolved & expected) == expected;
}

const char* SymbolGroupState(std::uint64_t resolved,
                             std::uint64_t expected) {
  return HasSymbolGroup(resolved, expected)
             ? "complete-symbols-only-semantics-unproven"
             : "incomplete";
}

const char* CapabilityClass(
    const Options& options,
    const kvaf_urma_runtime_probe_result_v2& result) {
  if (options.mode == KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER) {
    switch (result.outcome) {
      case KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK:
        break;
      case KVAF_URMA_RUNTIME_PROBE_OUTCOME_LIBRARY_UNAVAILABLE:
        return "official-runtime-library-unavailable";
      case KVAF_URMA_RUNTIME_PROBE_OUTCOME_SYMBOL_UNAVAILABLE:
        return "official-runtime-base-symbol-surface-incomplete";
      case KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH:
        return "kvaf-runtime-probe-provider-build-identity-mismatch";
      case KVAF_URMA_RUNTIME_PROBE_OUTCOME_CONCURRENT_USE:
        return "process-local-probe-concurrent";
      case KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROCESS_POISONED:
        return "process-local-runtime-state-poisoned";
      case KVAF_URMA_RUNTIME_PROBE_OUTCOME_CLEANUP_UNSAFE:
        return "official-loader-cleanup-contract-insufficient";
      default:
        return "discover-invalid-or-unexpected-runtime-result";
    }
    if (HasSymbolGroup(
            result.resolved_symbol_mask,
            KVAF_URMA_RUNTIME_PROBE_CURRENT_DIRECT_UB_PUBLIC_SYMBOLS)) {
      return "current-direct-ub-public-symbol-surface-present-semantics-unproven";
    }
    if (HasSymbolGroup(
            result.resolved_symbol_mask,
            KVAF_URMA_RUNTIME_PROBE_CANDIDATE_AIV_CONTROL_SYMBOLS)) {
      return "candidate-modern-wrapper-surface-present-semantics-unproven";
    }
    return HasSymbolGroup(result.resolved_symbol_mask,
                          KVAF_URMA_RUNTIME_PROBE_LIFECYCLE_SYMBOLS)
        ? "legacy-public-lifecycle-symbols-present-modern-aiv-surface-incomplete"
        : "official-base-symbols-present-lifecycle-symbols-incomplete";
  }
  if (options.mode == KVAF_URMA_RUNTIME_PROBE_MODE_INIT_DEVICE &&
      result.outcome == KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK) {
    return "public-runtime-init-device-returned-success-internal-state-unproven";
  }
  switch (result.outcome) {
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK:
      return "public-context-jfc-jfr-returned-success-internal-state-unproven";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_LIBRARY_UNAVAILABLE:
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_SYMBOL_UNAVAILABLE:
      return "official-runtime-symbol-surface-incomplete";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_INIT_FAILED:
      return "official-runtime-init-returned-failure-cause-unproven";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_RUNTIME_ALREADY_INITIALIZED:
      return "process-global-runtime-owner-unattributed";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_DEVICE_NOT_FOUND:
      return "runtime-device-discovery-returned-null-cause-unproven";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_CONTEXT_CREATE_FAILED_UNPROVEN:
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_JFC_CREATE_FAILED_UNPROVEN:
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_JFR_CREATE_FAILED_UNPROVEN:
      if (result.operation_errno_valid != 0U &&
          (result.operation_errno == EINVAL ||
          result.operation_errno == ENOSYS ||
          result.operation_errno == EOPNOTSUPP)) {
        return "public-create-rejected-abi-or-support-unknown";
      }
      return "public-create-rejected-cleanup-contract-insufficient";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_CLEANUP_UNSAFE:
      return "public-lifecycle-cleanup-contract-insufficient";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_CONCURRENT_USE:
      return "process-local-probe-concurrent";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROCESS_POISONED:
      return "process-local-runtime-state-poisoned";
    case KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH:
      return "kvaf-runtime-probe-provider-build-identity-mismatch";
    default:
      return "invalid-or-unknown-runtime-result";
  }
}

class ProbeResources {
 public:
  using AicpuLaunch = decltype(&rtAicpuKernelLaunchExWithArgs);

  ~ProbeResources() {
    if (closed) {
      return;
    }
    if ((!submitted || quiesced) && result != nullptr) {
      (void)aclrtFree(result);
      result = nullptr;
    }
    if ((!submitted || quiesced) && stream != nullptr) {
      (void)aclrtDestroyStream(stream);
      stream = nullptr;
    }
    if ((!submitted || quiesced) && runtime_library != nullptr) {
      (void)dlclose(runtime_library);
      runtime_library = nullptr;
    }
    if (device_set && (!submitted || quiesced) && cleanup_proven) {
      (void)aclrtResetDevice(device_id);
      device_set = false;
    }
    if (acl_initialized && (!submitted || quiesced) && cleanup_proven) {
      (void)aclFinalize();
      acl_initialized = false;
    }
  }

  bool CloseChecked(const char** failed_phase, int* failed_status) {
    auto fail = [&](const char* phase, int status) {
      if (failed_phase != nullptr) {
        *failed_phase = phase;
      }
      if (failed_status != nullptr) {
        *failed_status = status;
      }
      return false;
    };
    if (submitted && !quiesced) {
      return fail("stream_not_quiesced", -1);
    }
    if (result != nullptr) {
      const aclError status = aclrtFree(result);
      if (status != ACL_SUCCESS) {
        return fail("free_result", status);
      }
      result = nullptr;
    }
    if (stream != nullptr) {
      const aclError status = aclrtDestroyStream(stream);
      if (status != ACL_SUCCESS) {
        return fail("destroy_stream", status);
      }
      stream = nullptr;
    }
    if (runtime_library != nullptr) {
      const int status = dlclose(runtime_library);
      if (status != 0) {
        return fail("close_runtime_library", status);
      }
      runtime_library = nullptr;
    }
    if (device_set) {
      const aclError status = aclrtResetDevice(device_id);
      if (status != ACL_SUCCESS) {
        return fail("reset_device_context", status);
      }
      device_set = false;
    }
    if (acl_initialized) {
      const aclError status = aclFinalize();
      if (status != ACL_SUCCESS) {
        return fail("acl_finalize", status);
      }
      acl_initialized = false;
    }
    cleanup_proven = true;
    closed = true;
    return true;
  }

  bool acl_initialized{false};
  bool device_set{false};
  bool submitted{false};
  bool quiesced{false};
  bool cleanup_proven{true};
  bool closed{false};
  std::uint32_t mode{KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER};
  std::int32_t device_id{-1};
  void* runtime_library{nullptr};
  AicpuLaunch aicpu_launch{nullptr};
  aclrtStream stream{nullptr};
  kvaf_urma_runtime_probe_result_v2* result{nullptr};
};

[[noreturn]] void HoldUnsafeOwner(ProbeResources* resources,
                                  const char* phase, int status) {
  resources->cleanup_proven = false;
  std::cerr << "KVAF_A5_RUNTIME_URMA_OWNER_HELD"
            << " phase=" << phase
            << " status=" << status
            << " pid=" << static_cast<long long>(::getpid())
            << " action=reboot-or-vendor-recovery-required\n";
  std::cerr.flush();
  for (;;) {
    (void)::pause();
  }
}

int ReturnPrelaunchFailure(ProbeResources* resources, const char* phase,
                           int status) {
  const char* cleanup_phase = nullptr;
  int cleanup_status = 0;
  if (!resources->CloseChecked(&cleanup_phase, &cleanup_status)) {
    HoldUnsafeOwner(resources, cleanup_phase, cleanup_status);
  }
  std::cerr << "KVAF_A5_RUNTIME_URMA_PRELAUNCH_SAFE mode="
            << ModeName(resources->mode) << '\n';
  std::cerr << "KVAF_A5_RUNTIME_URMA_PRELAUNCH_DETAIL"
            << " phase=" << phase << " status=" << status << '\n';
  return 1;
}

int RunProbe(const Options& options) {
  ProbeResources resources;
  resources.device_id = options.logical_device_id;
  resources.mode = options.mode;

  aclError acl_result = aclInit(nullptr);
  if (acl_result != ACL_SUCCESS) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=acl_init"
              << " acl_ret=" << acl_result << '\n';
    return ReturnPrelaunchFailure(&resources, "acl_init", acl_result);
  }
  resources.acl_initialized = true;
  acl_result = aclrtSetDevice(options.logical_device_id);
  if (acl_result != ACL_SUCCESS) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=set_device"
              << " acl_ret=" << acl_result << '\n';
    return ReturnPrelaunchFailure(&resources, "set_device", acl_result);
  }
  resources.device_set = true;

  resources.runtime_library = dlopen(
      "libascendcl.so", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (resources.runtime_library == nullptr) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=load_runtime\n";
    return ReturnPrelaunchFailure(&resources, "load_runtime", -1);
  }
  resources.aicpu_launch =
      reinterpret_cast<ProbeResources::AicpuLaunch>(dlsym(
          resources.runtime_library, "rtAicpuKernelLaunchExWithArgs"));
  if (resources.aicpu_launch == nullptr) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=load_symbol\n";
    return ReturnPrelaunchFailure(&resources, "load_symbol", -1);
  }
  acl_result = aclrtCreateStream(&resources.stream);
  if (acl_result != ACL_SUCCESS) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=create_stream"
              << " acl_ret=" << acl_result << '\n';
    return ReturnPrelaunchFailure(&resources, "create_stream", acl_result);
  }
  acl_result = aclrtMalloc(
      reinterpret_cast<void**>(&resources.result),
      sizeof(*resources.result), ACL_MEM_MALLOC_HUGE_FIRST);
  if (acl_result != ACL_SUCCESS) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=allocate_result"
              << " acl_ret=" << acl_result << '\n';
    return ReturnPrelaunchFailure(&resources, "allocate_result", acl_result);
  }
  if ((reinterpret_cast<std::uintptr_t>(resources.result) %
       KVAF_URMA_RUNTIME_PROBE_CACHE_LINE_BYTES) != 0U) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED"
              << " phase=result_alignment"
              << " required="
              << KVAF_URMA_RUNTIME_PROBE_CACHE_LINE_BYTES << '\n';
    return ReturnPrelaunchFailure(&resources, "result_alignment", -1);
  }
  const kvaf_urma_runtime_probe_result_v2 empty{};
  acl_result = aclrtMemcpy(resources.result, sizeof(empty), &empty,
                          sizeof(empty), ACL_MEMCPY_HOST_TO_DEVICE);
  if (acl_result != ACL_SUCCESS) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=zero_result"
              << " acl_ret=" << acl_result << '\n';
    return ReturnPrelaunchFailure(&resources, "zero_result", acl_result);
  }

  kvaf_urma_aicpu_runtime_probe_args_v2 args{};
  if (!CopyFixed(kRuntimeProbeEntry, args.kernel_name,
                 sizeof(args.kernel_name)) ||
      !CopyFixed(options.provider_soname, args.so_name,
                 sizeof(args.so_name)) ||
      (options.mode != KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER &&
       !CopyFixed(options.urma_device, args.device_name,
                  sizeof(args.device_name)))) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=build_args\n";
    return ReturnPrelaunchFailure(&resources, "build_args", -1);
  }
  args.magic = KVAF_URMA_RUNTIME_PROBE_MAGIC;
  args.abi_version = KVAF_URMA_RUNTIME_PROBE_ABI_VERSION;
  args.struct_size = sizeof(args);
  args.eid_index = options.eid_index;
  args.result_address =
      reinterpret_cast<std::uintptr_t>(resources.result);
  args.queue_depth = options.queue_depth;
  args.mode = options.mode;
  if (!FillRequestNonce(args.request_nonce)) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=request_nonce\n";
    return ReturnPrelaunchFailure(&resources, "request_nonce", errno);
  }
  args.expected_provider_abi_token =
      KVAF_URMA_RUNTIME_PROBE_PROVIDER_ABI_TOKEN;
  args.logical_device_id = options.logical_device_id;
  std::memcpy(args.expected_provider_build_id_sha256,
              options.provider_build_id_sha256.data(),
              sizeof(args.expected_provider_build_id_sha256));

  rtAicpuArgsEx_t launch_args{};
  launch_args.args = &args;
  launch_args.argsSize = sizeof(args);
  launch_args.soNameAddrOffset =
      offsetof(kvaf_urma_aicpu_runtime_probe_args_v2, so_name);
  launch_args.kernelNameAddrOffset =
      offsetof(kvaf_urma_aicpu_runtime_probe_args_v2, kernel_name);
  // Even discover calls dlopen(liburma.so.0), which may execute official
  // constructors.  No AICPU mode is therefore safe to terminate forcibly;
  // the detached supervisor retains ownership until a checked result exists.
  launch_args.timeout = 0U;

  resources.submitted = true;
  const auto launch_result = resources.aicpu_launch(
      kAicpuKfcKernelType, "kvaf_aicpu_runtime_urma_probe", 1,
      &launch_args, nullptr,
      reinterpret_cast<rtStream_t>(resources.stream), 0);
  if (launch_result != 0) {
    resources.cleanup_proven = false;
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=launch"
              << " runtime_ret=" << launch_result << '\n';
    HoldUnsafeOwner(&resources, "launch_unknown", launch_result);
  }
  acl_result = aclrtSynchronizeStream(resources.stream);
  if (acl_result != ACL_SUCCESS) {
    resources.cleanup_proven = false;
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=synchronize"
              << " acl_ret=" << acl_result
              << " cleanup="
              << "unknown"
              << '\n';
    HoldUnsafeOwner(&resources, "synchronize_unknown", acl_result);
  }
  resources.quiesced = true;

  kvaf_urma_runtime_probe_result_v2 result{};
  acl_result = aclrtMemcpy(&result, sizeof(result), resources.result,
                          sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
  if (acl_result != ACL_SUCCESS) {
    resources.cleanup_proven = false;
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED phase=read_result"
              << " acl_ret=" << acl_result << '\n';
    HoldUnsafeOwner(&resources, "read_result_unknown", acl_result);
  }
  const char* validation_failure = nullptr;
  if (!ValidateResultAgainstRequest(
          options, args, result, &validation_failure)) {
    resources.cleanup_proven = false;
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_FAILED"
              << " phase=result_validation"
              << " reason=" << validation_failure << '\n';
    HoldUnsafeOwner(&resources, "result_validation_unknown", -1);
  }
  const bool cleanup_complete =
      result.cleanup_required == result.cleanup_completed;
  const bool create_failure_unproven =
      (result.flags &
       KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN) != 0U;
  const bool official_side_effects_cleanup_unproven =
      (result.flags &
       KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN) != 0U;
  resources.cleanup_proven = cleanup_complete &&
      !create_failure_unproven &&
      !official_side_effects_cleanup_unproven &&
      (result.flags & KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED) == 0U &&
      result.outcome != KVAF_URMA_RUNTIME_PROBE_OUTCOME_CLEANUP_UNSAFE;

  std::cout << "KVAF_A5_RUNTIME_URMA_PROBE_RESULT"
            << " mode=" << ModeName(options.mode)
            << " outcome=" << OutcomeName(result.outcome)
            << " outcome_id=" << result.outcome
            << " primary_outcome=" << result.primary_outcome
            << " stage=" << result.stage
            << " operation_status=" << result.operation_status
            << " operation_errno=" << result.operation_errno
            << " operation_errno_valid="
            << result.operation_errno_valid
            << " compiled_urma_api_version="
            << result.compiled_urma_api_version
            << " runtime_urma_api_version=unavailable"
            << " flags=" << result.flags
            << " cleanup_required=" << result.cleanup_required
            << " cleanup_completed=" << result.cleanup_completed
            << " cleanup_proven=" << (resources.cleanup_proven ? 1 : 0)
            << " jfc_config_size=" << result.jfc_config_size
            << " jfr_config_size=" << result.jfr_config_size
            << " runtime_library="
            << (result.runtime_library_path[0] == '\0'
                    ? "unavailable" : result.runtime_library_path)
            << " required_symbol_mask=" << result.required_symbol_mask
            << " resolved_symbol_mask=" << result.resolved_symbol_mask
            << " loader_error="
            << (result.loader_error[0] == '\0'
                    ? "unavailable" : result.loader_error)
            << " capability_class="
            << CapabilityClass(options, result)
            << " official_abi_handshake=unavailable"
            << " documented_official_aicpu_call_contract="
               "not-found-in-audited-sources"
            << " local_runtime_compatibility=unproven"
            << " process_global_exclusivity=unproven"
            << " official_function_semantics=unproven"
            << " audited_pinned_source_snapshot_known_gaps=present"
            << " runtime_liburma_source_identity=unproven"
            << " official_loader_constructor_side_effects=unproven"
            << " official_public_cleanup="
            << (result.cleanup_required == 0U
                    ? "not-entered-or-not-owned"
                    : (cleanup_complete ? "returned-complete"
                                        : "incomplete"))
            << " official_side_effects_cleanup="
            << (official_side_effects_cleanup_unproven
                    ? "unproven" : "official-dso-not-entered")
            << " same_as_urma_init_dso_binding="
            << ((result.flags &
                 KVAF_URMA_RUNTIME_PROBE_SYMBOL_ORIGIN_MISMATCH) == 0U
                    ? "verified-for-resolved-symbols"
                    : "mismatch-detected")
            << " legacy_lifecycle_symbol_surface="
            << SymbolGroupState(result.resolved_symbol_mask,
                                KVAF_URMA_RUNTIME_PROBE_LIFECYCLE_SYMBOLS)
            << " candidate_modern_queue_wrapper_symbol_surface="
            << SymbolGroupState(result.resolved_symbol_mask,
                KVAF_URMA_RUNTIME_PROBE_CANDIDATE_MODERN_QUEUE_SYMBOLS)
            << " segment_symbol_surface="
            << SymbolGroupState(result.resolved_symbol_mask,
                                KVAF_URMA_RUNTIME_PROBE_SEGMENT_SYMBOLS)
            << " candidate_public_posting_symbol_surface="
            << SymbolGroupState(result.resolved_symbol_mask,
                KVAF_URMA_RUNTIME_PROBE_CANDIDATE_PUBLIC_POSTING_SYMBOLS)
            << " legacy_jetty_link_wrapper_symbol_surface="
            << SymbolGroupState(result.resolved_symbol_mask,
                KVAF_URMA_RUNTIME_PROBE_LEGACY_JETTY_LINK_WRAPPER_SYMBOLS)
            << " current_endpoint_discovery_symbol_surface="
            << SymbolGroupState(result.resolved_symbol_mask,
                KVAF_URMA_RUNTIME_PROBE_CURRENT_ENDPOINT_DISCOVERY_SYMBOLS)
            << " current_resource_lifecycle_symbol_surface="
            << SymbolGroupState(result.resolved_symbol_mask,
                KVAF_URMA_RUNTIME_PROBE_CURRENT_RESOURCE_LIFECYCLE_SYMBOLS)
            << " current_ctp_link_symbol_surface="
            << SymbolGroupState(result.resolved_symbol_mask,
                KVAF_URMA_RUNTIME_PROBE_CURRENT_CTP_LINK_SYMBOLS)
            << " current_ub_management_io_symbol_surface="
            << SymbolGroupState(result.resolved_symbol_mask,
                KVAF_URMA_RUNTIME_PROBE_CURRENT_UB_MANAGEMENT_IO_SYMBOLS)
            << " current_checked_teardown_symbol_surface="
            << SymbolGroupState(result.resolved_symbol_mask,
                KVAF_URMA_RUNTIME_PROBE_CURRENT_CHECKED_TEARDOWN_SYMBOLS)
            << " current_direct_ub_symbol_surface="
            << (HasSymbolGroup(
                    result.resolved_symbol_mask,
                    KVAF_URMA_RUNTIME_PROBE_CURRENT_DIRECT_UB_PUBLIC_SYMBOLS)
                    ? "complete-semantics-unproven" : "incomplete")
            << " aiv_address_visibility=unproven"
            << " wqe_cqe_private_layout=unproven"
            << " private_queue_abi=unproven"
            << " request_binding=verified-v2"
            << " provider_build_id_sha256="
            << HexSha256(result.provider_build_id_sha256)
            << " provider_build_identity=verified-random-build-instance-v1"
            << " provider_versioned_loader_key=verified-build-instance-v1"
            << " provider_self_file_hash=unavailable"
            << '\n';
  std::cout.flush();

  const bool borrowed_runtime =
      result.outcome ==
      KVAF_URMA_RUNTIME_PROBE_OUTCOME_RUNTIME_ALREADY_INITIALIZED;
  if (!resources.cleanup_proven && !borrowed_runtime) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_UNSAFE\n";
    HoldUnsafeOwner(&resources, "provider_cleanup_unproven",
                    static_cast<int>(result.outcome));
  }
  const char* cleanup_phase = nullptr;
  int cleanup_status = 0;
  if (!resources.CloseChecked(&cleanup_phase, &cleanup_status)) {
    HoldUnsafeOwner(&resources, cleanup_phase, cleanup_status);
  }
  std::cout << "KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK\n";

  if (result.outcome == KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK &&
      resources.cleanup_proven) {
    std::cout << (options.mode == KVAF_URMA_RUNTIME_PROBE_MODE_INIT_DEVICE
        ? "KVAF_A5_RUNTIME_ACCEPTED_INIT_DEVICE\n"
        : "KVAF_A5_RUNTIME_ACCEPTED_MINIMAL_CREATE_DELETE\n");
    return 0;
  }
  if (result.outcome ==
      KVAF_URMA_RUNTIME_PROBE_OUTCOME_RUNTIME_ALREADY_INITIALIZED) {
    std::cerr << "KVAF_A5_RUNTIME_URMA_PROCESS_CONTAMINATED\n";
    return 5;
  }
  std::cerr << "KVAF_A5_RUNTIME_URMA_PROBE_NON_ACCEPTED\n";
  return 3;
}

}  // namespace

int main(int argc, char** argv) {
  // The wrapper script is intentionally fail-closed, but this executable can
  // also be launched directly from the build directory.  Do not let that
  // bypass the capability-observation gate.  Recovery is implemented by the
  // supervisor script, not by this resource-bearing Host binary.
  std::cerr << "KVAF_A5_RUNTIME_URMA_BINARY_DYNAMIC_PROBE_DISABLED\n"
            << "CAPABILITY_OBSERVATION_LEDGER=not-implemented\n"
            << "No ACL/AICPU/URMA operation was attempted.\n";
  return 2;

  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }
  return RunProbe(options);
}
