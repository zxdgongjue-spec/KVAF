#include "kvaf_urma_provider_abi.h"

#include <dlfcn.h>
#include <errno.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*aicpu_entry_fn)(void *);

static aicpu_entry_fn load_entry(void *library, const char *name)
{
    void *symbol;

    (void)dlerror();
    symbol = dlsym(library, name);
    if (symbol == NULL || dlerror() != NULL) {
        return NULL;
    }
    return (aicpu_entry_fn)symbol;
}

static void initialize_valid_device_config(
    struct kvaf_urma_provider_config_v2 *config,
    uint64_t local_address, uint64_t session_id)
{
    (void)memset(config, 0, sizeof(*config));
    config->magic = KVAF_URMA_PROVIDER_MAGIC;
    config->abi_version = KVAF_URMA_PROVIDER_ABI_VERSION;
    config->struct_size = (uint32_t)sizeof(*config);
    config->role = KVAF_URMA_PROVIDER_CLIENT;
    config->flags = KVAF_URMA_PROVIDER_USE_CTP |
        KVAF_URMA_PROVIDER_EXPECT_DEVICE_PRIVATE_ABI |
        KVAF_URMA_PROVIDER_EXPLICIT_TCP_MANAGEMENT;
    config->control_port = 21115U;
    config->address_kind = KVAF_URMA_PROVIDER_ADDRESS_DEVICE_VA;
    config->control_timeout_seconds = 1U;
    config->session_id = session_id;
    config->local_address = local_address;
    config->local_allocation_bytes = UINT64_C(8192);
    config->segment_payload_bytes = UINT64_C(4096);
    (void)snprintf(config->device_name, sizeof(config->device_name),
        "%s", "udmac0d0e2");
    (void)snprintf(config->control_server_address,
        sizeof(config->control_server_address), "%s", "127.0.0.1");
}

int main(int argc, char **argv)
{
    struct kvaf_urma_aicpu_open_args_v2 open_args;
    struct kvaf_urma_aicpu_close_args_v2 close_args;
    _Alignas(64) struct kvaf_urma_provider_result_v2 open_result;
    struct kvaf_urma_owner_probe_command_v1 owner_command;
    struct kvaf_urma_owner_probe_status_v1 owner_status;
    struct kvaf_urma_aicpu_owner_probe_args_v1 owner_args;
    struct kvaf_urma_epoch_owner_command_v1 epoch_command;
    struct kvaf_urma_epoch_owner_status_v1 epoch_status;
    _Alignas(64) struct kvaf_urma_provider_result_v2 epoch_open_result;
    struct kvaf_urma_aicpu_epoch_owner_args_v1 epoch_owner_args;
    struct kvaf_urma_aicpu_runtime_probe_args_v2 runtime_probe_args;
    struct kvaf_urma_runtime_probe_result_v2 runtime_probe_result;
    _Alignas(64) uint8_t fake_device_memory[8192];
    uint8_t compiled_runtime_probe_build_id[
        KVAF_URMA_RUNTIME_PROBE_BUILD_ID_BYTES];
    size_t index;
    int32_t close_result = 0;
    aicpu_entry_fn open_entry;
    aicpu_entry_fn close_entry;
    aicpu_entry_fn owner_entry;
    aicpu_entry_fn runtime_probe_entry;
    aicpu_entry_fn epoch_owner_entry;
    void *library;
    void *runtime_probe_library;

    if (argc != 3) {
        (void)fprintf(stderr,
            "usage: %s DEVICE_PROVIDER RUNTIME_PROBE_PROVIDER\n", argv[0]);
        return 2;
    }
    library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        (void)fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    open_entry = load_entry(library, "aicpu_kvaf_urma_open_v2");
    close_entry = load_entry(library, "aicpu_kvaf_urma_close_v2");
    owner_entry = load_entry(library, "aicpu_kvaf_urma_owner_probe_v1");
    epoch_owner_entry = load_entry(
        library, "aicpu_kvaf_urma_epoch_owner_v1");
    if (open_entry == NULL || close_entry == NULL || owner_entry == NULL ||
        epoch_owner_entry == NULL) {
        (void)fprintf(stderr, "AICPU provider entry is unavailable\n");
        (void)dlclose(library);
        return 1;
    }

    /* A valid request, rather than malformed input, must still be rejected by
     * the provider entry itself before it can create any resource. */
    (void)memset(&open_args, 0, sizeof(open_args));
    (void)memset(&open_result, 0, sizeof(open_result));
    initialize_valid_device_config(&open_args.config,
        (uint64_t)(uintptr_t)fake_device_memory,
        UINT64_C(0x6b76616600000011));
    open_args.result_address = (uint64_t)(uintptr_t)&open_result;
    if (open_entry(&open_args) != 0 ||
        open_result.magic != KVAF_URMA_PROVIDER_MAGIC ||
        open_result.abi_version != KVAF_URMA_PROVIDER_ABI_VERSION ||
        open_result.struct_size != sizeof(open_result) ||
        open_result.status !=
            KVAF_URMA_PROVIDER_CAPABILITY_LEDGER_UNAVAILABLE ||
        open_result.provider_status != -ENOTSUP ||
        open_result.session_id != 0U ||
        open_result.flags != 0U) {
        (void)fprintf(stderr,
            "provider Open capability gate is not fail-closed\n");
        (void)dlclose(library);
        return 1;
    }
    open_args.result_address = 0;
    if (open_entry(&open_args) == 0 || open_entry(NULL) == 0) {
        (void)fprintf(stderr,
            "invalid AICPU open dispatch unexpectedly succeeded\n");
        (void)dlclose(library);
        return 1;
    }

    (void)memset(&close_args, 0, sizeof(close_args));
    close_args.result_address = (uint64_t)(uintptr_t)&close_result;
    close_args.session_id = open_args.config.session_id;
    if (close_entry(&close_args) != 0 || close_result != -1) {
        (void)fprintf(stderr,
            "capability-gated Open unexpectedly left provider state\n");
        (void)dlclose(library);
        return 1;
    }
    close_args.session_id = 1;
    close_result = 0;
    if (close_entry(&close_args) != 0 || close_result != -1) {
        (void)fprintf(stderr,
            "empty provider instance acknowledged a foreign session\n");
        (void)dlclose(library);
        return 1;
    }
    close_args.session_id = 2;
    close_result = 0;
    if (close_entry(&close_args) != 0 || close_result != -1) {
        (void)fprintf(stderr,
            "a second foreign session was acknowledged by an empty provider\n");
        (void)dlclose(library);
        return 1;
    }
    close_args.result_address = 0;
    if (close_entry(&close_args) == 0 || close_entry(NULL) == 0) {
        (void)fprintf(stderr,
            "invalid AICPU close dispatch unexpectedly succeeded\n");
        (void)dlclose(library);
        return 1;
    }

    /* A valid owner-probe request must publish the fixed ledger-unavailable
     * terminal without reading its command or entering clock/sleep. */
    (void)memset(&owner_command, 0, sizeof(owner_command));
    (void)memset(&owner_status, 0, sizeof(owner_status));
    (void)memset(&owner_args, 0, sizeof(owner_args));
    owner_args.magic = KVAF_URMA_OWNER_PROBE_MAGIC;
    owner_args.abi_version = KVAF_URMA_OWNER_PROBE_ABI_VERSION;
    owner_args.struct_size = (uint32_t)sizeof(owner_args);
    owner_args.timeout_milliseconds = 10;
    owner_args.session_id = UINT64_C(0x6b76616600000001);
    owner_args.command_address =
        (uint64_t)(uintptr_t)&owner_command;
    owner_args.status_address =
        (uint64_t)(uintptr_t)&owner_status;
    owner_args.poll_interval_microseconds = 100;
    owner_command.magic = KVAF_URMA_OWNER_PROBE_MAGIC;
    owner_command.abi_version = KVAF_URMA_OWNER_PROBE_ABI_VERSION;
    owner_command.struct_size = (uint32_t)sizeof(owner_command);
    owner_command.session_id = owner_args.session_id;
    owner_command.sequence = KVAF_URMA_OWNER_PROBE_RELEASE_SEQUENCE;
    __atomic_store_n(&owner_command.opcode,
        (uint32_t)KVAF_URMA_OWNER_PROBE_COMMAND_RELEASE,
        __ATOMIC_RELEASE);
    if (owner_entry(&owner_args) != 0 ||
        __atomic_load_n(&owner_status.state, __ATOMIC_ACQUIRE) !=
            KVAF_URMA_OWNER_PROBE_STATE_FAILED ||
        owner_status.magic != KVAF_URMA_OWNER_PROBE_MAGIC ||
        owner_status.abi_version != KVAF_URMA_OWNER_PROBE_ABI_VERSION ||
        owner_status.struct_size != sizeof(owner_status) ||
        owner_status.session_id != owner_args.session_id ||
        owner_status.sequence != 0 ||
        owner_status.result_code !=
            KVAF_URMA_OWNER_PROBE_RESULT_CAPABILITY_LEDGER_UNAVAILABLE ||
        owner_status.elapsed_nanoseconds != 0 ||
        __atomic_load_n(&owner_command.opcode, __ATOMIC_ACQUIRE) !=
            KVAF_URMA_OWNER_PROBE_COMMAND_RELEASE) {
        (void)fprintf(stderr,
            "owner-probe capability gate is not fail-closed\n");
        (void)dlclose(library);
        return 1;
    }
    owner_args.session_id = 0;
    if (owner_entry(&owner_args) == 0 || owner_entry(NULL) == 0) {
        (void)fprintf(stderr,
            "invalid AICPU owner probe dispatch unexpectedly succeeded\n");
        (void)dlclose(library);
        return 1;
    }
    if (load_entry(library, "aicpu_kvaf_urma_rt_probe_v2") != NULL) {
        (void)fprintf(stderr,
            "canonical provider unexpectedly exports the isolated runtime probe\n");
        (void)dlclose(library);
        return 1;
    }
    if (epoch_owner_entry(NULL) == 0) {
        (void)fprintf(stderr,
            "invalid AICPU epoch owner dispatch unexpectedly succeeded\n");
        (void)dlclose(library);
        return 1;
    }

    /* A fully valid epoch-owner request is the strongest alternate-caller
     * bypass attempt.  It must publish a safe terminal without acquiring the
     * owner claim or entering provider Open. */
    (void)memset(&epoch_command, 0, sizeof(epoch_command));
    (void)memset(&epoch_status, 0, sizeof(epoch_status));
    (void)memset(&epoch_open_result, 0, sizeof(epoch_open_result));
    (void)memset(&epoch_owner_args, 0, sizeof(epoch_owner_args));
    initialize_valid_device_config(&epoch_owner_args.config,
        (uint64_t)(uintptr_t)fake_device_memory,
        UINT64_C(0x6b76616600000012));
    epoch_owner_args.epoch = UINT64_C(1);
    epoch_owner_args.command_address =
        (uint64_t)(uintptr_t)&epoch_command;
    epoch_owner_args.status_address =
        (uint64_t)(uintptr_t)&epoch_status;
    epoch_owner_args.open_result_address =
        (uint64_t)(uintptr_t)&epoch_open_result;
    epoch_owner_args.magic = KVAF_URMA_EPOCH_OWNER_MAGIC;
    epoch_owner_args.abi_version = KVAF_URMA_EPOCH_OWNER_ABI_VERSION;
    epoch_owner_args.struct_size = (uint32_t)sizeof(epoch_owner_args);
    if (epoch_owner_entry(&epoch_owner_args) != 0 ||
        epoch_status.magic != KVAF_URMA_EPOCH_OWNER_MAGIC ||
        epoch_status.abi_version != KVAF_URMA_EPOCH_OWNER_ABI_VERSION ||
        epoch_status.struct_size != sizeof(epoch_status) ||
        epoch_status.epoch != epoch_owner_args.epoch ||
        epoch_status.session_id != epoch_owner_args.config.session_id ||
        __atomic_load_n(&epoch_status.state, __ATOMIC_ACQUIRE) !=
            KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE ||
        epoch_status.result_code !=
            KVAF_URMA_EPOCH_OWNER_RESULT_CAPABILITY_LEDGER_UNAVAILABLE ||
        epoch_status.provider_status != -ENOTSUP ||
        epoch_status.flags !=
            (KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
             KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE) ||
        epoch_open_result.magic != KVAF_URMA_PROVIDER_MAGIC ||
        epoch_open_result.abi_version != KVAF_URMA_PROVIDER_ABI_VERSION ||
        epoch_open_result.struct_size != sizeof(epoch_open_result) ||
        epoch_open_result.status !=
            KVAF_URMA_PROVIDER_CAPABILITY_LEDGER_UNAVAILABLE ||
        epoch_open_result.provider_status != -ENOTSUP ||
        epoch_open_result.session_id != epoch_owner_args.config.session_id) {
        (void)fprintf(stderr,
            "epoch-owner capability gate is not fail-closed\n");
        (void)dlclose(library);
        return 1;
    }
    close_args.result_address = (uint64_t)(uintptr_t)&close_result;
    close_args.session_id = epoch_owner_args.config.session_id;
    close_result = 0;
    if (close_entry(&close_args) != 0 || close_result != -1) {
        (void)fprintf(stderr,
            "capability-gated epoch owner unexpectedly left provider state\n");
        (void)dlclose(library);
        return 1;
    }

    (void)dlclose(library);

    runtime_probe_library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (runtime_probe_library == NULL) {
        (void)fprintf(stderr, "runtime probe dlopen failed: %s\n", dlerror());
        return 1;
    }
    runtime_probe_entry = load_entry(
        runtime_probe_library, "aicpu_kvaf_urma_rt_probe_v2");
    if (runtime_probe_entry == NULL ||
        load_entry(runtime_probe_library, "aicpu_kvaf_urma_open_v2") != NULL ||
        load_entry(runtime_probe_library, "aicpu_kvaf_urma_close_v2") != NULL ||
        load_entry(runtime_probe_library,
            "aicpu_kvaf_urma_owner_probe_v1") != NULL ||
        load_entry(runtime_probe_library,
            "aicpu_kvaf_urma_epoch_owner_v1") != NULL) {
        (void)fprintf(stderr,
            "runtime probe provider export isolation failed\n");
        (void)dlclose(runtime_probe_library);
        return 1;
    }

    /* First recover the provider-compiled identity through the mismatch
     * result.  The second, correctly-bound request proves that even a custom
     * AICPU caller cannot cross the gate into dlopen or official URMA APIs. */
    (void)memset(&runtime_probe_args, 0, sizeof(runtime_probe_args));
    (void)memset(&runtime_probe_result, 0, sizeof(runtime_probe_result));
    (void)snprintf(runtime_probe_args.kernel_name,
        sizeof(runtime_probe_args.kernel_name), "%s",
        "aicpu_kvaf_urma_rt_probe_v2");
    (void)snprintf(runtime_probe_args.so_name,
        sizeof(runtime_probe_args.so_name), "%s", argv[2]);
    runtime_probe_args.magic = KVAF_URMA_RUNTIME_PROBE_MAGIC;
    runtime_probe_args.abi_version = KVAF_URMA_RUNTIME_PROBE_ABI_VERSION;
    runtime_probe_args.struct_size = (uint32_t)sizeof(runtime_probe_args);
    runtime_probe_args.result_address =
        (uint64_t)(uintptr_t)&runtime_probe_result;
    runtime_probe_args.queue_depth =
        KVAF_URMA_RUNTIME_PROBE_DEFAULT_QUEUE_DEPTH;
    runtime_probe_args.mode = KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER;
    runtime_probe_args.request_nonce[0] = UINT64_C(0x6b76616600000013);
    runtime_probe_args.expected_provider_abi_token =
        KVAF_URMA_RUNTIME_PROBE_PROVIDER_ABI_TOKEN;
    runtime_probe_args.logical_device_id = 0;
    (void)memset(runtime_probe_args.expected_provider_build_id_sha256,
        0xa5, sizeof(runtime_probe_args.expected_provider_build_id_sha256));
    if (runtime_probe_entry(&runtime_probe_args) != 0 ||
        runtime_probe_result.outcome !=
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH) {
        (void)fprintf(stderr,
            "runtime URMA probe identity binding contract failed\n");
        (void)dlclose(runtime_probe_library);
        return 1;
    }
    (void)memcpy(compiled_runtime_probe_build_id,
        runtime_probe_result.provider_build_id_sha256,
        sizeof(compiled_runtime_probe_build_id));
    for (index = 0; index < sizeof(compiled_runtime_probe_build_id); ++index) {
        if (compiled_runtime_probe_build_id[index] != 0U) {
            break;
        }
    }
    if (index == sizeof(compiled_runtime_probe_build_id)) {
        (void)fprintf(stderr,
            "runtime URMA probe omitted its compiled build identity\n");
        (void)dlclose(runtime_probe_library);
        return 1;
    }

    (void)memcpy(runtime_probe_args.expected_provider_build_id_sha256,
        compiled_runtime_probe_build_id,
        sizeof(compiled_runtime_probe_build_id));
    (void)memset(&runtime_probe_result, 0, sizeof(runtime_probe_result));
    if (runtime_probe_entry(&runtime_probe_args) != 0 ||
        runtime_probe_result.magic != KVAF_URMA_RUNTIME_PROBE_MAGIC ||
        runtime_probe_result.abi_version !=
            KVAF_URMA_RUNTIME_PROBE_ABI_VERSION ||
        runtime_probe_result.struct_size != sizeof(runtime_probe_result) ||
        runtime_probe_result.outcome !=
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_CAPABILITY_LEDGER_UNAVAILABLE ||
        runtime_probe_result.primary_outcome !=
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_CAPABILITY_LEDGER_UNAVAILABLE ||
        runtime_probe_result.stage != KVAF_URMA_RUNTIME_PROBE_STAGE_NONE ||
        runtime_probe_result.operation_status != -ENOTSUP ||
        runtime_probe_result.operation_errno != 0 ||
        runtime_probe_result.operation_errno_valid != 0U ||
        runtime_probe_result.flags !=
            KVAF_URMA_RUNTIME_PROBE_PUBLIC_ABI_UNPROVEN ||
        runtime_probe_result.cleanup_required != 0U ||
        runtime_probe_result.cleanup_completed != 0U ||
        runtime_probe_result.resolved_symbol_mask != UINT64_C(0) ||
        runtime_probe_result.runtime_library_path[0] != '\0' ||
        runtime_probe_result.loader_error[0] != '\0' ||
        (runtime_probe_result.flags &
            KVAF_URMA_RUNTIME_PROBE_RESOURCE_OPERATION_ATTEMPTED) != 0U ||
        memcmp(runtime_probe_result.provider_build_id_sha256,
            compiled_runtime_probe_build_id,
            sizeof(compiled_runtime_probe_build_id)) != 0 ||
        runtime_probe_result.request_nonce[0] !=
            runtime_probe_args.request_nonce[0] ||
        runtime_probe_result.request_nonce[1] !=
            runtime_probe_args.request_nonce[1] ||
        runtime_probe_result.provider_abi_token !=
            KVAF_URMA_RUNTIME_PROBE_PROVIDER_ABI_TOKEN ||
        runtime_probe_entry(NULL) == 0) {
        (void)fprintf(stderr,
            "runtime URMA probe capability gate is not fail-closed\n");
        (void)dlclose(runtime_probe_library);
        return 1;
    }
    (void)dlclose(runtime_probe_library);
    (void)puts("KVAF_URMA_PROVIDER_AICPU_CONTRACT=PASS");
    return 0;
}
