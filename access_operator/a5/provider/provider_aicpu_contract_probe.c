#include "kvaf_urma_provider_abi.h"

#include <dlfcn.h>

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

int main(int argc, char **argv)
{
    struct kvaf_urma_aicpu_open_args_v2 open_args;
    struct kvaf_urma_aicpu_close_args_v2 close_args;
    struct kvaf_urma_provider_result_v2 open_result;
    struct kvaf_urma_owner_probe_command_v1 owner_command;
    struct kvaf_urma_owner_probe_status_v1 owner_status;
    struct kvaf_urma_aicpu_owner_probe_args_v1 owner_args;
    int32_t close_result = 0;
    aicpu_entry_fn open_entry;
    aicpu_entry_fn close_entry;
    aicpu_entry_fn owner_entry;
    aicpu_entry_fn epoch_owner_entry;
    void *library;

    if (argc != 2) {
        (void)fprintf(stderr, "usage: %s DEVICE_PROVIDER\n", argv[0]);
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

    (void)memset(&open_args, 0, sizeof(open_args));
    (void)memset(&open_result, 0, sizeof(open_result));
    open_args.result_address = (uint64_t)(uintptr_t)&open_result;
    if (open_entry(&open_args) != 0 ||
        open_result.magic != KVAF_URMA_PROVIDER_MAGIC ||
        open_result.abi_version != KVAF_URMA_PROVIDER_ABI_VERSION ||
        open_result.struct_size != sizeof(open_result) ||
        open_result.status != KVAF_URMA_PROVIDER_INVALID_ARGUMENT) {
        (void)fprintf(stderr,
            "provider failure escaped as an AICPU task failure\n");
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
    if (close_entry(&close_args) != 0 || close_result != -1) {
        (void)fprintf(stderr,
            "provider close failure escaped as an AICPU task failure\n");
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

    /* This process-local probe validates the bounded command protocol and
     * export.  It does not prove AICPU worker affinity or device coherence. */
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
            KVAF_URMA_OWNER_PROBE_STATE_RELEASED ||
        owner_status.magic != KVAF_URMA_OWNER_PROBE_MAGIC ||
        owner_status.abi_version != KVAF_URMA_OWNER_PROBE_ABI_VERSION ||
        owner_status.struct_size != sizeof(owner_status) ||
        owner_status.session_id != owner_args.session_id ||
        owner_status.sequence != KVAF_URMA_OWNER_PROBE_RELEASE_SEQUENCE ||
        owner_status.result_code != KVAF_URMA_OWNER_PROBE_RESULT_OK) {
        (void)fprintf(stderr,
            "owner probe did not acknowledge a release command\n");
        (void)dlclose(library);
        return 1;
    }

    (void)memset(&owner_command, 0, sizeof(owner_command));
    (void)memset(&owner_status, 0, sizeof(owner_status));
    owner_args.timeout_milliseconds = 1;
    if (owner_entry(&owner_args) != 0 ||
        __atomic_load_n(&owner_status.state, __ATOMIC_ACQUIRE) !=
            KVAF_URMA_OWNER_PROBE_STATE_TIMED_OUT ||
        owner_status.result_code != KVAF_URMA_OWNER_PROBE_RESULT_TIMED_OUT ||
        owner_status.session_id != owner_args.session_id ||
        owner_status.sequence != 0) {
        (void)fprintf(stderr,
            "owner probe did not honor its bounded timeout\n");
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
    if (epoch_owner_entry(NULL) == 0) {
        (void)fprintf(stderr,
            "invalid AICPU epoch owner dispatch unexpectedly succeeded\n");
        (void)dlclose(library);
        return 1;
    }

    (void)dlclose(library);
    (void)puts("KVAF_URMA_PROVIDER_AICPU_CONTRACT=PASS");
    return 0;
}
