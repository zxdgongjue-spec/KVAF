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
    int32_t close_result = 0;
    aicpu_entry_fn open_entry;
    aicpu_entry_fn close_entry;
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
    if (open_entry == NULL || close_entry == NULL) {
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

    (void)dlclose(library);
    (void)puts("KVAF_URMA_PROVIDER_AICPU_CONTRACT=PASS");
    return 0;
}
