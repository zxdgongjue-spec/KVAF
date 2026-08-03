#define _GNU_SOURCE
#include "kvaf_urma_provider_abi.h"

#include <dlfcn.h>
#include <errno.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

#include "urma_api.h"

#define KVAF_RUNTIME_URMA_SONAME "liburma.so.0"
#define KVAF_RUNTIME_PROBE_TOKEN UINT32_C(0x00abcdef)

#ifndef KVAF_RUNTIME_PROBE_PROVIDER_BUILD_ID_HEX
#error "KVAF_RUNTIME_PROBE_PROVIDER_BUILD_ID_HEX must be supplied by the build"
#endif
_Static_assert(sizeof(KVAF_RUNTIME_PROBE_PROVIDER_BUILD_ID_HEX) == 65,
    "runtime probe provider build ID must contain 64 hex digits");

typedef urma_status_t (*kvaf_urma_init_fn)(urma_init_attr_t *);
typedef urma_status_t (*kvaf_urma_uninit_fn)(void);
typedef urma_device_t *(*kvaf_urma_get_device_by_name_fn)(char *);
typedef urma_context_t *(*kvaf_urma_create_context_fn)(
    urma_device_t *, uint32_t);
typedef urma_status_t (*kvaf_urma_delete_context_fn)(urma_context_t *);
typedef urma_jfc_t *(*kvaf_urma_create_jfc_fn)(
    urma_context_t *, urma_jfc_cfg_t *);
typedef urma_status_t (*kvaf_urma_delete_jfc_fn)(urma_jfc_t *);
typedef urma_jfr_t *(*kvaf_urma_create_jfr_fn)(
    urma_context_t *, urma_jfr_cfg_t *);
typedef urma_status_t (*kvaf_urma_delete_jfr_fn)(urma_jfr_t *);

struct kvaf_runtime_urma_api {
    kvaf_urma_init_fn init;
    kvaf_urma_uninit_fn uninit;
    kvaf_urma_get_device_by_name_fn get_device_by_name;
    kvaf_urma_create_context_fn create_context;
    kvaf_urma_delete_context_fn delete_context;
    kvaf_urma_create_jfc_fn create_jfc;
    kvaf_urma_delete_jfc_fn delete_jfc;
    kvaf_urma_create_jfr_fn create_jfr;
    kvaf_urma_delete_jfr_fn delete_jfr;
};

/* liburma state is process-global.  This excludes concurrent probes in this
 * DSO; URMA_EEXIST remains the fail-closed inter-DSO ownership check. */
static pthread_mutex_t g_kvaf_runtime_probe_lock =
    PTHREAD_MUTEX_INITIALIZER;
static bool g_kvaf_runtime_probe_poisoned;

_Static_assert(sizeof(void *) == sizeof(kvaf_urma_init_fn),
    "POSIX dlsym function pointers must fit in a data pointer");

static bool kvaf_runtime_probe_has_terminator(
    const char *text, size_t capacity)
{
    return text != NULL && memchr(text, '\0', capacity) != NULL;
}

static bool kvaf_runtime_probe_bytes_are_zero(
    const uint8_t *bytes, size_t size)
{
    size_t index;

    if (bytes == NULL) {
        return false;
    }
    for (index = 0; index < size; ++index) {
        if (bytes[index] != 0U) {
            return false;
        }
    }
    return true;
}

static int kvaf_runtime_probe_hex_nibble(char value)
{
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    return -1;
}

static bool kvaf_runtime_probe_compiled_build_id(uint8_t output[
    KVAF_URMA_RUNTIME_PROBE_BUILD_ID_BYTES])
{
    const char *hex = KVAF_RUNTIME_PROBE_PROVIDER_BUILD_ID_HEX;
    size_t index;

    if (output == NULL) {
        return false;
    }
    for (index = 0; index < KVAF_URMA_RUNTIME_PROBE_BUILD_ID_BYTES;
         ++index) {
        int upper = kvaf_runtime_probe_hex_nibble(hex[index * 2U]);
        int lower = kvaf_runtime_probe_hex_nibble(hex[index * 2U + 1U]);

        if (upper < 0 || lower < 0) {
            return false;
        }
        output[index] = (uint8_t)((unsigned int)upper << 4U |
            (unsigned int)lower);
    }
    return hex[KVAF_URMA_RUNTIME_PROBE_BUILD_ID_BYTES * 2U] == '\0';
}

static void kvaf_runtime_probe_initialize_result(
    struct kvaf_urma_runtime_probe_result_v2 *result, uint32_t queue_depth)
{
    (void)memset(result, 0, sizeof(*result));
    result->magic = KVAF_URMA_RUNTIME_PROBE_MAGIC;
    result->abi_version = KVAF_URMA_RUNTIME_PROBE_ABI_VERSION;
    result->struct_size = (uint32_t)sizeof(*result);
    result->outcome = KVAF_URMA_RUNTIME_PROBE_OUTCOME_INVALID_ARGUMENT;
    result->primary_outcome = result->outcome;
    result->flags = KVAF_URMA_RUNTIME_PROBE_PUBLIC_ABI_UNPROVEN;
    result->jfc_config_size = (uint32_t)sizeof(urma_jfc_cfg_t);
    result->jfr_config_size = (uint32_t)sizeof(urma_jfr_cfg_t);
    result->queue_depth = queue_depth;
    result->compiled_urma_api_version = URMA_API_VERSION;
}

static void kvaf_runtime_probe_record_primary(
    struct kvaf_urma_runtime_probe_result_v2 *result,
    enum kvaf_urma_runtime_probe_outcome outcome,
    enum kvaf_urma_runtime_probe_stage stage,
    int operation_status, int operation_errno)
{
    result->outcome = (uint32_t)outcome;
    result->primary_outcome = (uint32_t)outcome;
    result->stage = (uint32_t)stage;
    result->operation_status = operation_status;
    result->operation_errno = operation_errno;
    result->operation_errno_valid = operation_errno != 0 ? 1U : 0U;
}

static void kvaf_runtime_probe_record_cleanup_failure(
    struct kvaf_urma_runtime_probe_result_v2 *result,
    enum kvaf_urma_runtime_probe_stage stage,
    int operation_status, int operation_errno)
{
    result->outcome = KVAF_URMA_RUNTIME_PROBE_OUTCOME_CLEANUP_UNSAFE;
    result->stage = (uint32_t)stage;
    result->operation_status = operation_status;
    result->operation_errno = operation_errno;
    result->operation_errno_valid = operation_errno != 0 ? 1U : 0U;
}

static void kvaf_runtime_probe_record_loader_error(
    struct kvaf_urma_runtime_probe_result_v2 *result, const char *error)
{
    size_t source_index;
    size_t target_index = 0;

    if (result == NULL || error == NULL) {
        return;
    }
    for (source_index = 0; error[source_index] != '\0'; ++source_index) {
        unsigned char value = (unsigned char)error[source_index];

        if (target_index + 1U >= sizeof(result->loader_error)) {
            result->flags |=
                KVAF_URMA_RUNTIME_PROBE_LOADER_ERROR_TRUNCATED;
            break;
        }
        result->loader_error[target_index++] =
            value >= 0x21U && value <= 0x7eU ? (char)value : '_';
    }
    result->loader_error[target_index] = '\0';
}

static bool kvaf_runtime_probe_load_symbol(
    void *library, const char *name, void *destination, size_t size,
    uint64_t symbol_bit, bool required, const void *required_dso_base,
    struct kvaf_urma_runtime_probe_result_v2 *result)
{
    void *symbol;
    const char *error;
    Dl_info info = {0};

    if (destination == NULL || size != sizeof(symbol)) {
        return false;
    }
    (void)dlerror();
    symbol = dlsym(library, name);
    error = dlerror();
    if (symbol == NULL || error != NULL) {
        if (required) {
            kvaf_runtime_probe_record_loader_error(
                result, error == NULL ? name : error);
        }
        return !required;
    }
    if (required_dso_base != NULL &&
        (dladdr(symbol, &info) == 0 || info.dli_fbase == NULL ||
         info.dli_fbase != required_dso_base)) {
        result->flags |= KVAF_URMA_RUNTIME_PROBE_SYMBOL_ORIGIN_MISMATCH;
        if (required) {
            kvaf_runtime_probe_record_loader_error(
                result, "symbol_resolved_from_different_dso");
        }
        return !required;
    }
    (void)memcpy(destination, &symbol, sizeof(symbol));
    result->resolved_symbol_mask |= symbol_bit;
    return true;
}

static bool kvaf_runtime_probe_load_api(
    void *library, struct kvaf_runtime_urma_api *api,
    uint64_t required_symbol_mask,
    struct kvaf_urma_runtime_probe_result_v2 *result)
{
#define KVAF_LOAD_RUNTIME_SYMBOL(member, symbol_name, symbol_bit) \
    kvaf_runtime_probe_load_symbol(library, symbol_name, \
        &api->member, sizeof(api->member), symbol_bit, \
        (required_symbol_mask & symbol_bit) != 0U, init_dso_base, result)
    bool load_ok = true;
    void *init_address = NULL;
    Dl_info init_info = {0};
    const void *init_dso_base = NULL;

    (void)memset(api, 0, sizeof(*api));
    result->required_symbol_mask = required_symbol_mask;
    load_ok &= kvaf_runtime_probe_load_symbol(library, "urma_init",
        &api->init, sizeof(api->init),
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_INIT,
        (required_symbol_mask &
         KVAF_URMA_RUNTIME_PROBE_SYMBOL_INIT) != 0U,
        NULL, result);
    (void)memcpy(&init_address, &api->init, sizeof(init_address));
    if (init_address == NULL || dladdr(init_address, &init_info) == 0 ||
        init_info.dli_fbase == NULL) {
        kvaf_runtime_probe_record_loader_error(
            result, "unable_to_identify_urma_init_dso");
        return false;
    }
    init_dso_base = init_info.dli_fbase;
    load_ok &= KVAF_LOAD_RUNTIME_SYMBOL(uninit, "urma_uninit",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNINIT);
    load_ok &= KVAF_LOAD_RUNTIME_SYMBOL(get_device_by_name,
        "urma_get_device_by_name",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_DEVICE);
    load_ok &= KVAF_LOAD_RUNTIME_SYMBOL(create_context,
        "urma_create_context",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_CONTEXT);
    load_ok &= KVAF_LOAD_RUNTIME_SYMBOL(delete_context,
        "urma_delete_context",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_CONTEXT);
    load_ok &= KVAF_LOAD_RUNTIME_SYMBOL(create_jfc, "urma_create_jfc",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JFC);
    load_ok &= KVAF_LOAD_RUNTIME_SYMBOL(delete_jfc, "urma_delete_jfc",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_JFC);
    load_ok &= KVAF_LOAD_RUNTIME_SYMBOL(create_jfr, "urma_create_jfr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JFR);
    load_ok &= KVAF_LOAD_RUNTIME_SYMBOL(delete_jfr, "urma_delete_jfr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_JFR);

#define KVAF_PROBE_OPTIONAL_SYMBOL(symbol_name, symbol_bit) do { \
    void *optional_symbol = NULL; \
    (void)kvaf_runtime_probe_load_symbol(library, symbol_name, \
        &optional_symbol, sizeof(optional_symbol), symbol_bit, false, \
        init_dso_base, result); \
} while (0)
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_alloc_jfc",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JFC);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_set_jfc_opt",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JFC_OPT);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_get_jfc_opt",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JFC_OPT);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_active_jfc",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JFC);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_deactive_jfc",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JFC);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_free_jfc",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JFC);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_alloc_jfs",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JFS);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_set_jfs_opt",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JFS_OPT);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_get_jfs_opt",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JFS_OPT);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_active_jfs",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JFS);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_deactive_jfs",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JFS);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_free_jfs",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JFS);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_alloc_jfr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JFR);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_set_jfr_opt",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JFR_OPT);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_get_jfr_opt",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JFR_OPT);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_active_jfr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JFR);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_deactive_jfr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JFR);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_free_jfr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JFR);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_alloc_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_set_jetty_opt",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_SET_JETTY_OPT);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_get_jetty_opt",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_JETTY_OPT);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_active_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACTIVE_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_deactive_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_DEACTIVE_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_free_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_register_seg",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_REGISTER_SEG);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_unregister_seg",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNREGISTER_SEG);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_import_seg",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_IMPORT_SEG);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_unimport_seg",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNIMPORT_SEG);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_post_jfs_wr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JFS_WR);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_post_jfr_wr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JFR_WR);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_poll_jfc",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_POLL_JFC);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_create_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_CREATE_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_delete_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_DELETE_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_import_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_IMPORT_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_unimport_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNIMPORT_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_bind_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_BIND_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_unbind_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_UNBIND_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_query_device",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_QUERY_DEVICE);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_str_to_eid",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_STR_TO_EID);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_get_device_by_eid",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_DEVICE_BY_EID);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_get_eid_list",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_EID_LIST);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_free_eid_list",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_EID_LIST);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_alloc_token_id",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_ALLOC_TOKEN_ID);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_free_token_id",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_FREE_TOKEN_ID);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_get_tp_list",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_TP_LIST);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_import_jetty_ex",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_IMPORT_JETTY_EX);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_post_jetty_send_wr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JETTY_SEND_WR);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_post_jetty_recv_wr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_POST_JETTY_RECV_WR);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_modify_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_MODIFY_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_query_jetty",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_QUERY_JETTY);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_modify_jfr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_MODIFY_JFR);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_query_jfr",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_QUERY_JFR);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_get_async_event",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_GET_ASYNC_EVENT);
    KVAF_PROBE_OPTIONAL_SYMBOL("urma_ack_async_event",
        KVAF_URMA_RUNTIME_PROBE_SYMBOL_ACK_ASYNC_EVENT);
#undef KVAF_PROBE_OPTIONAL_SYMBOL
    return load_ok;
#undef KVAF_LOAD_RUNTIME_SYMBOL
}

static void kvaf_runtime_probe_record_library_path(
    struct kvaf_urma_runtime_probe_result_v2 *result,
    kvaf_urma_init_fn init_function)
{
    Dl_info info = {0};
    void *address = NULL;
    size_t source_index;
    size_t length;

    (void)memcpy(&address, &init_function, sizeof(address));
    if (dladdr(address, &info) == 0 || info.dli_fname == NULL) {
        return;
    }
    length = strlen(info.dli_fname);
    if (length >= sizeof(result->runtime_library_path)) {
        length = sizeof(result->runtime_library_path) - 1U;
        result->flags |= KVAF_URMA_RUNTIME_PROBE_LIBRARY_PATH_TRUNCATED;
    }
    for (source_index = 0; source_index < length; ++source_index) {
        unsigned char value = (unsigned char)info.dli_fname[source_index];

        result->runtime_library_path[source_index] =
            value >= 0x21U && value <= 0x7eU ? (char)value : '_';
    }
    result->runtime_library_path[length] = '\0';
}

static bool kvaf_runtime_probe_args_are_valid(
    const struct kvaf_urma_aicpu_runtime_probe_args_v2 *args)
{
    return args != NULL &&
        args->magic == KVAF_URMA_RUNTIME_PROBE_MAGIC &&
        args->abi_version == KVAF_URMA_RUNTIME_PROBE_ABI_VERSION &&
        args->struct_size == sizeof(*args) &&
        args->expected_provider_abi_token ==
            KVAF_URMA_RUNTIME_PROBE_PROVIDER_ABI_TOKEN &&
        (args->request_nonce[0] != 0U || args->request_nonce[1] != 0U) &&
        args->logical_device_id >= 0 &&
        args->reserved0 == 0U &&
        !kvaf_runtime_probe_bytes_are_zero(
            args->expected_provider_build_id_sha256,
            sizeof(args->expected_provider_build_id_sha256)) &&
        args->result_address != 0 &&
        (args->result_address %
         KVAF_URMA_RUNTIME_PROBE_CACHE_LINE_BYTES) == 0 &&
        args->mode <= KVAF_URMA_RUNTIME_PROBE_MODE_INIT_DEVICE &&
        args->queue_depth != 0 &&
        args->queue_depth <= KVAF_URMA_RUNTIME_PROBE_MAX_QUEUE_DEPTH &&
        (args->queue_depth & (args->queue_depth - 1U)) == 0 &&
        kvaf_runtime_probe_has_terminator(
            args->kernel_name, sizeof(args->kernel_name)) &&
        args->kernel_name[0] != '\0' &&
        kvaf_runtime_probe_has_terminator(
            args->so_name, sizeof(args->so_name)) &&
        args->so_name[0] != '\0' &&
        (args->mode == KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER ||
         (kvaf_runtime_probe_has_terminator(
              args->device_name, sizeof(args->device_name)) &&
          args->device_name[0] != '\0'));
}

int aicpu_kvaf_urma_rt_probe_v2(void *opaque_args)
{
    struct kvaf_urma_aicpu_runtime_probe_args_v2 *args =
        (struct kvaf_urma_aicpu_runtime_probe_args_v2 *)opaque_args;
    struct kvaf_urma_runtime_probe_result_v2 *result;
    struct kvaf_runtime_urma_api api;
    urma_jfc_cfg_t jfc_config = {0};
    urma_jfr_cfg_t jfr_config = {0};
    urma_device_t *device = NULL;
    urma_context_t *context = NULL;
    urma_jfc_t *jfc = NULL;
    urma_jfr_t *jfr = NULL;
    void *library = NULL;
    char device_name[KVAF_URMA_PROVIDER_NAME_BYTES];
    urma_status_t status;
    bool runtime_owned = false;
    bool cleanup_failed = false;
    bool lock_acquired = false;
    int operation_errno;
    uint64_t required_symbol_mask;

    if (args == NULL || args->result_address == 0 ||
        (args->result_address %
         KVAF_URMA_RUNTIME_PROBE_CACHE_LINE_BYTES) != 0) {
        return -1;
    }
    result = (struct kvaf_urma_runtime_probe_result_v2 *)(uintptr_t)
        args->result_address;
    kvaf_runtime_probe_initialize_result(result, args->queue_depth);
    if (!kvaf_runtime_probe_args_are_valid(args)) {
        return 0;
    }
    result->request_nonce[0] = args->request_nonce[0];
    result->request_nonce[1] = args->request_nonce[1];
    result->provider_abi_token =
        KVAF_URMA_RUNTIME_PROBE_PROVIDER_ABI_TOKEN;
    result->mode = args->mode;
    result->eid_index = args->eid_index;
    result->logical_device_id = args->logical_device_id;
    result->args_size = (uint32_t)sizeof(*args);
    required_symbol_mask =
        args->mode == KVAF_URMA_RUNTIME_PROBE_MODE_CREATE_DELETE
        ? KVAF_URMA_RUNTIME_PROBE_LIFECYCLE_SYMBOLS
        : KVAF_URMA_RUNTIME_PROBE_BASE_SYMBOLS;
    result->required_symbol_mask = required_symbol_mask;
    if (!kvaf_runtime_probe_compiled_build_id(
            result->provider_build_id_sha256)) {
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH,
            KVAF_URMA_RUNTIME_PROBE_STAGE_NONE, EPROTO, 0);
        return 0;
    }
    if (memcmp(args->expected_provider_build_id_sha256,
               result->provider_build_id_sha256,
               sizeof(result->provider_build_id_sha256)) != 0) {
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROVIDER_IDENTITY_MISMATCH,
            KVAF_URMA_RUNTIME_PROBE_STAGE_NONE, ESTALE, 0);
        return 0;
    }
    /*
     * The observation ledger which would authorize even loader constructors
     * has not been implemented.  Stop before the process-global mutex and,
     * critically, before dlopen(liburma.so.0), urma_init or any resource API.
     * This entry-level gate applies to arbitrary KFC callers; no Host binary,
     * shell variable, probe mode or request field can enable it.
     */
    kvaf_runtime_probe_record_primary(result,
        KVAF_URMA_RUNTIME_PROBE_OUTCOME_CAPABILITY_LEDGER_UNAVAILABLE,
        KVAF_URMA_RUNTIME_PROBE_STAGE_NONE, -ENOTSUP, 0);
    return 0;

    operation_errno = pthread_mutex_trylock(&g_kvaf_runtime_probe_lock);
    if (operation_errno != 0) {
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_CONCURRENT_USE,
            KVAF_URMA_RUNTIME_PROBE_STAGE_NONE, operation_errno, 0);
        return 0;
    }
    lock_acquired = true;
    if (g_kvaf_runtime_probe_poisoned) {
        result->flags |= KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED;
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_PROCESS_POISONED,
            KVAF_URMA_RUNTIME_PROBE_STAGE_NONE, EOWNERDEAD, 0);
        goto finish;
    }
    (void)memcpy(device_name, args->device_name, sizeof(device_name));
    device_name[sizeof(device_name) - 1U] = '\0';

    /*
     * The official DSO may execute constructors before dlopen returns, and a
     * failed urma_init may still leave opaque process or kernel state.  A
     * dlclose return value proves only handle release.  Even an owned
     * init/uninit round trip has no documented AICPU cleanup contract, so the
     * uncertainty remains latched for the entire boot.
     */
    result->flags |=
        KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN;
    errno = 0;
    library = dlopen(KVAF_RUNTIME_URMA_SONAME, RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        kvaf_runtime_probe_record_loader_error(result, dlerror());
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_LIBRARY_UNAVAILABLE,
            KVAF_URMA_RUNTIME_PROBE_STAGE_LOAD_LIBRARY, -1, 0);
        goto finish;
    }
    result->cleanup_required |= KVAF_URMA_RUNTIME_PROBE_CLEANUP_LIBRARY;
    if (!kvaf_runtime_probe_load_api(
            library, &api, required_symbol_mask, result)) {
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_SYMBOL_UNAVAILABLE,
            KVAF_URMA_RUNTIME_PROBE_STAGE_RESOLVE_SYMBOLS, -1, 0);
        goto cleanup;
    }
    kvaf_runtime_probe_record_library_path(result, api.init);
    if (args->mode == KVAF_URMA_RUNTIME_PROBE_MODE_DISCOVER) {
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK,
            KVAF_URMA_RUNTIME_PROBE_STAGE_COMPLETE, 0, 0);
        goto cleanup;
    }
    result->flags |=
        KVAF_URMA_RUNTIME_PROBE_RESOURCE_OPERATION_ATTEMPTED;

    errno = 0;
    status = api.init(NULL);
    operation_errno = errno;
    if (status == URMA_EEXIST) {
        result->flags |= KVAF_URMA_RUNTIME_PROBE_RUNTIME_BORROWED;
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_RUNTIME_ALREADY_INITIALIZED,
            KVAF_URMA_RUNTIME_PROBE_STAGE_INIT, status, operation_errno);
        /* liburma state is process-global.  A probe that did not acquire it
         * must not create resources in, or uninitialize, another owner's
         * runtime domain. */
        goto cleanup;
    } else if (status != URMA_SUCCESS) {
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_INIT_FAILED,
            KVAF_URMA_RUNTIME_PROBE_STAGE_INIT, status, operation_errno);
        goto cleanup;
    } else {
        runtime_owned = true;
        result->cleanup_required |=
            KVAF_URMA_RUNTIME_PROBE_CLEANUP_RUNTIME;
    }

    errno = 0;
    device = api.get_device_by_name(device_name);
    operation_errno = errno;
    if (device == NULL) {
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_DEVICE_NOT_FOUND,
            KVAF_URMA_RUNTIME_PROBE_STAGE_GET_DEVICE, -1,
            operation_errno);
        goto cleanup;
    }
    if (args->mode == KVAF_URMA_RUNTIME_PROBE_MODE_INIT_DEVICE) {
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK,
            KVAF_URMA_RUNTIME_PROBE_STAGE_COMPLETE, 0, 0);
        goto cleanup;
    }

    errno = 0;
    context = api.create_context(device, args->eid_index);
    operation_errno = errno;
    if (context == NULL) {
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_CONTEXT_CREATE_FAILED_UNPROVEN,
            KVAF_URMA_RUNTIME_PROBE_STAGE_CREATE_CONTEXT, -1,
            operation_errno);
        result->flags |=
            KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN;
        goto cleanup;
    }
    result->cleanup_required |= KVAF_URMA_RUNTIME_PROBE_CLEANUP_CONTEXT;

    jfc_config.depth = args->queue_depth;
    errno = 0;
    jfc = api.create_jfc(context, &jfc_config);
    operation_errno = errno;
    if (jfc == NULL) {
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_JFC_CREATE_FAILED_UNPROVEN,
            KVAF_URMA_RUNTIME_PROBE_STAGE_CREATE_JFC, -1,
            operation_errno);
        result->flags |=
            KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN;
        goto cleanup;
    }
    result->cleanup_required |= KVAF_URMA_RUNTIME_PROBE_CLEANUP_JFC;

    jfr_config.depth = args->queue_depth;
    jfr_config.trans_mode = URMA_TM_RM;
    jfr_config.max_sge = 1;
    jfr_config.min_rnr_timer = URMA_TYPICAL_MIN_RNR_TIMER;
    jfr_config.jfc = jfc;
    jfr_config.token_value.token = KVAF_RUNTIME_PROBE_TOKEN;
    errno = 0;
    jfr = api.create_jfr(context, &jfr_config);
    operation_errno = errno;
    if (jfr == NULL) {
        kvaf_runtime_probe_record_primary(result,
            KVAF_URMA_RUNTIME_PROBE_OUTCOME_JFR_CREATE_FAILED_UNPROVEN,
            KVAF_URMA_RUNTIME_PROBE_STAGE_CREATE_JFR, -1,
            operation_errno);
        result->flags |=
            KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN;
        goto cleanup;
    }
    result->cleanup_required |= KVAF_URMA_RUNTIME_PROBE_CLEANUP_JFR;
    kvaf_runtime_probe_record_primary(result,
        KVAF_URMA_RUNTIME_PROBE_OUTCOME_OK,
        KVAF_URMA_RUNTIME_PROBE_STAGE_COMPLETE, 0, 0);

cleanup:
    if (jfr != NULL) {
        errno = 0;
        status = api.delete_jfr(jfr);
        operation_errno = errno;
        if (status == URMA_SUCCESS) {
            result->cleanup_completed |=
                KVAF_URMA_RUNTIME_PROBE_CLEANUP_JFR;
            jfr = NULL;
        } else {
            kvaf_runtime_probe_record_cleanup_failure(result,
                KVAF_URMA_RUNTIME_PROBE_STAGE_DELETE_JFR,
                status, operation_errno);
            cleanup_failed = true;
        }
    }
    if (!cleanup_failed && jfc != NULL) {
        errno = 0;
        status = api.delete_jfc(jfc);
        operation_errno = errno;
        if (status == URMA_SUCCESS) {
            result->cleanup_completed |=
                KVAF_URMA_RUNTIME_PROBE_CLEANUP_JFC;
            jfc = NULL;
        } else {
            kvaf_runtime_probe_record_cleanup_failure(result,
                KVAF_URMA_RUNTIME_PROBE_STAGE_DELETE_JFC,
                status, operation_errno);
            cleanup_failed = true;
        }
    }
    if (!cleanup_failed && context != NULL) {
        errno = 0;
        status = api.delete_context(context);
        operation_errno = errno;
        if (status == URMA_SUCCESS) {
            result->cleanup_completed |=
                KVAF_URMA_RUNTIME_PROBE_CLEANUP_CONTEXT;
            context = NULL;
        } else {
            kvaf_runtime_probe_record_cleanup_failure(result,
                KVAF_URMA_RUNTIME_PROBE_STAGE_DELETE_CONTEXT,
                status, operation_errno);
            cleanup_failed = true;
        }
    }
    if (!cleanup_failed && runtime_owned &&
        (result->flags &
         KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN) == 0U) {
        errno = 0;
        status = api.uninit();
        operation_errno = errno;
        if (status == URMA_SUCCESS) {
            result->cleanup_completed |=
                KVAF_URMA_RUNTIME_PROBE_CLEANUP_RUNTIME;
            runtime_owned = false;
        } else {
            kvaf_runtime_probe_record_cleanup_failure(result,
                KVAF_URMA_RUNTIME_PROBE_STAGE_UNINIT,
                status, operation_errno);
            cleanup_failed = true;
        }
    }
    if (!cleanup_failed && library != NULL &&
        (result->flags &
         KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN) == 0U) {
        errno = 0;
        status = dlclose(library);
        operation_errno = errno;
        if (status == 0) {
            result->cleanup_completed |=
                KVAF_URMA_RUNTIME_PROBE_CLEANUP_LIBRARY;
            library = NULL;
        } else {
            kvaf_runtime_probe_record_loader_error(result, dlerror());
            kvaf_runtime_probe_record_cleanup_failure(result,
                KVAF_URMA_RUNTIME_PROBE_STAGE_CLOSE_LIBRARY,
                status, operation_errno);
        }
    }
finish:
    if (lock_acquired) {
        if ((result->flags &
             (KVAF_URMA_RUNTIME_PROBE_RUNTIME_BORROWED |
              KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN |
              KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED |
              KVAF_URMA_RUNTIME_PROBE_OFFICIAL_SIDE_EFFECTS_CLEANUP_UNPROVEN)) != 0U ||
            result->outcome ==
                KVAF_URMA_RUNTIME_PROBE_OUTCOME_CLEANUP_UNSAFE ||
            result->cleanup_required != result->cleanup_completed) {
            g_kvaf_runtime_probe_poisoned = true;
            if ((result->flags &
                 KVAF_URMA_RUNTIME_PROBE_RUNTIME_BORROWED) == 0U) {
                result->flags |=
                    KVAF_URMA_RUNTIME_PROBE_PROCESS_POISONED;
            }
        }
        (void)pthread_mutex_unlock(&g_kvaf_runtime_probe_lock);
    }
    return 0;
}
