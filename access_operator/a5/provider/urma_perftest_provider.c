#include "kvaf_urma_provider_abi.h"

#include <arpa/inet.h>
#include <errno.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "perftest_parameters.h"
#include "perftest_resources.h"
#include "kvaf_private_udma_accessor.h"
#include "kvaf_source_matched_perftest.h"
#include "kvaf_source_matched_udma.h"
#include "kvaf_source_matched_ummu.h"

#define KVAF_PROVIDER_JETTYS 1U
#define KVAF_PROVIDER_PAGE_SIZE 4096U
#define KVAF_PROVIDER_TID_SHIFT 8U
#define KVAF_PROVIDER_PRIVATE_ABI_MASK \
    (KVAF_URMA_PROVIDER_EXPECT_HOST_PRIVATE_ABI | \
     KVAF_URMA_PROVIDER_EXPECT_DEVICE_PRIVATE_ABI)
#define KVAF_PROVIDER_MANAGEMENT_MASK \
    (KVAF_URMA_PROVIDER_EXPLICIT_TCP_MANAGEMENT | \
     KVAF_URMA_PROVIDER_URMA_MANAGEMENT)
#define KVAF_PROVIDER_ALLOWED_CONFIG_FLAGS \
    (KVAF_URMA_PROVIDER_USE_CTP | \
     KVAF_PROVIDER_PRIVATE_ABI_MASK | \
     KVAF_PROVIDER_MANAGEMENT_MASK)

static perftest_config_t g_kvaf_config;
static perftest_context_t g_kvaf_context;
static bool g_kvaf_config_initialized;
static bool g_kvaf_connection_open;
static bool g_kvaf_context_open;
static bool g_kvaf_runtime_acquired;
static bool g_kvaf_cleanup_poisoned;
static uint64_t g_kvaf_session_id;
static uint64_t g_kvaf_last_closed_session_id;
static struct kvaf_udma_direct_queue_info g_kvaf_queue_info;
static struct kvaf_perftest_teardown_state g_kvaf_teardown_state;
static pthread_mutex_t g_kvaf_lock = PTHREAD_MUTEX_INITIALIZER;
/*
 * The provider state above is process/SO-global.  A resource-owning epoch
 * invocation must therefore exclude every legacy/public Open and Close for
 * its complete lifetime, not merely while one provider call holds the mutex.
 * The global claim is protected by g_kvaf_lock; the thread-local identity is
 * the capability which permits only that exact invocation to enter the public
 * provider wrappers while the claim is active.
 */
static bool g_kvaf_epoch_claim_active;
static uint64_t g_kvaf_epoch_claim_epoch;
static uint64_t g_kvaf_epoch_claim_session_id;
static _Thread_local bool g_kvaf_epoch_owner_authorized;
static _Thread_local uint64_t g_kvaf_epoch_owner_epoch;
static _Thread_local uint64_t g_kvaf_epoch_owner_session_id;

static bool kvaf_epoch_owner_is_authorized_unlocked(uint64_t session_id)
{
    return g_kvaf_epoch_claim_active &&
        g_kvaf_epoch_owner_authorized &&
        g_kvaf_epoch_claim_epoch != 0 &&
        g_kvaf_epoch_claim_epoch == g_kvaf_epoch_owner_epoch &&
        g_kvaf_epoch_claim_session_id != 0 &&
        g_kvaf_epoch_claim_session_id == session_id &&
        g_kvaf_epoch_owner_session_id == session_id;
}

static bool kvaf_provider_has_local_state_unlocked(void)
{
    return g_kvaf_config_initialized || g_kvaf_connection_open ||
        g_kvaf_context_open || g_kvaf_runtime_acquired ||
        g_kvaf_cleanup_poisoned || g_kvaf_session_id != 0;
}

static bool kvaf_is_power_of_two(uint32_t value)
{
    return value != 0 && (value & (value - 1U)) == 0;
}

static bool kvaf_has_terminator(const char *text, size_t capacity)
{
    return text != NULL && memchr(text, '\0', capacity) != NULL;
}

static int kvaf_numeric_ip_family(const char *text)
{
    struct in_addr ipv4;
    struct in6_addr ipv6;

    if (text == NULL || text[0] == '\0') {
        return AF_UNSPEC;
    }
    if (inet_pton(AF_INET, text, &ipv4) == 1) {
        return AF_INET;
    }
    if (inet_pton(AF_INET6, text, &ipv6) == 1) {
        return AF_INET6;
    }
    return AF_UNSPEC;
}

static bool kvaf_valid_nonzero_urma_eid(const char *text)
{
    urma_eid_t eid = {0};
    uint8_t combined = 0;
    uint32_t i;

    if (text == NULL || text[0] == '\0' ||
        urma_str_to_eid(text, &eid) != 0) {
        return false;
    }
    for (i = 0; i < sizeof(eid.raw); ++i) {
        combined |= eid.raw[i];
    }
    return combined != 0;
}

static bool kvaf_same_urma_eid(const char *left, const char *right)
{
    urma_eid_t left_eid = {0};
    urma_eid_t right_eid = {0};

    return left != NULL && right != NULL &&
        urma_str_to_eid(left, &left_eid) == 0 &&
        urma_str_to_eid(right, &right_eid) == 0 &&
        memcmp(&left_eid, &right_eid, sizeof(left_eid)) == 0;
}

static int kvaf_negative_errno_or(int fallback)
{
    return errno > 0 ? -errno : -fallback;
}

static bool kvaf_allocation_covers_parsed_config(
    const struct kvaf_urma_provider_config_v2 *input)
{
    uint64_t half = input->segment_payload_bytes;
    uint64_t alignment = g_kvaf_config.cache_line_size;

    if (half < g_kvaf_config.page_size) {
        half = g_kvaf_config.page_size;
    }
    if (alignment == 0) {
        return false;
    }
    if (half <= alignment) {
        half = alignment;
    } else {
        uint64_t remainder = half % alignment;
        if (remainder != 0) {
            if (half > UINT64_MAX - (alignment - remainder)) {
                return false;
            }
            half += alignment - remainder;
        }
    }
    return half <= UINT64_MAX / 2U &&
        input->local_allocation_bytes >= half * 2U;
}

static void kvaf_init_result(struct kvaf_urma_provider_result_v2 *result,
    enum kvaf_urma_provider_status status, int provider_status)
{
    (void)memset(result, 0, sizeof(*result));
    result->magic = KVAF_URMA_PROVIDER_MAGIC;
    result->abi_version = KVAF_URMA_PROVIDER_ABI_VERSION;
    result->struct_size = (uint32_t)sizeof(*result);
    result->status = (uint32_t)status;
    result->provider_status = provider_status;
}

static uint64_t kvaf_pack_signed_pair(int32_t low, int32_t high)
{
    return (uint64_t)(uint32_t)low |
        ((uint64_t)(uint32_t)high << 32U);
}

static void kvaf_attach_management_diagnostic(
    struct kvaf_urma_provider_result_v2 *result)
{
    struct kvaf_perftest_management_diagnostic_v1 diagnostic = {
        .structure_size = sizeof(diagnostic),
        .abi_version = KVAF_URMA_MANAGEMENT_DIAGNOSTIC_VERSION,
    };

    if (kvaf_perftest_management_diagnostic_get(&diagnostic) != 0 ||
        diagnostic.structure_size != sizeof(diagnostic) ||
        diagnostic.abi_version !=
            KVAF_URMA_MANAGEMENT_DIAGNOSTIC_VERSION ||
        diagnostic.stage == KVAF_URMA_MANAGEMENT_STAGE_NONE) {
        return;
    }
    result->reserved[KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RESERVED_STAGE] =
        (uint64_t)diagnostic.stage |
        ((uint64_t)diagnostic.abi_version << 32U);
    result->reserved[KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RESERVED_ERRNO] =
        kvaf_pack_signed_pair(
            diagnostic.raw_errno, diagnostic.effective_errno);
    result->reserved[KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RESERVED_STATUS] =
        kvaf_pack_signed_pair(
            diagnostic.operation_status,
            diagnostic.completion_status);
    result->reserved[KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RESERVED_FLAGS] =
        diagnostic.flags;
    /* Publish the validity marker last so a progress reader cannot accept a
     * partially populated diagnostic payload. */
    result->reserved[KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RESERVED_MAGIC] =
        KVAF_URMA_MANAGEMENT_DIAGNOSTIC_MAGIC;
}

static bool kvaf_config_is_valid(
    const struct kvaf_urma_provider_config_v2 *config)
{
    uint64_t minimum_bytes;
    uint32_t expected_private_abi;
    uint32_t management_flags;
    int control_family;
    int bind_family;
    bool use_urma_management;

    if (config == NULL ||
        config->magic != KVAF_URMA_PROVIDER_MAGIC ||
        config->abi_version != KVAF_URMA_PROVIDER_ABI_VERSION ||
        config->struct_size != sizeof(*config) ||
        config->role > KVAF_URMA_PROVIDER_CLIENT ||
        (config->flags & KVAF_URMA_PROVIDER_USE_CTP) == 0 ||
        (config->flags & KVAF_PROVIDER_PRIVATE_ABI_MASK) == 0 ||
        (config->flags & KVAF_PROVIDER_PRIVATE_ABI_MASK) ==
            KVAF_PROVIDER_PRIVATE_ABI_MASK ||
        (config->flags & ~KVAF_PROVIDER_ALLOWED_CONFIG_FLAGS) != 0 ||
        config->address_kind != KVAF_URMA_PROVIDER_ADDRESS_DEVICE_VA ||
        config->session_id == 0 ||
        config->jetty_index != 0 ||
        config->control_timeout_seconds == 0 ||
        config->control_timeout_seconds > 3600U ||
        config->reserved0 != 0U || config->reserved[0] != 0U ||
        config->reserved[1] != 0U ||
        config->local_address == 0 ||
        config->segment_payload_bytes == 0 ||
        config->segment_payload_bytes > UINT32_MAX ||
        !kvaf_has_terminator(config->device_name,
            sizeof(config->device_name)) ||
        config->device_name[0] == '\0' ||
        !kvaf_has_terminator(config->control_server_address,
            sizeof(config->control_server_address)) ||
        !kvaf_has_terminator(config->control_bind_address,
            sizeof(config->control_bind_address))) {
        return false;
    }

    management_flags = config->flags & KVAF_PROVIDER_MANAGEMENT_MASK;
    if (management_flags == 0 ||
        management_flags == KVAF_PROVIDER_MANAGEMENT_MASK) {
        return false;
    }
    use_urma_management =
        management_flags == KVAF_URMA_PROVIDER_URMA_MANAGEMENT;
    if (use_urma_management) {
        /* 1..1023 is the agreed well-known management-Jetty range.  It is
         * intentionally disjoint from ordinary TCP listener ports. */
        if (config->management_jetty_id == 0 ||
            config->management_jetty_id > 1023U ||
            !kvaf_valid_nonzero_urma_eid(
                config->management_local_address) ||
            !kvaf_valid_nonzero_urma_eid(
                config->management_peer_address) ||
            kvaf_same_urma_eid(config->management_local_address,
                config->management_peer_address)) {
            return false;
        }
    } else {
        if (config->control_port < 1024U ||
            config->control_port > 65535U) {
            return false;
        }
        control_family = kvaf_numeric_ip_family(
            config->control_server_address);
        bind_family = config->control_bind_address[0] == '\0'
            ? AF_UNSPEC
            : kvaf_numeric_ip_family(config->control_bind_address);
        if (control_family == AF_UNSPEC ||
            (config->role == KVAF_URMA_PROVIDER_SERVER &&
             config->control_bind_address[0] != '\0') ||
            (config->control_bind_address[0] != '\0' &&
             bind_family != control_family)) {
            return false;
        }
    }

    expected_private_abi =
        (config->flags &
         KVAF_URMA_PROVIDER_EXPECT_HOST_PRIVATE_ABI) != 0
            ? KVAF_UDMA_PRIVATE_ABI_HOST_PERFTEST_AIV_V1
            : KVAF_UDMA_PRIVATE_ABI_DEVICE_PERFTEST_AIV_V1;
    if (kvaf_urma_provider_private_abi_v2() !=
        expected_private_abi) {
        return false;
    }

    minimum_bytes = config->segment_payload_bytes;
    if (minimum_bytes < KVAF_PROVIDER_PAGE_SIZE) {
        minimum_bytes = KVAF_PROVIDER_PAGE_SIZE;
    }
    if (minimum_bytes > UINT64_MAX / 2U) {
        return false;
    }
    minimum_bytes *= 2U;
    return config->local_allocation_bytes >= minimum_bytes;
}

static int kvaf_set_control_socket_deadlines(
    const struct kvaf_urma_provider_config_v2 *config)
{
    uint32_t i;
    const struct timeval timeout = {
        .tv_sec = (time_t)config->control_timeout_seconds,
        .tv_usec = 0,
    };

    for (i = 0; i < g_kvaf_config.pair_num; ++i) {
        const int fd = kvaf_perftest_control_fd(i);
        if (fd < 0) {
            return kvaf_negative_errno_or(EBADF);
        }
        if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                &timeout, sizeof(timeout)) != 0 ||
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO,
                &timeout, sizeof(timeout)) != 0) {
            return kvaf_negative_errno_or(EIO);
        }
    }
    return 0;
}

static int kvaf_parse_config(
    const struct kvaf_urma_provider_config_v2 *input)
{
    char size_text[32];
    char eid_index_text[32];
    char port_text[16];
    char *argv[32];
    int argc = 0;
    const bool use_urma_management =
        (input->flags & KVAF_URMA_PROVIDER_URMA_MANAGEMENT) != 0;
    const int control_family = use_urma_management
        ? AF_UNSPEC
        : kvaf_numeric_ip_family(input->control_server_address);
    const uint32_t endpoint = use_urma_management
        ? input->management_jetty_id
        : input->control_port;

    if (snprintf(size_text, sizeof(size_text), "%llu",
        (unsigned long long)input->segment_payload_bytes) < 0 ||
        snprintf(eid_index_text, sizeof(eid_index_text), "%u",
        input->eid_index) < 0 ||
        snprintf(port_text, sizeof(port_text), "%u",
        endpoint) < 0) {
        return -1;
    }

#define KVAF_ADD_ARG(value) do { argv[argc++] = (char *)(value); } while (0)
    KVAF_ADD_ARG("kvaf_urma_provider");
    KVAF_ADD_ARG("write_bw");
    KVAF_ADD_ARG("-d");
    KVAF_ADD_ARG(input->device_name);
    KVAF_ADD_ARG("-J");
    KVAF_ADD_ARG("1");
    KVAF_ADD_ARG("-s");
    KVAF_ADD_ARG(size_text);
    if (use_urma_management) {
        KVAF_ADD_ARG("--mgmt_proto");
        KVAF_ADD_ARG("ub");
        KVAF_ADD_ARG("--mgmt_addr");
        KVAF_ADD_ARG(input->management_local_address);
        if (input->role == KVAF_URMA_PROVIDER_CLIENT) {
            KVAF_ADD_ARG("-S");
            KVAF_ADD_ARG(input->management_peer_address);
        }
    } else {
        if (control_family == AF_INET6) {
            KVAF_ADD_ARG("--enable_ipv6");
        }
        if (input->role == KVAF_URMA_PROVIDER_CLIENT) {
            KVAF_ADD_ARG("-S");
            KVAF_ADD_ARG(input->control_server_address);
            if (input->control_bind_address[0] != '\0') {
                KVAF_ADD_ARG("--bind_ip");
                KVAF_ADD_ARG(input->control_bind_address);
            }
        }
    }
    KVAF_ADD_ARG("--tp_aware");
    KVAF_ADD_ARG("--ctp");
    KVAF_ADD_ARG("--page_size");
    KVAF_ADD_ARG("4096");
    KVAF_ADD_ARG("--eid_idx");
    KVAF_ADD_ARG(eid_index_text);
    KVAF_ADD_ARG("--port");
    KVAF_ADD_ARG(port_text);
    // UB_MicroBenchmark_new's AIV WQE path was validated with tokenEn=0.
    // Make that policy explicit instead of relying on the perftest default.
    KVAF_ADD_ARG("--token_policy");
    KVAF_ADD_ARG("0");
#undef KVAF_ADD_ARG

    optind = 1;
    return perftest_parse_args(argc, argv, &g_kvaf_config);
}

static bool kvaf_ummu_checked_runtime_is_armed(void)
{
    struct kvaf_source_ummu_status status = {
        .structure_size = sizeof(status),
    };
    const uint64_t expected_capability =
        KVAF_SOURCE_UMMU_CHECKED_RETRYABLE_V1 |
        KVAF_SOURCE_UMMU_PINNED_FULL_UAPI;

    /*
     * Do not advertise the checked UDMA teardown capability merely because
     * the provider was compiled with the private UMMU object.  Prove that
     * the running instance exposes the pinned full-UAPI contract and that
     * the just-created TID has no deferred or state-unknown cleanup work.
     */
    return kvaf_source_ummu_checked_contract_ready() == 1 &&
        kvaf_source_ummu_get_status(&status) == 0 &&
        status.abi_version == KVAF_SOURCE_UMMU_STATUS_ABI_VERSION &&
        status.structure_size == sizeof(status) &&
        status.capability_magic == KVAF_SOURCE_UMMU_CAPABILITY_MAGIC &&
        status.capability_bits == expected_capability &&
        status.live_tid_count != 0U &&
        status.pending_operation_count == 0U &&
        status.retained_mapping_count == 0U &&
        status.last_state_unknown == 0U;
}

static bool kvaf_context_is_complete(
    const struct kvaf_urma_provider_config_v2 *config)
{
    uint32_t expected_private_abi;

    if (g_kvaf_context.urma_ctx == NULL ||
        kvaf_udma_source_provider_is_registered() != 1 ||
        !kvaf_ummu_checked_runtime_is_armed() ||
        kvaf_udma_source_public_abi_matches(
            kvaf_urma_public_abi_signature()) != 1 ||
        kvaf_udma_source_provider_owns_context(
            g_kvaf_context.urma_ctx) != 1 ||
        g_kvaf_context.jetty_num != KVAF_PROVIDER_JETTYS ||
        g_kvaf_context.local_buf == NULL ||
        !g_kvaf_config.enable_va ||
        g_kvaf_config.v_address != config->local_address ||
        g_kvaf_context.local_buf[0] !=
            (void *)(uintptr_t)config->local_address ||
        g_kvaf_context.local_tseg == NULL ||
        g_kvaf_context.local_tseg[0] == NULL ||
        g_kvaf_context.remote_seg == NULL ||
        g_kvaf_context.remote_jetty_id == NULL ||
        g_kvaf_context.import_tseg == NULL ||
        g_kvaf_context.import_tseg[0] == NULL ||
        g_kvaf_context.jfc_s == NULL ||
        g_kvaf_context.jfc_s[0] == NULL ||
        (g_kvaf_config.share_jfr &&
         (g_kvaf_context.jfr == NULL ||
          g_kvaf_context.jfr[0] == NULL)) ||
        g_kvaf_context.jetty == NULL ||
        g_kvaf_context.jetty[0] == NULL ||
        g_kvaf_context.buf_len == 0 ||
        g_kvaf_context.buf_len > config->local_allocation_bytes) {
        return false;
    }

    if (kvaf_udma_extract_direct_queue_info(
            g_kvaf_context.jetty[0], g_kvaf_context.jfc_s[0],
            &g_kvaf_queue_info) != 0) {
        return false;
    }
    expected_private_abi =
        (config->flags &
         KVAF_URMA_PROVIDER_EXPECT_HOST_PRIVATE_ABI) != 0
            ? KVAF_UDMA_PRIVATE_ABI_HOST_PERFTEST_AIV_V1
            : KVAF_UDMA_PRIVATE_ABI_DEVICE_PERFTEST_AIV_V1;
    return g_kvaf_queue_info.struct_size ==
            sizeof(g_kvaf_queue_info) &&
        g_kvaf_queue_info.private_abi == expected_private_abi &&
        g_kvaf_queue_info.sq_reserved == 0 &&
        g_kvaf_queue_info.sq_base != 0 &&
        g_kvaf_queue_info.sq_doorbell != 0 &&
        g_kvaf_queue_info.sq_baseblock_shift == 6U &&
        g_kvaf_queue_info.sq_baseblock_count >=
            g_kvaf_config.jfs_depth &&
        kvaf_is_power_of_two(
            g_kvaf_queue_info.sq_baseblock_count) &&
        g_kvaf_queue_info.cq_base != 0 &&
        g_kvaf_queue_info.cq_doorbell != 0 &&
        g_kvaf_queue_info.cqe_size == 64U &&
        g_kvaf_queue_info.cq_baseblock_shift == 6U &&
        g_kvaf_queue_info.cq_baseblock_count >=
            g_kvaf_config.jfc_depth &&
        kvaf_is_power_of_two(
            g_kvaf_queue_info.cq_baseblock_count) &&
        g_kvaf_queue_info.sq_producer_index == 0 &&
        g_kvaf_queue_info.sq_consumer_index == 0 &&
        g_kvaf_queue_info.cq_producer_index == 0 &&
        g_kvaf_queue_info.cq_consumer_index == 0 &&
        g_kvaf_context.import_tjetty != NULL &&
        g_kvaf_context.import_tjetty[0] != NULL &&
        g_kvaf_context.import_tjetty[0]->tp.tpn < (1U << 24U) &&
        (g_kvaf_context.remote_seg[0].token_id &
            ((1U << KVAF_PROVIDER_TID_SHIFT) - 1U)) == 0 &&
        (g_kvaf_context.remote_seg[0].token_id >>
            KVAF_PROVIDER_TID_SHIFT) < (1U << 20U) &&
        (g_kvaf_context.local_tseg[0]->seg.token_id &
            ((1U << KVAF_PROVIDER_TID_SHIFT) - 1U)) == 0 &&
        (g_kvaf_context.local_tseg[0]->seg.token_id >>
            KVAF_PROVIDER_TID_SHIFT) < (1U << 20U) &&
        g_kvaf_context.remote_seg[0].len != 0;
}

static bool kvaf_remote_eid_is_nonzero(const uint8_t eid[16])
{
    uint32_t i;
    uint8_t combined = 0;
    for (i = 0; i < 16U; ++i) {
        combined |= eid[i];
    }
    return combined != 0;
}

static int kvaf_fill_result(
    struct kvaf_urma_provider_result_v2 *result)
{
    const uint32_t raw_sq_depth =
        g_kvaf_queue_info.sq_baseblock_count;

    /*
     * Public liburma query calls do not expose userspace SQ/CQ mappings.
     * Consume only the isolated ABI-pinned accessor output; never use
     * any copied perftest private layout for device addresses.
     */
    if (!kvaf_is_power_of_two(raw_sq_depth) ||
        !kvaf_is_power_of_two(
            g_kvaf_queue_info.cq_baseblock_count) ||
        !kvaf_ummu_checked_runtime_is_armed() ||
        !kvaf_remote_eid_is_nonzero(
            g_kvaf_context.import_tjetty[0]->id.eid.raw)) {
        kvaf_init_result(result,
            KVAF_URMA_PROVIDER_RESOURCE_INCOMPATIBLE, -1);
        return -1;
    }

    kvaf_init_result(result, KVAF_URMA_PROVIDER_OK, 0);
    result->flags = KVAF_URMA_PROVIDER_CONNECTED |
        KVAF_URMA_PROVIDER_REMOTE_EID_NETWORK_ORDER;
    if (g_kvaf_queue_info.private_abi ==
        KVAF_UDMA_PRIVATE_ABI_DEVICE_PERFTEST_AIV_V1) {
        result->flags |=
            KVAF_URMA_PROVIDER_QUEUE_IS_AICPU_SHARED_VA;
    }
    result->sq_depth = raw_sq_depth;
    result->cq_depth = g_kvaf_queue_info.cq_baseblock_count;
    result->sq_base = g_kvaf_queue_info.sq_base;
    result->sq_doorbell = g_kvaf_queue_info.sq_doorbell;
    result->cq_base = g_kvaf_queue_info.cq_base;
    result->cq_doorbell = g_kvaf_queue_info.cq_doorbell;
    result->local_base = g_kvaf_context.local_tseg[0]->seg.ubva.va;
    result->local_bytes = g_kvaf_context.local_tseg[0]->seg.len;
    result->remote_base = g_kvaf_context.remote_seg[0].ubva.va;
    result->remote_bytes = g_kvaf_context.remote_seg[0].len;
    result->wqe_size = 1U << g_kvaf_queue_info.sq_baseblock_shift;
    result->cqe_size = g_kvaf_queue_info.cqe_size;
    /*
     * Use the imported target Jetty's public TP result. This is the exact TPN
     * used by udma_set_sqe(). The selected UDMA provider does not implement
     * urma_ops.get_tpn, so perftest's ai->sq.tp_id assignment from
     * urma_get_tpn(local_jetty) contains a negative error converted to
     * uint32_t. The older AIV adapter worked around that bug by posting a
     * probe WQE and reading word2 back; consuming import_tjetty->tp.tpn avoids
     * both the unsupported API and queue pollution.
     */
    result->tp_id = g_kvaf_context.import_tjetty[0]->tp.tpn;
    result->remote_object_id =
        g_kvaf_context.remote_seg[0].token_id >>
        KVAF_PROVIDER_TID_SHIFT;
    // register_host_mem() deliberately uses URMA_TOKEN_NONE, matching the
    // validated UB_MicroBenchmark AIV WQE. Keep token values zero so a stale
    // sidecar value can never accidentally enable a different policy.
    result->remote_token_value = 0;
    result->local_object_id =
        g_kvaf_context.local_tseg[0]->seg.token_id >>
        KVAF_PROVIDER_TID_SHIFT;
    result->local_token_value = 0;
    result->sq_ring_shift = (uint32_t)__builtin_ctz(raw_sq_depth);
    /*
     * create_duplex_ctx() allocates fresh queues and creates only software
     * run descriptors; KVAF deliberately does not call prepare/post/warm-up.
     * Therefore raw SQ PI and CQ CI both start at zero.
     */
    result->sq_head = 0;
    result->cq_tail = 0;
    result->wqe_count = 0;
    result->private_abi = g_kvaf_queue_info.private_abi;
    result->session_id = g_kvaf_session_id;
    result->reserved[0] = g_kvaf_queue_info.layout_signature;
    /*
     * Runtime evidence, not a build-manifest assertion: owns_context() and
     * the source UMMU status/capability were checked immediately before the
     * private queue accessor/result were accepted. Consumers must require
     * both words before assigning any teardown assurance.
     */
    result->reserved[1] = KVAF_UDMA_SOURCE_CAPABILITY_MAGIC;
    result->reserved[2] =
        KVAF_UDMA_TEARDOWN_RUNTIME_CAPABILITY;
    result->reserved[3] =
        kvaf_udma_source_public_abi_signature();
    (void)memcpy(result->remote_eid,
        g_kvaf_context.import_tjetty[0]->id.eid.raw,
        sizeof(result->remote_eid));
    return 0;
}

static int kvaf_quiesce_aiv_queue(uint32_t final_sq_head,
    uint32_t final_cq_tail)
{
    if (!g_kvaf_context_open || final_sq_head != final_cq_tail) {
        return -1;
    }
    /*
     * The AIV stream is synchronized before this entry point and every posted
     * WQE requested and consumed one CQE, so equal counters prove the raw
     * queue is drained. Do not write private liburma PI/CI fields: their
     * layout is not an ABI, and destroy_ctx() does not consume those counters.
     */
    return 0;
}

static int kvaf_close_unlocked(uint64_t session_id);

static bool kvaf_source_ummu_state_is_unknown(void)
{
    return kvaf_source_ummu_cleanup_is_unknown() != 0;
}

static bool kvaf_source_cleanup_state_is_unknown(void)
{
    return kvaf_udma_source_cleanup_is_unknown() != 0 ||
        kvaf_source_ummu_state_is_unknown();
}

static bool kvaf_udma_source_has_no_live_resources(void)
{
    struct kvaf_source_ummu_status ummu_status = {
        .structure_size = sizeof(ummu_status),
    };

    /* Open can fail before exposing a context handle; drain that sidecar
     * before consulting any ordinary live-resource counter. */
    return kvaf_udma_source_cleanup_is_unknown() == 0 &&
        kvaf_udma_source_retry_orphan_contexts() == 0 &&
        kvaf_udma_source_live_segment_count() == 0U &&
        kvaf_udma_source_live_token_count() == 0U &&
        kvaf_udma_source_live_jetty_count() == 0U &&
        kvaf_udma_source_live_jfr_count() == 0U &&
        kvaf_udma_source_live_jfc_count() == 0U &&
        kvaf_udma_source_live_context_count() == 0U &&
        kvaf_source_ummu_checked_contract_ready() == 1 &&
        kvaf_source_ummu_get_status(&ummu_status) == 0 &&
        kvaf_source_ummu_cleanup_is_unknown() == 0 &&
        kvaf_source_ummu_live_tid_count() == 0U &&
        kvaf_source_ummu_pending_operation_count() == 0U &&
        kvaf_source_ummu_retained_mapping_count() == 0U;
}

enum kvaf_epoch_claim_result {
    KVAF_EPOCH_CLAIM_ACQUIRED = 0,
    KVAF_EPOCH_CLAIM_CONFLICT = 1,
    KVAF_EPOCH_CLAIM_LOCK_FAILED = 2,
};

static enum kvaf_epoch_claim_result kvaf_epoch_owner_acquire_claim(
    const struct kvaf_urma_aicpu_epoch_owner_args_v1 *args)
{
    if (pthread_mutex_lock(&g_kvaf_lock) != 0) {
        return KVAF_EPOCH_CLAIM_LOCK_FAILED;
    }
    if (g_kvaf_epoch_claim_active ||
        kvaf_provider_has_local_state_unlocked()) {
        (void)pthread_mutex_unlock(&g_kvaf_lock);
        return KVAF_EPOCH_CLAIM_CONFLICT;
    }
    g_kvaf_epoch_claim_active = true;
    g_kvaf_epoch_claim_epoch = args->epoch;
    g_kvaf_epoch_claim_session_id = args->config.session_id;
    g_kvaf_epoch_owner_authorized = true;
    g_kvaf_epoch_owner_epoch = args->epoch;
    g_kvaf_epoch_owner_session_id = args->config.session_id;
    (void)pthread_mutex_unlock(&g_kvaf_lock);
    return KVAF_EPOCH_CLAIM_ACQUIRED;
}

static bool kvaf_epoch_owner_snapshot_local_state(
    const struct kvaf_urma_aicpu_epoch_owner_args_v1 *args,
    bool *local_state_present, bool *provider_session_matches)
{
    bool authorized;

    if (local_state_present == NULL || provider_session_matches == NULL ||
        pthread_mutex_lock(&g_kvaf_lock) != 0) {
        return false;
    }
    authorized = kvaf_epoch_owner_is_authorized_unlocked(
        args->config.session_id) &&
        g_kvaf_epoch_claim_epoch == args->epoch;
    if (authorized) {
        *local_state_present = kvaf_provider_has_local_state_unlocked();
        *provider_session_matches =
            g_kvaf_session_id == args->config.session_id;
    }
    (void)pthread_mutex_unlock(&g_kvaf_lock);
    return authorized;
}

static bool kvaf_epoch_owner_release_claim(
    const struct kvaf_urma_aicpu_epoch_owner_args_v1 *args)
{
    bool released = false;

    if (pthread_mutex_lock(&g_kvaf_lock) != 0) {
        return false;
    }
    if (kvaf_epoch_owner_is_authorized_unlocked(
            args->config.session_id) &&
        g_kvaf_epoch_claim_epoch == args->epoch &&
        !kvaf_provider_has_local_state_unlocked()) {
        g_kvaf_epoch_claim_active = false;
        g_kvaf_epoch_claim_epoch = 0;
        g_kvaf_epoch_claim_session_id = 0;
        g_kvaf_epoch_owner_authorized = false;
        g_kvaf_epoch_owner_epoch = 0;
        g_kvaf_epoch_owner_session_id = 0;
        released = true;
    }
    (void)pthread_mutex_unlock(&g_kvaf_lock);
    return released;
}

int kvaf_urma_provider_open_v2(
    const struct kvaf_urma_provider_config_v2 *config,
    struct kvaf_urma_provider_result_v2 *result)
{
    int ret;
    int ready_status;
    const bool use_urma_management = config != NULL &&
        (config->flags & KVAF_URMA_PROVIDER_URMA_MANAGEMENT) != 0;

    if (result == NULL) {
        return -1;
    }
    kvaf_init_result(result, KVAF_URMA_PROVIDER_INVALID_ARGUMENT, -1);
    /* A provider instance can serve multiple sequential sessions. Clear the
     * thread-local diagnostic before any early-return/fail path so one Open
     * can never report a leaf failure from an older session. */
    kvaf_perftest_management_diagnostic_reset();
    if (!kvaf_config_is_valid(config)) {
        return -1;
    }
    if (pthread_mutex_lock(&g_kvaf_lock) != 0) {
        kvaf_init_result(result, KVAF_URMA_PROVIDER_ALREADY_OPEN, -1);
        return -1;
    }
    if (g_kvaf_epoch_claim_active &&
        !kvaf_epoch_owner_is_authorized_unlocked(config->session_id)) {
        /* A live epoch owns the complete process-local provider lifetime.
         * Legacy/public callers must not mutate that instance between READY
         * and checked teardown, even when they know its session nonce. */
        kvaf_init_result(result, KVAF_URMA_PROVIDER_ALREADY_OPEN, -1);
        (void)pthread_mutex_unlock(&g_kvaf_lock);
        return -1;
    }
    if (g_kvaf_config_initialized || g_kvaf_connection_open ||
        g_kvaf_context_open || g_kvaf_runtime_acquired ||
        g_kvaf_cleanup_poisoned) {
        kvaf_init_result(result, KVAF_URMA_PROVIDER_ALREADY_OPEN, -1);
        (void)pthread_mutex_unlock(&g_kvaf_lock);
        return -1;
    }
    /* Validate the public layout before urma_init discovers any device. */
    if (kvaf_udma_source_public_abi_matches(
            kvaf_urma_public_abi_signature()) != 1) {
        kvaf_init_result(result,
            KVAF_URMA_PROVIDER_RESOURCE_INCOMPATIBLE, -1);
        (void)pthread_mutex_unlock(&g_kvaf_lock);
        return -1;
    }
    g_kvaf_session_id = config->session_id;

    (void)memset(&g_kvaf_config, 0, sizeof(g_kvaf_config));
    (void)memset(&g_kvaf_context, 0, sizeof(g_kvaf_context));
    (void)memset(&g_kvaf_queue_info, 0, sizeof(g_kvaf_queue_info));
    kvaf_perftest_teardown_state_init(&g_kvaf_teardown_state);
    ret = kvaf_parse_config(config);
    g_kvaf_config_initialized = true;
    if (ret != 0) {
        kvaf_init_result(result, KVAF_URMA_PROVIDER_PARSE_FAILED, ret);
        goto fail;
    }
    // The source snapshot defaults to 2 MiB huge pages even when an external
    // device VA is supplied. The direct provider owns no allocation, so use
    // the explicit 4 KiB page-size contract and avoid a hidden 4 MiB MR.
    g_kvaf_config.use_huge_page = false;
    if (!kvaf_allocation_covers_parsed_config(config)) {
        kvaf_init_result(result,
            KVAF_URMA_PROVIDER_INVALID_ARGUMENT, -1);
        goto fail;
    }
    ret = check_local_cfg(&g_kvaf_config);
    if (ret != 0) {
        kvaf_init_result(result,
            KVAF_URMA_PROVIDER_LOCAL_CONFIG_FAILED, ret);
        goto fail;
    }

    /* UB management creates a context, queues and registered segments of its
     * own, so the source-linked runtime must already be alive.  The inlined
     * UMDK helper is compiled with its duplicate urma_init suppressed. */
    if (use_urma_management) {
        ret = kvaf_udma_source_runtime_init();
        g_kvaf_runtime_acquired = ret == 0;
        if (ret != 0) {
            kvaf_init_result(result,
                KVAF_URMA_PROVIDER_RESOURCE_CREATE_FAILED, ret);
            goto fail;
        }
    }

    errno = 0;
    ret = kvaf_perftest_set_control_timeout(
        config->control_timeout_seconds);
    /* Mark ownership before establish: UB publishes a partially-created
     * context on failure so fail/Close can retry exact public handles. TCP's
     * close helper is likewise idempotent after partial connect cleanup. */
    g_kvaf_connection_open = ret == 0;
    if (ret == 0 && kvaf_perftest_establish_connection(
            &g_kvaf_config,
            use_urma_management
                ? config->management_peer_address : NULL) != 0) {
        ret = kvaf_negative_errno_or(EIO);
    } else if (ret != 0) {
        ret = kvaf_negative_errno_or(EIO);
    }
    if (ret != 0) {
        kvaf_init_result(result, KVAF_URMA_PROVIDER_CONNECT_FAILED,
            ret);
        goto fail;
    }
    if (!use_urma_management) {
        ret = kvaf_set_control_socket_deadlines(config);
        if (ret != 0) {
            kvaf_init_result(result,
                KVAF_URMA_PROVIDER_CONNECT_FAILED, ret);
            goto fail;
        }
    }
    ret = check_remote_cfg(&g_kvaf_config);
    if (ret != 0) {
        kvaf_init_result(result,
            KVAF_URMA_PROVIDER_REMOTE_CONFIG_FAILED, ret);
        goto fail;
    }

    if (!use_urma_management) {
        ret = kvaf_udma_source_runtime_init();
        g_kvaf_runtime_acquired = ret == 0;
    } else {
        ret = 0;
    }
    if (kvaf_perftest_confirm_runtime_ready(
            &g_kvaf_config, ret) != KVAF_PERFTEST_CHECKED_OK) {
        kvaf_init_result(result,
            KVAF_URMA_PROVIDER_RESOURCE_CREATE_FAILED, ret);
        goto fail;
    }

    g_kvaf_config.v_address = config->local_address;
    g_kvaf_config.enable_va = true;
    ret = kvaf_perftest_create_duplex_ctx_checked(
        &g_kvaf_context, &g_kvaf_config);
    /*
     * Even a synchronized partial creation may own a context, queue or MR.
     * Route it through the checked phase machine instead of hiding it behind
     * a false context_open value.
     */
    g_kvaf_context_open = true;
    if (ret != KVAF_PERFTEST_CHECKED_OK) {
        kvaf_init_result(result,
            KVAF_URMA_PROVIDER_RESOURCE_CREATE_FAILED, ret);
        if (ret != KVAF_PERFTEST_CHECKED_CREATE_CLEANUP_REQUIRED) {
            g_kvaf_cleanup_poisoned = true;
            result->flags |= KVAF_URMA_PROVIDER_CLEANUP_UNCERTAIN;
        }
        goto fail;
    }
    ret = 0;
    if (!kvaf_context_is_complete(config)) {
        kvaf_init_result(result,
            KVAF_URMA_PROVIDER_RESOURCE_CREATE_FAILED, -1);
        ret = -1;
    } else if (kvaf_fill_result(result) != 0) {
        ret = -1;
    }
    ready_status = kvaf_perftest_confirm_provider_ready(
        &g_kvaf_context, &g_kvaf_config, ret);
    if (ready_status != KVAF_PERFTEST_CHECKED_OK) {
        if (ret == 0) {
            kvaf_init_result(result,
                KVAF_URMA_PROVIDER_RESOURCE_CREATE_FAILED,
                ready_status);
        }
        if (ready_status !=
            KVAF_PERFTEST_CHECKED_CREATE_CLEANUP_REQUIRED) {
            g_kvaf_cleanup_poisoned = true;
            result->flags |= KVAF_URMA_PROVIDER_CLEANUP_UNCERTAIN;
        }
        goto fail;
    }
    (void)pthread_mutex_unlock(&g_kvaf_lock);
    return 0;

fail:
    /* Snapshot the first management leaf failure before checked teardown can
     * execute secondary management operations on the same thread. This also
     * covers failures discovered after the initial connection handshake. */
    kvaf_attach_management_diagnostic(result);
    ret = kvaf_close_unlocked(config->session_id);
    if (ret == KVAF_URMA_PROVIDER_CLOSE_RETRYABLE) {
        /*
         * Preserve the nonce in the error result.  The caller must keep the
         * provider active and issue Close again for this exact session; this
         * is not protocol ambiguity and must not be promoted to quarantine.
         */
        result->flags |= KVAF_URMA_PROVIDER_CLEANUP_RETRYABLE;
        result->session_id = config->session_id;
    } else if (ret != KVAF_URMA_PROVIDER_CLOSE_OK) {
        result->flags |= KVAF_URMA_PROVIDER_CLEANUP_UNCERTAIN;
        result->session_id = config->session_id;
    }
    (void)pthread_mutex_unlock(&g_kvaf_lock);
    return -1;
}

static int kvaf_close_unlocked(uint64_t session_id)
{
    /*
     * AICPU KFC may dispatch two kernel launches to different workers or SO
     * instances.  An empty instance must never acknowledge another
     * instance's Close: doing so would let the caller release DDR/HBM while
     * the original provider still owns DMA resources.  Close succeeds only
     * for the exact non-zero session installed by Open in this instance, or
     * an idempotent retry of the last session closed by this same instance.
     */
    if (session_id == 0) {
        return KVAF_URMA_PROVIDER_CLOSE_UNSAFE;
    }
    /*
     * A Close result can be lost after this instance has already completed
     * teardown.  A retry for that exact session is safe and must be
     * idempotent.  Never generalize this to an arbitrary empty instance: an
     * AICPU KFC retry may otherwise land on a worker that never owned Open.
     */
    if (g_kvaf_session_id == 0) {
        return g_kvaf_last_closed_session_id == session_id
            ? KVAF_URMA_PROVIDER_CLOSE_OK
            : KVAF_URMA_PROVIDER_CLOSE_UNSAFE;
    }
    if (g_kvaf_session_id != session_id) {
        return KVAF_URMA_PROVIDER_CLOSE_UNSAFE;
    }
    if (g_kvaf_cleanup_poisoned) {
        return KVAF_URMA_PROVIDER_CLOSE_UNSAFE;
    }
    if (g_kvaf_connection_open &&
        kvaf_perftest_rearm_control_timeout() != 0) {
        /* No resource was touched. Retrying Close can safely acquire a fresh
         * deadline and resume the same session. */
        return KVAF_URMA_PROVIDER_CLOSE_RETRYABLE;
    }
    if (g_kvaf_context_open) {
        const int teardown_status =
            kvaf_perftest_destroy_duplex_ctx_checked(
                &g_kvaf_context, &g_kvaf_config,
                &g_kvaf_teardown_state);
        if (teardown_status ==
            KVAF_PERFTEST_CHECKED_DESTROY_RETRYABLE) {
            if (kvaf_source_cleanup_state_is_unknown()) {
                /*
                 * A public destroy API collapses lower UMMU errno values to
                 * URMA_FAIL. Consult the source-owned diagnostic before
                 * advertising a retry: a command whose completion is
                 * ambiguous must be quarantined, not retried forever.
                 */
                g_kvaf_cleanup_poisoned = true;
                return KVAF_URMA_PROVIDER_CLOSE_UNSAFE;
            }
            /*
             * The checked source-matched provider retained the exact failed
             * handle and phase. Keep this session installed so a second
             * Close for the same nonce resumes rather than re-running
             * successful teardown actions.
             */
            return KVAF_URMA_PROVIDER_CLOSE_RETRYABLE;
        }
        if (teardown_status != KVAF_PERFTEST_CHECKED_OK) {
            /* Only protocol/ownership uncertainty permanently poisons it. */
            g_kvaf_cleanup_poisoned = true;
            return KVAF_URMA_PROVIDER_CLOSE_UNSAFE;
        }
        g_kvaf_context_open = false;
    }
    /* The UB management context is a real URMA resource owner.  Destroy it
     * after the data teardown protocol has used it, but before runtime uninit
     * can invalidate its provider ops. A failed public destroy keeps its
     * pointer in g_ub_ctx and is resumed by the next Close. */
    if (g_kvaf_connection_open) {
        const int management_status =
            kvaf_perftest_close_connection(&g_kvaf_config);
        if (management_status !=
            KVAF_PERFTEST_MANAGEMENT_CLOSE_OK) {
            if (management_status ==
                    KVAF_PERFTEST_MANAGEMENT_CLOSE_UNSAFE ||
                kvaf_source_cleanup_state_is_unknown()) {
                g_kvaf_cleanup_poisoned = true;
                return KVAF_URMA_PROVIDER_CLOSE_UNSAFE;
            }
            return KVAF_URMA_PROVIDER_CLOSE_RETRYABLE;
        }
        g_kvaf_connection_open = false;
    }
    if (g_kvaf_runtime_acquired) {
        if (!kvaf_udma_source_has_no_live_resources() ||
            kvaf_udma_source_runtime_uninit() != 0) {
            /*
             * urma_uninit mutates process-global state even when it reports
             * failure. It is not retryable; retain this session and refuse
             * any future Open in this provider instance.
             */
            g_kvaf_cleanup_poisoned = true;
            return KVAF_URMA_PROVIDER_CLOSE_UNSAFE;
        }
        g_kvaf_runtime_acquired = false;
    }
    if (g_kvaf_config_initialized) {
        destroy_cfg(&g_kvaf_config);
        g_kvaf_config_initialized = false;
    }
    (void)memset(&g_kvaf_config, 0, sizeof(g_kvaf_config));
    (void)memset(&g_kvaf_context, 0, sizeof(g_kvaf_context));
    (void)memset(&g_kvaf_queue_info, 0, sizeof(g_kvaf_queue_info));
    (void)memset(
        &g_kvaf_teardown_state, 0, sizeof(g_kvaf_teardown_state));
    g_kvaf_last_closed_session_id = g_kvaf_session_id;
    g_kvaf_session_id = 0;
    return KVAF_URMA_PROVIDER_CLOSE_OK;
}

int kvaf_urma_provider_close_v2(uint64_t session_id)
{
    int ret;

    if (pthread_mutex_lock(&g_kvaf_lock) != 0) {
        return -1;
    }
    if (g_kvaf_epoch_claim_active &&
        !kvaf_epoch_owner_is_authorized_unlocked(session_id)) {
        (void)pthread_mutex_unlock(&g_kvaf_lock);
        return KVAF_URMA_PROVIDER_CLOSE_UNSAFE;
    }
    ret = kvaf_close_unlocked(session_id);
    (void)pthread_mutex_unlock(&g_kvaf_lock);
    return ret;
}

int aicpu_kvaf_urma_open_v2(void *opaque_args)
{
    struct kvaf_urma_aicpu_open_args_v2 *args =
        (struct kvaf_urma_aicpu_open_args_v2 *)opaque_args;
    struct kvaf_urma_provider_result_v2 *result;

    if (args == NULL || args->result_address == 0) {
        return -1;
    }
    result = (struct kvaf_urma_provider_result_v2 *)(uintptr_t)
        args->result_address;
    (void)kvaf_urma_provider_open_v2(&args->config, result);
    /*
     * KFC treats a non-zero AICPU entry-point return as a task exception and
     * aclrtSynchronizeStream then reports 507018.  Provider failures are
     * ordinary transport results, not malformed KFC tasks: the stable result
     * block already carries status, provider_status and cleanup certainty.
     * Return task success once that block is writable so the Host can inspect
     * the real provider result and decide whether cleanup is safe.
     */
    return 0;
}

int aicpu_kvaf_urma_close_v2(void *opaque_args)
{
    struct kvaf_urma_aicpu_close_args_v2 *args =
        (struct kvaf_urma_aicpu_close_args_v2 *)opaque_args;
    int32_t *result;
    int ret;

    if (args == NULL || args->result_address == 0) {
        return -1;
    }
    result = (int32_t *)(uintptr_t)args->result_address;
    *result = -1;
    if (pthread_mutex_lock(&g_kvaf_lock) != 0) {
        return 0;
    }
    if (g_kvaf_epoch_claim_active &&
        !kvaf_epoch_owner_is_authorized_unlocked(args->session_id)) {
        (void)pthread_mutex_unlock(&g_kvaf_lock);
        return 0;
    }
    if (args->session_id == 0 ||
        (g_kvaf_session_id == 0 &&
         g_kvaf_last_closed_session_id != args->session_id) ||
        (g_kvaf_session_id != 0 &&
         g_kvaf_session_id != args->session_id)) {
        (void)pthread_mutex_unlock(&g_kvaf_lock);
        return 0;
    }
    if (g_kvaf_context_open &&
        kvaf_quiesce_aiv_queue(
            args->final_sq_head, args->final_cq_tail) != 0) {
        (void)pthread_mutex_unlock(&g_kvaf_lock);
        return 0;
    }
    ret = kvaf_close_unlocked(args->session_id);
    (void)pthread_mutex_unlock(&g_kvaf_lock);
    *result = ret;
    /* See aicpu_kvaf_urma_open_v2: result reports provider failure. */
    return 0;
}

static bool kvaf_owner_probe_args_are_valid(
    const struct kvaf_urma_aicpu_owner_probe_args_v1 *args)
{
    const uint64_t alignment_mask =
        KVAF_URMA_OWNER_PROBE_CACHE_LINE_BYTES - UINT64_C(1);

    return args != NULL &&
        args->magic == KVAF_URMA_OWNER_PROBE_MAGIC &&
        args->abi_version == KVAF_URMA_OWNER_PROBE_ABI_VERSION &&
        args->struct_size == sizeof(*args) &&
        args->session_id != 0 &&
        args->command_address != 0 &&
        args->status_address != 0 &&
        args->command_address != args->status_address &&
        (args->command_address & alignment_mask) == 0 &&
        (args->status_address & alignment_mask) == 0 &&
        args->reserved0 == 0 &&
        args->reserved[0] == 0 &&
        args->reserved[1] == 0;
}

static int kvaf_owner_probe_monotonic_nanoseconds(
    clockid_t clock_id, uint64_t *nanoseconds)
{
    struct timespec now;

    if (nanoseconds == NULL ||
        clock_gettime(clock_id, &now) != 0 ||
        now.tv_sec < 0 || now.tv_nsec < 0 ||
        now.tv_nsec >= 1000000000L ||
        (uint64_t)now.tv_sec > UINT64_MAX / UINT64_C(1000000000)) {
        return -1;
    }
    *nanoseconds = (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
    return 0;
}

static void kvaf_owner_probe_publish(
    struct kvaf_urma_owner_probe_status_v1 *status,
    enum kvaf_urma_owner_probe_state state,
    enum kvaf_urma_owner_probe_result result,
    uint64_t sequence,
    uint64_t elapsed_nanoseconds)
{
    status->sequence = sequence;
    status->result_code = (int32_t)result;
    status->elapsed_nanoseconds = elapsed_nanoseconds;
    /* Publish every status payload field before the externally polled state. */
    __atomic_store_n(&status->state, (uint32_t)state, __ATOMIC_RELEASE);
}

static void kvaf_owner_probe_sleep(uint32_t microseconds)
{
    const struct timespec delay = {
        .tv_sec = (time_t)(microseconds / 1000000U),
        .tv_nsec = (long)(microseconds % 1000000U) * 1000L,
    };

    /* A signal merely shortens one poll interval; the monotonic deadline is
     * checked again before any subsequent sleep. */
    (void)nanosleep(&delay, NULL);
}

int aicpu_kvaf_urma_owner_probe_v1(void *opaque_args)
{
    struct kvaf_urma_aicpu_owner_probe_args_v1 *args =
        (struct kvaf_urma_aicpu_owner_probe_args_v1 *)opaque_args;
    struct kvaf_urma_owner_probe_command_v1 *command;
    struct kvaf_urma_owner_probe_status_v1 *status;
    uint64_t start_nanoseconds;
    uint64_t now_nanoseconds;
    uint64_t timeout_nanoseconds;
    uint32_t poll_microseconds;

    if (args == NULL ||
        args->timeout_milliseconds == 0 ||
        args->timeout_milliseconds >
            KVAF_URMA_OWNER_PROBE_MAX_TIMEOUT_MILLISECONDS ||
        args->poll_interval_microseconds >
            KVAF_URMA_OWNER_PROBE_MAX_POLL_MICROSECONDS ||
        !kvaf_owner_probe_args_are_valid(args)) {
        return -1;
    }
    command = (struct kvaf_urma_owner_probe_command_v1 *)(uintptr_t)
        args->command_address;
    status = (struct kvaf_urma_owner_probe_status_v1 *)(uintptr_t)
        args->status_address;

    /* Status has one producer.  Reset state first so a reused block cannot
     * expose an earlier READY while this invocation initializes its payload. */
    __atomic_store_n(&status->state,
        (uint32_t)KVAF_URMA_OWNER_PROBE_STATE_EMPTY, __ATOMIC_RELEASE);
    status->magic = KVAF_URMA_OWNER_PROBE_MAGIC;
    status->abi_version = KVAF_URMA_OWNER_PROBE_ABI_VERSION;
    status->struct_size = (uint32_t)sizeof(*status);
    status->session_id = args->session_id;
    status->sequence = 0;
    status->result_code = KVAF_URMA_OWNER_PROBE_RESULT_OK;
    status->reserved0 = 0;
    status->elapsed_nanoseconds = 0;
    status->reserved[0] = 0;
    status->reserved[1] = 0;

    if (kvaf_owner_probe_monotonic_nanoseconds(
            CLOCK_MONOTONIC, &start_nanoseconds) != 0) {
        kvaf_owner_probe_publish(status,
            KVAF_URMA_OWNER_PROBE_STATE_FAILED,
            KVAF_URMA_OWNER_PROBE_RESULT_CLOCK_FAILED, 0, 0);
        return 0;
    }
    timeout_nanoseconds =
        (uint64_t)args->timeout_milliseconds * UINT64_C(1000000);
    poll_microseconds = args->poll_interval_microseconds;
    if (poll_microseconds == 0) {
        poll_microseconds = 1000U;
    }

    kvaf_owner_probe_publish(status,
        KVAF_URMA_OWNER_PROBE_STATE_READY,
        KVAF_URMA_OWNER_PROBE_RESULT_OK, 0, 0);

    for (;;) {
        const uint32_t opcode =
            __atomic_load_n(&command->opcode, __ATOMIC_ACQUIRE);

        if (opcode == KVAF_URMA_OWNER_PROBE_COMMAND_RELEASE) {
            if (command->magic != KVAF_URMA_OWNER_PROBE_MAGIC ||
                command->abi_version !=
                    KVAF_URMA_OWNER_PROBE_ABI_VERSION ||
                command->struct_size != sizeof(*command) ||
                command->session_id != args->session_id ||
                command->sequence !=
                    KVAF_URMA_OWNER_PROBE_RELEASE_SEQUENCE) {
                kvaf_owner_probe_publish(status,
                    KVAF_URMA_OWNER_PROBE_STATE_FAILED,
                    KVAF_URMA_OWNER_PROBE_RESULT_INVALID_COMMAND,
                    command->sequence, 0);
                return 0;
            }
            if (kvaf_owner_probe_monotonic_nanoseconds(
                    CLOCK_MONOTONIC, &now_nanoseconds) != 0 ||
                now_nanoseconds < start_nanoseconds) {
                kvaf_owner_probe_publish(status,
                    KVAF_URMA_OWNER_PROBE_STATE_FAILED,
                    KVAF_URMA_OWNER_PROBE_RESULT_CLOCK_FAILED,
                    command->sequence, 0);
                return 0;
            }
            kvaf_owner_probe_publish(status,
                KVAF_URMA_OWNER_PROBE_STATE_RELEASED,
                KVAF_URMA_OWNER_PROBE_RESULT_OK,
                command->sequence,
                now_nanoseconds - start_nanoseconds);
            return 0;
        }
        if (opcode != KVAF_URMA_OWNER_PROBE_COMMAND_NONE) {
            kvaf_owner_probe_publish(status,
                KVAF_URMA_OWNER_PROBE_STATE_FAILED,
                KVAF_URMA_OWNER_PROBE_RESULT_INVALID_COMMAND, 0, 0);
            return 0;
        }

        if (kvaf_owner_probe_monotonic_nanoseconds(
                CLOCK_MONOTONIC, &now_nanoseconds) != 0 ||
            now_nanoseconds < start_nanoseconds) {
            kvaf_owner_probe_publish(status,
                KVAF_URMA_OWNER_PROBE_STATE_FAILED,
                KVAF_URMA_OWNER_PROBE_RESULT_CLOCK_FAILED, 0, 0);
            return 0;
        }
        if (now_nanoseconds - start_nanoseconds >= timeout_nanoseconds) {
            kvaf_owner_probe_publish(status,
                KVAF_URMA_OWNER_PROBE_STATE_TIMED_OUT,
                KVAF_URMA_OWNER_PROBE_RESULT_TIMED_OUT, 0,
                now_nanoseconds - start_nanoseconds);
            return 0;
        }
        kvaf_owner_probe_sleep(poll_microseconds);
    }
}

static bool kvaf_epoch_owner_range_is_valid(uint64_t address,
    uint64_t length)
{
    return address != 0 && length != 0 &&
        address <= UINT64_MAX - (length - 1U);
}

static bool kvaf_epoch_owner_ranges_overlap(uint64_t first_address,
    uint64_t first_length, uint64_t second_address,
    uint64_t second_length)
{
    return first_address <= second_address
        ? second_address - first_address < first_length
        : first_address - second_address < second_length;
}

static bool kvaf_epoch_owner_args_are_valid(
    const struct kvaf_urma_aicpu_epoch_owner_args_v1 *args)
{
    const uint64_t alignment_mask =
        KVAF_URMA_EPOCH_OWNER_CACHE_LINE_BYTES - UINT64_C(1);

    if (args == NULL ||
        args->magic != KVAF_URMA_EPOCH_OWNER_MAGIC ||
        args->abi_version != KVAF_URMA_EPOCH_OWNER_ABI_VERSION ||
        args->struct_size != sizeof(*args) ||
        args->flags != 0 ||
        args->epoch == 0 ||
        args->lease_timeout_milliseconds >
            KVAF_URMA_EPOCH_OWNER_MAX_LEASE_MILLISECONDS ||
        args->poll_interval_microseconds >
            KVAF_URMA_EPOCH_OWNER_MAX_POLL_MICROSECONDS ||
        args->close_retry_limit >
            KVAF_URMA_EPOCH_OWNER_MAX_CLOSE_RETRY_LIMIT ||
        args->close_retry_interval_microseconds >
            KVAF_URMA_EPOCH_OWNER_MAX_CLOSE_RETRY_MICROSECONDS ||
        !kvaf_config_is_valid(&args->config) ||
        (args->config.flags &
            KVAF_URMA_PROVIDER_EXPECT_DEVICE_PRIVATE_ABI) == 0 ||
        (args->config.flags &
            KVAF_URMA_PROVIDER_EXPECT_HOST_PRIVATE_ABI) != 0 ||
        !kvaf_epoch_owner_range_is_valid(args->command_address,
            sizeof(struct kvaf_urma_epoch_owner_command_v1)) ||
        !kvaf_epoch_owner_range_is_valid(args->status_address,
            sizeof(struct kvaf_urma_epoch_owner_status_v1)) ||
        !kvaf_epoch_owner_range_is_valid(args->open_result_address,
            sizeof(struct kvaf_urma_provider_result_v2)) ||
        (args->command_address & alignment_mask) != 0 ||
        (args->status_address & alignment_mask) != 0 ||
        (args->open_result_address & alignment_mask) != 0 ||
        kvaf_epoch_owner_ranges_overlap(args->command_address,
            sizeof(struct kvaf_urma_epoch_owner_command_v1),
            args->status_address,
            sizeof(struct kvaf_urma_epoch_owner_status_v1)) ||
        kvaf_epoch_owner_ranges_overlap(args->command_address,
            sizeof(struct kvaf_urma_epoch_owner_command_v1),
            args->open_result_address,
            sizeof(struct kvaf_urma_provider_result_v2)) ||
        kvaf_epoch_owner_ranges_overlap(args->status_address,
            sizeof(struct kvaf_urma_epoch_owner_status_v1),
            args->open_result_address,
            sizeof(struct kvaf_urma_provider_result_v2))) {
        return false;
    }
    return true;
}

static void kvaf_epoch_owner_initialize_status(
    struct kvaf_urma_epoch_owner_status_v1 *status,
    const struct kvaf_urma_aicpu_epoch_owner_args_v1 *args)
{
    /* Clear the externally polled state before replacing any payload from a
     * prior epoch.  This status line has exactly one producer. */
    __atomic_store_n(&status->state,
        (uint32_t)KVAF_URMA_EPOCH_OWNER_STATE_EMPTY,
        __ATOMIC_RELEASE);
    status->magic = KVAF_URMA_EPOCH_OWNER_MAGIC;
    status->abi_version = KVAF_URMA_EPOCH_OWNER_ABI_VERSION;
    status->struct_size = (uint32_t)sizeof(*status);
    status->epoch = args->epoch;
    status->session_id = args->config.session_id;
    status->sequence = 0;
    status->result_code = KVAF_URMA_EPOCH_OWNER_RESULT_OK;
    status->provider_status = 0;
    status->close_attempts = 0;
    status->flags = 0;
    status->elapsed_nanoseconds = 0;
}

static void kvaf_epoch_owner_publish(
    struct kvaf_urma_epoch_owner_status_v1 *status,
    enum kvaf_urma_epoch_owner_state state,
    enum kvaf_urma_epoch_owner_result result,
    int provider_status, uint64_t sequence,
    uint32_t close_attempts, uint32_t flags,
    uint64_t elapsed_nanoseconds)
{
    status->sequence = sequence;
    status->result_code = (int32_t)result;
    status->provider_status = provider_status;
    status->close_attempts = close_attempts;
    status->flags = flags;
    status->elapsed_nanoseconds = elapsed_nanoseconds;
    /* Publish the complete payload before AIV/Host observes the state. */
    __atomic_store_n(&status->state, (uint32_t)state, __ATOMIC_RELEASE);
}

static uint64_t kvaf_epoch_owner_elapsed(uint64_t start_nanoseconds)
{
    uint64_t now_nanoseconds;

    if (start_nanoseconds == 0 ||
        kvaf_owner_probe_monotonic_nanoseconds(
            CLOCK_MONOTONIC, &now_nanoseconds) != 0 ||
        now_nanoseconds < start_nanoseconds) {
        return 0;
    }
    return now_nanoseconds - start_nanoseconds;
}

static void __attribute__((noreturn)) kvaf_epoch_owner_hold_unsafe(
    struct kvaf_urma_epoch_owner_status_v1 *status,
    enum kvaf_urma_epoch_owner_result result,
    int provider_status, uint64_t sequence,
    uint32_t close_attempts, uint32_t poll_microseconds,
    uint64_t start_nanoseconds, bool open_result_valid)
{
    const uint32_t open_result_flag = open_result_valid
        ? KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID
        : 0U;

    kvaf_epoch_owner_publish(status,
        KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING, result,
        provider_status, sequence, close_attempts,
        open_result_flag |
            KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE |
            KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE,
        kvaf_epoch_owner_elapsed(start_nanoseconds));

    /* Returning would discard the only process/SO instance that still owns
     * the provider's pointers and checked teardown cursor.  An UNSAFE owner
     * therefore remains resident until device reset/process isolation. */
    for (;;) {
        kvaf_owner_probe_sleep(poll_microseconds);
    }
}

static void __attribute__((noreturn))
kvaf_epoch_owner_hold_claim_after_teardown(
    struct kvaf_urma_epoch_owner_status_v1 *status,
    uint64_t sequence, uint32_t close_attempts,
    uint32_t poll_microseconds, uint64_t start_nanoseconds)
{
    kvaf_epoch_owner_publish(status,
        KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING,
        KVAF_URMA_EPOCH_OWNER_RESULT_CLOSE_UNSAFE,
        KVAF_URMA_PROVIDER_CLOSE_OK, sequence, close_attempts,
        KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
            KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE |
            KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE,
        kvaf_epoch_owner_elapsed(start_nanoseconds));

    /* Transport teardown succeeded, but losing the still-installed global
     * claim would allow a new session to race an indeterminate owner record.
     * Keep this invocation resident and fail closed instead. */
    for (;;) {
        kvaf_owner_probe_sleep(poll_microseconds);
    }
}

static int kvaf_epoch_owner_close_until_complete(
    const struct kvaf_urma_aicpu_epoch_owner_args_v1 *args,
    struct kvaf_urma_epoch_owner_status_v1 *status,
    enum kvaf_urma_epoch_owner_state safe_terminal_state,
    enum kvaf_urma_epoch_owner_result safe_terminal_result,
    int safe_terminal_provider_status, uint64_t sequence,
    uint64_t start_nanoseconds)
{
    uint32_t close_attempts = 0;
    uint32_t retry_microseconds =
        args->close_retry_interval_microseconds;

    if (retry_microseconds == 0) {
        retry_microseconds = 1000U;
    }
    kvaf_epoch_owner_publish(status,
        KVAF_URMA_EPOCH_OWNER_STATE_CLOSING,
        KVAF_URMA_EPOCH_OWNER_RESULT_OK, 0, sequence, 0,
        KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
            KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE,
        kvaf_epoch_owner_elapsed(start_nanoseconds));

    for (;;) {
        int close_status;

        if (close_attempts != UINT32_MAX) {
            close_attempts++;
        }
        close_status = kvaf_urma_provider_close_v2(
            args->config.session_id);
        if (close_status == KVAF_URMA_PROVIDER_CLOSE_OK) {
            if (!kvaf_epoch_owner_release_claim(args)) {
                kvaf_epoch_owner_hold_claim_after_teardown(status,
                    sequence, close_attempts, retry_microseconds,
                    start_nanoseconds);
            }
            kvaf_epoch_owner_publish(status, safe_terminal_state,
                safe_terminal_result, safe_terminal_provider_status,
                sequence, close_attempts,
                KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                    KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE,
                kvaf_epoch_owner_elapsed(start_nanoseconds));
            return 0;
        }
        if (close_status != KVAF_URMA_PROVIDER_CLOSE_RETRYABLE) {
            kvaf_epoch_owner_hold_unsafe(status,
                KVAF_URMA_EPOCH_OWNER_RESULT_CLOSE_UNSAFE,
                close_status, sequence, close_attempts,
                retry_microseconds, start_nanoseconds, true);
        }

        if (args->close_retry_limit != 0 &&
            close_attempts >= args->close_retry_limit) {
            /* The threshold escalates observability only.  The provider
             * explicitly retained a safe retry cursor, so keep retrying it
             * inside this same invocation instead of returning. */
            kvaf_epoch_owner_publish(status,
                KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING,
                KVAF_URMA_EPOCH_OWNER_RESULT_CLOSE_RETRY_LIMIT,
                close_status, sequence, close_attempts,
                KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                    KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE |
                    KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE,
                kvaf_epoch_owner_elapsed(start_nanoseconds));
        } else {
            kvaf_epoch_owner_publish(status,
                KVAF_URMA_EPOCH_OWNER_STATE_CLOSE_RETRYING,
                KVAF_URMA_EPOCH_OWNER_RESULT_OK,
                close_status, sequence, close_attempts,
                KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                    KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE,
                kvaf_epoch_owner_elapsed(start_nanoseconds));
        }
        kvaf_owner_probe_sleep(retry_microseconds);
    }
}

static bool kvaf_epoch_owner_open_result_is_valid(
    const struct kvaf_urma_provider_result_v2 *result)
{
    return result->magic == KVAF_URMA_PROVIDER_MAGIC &&
        result->abi_version == KVAF_URMA_PROVIDER_ABI_VERSION &&
        result->struct_size == sizeof(*result);
}

static void kvaf_epoch_owner_make_safe_failed_open_result(
    struct kvaf_urma_provider_result_v2 *result,
    const struct kvaf_urma_aicpu_epoch_owner_args_v1 *args,
    enum kvaf_urma_provider_status status, int provider_status)
{
    kvaf_init_result(result, status, provider_status);
    /* OPEN_FAILED_SAFE is useful to Host only when its result is bound to
     * this launch's nonce.  In particular, a claim conflict must not expose
     * the active owner's session and an ordinary no-resource failure from
     * kvaf_urma_provider_open_v2 otherwise leaves session_id as zero. */
    result->session_id = args->config.session_id;
}

int aicpu_kvaf_urma_epoch_owner_v1(void *opaque_args)
{
    struct kvaf_urma_aicpu_epoch_owner_args_v1 *args =
        (struct kvaf_urma_aicpu_epoch_owner_args_v1 *)opaque_args;
    struct kvaf_urma_epoch_owner_command_v1 *command;
    struct kvaf_urma_epoch_owner_status_v1 *status;
    struct kvaf_urma_provider_result_v2 *open_result;
    uint64_t start_nanoseconds = 0;
    uint64_t lease_nanoseconds = 0;
    uint32_t poll_microseconds;
    bool lease_reported = false;
    bool clock_failure_reported = false;
    enum kvaf_epoch_claim_result claim_result;
    int open_status;

    if (!kvaf_epoch_owner_args_are_valid(args)) {
        return -1;
    }
    command = (struct kvaf_urma_epoch_owner_command_v1 *)(uintptr_t)
        args->command_address;
    status = (struct kvaf_urma_epoch_owner_status_v1 *)(uintptr_t)
        args->status_address;
    open_result = (struct kvaf_urma_provider_result_v2 *)(uintptr_t)
        args->open_result_address;
    poll_microseconds = args->poll_interval_microseconds;
    if (poll_microseconds == 0) {
        poll_microseconds = 1000U;
    }

    kvaf_epoch_owner_initialize_status(status, args);
    if (args->lease_timeout_milliseconds != 0) {
        if (kvaf_owner_probe_monotonic_nanoseconds(
                CLOCK_MONOTONIC, &start_nanoseconds) != 0) {
            kvaf_epoch_owner_make_safe_failed_open_result(open_result, args,
                KVAF_URMA_PROVIDER_RESOURCE_CREATE_FAILED, -1);
            kvaf_epoch_owner_publish(status,
                KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE,
                KVAF_URMA_EPOCH_OWNER_RESULT_CLOCK_FAILED, 0, 0, 0,
                KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                    KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE, 0);
            return 0;
        }
        lease_nanoseconds =
            (uint64_t)args->lease_timeout_milliseconds *
            UINT64_C(1000000);
    }

    claim_result = kvaf_epoch_owner_acquire_claim(args);
    if (claim_result == KVAF_EPOCH_CLAIM_CONFLICT) {
        /* This invocation created no resource and owns no provider pointer.
         * The already-live owner's invocation retains the SO and its claim. */
        kvaf_epoch_owner_make_safe_failed_open_result(open_result, args,
            KVAF_URMA_PROVIDER_ALREADY_OPEN, -1);
        kvaf_epoch_owner_publish(status,
            KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE,
            KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED,
            open_result->provider_status, 0, 0,
            KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE,
            kvaf_epoch_owner_elapsed(start_nanoseconds));
        return 0;
    }
    if (claim_result != KVAF_EPOCH_CLAIM_ACQUIRED) {
        kvaf_epoch_owner_publish(status,
            KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING,
            KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED, -1, 0, 0,
            KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE,
            kvaf_epoch_owner_elapsed(start_nanoseconds));
        /* Without the provider mutex we cannot prove whether another owner is
         * live. Returning could unload its process-local state. */
        for (;;) {
            kvaf_owner_probe_sleep(poll_microseconds);
        }
    }
    kvaf_epoch_owner_publish(status,
        KVAF_URMA_EPOCH_OWNER_STATE_OPENING,
        KVAF_URMA_EPOCH_OWNER_RESULT_OK, 0, 0, 0, 0, 0);

    open_status = kvaf_urma_provider_open_v2(
        &args->config, open_result);
    if (open_status != 0) {
        const uint32_t cleanup_mask =
            KVAF_URMA_PROVIDER_CLEANUP_RETRYABLE |
            KVAF_URMA_PROVIDER_CLEANUP_UNCERTAIN;
        const bool result_valid =
            kvaf_epoch_owner_open_result_is_valid(open_result);
        const uint32_t cleanup_flags = result_valid
            ? open_result->flags & cleanup_mask
            : 0U;
        const bool result_session_matches = result_valid &&
            open_result->session_id == args->config.session_id;
        const bool cleanup_retryable =
            (cleanup_flags &
             KVAF_URMA_PROVIDER_CLEANUP_RETRYABLE) != 0;
        const bool cleanup_uncertain =
            (cleanup_flags &
             KVAF_URMA_PROVIDER_CLEANUP_UNCERTAIN) != 0;
        bool local_state_present = false;
        bool provider_session_matches = false;
        const bool snapshot_valid =
            kvaf_epoch_owner_snapshot_local_state(args,
                &local_state_present, &provider_session_matches);
        const bool failure_result_contradicts_ownership =
            !result_valid ||
            open_result->status == KVAF_URMA_PROVIDER_OK ||
            (open_result->flags & ~cleanup_mask) != 0 ||
            (cleanup_flags != 0 && !result_session_matches);

        /* Only the provider's explicit RETRYABLE result grants this exact
         * invocation permission to resume checked teardown.  Never infer
         * ownership from a process-global session alone: it could belong to
         * an earlier KFC task in the same worker/SO instance. */
        if (!snapshot_valid) {
            kvaf_epoch_owner_hold_unsafe(status,
                KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED,
                result_valid ? open_result->provider_status : open_status,
                0, 0, poll_microseconds, start_nanoseconds,
                result_valid);
        }
        if (cleanup_retryable &&
            (!local_state_present || !provider_session_matches)) {
            /* RETRYABLE is meaningful only while this claimed instance still
             * owns the exact session and teardown cursor. */
            kvaf_epoch_owner_hold_unsafe(status,
                KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED,
                result_valid ? open_result->provider_status : open_status,
                0, 0, poll_microseconds, start_nanoseconds,
                result_valid);
        }
        if (!failure_result_contradicts_ownership &&
            cleanup_retryable && !cleanup_uncertain) {
            return kvaf_epoch_owner_close_until_complete(args, status,
                KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE,
                KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED,
                open_result->provider_status,
                0, start_nanoseconds);
        }
        if (failure_result_contradicts_ownership || cleanup_uncertain ||
            cleanup_retryable || local_state_present ||
            provider_session_matches) {
            kvaf_epoch_owner_hold_unsafe(status,
                KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED,
                result_valid ? open_result->provider_status : open_status,
                0, 0, poll_microseconds, start_nanoseconds,
                result_valid);
        }
        if (!kvaf_epoch_owner_release_claim(args)) {
            kvaf_epoch_owner_hold_claim_after_teardown(status,
                0, 0, poll_microseconds, start_nanoseconds);
        }
        /* The provider's ordinary no-resource failure is structurally valid
         * but does not carry a session nonce.  Bind it only after the locked
         * snapshot proved no local state and claim release succeeded. */
        open_result->session_id = args->config.session_id;
        kvaf_epoch_owner_publish(status,
            KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE,
            KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED,
            open_result->provider_status, 0, 0,
            KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE,
            kvaf_epoch_owner_elapsed(start_nanoseconds));
        return 0;
    }

    {
        const bool result_valid =
            kvaf_epoch_owner_open_result_is_valid(open_result);
        bool local_state_present = false;
        bool provider_session_matches = false;
        const bool snapshot_valid =
            kvaf_epoch_owner_snapshot_local_state(args,
                &local_state_present, &provider_session_matches);

        if (!snapshot_valid || !local_state_present ||
            !result_valid ||
            open_result->status != KVAF_URMA_PROVIDER_OK ||
            open_result->provider_status != 0 ||
            open_result->session_id != args->config.session_id ||
            !provider_session_matches) {
            kvaf_epoch_owner_hold_unsafe(status,
                KVAF_URMA_EPOCH_OWNER_RESULT_OPEN_FAILED,
                result_valid ? open_result->provider_status : open_status,
                0, 0, poll_microseconds, start_nanoseconds,
                result_valid);
        }
    }

    if (lease_nanoseconds != 0) {
        uint64_t now_nanoseconds;
        enum kvaf_urma_epoch_owner_result pre_ready_result =
            KVAF_URMA_EPOCH_OWNER_RESULT_OK;

        if (kvaf_owner_probe_monotonic_nanoseconds(
                CLOCK_MONOTONIC, &now_nanoseconds) != 0 ||
            now_nanoseconds < start_nanoseconds) {
            pre_ready_result =
                KVAF_URMA_EPOCH_OWNER_RESULT_CLOCK_FAILED;
        } else if (now_nanoseconds - start_nanoseconds >=
            lease_nanoseconds) {
            pre_ready_result =
                KVAF_URMA_EPOCH_OWNER_RESULT_LEASE_EXPIRED;
        }
        if (pre_ready_result != KVAF_URMA_EPOCH_OWNER_RESULT_OK) {
            /* READY has not been published, so AIV cannot have submitted a
             * WQE. Close is safe here even though no command exists. */
            return kvaf_epoch_owner_close_until_complete(args, status,
                KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE,
                pre_ready_result,
                0, 0, start_nanoseconds);
        }
    }

    kvaf_epoch_owner_publish(status,
        KVAF_URMA_EPOCH_OWNER_STATE_READY,
        KVAF_URMA_EPOCH_OWNER_RESULT_OK, 0, 0, 0,
        KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
            KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE,
        kvaf_epoch_owner_elapsed(start_nanoseconds));

    for (;;) {
        const uint32_t opcode =
            __atomic_load_n(&command->opcode, __ATOMIC_ACQUIRE);

        if (opcode ==
            KVAF_URMA_EPOCH_OWNER_COMMAND_SHUTDOWN_QUIESCED) {
            const bool identity_valid =
                command->magic == KVAF_URMA_EPOCH_OWNER_MAGIC &&
                command->abi_version ==
                    KVAF_URMA_EPOCH_OWNER_ABI_VERSION &&
                command->struct_size == sizeof(*command) &&
                command->epoch == args->epoch &&
                command->session_id == args->config.session_id &&
                command->sequence ==
                    KVAF_URMA_EPOCH_OWNER_SHUTDOWN_SEQUENCE &&
                command->reserved[0] == 0;
            const bool queue_quiesced =
                command->final_sq_head == command->final_sq_tail &&
                command->final_sq_tail == command->final_cq_tail &&
                command->final_cq_tail == command->final_wqe_count;

            if (!identity_valid) {
                kvaf_epoch_owner_publish(status,
                    KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING,
                    KVAF_URMA_EPOCH_OWNER_RESULT_INVALID_COMMAND,
                    0, command->sequence, 0,
                    KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                        KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE |
                        KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE,
                    kvaf_epoch_owner_elapsed(start_nanoseconds));
            } else if (!queue_quiesced ||
                kvaf_quiesce_aiv_queue(command->final_sq_head,
                    command->final_cq_tail) != 0) {
                kvaf_epoch_owner_publish(status,
                    KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING,
                    KVAF_URMA_EPOCH_OWNER_RESULT_QUEUE_NOT_QUIESCED,
                    0, command->sequence, 0,
                    KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                        KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE |
                        KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE,
                    kvaf_epoch_owner_elapsed(start_nanoseconds));
            } else {
                return kvaf_epoch_owner_close_until_complete(args,
                    status, KVAF_URMA_EPOCH_OWNER_STATE_CLOSED,
                    KVAF_URMA_EPOCH_OWNER_RESULT_OK,
                    KVAF_URMA_PROVIDER_CLOSE_OK, command->sequence,
                    start_nanoseconds);
            }
        } else if (opcode != KVAF_URMA_EPOCH_OWNER_COMMAND_NONE) {
            kvaf_epoch_owner_publish(status,
                KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING,
                KVAF_URMA_EPOCH_OWNER_RESULT_INVALID_COMMAND,
                0, command->sequence, 0,
                KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                    KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE |
                    KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE,
                kvaf_epoch_owner_elapsed(start_nanoseconds));
        }

        if (lease_nanoseconds != 0 && !lease_reported) {
            uint64_t now_nanoseconds;

            if (kvaf_owner_probe_monotonic_nanoseconds(
                    CLOCK_MONOTONIC, &now_nanoseconds) != 0 ||
                now_nanoseconds < start_nanoseconds) {
                clock_failure_reported = true;
                kvaf_epoch_owner_publish(status,
                    KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING,
                    KVAF_URMA_EPOCH_OWNER_RESULT_CLOCK_FAILED,
                    0, 0, 0,
                    KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                        KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE |
                        KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE,
                    0);
            } else if (now_nanoseconds - start_nanoseconds >=
                lease_nanoseconds) {
                lease_reported = true;
                kvaf_epoch_owner_publish(status,
                    KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING,
                    KVAF_URMA_EPOCH_OWNER_RESULT_LEASE_EXPIRED,
                    0, 0, 0,
                    KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID |
                        KVAF_URMA_EPOCH_OWNER_PROVIDER_ACTIVE |
                        KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE,
                    now_nanoseconds - start_nanoseconds);
            }
            if (clock_failure_reported) {
                /* Clock failure disables the diagnostic lease.  Continue
                 * holding the provider for cooperative shutdown. */
                lease_reported = true;
            }
        }
        kvaf_owner_probe_sleep(poll_microseconds);
    }
}
