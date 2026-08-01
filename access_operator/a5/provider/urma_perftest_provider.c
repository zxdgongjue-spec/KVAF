#include "kvaf_urma_provider_abi.h"

#include <stdbool.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "perftest_communication.h"
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

static bool kvaf_is_power_of_two(uint32_t value)
{
    return value != 0 && (value & (value - 1U)) == 0;
}

static bool kvaf_has_terminator(const char *text, size_t capacity)
{
    return text != NULL && memchr(text, '\0', capacity) != NULL;
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

static bool kvaf_config_is_valid(
    const struct kvaf_urma_provider_config_v2 *config)
{
    uint64_t minimum_bytes;
    uint32_t expected_private_abi;

    if (config == NULL ||
        config->magic != KVAF_URMA_PROVIDER_MAGIC ||
        config->abi_version != KVAF_URMA_PROVIDER_ABI_VERSION ||
        config->struct_size != sizeof(*config) ||
        config->role > KVAF_URMA_PROVIDER_CLIENT ||
        (config->flags & KVAF_URMA_PROVIDER_USE_CTP) == 0 ||
        (config->flags & KVAF_PROVIDER_PRIVATE_ABI_MASK) == 0 ||
        (config->flags & KVAF_PROVIDER_PRIVATE_ABI_MASK) ==
            KVAF_PROVIDER_PRIVATE_ABI_MASK ||
        (config->flags &
            ~(KVAF_URMA_PROVIDER_USE_CTP |
              KVAF_PROVIDER_PRIVATE_ABI_MASK)) != 0 ||
        config->address_kind != KVAF_URMA_PROVIDER_ADDRESS_DEVICE_VA ||
        config->session_id == 0 ||
        config->jetty_index != 0 ||
        config->control_port < 1024U ||
        config->control_port > 65535U ||
        config->control_timeout_seconds == 0 ||
        config->control_timeout_seconds > 3600U ||
        config->local_address == 0 ||
        config->segment_payload_bytes == 0 ||
        config->segment_payload_bytes > UINT32_MAX ||
        !kvaf_has_terminator(config->device_name,
            sizeof(config->device_name)) ||
        config->device_name[0] == '\0' ||
        !kvaf_has_terminator(config->local_eid,
            sizeof(config->local_eid)) ||
        !kvaf_has_terminator(config->peer_eid,
            sizeof(config->peer_eid))) {
        return false;
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
    return config->local_allocation_bytes >= minimum_bytes &&
        (config->role != KVAF_URMA_PROVIDER_CLIENT ||
         (config->local_eid[0] != '\0' &&
          config->peer_eid[0] != '\0'));
}

static int kvaf_set_control_socket_deadlines(
    const struct kvaf_urma_provider_config_v2 *config)
{
    uint32_t i;
    const struct timeval timeout = {
        .tv_sec = (time_t)config->control_timeout_seconds,
        .tv_usec = 0,
    };

    if (g_kvaf_config.comm.sock_fd == NULL) {
        return -1;
    }
    for (i = 0; i < g_kvaf_config.pair_num; ++i) {
        const int fd = g_kvaf_config.comm.sock_fd[i];
        if (fd < 0 ||
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                &timeout, sizeof(timeout)) != 0 ||
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO,
                &timeout, sizeof(timeout)) != 0) {
            return -1;
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
    char *argv[24];
    int argc = 0;

    if (snprintf(size_text, sizeof(size_text), "%llu",
        (unsigned long long)input->segment_payload_bytes) < 0 ||
        snprintf(eid_index_text, sizeof(eid_index_text), "%u",
        input->eid_index) < 0 ||
        snprintf(port_text, sizeof(port_text), "%u",
        input->control_port) < 0) {
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
    KVAF_ADD_ARG("--enable_ipv6");
    if (input->role == KVAF_URMA_PROVIDER_CLIENT) {
        KVAF_ADD_ARG("-S");
        KVAF_ADD_ARG(input->peer_eid);
        KVAF_ADD_ARG("--bind_ip");
        KVAF_ADD_ARG(input->local_eid);
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
     * urma_perftest's copied private ai_jettys layout for device addresses.
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
    struct kvaf_source_ummu_status status = {
        .structure_size = sizeof(status),
    };

    return kvaf_source_ummu_get_status(&status) != 0 ||
        status.last_state_unknown != 0U;
}

static bool kvaf_udma_source_has_no_live_resources(void)
{
    struct kvaf_source_ummu_status ummu_status = {
        .structure_size = sizeof(ummu_status),
    };

    /* Open can fail before exposing a context handle; drain that sidecar
     * before consulting any ordinary live-resource counter. */
    return kvaf_udma_source_retry_orphan_contexts() == 0 &&
        kvaf_udma_source_live_segment_count() == 0U &&
        kvaf_udma_source_live_token_count() == 0U &&
        kvaf_udma_source_live_jetty_count() == 0U &&
        kvaf_udma_source_live_jfc_count() == 0U &&
        kvaf_udma_source_live_context_count() == 0U &&
        kvaf_source_ummu_checked_contract_ready() == 1 &&
        kvaf_source_ummu_get_status(&ummu_status) == 0 &&
        ummu_status.last_state_unknown == 0U &&
        kvaf_source_ummu_live_tid_count() == 0U &&
        kvaf_source_ummu_pending_operation_count() == 0U &&
        kvaf_source_ummu_retained_mapping_count() == 0U;
}

int kvaf_urma_provider_open_v2(
    const struct kvaf_urma_provider_config_v2 *config,
    struct kvaf_urma_provider_result_v2 *result)
{
    int ret;
    int ready_status;

    if (result == NULL) {
        return -1;
    }
    kvaf_init_result(result, KVAF_URMA_PROVIDER_INVALID_ARGUMENT, -1);
    if (!kvaf_config_is_valid(config)) {
        return -1;
    }
    if (pthread_mutex_lock(&g_kvaf_lock) != 0) {
        kvaf_init_result(result, KVAF_URMA_PROVIDER_ALREADY_OPEN, -1);
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
    ret = kvaf_perftest_set_control_timeout(
        config->control_timeout_seconds);
    if (ret == 0) {
        ret = establish_connection(&g_kvaf_config);
    }
    if (ret != 0) {
        kvaf_init_result(result, KVAF_URMA_PROVIDER_CONNECT_FAILED, ret);
        goto fail;
    }
    g_kvaf_connection_open = true;
    ret = kvaf_set_control_socket_deadlines(config);
    if (ret != 0) {
        kvaf_init_result(result, KVAF_URMA_PROVIDER_CONNECT_FAILED, ret);
        goto fail;
    }
    ret = check_remote_cfg(&g_kvaf_config);
    if (ret != 0) {
        kvaf_init_result(result,
            KVAF_URMA_PROVIDER_REMOTE_CONFIG_FAILED, ret);
        goto fail;
    }

    ret = kvaf_udma_source_runtime_init();
    g_kvaf_runtime_acquired = ret == 0;
    if (kvaf_perftest_confirm_runtime_ready(
            &g_kvaf_config, ret) != KVAF_PERFTEST_CHECKED_OK) {
        kvaf_init_result(result,
            KVAF_URMA_PROVIDER_RESOURCE_CREATE_FAILED, ret);
        goto fail;
    }

    g_kvaf_config.dev_buf = (void *)(uintptr_t)config->local_address;
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
    if (g_kvaf_context_open) {
        const int teardown_status =
            kvaf_perftest_destroy_duplex_ctx_checked(
                &g_kvaf_context, &g_kvaf_config,
                &g_kvaf_teardown_state);
        if (teardown_status ==
            KVAF_PERFTEST_CHECKED_DESTROY_RETRYABLE) {
            if (kvaf_source_ummu_state_is_unknown()) {
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
        if (!kvaf_udma_source_has_no_live_resources()) {
            /*
             * The public context is already gone, so a non-zero private
             * counter contradicts the checked teardown contract. Never
             * acknowledge Close or release the session in this state.
             */
            g_kvaf_cleanup_poisoned = true;
            return KVAF_URMA_PROVIDER_CLOSE_UNSAFE;
        }
        g_kvaf_context_open = false;
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
    if (g_kvaf_connection_open) {
        close_connection(&g_kvaf_config);
        g_kvaf_connection_open = false;
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
