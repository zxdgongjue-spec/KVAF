#include "kvaf_source_matched_udma.h"
#include "kvaf_source_matched_ummu.h"

#include <pthread.h>
#include <stdbool.h>
#include <string.h>

#include "udma_u_ops.h"

static bool g_kvaf_udma_provider_registered;
static pthread_mutex_t g_kvaf_udma_runtime_lock =
    PTHREAD_MUTEX_INITIALIZER;
static bool g_kvaf_udma_runtime_owned;
static bool g_kvaf_udma_runtime_uninit_failed;

int kvaf_udma_register_source_provider(void)
{
    int status;

    if (g_kvaf_udma_provider_registered ||
        kvaf_udma_prepare_provider_ops() != 0) {
        return -1;
    }
    status = urma_register_provider_ops(&g_udma_provider_ops);
    if (status != 0) {
        return -1;
    }
    g_kvaf_udma_provider_registered = true;
    return 0;
}

void kvaf_udma_unregister_source_provider(void)
{
    if (!g_kvaf_udma_provider_registered) {
        return;
    }
    if (kvaf_udma_source_live_context_count() != 0U ||
        kvaf_udma_source_live_segment_count() != 0U ||
        kvaf_udma_source_live_token_count() != 0U ||
        kvaf_udma_source_live_jetty_count() != 0U ||
        kvaf_udma_source_live_jfc_count() != 0U ||
        kvaf_source_ummu_live_tid_count() != 0U ||
        kvaf_source_ummu_pending_operation_count() != 0U ||
        kvaf_source_ummu_retained_mapping_count() != 0U) {
        /*
         * dlclose with live DMA objects would make their retry callbacks
         * unreachable. The owning process is deliberately retained by the
         * higher-level KVAF fail-closed path, so never unregister here.
         */
        return;
    }
    if (urma_unregister_provider_ops(&g_udma_provider_ops) == 0) {
        g_kvaf_udma_provider_registered = false;
    }
}

int kvaf_udma_source_provider_is_registered(void)
{
    return g_kvaf_udma_provider_registered ? 1 : 0;
}

int kvaf_udma_source_runtime_init(void)
{
    urma_init_attr_t init_attr;
    urma_status_t status;

    if (pthread_mutex_lock(&g_kvaf_udma_runtime_lock) != 0) {
        return -1;
    }
    if (!g_kvaf_udma_provider_registered ||
        g_kvaf_udma_runtime_owned ||
        g_kvaf_udma_runtime_uninit_failed ||
        kvaf_source_ummu_checked_contract_ready() != 1 ||
        kvaf_source_ummu_live_tid_count() != 0U ||
        kvaf_source_ummu_pending_operation_count() != 0U ||
        kvaf_source_ummu_retained_mapping_count() != 0U) {
        (void)pthread_mutex_unlock(&g_kvaf_udma_runtime_lock);
        return -1;
    }
    (void)memset(&init_attr, 0, sizeof(init_attr));
    status = urma_init(&init_attr);
    if (status != URMA_SUCCESS) {
        /* In particular, URMA_EEXIST means unknown process-global state. */
        (void)pthread_mutex_unlock(&g_kvaf_udma_runtime_lock);
        return -1;
    }
    g_kvaf_udma_runtime_owned = true;
    (void)pthread_mutex_unlock(&g_kvaf_udma_runtime_lock);
    return 0;
}

int kvaf_udma_source_runtime_uninit(void)
{
    urma_status_t status;

    if (kvaf_udma_source_retry_orphan_contexts() != 0) {
        return -1;
    }
    if (pthread_mutex_lock(&g_kvaf_udma_runtime_lock) != 0) {
        return -1;
    }
    if (!g_kvaf_udma_runtime_owned ||
        g_kvaf_udma_runtime_uninit_failed ||
        kvaf_udma_source_live_context_count() != 0U ||
        kvaf_udma_source_live_segment_count() != 0U ||
        kvaf_udma_source_live_token_count() != 0U ||
        kvaf_udma_source_live_jetty_count() != 0U ||
        kvaf_udma_source_live_jfc_count() != 0U ||
        kvaf_source_ummu_checked_contract_ready() != 1 ||
        kvaf_source_ummu_live_tid_count() != 0U ||
        kvaf_source_ummu_pending_operation_count() != 0U ||
        kvaf_source_ummu_retained_mapping_count() != 0U) {
        (void)pthread_mutex_unlock(&g_kvaf_udma_runtime_lock);
        return -1;
    }
    status = urma_uninit();
    g_kvaf_udma_runtime_owned = false;
    if (status != URMA_SUCCESS) {
        /* urma_uninit already mutates global state, so retry is not safe. */
        g_kvaf_udma_runtime_uninit_failed = true;
        (void)pthread_mutex_unlock(&g_kvaf_udma_runtime_lock);
        return -1;
    }
    (void)pthread_mutex_unlock(&g_kvaf_udma_runtime_lock);
    return 0;
}

int kvaf_udma_source_runtime_is_owned(void)
{
    int owned;

    if (pthread_mutex_lock(&g_kvaf_udma_runtime_lock) != 0) {
        return 0;
    }
    owned = g_kvaf_udma_runtime_owned &&
        !g_kvaf_udma_runtime_uninit_failed ? 1 : 0;
    (void)pthread_mutex_unlock(&g_kvaf_udma_runtime_lock);
    return owned;
}

int kvaf_udma_source_provider_owns_device(const urma_device_t *device)
{
    return device != NULL &&
        kvaf_udma_source_runtime_is_owned() != 0 &&
        device->ops == &g_udma_provider_ops;
}

uint64_t kvaf_udma_source_public_abi_signature(void)
{
    return kvaf_urma_public_abi_signature();
}

int kvaf_udma_source_public_abi_matches(uint64_t caller_signature)
{
    return caller_signature == kvaf_urma_public_abi_signature() ? 1 : 0;
}

static __attribute__((constructor)) void kvaf_udma_provider_init(void)
{
    (void)kvaf_udma_register_source_provider();
}

static __attribute__((destructor)) void kvaf_udma_provider_uninit(void)
{
    kvaf_udma_unregister_source_provider();
}
