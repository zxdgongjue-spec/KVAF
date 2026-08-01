#ifndef KVAF_SOURCE_MATCHED_UDMA_H
#define KVAF_SOURCE_MATCHED_UDMA_H

#include <stddef.h>
#include <stdint.h>

#include "kvaf_urma_provider_abi.h"
#include "urma_types.h"

/* Runtime evidence written into provider result reserved fields. */
#define KVAF_UDMA_TEARDOWN_RUNTIME_CAPABILITY \
  KVAF_UDMA_TEARDOWN_SOURCE_OWNED_CHECKED_RETRYABLE_V1

#ifdef __cplusplus
extern "C" {
#endif

static inline uint64_t kvaf_urma_public_abi_signature(void)
{
  const uint64_t fields[] = {
    sizeof(urma_device_t),
    offsetof(urma_device_t, ops),
    offsetof(urma_device_t, sysfs_dev),
    sizeof(urma_device_cap_t),
    offsetof(urma_device_cap_t, max_jfc_depth),
    offsetof(urma_device_cap_t, ceq_cnt),
    sizeof(urma_context_t),
    offsetof(urma_context_t, ops),
    offsetof(urma_context_t, dev_fd),
    offsetof(urma_context_t, eid),
    sizeof(urma_jfc_t),
    offsetof(urma_jfc_t, jfc_cfg),
    offsetof(urma_jfc_t, handle),
    offsetof(urma_jfc_t, event_mutex),
    sizeof(urma_jetty_t),
    offsetof(urma_jetty_t, remote_jetty),
    offsetof(urma_jetty_t, jetty_cfg),
    offsetof(urma_jetty_t, handle),
    offsetof(urma_jetty_t, event_mutex),
  };
  uint64_t hash = UINT64_C(1469598103934665603);
  size_t i;

  for (i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
    hash ^= fields[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

/*
 * These functions are private to libkvaf_urma_provider_{host,device}.so.
 * The source-built UDMA provider registers into KVAF's private liburma core;
 * that core is compiled with dynamic provider discovery disabled. KVAF
 * refuses to export queue addresses unless the resulting context is owned by
 * this exact provider instance.
 */
int kvaf_udma_prepare_provider_ops(void);
int kvaf_udma_register_source_provider(void);
void kvaf_udma_unregister_source_provider(void);
int kvaf_udma_source_provider_is_registered(void);
int kvaf_udma_source_runtime_init(void);
int kvaf_udma_source_runtime_uninit(void);
int kvaf_udma_source_runtime_is_owned(void);
int kvaf_udma_source_provider_owns_device(const urma_device_t *device);
int kvaf_udma_source_provider_owns_context(
    const urma_context_t *context);
uint64_t kvaf_udma_source_public_abi_signature(void);
int kvaf_udma_source_public_abi_matches(uint64_t caller_signature);

/* Number of retryable private objects still retained by the provider. */
uint32_t kvaf_udma_source_live_segment_count(void);
uint32_t kvaf_udma_source_live_token_count(void);
uint32_t kvaf_udma_source_live_context_count(void);
uint32_t kvaf_udma_source_live_jetty_count(void);
uint32_t kvaf_udma_source_live_jfc_count(void);

/*
 * A create operation can return NULL after the kernel or UMMU has already
 * acquired a resource.  Such an object is not visible to the caller and must
 * therefore be retried by its owning context before context destruction.
 * The drain only touches objects which were never handed to the caller.
 */
uint32_t kvaf_udma_source_orphan_segment_count(void);
uint32_t kvaf_udma_source_orphan_token_count(void);
uint32_t kvaf_udma_source_orphan_jetty_count(void);
uint32_t kvaf_udma_source_orphan_context_count(void);
int kvaf_udma_source_retry_orphan_segments(urma_context_t *context);
int kvaf_udma_source_retry_orphan_tokens(urma_context_t *context);
int kvaf_udma_source_retry_orphan_jetties(urma_context_t *context);
int kvaf_udma_source_retry_orphan_jfcs(urma_context_t *context);
int kvaf_udma_source_retry_orphan_cleanup(urma_context_t *context);
int kvaf_udma_source_retry_orphan_contexts(void);

/* Installed into the private UDMA urma_ops table before registration. */
urma_jetty_t *kvaf_udma_create_jetty_checked(
    urma_context_t *context, urma_jetty_cfg_t *cfg);
urma_status_t kvaf_udma_delete_jetty_checked(urma_jetty_t *jetty);
urma_jfc_t *kvaf_udma_create_jfc_checked(
    urma_context_t *context, urma_jfc_cfg_t *cfg);
urma_status_t kvaf_udma_delete_jfc_checked(urma_jfc_t *jfc);

#ifdef __cplusplus
}
#endif

#endif
