#ifndef KVAF_SOURCE_MATCHED_PERFTEST_H
#define KVAF_SOURCE_MATCHED_PERFTEST_H

#include "perftest_parameters.h"
#include "perftest_resources.h"

#ifdef __cplusplus
extern "C" {
#endif

enum kvaf_perftest_checked_status {
  KVAF_PERFTEST_CHECKED_OK = 0,
  // Creation failed before a usable context was returned. No generic unwind
  // is attempted after a local MR may have been exposed to the peer; callers
  // must retain the process and backing memory until reset.
  KVAF_PERFTEST_CHECKED_CREATE_UNSAFE = -2,
  // Teardown reached a public URMA operation that did not report success.
  KVAF_PERFTEST_CHECKED_DESTROY_UNSAFE = -3,
};

// Arms one monotonic deadline for the upstream connect/accept helper. It must
// be called immediately before establish_connection().
int kvaf_perftest_set_control_timeout(unsigned int timeout_seconds);

int kvaf_perftest_create_duplex_ctx_checked(
    perftest_context_t *ctx, perftest_config_t *cfg);

int kvaf_perftest_confirm_provider_ready(
    perftest_context_t *ctx, perftest_config_t *cfg, int local_status);

int kvaf_perftest_destroy_duplex_ctx_checked(
    perftest_context_t *ctx, perftest_config_t *cfg);

#ifdef __cplusplus
}
#endif

#endif
