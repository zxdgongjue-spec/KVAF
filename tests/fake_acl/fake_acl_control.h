#ifndef KVAF_TEST_FAKE_ACL_CONTROL_H
#define KVAF_TEST_FAKE_ACL_CONTROL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void kvaf_fake_acl_set_scenario(const char* scenario);
int kvaf_fake_acl_sync_calls(void);
int kvaf_fake_acl_close_launches(void);
int kvaf_fake_acl_free_calls(void);
int kvaf_fake_acl_destroy_stream_calls(void);
unsigned long long kvaf_fake_acl_open_session(void);
unsigned long long kvaf_fake_acl_last_close_session(void);
uint32_t kvaf_fake_acl_last_kernel_type(void);
uint32_t kvaf_fake_acl_last_config_flags(void);
uint32_t kvaf_fake_acl_last_management_jetty_id(void);
const char* kvaf_fake_acl_last_management_local_address(void);
const char* kvaf_fake_acl_last_management_peer_address(void);
int kvaf_fake_acl_data_launches(void);
int kvaf_fake_acl_data_stream_matches_control(void);

#ifdef __cplusplus
}
#endif

#endif
