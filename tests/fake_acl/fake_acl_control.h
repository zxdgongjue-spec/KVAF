#ifndef KVAF_TEST_FAKE_ACL_CONTROL_H
#define KVAF_TEST_FAKE_ACL_CONTROL_H

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

#ifdef __cplusplus
}
#endif

#endif
