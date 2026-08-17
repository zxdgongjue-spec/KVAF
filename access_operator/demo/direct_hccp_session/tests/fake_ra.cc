#include <cstdint>

#include "direct_hccp_abi.h"

namespace {

int g_host_init_calls = 0;
int g_device_init_calls = 0;
int g_host_deinit_calls = 0;
int g_device_deinit_calls = 0;
int g_device_init_result = 328002;

}  // namespace

extern "C" int RaInit(kvaf::direct_hccp::RaInitConfig *config) {
  if (config == nullptr) {
    return -1;
  }
  if (config->nic_position == kvaf::direct_hccp::kPeerOnline) {
    ++g_host_init_calls;
    return 0;
  }
  if (config->nic_position == kvaf::direct_hccp::kOfflineHdc) {
    ++g_device_init_calls;
    return g_device_init_result;
  }
  return -2;
}

extern "C" int RaDeinit(kvaf::direct_hccp::RaInitConfig *config) {
  if (config == nullptr) {
    return -1;
  }
  if (config->nic_position == kvaf::direct_hccp::kPeerOnline) {
    ++g_host_deinit_calls;
    return 0;
  }
  if (config->nic_position == kvaf::direct_hccp::kOfflineHdc) {
    ++g_device_deinit_calls;
    return 0;
  }
  return -2;
}

extern "C" void KvafFakeRaReset() {
  g_host_init_calls = 0;
  g_device_init_calls = 0;
  g_host_deinit_calls = 0;
  g_device_deinit_calls = 0;
  g_device_init_result = 328002;
}

extern "C" void KvafFakeRaSetDeviceInitResult(int result) {
  g_device_init_result = result;
}

extern "C" int KvafFakeRaHostInitCalls() { return g_host_init_calls; }
extern "C" int KvafFakeRaDeviceInitCalls() { return g_device_init_calls; }
extern "C" int KvafFakeRaHostDeinitCalls() { return g_host_deinit_calls; }
extern "C" int KvafFakeRaDeviceDeinitCalls() { return g_device_deinit_calls; }
