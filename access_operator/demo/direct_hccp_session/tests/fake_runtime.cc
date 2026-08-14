#include "direct_hccp_abi.h"

namespace {

int g_open_net_service_calls = 0;

}  // namespace

extern "C" int rtOpenNetService(kvaf::direct_hccp::RtNetServiceOpenArgs *service) {
  if (service == nullptr || service->ext_param_list == nullptr ||
      service->ext_param_count != 1U) {
    return -1;
  }
  ++g_open_net_service_calls;
  return 0;
}

extern "C" int KvafFakeRuntimeOpenNetServiceCalls() {
  return g_open_net_service_calls;
}
