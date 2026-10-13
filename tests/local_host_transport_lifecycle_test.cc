#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "direct_hccp/direct_hccp_local_host.h"

namespace hccp = kvaf::direct_hccp;
namespace {
std::vector<std::string> events;
void* failing = nullptr;
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void* Handle(unsigned value) { return reinterpret_cast<void*>(static_cast<uintptr_t>(value)); }
int Unimport(void*, void* handle) { events.push_back("unimport"); return handle == failing ? 1 : 0; }
int Destroy(void* handle) { events.push_back("destroy"); return handle == failing ? 1 : 0; }
int CqDestroy(void*, void* handle) { events.push_back("cq"); return handle == failing ? 1 : 0; }
int Unregister(void*, void* handle) { events.push_back("memory"); return handle == failing ? 1 : 0; }
int TokenFree(void*, void* handle) { events.push_back("token"); return handle == failing ? 1 : 0; }
int ContextFree(void* handle) { events.push_back("context"); return handle == failing ? 1 : 0; }
void TestRetirement() {
  hccp::DirectHccpApi api{};
  api.qp_unimport = Unimport; api.qp_destroy = Destroy; api.cq_destroy = CqDestroy;
  api.lmem_unregister = Unregister; api.token_free = TokenFree; api.ctx_deinit = ContextFree;
  hccp::LocalHostEndpoint host{};
  host.context = Handle(10); host.token_handle = Handle(11); host.memory.handle = Handle(12);
  host.workers[0].device_target = Handle(13); host.workers[0].jetty.handle = Handle(14);
  host.workers[0].cq = Handle(15);
  hccp::DeviceAivWorkerResources device{};
  device.device_peer_target_jetty = Handle(20); device.device_user_ctl_jetty.handle = Handle(21);
  device.device_user_ctl_cq = Handle(22);
  failing = Handle(13);
  Require(!hccp::RetireLocalAndDeviceQueues(api, Handle(30), &device, 1, &host), "reject failed reciprocal import retirement");
  Require(events == std::vector<std::string>({"unimport", "unimport"}), "do not destroy queues after failed import retirement");
  Require(!device.device_peer_target_jetty && host.workers[0].device_target == failing, "retain failed handle, clear retired handle");
  Require(host.memory.handle == Handle(12), "DDR MR still owned");
  events.clear(); failing = nullptr;
  Require(hccp::RetireLocalAndDeviceQueues(api, Handle(30), &device, 1, &host), "retire remaining queues");
  Require(events == std::vector<std::string>({"unimport", "destroy", "destroy", "cq", "cq"}), "reciprocal imports before QPs before CQs");
  events.clear(); failing = host.memory.handle;
  Require(!hccp::ReleaseLocalHostEndpoint(api, &host), "retain failed host MR");
  Require(events == std::vector<std::string>({"memory"}) && host.token_handle && host.context, "do not free MR dependencies");
  failing = nullptr; events.clear();
  Require(hccp::ReleaseLocalHostEndpoint(api, &host), "release registered host endpoint");
  Require(events == std::vector<std::string>({"memory", "token", "context"}), "MR before token before context");
  Require(!host.memory.handle && !host.token_handle && !host.context, "clear retired resources");
  events.clear();
  Require(hccp::ReleaseLocalHostEndpoint(api, &host) && events.empty(), "idempotent retired endpoint");
}
}  // namespace
// The lifecycle test links the real local RA helper without the production
// PeerLink library. Channel setup is intentionally outside this host test.
namespace kvaf::direct_hccp {
bool ImportPeerJetty(const DirectHccpApi&, void*, const uint8_t*, uint32_t,
                     uint32_t, uint64_t, uint64_t, uint32_t, uint32_t,
                     const char*, QpImportInfoT*, void**) { return false; }
}
int main() {
  try { TestRetirement(); std::cout << "local Host DDR transport lifecycle passed (mock RA)\n"; }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
