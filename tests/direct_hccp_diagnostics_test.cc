// Portable tests for initialization-only diagnostics, using the real adapter
// and fake RA/peer responses. No CANN installation or device is required.
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "acl/acl.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "peer_link.h"

namespace h = kvaf::direct_hccp;
namespace {
enum class Reply { kSuccess, kSubmitError, kRequestError, kApiError, kTimeout };
Reply reply = Reply::kSuccess;
int device_queries = 0;
int jetty_queries = 0;
int tp_queries = 0;
bool query_failure = false;
bool query_throw = false;
std::uint32_t* pending_bitmap = nullptr;
h::TpAttr* pending_attr = nullptr;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::string Capture(const std::function<void()>& function) {
  std::fflush(stderr);
  FILE* file = std::tmpfile();
  Require(file != nullptr, "tmpfile");
  const int original = dup(STDERR_FILENO);
  Require(original >= 0 && dup2(fileno(file), STDERR_FILENO) >= 0, "capture stderr");
  try {
    function();
  } catch (...) {
    std::fflush(stderr);
    (void)dup2(original, STDERR_FILENO);
    close(original);
    std::fclose(file);
    throw;
  }
  std::fflush(stderr);
  (void)dup2(original, STDERR_FILENO);
  close(original);
  std::rewind(file);
  std::string result;
  char buffer[4096];
  while (const auto size = std::fread(buffer, 1, sizeof(buffer), file)) result.append(buffer, size);
  std::fclose(file);
  return result;
}

int DeviceQuery(void*, h::DevBaseAttr* attr) {
  ++device_queries;
  errno = EIO;
  if (query_throw) throw std::runtime_error("fake diagnostic exception");
  if (query_failure) return 31;
  attr->ub.chip_id = 3;
  attr->ub.die_id = 4;
  attr->ub.func_id = 5;
  attr->ub.priority_info[2] = {7, 2};
  return 0;
}

int JettyQuery(void**, h::JettyAttr* attr, std::uint32_t* count) {
  ++jetty_queries;
  if (query_failure) return 32;
  *count = 1;
  attr->mask = 0;  // Modify mask is zero on a successful UMDK query.
  attr->state = 1;
  attr->rx_threshold = 99;
  return 0;
}

int TpQuery(void*, std::uint64_t, std::uint32_t* bitmap, h::TpAttr* attr, void** request) {
  ++tp_queries;
  if (reply == Reply::kSubmitError) return 33;
  pending_bitmap = bitmap;
  pending_attr = attr;
  *request = reinterpret_cast<void*>(2);
  return 0;
}

int TpInfoQuery(void*, h::GetTpCfg*, h::TpInfo* info, std::uint32_t* count, void** request) {
  info->tp_handle = 100;
  *count = 1;
  *request = reinterpret_cast<void*>(1);
  return 0;
}

int AsyncResult(void* request, int* result) {
  *result = 0;
  if (request == reinterpret_cast<void*>(1)) return 0;
  if (reply == Reply::kTimeout) return 128301;
  if (reply == Reply::kApiError) return 35;
  if (reply == Reply::kRequestError) {
    *result = 34;
    return 0;
  }
  *pending_bitmap = 1U << 10U;
  pending_attr->sl = 7;
  pending_attr->retry_times_init = 5;  // Should not be printed: returned bit 0 is clear.
  return 0;
}

int ImportJetty(void*, h::QpImportInfoT* info, void** target) {
  Require(info->in.ub.exp_import_cfg.tp_handle == 100, "local TP changed");
  Require(info->in.ub.exp_import_cfg.peer_tp_handle == 200, "peer TP changed");
  info->out.ub.tpn = 300;
  *target = reinterpret_cast<void*>(3);
  return 0;
}

int CreateJetty(void*, h::QpCreateAttr*, h::QpCreateInfo* info, void** jetty) {
  h::JettyKeyIdentity id{};
  id.eid.raw[15] = 9;
  id.uasid = 10;
  id.id = 11;
  id.transport_mode = 1;
  info->key.size = sizeof(id);
  std::memcpy(info->key.value, &id, sizeof(id));
  info->ub.uasid = id.uasid;
  info->ub.id = id.id;
  info->ub.sq_buffer_va = 0x1000;
  info->ub.db_addr = 0x2000;
  info->ub.wqebb_size = 64;
  *jetty = reinterpret_cast<void*>(4);
  return 0;
}

h::DirectHccpApi Api() {
  h::DirectHccpApi api{};
  api.get_dev_base_attr = DeviceQuery;
  api.qp_query_batch = JettyQuery;
  api.get_tp_attr_async = TpQuery;
  api.get_tp_info_async = TpInfoQuery;
  api.get_async_req_result = AsyncResult;
  api.qp_create = CreateJetty;
  api.qp_import = ImportJetty;
  return api;
}

void Exercise(const h::DirectHccpApi& api) {
  std::array<std::uint8_t, h::kEidBytes> eid{};
  eid[15] = 9;
  void* context = reinterpret_cast<void*>(10);
  h::LogLinkContextDiagnostics(api, context, 0, eid, "test");
  h::QpCreateInfo info{};
  void* jetty = nullptr;
  Require(h::CreateDeviceAivJetty(api, context, reinterpret_cast<void*>(11),
                                reinterpret_cast<void*>(12), 1, 8192, "test", &info, &jetty),
          "diagnostics broke jetty creation");
  h::DirectMemory memory{};
  memory.token_value = 1;
  memory.registered_bytes = 4096;
  kvaf::peer::PeerLink link("fake");
  h::PeerChannel channel{};
  Require(h::CreatePeerChannel({&link, &api, context, 0, 4096, &memory, &eid, &eid, &info, "test"},
                               &channel), "diagnostics broke channel creation");
  Require(channel.local_import_info.out.ub.tpn == 300, "diagnostics changed imported TPN");
}
}  // namespace

aclError aclrtGetDevice(std::int32_t* device) { *device = 0; return ACL_SUCCESS; }
namespace kvaf::direct_hccp {
void ReleaseDirectHccpSession(DirectHccpSession*) {}
}
namespace kvaf::peer {
PeerLink::PeerLink(std::string address) : address_(std::move(address)) {}
Status PeerLink::LookupRegion(std::uint64_t, PeerRegion*) const { return Status::NotFound("fake"); }
Status PeerLink::CreateChannel(std::uint64_t, const std::uint8_t*, std::uint32_t,
    const std::uint8_t*, std::uint32_t, std::uint64_t, std::uint32_t,
    std::array<std::uint8_t, kQpKeyBytes>* key, std::uint32_t* bytes,
    std::uint64_t* tp, std::uint32_t* psn) {
  h::JettyKeyIdentity id{};
  id.eid.raw[15] = 8;
  id.uasid = 20;
  id.id = 21;
  id.transport_mode = 1;
  // The CPU service's packed identity is 28 bytes, unlike RA's padded 32.
  *bytes = 28;
  std::memcpy(key->data(), &id, *bytes);
  *tp = 200;
  *psn = 1;
  return Status::OK();
}
}

int main() {
  try {
    unsetenv("KVAF_LINK_DIAGNOSTICS");
    auto output = Capture([] { Exercise(Api()); });
    Require(device_queries == 0 && jetty_queries == 0 && tp_queries == 0, "disabled diagnostics queried RA");
    Require(output.find("KVAF_LINK_DIAG") == std::string::npos, "disabled diagnostics logged");
    setenv("KVAF_LINK_DIAGNOSTICS", "1", 1);
    output = Capture([] { Exercise(Api()); });
    Require(device_queries == 1 && jetty_queries == 1 && tp_queries == 1, "missing queries");
    Require(output.find("priority=2 sl=7 tp_type=0x2") != std::string::npos, "priority mapping");
    Require(output.find("returned_bitmap=0x400 sl=7") != std::string::npos, "TP readback bitmap");
    Require(output.find(" retry_times_init=") == std::string::npos, "invalid TP field printed");
    Require(output.find("modify_mask=0x0 state_valid=1 state=1 rx_threshold_valid=1 rx_threshold=99") !=
                std::string::npos, "Jetty modify mask wrongly invalidated successful query");
    Require(output.find("endpoint=peer source=exported_key status=ok eid=") != std::string::npos,
            "missing peer identity");
    auto missing = Api();
    missing.get_dev_base_attr = nullptr;
    missing.qp_query_batch = nullptr;
    missing.get_tp_attr_async = nullptr;
    output = Capture([&] { Exercise(missing); });
    Require(output.find("reason=missing_symbol") != std::string::npos, "missing API unreported");
    const int queries_before_import_readback = jetty_queries;
    output = Capture([] {
      h::LogLinkJettyDiagnostics(Api(), reinterpret_cast<void*>(1),
                                reinterpret_cast<void*>(4), "test", "after_peer_import");
    });
    Require(jetty_queries == queries_before_import_readback + 1, "post-import Jetty query missing");
    Require(output.find("phase=after_peer_import") != std::string::npos, "post-import phase missing");
    output = Capture([] {
      errno = EBUSY;
      h::LogLinkContextDiagnostics(Api(), reinterpret_cast<void*>(1), 0, {}, "errno_test");
      Require(errno == EBUSY, "diagnostic changed errno");
    });
    query_throw = true;
    output = Capture([] { Exercise(Api()); });
    Require(output.find("reason=cpp_exception") != std::string::npos, "exception unreported");
    query_throw = false;
    query_failure = true;
    output = Capture([] { Exercise(Api()); });
    Require(output.find("status=failed ret=31") != std::string::npos, "device failure unreported");
    Require(output.find("status=failed ret=32") != std::string::npos, "Jetty failure unreported");
    query_failure = false;
    for (const auto mode : {Reply::kSubmitError, Reply::kRequestError, Reply::kApiError}) {
      reply = mode;
      output = Capture([] { Exercise(Api()); });
      Require(output.find("event=tp_readback") != std::string::npos &&
              output.find("status=failed phase=") != std::string::npos, "TP failure unreported");
    }
    reply = Reply::kTimeout;
    output = Capture([] { Exercise(Api()); });
    Require(output.find("status=timeout timeout_ms=250") != std::string::npos, "TP timeout unreported");
    // Simulate the provider delivering a late reply. ASan must not see a
    // use-after-free; retained output storage intentionally lives to exit.
    *pending_bitmap = 1U << 10U;
    pending_attr->sl = 6;
    const int queries_after_timeout = tp_queries;
    output = Capture([] { Exercise(Api()); });
    Require(tp_queries == queries_after_timeout, "timeout did not stop further TP queries");
    Require(output.find("reason=previous_timeout") != std::string::npos, "timeout skip unreported");
    std::puts("direct HCCP diagnostics tests passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAILED: %s\n", error.what());
    return 1;
  }
}
