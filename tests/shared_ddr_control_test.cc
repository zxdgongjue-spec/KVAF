// Exercise the real CPU service + PeerLink over TCP with a fake UMDK provider.
// This verifies ownership and protocol behavior, not RMA hardware semantics.
#include <dlfcn.h>
void* FakeDlsym(void*, const char*);
#define dlsym FakeDlsym
#include "../runtime/src/cpu_runtime.cc"
#undef dlsym
#include "peer_link.h"

#include <atomic>
#include <iostream>
#include <thread>

#define CHECK(expr) do { if (!(expr)) { \
  std::cerr << "line " << __LINE__ << ": " #expr "\n"; std::abort(); \
} } while (false)

namespace {
std::atomic<int> segments{0}, tokens{0}, jetties{0}, imports{0}, jfcs{0}, jfrs{0};
uint32_t next_id = 10;
void* registered_memory = nullptr;
uint64_t registered_bytes = 0;
urma_device_t device{};
urma_device_t* devices[] = {&device};
urma_eid_info_t eid{};
std::atomic<bool> fail_import{false};
}

extern "C" {
urma_status_t urma_init(urma_init_attr_t*) {
  std::strcpy(device.name, "fake_ub");
  device.type = URMA_TRANSPORT_UB;
  eid.eid.raw[0] = 1;
  return URMA_SUCCESS;
}
urma_status_t urma_uninit() { return URMA_SUCCESS; }
urma_device_t** urma_get_device_list(int* count) { *count = 1; return devices; }
void urma_free_device_list(urma_device_t**) {}
urma_eid_info_t* urma_get_eid_list(urma_device_t*, uint32_t* count) { *count = 1; return &eid; }
void urma_free_eid_list(urma_eid_info_t*) {}
urma_device_t* urma_get_device_by_name(char*) { return &device; }
urma_context_t* urma_create_context(urma_device_t*, uint32_t) {
  auto* result = new urma_context_t{};
  result->eid = eid.eid;
  return result;
}
urma_status_t urma_delete_context(urma_context_t* context) { delete context; return URMA_SUCCESS; }
urma_token_id_t* urma_alloc_token_id(urma_context_t*) {
  ++tokens;
  auto* result = new urma_token_id_t{};
  result->token_id = ++next_id;
  return result;
}
urma_status_t urma_free_token_id(urma_token_id_t* token) {
  --tokens; delete token; return URMA_SUCCESS;
}
urma_target_seg_t* urma_register_seg(urma_context_t* context, urma_seg_cfg_t* config) {
  ++segments;
  registered_memory = reinterpret_cast<void*>(config->va);
  registered_bytes = config->len;
  auto* result = new urma_target_seg_t{};
  result->seg.ubva.va = config->va;
  result->seg.ubva.eid = context->eid;
  result->seg.len = config->len;
  result->seg.token_id = config->token_id->token_id;
  return result;
}
urma_status_t urma_unregister_seg(urma_target_seg_t* segment) {
  --segments; delete segment; return URMA_SUCCESS;
}
urma_jfc_t* urma_create_jfc(urma_context_t*, urma_jfc_cfg_t*) {
  ++jfcs;
  auto* result = new urma_jfc_t{};
  result->jfc_id.id = ++next_id;
  return result;
}
urma_status_t urma_delete_jfc(urma_jfc_t* jfc) { --jfcs; delete jfc; return URMA_SUCCESS; }
urma_jfr_t* urma_create_jfr(urma_context_t*, urma_jfr_cfg_t*) { ++jfrs; return new urma_jfr_t{}; }
urma_status_t urma_delete_jfr(urma_jfr_t* jfr) { --jfrs; delete jfr; return URMA_SUCCESS; }
urma_jetty_t* urma_create_jetty(urma_context_t*, urma_jetty_cfg_t* config) {
  ++jetties;
  auto* result = new urma_jetty_t{};
  result->jetty_id.id = ++next_id;
  result->jetty_cfg = *config;
  return result;
}
urma_status_t urma_delete_jetty(urma_jetty_t* jetty) { --jetties; delete jetty; return URMA_SUCCESS; }
urma_status_t urma_unimport_jetty(urma_target_jetty_t* target) {
  --imports; delete target; return URMA_SUCCESS;
}
// Test implementations of the expanded symbols used by the service.
int FakeGetTpList(urma_context_t*, kvaf::UrmaExtGetTpCfg*, uint32_t* count,
                     kvaf::UrmaExtTpInfo* info) {
  *count = 1;
  info->tp_handle = 123;
  return 0;
}
urma_target_jetty_t* FakeImportJettyEx(urma_context_t*, kvaf::UrmaExtRjetty*,
                                        urma_token_t*, kvaf::UrmaExtActiveTpCfg*) {
  if (fail_import) return nullptr;
  ++imports;
  return new urma_target_jetty_t{};
}
}

void* FakeDlsym(void*, const char* name) {
  if (std::strcmp(name, "urma_get_tp_list") == 0)
    return reinterpret_cast<void*>(&FakeGetTpList);
  if (std::strcmp(name, "urma_import_jetty_ex") == 0)
    return reinterpret_cast<void*>(&FakeImportJettyEx);
  return nullptr;
}

int main() {
  using namespace kvaf;
  using namespace kvaf::peer;
  CpuRuntime cpu;
  CpuRuntimeOptions options;
  options.memory_page_mode = CpuMemoryPageMode::kDefault;
  CHECK(cpu.Initialize(options).ok());
  int listener = socket(AF_INET, SOCK_STREAM, 0);
  CHECK(listener >= 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  CHECK(bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
  socklen_t length = sizeof(address);
  CHECK(getsockname(listener, reinterpret_cast<sockaddr*>(&address), &length) == 0);
  CHECK(listen(listener, 8) == 0);
  std::thread server([&] {
    for (;;) {
      int connection = accept(listener, nullptr, nullptr);
      if (connection < 0) break;
      CHECK(cpu.ServeConnection(connection).ok());
      close(connection);
    }
  });
  const std::string endpoint = "127.0.0.1:" + std::to_string(ntohs(address.sin_port));
  PeerLink writer(endpoint), reader(endpoint), other(endpoint);
  PeerRegion a{}, b{}, c{};
  CHECK(writer.AllocateRegion(4096, 256, 0, &a).ok());
  CHECK(segments == 1 && tokens == 1);
  CHECK(a.base == reinterpret_cast<uint64_t>(registered_memory) && registered_bytes == 4096);
  CHECK(a.handle.region_id != 0 && a.handle.server_instance_id != 0);
  std::memset(registered_memory, 0x73, 4096);

  auto bad = a.handle;
  ++bad.server_instance_id;
  CHECK(!reader.AttachRegion(bad, &b).ok());
  bad = a.handle; ++bad.region_id;
  CHECK(!reader.AttachRegion(bad, &b).ok());
  bad = a.handle; bad.length *= 2;
  CHECK(!reader.AttachRegion(bad, &b).ok());
  bad = a.handle; bad.slot_size *= 2;
  CHECK(!reader.AttachRegion(bad, &b).ok());
  bad = a.handle; ++bad.candidate_id;
  CHECK(!reader.AttachRegion(bad, &b).ok());
  CHECK(!reader.has_region() && segments == 1);
  CHECK(reader.AttachRegion(a.handle, &b).ok());
  CHECK(b.base == a.base && b.length == a.length && b.handle.region_id == a.handle.region_id);
  CHECK(b.session_id != a.session_id && b.segment_key == a.segment_key);
  CHECK(b.token_id == a.token_id && b.token_value == a.token_value);
  CHECK(!reader.AttachRegion(a.handle, &c).ok());
  CHECK(segments == 1 && tokens == 1);
  CHECK(static_cast<uint8_t*>(registered_memory)[4095] == 0x73);

  auto channel = [](PeerLink& link, const PeerRegion& region) {
    std::array<uint8_t, kQpKeyBytes> key{}, peer_key{};
    std::array<uint8_t, kEidBytes> local_eid{};
    local_eid[0] = 2;
    uint32_t key_bytes = 0, psn = 0;
    uint64_t tp = 0;
    return link.CreateChannel(region.base, key.data(), kNpuJettyKeyBytes,
                              local_eid.data(), 2, 3, 4, &peer_key, &key_bytes, &tp, &psn);
  };
  CHECK(channel(writer, a).ok());
  CHECK(channel(reader, b).ok());
  CHECK(jetties == 2 && imports == 2 && jfcs == 2 && jfrs == 2);
  // A failed attachment's channel creation must not damage the existing users.
  CHECK(other.AttachRegion(a.handle, &c).ok());
  fail_import = true;
  CHECK(!channel(other, c).ok());
  fail_import = false;
  CHECK(other.ReleaseRegion(c.base).ok());
  CHECK(jetties == 2 && imports == 2 && jfcs == 2 && jfrs == 2 && segments == 1);

  CHECK(writer.ReleaseRegion(a.base).ok());
  CHECK(!writer.has_region());
  CHECK(segments == 1 && tokens == 1 && jetties == 1 && imports == 1);
  CHECK(reinterpret_cast<uint8_t*>(b.base)[4095] == 0x73);
  CHECK(channel(reader, b).ok());
  CHECK(other.AttachRegion(a.handle, &c).ok());  // creator is no longer required
  CHECK(other.ReleaseRegion(c.base).ok());
  CHECK(reader.ReleaseRegion(b.base).ok());
  CHECK(segments == 0 && tokens == 0 && jetties == 0 && imports == 0 && jfcs == 0 && jfrs == 0);
  CHECK(!other.AttachRegion(a.handle, &c).ok());
  CHECK(segments == 0 && !other.has_region());

  // Fresh allocations have new identities, even if malloc reuses the address.
  CHECK(other.AllocateRegion(4096, 256, 0, &c).ok());
  CHECK(c.handle.region_id != a.handle.region_id);
  CHECK(!writer.AttachRegion(a.handle, &a).ok());
  CHECK(other.ReleaseRegion(c.base).ok());
  shutdown(listener, SHUT_RDWR);
  close(listener);
  server.join();
  CHECK(cpu.Shutdown().ok());
  {
    UrmaResourcePool pool;
    CHECK(pool.Initialize("", false, 0));
    CpuService restarted(&pool, CpuMemoryPageMode::kDefault, false);
    AllocateRegionResponse fresh{};
    CHECK(restarted.Allocate(4096, 256, 0, &fresh) == Result::kOk);
    CHECK(fresh.region_id == a.handle.region_id);  // numeric ID is reused after restart
    CHECK(fresh.server_instance_id != a.handle.server_instance_id);
    AttachRegionRequest stale{};
    stale.server_instance_id = a.handle.server_instance_id;
    stale.region_id = a.handle.region_id;
    stale.ddr_bytes = a.handle.length;
    stale.slot_size = a.handle.slot_size;
    stale.candidate_id = a.handle.candidate_id;
    AllocateRegionResponse rejected{};
    CHECK(restarted.Attach(stale, &rejected) == Result::kNotFound);
    CHECK(restarted.Release(fresh.session_id, a.handle.server_instance_id) == Result::kNotFound);
    CHECK(segments == 1);
    CHECK(restarted.Release(fresh.session_id, fresh.server_instance_id) == Result::kOk);
    CHECK(segments == 0 && tokens == 0);
  }
  std::cout << "PASS: shared DDR control plane and per-session resource lifetime\n";
}
