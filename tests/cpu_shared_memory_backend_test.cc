#include "cpu_service/backends/cpu_backends.h"
#include "cpu_gather_executor.h"
#include <urma_api.h>
#include <atomic>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

#define CHECK(c) do { if (!(c)) throw std::runtime_error(#c); } while (false)
namespace {
std::atomic<unsigned> contexts_created{}, contexts_deleted{}, registered{}, unregistered{}, next_id{1};
std::mutex driver;
std::set<urma_context_t*> contexts;
std::map<urma_target_seg_t*, urma_context_t*> segments;
std::set<urma_jetty_t*> jettys;
bool fail_unregister{}, fail_register{}, fail_delete_jetty{}, fail_delete_context{};
unsigned live_tokens{};
urma_device_t device;
urma_eid_info_t eids[2];
}
extern "C" {
urma_status_t urma_init(urma_init_attr_t*) { return URMA_SUCCESS; }
urma_status_t urma_uninit() { return URMA_SUCCESS; }
urma_device_t** urma_get_device_list(int* count) {
  static urma_device_t* devices[]{&device}; *count = 1; return devices;
}
void urma_free_device_list(urma_device_t**) {}
urma_eid_info_t* urma_get_eid_list(urma_device_t*, std::uint32_t* count) { *count = 2; return eids; }
void urma_free_eid_list(urma_eid_info_t*) {}
urma_device_t* urma_get_device_by_name(char* name) { return std::strcmp(name, device.name) ? nullptr : &device; }
urma_context_t* urma_create_context(urma_device_t*, std::uint32_t index) {
  auto* context = new urma_context_t{}; context->eid = eids[index].eid;
  std::lock_guard<std::mutex> lock(driver); contexts.insert(context); ++contexts_created; return context;
}
urma_status_t urma_delete_context(urma_context_t* context) {
  std::lock_guard<std::mutex> lock(driver);
  if (fail_delete_context) return URMA_FAIL;
  for (const auto& entry : segments) CHECK(entry.second != context);
  CHECK(contexts.erase(context) == 1); ++contexts_deleted; delete context; return URMA_SUCCESS;
}
urma_token_id_t* urma_alloc_token_id(urma_context_t*) {
  auto* token = new urma_token_id_t{}; token->token_id = next_id++; ++live_tokens; return token;
}
urma_status_t urma_free_token_id(urma_token_id_t* token) { --live_tokens; delete token; return URMA_SUCCESS; }
urma_target_seg_t* urma_register_seg(urma_context_t* context, urma_seg_cfg_t* config) {
  std::lock_guard<std::mutex> lock(driver); CHECK(contexts.count(context));
  if (fail_register) return nullptr;
  auto* segment = new urma_target_seg_t{};
  segment->seg.ubva.va = config->va + 0x1000;
  segment->seg.ubva.eid = context->eid;
  segment->seg.len = config->len; segment->seg.token_id = config->token_id->token_id;
  segments.emplace(segment, context); ++registered; return segment;
}
urma_status_t urma_unregister_seg(urma_target_seg_t* segment) {
  std::lock_guard<std::mutex> lock(driver);
  if (fail_unregister) return URMA_FAIL;
  CHECK(segments.erase(segment) == 1); ++unregistered; delete segment; return URMA_SUCCESS;
}
urma_jfc_t* urma_create_jfc(urma_context_t*, urma_jfc_cfg_t*) { return new urma_jfc_t{}; }
urma_status_t urma_delete_jfc(urma_jfc_t* value) { delete value; return URMA_SUCCESS; }
urma_jfr_t* urma_create_jfr(urma_context_t*, urma_jfr_cfg_t*) { return new urma_jfr_t{}; }
urma_status_t urma_delete_jfr(urma_jfr_t* value) { delete value; return URMA_SUCCESS; }
urma_jetty_t* urma_create_jetty(urma_context_t*, urma_jetty_cfg_t* config) {
  auto* value = new urma_jetty_t{}; value->jetty_cfg = *config; value->jetty_id.id = next_id++;
  jettys.insert(value); return value;
}
urma_status_t urma_delete_jetty(urma_jetty_t* value) {
  if (fail_delete_jetty) return URMA_FAIL;
  CHECK(jettys.erase(value) == 1); delete value; return URMA_SUCCESS;
}
urma_target_jetty_t* urma_import_jetty_ex(urma_context_t*, urma_rjetty_t*, urma_token_t*,
                                        urma_import_jetty_ex_cfg_t*) { return new urma_target_jetty_t{}; }
urma_status_t urma_unimport_jetty(urma_target_jetty_t* value) { delete value; return URMA_SUCCESS; }
urma_status_t urma_get_tp_list(urma_context_t*, urma_get_tp_cfg_t*, std::uint32_t* count, urma_tp_info_t* info) {
  *count = 1; info->tp_handle = next_id++; return URMA_SUCCESS;
}
urma_status_t urma_query_device(urma_device_t*, urma_device_attr_t*) { return URMA_FAIL; }
urma_target_seg_t* urma_import_seg(urma_context_t*, urma_seg_t*, urma_token_t*, std::uint64_t,
                                  urma_import_seg_flag_t) { return nullptr; }
urma_status_t urma_unimport_seg(urma_target_seg_t*) { return URMA_FAIL; }
urma_status_t urma_post_jetty_send_wr(urma_jetty_t*, urma_jfs_wr_t*, urma_jfs_wr_t**) { return URMA_FAIL; }
int urma_poll_jfc(urma_jfc_t*, int, urma_cr_t*) { return -1; }
}

std::uint32_t Channel(kvaf::UrmaCpuBackend& backend, const kvaf::peer::PeerRegion& binding) {
  using namespace kvaf::peer;
  CreateChannelRequest request{};
  request.header = MakeHeader(Command::kCreateChannel, sizeof(request));
  request.session_id = binding.session_id;
  request.local_qp_key_bytes = 28; request.local_eid[0] = 9;
  request.local_tp_handle = 10; request.local_psn = 1;
  request.peer_token_value = binding.token_value;
  request.reserved = kChannelSharedTp;
  const auto* data = reinterpret_cast<const std::uint8_t*>(&request);
  std::vector<std::uint8_t> response;
  CHECK(backend.Call({data, data + sizeof(request)}, &response).ok());
  CHECK(response.size() == sizeof(CreateChannelResponse));
  CreateChannelResponse result{};
  std::memcpy(&result, response.data(), sizeof(result));
  CHECK(result.result == static_cast<int>(Result::kOk));
  urma_jetty_id_t jetty{};
  std::memcpy(&jetty, result.peer_qp_key, sizeof(jetty));
  return jetty.id;
}

template <typename Response, typename Request>
Response Call(kvaf::UrmaCpuBackend& backend, const Request& request) {
  const auto* data = reinterpret_cast<const std::uint8_t*>(&request);
  std::vector<std::uint8_t> response;
  CHECK(backend.Call({data, data + sizeof(request)}, &response).ok());
  CHECK(response.size() == sizeof(Response));
  Response result{};
  std::memcpy(&result, response.data(), sizeof(result));
  CHECK(result.header.command == (request.header.command | 0x8000));
  return result;
}

void LazyGatherWorkspace() {
  using namespace kvaf;
  using namespace kvaf::peer;
  alignas(64) std::uint8_t memory[4096];
  for (std::size_t i = 0; i < sizeof(memory); ++i) memory[i] = i % 251;
  std::vector<std::uint8_t> source_snapshot(memory, memory + sizeof(memory));
  CpuRuntimeOptions options;
  options.memory_page_mode = CpuMemoryPageMode::kDefault;
  options.gather_threads = 2;
  auto pool = std::make_shared<CpuUrmaMemoryPool>(options, memory, sizeof(memory), 123, 999);
  UrmaCpuBackend backend;
  std::uint64_t next_session = 1;
  CHECK(backend.Initialize(options, next_session, pool).ok());
  PeerRegion source;
  const auto old_registered = registered.load();
  const auto old_contexts = contexts_created.load();
  CHECK(backend.AttachRegion(memory, sizeof(memory), 1, 0, &source).ok());
  CHECK(registered == old_registered + 1 && contexts_created == old_contexts + 1);
  PrepareGatherWorkspaceRequest prepare{};
  prepare.header = MakeHeader(Command::kPrepareGatherWorkspace, sizeof(prepare));
  prepare.session_id = source.session_id;
  prepare.bytes = 4096;
  fail_register = true;
  CHECK(Call<AllocateRegionResponse>(backend, prepare).result == int(Result::kResourceError));
  CHECK(live_tokens == 2);  // Source and the failed workspace registration token.
  fail_register = false;
  const auto workspace = Call<AllocateRegionResponse>(backend, prepare);
  CHECK(workspace.result == int(Result::kOk));
  CHECK(workspace.session_id == source.session_id && workspace.peer_base != source.base);
  CHECK(workspace.token_id != source.token_id && workspace.token_value == source.token_value);
  CHECK(live_tokens == 2 && registered == old_registered + 2 && contexts_created == old_contexts + 1);
  prepare.bytes = 2048;
  const auto reused = Call<AllocateRegionResponse>(backend, prepare);
  CHECK(reused.peer_base == workspace.peer_base && reused.peer_length == 4096 &&
        reused.token_id == workspace.token_id && registered == old_registered + 2);
  prepare.bytes = 8192;
  CHECK(Call<AllocateRegionResponse>(backend, prepare).result == int(Result::kInvalidRequest));
  PeerRegion workspace_binding;
  workspace_binding.session_id = workspace.session_id;
  workspace_binding.token_value = workspace.token_value;
  CHECK(Channel(backend, workspace_binding) != 0);

  // Stub registrations intentionally translate addresses by 4 KiB. Source
  // spans still use the allocation's canonical address, while every command,
  // descriptor and staging address uses the workspace's registered address.
  auto* workspace_memory = reinterpret_cast<std::uint8_t*>(workspace.peer_base - 0x1000);
  auto* header = reinterpret_cast<GatherDescriptorHeader*>(workspace_memory);
  auto* spans = reinterpret_cast<GatherSpan*>(workspace_memory + sizeof(*header));
  auto* control = reinterpret_cast<GatherControl*>(workspace_memory + 4096 - 128);
  EnableGatherLanesRequest enable{};
  enable.header = MakeHeader(Command::kEnableGatherLanes, sizeof(enable));
  enable.session_id = source.session_id;
  enable.control_address = workspace.peer_base + 4096 - 128;
  enable.lanes = 1;
  CHECK(Call<StatusResponse>(backend, enable).result == int(Result::kOk));
  for (std::uint64_t sequence = 1; sequence <= 2; ++sequence) {
    *header = {};
    header->magic = kGatherDescriptorMagic;
    header->sequence = sequence;
    header->span_count = 4;
    header->span_bytes = sequence == 1 ? 64 : 0;
    header->total_bytes = 256;
    header->staging_address = workspace.peer_base + 512;
    header->pipeline_chunk_bytes = sequence == 1 ? 128 : 0;
    for (unsigned i = 0; i < 4; ++i) {
      spans[i].address = reinterpret_cast<std::uint64_t>(memory) + i * 128;
      spans[i].length = 64;
    }
    __atomic_store_n(&control->command, workspace.peer_base, __ATOMIC_RELEASE);
    __atomic_store_n(&control->notify, sequence, __ATOMIC_RELEASE);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (__atomic_load_n(&control->ack, __ATOMIC_ACQUIRE) != sequence) {
      CHECK(std::chrono::steady_clock::now() < deadline);
      std::this_thread::yield();
    }
    for (unsigned i = 0; i < 4; ++i)
      CHECK(std::memcmp(workspace_memory + 512 + i * 64, memory + i * 128, 64) == 0);
    CHECK(std::memcmp(memory, source_snapshot.data(), sizeof(memory)) == 0);
  }
  prepare.bytes = 4096;
  CHECK(Call<AllocateRegionResponse>(backend, prepare).peer_base == workspace.peer_base);
  fail_delete_jetty = true;
  CHECK(!backend.Shutdown().ok() && segments.size() == 2 && live_tokens == 2);
  fail_delete_jetty = false;
  fail_unregister = true;
  CHECK(!backend.Shutdown().ok() && jettys.empty() && segments.size() == 2 && live_tokens == 2);
  fail_unregister = false;
  CHECK(backend.Shutdown().ok() && segments.size() == 1 && live_tokens == 1);
  CHECK(pool->Close().ok() && segments.empty() && contexts.empty() && live_tokens == 0);

  // Existing same-region descriptors retain their original interpretation.
  alignas(64) std::uint8_t legacy[1024]{};
  const std::uint64_t base = 0x100000;
  auto* legacy_header = reinterpret_cast<GatherDescriptorHeader*>(legacy);
  legacy_header->magic = kGatherDescriptorMagic;
  legacy_header->sequence = 1;
  legacy_header->span_count = 1;
  legacy_header->span_bytes = legacy_header->total_bytes = 64;
  legacy_header->staging_address = base + 512;
  auto* legacy_span = reinterpret_cast<GatherSpan*>(legacy + sizeof(*legacy_header));
  legacy_span->address = base + 256;
  std::memset(legacy + 256, 0x42, 64);
  CpuGatherExecutor executor(legacy, base, 1);
  CHECK(executor.PrepareDirect(sizeof(legacy), base, 1, 1) == 1);
  executor.Copy();
  CHECK(std::memcmp(legacy + 256, legacy + 512, 64) == 0);
}
int main() {
  using namespace kvaf;
  device.type = URMA_TRANSPORT_UB; std::strcpy(device.name, "test_ub");
  for (unsigned i = 0; i < 2; ++i) { eids[i].eid_index = i; eids[i].eid.raw[0] = i + 1; }
  std::uint8_t memory[4096]{};
  CpuRuntimeOptions options;
  auto pool = std::make_shared<CpuUrmaMemoryPool>(options, memory, sizeof(memory), 123, 456);
  UrmaCpuBackend put, get, other_eid;
  std::uint64_t put_id = 1, get_id = 1, other_id = 1;
  CHECK(put.Initialize(options, put_id, pool).ok());
  CHECK(get.Initialize(options, get_id, pool).ok());
  peer::PeerRegion p, g, o;
  auto a = std::async(std::launch::async, [&] { return put.AttachRegion(memory, sizeof(memory), 1, 0, &p); });
  auto b = std::async(std::launch::async, [&] { return get.AttachRegion(memory, sizeof(memory), 1, 0, &g); });
  CHECK(a.get().ok() && b.get().ok());
  CHECK(contexts_created == 1 && registered == 1);
  CHECK(p.base == g.base && p.token_id == g.token_id && p.segment_key == g.segment_key);
  CHECK(Channel(put, p) != Channel(get, g) && jettys.size() == 2);
  CHECK(!pool->Close().ok());
  fail_delete_jetty = true;
  CHECK(!put.Shutdown().ok() && !pool->Close().ok() && unregistered == 0);
  fail_delete_jetty = false;
  CHECK(put.Shutdown().ok());
  CHECK(jettys.size() == 1 && unregistered == 0 && contexts_deleted == 0);
  CHECK(Channel(get, g) && jettys.size() == 2);  // The surviving Get remains usable.
  CHECK(other_eid.Initialize(options, other_id, pool).ok());
  CHECK(other_eid.AttachRegion(memory, sizeof(memory), 1, 1, &o).ok());
  CHECK(registered == 2 && contexts_created == 2 && o.token_id != g.token_id);
  CHECK(other_eid.Shutdown().ok() && get.Shutdown().ok());
  fail_unregister = true;
  CHECK(!pool->Close().ok() && contexts_deleted == 0 && segments.size() == 2);
  fail_unregister = false; fail_delete_context = true;
  CHECK(!pool->Close().ok() && unregistered == 2 && contexts_deleted == 0);
  fail_delete_context = false;
  CHECK(pool->Close().ok() && contexts_deleted == 2);
  CHECK(segments.empty() && contexts.empty() && jettys.empty());
  // A new allocation owner at the same address must make a new registration.
  pool = std::make_shared<CpuUrmaMemoryPool>(options, memory, sizeof(memory), 123, 789);
  CHECK(put.Initialize(options, put_id, pool).ok());
  CHECK(put.AttachRegion(memory, sizeof(memory), 1, 0, &p).ok() && registered == 3);
  CHECK(put.Shutdown().ok() && pool->Close().ok());
  CHECK(live_tokens == 0);
  LazyGatherWorkspace();
  std::cout << "CPU shared context/DDR with independent execution queues passed\n";
}
