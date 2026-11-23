// Real CPU SGL WRITE transport with a synchronous fake URMA provider.
// Requires the public UMDK headers, no URMA library or hardware.
#include "cpu_service/push/cpu_push_get_transport.h"
#include "urma_device_abi.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <vector>

namespace {
using kvaf::urma::Sge;
std::uint32_t device_sges = 13, queue_sges{}, data_wrs{}, data_sges{}, segments{};
std::uint64_t output_base{}, output_bytes{}, done_address{};
std::map<urma_jfc_t*, std::deque<std::uint64_t>> completions;
int QueryTp(urma_context_t*, kvaf::urma::TpQuery*, std::uint32_t* count, kvaf::urma::TpInfo* info) {
  *count = 1; info->tp_handle = 1; return 0;
}
urma_target_jetty_t* ImportJetty(urma_context_t*, kvaf::urma::RemoteJettyEx*, urma_token_t*,
                                kvaf::urma::ActiveTp*) {
  return new urma_target_jetty_t{};
}
}  // namespace

extern "C" {
urma_device_t** urma_get_device_list(int*) { return nullptr; }
void urma_free_device_list(urma_device_t**) {}
urma_eid_info_t* urma_get_eid_list(urma_device_t*, std::uint32_t*) { return nullptr; }
void urma_free_eid_list(urma_eid_info_t*) {}
urma_status_t urma_query_device(urma_device_t*, urma_device_attr_t* attributes) {
  auto& value = *reinterpret_cast<kvaf::urma::UbDeviceAttributes*>(attributes);
  value = {};
  value.dev_cap.max_jfs_sge = device_sges;
  value.dev_cap.max_msg_size = 65536;
  return URMA_SUCCESS;
}
urma_token_id_t* urma_alloc_token_id(urma_context_t*) { return new urma_token_id_t{}; }
urma_status_t urma_free_token_id(urma_token_id_t* token) { delete token; return URMA_SUCCESS; }
urma_target_seg_t* urma_register_seg(urma_context_t*, urma_seg_cfg_t* config) {
  auto* segment = new urma_target_seg_t{};
  segment->seg.ubva.va = config->va;
  segment->seg.len = config->len;
  ++segments;
  return segment;
}
urma_status_t urma_unregister_seg(urma_target_seg_t* segment) {
  --segments; delete segment; return URMA_SUCCESS;
}
urma_target_seg_t* urma_import_seg(urma_context_t*, urma_seg_t* value, urma_token_t*,
                                  std::uint64_t, urma_import_seg_flag_t) {
  auto* segment = new urma_target_seg_t{};
  segment->seg = *value;
  ++segments;
  return segment;
}
urma_status_t urma_unimport_seg(urma_target_seg_t* segment) {
  --segments; delete segment; return URMA_SUCCESS;
}
urma_jfc_t* urma_create_jfc(urma_context_t*, urma_jfc_cfg_t*) {
  auto* cq = new urma_jfc_t{};
  completions[cq] = {};
  return cq;
}
urma_status_t urma_delete_jfc(urma_jfc_t* cq) {
  assert(completions[cq].empty()); completions.erase(cq); delete cq; return URMA_SUCCESS;
}
urma_jfr_t* urma_create_jfr(urma_context_t*, urma_jfr_cfg_t*) { return new urma_jfr_t{}; }
urma_status_t urma_delete_jfr(urma_jfr_t* queue) { delete queue; return URMA_SUCCESS; }
urma_jetty_t* urma_create_jetty(urma_context_t*, urma_jetty_cfg_t* config) {
  queue_sges = config->jfs_cfg.max_sge;
  auto* queue = new urma_jetty_t{};
  queue->jetty_cfg = *config;
  return queue;
}
urma_status_t urma_delete_jetty(urma_jetty_t* queue) { delete queue; return URMA_SUCCESS; }
urma_status_t urma_unimport_jetty(urma_target_jetty_t* queue) { delete queue; return URMA_SUCCESS; }
urma_status_t urma_post_jetty_send_wr(urma_jetty_t* queue, urma_jfs_wr_t* wr, urma_jfs_wr_t**) {
  for (; wr; wr = wr->next) {
    const auto* src = reinterpret_cast<const Sge*>(wr->rw.src.sge);
    const auto* dst = reinterpret_cast<const Sge*>(wr->rw.dst.sge);
    assert(wr->rw.dst.num_sge == 1 && wr->rw.src.num_sge <= queue_sges);
    assert(dst->addr >= dst->tseg->seg.ubva.va);
    assert(dst->len <= dst->tseg->seg.len - (dst->addr - dst->tseg->seg.ubva.va));
    std::uint32_t bytes = 0;
    for (std::uint32_t i = 0; i < wr->rw.src.num_sge; ++i) {
      assert(src[i].addr >= src[i].tseg->seg.ubva.va);
      assert(src[i].len <= src[i].tseg->seg.len - (src[i].addr - src[i].tseg->seg.ubva.va));
      std::memcpy(reinterpret_cast<void*>(dst->addr + bytes), reinterpret_cast<void*>(src[i].addr), src[i].len);
      bytes += src[i].len;
    }
    assert(bytes == dst->len && bytes <= 65536);
    if (wr->opcode == URMA_OPC_WRITE && dst->addr >= output_base && dst->addr < output_base + output_bytes) {
      assert(wr->rw.src.num_sge <= queue_sges);
      ++data_wrs; data_sges += wr->rw.src.num_sge;
    }
    if (wr->opcode == URMA_OPC_WRITE && dst->addr == done_address)
      assert(segments == 2);  // Workspace + control; output is unimported before done.
    if (wr->flag.bs.complete_enable)
      completions[queue->jetty_cfg.jfs_cfg.jfc].push_back(wr->user_ctx);
  }
  return URMA_SUCCESS;
}
int urma_poll_jfc(urma_jfc_t* cq, int capacity, urma_cr_t* result) {
  auto& pending = completions[cq];
  int count = 0;
  while (count < capacity && !pending.empty()) {
    result[count] = {};
    result[count].status = URMA_CR_SUCCESS;
    result[count++].user_ctx = pending.front();
    pending.pop_front();
  }
  return count;
}
}  // extern "C"

namespace {
void Run(std::uint32_t requested, std::uint32_t expected, std::uint32_t count = 4096,
         std::uint32_t token_bytes = 656) {
  using namespace kvaf::cpu_push;
  Layout layout;
  assert(MakeLayout(token_bytes, 4, &layout, true));
  auto* hbm = static_cast<std::uint8_t*>(std::aligned_alloc(64, layout.bytes));
  assert(hbm);
  std::memset(hbm, 0, layout.bytes);
  std::vector<std::uint8_t> source(count * token_bytes), output(source.size()), second(output.size());
  for (std::size_t i = 0; i < source.size(); ++i) source[i] = (i * 17 + i / token_bytes) % 251;
  urma_device_t device{};
  urma_context_t context{};
  context.dev = &device;
  urma_target_seg_t source_segment{};
  source_segment.seg.ubva.va = reinterpret_cast<std::uint64_t>(source.data());
  source_segment.seg.len = source.size();
  urma_seg_t hbm_segment{};
  hbm_segment.ubva.va = reinterpret_cast<std::uint64_t>(hbm);
  hbm_segment.len = layout.bytes;
  kvaf::peer::OpenCpuPushGetRequest setup{};
  setup.source_bytes = source.size(); setup.hbm_base = hbm_segment.ubva.va; setup.hbm_bytes = layout.bytes;
  setup.npu_tp = 1; setup.npu_psn = 1; setup.token_value = 7;
  setup.segment_bytes = sizeof(hbm_segment); setup.token_bytes = token_bytes; setup.workers = 4;
  setup.output_mode = kDirectOutput; setup.sge_per_wr = requested;
  std::memcpy(setup.segment, &hbm_segment, sizeof(hbm_segment));
  for (auto& channel : setup.channels) channel.bytes = 28;
  kvaf::peer::OpenCpuPushGetResponse response{};
  Transport transport;
  assert(transport.Initialize(&context, &source_segment, setup, QueryTp, ImportJetty, 1, &response).ok());
  assert(queue_sges == expected);
  done_address = setup.hbm_base + layout.done;
  for (auto sequence : {1U, 2U}) {
    auto& result = sequence == 1 ? output : second;
    std::fill(result.begin(), result.end(), 0xA5);
    output_base = reinterpret_cast<std::uint64_t>(result.data()); output_bytes = result.size();
    auto* request = reinterpret_cast<Request*>(hbm);
    *request = {kMagic, sequence, count, token_bytes, 4, kDirectOutput, {}};
    auto* target = reinterpret_cast<OutputTarget*>(hbm + layout.output_target);
    *target = {};
    target->base = output_base; target->bytes = result.size(); target->token_value = 7;
    target->segment_bytes = sizeof(urma_seg_t);
    urma_seg_t output_segment{};
    output_segment.ubva.va = target->base; output_segment.len = target->bytes;
    std::memcpy(target->segment, &output_segment, sizeof(output_segment));
    auto* addresses = reinterpret_cast<std::uint64_t*>(hbm + layout.descriptor);
    for (std::uint32_t i = 0; i < count; ++i)
      addresses[i] = source_segment.seg.ubva.va + ((count - i + sequence - 1) % count) * token_bytes;
    data_wrs = data_sges = 0;
    assert(transport.ReadDescriptor(sequence).ok());
    assert(transport.WriteBatch().ok());
    assert(transport.Complete(sequence, Result::kOk).ok());
    if (count == 4096 && token_bytes == 656) {
      assert(data_wrs == (expected == 4 ? 1024 : expected == 8 ? 512 : 320));
      assert(data_sges == count);
    }
    for (std::uint32_t i = 0; i < count; ++i)
      assert(std::memcmp(result.data() + i * token_bytes, reinterpret_cast<void*>(addresses[i]), token_bytes) == 0);
  }
  assert(transport.Shutdown().ok());
  assert(segments == 0 && completions.empty());
  std::free(hbm);
}
void Reject(std::uint32_t requested, std::uint32_t device_limit, std::uint32_t mode) {
  device_sges = device_limit;
  urma_context_t context{};
  urma_device_t device{};
  context.dev = &device;
  urma_target_seg_t source{};
  kvaf::cpu_push::Layout layout;
  assert(kvaf::cpu_push::MakeLayout(656, 4, &layout, mode == kvaf::cpu_push::kDirectOutput));
  kvaf::peer::OpenCpuPushGetRequest setup{};
  setup.npu_tp = 1; setup.npu_psn = 1; setup.token_value = 7; setup.source_bytes = 656;
  setup.segment_bytes = sizeof(urma_seg_t); setup.token_bytes = 656; setup.workers = 4;
  setup.hbm_base = 4096; setup.hbm_bytes = layout.bytes;
  setup.output_mode = mode; setup.sge_per_wr = requested;
  kvaf::peer::OpenCpuPushGetResponse response{};
  kvaf::cpu_push::Transport transport;
  assert(!transport.Initialize(&context, &source, setup, QueryTp, ImportJetty, 1, &response).ok());
  assert(transport.Shutdown().ok());
  assert(segments == 0 && completions.empty());
}
}  // namespace

int main() {
  Run(4, 4);
  Run(8, 8);
  Run(0, 13);
  Run(8, 8, 65, 10000);  // Byte-limited WRs and split-token source offsets.
  Reject(8, 4, kvaf::cpu_push::kDirectOutput);
  Reject(17, 32, kvaf::cpu_push::kDirectOutput);
  Reject(0, 0, kvaf::cpu_push::kDirectOutput);
  Reject(4, 13, kvaf::cpu_push::kScatterOutput);
}
