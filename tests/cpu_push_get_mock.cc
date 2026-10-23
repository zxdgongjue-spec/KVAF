#include "cpu_push_get_mock.h"

#include <atomic>
#include <array>
#include <cstring>
#include <deque>
#include <stdexcept>

namespace {
struct Cq { urma_jfc_t handle{}; std::deque<urma_cr_t> completions; };
std::atomic<int> live{0};
std::atomic<std::uint64_t> writes{0}, markers{0};
std::uint32_t max_sge = 3, max_bytes = 1024;
bool fail_post = false, fail_delete = false, error_completion = false;
bool expanded_attributes = false;
void Check(bool valid) { if (!valid) throw std::runtime_error("invalid CPU push provider contract"); }
}  // namespace
namespace push_mock {
void Reset(std::uint32_t sge, std::uint32_t bytes) {
  Check(live == 0); max_sge = sge; max_bytes = bytes;
  writes = 0; markers = 0; fail_post = fail_delete = error_completion = false;
  expanded_attributes = false;
}
void UseExpandedDeviceAttributes() { expanded_attributes = true; }
void FailNextPost() { fail_post = true; }
void FailNextDelete() { fail_delete = true; }
void ErrorNextCompletion() { error_completion = true; }
int LiveResources() { return live; }
std::uint64_t Writes() { return writes; }
std::uint64_t Markers() { return markers; }
int QueryTp(urma_context_t*, kvaf::urma::TpQuery*, std::uint32_t* count, kvaf::urma::TpInfo* info) {
  *count = 1; info->tp_handle = 17; return 0;
}
urma_target_jetty_t* ImportJetty(urma_context_t*, kvaf::urma::RemoteJettyEx*, urma_token_t*, kvaf::urma::ActiveTp*) {
  ++live; return new urma_target_jetty_t{};
}
}  // namespace push_mock

extern "C" {
urma_status_t urma_query_device(urma_device_t*, urma_device_attr_t* attr) {
  if (expanded_attributes) {
    // Expanded UB SDK response: 496 B, max_jfs_sge at 60, max_msg_size at 72,
    // reserved_jetty_id_max at 488. Independent of the caller's header layout.
    std::array<std::uint8_t, 496> response{};
    std::memcpy(response.data() + 60, &max_sge, 4);
    const std::uint64_t message = max_bytes;
    std::memcpy(response.data() + 72, &message, 8);
    const std::uint32_t reserved_max = 1023;
    std::memcpy(response.data() + 488, &reserved_max, 4);
    std::memcpy(attr, response.data(), response.size());
    return URMA_SUCCESS;
  }
  *attr = {}; attr->dev_cap.max_jfs_sge = max_sge; attr->dev_cap.max_msg_size = max_bytes; return URMA_SUCCESS;
}
urma_jfc_t* urma_create_jfc(urma_context_t*, urma_jfc_cfg_t*) { ++live; return &(new Cq)->handle; }
urma_status_t urma_delete_jfc(urma_jfc_t* p) { --live; delete reinterpret_cast<Cq*>(p); return URMA_SUCCESS; }
urma_jfr_t* urma_create_jfr(urma_context_t*, urma_jfr_cfg_t*) { ++live; return new urma_jfr_t{}; }
urma_status_t urma_delete_jfr(urma_jfr_t* p) { --live; delete p; return URMA_SUCCESS; }
urma_jetty_t* urma_create_jetty(urma_context_t*, urma_jetty_cfg_t* cfg) {
  Check(cfg->jfs_cfg.max_sge <= max_sge);
  Check(cfg->jfs_cfg.max_rsge == 1 && cfg->jfs_cfg.priority == URMA_MAX_PRIORITY &&
        cfg->jfs_cfg.rnr_retry == URMA_TYPICAL_RNR_RETRY &&
        cfg->jfs_cfg.err_timeout == URMA_TYPICAL_ERR_TIMEOUT);
  auto* q = new urma_jetty_t{}; q->jetty_cfg = *cfg; ++live; return q;
}
urma_status_t urma_delete_jetty(urma_jetty_t* q) {
  if (fail_delete) { fail_delete = false; return URMA_FAIL; }
  --live; delete q; return URMA_SUCCESS;
}
urma_status_t urma_unimport_jetty(urma_target_jetty_t* q) { --live; delete q; return URMA_SUCCESS; }
urma_token_id_t* urma_alloc_token_id(urma_context_t*) { ++live; return new urma_token_id_t{}; }
urma_status_t urma_free_token_id(urma_token_id_t* p) { --live; delete p; return URMA_SUCCESS; }
urma_target_seg_t* urma_register_seg(urma_context_t*, urma_seg_cfg_t* cfg) {
  Check(cfg->token_id && cfg->flag.bs.token_id_valid && !cfg->flag.bs.non_pin &&
        cfg->flag.bs.token_policy == URMA_TOKEN_PLAIN_TEXT && cfg->flag.bs.access == 1);
  auto* p = new urma_target_seg_t{}; p->seg.ubva.va = cfg->va; p->seg.len = cfg->len; ++live; return p;
}
urma_status_t urma_unregister_seg(urma_target_seg_t* p) { --live; delete p; return URMA_SUCCESS; }
urma_target_seg_t* urma_import_seg(urma_context_t*, urma_seg_t* seg, urma_token_t*, uint64_t, urma_import_seg_flag_t flag) {
  Check(flag.bs.cacheable == URMA_NON_CACHEABLE && flag.bs.mapping == URMA_SEG_NOMAP &&
        flag.bs.access == ((1U << 1U) | (1U << 2U)));
  auto* p = new urma_target_seg_t{}; p->seg = *seg; ++live; return p;
}
urma_status_t urma_unimport_seg(urma_target_seg_t* p) { --live; delete p; return URMA_SUCCESS; }
urma_status_t urma_post_jetty_send_wr(urma_jetty_t* jetty, urma_jfs_wr_t* first, urma_jfs_wr_t** bad) {
  if (fail_post) { fail_post = false; *bad = first; return URMA_FAIL; }
  auto* cq = reinterpret_cast<Cq*>(jetty->jetty_cfg.jfs_cfg.jfc);
  for (auto* wr = first; wr; wr = wr->next) {
    Check(wr->rw.src.num_sge && wr->rw.dst.num_sge && wr->tjetty);
    const bool write = wr->opcode == URMA_OPC_WRITE;
    Check(write || wr->opcode == URMA_OPC_READ);
    const auto& sources = wr->rw.src;
    const auto& targets = wr->rw.dst;
    Check((write ? sources.num_sge : targets.num_sge) <= max_sge);
    Check((write ? targets.num_sge : sources.num_sge) == 1);
    std::uint32_t total = 0, src = 0, dst = 0, src_offset = 0, dst_offset = 0;
    while (src < sources.num_sge && dst < targets.num_sge) {
      const auto* s = reinterpret_cast<const kvaf::urma::Sge*>(sources.sge) + src;
      const auto* d = reinterpret_cast<const kvaf::urma::Sge*>(targets.sge) + dst;
      const auto bytes = std::min(s->len - src_offset, d->len - dst_offset);
      Check(bytes && s->tseg && d->tseg);
      std::memcpy(reinterpret_cast<void*>(d->addr + dst_offset), reinterpret_cast<const void*>(s->addr + src_offset), bytes);
      total += bytes; src_offset += bytes; dst_offset += bytes;
      if (src_offset == s->len) { ++src; src_offset = 0; }
      if (dst_offset == d->len) { ++dst; dst_offset = 0; }
    }
    Check(src == sources.num_sge && dst == targets.num_sge && total <= max_bytes);
    if (write) {
      ++writes;
      if (total == 8) { Check(wr->flag.bs.place_order == URMA_STRONG_ORDER); ++markers; }
    }
    if (wr->flag.bs.complete_enable) {
      Check(wr->flag.bs.place_order == URMA_STRONG_ORDER);
      urma_cr_t cr{}; cr.status = error_completion ? URMA_CR_WR_FLUSH_ERR : URMA_CR_SUCCESS;
      error_completion = false;
      cr.user_ctx = wr->user_ctx; cq->completions.push_back(cr);
    }
  }
  return URMA_SUCCESS;
}
int urma_poll_jfc(urma_jfc_t* handle, int capacity, urma_cr_t* crs) {
  auto* cq = reinterpret_cast<Cq*>(handle);
  int count = 0;
  while (count < capacity && !cq->completions.empty()) {
    crs[count++] = cq->completions.front(); cq->completions.pop_front();
  }
  return count;
}
// NativeApi also carries discovery entry points; this fixture does not discover endpoints.
urma_device_t** urma_get_device_list(int*) { return nullptr; }
void urma_free_device_list(urma_device_t**) {}
urma_eid_info_t* urma_get_eid_list(urma_device_t*, std::uint32_t*) { return nullptr; }
void urma_free_eid_list(urma_eid_info_t*) {}
}  // extern "C"
