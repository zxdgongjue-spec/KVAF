#pragma once

// Core 0 publishes request phases to registered CPU memory using the same
// device USER_CTL WRITE format as BatchPut. UB is reused only after list-build
// DMA has drained, and returned to payload scatter after the control CQ drains.
namespace kvaf_scatter {
using namespace AscendC;
using namespace kvaf_aiv;
enum class MailboxPhase { Full, Head, Suffix };

__aicore__ inline uint64_t PushRequest(const KvafScatterBatchGetArgs& a, uint32_t count,
    __gm__ kvaf::ScatterBatchGetCoreProfile* profile,
    MailboxPhase phase = MailboxPhase::Full) {
  auto* context = reinterpret_cast<__gm__ KvafScatterMailboxContext*>(a.request.channel_entity);
  auto* state = reinterpret_cast<__gm__ KvafBatchGetQueueState*>(a.request.queue_state);
  if (!context || !state) return KVAF_SEQ_QUEUE_ERROR;
  // Templates are read by DMA, not Scalar cache.
  Flush(reinterpret_cast<uint64_t>(context), kvaf::kScatterMailboxTemplateOffset);
  const auto& e = context->entity;
  if (e.magic != 0x4b564155U || e.version != 1 || !e.sq_va || !e.sq_db_va ||
      !e.cq_va || !e.cq_db_va || e.sq_depth < 4 || e.sq_depth % 4 ||
      !e.cq_depth || e.cqe_size < 32 || e.wqe_size != 64 || !e.local_token_id ||
      !context->max_write_bytes || context->max_write_bytes > 65536)
    return KVAF_SEQ_QUEUE_ERROR;
  if (state->initialized && (state->magic != 0x4b564151U || state->poisoned))
    return KVAF_SEQ_QUEUE_ERROR;
  if (!state->initialized) {
    state->magic = 0x4b564151U; state->initialized = 1;
    state->sq_producer_index = state->cq_consumer_index = 0;
  }
  // Never copy the intervening done/Ready lines: CPU owns them as WRITE
  // sources and could still be retiring the previous request's final CQE.
  const uint32_t head = phase == MailboxPhase::Full ? count : kvaf::cpu_push::kChunkRows;
  const bool suffix = phase == MailboxPhase::Suffix;
  const uint64_t offsets[3] = {suffix ? a.descriptor_offset + uint64_t(head) * 8 : 0,
      suffix ? kvaf::cpu_push::kSuffixNotifyOffset : a.descriptor_offset, 64};
  const uint64_t lengths[3] = {suffix ? uint64_t(count - head) * 8 : 64,
      suffix ? 8 : uint64_t(head) * 8, 8};
  const uint32_t regions = suffix ? 2 : 3;
  uint32_t region = 0;
  uint64_t offset = 0;
  const auto limit = e.sq_depth / 4 < kvaf::kScatterMailboxWqesPerBatch ?
      e.sq_depth / 4 : kvaf::kScatterMailboxWqesPerBatch;
  LocalTensor<uint32_t> buffer(TPosition::VECIN, 0, 16 * 16);
  LocalTensor<uint8_t> encoded(TPosition::VECIN, 0, 16 * 64);
  GlobalTensor<uint8_t> sq;
  sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(e.sq_va));
  GlobalTensor<uint32_t> templates;
  templates.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
      reinterpret_cast<uint64_t>(context) + kvaf::kScatterMailboxTemplateOffset));
  while (region < regions) {
    // One DMA replaces per-WQE Scalar reconstruction of all invariant words.
    DataCopy(buffer, templates, kvaf::kScatterMailboxWqesPerBatch * 16);
    set_flag(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
    wait_flag(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
    const auto pi = state->sq_producer_index;
    uint32_t wrs = 0;
    while (wrs < limit && region < regions) {
      const auto remaining = lengths[region] - offset;
      const auto bytes = remaining < context->max_write_bytes ?
          static_cast<uint32_t>(remaining) : context->max_write_bytes;
      const auto local = a.workspace + offsets[region] + offset;
      const auto remote = e.remote_host_ddr_address + offsets[region] + offset;
      offset += bytes;
      if (offset == lengths[region]) { ++region; offset = 0; }
      const bool tail = wrs + 1 == limit || region == regions;
      const auto owner = 1U - static_cast<uint32_t>(((pi + wrs) / e.sq_depth) & 1U);
      const auto base = wrs * 16;
      const uint32_t control = ((tail ? 2U : 1U) << 16) | (1U << 18) | (1U << 20) |
          (uint32_t(tail) << 21) | (1U << 28) | (1U << 29) | (owner << 31);
      buffer.SetValue(base, control);
      buffer.SetValue(base + 10, static_cast<uint32_t>(remote));
      buffer.SetValue(base + 11, static_cast<uint32_t>(remote >> 32));
      buffer.SetValue(base + 12, bytes);
      buffer.SetValue(base + 14, static_cast<uint32_t>(local));
      buffer.SetValue(base + 15, static_cast<uint32_t>(local >> 32));
      ++wrs;
    }
    set_flag(PIPE_S, PIPE_MTE3, static_cast<event_t>(0));
    wait_flag(PIPE_S, PIPE_MTE3, static_cast<event_t>(0));
    const auto head = static_cast<uint32_t>(pi % e.sq_depth);
    const auto first = wrs < e.sq_depth - head ? wrs : e.sq_depth - head;
    DataCopyPad(sq[head * 64], encoded,
                DataCopyExtParams{1, first * 64, 0, 0, 0});
    if (first < wrs)
      DataCopyPad(sq, encoded[first * 64],
                  DataCopyExtParams{1, (wrs - first) * 64, 0, 0, 0});
    set_flag(PIPE_MTE3, PIPE_S, static_cast<event_t>(0));
    wait_flag(PIPE_MTE3, PIPE_S, static_cast<event_t>(0));
    state->sq_producer_index = pi + wrs;
    // Timestamp publication, before waiting for its CQE. CPU can already
    // start payload DMA while core 0 retires this control completion.
    if (region == regions && profile && !suffix) profile->notify_cycle = GetSystemCycle();
    st_dev(static_cast<uint32_t>(pi + wrs), reinterpret_cast<__gm__ uint32_t*>(e.sq_db_va), 0);
    const auto ci = state->cq_consumer_index;
    const auto cqe = e.cq_va + (ci % e.cq_depth) * e.cqe_size;
    uint64_t result = KVAF_SEQ_TIMEOUT;
    for (uint32_t poll = 0; poll < a.max_polls; ++poll) {
      Flush(cqe, 32);
      const auto word = *reinterpret_cast<__gm__ volatile uint32_t*>(cqe);
      if (((word >> 2) & 1) == ((ci / e.cq_depth) & 1)) continue;
      ++state->cq_consumer_index;
      st_dev(static_cast<uint32_t>(ci + 1) & 0x003fffffU,
             reinterpret_cast<__gm__ uint32_t*>(e.cq_db_va), 0);
      result = (word & 0xffff0000U) ? KVAF_SEQ_CQ_ERROR : KVAF_SEQ_OK;
      break;
    }
    if (result) {
      state->last_error = result; state->poisoned = 1;
      Flush(reinterpret_cast<uint64_t>(state), sizeof(*state));
      return result;
    }
  }
  ++state->completed_batch_count;
  Flush(reinterpret_cast<uint64_t>(state), sizeof(*state));
  return KVAF_SEQ_OK;
}
}  // namespace kvaf_scatter
