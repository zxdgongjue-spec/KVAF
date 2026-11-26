#pragma once
// USER_CTL WRITE encoding and deferred tail-CQE retirement from aiv-scatter.
#include "../include/aiv_mailbox_abi.h"
__aicore__ inline uint64_t WaitMailboxCqe(const MailboxLaunch& mailbox,
    const __gm__ KvafBatchGetDirectUrmaEntity* e, __gm__ KvafBatchGetQueueState* state) {
  const auto ci = state->cq_consumer_index;
  const auto cqe = e->cq_va + (ci % e->cq_depth) * e->cqe_size;
  uint64_t result = uint64_t(Result::kTimeout);
  for (uint32_t poll = 0; poll < 300000000U; ++poll) {
    Flush(cqe, 32);
    const auto word = *reinterpret_cast<__gm__ volatile uint32_t*>(cqe);
    if (((word >> 2) & 1) == ((ci / e->cq_depth) & 1)) continue;
    ++state->cq_consumer_index;
    st_dev(static_cast<uint32_t>(ci + 1) & 0x003fffffU,
           reinterpret_cast<__gm__ uint32_t*>(e->cq_db_va), 0);
    result = (word & 0xffff0000U) ? uint64_t(Result::kTransportError) : 0U;
    break;
  }
  if (result) {
    state->last_error = result;
    state->poisoned = 1;
    Flush(reinterpret_cast<uint64_t>(state), sizeof(*state));
  }
  return result;
}

__aicore__ inline void MarkMailboxPhaseComplete(__gm__ KvafBatchGetQueueState* state) {
  ++state->completed_batch_count;
  Flush(reinterpret_cast<uint64_t>(state), sizeof(*state));
}

__aicore__ inline uint64_t RetireDeferredMailbox(const MailboxLaunch& mailbox) {
  auto* context = reinterpret_cast<__gm__ AivMailboxContext*>(mailbox.aiv_context);
  auto* state = reinterpret_cast<__gm__ KvafBatchGetQueueState*>(mailbox.aiv_context + offsetof(AivMailboxContext, queue));
  const auto result = WaitMailboxCqe(mailbox, &context->entity, state);
  if (result == 0U) MarkMailboxPhaseComplete(state);
  return result;
}

__aicore__ inline uint64_t PostMailboxRegions(const MailboxLaunch& mailbox, const KernelArgs& a,
    const uint64_t* local_offsets, const uint64_t* remote_offsets, const uint64_t* lengths,
    uint32_t regions, bool head_request, __gm__ volatile BatchGetWorkerProfile* profile = nullptr) {
  auto* context = reinterpret_cast<__gm__ AivMailboxContext*>(mailbox.aiv_context);
  auto* state = reinterpret_cast<__gm__ KvafBatchGetQueueState*>(mailbox.aiv_context + offsetof(AivMailboxContext, queue));
  if (!context || !state) return uint64_t(Result::kTransportError);
  // Templates are read by DMA, not Scalar cache.
  Flush(reinterpret_cast<uint64_t>(context), offsetof(AivMailboxContext, write_templates));
  const auto& e = context->entity;
  if (e.magic != 0x4b564155U || e.version != 1 || !e.sq_va || !e.sq_db_va ||
      !e.cq_va || !e.cq_db_va || e.sq_depth < 4 || e.sq_depth % 4 ||
      !e.cq_depth || e.cqe_size < 32 || e.wqe_size != 64 || !e.local_token_id ||
      !context->max_write_bytes || context->max_write_bytes > 65536)
    return uint64_t(Result::kTransportError);
  if (state->initialized && (state->magic != 0x4b564151U || state->poisoned))
    return uint64_t(Result::kTransportError);
  if (!state->initialized) {
    state->magic = 0x4b564151U; state->initialized = 1;
    state->sq_producer_index = state->cq_consumer_index = 0;
  }
  const auto max_write_bytes = context->max_write_bytes;
  uint32_t region = 0;
  uint64_t offset = 0;
  const auto limit = e.sq_depth / 4 < 16U ?
      e.sq_depth / 4 : 16U;
  LocalTensor<uint32_t> buffer(TPosition::VECIN, 0, 16 * 16);
  LocalTensor<uint8_t> encoded(TPosition::VECIN, 0, 16 * 64);
  GlobalTensor<uint8_t> sq;
  sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(e.sq_va));
  GlobalTensor<uint32_t> templates;
  templates.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
      reinterpret_cast<uint64_t>(context) + offsetof(AivMailboxContext, write_templates)));
  while (region < regions) {
    if (!lengths[region]) { ++region; continue; }
    // One DMA replaces per-WQE Scalar reconstruction of all invariant words.
    DataCopy(buffer, templates, 16U * 16);
    set_flag(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
    wait_flag(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
    const auto pi = state->sq_producer_index;
    uint32_t wrs = 0;
    while (wrs < limit && region < regions) {
      const auto remaining = lengths[region] - offset;
      const auto bytes = remaining < max_write_bytes ? static_cast<uint32_t>(remaining) : max_write_bytes;
      const auto local = a.workspace + local_offsets[region] + offset;
      const auto remote = e.remote_host_ddr_address + remote_offsets[region] + offset;
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
    const bool last = region == regions;
    st_dev(static_cast<uint32_t>(pi + wrs), reinterpret_cast<__gm__ uint32_t*>(e.sq_db_va), 0);
    if (last && head_request && profile)
      profile->head_doorbell_ns = GetSystemCycle() - profile->enter_ns;
    if (last) return 0U;
    const auto result = WaitMailboxCqe(mailbox, &context->entity, state);
    if (result) return result;
  }
  return 0U;
}

__aicore__ inline uint64_t PostMailbox(const MailboxLaunch& mailbox, const KernelArgs& a,
    uint64_t descriptor_offset, uint64_t target_offset, uint32_t count, bool suffix, bool failed = false,
    __gm__ volatile BatchGetWorkerProfile* profile = nullptr) {
  const uint32_t head = count < kPipelineHeadRows ? count : kPipelineHeadRows;
  const uint64_t offsets[4] = {suffix ? descriptor_offset + uint64_t(head) * 8 : 0,
      suffix ? kSuffixNotifyOffset : target_offset, descriptor_offset, 64};
  const uint64_t lengths[4] = {suffix ? (failed ? 0U : uint64_t(count - head) * 8) : 64,
      suffix ? 8U : sizeof(OutputTarget), uint64_t(head) * 8, 8};
  return PostMailboxRegions(mailbox, a, offsets, offsets, lengths, suffix ? 2 : 4, !suffix, profile);
}

__aicore__ inline uint64_t PostChunkMailbox(const MailboxLaunch& mailbox, const KernelArgs& a,
    uint64_t descriptor_offset, uint64_t target_offset, uint32_t count, uint32_t chunk,
    uint32_t owner, bool first, bool failed, __gm__ volatile BatchGetWorkerProfile* profile) {
  const auto staging = (descriptor_offset + uint64_t(kMaxRequests) * 8 + 63) & ~uint64_t(63);
  const auto slot = failed ? kPushChunkCount + owner : chunk;
  const auto commit = staging + (failed ? uint64_t(slot) * sizeof(Word) : PushCommitOffset(chunk));
  const auto head = PushLaunchHeadRows(a.sequence);
  const auto begin = PushChunkBegin(chunk, head);
  const auto descriptor = descriptor_offset + uint64_t(begin) * 8;
  const auto rows = PushChunkEnd(count, chunk, head) - begin;
  const uint64_t local[5] = {0, target_offset, descriptor, commit, 64};
  const uint64_t remote[5] = {0, target_offset, descriptor, 192 + uint64_t(owner) * 64, 64};
  const uint64_t lengths[5] = {64, sizeof(OutputTarget), failed ? 0U : uint64_t(rows) * 8, 8, 8};
  if (failed) return PostMailboxRegions(mailbox, a, local + 3, remote + 3, lengths + 3, 1, false);
  return PostMailboxRegions(mailbox, a, local + (first ? 0 : 2), remote + (first ? 0 : 2),
      lengths + (first ? 0 : 2), first ? 5 : 2, first, profile);
}
