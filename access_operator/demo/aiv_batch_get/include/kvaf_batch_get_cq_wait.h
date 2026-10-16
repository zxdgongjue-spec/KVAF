#ifndef KVAF_BATCH_GET_CQ_WAIT_H_
#define KVAF_BATCH_GET_CQ_WAIT_H_

#include "kernel_operator.h"
#include "batch_get_abi.h"

namespace kvaf_get_cq {
using namespace AscendC;

constexpr uint32_t kFastPolls = 20000U;
constexpr uint32_t kSlowPolls = 1000000U;
// The A5 harness assumes 1GHz: 100M raw ticks then mean 100ms. Report raw
// ticks and verify the target frequency when interpreting elapsed time. The
// independent poll cap also bounds simulators or a stalled device clock.
constexpr uint64_t kSlowDeadlineTicks = 100000000ULL;
constexpr uint32_t kConsumerMask = 0x003fffffU;

__aicore__ inline uint32_t Probe(__gm__ uint32_t *address) {
  // The transport writes CQ memory independently. ld_dev bypasses scalar
  // L1; invalidating then immediately reloading cached memory is unnecessary.
  __asm__ __volatile__("" ::: "memory");
  const uint32_t word = AscendC::ReadGmByPassDCache<uint32_t>(address);
  __asm__ __volatile__("" ::: "memory");
  return word;
}

template<class Queue>
__aicore__ inline void Trace(__gm__ Queue *queue, uint32_t word, uint32_t expected,
                             uint64_t elapsed, uint32_t polls, uint32_t flags) {
  // Queue indices use bypass stores too: never dirty their shared cache line.
  AscendC::WriteGmByPassDCache<uint64_t>(&queue->reserved[0],
      uint64_t(word) | (uint64_t(expected) << 32U));
  AscendC::WriteGmByPassDCache<uint64_t>(&queue->reserved[1], elapsed);
  AscendC::WriteGmByPassDCache<uint64_t>(&queue->reserved[2], uint64_t(polls) |
      (uint64_t(KVAF_BATCH_GET_CQ_TRACE_TAG | flags) << 32U));
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

// Exactly one completion is consumed per posted terminal fence. Exhausting
// the cheap fast loop starts a bounded wait; it does not resubmit any READ,
// alter routing, or declare healthy-but-delayed transport permanently broken.
template<class Entity, class Queue>
__aicore__ inline uint64_t WaitForCompletion(
    const __gm__ Entity *entity, __gm__ Queue *queue, uint32_t *poll_count) {
  const uint64_t consumer = AscendC::ReadGmByPassDCache<uint64_t>(
      &queue->cq_consumer_index);
  auto *address = reinterpret_cast<__gm__ uint32_t *>(entity->cq_va +
      (consumer % entity->cq_depth) * entity->cqe_size);
  const uint32_t expected = static_cast<uint32_t>((consumer / entity->cq_depth) & 1U);
  uint32_t word = 0U, polls = 0U;
  bool ready = false;
  for (; polls < kFastPolls; ++polls) {
    word = Probe(address);
    if (((word >> 2U) & 1U) != expected) {
      ++polls;
      ready = true;
      break;
    }
  }
  uint32_t flags = KVAF_BATCH_GET_CQ_TRACE_BYPASS_PROBE;
  uint64_t elapsed = 0U;
  if (!ready) {
    flags |= KVAF_BATCH_GET_CQ_TRACE_SLOW_ENTERED;
    const uint64_t started = static_cast<uint64_t>(AscendC::GetSystemCycle());
    for (uint32_t slow = 0U; slow < kSlowPolls; ++slow) {
      elapsed = static_cast<uint64_t>(AscendC::GetSystemCycle()) - started;
      if (elapsed >= kSlowDeadlineTicks) {
        flags |= KVAF_BATCH_GET_CQ_TRACE_DEADLINE;
        break;
      }
      word = Probe(address);
      ++polls;
      if (((word >> 2U) & 1U) != expected) {
        ready = true;
        break;
      }
    }
    elapsed = static_cast<uint64_t>(AscendC::GetSystemCycle()) - started;
    if (!ready && !(flags & KVAF_BATCH_GET_CQ_TRACE_DEADLINE))
      flags |= KVAF_BATCH_GET_CQ_TRACE_POLL_LIMIT;
  }
  *poll_count = polls;
  if (!ready) {
    Trace(queue, word, expected, elapsed, polls, flags | KVAF_BATCH_GET_CQ_TRACE_TIMEOUT);
    return 6U;
  }
  const bool success = (word & 0xffff0000U) == 0U;
  const uint64_t next = consumer + 1U;
  AscendC::WriteGmByPassDCache<uint64_t>(&queue->cq_consumer_index, next);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  st_dev(static_cast<uint32_t>(next) & kConsumerMask,
      reinterpret_cast<__gm__ uint32_t *>(entity->cq_db_va), 0);
  if (!success) flags |= KVAF_BATCH_GET_CQ_TRACE_ERROR_CQE;
  if (success && (flags & KVAF_BATCH_GET_CQ_TRACE_SLOW_ENTERED))
    flags |= KVAF_BATCH_GET_CQ_TRACE_LATE_SUCCESS;
  if (!success || (flags & KVAF_BATCH_GET_CQ_TRACE_SLOW_ENTERED))
    Trace(queue, word, expected, elapsed, polls, flags);
  return success ? 0U : 7U;
}

}  // namespace kvaf_get_cq
#endif  // KVAF_BATCH_GET_CQ_WAIT_H_
