#pragma once

#include "status.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace kvaf::cpu_urma_detail {
// CQ context encodes slot, generation and queue. Handles belong to stable
// prepared plans; no allocation, task scan or hash lookup is needed on a CQE.
class IndexedAsyncWriteTracker {
 public:
  struct Fence { std::size_t queue{}, wrs{}; std::uint64_t context{}; };
  static constexpr std::size_t kMaxQueues = 16, kMaxBatches = 64;
  struct Handle { std::uint64_t token{}; };
  explicit IndexedAsyncWriteTracker(std::size_t queues = 0, std::size_t depth = 8192)
      : queues_(queues), depth_(depth) {}
  Status TryBegin(std::uint64_t id, Handle& handle, Fence* fences,
                  std::size_t count, bool* submitted) {
    if (!submitted) return Status::InvalidArgument("missing submission result");
    *submitted = false;
    if (!id || !fences || !count || count > queues_ || queues_ > kMaxQueues)
      return Status::InvalidArgument("invalid indexed async fences");
    if (handle.token) return Status::InvalidState("prepared WRITE is already in flight");
    std::uint32_t mask = 0;
    bool capacity = active_ != UINT64_MAX;
    for (std::size_t i = 0; i < count; ++i) {
      const auto& f = fences[i];
      if (f.queue >= queues_ || !f.wrs || f.wrs > depth_ || (mask & (1U << f.queue)))
        return Status::InvalidArgument("invalid/duplicate indexed async fence");
      mask |= 1U << f.queue;
      capacity &= f.wrs <= depth_ - used_[f.queue];
    }
    if (!capacity) return Status::OK();  // All queue credits reserved atomically.
    if (generation_ == kGenerationMask)
      return Status::InvalidState("async completion generation exhausted; reinitialize transport");
    const auto slot = FirstBit(~active_);
    const auto token = kTag | (++generation_ << 10) | (std::uint64_t(slot) << 4);
    auto& batch = batches_[slot];
    batch.id = id; batch.mask = mask; batch.handle = &handle;
    handle.token = token;
    for (std::size_t i = 0; i < count; ++i) {
      auto& f = fences[i];
      f.context = token | f.queue;
      batch.wrs[f.queue] = f.wrs; used_[f.queue] += f.wrs;
      ++pending_[f.queue];
    }
    active_ |= UINT64_C(1) << slot; queues_mask_ |= mask;
    *submitted = true;
    return Status::OK();
  }
  Status Complete(std::size_t queue, std::uint64_t context, std::uint64_t* completed) {
    if (!completed || queue >= queues_ || queues_ > kMaxQueues)
      return Status::InvalidArgument("invalid indexed CQ queue");
    *completed = 0;
    const auto slot = static_cast<std::size_t>((context >> 4) & 63);
    const auto bit = 1U << queue;
    auto& batch = batches_[slot];
    if ((context & kTagMask) != kTag || (context & 15) != queue ||
        !(active_ & (UINT64_C(1) << slot)) || !(batch.mask & bit) ||
        (context & ~UINT64_C(15)) != batch.handle->token)
      return Status::Internal("unexpected/stale indexed async completion context");
    used_[queue] -= batch.wrs[queue]; batch.mask &= ~bit;
    if (!--pending_[queue]) queues_mask_ &= ~bit;
    if (!batch.mask) {
      *completed = batch.id; batch.handle->token = 0;
      active_ &= ~(UINT64_C(1) << slot);
    }
    return Status::OK();
  }
  bool Empty() const { return active_ == 0; }
  std::uint32_t QueueMask() const { return queues_mask_; }
  std::size_t Pending(std::size_t queue) const { return pending_[queue]; }
  template <class Visit> void ForEachActive(Visit visit) const {
    for (auto bits = active_; bits; bits &= bits - 1) visit(batches_[FirstBit(bits)].id);
  }
 private:
  static unsigned FirstBit(std::uint64_t bits) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_ctzll(bits));
#else
    unsigned bit = 0; while (!(bits & 1)) { bits >>= 1; ++bit; } return bit;
#endif
  }
  static constexpr std::uint64_t kTag = UINT64_C(0xA700000000000000);
  static constexpr std::uint64_t kTagMask = UINT64_C(0xFF00000000000000);
  static constexpr std::uint64_t kGenerationMask = (UINT64_C(1) << 46) - 1;
  struct Batch {
    std::uint64_t id{}; std::uint32_t mask{}; Handle* handle{};
    std::array<std::size_t, kMaxQueues> wrs{};
  };
  std::size_t queues_, depth_;
  std::uint64_t active_{}, generation_{};
  std::uint32_t queues_mask_{};
  std::array<std::size_t, kMaxQueues> used_{}, pending_{};
  std::array<Batch, kMaxBatches> batches_{};
};

// One bounded read per active queue, rotating the first queue each sweep.
// A CQE can finish at most one plan. Limit reads by output space so even an
// out-of-order burst cannot consume completions that the caller cannot hold.
// read(queue,limit,&polled); complete(queue,index,&finished) own CR storage.
template <class Pending, class Read, class Complete>
Status SweepAsyncCqs(std::size_t queues, std::size_t first, std::size_t batch_size,
                    std::size_t capacity, std::size_t* count,
                    Pending pending, Read read, Complete complete) {
  if (!queues || !batch_size || !capacity || !count)
    return Status::InvalidArgument("invalid async CQ sweep");
  *count = 0;
  for (std::size_t i = 0, q = first % queues; i < queues && *count < capacity;
       ++i, q = q + 1 == queues ? 0 : q + 1) {
    const auto limit = std::min({batch_size, capacity - *count, pending(q)});
    if (!limit) continue;
    std::size_t polled = 0;
    auto status = read(q, limit, &polled);
    if (!status.ok()) return status;
    if (polled > limit) return Status::Internal("CQ returned more entries than requested");
    for (std::size_t j = 0; j < polled; ++j) {
      bool finished = false;
      status = complete(q, j, &finished);
      if (!status.ok()) return status;
      if (finished) ++*count;
    }
  }
  return Status::OK();
}
}  // namespace kvaf::cpu_urma_detail
