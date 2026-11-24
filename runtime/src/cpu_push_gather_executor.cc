#include "cpu_push_gather_executor.h"

#include <exception>
#include <stdexcept>

namespace kvaf::peer {

std::size_t CpuPushGatherExecutor::PrepareFixed(void* staging, const std::uint64_t* addresses,
    std::uint32_t count, std::uint32_t span_bytes, std::uint64_t source_bytes,
    const std::vector<cpu_push::GatherWave>& waves, const FixedSuffix* suffix,
    std::size_t first_shard_wave, bool jetty_supply, std::uint32_t jettys) {
  if (!count || !span_bytes || span_bytes > source_bytes || base_ > UINT64_MAX - source_bytes ||
      !jettys || jettys > cpu_push::kMaxJettyCount)
    return 0U;
  if ((!waves.empty() && reinterpret_cast<std::uintptr_t>(staging) % 64U) ||
      (suffix && (!suffix->ready_sequence || suffix->first_wave >= waves.size())) ||
      (first_shard_wave != SIZE_MAX && first_shard_wave >= waves.size()))
    return 0U;
  fixed_progress_.resize(waves.size());
  std::uint64_t planned_rows = 0U;
  for (std::size_t i = 0; i < fixed_progress_.size(); ++i) {
    auto& wave = fixed_progress_[i];
    if (i >= first_shard_wave) wave.first_shard = (i - first_shard_wave) * jettys;
    wave.aligned_bytes = 0U;
    wave.rows = waves[i];
    for (std::uint32_t q = 0; q < jettys; ++q) {
      const auto rows = wave.rows[q];
      if (rows.begin == rows.end) continue;
      if (rows.begin > rows.end || rows.end > count ||
          std::uint64_t{rows.begin} * span_bytes % 64U ||
          (rows.end != count && std::uint64_t{rows.end} * span_bytes % 64U))
        return 0U;
      const auto row_count = rows.end - rows.begin;
      planned_rows += row_count;
      if (planned_rows > count) return 0U;
      wave.aligned_bytes += cpu_push::Align64(std::uint64_t{row_count} * span_bytes);
    }
    if (!wave.aligned_bytes) return 0U;
  }
  if (!waves.empty() && planned_rows != count) return 0U;
  fixed_first_shard_wave_ = first_shard_wave;
  fixed_jetty_supply_ = jetty_supply && first_shard_wave != SIZE_MAX;
  fixed_jetty_count_ = jettys;
  const auto shard_waves = first_shard_wave == SIZE_MAX ? 0U : waves.size() - first_shard_wave;
  fixed_shards_.resize(shard_waves * jettys);
  fixed_last_source_offset_ = source_bytes - span_bytes;
  fixed_suffix_ = suffix ? *suffix : FixedSuffix{};
  fixed_error_.store(FixedError::kNone, std::memory_order_relaxed);
  span_bytes_ = span_bytes;
  return PrepareJob(staging, addresses, std::uint64_t{count} * span_bytes);
}

void CpuPushGatherExecutor::DispatchJob() {
  if (active_threads_ > 1U) {
    {
      // Use the same mutex as the sleeping helpers so they cannot miss the
      // notification. Helpers already spinning acquire the job via dispatch_.
      std::lock_guard<std::mutex> lock(mutex_);
      const auto previous = dispatch_.load(std::memory_order_relaxed);
      const auto generation = (previous & ~kDispatchThreadMask) + kDispatchGenerationStep;
      dispatch_.store(generation | active_threads_, std::memory_order_release);
    }
    ready_.notify_all();
  }
}

void CpuPushGatherExecutor::Copy(const WaveReady& on_wave, const ShardProgress& on_shards) {
  completed_.store(0U, std::memory_order_relaxed);
  DispatchJob();
  std::exception_ptr publish_error;
  std::size_t next_ready = 0;
  bool publishing = true;
  const auto whole_waves = std::min(fixed_progress_.size(), fixed_first_shard_wave_);
  const auto publish_ready = [&] {
    if (!publishing) return;
    try {
      while (next_ready < whole_waves &&
             !__atomic_load_n(&fixed_progress_[next_ready].remaining, __ATOMIC_ACQUIRE) &&
             fixed_error_.load(std::memory_order_relaxed) == FixedError::kNone) {
        if (on_wave && !on_wave(next_ready + 1U)) {
          publishing = false;
          SetFixedError(FixedError::kCancelled);
          return;
        }
        ++next_ready;
      }
      if (on_shards && next_ready == whole_waves &&
          fixed_error_.load(std::memory_order_relaxed) == FixedError::kNone && !on_shards()) {
        publishing = false;
        SetFixedError(FixedError::kCancelled);
        return;
      }
    } catch (...) {
      publish_error = std::current_exception();
      publishing = false;
      SetFixedError(FixedError::kCancelled);
    }
  };
  if (fixed_progress_.empty()) {
    ProcessSource(0U);
    GatherBarrier();
  } else {
    for (std::size_t wave = 0; wave < fixed_progress_.size(); ++wave) {
      if (fixed_error_.load(std::memory_order_relaxed) != FixedError::kNone) break;
      if (fixed_suffix_.ready_sequence && wave == fixed_suffix_.first_wave &&
          !WaitFixedSuffix(publish_ready)) break;
      CopyFixedWave(0U, wave, publish_ready);
      publish_ready();
    }
  }
  while (completed_.load(std::memory_order_acquire) != active_threads_ - 1U) {
    publish_ready();
    GatherSpinPause();
  }
  publish_ready();
  if (publish_error) std::rethrow_exception(publish_error);
  const auto error = fixed_error_.load(std::memory_order_relaxed);
  if (error == FixedError::kSource)
    throw std::invalid_argument("CPU gather source outside registered DDR");
  if (error == FixedError::kSuffix)
    throw std::runtime_error("CPU gather descriptor suffix wait failed");
}

std::uint64_t CpuPushGatherExecutor::CopyFixedRange(std::uint64_t begin, std::uint64_t end) {
  if (begin == end) return 0U;
  const auto span_bytes = span_bytes_;
  const auto last_source = fixed_last_source_offset_;
  auto row = begin / span_bytes;
  auto in_span = begin % span_bytes;
  auto offset = begin;
  auto source = addresses_[row] - base_;
  while (offset < end) {
    // Check the full token even when a worker owns only part of it. Unsigned
    // subtraction also rejects addresses below the registered base.
    if (source > last_source) {
      SetFixedError(FixedError::kSource);
      return offset - begin;
    }
    const auto source_begin = source + in_span;
    const auto remaining = end - offset;
    auto bytes = std::min(remaining, span_bytes - in_span);
    in_span = 0U;
    while (bytes < remaining) {
      source = addresses_[++row] - base_;
      if (source != source_begin + bytes) break;
      if (source > last_source) {
        SetFixedError(FixedError::kSource);
        return offset - begin;
      }
      bytes += std::min<std::uint64_t>(span_bytes, remaining - bytes);
    }
    std::memcpy(staging_ + offset, memory_ + source_begin, bytes);
    offset += bytes;
  }
  return offset - begin;
}

template <typename OnWait>
bool CpuPushGatherExecutor::WaitFixedSuffix(OnWait on_wait) {
  const auto& suffix = fixed_suffix_;
  auto observed = __atomic_load_n(suffix.ready_sequence, __ATOMIC_ACQUIRE);
  if (observed == suffix.sequence) {
    return true;
  }
  const auto deadline = std::chrono::steady_clock::now() + suffix.timeout;
  std::uint32_t polls = 0;
  do {
    if (observed > suffix.sequence ||
        fixed_error_.load(std::memory_order_relaxed) != FixedError::kNone)
      break;
    if ((++polls & 31U) == 0U &&
        ((suffix.stop && suffix.stop->load(std::memory_order_acquire)) ||
         std::chrono::steady_clock::now() >= deadline))
      break;
    on_wait();
    GatherSpinPause();
    observed = __atomic_load_n(suffix.ready_sequence, __ATOMIC_ACQUIRE);
  } while (observed != suffix.sequence);
  if (observed == suffix.sequence &&
      fixed_error_.load(std::memory_order_relaxed) == FixedError::kNone) {
    return true;
  }
  SetFixedError(FixedError::kSuffix);
  return false;
}

void CpuPushGatherExecutor::PrepareShardProgress() {
  const auto queues = fixed_jetty_count_;
  fixed_jetty_supply_ &= active_threads_ >= queues;
  for (auto& shard : fixed_shards_) {
    shard.remaining = 0;
  }
  for (std::size_t index = fixed_first_shard_wave_; index < fixed_progress_.size(); ++index) {
    const auto& wave = fixed_progress_[index];
    auto* shards = fixed_shards_.data() + wave.first_shard;
    if (fixed_jetty_supply_) {
      for (std::size_t q = 0; q < queues; ++q) {
        const auto rows = wave.rows[q];
        const auto bytes = std::uint64_t{rows.end - rows.begin} * span_bytes_;
        const auto threads = active_threads_ / queues + (q < active_threads_ % queues);
        std::uint64_t first = 0;
        for (std::size_t i = 1; i <= threads && first < bytes; ++i) {
          const auto last = Boundary(i, 0U, bytes, threads);
          shards[q].remaining += first < last;
          first = last;
        }
      }
      continue;
    }
    std::array<std::uint64_t, cpu_push::kMaxJettyCount> starts{}, ends{};
    std::uint64_t offset = 0U;
    for (std::size_t queue = 0; queue < queues; ++queue) {
      const auto rows = wave.rows[queue];
      const auto bytes = std::uint64_t{rows.end - rows.begin} * span_bytes_;
      starts[queue] = offset;
      ends[queue] = offset + bytes;
      offset += cpu_push::Align64(bytes);
    }
    std::uint64_t first = 0U;
    for (std::size_t worker = 1; worker <= active_threads_; ++worker) {
      const auto last = Boundary(worker, 0U, wave.aligned_bytes, active_threads_);
      for (std::size_t queue = 0; queue < queues; ++queue) {
        if (std::max(first, starts[queue]) < std::min(last, ends[queue]))
          ++shards[queue].remaining;
      }
      first = last;
    }
  }
}

bool CpuPushGatherExecutor::ShardReady(std::size_t wave, std::size_t queue) const {
  if (wave < fixed_first_shard_wave_ || wave >= fixed_progress_.size() ||
      queue >= fixed_jetty_count_)
    return false;
  const auto& progress = fixed_progress_[wave];
  const auto rows = progress.rows[queue];
  return rows.begin < rows.end &&
         !__atomic_load_n(&fixed_shards_[progress.first_shard + queue].remaining, __ATOMIC_ACQUIRE) &&
         fixed_error_.load(std::memory_order_relaxed) == FixedError::kNone;
}

void CpuPushGatherExecutor::FinishFixedRange(std::size_t& remaining) {
  __atomic_fetch_sub(&remaining, 1U, __ATOMIC_ACQ_REL);
}

template <typename OnProgress>
std::uint64_t CpuPushGatherExecutor::CopyFixedWave(std::size_t worker, std::size_t index,
                                             OnProgress on_progress) {
  auto& wave = fixed_progress_[index];
  const bool shard_mode = index >= fixed_first_shard_wave_;
  if (shard_mode && fixed_jetty_supply_) {
    // Each group advances through its Jetty's shards independently, even
    // when another group is still copying the preceding wave.
    const auto queues = fixed_jetty_count_;
    const auto q = worker % queues;
    const auto rank = worker / queues;
    const auto threads = active_threads_ / queues + (q < active_threads_ % queues);
    const auto rows = wave.rows[q];
    const auto base = std::uint64_t{rows.begin} * span_bytes_;
    const auto bytes = std::uint64_t{rows.end - rows.begin} * span_bytes_;
    const auto first = Boundary(rank, 0U, bytes, threads);
    const auto last = Boundary(rank + 1U, 0U, bytes, threads);
    if (first == last) return 0;
    const auto copied = CopyFixedRange(base + first, base + last);
    if (copied) GatherBarrier();
    auto& shard = fixed_shards_[wave.first_shard + q];
    FinishFixedRange(shard.remaining);
    // The coordinator calls publish_ready after returning from this slice.
    return copied;
  }
  const auto first = Boundary(worker, 0U, wave.aligned_bytes, active_threads_);
  const auto last = Boundary(worker + 1U, 0U, wave.aligned_bytes, active_threads_);
  std::uint64_t copied = 0U;
  std::uint64_t offset = 0U;
  // Divide the combined wave evenly, then translate each worker's slice
  // back to the original staging addresses. Pad a short final range in
  // this calculation only, so the next range still starts on a cache line.
  for (std::size_t queue = 0; queue < fixed_jetty_count_; ++queue) {
    const auto rows = wave.rows[queue];
    if (rows.begin == rows.end) continue;
    const auto bytes = std::uint64_t{rows.end - rows.begin} * span_bytes_;
    const auto begin = std::max(first, offset);
    const auto end = std::min(last, offset + bytes);
    if (begin < end) {
      const auto staging_offset = std::uint64_t{rows.begin} * span_bytes_;
      const auto bytes_copied =
          CopyFixedRange(staging_offset + (begin - offset), staging_offset + (end - offset));
      copied += bytes_copied;
      if (shard_mode) {
        if (bytes_copied) GatherBarrier();
        auto& shard = fixed_shards_[wave.first_shard + queue];
        FinishFixedRange(shard.remaining);
        on_progress();
        if (fixed_error_.load(std::memory_order_relaxed) != FixedError::kNone) break;
      }
    }
    offset += cpu_push::Align64(bytes);
    if (offset >= last) break;
  }
  if (!shard_mode) {
    if (copied) GatherBarrier();
    FinishFixedRange(wave.remaining);
  }
  return copied;
}

std::uint64_t CpuPushGatherExecutor::ProcessSource(std::size_t worker) {
  if (fixed_progress_.empty()) {
    return CopyFixedRange(Boundary(worker, 0U, total_bytes_, active_threads_),
                          Boundary(worker + 1U, 0U, total_bytes_, active_threads_));
  }
  std::uint64_t copied = 0U;
  for (std::size_t wave = 0; wave < fixed_progress_.size(); ++wave) {
    if (fixed_error_.load(std::memory_order_relaxed) != FixedError::kNone) break;
    if (fixed_suffix_.ready_sequence && wave == fixed_suffix_.first_wave &&
        !WaitFixedSuffix([] {})) break;
    copied += CopyFixedWave(worker, wave, [] {});
  }
  return copied;
}

void CpuPushGatherExecutor::RunHelper(std::size_t worker) {
  BindWorker(worker);
  std::uint64_t previous = 0U;
  bool was_active = false;
  for (;;) {
    auto dispatch = dispatch_.load(std::memory_order_acquire);
    if (wait_mode_ != GatherWaitMode::kPassive && was_active && dispatch == previous) {
      const auto spin_start = std::chrono::steady_clock::now();
      std::uint32_t spin_count = 0U;
      while (dispatch == previous) {
        if (stopping_.load(std::memory_order_acquire)) return;
        if (wait_mode_ == GatherWaitMode::kAdaptive && (++spin_count & 255U) == 0U &&
            std::chrono::steady_clock::now() - spin_start >= std::chrono::microseconds(100))
          break;
        GatherSpinPause();
        dispatch = dispatch_.load(std::memory_order_acquire);
      }
    }
    const bool wait_path = wait_mode_ == GatherWaitMode::kPassive || dispatch == previous;
    if (wait_path) {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [this, previous] {
        return stopping_.load(std::memory_order_acquire) ||
               dispatch_.load(std::memory_order_acquire) != previous;
      });
      dispatch = dispatch_.load(std::memory_order_acquire);
    }
    if (stopping_.load(std::memory_order_acquire)) return;
    previous = dispatch;
    was_active = worker < (dispatch & kDispatchThreadMask);
    if (!was_active) continue;
    ProcessSource(worker);
    if (fixed_progress_.empty()) GatherBarrier();
    completed_.fetch_add(1U, std::memory_order_release);
  }
}

}  // namespace kvaf::peer
