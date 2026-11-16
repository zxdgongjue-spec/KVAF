#include "cpu_gather_executor.h"

#include <exception>
#include <ctime>
#include <stdexcept>

namespace kvaf::peer {
namespace {

std::uint64_t ThreadCpuNs() {
#if defined(CLOCK_THREAD_CPUTIME_ID)
  timespec value{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0)
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000U + value.tv_nsec;
#endif
  return 0U;
}

bool ValidateEnvelope(std::uint64_t base, std::uint64_t bytes, std::uint64_t descriptor_address,
                      const GatherDescriptorHeader& header) {
  if (bytes < 64U || base > UINT64_MAX - bytes || header.magic != kGatherDescriptorMagic ||
      header.span_count == 0U ||
      header.ready_bytes != 0U ||
      (header.pipeline_chunk_bytes != 0U &&
       (header.pipeline_chunk_bytes < 64U || header.pipeline_chunk_bytes % 64U != 0U)))
    return false;
  const auto data_bytes = GatherControlOffset(bytes);
  if (!GatherRange(base, data_bytes, descriptor_address, GatherDescriptorBytes(header.span_count)) ||
      !GatherRange(base, data_bytes, header.staging_address, header.total_bytes))
    return false;
  return descriptor_address >= header.staging_address + header.total_bytes ||
         header.staging_address >= descriptor_address + GatherDescriptorBytes(header.span_count);
}

}  // namespace

std::size_t CpuGatherExecutor::PrepareDirect(std::uint64_t region_bytes,
                                         std::uint64_t descriptor_address, std::uint64_t sequence,
                                         std::size_t max_spans) {
  if (region_bytes < 64U || base_ > UINT64_MAX - region_bytes || descriptor_address % 64U != 0U ||
      !GatherRange(base_, GatherControlOffset(region_bytes), descriptor_address,
                   sizeof(GatherDescriptorHeader)))
    return 0U;
  const auto* descriptor = memory_ + (descriptor_address - base_);
  GatherDescriptorHeader header;
  std::memcpy(&header, descriptor, sizeof(header));
  if (header.sequence != sequence || header.span_count > max_spans ||
      !ValidateEnvelope(base_, region_bytes, descriptor_address, header))
    return 0U;
  const auto* source_spans = reinterpret_cast<const GatherSpan*>(descriptor + sizeof(header));
  span_bytes_ = header.span_bytes;
  if (span_bytes_ != 0U) {
    // BatchGet has fixed-size tokens: locate any staging byte by division/remainder.
    if (static_cast<std::uint64_t>(span_bytes_) * header.span_count != header.total_bytes)
      return 0U;
  } else {
    // Older variable-length descriptors still need a prefix index, but no snapshot.
    variable_offsets_.resize(header.span_count + 1U);
    variable_offsets_[0] = 0U;
    for (std::size_t index = 0; index < header.span_count; ++index) {
      const auto length = source_spans[index].length;
      const auto end = variable_offsets_[index] + length;
      if (length == 0U || end > header.total_bytes) return 0U;
      variable_offsets_[index + 1U] = end;
    }
    if (variable_offsets_.back() != header.total_bytes) return 0U;
  }
  fixed_progress_.clear();
  fixed_shards_.clear();
  fixed_first_shard_wave_ = SIZE_MAX;
  fixed_jetty_supply_ = false;
  pipeline_ready_ = nullptr;
  if (header.pipeline_chunk_bytes != 0U) {
    pipeline_chunk_bytes_ = header.pipeline_chunk_bytes;
    const auto chunks = header.total_bytes / pipeline_chunk_bytes_ +
                        (header.total_bytes % pipeline_chunk_bytes_ != 0U);
    chunk_done_.assign(chunks, 0U);
    next_chunk_.store(0U, std::memory_order_relaxed);
    pipeline_ready_ = reinterpret_cast<std::uint64_t*>(
        memory_ + (descriptor_address - base_) + offsetof(GatherDescriptorHeader, ready_bytes));
  }
  return PrepareJob(memory_ + (header.staging_address - base_), source_spans, nullptr,
                    header.span_count, header.total_bytes);
}

std::size_t CpuGatherExecutor::PrepareFixed(void* staging, const std::uint64_t* addresses,
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
  fixed_shards_.resize(first_shard_wave == SIZE_MAX ? 0U : waves.size() - first_shard_wave);
  fixed_last_source_offset_ = source_bytes - span_bytes;
  fixed_suffix_ = suffix ? *suffix : FixedSuffix{};
  fixed_error_.store(FixedError::kNone, std::memory_order_relaxed);
  span_bytes_ = span_bytes;
  pipeline_ready_ = nullptr;
  return PrepareJob(staging, nullptr, addresses, count, std::uint64_t{count} * span_bytes);
}

void CpuGatherExecutor::Copy(const WaveReady& on_wave, const ShardProgress& on_shards) {
  completed_.store(0U, std::memory_order_relaxed);
  if (profiling_enabled_) {
    copy_profile_.begin_ns = NowNs();
    thread_profiles_[0].ready_ns = copy_profile_.begin_ns;
  }
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
  BeginWorker(0U);
  std::uint64_t copied = 0U;
  std::exception_ptr publish_error;
  std::size_t next_ready = 0;
  bool publishing = true;
  const auto whole_waves = std::min(fixed_progress_.size(), fixed_first_shard_wave_);
  const auto publish_ready = [&]() {
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
      }
    } catch (...) {
      publish_error = std::current_exception();
      publishing = false;
      SetFixedError(FixedError::kCancelled);
    }
  };
  if (fixed_progress_.empty()) {
    copied = ProcessSource(0U);
  } else {
    for (std::size_t wave = 0; wave < fixed_progress_.size(); ++wave) {
      if (fixed_error_.load(std::memory_order_relaxed) != FixedError::kNone) break;
      if (fixed_suffix_.ready_sequence && wave == fixed_suffix_.first_wave) {
        const auto begin = profiling_enabled_ &&
            __atomic_load_n(fixed_suffix_.ready_sequence, __ATOMIC_ACQUIRE) != fixed_suffix_.sequence
            ? NowNs() : 0U;
        // Keep posting completed head waves if suffix has yet to arrive.
        const bool ready = WaitFixedSuffix(publish_ready);
        if (begin) copy_profile_.suffix_wait_ns = NowNs() - begin;
        if (!ready) break;
      }
      copied += CopyFixedWave(0U, wave, publish_ready);
      publish_ready();
    }
  }
  FinishWorker(0U, copied);
  if (profiling_enabled_) copy_profile_.coordinator_done_ns = thread_profiles_[0].done_ns;
  while (completed_.load(std::memory_order_acquire) != active_threads_ - 1U) {
    publish_ready();
    GatherSpinPause();
  }
  publish_ready();
  if (profiling_enabled_) copy_profile_.done_ns = NowNs();
  if (publish_error) std::rethrow_exception(publish_error);
  if (addresses_) {
    const auto error = fixed_error_.load(std::memory_order_relaxed);
    if (error == FixedError::kSource)
      throw std::invalid_argument("CPU gather source outside registered DDR");
    if (error == FixedError::kSuffix)
      throw std::runtime_error("CPU gather descriptor suffix wait failed");
  }
  if (pipeline_ready_ != nullptr) __atomic_store_n(pipeline_ready_, total_bytes_, __ATOMIC_RELEASE);
}

// Copy only this thread's staging range, combining adjacent source spans into one memcpy.
std::uint64_t CpuGatherExecutor::CopyRange(std::uint64_t offset, std::uint64_t end) const {
  if (offset >= end) return 0U;
  const auto begin = offset;
  const auto index_for_offset = span_bytes_ != 0U
      ? offset / span_bytes_
      : static_cast<std::uint64_t>(std::upper_bound(variable_offsets_.begin(),
            variable_offsets_.end(), offset) - variable_offsets_.begin() - 1U);
  std::size_t index = static_cast<std::size_t>(index_for_offset);
  auto in_span = span_bytes_ != 0U ? offset % span_bytes_ : offset - variable_offsets_[index];
  while (offset < end) {
    const auto source_address = spans_[index].address + in_span;
    const auto length = span_bytes_ != 0U ? span_bytes_ : spans_[index].length;
    auto bytes = std::min<std::uint64_t>(length - in_span, end - offset);
    ++index;
    in_span = 0U;
    while (bytes < end - offset && index < span_count_ &&
           spans_[index].address == source_address + bytes) {
      const auto next_length = span_bytes_ != 0U ? span_bytes_ : spans_[index].length;
      bytes += std::min<std::uint64_t>(next_length, end - offset - bytes);
      ++index;
    }
    const auto* source = memory_ + (source_address - base_);
    std::memcpy(staging_ + offset, source, bytes);
    offset += bytes;
  }
  return offset - begin;
}

std::uint64_t CpuGatherExecutor::CopyFixedRange(std::uint64_t begin, std::uint64_t end) {
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
bool CpuGatherExecutor::WaitFixedSuffix(OnWait on_wait) {
  const auto& suffix = fixed_suffix_;
  auto observed = __atomic_load_n(suffix.ready_sequence, __ATOMIC_ACQUIRE);
  if (observed == suffix.sequence) return true;
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
      fixed_error_.load(std::memory_order_relaxed) == FixedError::kNone)
    return true;
  SetFixedError(FixedError::kSuffix);
  return false;
}

void CpuGatherExecutor::PrepareShardProgress() {
  const auto queues = fixed_jetty_count_;
  fixed_jetty_supply_ &= active_threads_ >= queues;
  for (std::size_t index = 0; index < fixed_shards_.size(); ++index) {
    const auto& wave = fixed_progress_[fixed_first_shard_wave_ + index];
    auto& shards = fixed_shards_[index];
    shards = {};
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

bool CpuGatherExecutor::ShardReady(std::size_t wave, std::size_t queue) const {
  if (wave < fixed_first_shard_wave_ || wave >= fixed_progress_.size() ||
      queue >= fixed_jetty_count_)
    return false;
  const auto rows = fixed_progress_[wave].rows[queue];
  return rows.begin < rows.end &&
         !__atomic_load_n(&fixed_shards_[wave - fixed_first_shard_wave_][queue].remaining,
                          __ATOMIC_ACQUIRE) &&
         fixed_error_.load(std::memory_order_relaxed) == FixedError::kNone;
}

template <typename OnProgress>
std::uint64_t CpuGatherExecutor::CopyFixedWave(std::size_t worker, std::size_t index,
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
    auto& remaining = fixed_shards_[index - fixed_first_shard_wave_][q].remaining;
    __atomic_fetch_sub(&remaining, 1U, __ATOMIC_ACQ_REL);
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
        auto& remaining = fixed_shards_[index - fixed_first_shard_wave_][queue].remaining;
        // Acquire the previous writers' stores before making this shard ready.
        __atomic_fetch_sub(&remaining, 1U, __ATOMIC_ACQ_REL);
        on_progress();
        if (fixed_error_.load(std::memory_order_relaxed) != FixedError::kNone) break;
      }
    }
    offset += cpu_push::Align64(bytes);
    if (offset >= last) break;
  }
  if (!shard_mode) {
    if (copied) GatherBarrier();
    // The last decrement carries every writer's completed stores to the owner.
    __atomic_fetch_sub(&wave.remaining, 1U, __ATOMIC_ACQ_REL);
  }
  return copied;
}

std::uint64_t CpuGatherExecutor::ProcessSource(std::size_t worker) {
  if (!fixed_progress_.empty()) {
    std::uint64_t copied = 0U;
    for (std::size_t wave = 0; wave < fixed_progress_.size(); ++wave) {
      if (fixed_error_.load(std::memory_order_relaxed) != FixedError::kNone) break;
      if (fixed_suffix_.ready_sequence && wave == fixed_suffix_.first_wave &&
          !WaitFixedSuffix([] {})) break;
      copied += CopyFixedWave(worker, wave, [] {});
    }
    return copied;
  }
  if (pipeline_ready_ == nullptr) {
    const auto begin = Boundary(worker, 0U, total_bytes_, active_threads_);
    const auto end = Boundary(worker + 1U, 0U, total_bytes_, active_threads_);
    return addresses_ ? CopyFixedRange(begin, end) : CopyRange(begin, end);
  }
  std::uint64_t copied = 0U;
  for (;;) {
    const auto chunk = next_chunk_.fetch_add(1U, std::memory_order_relaxed);
    if (chunk >= chunk_done_.size()) return copied;
    const auto begin = static_cast<std::uint64_t>(chunk) * pipeline_chunk_bytes_;
    copied += CopyRange(begin, std::min(begin + pipeline_chunk_bytes_, total_bytes_));
    GatherBarrier();
    __atomic_store_n(&chunk_done_[chunk], 1U, __ATOMIC_RELEASE);
    auto observed = __atomic_load_n(pipeline_ready_, __ATOMIC_ACQUIRE);
    auto prefix = static_cast<std::size_t>(observed / pipeline_chunk_bytes_);
    while (prefix < chunk_done_.size() &&
           __atomic_load_n(&chunk_done_[prefix], __ATOMIC_ACQUIRE) != 0U)
      ++prefix;
    const auto ready_bytes =
        std::min(total_bytes_, static_cast<std::uint64_t>(prefix) * pipeline_chunk_bytes_);
    while (observed < ready_bytes) {
      if (!__atomic_compare_exchange_n(pipeline_ready_, &observed, ready_bytes, false,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        continue;
      GatherBarrier();
      break;
    }
  }
}

void CpuGatherExecutor::BeginWorker(std::size_t worker) {
  if (!profiling_enabled_) return;
  auto& profile = thread_profiles_[worker];
#if defined(__linux__)
  profile.tid = static_cast<std::uint64_t>(syscall(SYS_gettid));
  profile.cpu = sched_getcpu();
#endif
  profile.thread_cpu_ns = ThreadCpuNs();
  profile.start_ns = NowNs();
}

void CpuGatherExecutor::FinishWorker(std::size_t worker, std::uint64_t bytes) {
  if (profiling_enabled_) {
    thread_profiles_[worker].bytes = bytes;
    thread_profiles_[worker].copy_done_ns = NowNs();
  }
  // Chunk writers have already completed their stores before advertising progress.
  if (fixed_progress_.empty()) GatherBarrier();
  if (!profiling_enabled_) return;
  auto& profile = thread_profiles_[worker];
  profile.done_ns = NowNs();
  const auto cpu_end = ThreadCpuNs();
  const auto cpu_begin = profile.thread_cpu_ns;
  profile.cpu_time_valid = cpu_begin && cpu_end >= cpu_begin;
  profile.thread_cpu_ns = profile.cpu_time_valid ? cpu_end - cpu_begin : 0U;
}

void CpuGatherExecutor::RunHelper(std::size_t worker) {
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
    if (profiling_enabled_) {
      thread_profiles_[worker].ready_ns = NowNs();
      thread_profiles_[worker].wait_path = wait_path;
    }
    BeginWorker(worker);
    const auto bytes = ProcessSource(worker);
    FinishWorker(worker, bytes);
    completed_.fetch_add(1U, std::memory_order_release);
  }
}

}  // namespace kvaf::peer
