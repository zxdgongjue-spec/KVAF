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
                      const GatherDescriptorHeader& header, const GatherChunkConfig& config) {
  const bool configured = header.magic == kGatherConfiguredMagic;
  const bool bootstrap = header.magic == kGatherBootstrapMagic;
  const bool cooperative = configured || bootstrap || header.magic == kGatherCooperativeMagic;
  const auto first = configured ? config.first_chunk_divisor : bootstrap ? kGatherBootstrapDivisor : 0;
  const auto tails = configured ? config.tail_chunks : bootstrap ? kGatherBootstrapTailChunks : 0;
  if (bytes < 64U || base > UINT64_MAX - bytes ||
      (!cooperative && header.magic != kGatherDescriptorMagic) ||
      header.span_count == 0U ||
      header.ready_bytes != 0U ||
      (configured && (config.reserved || !first || !GatherCooperativeLayoutValid(tails, first))) ||
      ((configured || bootstrap) && header.pipeline_chunk_bytes != GatherCooperativeChunkBytes(
          header.total_bytes, tails, first)) ||
      (cooperative && (!header.pipeline_chunk_bytes ||
       GatherProgressChunkCount(header.total_bytes, header.pipeline_chunk_bytes,
           first) > (first ? tails + 1 : kGatherMaxCooperativeChunks))) ||
      (header.pipeline_chunk_bytes != 0U &&
       (header.pipeline_chunk_bytes < 64U || header.pipeline_chunk_bytes % 64U != 0U)))
    return false;
  const auto data_bytes = GatherControlOffset(bytes);
  const auto descriptor_bytes = GatherDescriptorBytes(header.span_count, configured);
  if (!GatherRange(base, data_bytes, descriptor_address, descriptor_bytes) ||
      !GatherRange(base, data_bytes, header.staging_address, header.total_bytes))
    return false;
  return descriptor_address >= header.staging_address + header.total_bytes ||
         header.staging_address >= descriptor_address + descriptor_bytes;
}

}  // namespace

std::size_t CpuGatherExecutor::PrepareDirect(std::uint64_t region_bytes,
                                         std::uint64_t descriptor_address, std::uint64_t sequence,
                                         std::size_t max_spans, const GatherControl* control,
                                         const std::atomic<bool>* stop) {
  if (region_bytes < 64U || base_ > UINT64_MAX - region_bytes || descriptor_address % 64U != 0U ||
      !GatherRange(base_, GatherControlOffset(region_bytes), descriptor_address,
                   sizeof(GatherDescriptorHeader)))
    return 0U;
  const auto* descriptor = memory_ + (descriptor_address - base_);
  GatherDescriptorHeader header;
  std::memcpy(&header, descriptor, sizeof(header));
  const bool head_start = (header.magic & kGatherHeadStartFlag) != 0;
  header.magic &= ~kGatherHeadStartFlag;
  if (head_start && (!control || !header.span_bytes || header.span_count <= kGatherHeadRows))
    return 0U;
  const bool configured = header.magic == kGatherConfiguredMagic;
  GatherChunkConfig config{};
  if (configured) {
    if (!GatherRange(base_, GatherControlOffset(region_bytes), descriptor_address,
                     GatherPrefixHeaderBytes(true))) return 0;
    std::memcpy(&config, descriptor + sizeof(header), sizeof(config));
  }
  if (header.sequence != sequence || header.span_count > max_spans ||
      !ValidateEnvelope(base_, region_bytes, descriptor_address, header, config))
    return 0U;
  const auto* source_spans = reinterpret_cast<const GatherSpan*>(
      descriptor + GatherPrefixHeaderBytes(configured));
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
  first_chunk_divisor_ = configured ? config.first_chunk_divisor :
      header.magic == kGatherBootstrapMagic ? kGatherBootstrapDivisor : 0;
  cooperative_ = first_chunk_divisor_ || header.magic == kGatherCooperativeMagic;
  if (header.pipeline_chunk_bytes != 0U) {
    pipeline_chunk_bytes_ = header.pipeline_chunk_bytes;
    pipeline_ready_ = reinterpret_cast<std::uint64_t*>(
        memory_ + (descriptor_address - base_) + offsetof(GatherDescriptorHeader, ready_bytes));
  }
  descriptor_suffix_ = head_start ? &control->suffix : nullptr;
  request_stop_ = stop;
  sequence_ = sequence;
  head_bytes_ = std::uint64_t{kGatherHeadRows} * span_bytes_;
  copy_failed_.store(false, std::memory_order_relaxed);
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
  pipeline_ready_ = nullptr;
  cooperative_ = false;
  descriptor_suffix_ = nullptr;
  request_stop_ = nullptr;
  return PrepareJob(staging, nullptr, addresses, count, std::uint64_t{count} * span_bytes);
}

void CpuGatherExecutor::DispatchJob() {
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

void CpuGatherExecutor::Copy(const WaveReady& on_wave, const ShardProgress& on_shards) {
  completed_.store(0U, std::memory_order_relaxed);
  if (profiling_enabled_) {
    copy_profile_.begin_ns = NowNs();
    thread_profiles_[0].ready_ns = copy_profile_.begin_ns;
  }
  if (addresses_) DispatchJob();
  else phase_.store(1U, std::memory_order_release);
  BeginWorker(0U);
  if (!addresses_) {
    const auto copied = ProcessSource(0U);
    FinishWorker(0U, copied);
    if (profiling_enabled_) copy_profile_.coordinator_done_ns = thread_profiles_[0].done_ns;
    while (completed_.load(std::memory_order_acquire) != active_threads_ - 1U) {
      PublishReady();
      GatherSpinPause();
    }
    if (copy_failed_.load(std::memory_order_relaxed))
      throw std::runtime_error("gather descriptor suffix wait failed");
    PublishReady();
    if (profiling_enabled_) copy_profile_.done_ns = NowNs();
    return;
  }
  std::uint64_t copied = 0U;
  std::exception_ptr publish_error;
  std::size_t next_ready = 0;
  bool publishing = true;
  const auto whole_waves = std::min(fixed_progress_.size(), fixed_first_shard_wave_);
  const auto publish_ready = [&] {
    if (!publishing) return;
    try {
      while (next_ready < whole_waves &&
             !__atomic_load_n(&fixed_progress_[next_ready].remaining, __ATOMIC_ACQUIRE) &&
             (!trace_ready_ || FixedReadyNs(next_ready, 0U, false)) &&
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

bool CpuGatherExecutor::WaitSuffix() const {
  if (__atomic_load_n(descriptor_suffix_, __ATOMIC_ACQUIRE) == sequence_) return true;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  std::uint32_t polls = 0;
  for (;;) {
    const auto suffix = __atomic_load_n(descriptor_suffix_, __ATOMIC_ACQUIRE);
    if (suffix == sequence_) return true;
    if (suffix > sequence_ || copy_failed_.load(std::memory_order_relaxed) ||
        ((++polls & 255U) == 0U &&
         (stopping_.load(std::memory_order_acquire) ||
          (request_stop_ && request_stop_->load(std::memory_order_acquire)) ||
          std::chrono::steady_clock::now() >= deadline))) {
      copy_failed_.store(true, std::memory_order_relaxed);
      return false;
    }
    GatherSpinPause();
  }
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
  auto ready_end = descriptor_suffix_ ? std::min(end, head_bytes_) : end;
  while (offset < end) {
    if (offset >= ready_end) {
      if (!WaitSuffix()) return offset - begin;
      ready_end = end;
    }
    const auto source_address = spans_[index].address + in_span;
    const auto length = span_bytes_ != 0U ? span_bytes_ : spans_[index].length;
    auto bytes = std::min<std::uint64_t>(length - in_span, ready_end - offset);
    ++index;
    in_span = 0U;
    while (bytes < ready_end - offset && index < span_count_ &&
           spans_[index].address == source_address + bytes) {
      const auto next_length = span_bytes_ != 0U ? span_bytes_ : spans_[index].length;
      bytes += std::min<std::uint64_t>(next_length, ready_end - offset - bytes);
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

void CpuGatherExecutor::RecordSuffixSeen() {
  if (!trace_ready_ || __atomic_load_n(&copy_profile_.suffix_seen_ns, __ATOMIC_RELAXED)) return;
  std::uint64_t expected = 0;
  __atomic_compare_exchange_n(&copy_profile_.suffix_seen_ns, &expected, NowNs(), false,
                              __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}

template <typename OnWait>
bool CpuGatherExecutor::WaitFixedSuffix(OnWait on_wait) {
  const auto& suffix = fixed_suffix_;
  auto observed = __atomic_load_n(suffix.ready_sequence, __ATOMIC_ACQUIRE);
  if (observed == suffix.sequence) {
    RecordSuffixSeen();
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
    RecordSuffixSeen();
    return true;
  }
  SetFixedError(FixedError::kSuffix);
  return false;
}

void CpuGatherExecutor::PrepareShardProgress() {
  const auto queues = fixed_jetty_count_;
  fixed_jetty_supply_ &= active_threads_ >= queues;
  for (auto& shard : fixed_shards_) {
    shard.remaining = 0;
    shard.ready_ns = 0;
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

bool CpuGatherExecutor::ShardReady(std::size_t wave, std::size_t queue) const {
  if (wave < fixed_first_shard_wave_ || wave >= fixed_progress_.size() ||
      queue >= fixed_jetty_count_)
    return false;
  const auto& progress = fixed_progress_[wave];
  const auto rows = progress.rows[queue];
  return rows.begin < rows.end &&
         !__atomic_load_n(&fixed_shards_[progress.first_shard + queue].remaining, __ATOMIC_ACQUIRE) &&
         (!trace_ready_ || FixedReadyNs(wave, queue, true)) &&
         fixed_error_.load(std::memory_order_relaxed) == FixedError::kNone;
}

std::uint64_t CpuGatherExecutor::FixedReadyNs(std::size_t wave, std::size_t queue, bool shard) const {
  const auto& progress = fixed_progress_[wave];
  const auto& ns = shard ? fixed_shards_[progress.first_shard + queue].ready_ns : progress.ready_ns;
  return __atomic_load_n(&ns, __ATOMIC_ACQUIRE);
}

void CpuGatherExecutor::FinishFixedRange(std::size_t& remaining, std::uint64_t& ready_ns) {
  // The final decrement acquires the preceding writers' stores. A traced
  // consumer also waits for that writer's timestamp before recording its post.
  if (__atomic_fetch_sub(&remaining, 1U, __ATOMIC_ACQ_REL) == 1U && trace_ready_)
    __atomic_store_n(&ready_ns, NowNs(), __ATOMIC_RELEASE);
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
    auto& shard = fixed_shards_[wave.first_shard + q];
    FinishFixedRange(shard.remaining, shard.ready_ns);
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
        FinishFixedRange(shard.remaining, shard.ready_ns);
        on_progress();
        if (fixed_error_.load(std::memory_order_relaxed) != FixedError::kNone) break;
      }
    }
    offset += cpu_push::Align64(bytes);
    if (offset >= last) break;
  }
  if (!shard_mode) {
    if (copied) GatherBarrier();
    FinishFixedRange(wave.remaining, wave.ready_ns);
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
  if (cooperative_) {
    std::uint64_t copied = 0;
    for (std::uint32_t chunk = 0; chunk < cooperative_count_; ++chunk) {
      const auto begin = chunk ? GatherProgressChunkEnd(
          total_bytes_, pipeline_chunk_bytes_, first_chunk_divisor_, chunk - 1) : 0;
      const auto end = GatherProgressChunkEnd(total_bytes_, pipeline_chunk_bytes_, first_chunk_divisor_, chunk);
      // Partition aligned cache lines inside EACH prefix block, including a
      // short tail. Empty shares still participate in block completion.
      const auto lines = (end - begin + 63U) / 64U;
      const auto local_begin = std::min(end, begin + lines * worker / active_threads_ * 64U);
      const auto local_end = std::min(end, begin + lines * (worker + 1U) / active_threads_ * 64U);
      copied += CopyRange(local_begin, local_end);
      if (copy_failed_.load(std::memory_order_relaxed)) return copied;
      if (local_begin != local_end) GatherBarrier();
      auto& progress = cooperative_progress_[chunk];
      // The acquire chain collects every writer's completed payload stores.
      if (progress.remaining.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
        progress.complete.store(true, std::memory_order_release);
        PublishCooperativeReady();
      }
      if (profiling_enabled_) ++thread_profiles_[worker].progress_publications;
    }
    return copied;
  }
  if (pipeline_ready_ == nullptr) {
    const auto begin = Boundary(worker, 0U, total_bytes_, active_threads_);
    const auto end = Boundary(worker + 1U, 0U, total_bytes_, active_threads_);
    return addresses_ ? CopyFixedRange(begin, end) : CopyRange(begin, end);
  }
  std::uint64_t copied = 0U;
  for (auto task = worker; task < task_progress_.size(); task += active_threads_) {
    const auto begin = static_cast<std::uint64_t>(task) * task_bytes_;
    const auto end = std::min(begin + task_bytes_, total_bytes_);
    for (auto offset = begin; offset < end;) {
      // Checkpoint at global staging chunk boundaries, not at task-local
      // multiples: a chunk may span two balanced tasks and two CPU threads.
      const auto boundary = (offset / pipeline_chunk_bytes_ + 1U) * pipeline_chunk_bytes_;
      const auto next = std::min(end, boundary);
      copied += CopyRange(offset, next);
      if (copy_failed_.load(std::memory_order_relaxed)) return copied;
      // This writer must finish its payload stores before advertising partial
      // progress. The task's remaining bytes can then be copied independently.
      GatherBarrier();
      __atomic_store_n(&task_progress_[task].bytes, next - begin, __ATOMIC_RELEASE);
      if (worker == 0U) PublishReady();
      offset = next;
    }
  }
  return copied;
}

void CpuGatherExecutor::PublishReady() {
  if (pipeline_ready_ == nullptr || cooperative_) return;
  auto copied = std::min(total_bytes_, static_cast<std::uint64_t>(ready_tasks_) * task_bytes_);
  while (ready_tasks_ < task_progress_.size()) {
    const auto begin = static_cast<std::uint64_t>(ready_tasks_) * task_bytes_;
    const auto end = std::min(begin + task_bytes_, total_bytes_);
    const auto progress = __atomic_load_n(&task_progress_[ready_tasks_].bytes, __ATOMIC_ACQUIRE);
    copied = begin + progress;
    if (copied < end) break;
    ++ready_tasks_;
  }
  const auto ready = copied == total_bytes_ ? copied : copied / pipeline_chunk_bytes_ * pipeline_chunk_bytes_;
  if (ready > published_bytes_) {
    __atomic_store_n(pipeline_ready_, ready, __ATOMIC_RELEASE);
    GatherBarrier();
    published_bytes_ = ready;
  }
}

void CpuGatherExecutor::PublishCooperativeReady() {
  std::uint32_t completed = 0;
  while (completed < cooperative_count_ &&
         cooperative_progress_[completed].complete.load(std::memory_order_acquire))
    ++completed;
  const auto ready = completed ? GatherProgressChunkEnd(
      total_bytes_, pipeline_chunk_bytes_, first_chunk_divisor_, completed - 1) : 0;
  auto previous = __atomic_load_n(pipeline_ready_, __ATOMIC_ACQUIRE);
  // Only completed-block finishers publish per batch. Concurrent finishers may
  // coalesce prefixes, but a delayed earlier finisher must never regress ready.
  while (previous < ready) {
    if (__atomic_compare_exchange_n(pipeline_ready_, &previous, ready, false,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
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
  if (fixed_progress_.empty() && pipeline_ready_ == nullptr) GatherBarrier();
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
    if (!addresses_) {
      while (phase_.load(std::memory_order_acquire) == 0U) {
        if (stopping_.load(std::memory_order_acquire)) return;
        GatherSpinPause();
      }
    }
    BeginWorker(worker);
    const auto bytes = ProcessSource(worker);
    FinishWorker(worker, bytes);
    completed_.fetch_add(1U, std::memory_order_release);
  }
}

}  // namespace kvaf::peer
