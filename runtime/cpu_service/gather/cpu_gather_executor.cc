#include "cpu_gather_executor.h"

#include <ctime>

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

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

bool IsPullGroupFormat(std::uint64_t magic) {
  return magic == kGatherPullPlanMagic || magic == kGatherChunkedPullMagic;
}

bool ValidateEnvelope(std::uint64_t base, std::uint64_t bytes, std::uint64_t descriptor_address,
                      const GatherDescriptorHeader& header, bool extended) {
  if (header.span_count == 0U || header.ready_bytes != 0U ||
      header.span_count == 0U ||
      header.ready_bytes != 0U ||
      (header.pipeline_chunk_bytes != 0U &&
       (header.pipeline_chunk_bytes < 64U || header.pipeline_chunk_bytes % 64U != 0U)))
    return false;
  const auto data_bytes = GatherControlOffset(bytes);
  const auto descriptor_bytes = GatherDescriptorBytes(header.span_count, extended);
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
  if (region_bytes < 64U || workspace_base_ > UINT64_MAX - region_bytes || descriptor_address % 64U != 0U ||
      !GatherRange(workspace_base_, GatherControlOffset(region_bytes), descriptor_address,
                   sizeof(GatherDescriptorHeader)))
    return 0U;
  const auto* descriptor = workspace_memory_ + (descriptor_address - workspace_base_);
  GatherDescriptorHeader header;
  std::memcpy(&header, descriptor, sizeof(header));
  if (read_mode_ == GatherReadMode::kAicpu && header.magic != kGatherDescriptorMagic)
    return 0U;
  const bool aiv = read_mode_ == GatherReadMode::kAiv;
  const bool head_start = aiv && (header.magic & kGatherHeadStartFlag) != 0;
  if (head_start) header.magic &= ~kGatherHeadStartFlag;
  const bool configured = aiv && header.magic == kGatherConfiguredMagic;
  const bool bootstrap = aiv && header.magic == kGatherBootstrapMagic;
  cooperative_ = configured || bootstrap || (aiv && header.magic == kGatherCooperativeMagic);
  const bool sized = !aiv && header.magic == kGatherSizedChunksMagic;
  const bool head256 = !aiv && header.magic == kGatherHead256Magic;
  chunked_addresses_ = !aiv && header.magic == kGatherChunkedPullMagic;
  pull_groups_ = !aiv && IsPullGroupFormat(header.magic);
  const bool extended = configured || sized || head256 || pull_groups_;
  if (header.sequence != sequence || header.span_count > max_spans ||
      (header.magic != kGatherDescriptorMagic && !cooperative_ && !extended) ||
      !ValidateEnvelope(workspace_base_, region_bytes, descriptor_address, header, extended))
    return 0U;

  if (pull_groups_) {
    if (!control || !sequence || !header.span_bytes) return 0U;
    std::memcpy(&group_plan_, descriptor + sizeof(header), sizeof(group_plan_));
    const bool valid = chunked_addresses_ ? group_plan_.ValidChunked() : group_plan_.Valid();
    if (!valid || group_plan_.total_rows != header.span_count ||
        group_plan_.token_bytes != header.span_bytes) return 0U;
  }
  first_chunk_bytes_ = 0U;
  if (sized) {
    GatherChunkConfig config{};
    std::memcpy(&config, descriptor + sizeof(header), sizeof(config));
    if (!config.valid() || !config.enabled() || !header.span_bytes ||
        header.pipeline_chunk_bytes != config.Tail(header.total_bytes)) return 0U;
    first_chunk_bytes_ = config.First(header.total_bytes);
  }
  AivGatherChunkConfig config{};
  if (configured) {
    std::memcpy(&config, descriptor + sizeof(header), sizeof(config));
    if (config.reserved || !config.first_chunk_divisor ||
        !GatherCooperativeLayoutValid(config.tail_chunks, config.first_chunk_divisor))
      return 0U;
  }
  first_chunk_divisor_ = configured ? config.first_chunk_divisor : bootstrap ? kGatherBootstrapDivisor : 0U;
  const auto tails = configured ? config.tail_chunks : bootstrap ? kGatherBootstrapTailChunks : 0U;
  if ((first_chunk_divisor_ && header.pipeline_chunk_bytes !=
          GatherCooperativeChunkBytes(header.total_bytes, tails, first_chunk_divisor_)) ||
      (cooperative_ && (!header.pipeline_chunk_bytes ||
          GatherProgressChunkCount(header.total_bytes, header.pipeline_chunk_bytes, first_chunk_divisor_) >
              (first_chunk_divisor_ ? tails + 1U : kGatherMaxCooperativeChunks))))
    return 0U;

  const auto head_rows = pull_groups_ ? group_plan_.head_rows : aiv ? kvaf::kGatherHeadRows : kGatherHeadRows;
  const bool early = pull_groups_ ? head_rows < header.span_count :
      head_start || head256 || (sized && header.span_count > head_rows);
  if (early && (!control || !sequence || !header.span_bytes ||
                (!pull_groups_ && (!aiv && !header.pipeline_chunk_bytes)) ||
                head_rows >= header.span_count))
    return 0U;
  const auto* source_spans = reinterpret_cast<const GatherSpan*>(
      descriptor + GatherPrefixHeaderBytes(extended));
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
  pipeline_ready_ = nullptr;
  pipeline_chunk_bytes_ = header.pipeline_chunk_bytes;
  if (pull_groups_ || pipeline_chunk_bytes_ != 0U) {
    pipeline_ready_ = reinterpret_cast<std::uint64_t*>(
        workspace_memory_ + (descriptor_address - workspace_base_) + offsetof(GatherDescriptorHeader, ready_bytes));
  }
  suffix_sequence_ = early ? &control->suffix_sequence : nullptr;
  sequence_ = sequence;
  request_stop_ = stop;
  head_bytes_ = std::uint64_t{head_rows} * span_bytes_;
  suffix_failed_.store(false, std::memory_order_relaxed);
  return Publish(header.staging_address, source_spans, header.span_count, header.total_bytes);
}

bool CpuGatherExecutor::Copy() {
  completed_.store(0U, std::memory_order_relaxed);
  if (!thread_profiles_.empty()) {
    copy_profile_.begin_ns = NowNs();
    thread_profiles_[0].ready_ns = copy_profile_.begin_ns;
  }
  phase_.store(1U, std::memory_order_release);
  CopyWorker(0U);
  if (!thread_profiles_.empty())
    copy_profile_.coordinator_done_ns = thread_profiles_[0].done_ns;
  const bool publish_tasks = read_mode_ == GatherReadMode::kAiv && pipeline_ready_ && !cooperative_;
  while (completed_.load(std::memory_order_acquire) != active_threads_ - 1U) {
    if (publish_tasks) PublishTaskReady();
    GatherSpinPause();
  }
  const bool ok = pull_groups_ ? !GroupStopped() : !suffix_failed_.load(std::memory_order_relaxed);
  if (ok) {
    if (publish_tasks) PublishTaskReady();
    if (pipeline_ready_ && !pull_groups_ && read_mode_ == GatherReadMode::kAicpu)
      __atomic_store_n(pipeline_ready_, total_bytes_, __ATOMIC_RELEASE);
  }
  if (!thread_profiles_.empty()) copy_profile_.done_ns = NowNs();
  return ok;
}

void CpuGatherExecutor::CopyWorker(std::size_t worker) {
  const bool finish_barrier = pipeline_ready_ == nullptr || read_mode_ == GatherReadMode::kAicpu;
  if (thread_profiles_.empty()) {
    (void)ProcessSource(worker);
    if (finish_barrier) GatherBarrier();
    return;
  }
  auto& profile = thread_profiles_[worker];
#if defined(__linux__)
  profile.tid = static_cast<std::uint64_t>(syscall(SYS_gettid));
  profile.cpu = sched_getcpu();
#endif
  const auto cpu_begin = ThreadCpuNs();
  profile.start_ns = NowNs();
  profile.bytes = ProcessSource(worker);
  profile.copy_done_ns = NowNs();
  if (finish_barrier) GatherBarrier();
  profile.done_ns = NowNs();
  const auto cpu_end = ThreadCpuNs();
  profile.cpu_time_valid = cpu_begin != 0U && cpu_end >= cpu_begin;
  profile.thread_cpu_ns = profile.cpu_time_valid ? cpu_end - cpu_begin : 0U;
}

void FinishGatherCopy(GatherControl& control, std::uint64_t sequence, bool success, GatherDescriptorHeader* header) {
  const bool terminal = header && IsPullGroupFormat(header->magic);
  const auto final_progress = success && terminal ? std::uint64_t{header->total_bytes} : kGatherErrorBit;
  if (!success && header && !terminal)
    __atomic_store_n(&header->ready_bytes, kGatherErrorBit, __ATOMIC_RELEASE);
  __atomic_store_n(&control.ack, sequence | (success ? 0U : kGatherErrorBit), __ATOMIC_RELEASE);
  GatherBarrier();
  if (terminal) __atomic_store_n(&header->ready_bytes, final_progress, __ATOMIC_RELEASE);
}

bool CpuGatherExecutor::WaitPublishedAddresses(std::uint64_t end) {
  // Chunked requests publish a row count. Legacy requests publish their
  // sequence when the whole suffix is available.
  const auto needed = chunked_addresses_ ? (end + span_bytes_ - 1U) / span_bytes_ : sequence_;
  const auto limit = chunked_addresses_ ? span_count_ : sequence_;
  const auto available = [needed, limit](std::uint64_t value) { return value >= needed && value <= limit; };
  if (available(__atomic_load_n(suffix_sequence_, __ATOMIC_ACQUIRE))) return true;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  for (unsigned polls = 0;; ++polls) {
    const auto observed = __atomic_load_n(suffix_sequence_, __ATOMIC_ACQUIRE);
    if (available(observed)) return true;
    if (observed > limit || suffix_failed_.load(std::memory_order_relaxed) ||
        ((polls & 255U) == 0 && ((request_stop_ && request_stop_->load(std::memory_order_acquire)) ||
                                 std::chrono::steady_clock::now() >= deadline))) {
      suffix_failed_.store(true, std::memory_order_relaxed);
      return false;
    }
    GatherSpinPause();
  }
}

std::uint64_t CpuGatherExecutor::CopyRange(std::uint64_t offset, std::uint64_t end) {
  if (!suffix_sequence_ || end <= head_bytes_ || offset >= end) return CopyPublishedRange(offset, end);
  const auto split = std::max(offset, head_bytes_);
  const auto copied = CopyPublishedRange(offset, split);
  if (!WaitPublishedAddresses(end)) return copied;
  return copied + CopyPublishedRange(split, end);
}

// Copy only this thread's staging range, combining adjacent source spans into one memcpy.
std::uint64_t CpuGatherExecutor::CopyPublishedRange(std::uint64_t offset, std::uint64_t end) const {
  if (offset >= end) return 0U;
  const auto begin = offset;
  const auto index_for_offset = span_bytes_ != 0U
      ? offset / span_bytes_
      : static_cast<std::uint64_t>(std::upper_bound(variable_offsets_.begin(),
            variable_offsets_.end(), offset) - variable_offsets_.begin() - 1U);
  std::size_t index = static_cast<std::size_t>(index_for_offset);
  auto in_span = span_bytes_ != 0U ? offset % span_bytes_ : offset - variable_offsets_[index];
  // Look ahead within this copy range. Adjacent spans still take the coalesced memcpy path.
  constexpr std::size_t lookahead = 6;
  const auto full_spans_end = span_bytes_ ? end / span_bytes_ : 0U;
  while (offset < end) {
    if (index + lookahead < full_spans_end) {
      const auto* next = memory_ + (spans_[index + lookahead].address - base_);
      // Bring the leading lines of a random token into cache while copying previous tokens.
      for (std::uint32_t byte = 0; byte < std::min(span_bytes_, 768U); byte += 128U)
        __builtin_prefetch(next + byte, 0, 3);
    }
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

std::uint64_t CpuGatherExecutor::ProcessSource(std::size_t worker) {
  if (pull_groups_) return CopyGroups(worker);
  if (pipeline_ready_ == nullptr) {
    return CopyRange(Boundary(worker), Boundary(worker + 1U));
  }
  if (read_mode_ == GatherReadMode::kAicpu)
    return first_chunk_bytes_ ? CopySizedChunks(worker) : CopyDynamicChunks();
  if (cooperative_) {
    std::uint64_t copied = 0;
    for (std::uint32_t chunk = 0; chunk < cooperative_count_; ++chunk) {
      const auto begin = chunk ? GatherProgressChunkEnd(
          total_bytes_, pipeline_chunk_bytes_, first_chunk_divisor_, chunk - 1) : 0;
      const auto end = GatherProgressChunkEnd(total_bytes_, pipeline_chunk_bytes_, first_chunk_divisor_, chunk);
      const auto before = copied;
      if (!CopyStripe(worker, begin, end, copied) || suffix_failed_.load(std::memory_order_relaxed))
        return copied;
      if (copied != before) GatherBarrier();
      auto& progress = cooperative_progress_[chunk];
      // The acquire chain collects every writer's completed payload stores.
      if (progress.remaining.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
        progress.complete.store(true, std::memory_order_release);
        PublishCooperativeReady();
      }
      if (!thread_profiles_.empty()) ++thread_profiles_[worker].progress_publications;
    }
    return copied;
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
      if (suffix_failed_.load(std::memory_order_relaxed)) return copied;
      // This writer must finish its payload stores before advertising partial
      // progress. The task's remaining bytes can then be copied independently.
      GatherBarrier();
      __atomic_store_n(&task_progress_[task].bytes, next - begin, __ATOMIC_RELEASE);
      if (worker == 0U) PublishTaskReady();
      offset = next;
    }
  }
  return copied;
}

std::uint64_t CpuGatherExecutor::CopyDynamicChunks() {
  std::uint64_t copied = 0U;
  for (;;) {
    const auto chunk = next_chunk_.fetch_add(1U, std::memory_order_relaxed);
    if (chunk >= chunk_done_.size()) return copied;
    const auto begin = static_cast<std::uint64_t>(chunk) * pipeline_chunk_bytes_;
    const auto end = std::min(begin + pipeline_chunk_bytes_, total_bytes_);
    const auto count = CopyRange(begin, end);
    copied += count;
    if (count != end - begin) return copied;
    GatherBarrier();
    __atomic_store_n(&chunk_done_[chunk], 1U, __ATOMIC_RELEASE);
    const auto observed = __atomic_load_n(pipeline_ready_, __ATOMIC_ACQUIRE);
    auto prefix = static_cast<std::size_t>(observed / pipeline_chunk_bytes_);
    while (prefix < chunk_done_.size() &&
           __atomic_load_n(&chunk_done_[prefix], __ATOMIC_ACQUIRE) != 0U)
      ++prefix;
    AdvanceReadyPrefix(observed, std::min(total_bytes_, static_cast<std::uint64_t>(prefix) * pipeline_chunk_bytes_));
  }
}

void CpuGatherExecutor::PublishTaskReady() {
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
  AdvanceReadyPrefix(__atomic_load_n(pipeline_ready_, __ATOMIC_ACQUIRE), ready);
}

void CpuGatherExecutor::AdvanceReadyPrefix(std::uint64_t observed, std::uint64_t ready) {
  // Concurrent chunk finishers can advance the prefix in either order.
  while (observed < ready) {
    if (__atomic_compare_exchange_n(pipeline_ready_, &observed, ready, false,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      GatherBarrier();
      break;
    }
  }
}

// Helpers copy disjoint cache lines in each chunk. Only its last participant
// can publish it, and only after every earlier chunk is complete.
std::uint64_t CpuGatherExecutor::CopySizedChunks(std::size_t worker) {
  std::uint64_t copied = 0;
  for (std::size_t chunk = 0; chunk < chunk_done_.size(); ++chunk) {
    const auto begin = ChunkBoundary(chunk), end = ChunkBoundary(chunk + 1);
    if (!CopyStripe(worker, begin, end, copied)) return copied;
    GatherBarrier();
    if (__atomic_add_fetch(&chunk_done_[chunk], 1U, __ATOMIC_ACQ_REL) != active_threads_) continue;
    // Every participant processes chunks in order: the last increment for this
    // chunk also certifies the whole prefix. A delayed publisher must not regress it.
    auto observed = __atomic_load_n(pipeline_ready_, __ATOMIC_ACQUIRE);
    const auto ready = end;
    while (observed < ready && !__atomic_compare_exchange_n(pipeline_ready_, &observed, ready, false,
                                                             __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {}
  }
  return copied;
}

bool CpuGatherExecutor::GroupStopped() const {
  return suffix_failed_.load(std::memory_order_relaxed) ||
         (request_stop_ && request_stop_->load(std::memory_order_acquire));
}

bool CpuGatherExecutor::CopyStripe(std::size_t worker, std::uint64_t begin,
                                  std::uint64_t end, std::uint64_t& copied) {
  const auto lines = (end - begin + 63) / 64;
  const auto lo = std::min(end, begin + 64 * (lines * worker / active_threads_));
  const auto hi = std::min(end, begin + 64 * (lines * (worker + 1) / active_threads_));
  if (hi != lo && CopyRange(lo, hi) != hi - lo) return false;
  copied += hi - lo;
  return true;
}

// Each stripe waits for its addresses. The coordinator publishes the contiguous
// prefix in registered DDR once all stripes of that group have been copied.
std::uint64_t CpuGatherExecutor::CopyGroups(std::size_t worker) {
  std::uint64_t copied = 0, begin = 0;
  for (std::size_t i = 0; i < chunk_done_.size() && !GroupStopped(); ++i) {
    const auto end = group_plan_.EndBytes(i);
    if (!CopyStripe(worker, begin, end, copied) || GroupStopped()) break;
    begin = end;
    if (i + 1 == chunk_done_.size()) break;
    GatherBarrier();
    __atomic_add_fetch(&chunk_done_[i], 1U, __ATOMIC_ACQ_REL);
    if (worker) continue;
    while (__atomic_load_n(&chunk_done_[i], __ATOMIC_ACQUIRE) != active_threads_) {
      if (GroupStopped()) return copied;
      GatherSpinPause();
    }
    __atomic_store_n(pipeline_ready_, end, __ATOMIC_RELEASE);
  }
  return copied;
}

void CpuGatherExecutor::RunHelper(std::size_t worker) {
  BindWorker(worker);
  std::uint64_t observed_generation = 0U;
  bool was_active = false;
  for (;;) {
    bool active = false;
    bool wait_path = false;
    const bool spin = wait_mode_ != GatherWaitMode::kPassive && was_active;
    if (spin) {
      const auto spin_start = std::chrono::steady_clock::now();
      std::uint32_t spin_count = 0U;
      std::uint64_t generation = generation_.load(std::memory_order_acquire);
      while (generation == observed_generation) {
        if (stopping_.load(std::memory_order_acquire)) return;
        if (wait_mode_ == GatherWaitMode::kAdaptive && (++spin_count & 255U) == 0U &&
            std::chrono::steady_clock::now() - spin_start >= std::chrono::microseconds(100))
          break;
        GatherSpinPause();
        generation = generation_.load(std::memory_order_acquire);
      }
    }
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!spin || generation_.load(std::memory_order_acquire) == observed_generation) {
        wait_path = true;
        ready_.wait(lock, [this, &observed_generation] {
          return stopping_.load(std::memory_order_acquire) ||
                 generation_.load(std::memory_order_acquire) != observed_generation;
        });
      }
      if (stopping_.load(std::memory_order_acquire)) return;
      observed_generation = generation_.load(std::memory_order_acquire);
      active = worker < active_threads_;
    }
    was_active = active;
    if (!active) continue;
    if (!thread_profiles_.empty()) {
      thread_profiles_[worker].ready_ns = NowNs();
      thread_profiles_[worker].wait_path = wait_path;
    }
    while (phase_.load(std::memory_order_acquire) == 0U) {
      if (stopping_.load(std::memory_order_acquire)) return;
      GatherSpinPause();
    }
    CopyWorker(worker);
    completed_.fetch_add(1U, std::memory_order_release);
  }
}

}  // namespace kvaf::peer
