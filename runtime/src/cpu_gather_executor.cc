#include "cpu_gather_executor.h"

#include <ctime>

namespace kvaf::peer {
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
// Test-only seams for controlling copy tails and observing helper idle policy.
void CpuGatherBeforeCopyRange(std::uint64_t begin, std::uint64_t end);
void CpuGatherHelperCompleted(std::size_t worker, std::uint64_t generation);
void CpuGatherHelperWait(std::size_t worker, std::uint64_t generation, bool spinning);
#endif
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
  const bool cooperative = header.magic == kGatherCooperativeMagic;
  if (bytes < 64U || base > UINT64_MAX - bytes ||
      (!cooperative && header.magic != kGatherDescriptorMagic) ||
      header.span_count == 0U ||
      header.ready_bytes != 0U ||
      (cooperative && (!header.pipeline_chunk_bytes ||
       (std::uint64_t(header.total_bytes) + header.pipeline_chunk_bytes - 1U) /
           header.pipeline_chunk_bytes > kGatherMaxCooperativeChunks)) ||
      (header.pipeline_chunk_bytes != 0U &&
       (header.pipeline_chunk_bytes < 64U || header.pipeline_chunk_bytes % 64U != 0U)))
    return false;
  const auto data_bytes = GatherControlOffset(bytes);
  const auto descriptor_bytes = GatherDescriptorBytes(header.span_count);
  if (!GatherRange(base, data_bytes, descriptor_address, descriptor_bytes) ||
      !GatherRange(base, data_bytes, header.staging_address, header.total_bytes))
    return false;
  return descriptor_address >= header.staging_address + header.total_bytes ||
         header.staging_address >= descriptor_address + descriptor_bytes;
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
  pipeline_ready_ = nullptr;
  cooperative_ = header.magic == kGatherCooperativeMagic;
  if (header.pipeline_chunk_bytes != 0U) {
    pipeline_chunk_bytes_ = header.pipeline_chunk_bytes;
    pipeline_ready_ = reinterpret_cast<std::uint64_t*>(
        memory_ + (descriptor_address - base_) + offsetof(GatherDescriptorHeader, ready_bytes));
  }
  return Publish(header.staging_address, source_spans, header.span_count, header.total_bytes);
}

void CpuGatherExecutor::Copy() {
  completed_.store(0U, std::memory_order_relaxed);
  if (!thread_profiles_.empty()) {
    copy_profile_.begin_ns = NowNs();
    thread_profiles_[0].ready_ns = copy_profile_.begin_ns;
  }
  phase_.store(CopyPhase::kCopying, std::memory_order_release);
  CopyWorker(0U);
  if (!thread_profiles_.empty())
    copy_profile_.coordinator_done_ns = thread_profiles_[0].done_ns;
  while (completed_.load(std::memory_order_acquire) != active_threads_ - 1U) {
    PublishReady();
    GatherSpinPause();
  }
  PublishReady();
  if (!thread_profiles_.empty()) copy_profile_.done_ns = NowNs();
  // A helper may keep warm only after the coordinator AND all helpers finish.
  // No wakeup is needed here: parked helpers have no more work this generation.
  phase_.store(CopyPhase::kComplete, std::memory_order_release);
}

void CpuGatherExecutor::CopyWorker(std::size_t worker) {
  if (thread_profiles_.empty()) {
    (void)ProcessSource(worker);
    if (pipeline_ready_ == nullptr) GatherBarrier();
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
  // Chunk writers already completed their stores before publishing progress.
  if (pipeline_ready_ == nullptr) GatherBarrier();
  profile.done_ns = NowNs();
  const auto cpu_end = ThreadCpuNs();
  profile.cpu_time_valid = cpu_begin != 0U && cpu_end >= cpu_begin;
  profile.thread_cpu_ns = profile.cpu_time_valid ? cpu_end - cpu_begin : 0U;
}

// Copy only this thread's staging range, combining adjacent source spans into one memcpy.
std::uint64_t CpuGatherExecutor::CopyRange(std::uint64_t offset, std::uint64_t end) const {
  if (offset >= end) return 0U;
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
  CpuGatherBeforeCopyRange(offset, end);
#endif
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

std::uint64_t CpuGatherExecutor::ProcessSource(std::size_t worker) {
  if (cooperative_) {
    std::uint64_t copied = 0;
    for (std::uint32_t chunk = 0; chunk < cooperative_count_; ++chunk) {
      const auto begin = std::uint64_t(chunk) * pipeline_chunk_bytes_;
      const auto end = std::min(begin + pipeline_chunk_bytes_, total_bytes_);
      // Partition aligned cache lines inside EACH prefix block, including a
      // short tail. Empty shares still participate in block completion.
      const auto lines = (end - begin + 63U) / 64U;
      const auto local_begin = std::min(end, begin + lines * worker / active_threads_ * 64U);
      const auto local_end = std::min(end, begin + lines * (worker + 1U) / active_threads_ * 64U);
      copied += CopyRange(local_begin, local_end);
      if (local_begin != local_end) GatherBarrier();
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
  if (pipeline_ready_ == nullptr) {
    return CopyRange(Boundary(worker), Boundary(worker + 1U));
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
  const auto ready = std::min(total_bytes_, std::uint64_t(completed) * pipeline_chunk_bytes_);
  auto previous = __atomic_load_n(pipeline_ready_, __ATOMIC_ACQUIRE);
  // At most eight block finishers publish per batch. Concurrent finishers may
  // coalesce prefixes, but a delayed earlier finisher must never regress ready.
  while (previous < ready) {
    if (__atomic_compare_exchange_n(pipeline_ready_, &previous, ready, false,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      GatherBarrier();
      break;
    }
  }
}

void CpuGatherExecutor::RunHelper(std::size_t worker) {
  BindWorker(worker);
  std::uint64_t observed_generation = 0U;
  bool was_active = false;
  for (;;) {
    bool active = false;
    bool wait_path = false;
    auto generation = generation_.load(std::memory_order_acquire);
    // Finished helpers must not compete with a lagging copy worker. At most
    // helper 1 can keep warm, opportunistically after the whole copy completes.
    // Both active and adaptive have a finite helper budget; mode-specific
    // coordinator notify polling is owned by CpuGatherWorker, not this loop.
    if (generation == observed_generation && was_active && worker == 1U &&
        wait_mode_ != GatherWaitMode::kPassive &&
        phase_.load(std::memory_order_acquire) == CopyPhase::kComplete) {
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
      CpuGatherHelperWait(worker, observed_generation, true);
#endif
      const auto spin_start = std::chrono::steady_clock::now();
      const auto budget = std::chrono::microseconds(
          wait_mode_ == GatherWaitMode::kActive ? 100 : 25);
      std::uint32_t spin_count = 0U;
      while (generation == observed_generation) {
        if (stopping_.load(std::memory_order_acquire)) return;
        if ((++spin_count & 63U) == 0U &&
            std::chrono::steady_clock::now() - spin_start >= budget)
          break;
        GatherSpinPause();
        generation = generation_.load(std::memory_order_acquire);
      }
    }
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (generation_.load(std::memory_order_acquire) == observed_generation) {
        wait_path = true;
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
        CpuGatherHelperWait(worker, observed_generation, false);
#endif
        // Publish/Shutdown change the predicate under this same mutex. A
        // generation arriving between spin and sleep cannot be lost.
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
    while (phase_.load(std::memory_order_acquire) == CopyPhase::kPreparing) {
      if (stopping_.load(std::memory_order_acquire)) return;
      GatherSpinPause();
    }
    CopyWorker(worker);
    completed_.fetch_add(1U, std::memory_order_release);
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
    CpuGatherHelperCompleted(worker, observed_generation);
#endif
  }
}

}  // namespace kvaf::peer
