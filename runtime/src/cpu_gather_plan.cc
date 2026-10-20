#include "cpu_gather_team.h"

namespace kvaf::peer {
namespace {

bool ValidateEnvelope(std::uint64_t base, std::uint64_t bytes, std::uint64_t descriptor_address,
                      const GatherDescriptorHeader& header, std::size_t span_count) {
  if (bytes < 64U || base > UINT64_MAX - bytes || header.magic != kGatherDescriptorMagic ||
      header.span_count == 0U || header.span_count != span_count || header.reserved != 0U ||
      header.ready_bytes != 0U ||
      (header.pipeline_chunk_bytes != 0U &&
       (header.pipeline_chunk_bytes < 64U || header.pipeline_chunk_bytes % 64U != 0U)))
    return false;
  const auto data_bytes = GatherControlOffset(bytes);
  if (!GatherRange(base, data_bytes, descriptor_address, GatherDescriptorBytes(span_count)) ||
      !GatherRange(base, data_bytes, header.staging_address, header.total_bytes))
    return false;
  return descriptor_address >= header.staging_address + header.total_bytes ||
         header.staging_address >= descriptor_address + GatherDescriptorBytes(span_count);
}

bool BuildOffsets(std::uint64_t base, std::uint64_t data_bytes, std::uint64_t descriptor_address,
                  const GatherDescriptorHeader& header, const void* spans, std::size_t span_count,
                  std::vector<std::uint64_t>* offsets, GatherSpan* snapshot = nullptr) {
  const auto staging_end = header.staging_address + header.total_bytes;
  const auto descriptor_end = descriptor_address + GatherDescriptorBytes(span_count);
  if (offsets != nullptr) {
    offsets->resize(span_count + 1U);
    (*offsets)[0U] = 0U;
  }
  std::uint64_t total = 0U;
  for (std::size_t index = 0U; index < span_count; ++index) {
    // Load once from the published descriptor, then retain the validated value.
    GatherSpan span;
    std::memcpy(&span, static_cast<const std::uint8_t*>(spans) + index * sizeof(span),
                sizeof(span));
    total += span.length;
    if (total > header.total_bytes || !GatherRange(base, data_bytes, span.address, span.length) ||
        (span.address < staging_end && header.staging_address < span.address + span.length) ||
        (span.address < descriptor_end && descriptor_address < span.address + span.length))
      return false;
    if (offsets != nullptr) (*offsets)[index + 1U] = total;
    if (snapshot != nullptr) snapshot[index] = span;
  }
  return total == header.total_bytes;
}

}  // namespace

bool ValidateDirectGather(std::uint64_t base, std::uint64_t bytes, std::uint64_t descriptor_address,
                          const GatherDescriptorHeader& header,
                          const std::vector<GatherSpan>& spans) {
  return ValidateEnvelope(base, bytes, descriptor_address, header, spans.size()) &&
         BuildOffsets(base, GatherControlOffset(bytes), descriptor_address, header, spans.data(),
                      spans.size(), nullptr);
}

std::size_t CpuGatherTeam::PrepareDirect(std::uint64_t region_bytes,
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
      !ValidateEnvelope(base_, region_bytes, descriptor_address, header, header.span_count))
    return 0U;
  direct_spans_.resize(header.span_count);
  const auto* source_spans = descriptor + sizeof(header);
  if (!BuildOffsets(base_, GatherControlOffset(region_bytes), descriptor_address, header,
                    source_spans, header.span_count, &offsets_, direct_spans_.data()))
    return 0U;
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
  return Publish(header.staging_address, direct_spans_, header.total_bytes);
}

void CpuGatherTeam::Copy() {
  if (copy_profiles_) {
    copy_profile_ = {};
    copy_profile_.workers = active_threads_;
    copy_profile_.begin_ns =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count());
  }
  completed_.store(0U, std::memory_order_relaxed);
  phase_.store(1U, std::memory_order_release);
  RunSource(0U);
  GatherBarrier();
  const auto wait_begin =
      copy_profiles_
          ? static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now().time_since_epoch())
                                           .count())
          : 0U;
  while (completed_.load(std::memory_order_acquire) != active_threads_ - 1U) GatherSpinPause();
  // Concurrent chunk scans may stop at a not-yet-observed completion. Once
  // every copier has joined, publish the full prefix before the final ACK.
  if (pipeline_ready_ != nullptr) __atomic_store_n(pipeline_ready_, total_bytes_, __ATOMIC_RELEASE);
  if (copy_profiles_) {
    copy_profile_.end_ns =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count());
    copy_profile_.join_wait_ns = copy_profile_.end_ns - wait_begin;
  }
}

std::uint64_t CpuGatherTeam::CopyRange(std::uint64_t offset, std::uint64_t end) const {
  const auto begin = offset;
  auto found = std::upper_bound(offsets_.begin(), offsets_.end(), offset);
  std::size_t index = static_cast<std::size_t>(found - offsets_.begin() - 1U);
  while (offset < end) {
    const GatherSpan& span = (*spans_)[index];
    const std::uint64_t in_span = offset - offsets_[index];
    const std::uint64_t bytes = std::min<std::uint64_t>(end - offset, span.length - in_span);
    auto* source = memory_ + (span.address - base_) + in_span;
    std::memcpy(staging_ + offset, source, bytes);
    offset += bytes;
    ++index;
  }
  return offset - begin;
}

std::uint64_t CpuGatherTeam::ProcessSource(std::size_t worker) {
  if (pipeline_ready_ == nullptr) {
    return CopyRange(Boundary(worker), Boundary(worker + 1U));
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

void CpuGatherTeam::RunSource(std::size_t worker) {
  if (!copy_profiles_) {
    (void)ProcessSource(worker);
    return;
  }
  auto& profile = copy_profiles_[worker];
  profile = {};
  const auto now = [] {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
  };
  const auto thread_cpu = []() -> std::uint64_t {
    timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) return std::uint64_t{0};
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
  };
#if defined(__linux__)
  profile.start_cpu = sched_getcpu();
#endif
  const auto cpu_begin = thread_cpu();
  profile.begin_ns = now();
  profile.bytes = ProcessSource(worker);
  profile.end_ns = now();
  const auto cpu_end = thread_cpu();
  profile.thread_cpu_available = cpu_begin != 0U && cpu_end >= cpu_begin;
  if (profile.thread_cpu_available) profile.thread_cpu_ns = cpu_end - cpu_begin;
#if defined(__linux__)
  profile.end_cpu = sched_getcpu();
#endif
}

void CpuGatherTeam::RunHelper(std::size_t worker) {
  BindWorker(worker);
  std::uint64_t observed_generation = 0U;
  bool was_active = false;
  for (;;) {
    bool active = false;
    if (wait_mode_ != GatherWaitMode::kPassive && was_active) {
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
      std::unique_lock<std::mutex> lock(mutex_);
      if (generation_.load(std::memory_order_acquire) == observed_generation)
        ready_.wait(lock, [this, &observed_generation] {
          return stopping_.load(std::memory_order_acquire) ||
                 generation_.load(std::memory_order_acquire) != observed_generation;
        });
      if (stopping_.load(std::memory_order_acquire)) return;
      observed_generation = generation_.load(std::memory_order_acquire);
      active = worker < active_threads_;
    } else {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [this, &observed_generation] {
        return stopping_.load(std::memory_order_acquire) ||
               generation_.load(std::memory_order_acquire) != observed_generation;
      });
      if (stopping_.load(std::memory_order_acquire)) return;
      observed_generation = generation_.load(std::memory_order_acquire);
      active = worker < active_threads_;
    }
    was_active = active;
    if (!active) continue;
    while (phase_.load(std::memory_order_acquire) == 0U) {
      if (stopping_.load(std::memory_order_acquire)) return;
      GatherSpinPause();
    }
    RunSource(worker);
    GatherBarrier();
    completed_.fetch_add(1U, std::memory_order_release);
  }
}

}  // namespace kvaf::peer
