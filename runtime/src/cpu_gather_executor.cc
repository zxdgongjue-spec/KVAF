#include "cpu_gather_executor.h"

#include <exception>
#include <stdexcept>

namespace kvaf::peer {
namespace {

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
  return Publish(memory_ + (header.staging_address - base_), source_spans, nullptr,
                 header.span_count, header.total_bytes);
}

std::size_t CpuGatherExecutor::PrepareFixed(void* staging, const std::uint64_t* addresses,
    std::uint32_t count, std::uint32_t span_bytes, std::uint64_t source_bytes,
    std::uint32_t chunk_rows) {
  if (!count || !span_bytes || span_bytes > source_bytes || base_ > UINT64_MAX - source_bytes)
    return 0U;
  if (chunk_rows && (reinterpret_cast<std::uintptr_t>(staging) % 64U ||
                     std::uint64_t{chunk_rows} * span_bytes % 64U))
    return 0U;
  fixed_chunk_rows_ = chunk_rows;
  fixed_progress_.resize(chunk_rows ? (count - 1U) / chunk_rows + 1U : 0U);
  fixed_source_bytes_ = source_bytes;
  fixed_copy_failed_.store(false, std::memory_order_relaxed);
  span_bytes_ = span_bytes;
  pipeline_ready_ = nullptr;
  return Publish(staging, nullptr, addresses, count, std::uint64_t{count} * span_bytes);
}

void CpuGatherExecutor::Copy(const ChunkReady& on_chunk) {
  completed_.store(0U, std::memory_order_relaxed);
  phase_.store(1U, std::memory_order_release);
  std::exception_ptr publish_error;
  if (fixed_progress_.empty()) {
    (void)ProcessSource(0U);
  } else {
    try {
      for (std::size_t chunk = 0; chunk < fixed_progress_.size(); ++chunk) {
        if (fixed_copy_failed_.load(std::memory_order_relaxed)) break;
        CopyFixedChunk(0U, chunk);
        while (__atomic_load_n(&fixed_progress_[chunk].remaining, __ATOMIC_ACQUIRE) &&
               !fixed_copy_failed_.load(std::memory_order_relaxed))
          GatherSpinPause();
        if (fixed_copy_failed_.load(std::memory_order_relaxed)) break;
        const auto rows = std::min<std::uint64_t>(
            span_count_, std::uint64_t{fixed_chunk_rows_} * (chunk + 1U));
        if (on_chunk && !on_chunk(static_cast<std::uint32_t>(rows))) break;
      }
    } catch (...) {
      publish_error = std::current_exception();
    }
  }
  GatherBarrier();
  while (completed_.load(std::memory_order_acquire) != active_threads_ - 1U) GatherSpinPause();
  if (publish_error) std::rethrow_exception(publish_error);
  if (addresses_ && fixed_copy_failed_.load(std::memory_order_relaxed))
    throw std::invalid_argument("CPU gather source outside registered DDR");
  if (pipeline_ready_ != nullptr) __atomic_store_n(pipeline_ready_, total_bytes_, __ATOMIC_RELEASE);
}

// Copy only this thread's staging range, combining adjacent source spans into one memcpy.
std::uint64_t CpuGatherExecutor::CopyRange(std::uint64_t offset, std::uint64_t end) const {
  if (addresses_)
    return CopyRange(offset, end, [this](std::size_t index) { return addresses_[index]; });
  return CopyRange(offset, end, [this](std::size_t index) { return spans_[index].address; });
}

template <typename Address>
std::uint64_t CpuGatherExecutor::CopyRange(std::uint64_t offset, std::uint64_t end,
                                          Address address) const {
  if (offset >= end) return 0U;
  const auto begin = offset;
  const auto index_for_offset = span_bytes_ != 0U
      ? offset / span_bytes_
      : static_cast<std::uint64_t>(std::upper_bound(variable_offsets_.begin(),
            variable_offsets_.end(), offset) - variable_offsets_.begin() - 1U);
  std::size_t index = static_cast<std::size_t>(index_for_offset);
  auto in_span = span_bytes_ != 0U ? offset % span_bytes_ : offset - variable_offsets_[index];
  while (offset < end) {
    const auto source_address = address(index) + in_span;
    const auto length = span_bytes_ != 0U ? span_bytes_ : spans_[index].length;
    auto bytes = std::min<std::uint64_t>(length - in_span, end - offset);
    ++index;
    in_span = 0U;
    while (bytes < end - offset && index < span_count_ &&
           address(index) == source_address + bytes) {
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
  // Validate full tokens even where workers split a token at a cache line.
  const auto last = (end - 1U) / span_bytes_;
  for (auto row = begin / span_bytes_; row <= last; ++row) {
    if (!GatherRange(base_, fixed_source_bytes_, addresses_[row], span_bytes_)) {
      fixed_copy_failed_.store(true, std::memory_order_relaxed);
      return 0U;
    }
  }
  return CopyRange(begin, end);
}

std::uint64_t CpuGatherExecutor::CopyFixedChunk(std::size_t worker, std::size_t chunk) {
  const auto chunk_bytes = std::uint64_t{fixed_chunk_rows_} * span_bytes_;
  const auto begin = chunk * chunk_bytes;
  const auto end = begin + std::min(chunk_bytes, total_bytes_ - begin);
  const auto copied = CopyFixedRange(Boundary(worker, begin, end), Boundary(worker + 1U, begin, end));
  if (copied) GatherBarrier();
  // The last decrement carries every writer's completed stores to the owner.
  __atomic_fetch_sub(&fixed_progress_[chunk].remaining, 1U, __ATOMIC_ACQ_REL);
  return copied;
}

std::uint64_t CpuGatherExecutor::ProcessSource(std::size_t worker) {
  if (!fixed_progress_.empty()) {
    std::uint64_t copied = 0U;
    for (std::size_t chunk = 0; chunk < fixed_progress_.size(); ++chunk) {
      if (fixed_copy_failed_.load(std::memory_order_relaxed)) break;
      copied += CopyFixedChunk(worker, chunk);
    }
    return copied;
  }
  if (pipeline_ready_ == nullptr) {
    const auto begin = Boundary(worker, 0U, total_bytes_);
    const auto end = Boundary(worker + 1U, 0U, total_bytes_);
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

void CpuGatherExecutor::RunHelper(std::size_t worker) {
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
    (void)ProcessSource(worker);
    GatherBarrier();
    completed_.fetch_add(1U, std::memory_order_release);
  }
}

}  // namespace kvaf::peer
