#include "cpu_gather_team.h"

namespace kvaf::peer {
namespace {

bool ValidateEnvelope(std::uint64_t base, std::uint64_t bytes,
                      std::uint64_t descriptor_address,
                      const GatherDescriptorHeader& header, std::size_t span_count) {
  if (bytes < 64U || base > UINT64_MAX - bytes ||
      header.magic != kGatherDescriptorMagic || header.span_count == 0U ||
      header.span_count != span_count || header.reserved != 0U ||
      header.ready_bytes != 0U ||
      (header.pipeline_chunk_bytes != 0U &&
       (header.pipeline_chunk_bytes < 64U || header.pipeline_chunk_bytes % 64U != 0U)))
    return false;
  const auto data_bytes = GatherControlOffset(bytes);
  if (!GatherRange(base, data_bytes, descriptor_address, GatherDescriptorBytes(span_count)) ||
      !GatherRange(base, data_bytes, header.staging_address, header.total_bytes)) return false;
  return descriptor_address >= header.staging_address + header.total_bytes ||
         header.staging_address >= descriptor_address + GatherDescriptorBytes(span_count);
}

bool BuildOffsets(std::uint64_t base, std::uint64_t data_bytes,
                  std::uint64_t descriptor_address, const GatherDescriptorHeader& header,
                  const std::vector<GatherSpan>& spans,
                  std::vector<std::uint64_t>* offsets) {
  const auto staging_end = header.staging_address + header.total_bytes;
  const auto descriptor_end = descriptor_address + GatherDescriptorBytes(spans.size());
  if (offsets != nullptr) {
    offsets->resize(spans.size() + 1U);
    (*offsets)[0U] = 0U;
  }
  std::uint64_t total = 0U;
  for (std::size_t index = 0U; index < spans.size(); ++index) {
    const auto& span = spans[index];
    total += span.length;
    if (total > header.total_bytes ||
        !GatherRange(base, data_bytes, span.address, span.length) ||
        (span.address < staging_end && header.staging_address < span.address + span.length) ||
        (span.address < descriptor_end && descriptor_address < span.address + span.length))
      return false;
    if (offsets != nullptr) (*offsets)[index + 1U] = total;
  }
  return total == header.total_bytes;
}

}

bool ValidateDirectGather(std::uint64_t base, std::uint64_t bytes,
                          std::uint64_t descriptor_address,
                          const GatherDescriptorHeader& header,
                          const std::vector<GatherSpan>& spans) {
  return ValidateEnvelope(base, bytes, descriptor_address, header, spans.size()) &&
         BuildOffsets(base, GatherControlOffset(bytes), descriptor_address, header,
                      spans, nullptr);
}

std::size_t CpuGatherTeam::PrepareDirect(std::uint64_t region_bytes,
                                        std::uint64_t descriptor_address,
                                        std::uint64_t sequence, std::size_t max_spans) {
  if (region_bytes < 64U || base_ > UINT64_MAX - region_bytes ||
      descriptor_address % 64U != 0U ||
      !GatherRange(base_, GatherControlOffset(region_bytes), descriptor_address,
                   sizeof(GatherDescriptorHeader))) return 0U;
  const auto* descriptor = memory_ + (descriptor_address - base_);
  GatherDescriptorHeader header;
  std::memcpy(&header, descriptor, sizeof(header));
  if (header.sequence != sequence || header.span_count > max_spans ||
      !ValidateEnvelope(base_, region_bytes, descriptor_address, header, header.span_count))
    return 0U;
  direct_spans_.resize(header.span_count);
  std::memcpy(direct_spans_.data(), descriptor + sizeof(header),
              header.span_count * sizeof(GatherSpan));
  if (!BuildOffsets(base_, GatherControlOffset(region_bytes), descriptor_address, header,
                     direct_spans_, &offsets_)) return 0U;
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

}
