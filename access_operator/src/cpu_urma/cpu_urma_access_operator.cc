#include "cpu_urma/cpu_urma_access_operator.h"

#include "meta_manager.h"

#include <limits>
#include <utility>

namespace kvaf {

CpuUrmaAccessOperator::CpuUrmaAccessOperator(CpuUrmaTransportOptions options,
                                             std::shared_ptr<peer::PeerLink> link)
    : options_(std::move(options)), link_(std::move(link)) {}
CpuUrmaAccessOperator::~CpuUrmaAccessOperator() { (void)Shutdown(); }

Status CpuUrmaAccessOperator::Initialize(const RemoteRegionDesc& remote_region) {
  if (initialized_) return Status::InvalidState("CPU URMA operator already initialized");
  if (remote_region.base == 0U || remote_region.length == 0U)
    return Status::InvalidArgument("invalid remote HBM region");
  options_.region_bytes = remote_region.length;
  if (link_ != nullptr) {
    peer::PeerRegion descriptor{};
    const Status lookup = link_->LookupRegion(remote_region.base, &descriptor);
    if (!lookup.ok()) return lookup;
    options_.has_external_region = true;
    options_.external_region = descriptor;
  }
  const Status status = transport_.Initialize(options_);
  if (!status.ok()) {
    (void)transport_.Shutdown();
    return status;
  }
  region_ = remote_region;
  initialized_ = true;
  return Status::OK();
}

Status CpuUrmaAccessOperator::Shutdown() {
  if (!initialized_) return Status::OK();
  initialized_ = false;
  region_ = {};
  return transport_.Shutdown();
}

Status CpuUrmaAccessOperator::ValidateLink() {
  std::unique_lock<std::mutex> lock(batch_mutex_);
  return transport_.ValidateLink();
}

Status CpuUrmaAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& request) {
  if (!initialized_) return Status::NotInitialized("CPU URMA operator is not initialized");
  std::unique_lock<std::mutex> batch_lock(batch_mutex_);
  if (request.plan == nullptr || request.put_desc == nullptr ||
      request.plan->lengths.empty() ||
      request.plan->lengths.size() != request.put_desc->entries.size() ||
      request.plan->lengths.size() != request.plan->source_indices.size() ||
      request.plan->lengths.size() != request.plan->offsets.size())
    return Status::InvalidArgument("invalid BatchPut execution plan");

  batch_buffers_.clear();
  batch_remote_offsets_.clear();
  batch_buffers_.reserve(request.put_desc->entries.size());
  batch_remote_offsets_.reserve(request.put_desc->entries.size());
  batch_used_sources_.assign(request.plan->sources.size(), 0U);
  for (std::size_t i = 0; i < request.put_desc->entries.size(); ++i) {
    const auto& allocation = request.put_desc->entries[i];
    const std::uint32_t source_index = request.plan->source_indices[i];
    if (source_index >= request.plan->sources.size())
      return Status::InvalidArgument("BatchPut source index is out of range");
    const auto& source = request.plan->sources[source_index];
    const std::uint64_t source_offset = request.plan->offsets[i];
    const std::uint64_t bytes = request.plan->lengths[i];
    if (!source.valid() || source.device != DeviceKind::kCpu ||
        source_offset > source.bytes || bytes > source.bytes - source_offset ||
        allocation.address < region_.base ||
        allocation.address - region_.base > region_.length ||
        bytes > region_.length - (allocation.address - region_.base))
      return Status::InvalidArgument("BatchPut entry is outside its buffer or remote region");
    const auto* data = static_cast<const std::uint8_t*>(source.data) + source_offset;
    batch_buffers_.push_back(CpuUrmaTransport::HostSge{
        const_cast<std::uint8_t*>(data), static_cast<std::size_t>(bytes)});
    batch_remote_offsets_.push_back(allocation.address - region_.base);
    batch_used_sources_[source_index] = 1U;
  }
  // AIV BatchGet binds its complete HBM output before posting READs. For the
  // corresponding CPU DRAM -> HBM flow, bind each complete CPU source buffer
  // once so its distinct row SGEs reuse the same registered host segment.
  // Providers that reject the larger range fall back to per-row registration.
  for (std::size_t index = 0; index < request.plan->sources.size(); ++index) {
    if (batch_used_sources_[index] == 0U) continue;
    const ConstBufferView& source = request.plan->sources[index];
    (void)transport_.RegisterDirectHostMemory(
        const_cast<void*>(source.data), source.bytes);
  }
  return transport_.WriteBatch(batch_remote_offsets_.data(),
                               batch_buffers_.data(), batch_buffers_.size());
}

CpuUrmaTransport::BatchTiming CpuUrmaAccessOperator::last_batch_timing() const {
  std::unique_lock<std::mutex> batch_lock(batch_mutex_);
  return transport_.last_batch_timing();
}

Status CpuUrmaAccessOperator::ExecuteBatchGet(const BatchGetExecRequest& request) {
  if (!initialized_) return Status::NotInitialized("CPU URMA operator is not initialized");
  std::unique_lock<std::mutex> batch_lock(batch_mutex_);
  if (request.request == nullptr || request.host_access_table.addresses == nullptr ||
      request.host_access_table.lengths == nullptr || request.host_access_table.valid == nullptr)
    return Status::InvalidArgument("CPU BatchGet requires a host AccessTable");
  const BatchGetRequest& batch = *request.request;
  if (!batch.indices.has_elements<std::int32_t>() || !batch.offsets.has_elements<std::int32_t>() ||
      !batch.output_indices.has_elements<std::int32_t>() || !batch.output.valid() ||
      batch.indices.bytes != batch.offsets.bytes ||
      batch.indices.bytes != batch.output_indices.bytes ||
      batch.output_block_bytes == 0U ||
      batch.output_block_bytes > std::numeric_limits<std::size_t>::max())
    return Status::InvalidArgument("invalid CPU BatchGet buffers");
  const std::size_t request_count =
      batch.indices.bytes / sizeof(std::int32_t);
  if (!batch.mask.empty() &&
      (!batch.mask.valid() || batch.mask.bytes != request_count)) {
    return Status::InvalidArgument("invalid CPU BatchGet mask");
  }
  const auto* indices = static_cast<const std::int32_t*>(batch.indices.data);
  const auto* offsets = static_cast<const std::int32_t*>(batch.offsets.data);
  const auto* output_indices = static_cast<const std::int32_t*>(batch.output_indices.data);
  const auto* mask = static_cast<const std::uint8_t*>(batch.mask.data);
  batch_buffers_.clear();
  batch_remote_offsets_.clear();
  batch_buffers_.reserve(request_count);
  batch_remote_offsets_.reserve(request_count);
  for (std::size_t i = 0; i < request_count; ++i) {
    if (mask != nullptr && mask[i] == 0U) continue;
    const std::int32_t index = indices[i];
    const std::int32_t offset = offsets[i];
    const std::int32_t output_index = output_indices[i];
    if (index < 0 || output_index < 0 || static_cast<std::size_t>(index) >= request.host_access_table.size || offset < 0 ||
        request.host_access_table.valid[index] == 0)
      return Status::InvalidArgument("CPU BatchGet index is invalid");
    const std::uint64_t length = request.host_access_table.lengths[index];
    if (static_cast<std::uint64_t>(offset) > length ||
        batch.output_block_bytes == 0U ||
        batch.output_block_bytes > length - static_cast<std::uint64_t>(offset))
      return Status::InvalidArgument("CPU BatchGet offset exceeds entry length");
    const std::uint64_t output_row = static_cast<std::uint64_t>(output_index);
    if (output_row >= batch.output.bytes / batch.output_block_bytes)
      return Status::InvalidArgument("CPU BatchGet output index exceeds output buffer");
    const std::uint64_t destination_offset =
        output_row * batch.output_block_bytes;
    const Address address = request.host_access_table.addresses[index];
    const std::uint64_t remote_offset = address < region_.base ? region_.length + 1U
                                                               : address - region_.base;
    if (remote_offset > region_.length || offset < 0 ||
        static_cast<std::uint64_t>(offset) > region_.length - remote_offset ||
        batch.output_block_bytes > region_.length - remote_offset - static_cast<std::uint64_t>(offset))
      return Status::InvalidArgument("CPU BatchGet remote address is invalid");
    auto* destination = static_cast<std::uint8_t*>(batch.output.data) + destination_offset;
    batch_buffers_.push_back(CpuUrmaTransport::HostSge{
        destination, static_cast<std::size_t>(batch.output_block_bytes)});
    batch_remote_offsets_.push_back(remote_offset +
                                    static_cast<std::uint64_t>(offset));
  }
  if (batch_buffers_.empty()) return Status::OK();
  // Prefer one segment for the contiguous output. Providers that reject the
  // larger registration still fall back to per-entry registration below.
  (void)transport_.RegisterDirectHostMemory(batch.output.data,
                                            batch.output.bytes);
  return transport_.ReadBatch(batch_remote_offsets_.data(),
                              batch_buffers_.data(), batch_buffers_.size());
}

}  // namespace kvaf
