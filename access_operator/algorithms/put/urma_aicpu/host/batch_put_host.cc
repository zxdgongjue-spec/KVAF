#include <utility>
#include <vector>

#include "put/urma_aicpu/include/kvaf_aicpu_batch_put.h"
#include "meta_manager.h"
#include "aicpu_urma/session.h"
#include "put_host_timing.h"
namespace kvaf::aicpu_urma {
std::unique_ptr<ExecutionState> BatchPutTransport::CreateExecutionState() {
  return std::make_unique<Preparation>(this);
}
Status BatchPutTransport::BindExecutionState(ExecutionState* state) {
  auto* typed = state ? dynamic_cast<Preparation*>(state) : &legacy_preparation_;
  if (!typed || typed->owner != this)
    return Status::InvalidArgument("Put preparation belongs to another resource");
  preparation_ = typed;
  views_.Bind(typed);
  return Status::OK();
}
Status BatchPutTransport::Shutdown() {
  views_.Close();
  auto drained = session_.Drain();
  if (!drained.ok()) return drained;
  if (transfers_) (void)aclrtFree(transfers_);
  transfers_ = nullptr;
  *preparation_ = Preparation(this);
  capacity_ = 0;
  host_transfers_ = std::vector<UmdkTransfer>{};
  return Status::OK();
}
Status BatchPutTransport::Reserve(std::uint32_t count) {
  if (!count || count > kUmdkMaxTransfers) return Status::InvalidArgument("invalid Put capacity");
  const auto bytes = std::size_t(count) * sizeof(UmdkTransfer);
  if (bytes <= capacity_) return Status::OK();
  views_.Prepare(ExecutionDirection::Put);
  void* allocation = nullptr;
  if (aclrtMalloc(&allocation, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    return Status::OutOfMemory("cannot reserve WRITE descriptors");
  if (transfers_) (void)aclrtFree(transfers_);
  transfers_ = allocation;
  capacity_ = bytes;
  host_transfers_.reserve(count);
  return Status::OK();
}
Status BatchPutTransport::Prepare(const RemoteRegionDesc& region, const BatchPutRequest& request,
                                  PutProviderTiming* timing) {
  views_.Prepare();
  preparation_->request = nullptr;
  if (!request.input.valid() || request.input.device != DeviceKind::kNpu || request.keys.empty() ||
      request.keys.size() > kUmdkMaxTransfers || request.keys.size() != request.input_offsets.size() ||
      request.keys.size() != request.lengths.size())
    return Status::InvalidArgument("invalid BatchPut layout");
  PutHostTimingScope registration_timing(timing, &PutProviderTiming::registration_ns);
  auto ready = session_.Ensure(region.base, const_cast<void*>(request.input.data), request.input.bytes);
  registration_timing.Finish();
  if (!ready.ok()) return ready;
  ready = Reserve(static_cast<std::uint32_t>(request.keys.size()));
  if (!ready.ok()) return ready;
  return RecordPreparation(region, request);
}
Status BatchPutTransport::PrepareBound(const RemoteRegionDesc& region, const BatchPutRequest& request,
                                       const PutInputOptions& input) {
  views_.Prepare();
  preparation_->request = nullptr;
  if (request.input.data != input.input.data || request.input.bytes != input.input.bytes ||
      request.input.device != input.input.device || request.keys.empty() ||
      request.keys.size() > input.request_capacity || request.keys.size() > kUmdkMaxTransfers ||
      request.keys.size() != request.input_offsets.size() ||
      request.keys.size() != request.lengths.size() ||
      request.keys.size() * sizeof(UmdkTransfer) > capacity_)
    return Status::InvalidArgument("request differs from prepared Put input/capacity");
  return RecordPreparation(region, request);
}
Status BatchPutTransport::RecordPreparation(const RemoteRegionDesc& region, const BatchPutRequest& request) {
  views_.Initialize(region);
  views_.Bind(preparation_);
  preparation_->request = &request;
  preparation_->region = region.base;
  preparation_->input = request.input.data;
  preparation_->bytes = request.input.bytes;
  preparation_->rows = request.keys.size();
  return Status::OK();
}
Status BatchPutTransport::Execute(const RemoteRegionDesc& region, const BatchPutExecRequest& exec) {
  if (!exec.request) return Status::InvalidArgument("null BatchPut request");
  PutHostTimingScope prepare_timing(exec.execution.host_timing, &PutProviderTiming::prepare_ns);
  auto prepared = Prepare(region, *exec.request, exec.execution.host_timing);
  prepare_timing.Finish();
  return prepared.ok() ? ExecutePrepared(region, exec) : prepared;
}
PutLaunchView BatchPutTransport::View() {
  return {views_.View(ExecutionDirection::Put), session_.WriteView(), transfers_, capacity_,
          &host_transfers_};
}
Status BatchPutTransport::ExecutePrepared(const RemoteRegionDesc& region, const BatchPutExecRequest& exec) {
  auto view = View();
  if (region.base != view.common.region().canonical_base)
    return Status::InvalidState("AICPU Put region differs from prepared resources");
  return LaunchPrepared(exec, view, preparation_);
}
Status BatchPutTransport::LaunchPrepared(const BatchPutExecRequest& exec, const PutLaunchView& view,
                                         ExecutionState* state) {
  auto* preparation = dynamic_cast<PutPreparation*>(state);
  auto valid = view.common.Validate(preparation, ExecutionDirection::Put);
  if (!valid.ok()) return valid;
  // Only this synchronous Prepare/Launch pair may borrow the caller's request.
  // Consume the marker even when validation fails; recovery never consults it.
  const auto* prepared_request = std::exchange(preparation->request, nullptr);
  if (!exec.request || !exec.put_desc || !exec.execution.stream)
    return Status::InvalidArgument(
        "UMDK BatchPut requires initialized operator, input, destination and stream");
  const auto& request = *exec.request;
  if (!request.input.valid() || request.input.device != DeviceKind::kNpu || request.keys.empty() ||
      request.keys.size() != request.input_offsets.size() ||
      request.keys.size() != request.lengths.size() ||
      request.keys.size() != exec.put_desc->entries.size())
    return Status::InvalidArgument("invalid BatchPut layout");
  if (prepared_request != &request || preparation->region != view.common.region().canonical_base ||
      preparation->input != request.input.data || preparation->bytes != request.input.bytes ||
      preparation->rows != request.keys.size() ||
      request.keys.size() * sizeof(UmdkTransfer) > view.capacity)
    return Status::InvalidState("AICPU Put resources were not prepared for this request");
  PutHostTimingScope metadata_timing(exec.execution.host_timing, &PutProviderTiming::metadata_ns);
  auto& transfers = (*view.host_descriptors);
  transfers.clear();
  for (std::size_t index = 0; index < request.keys.size(); ++index) {
    const auto offset = request.input_offsets[index], length = request.lengths[index];
    const auto& destination = exec.put_desc->entries[index];
    if (!length || offset > request.input.bytes || length > request.input.bytes - offset ||
        !destination.address || destination.reserved_length != length)
      return Status::InvalidArgument("invalid BatchPut span");
    UmdkTransfer transfer{destination.address,
                          reinterpret_cast<std::uint64_t>(request.input.data) + offset, length};
    if (!transfers.empty() && transfers.back().local + transfers.back().bytes == transfer.local &&
        transfers.back().remote + transfers.back().bytes == transfer.remote)
      transfers.back().bytes += length;
    else
      transfers.push_back(transfer);
  }
  if (transfers.empty() || transfers.size() > kUmdkMaxTransfers)
    return Status::InvalidArgument("UMDK BatchPut exceeds the transfer list capacity");
  const auto bytes = transfers.size() * sizeof(UmdkTransfer);
  if (aclrtMemcpy(view.descriptors, view.capacity, transfers.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) !=
      ACL_SUCCESS)
    return Status::Internal("failed to upload UMDK WRITE descriptors");
  metadata_timing.Finish();
  return view.transport.Submit(reinterpret_cast<std::uint64_t>(view.descriptors), transfers.size(),
      static_cast<aclrtStream>(exec.execution.stream), LaunchBatchPut, exec.execution.host_timing);
}

}  // namespace kvaf::aicpu_urma
