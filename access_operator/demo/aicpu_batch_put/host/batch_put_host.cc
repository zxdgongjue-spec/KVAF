#include <vector>

#include "aicpu_batch_put/include/kvaf_aicpu_batch_put.h"
#include "meta_manager.h"
#include "aicpu_urma/session.h"
namespace kvaf::aicpu_urma {
Status BatchPutTransport::Shutdown() {
  auto drained = session_.Drain();
  if (!drained.ok()) return drained;
  if (transfers_) (void)aclrtFree(transfers_);
  transfers_ = nullptr;
  capacity_ = 0;
  host_transfers_ = std::vector<UmdkTransfer>{};
  return Status::OK();
}
Status BatchPutTransport::Execute(const RemoteRegionDesc& region, const BatchPutExecRequest& exec) {
  if (!exec.request || !exec.put_desc || !exec.execution.stream)
    return Status::InvalidArgument(
        "UMDK BatchPut requires initialized operator, input, destination and stream");
  const auto& request = *exec.request;
  if (!request.input.valid() || request.input.device != DeviceKind::kNpu || request.keys.empty() ||
      request.keys.size() != request.input_offsets.size() ||
      request.keys.size() != request.lengths.size() ||
      request.keys.size() != exec.put_desc->entries.size())
    return Status::InvalidArgument("invalid BatchPut layout");
  auto ready =
      session_.Ensure(region.base, const_cast<void*>(request.input.data), request.input.bytes);
  if (!ready.ok()) return ready;
  auto& transfers = host_transfers_;
  transfers.clear();
  transfers.reserve(request.keys.size());
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
  if (bytes > capacity_) {
    void* allocation = nullptr;
    if (aclrtMalloc(&allocation, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
      return Status::Internal("failed to allocate UMDK WRITE descriptors");
    if (transfers_ != nullptr) (void)aclrtFree(transfers_);
    transfers_ = allocation;
    capacity_ = bytes;
  }
  if (aclrtMemcpy(transfers_, capacity_, transfers.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) !=
      ACL_SUCCESS)
    return Status::Internal("failed to upload UMDK WRITE descriptors");
  return session_.WriteDescriptors(reinterpret_cast<std::uint64_t>(transfers_), transfers.size(),
                                   static_cast<aclrtStream>(exec.execution.stream), LaunchBatchPut);
}

}  // namespace kvaf::aicpu_urma
