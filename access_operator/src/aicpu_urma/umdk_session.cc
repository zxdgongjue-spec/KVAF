#include "umdk_session.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace kvaf::aicpu_urma {

Status UmdkSession::Run(UmdkControlOp op, UmdkControlStatus& status, aclrtStream stream) {
  if (stream == nullptr) stream = stream_;
  args_.op = op;
  auto* device_status = reinterpret_cast<void*>(args_.status);
  if (aclrtMemset(device_status, sizeof(status), 0, sizeof(status)) != ACL_SUCCESS)
    return Status::Internal("failed to clear GatherRead UMDK setup status");
  const auto launch = LaunchUmdkControl(&args_, stream);
  const auto sync = aclrtSynchronizeStream(stream);
  const auto copy = aclrtMemcpy(&status, sizeof(status), device_status, sizeof(status),
                                ACL_MEMCPY_DEVICE_TO_HOST);
  if (copy == ACL_SUCCESS && status.magic == kUmdkControlMagic &&
      op == UmdkControlOp::kPrepareGather) {
    args_.session = status.session;
    args_.owner_pid = status.owner_pid;
  }
  if (launch != 0 || sync != ACL_SUCCESS || copy != ACL_SUCCESS ||
      status.magic != kUmdkControlMagic || status.result != 0) {
    if (copy == ACL_SUCCESS && status.magic == kUmdkControlMagic) {
      std::fprintf(stderr,
                   "GATHER_UMDK_SETUP_ERROR op=%u workers=%u host_device='%s' "
                   "result=%d errno=%d stage=%s detail=%s\n",
                   static_cast<unsigned>(op), args_.worker_count, args_.device, status.result,
                   status.error_number, status.stage, status.detail);
    }
    return Status::Internal("GatherRead UMDK setup/cleanup failed");
  }
  if (op == UmdkControlOp::kDestroy) args_.session = 0;
  return Status::OK();
}

Status UmdkSession::Shutdown() {
  connected_ = false;  // A partial teardown must never be reused as a live session.
  if (args_.session != 0) {
    UmdkControlStatus status{};
    auto result = Run(UmdkControlOp::kDestroy, status);
    if (!result.ok()) return result;
  }
  if (args_.status != 0) (void)aclrtFree(reinterpret_cast<void*>(args_.status));
  if (transfers_ != nullptr) (void)aclrtFree(transfers_);
  if (stream_ != nullptr) (void)aclrtDestroyStream(stream_);
  args_ = {};
  stream_ = nullptr;
  connected_ = false;
  transfers_ = nullptr;
  transfers_capacity_ = 0;
  registered_buffers_.clear();
  return Status::OK();
}

Status UmdkSession::Ensure(peer::PeerLink& link, std::uint64_t region_base,
                           const std::string& device, std::uint32_t eid_index, void* output,
                           std::size_t output_bytes, std::uint32_t workers) {
  if (connected_ && args_.remote_address == region_base && args_.worker_count == workers &&
      device == args_.device && eid_index == args_.eid_index) {
    const auto address = reinterpret_cast<std::uint64_t>(output);
    for (const auto& buffer : registered_buffers_)
      if (address >= buffer.first && address - buffer.first <= buffer.second &&
          output_bytes <= buffer.second - (address - buffer.first))
        return Status::OK();
    args_.local_buffer = address;
    args_.buffer_bytes = output_bytes;
    UmdkControlStatus registered{};
    auto result = Run(UmdkControlOp::kRegisterGatherBuffer, registered);
    if (result.ok()) registered_buffers_.push_back({address, output_bytes});
    return result;
  }
  auto result = Shutdown();
  if (!result.ok()) return result;
  if (device.size() >= sizeof(args_.device))
    return Status::InvalidArgument("GatherRead UMDK device name is too long");
  peer::PeerRegion region;
  result = link.LookupRegion(region_base, &region);
  if (!result.ok()) return result;
  if (aclrtCreateStream(&stream_) != ACL_SUCCESS)
    return Status::Internal("failed to create GatherRead UMDK setup stream");
  void* status_buffer = nullptr;
  if (aclrtMalloc(&status_buffer, sizeof(UmdkControlStatus), ACL_MEM_MALLOC_HUGE_FIRST) !=
      ACL_SUCCESS) {
    (void)Shutdown();
    return Status::Internal("failed to allocate GatherRead UMDK status");
  }
  args_.status = reinterpret_cast<std::uint64_t>(status_buffer);
  args_.local_buffer = reinterpret_cast<std::uint64_t>(output);
  args_.buffer_bytes = output_bytes;
  args_.worker_count = workers;
  args_.eid_index = eid_index;
  std::snprintf(args_.device, sizeof(args_.device), "%s", device.c_str());
  args_.token = region.token_value;
  args_.remote_address = region.base;
  std::copy(region.eid.begin(), region.eid.end(), args_.route_eid);
  UmdkControlStatus prepared{};
  result = Run(UmdkControlOp::kPrepareGather, prepared);
  if (!result.ok()) return result;
  if (prepared.key_bytes != 28 || region.segment_key_bytes > sizeof(args_.segment))
    return Status::Internal("invalid GatherRead UMDK peer descriptor");
  args_.segment_bytes = region.segment_key_bytes;
  std::copy_n(region.segment_key.begin(), args_.segment_bytes, args_.segment);
  for (std::uint32_t queue = 0; queue < workers + 1U; ++queue) {
    const bool control = queue == 0U;
    const auto index = control ? kUmdkControlQueueIndex : queue - 1U;
    args_.worker_index = index;
    auto descriptor = prepared;
    if (index != 0) {
      result = Run(UmdkControlOp::kQueryGatherWorker, descriptor);
      if (!result.ok()) return result;
    }
    std::array<std::uint8_t, peer::kQpKeyBytes> local_key{}, peer_key{};
    std::copy_n(descriptor.key, descriptor.key_bytes, local_key.begin());
    result = link.CreateChannel(region.base, local_key.data(), descriptor.key_bytes, descriptor.eid,
                                region.token_value, descriptor.tp, 1, &peer_key, &args_.key_bytes,
                                &args_.peer_tp, &args_.peer_psn, true);
    if (!result.ok()) return result;
    if (args_.key_bytes > sizeof(args_.peer_key))
      return Status::Internal("invalid GatherRead UMDK Jetty key size");
    std::copy_n(peer_key.begin(), args_.key_bytes, args_.peer_key);
    UmdkControlStatus connected{};
    result = Run(UmdkControlOp::kConnect, connected);
    if (!result.ok()) return result;
  }
  args_.worker_index = 0;
  connected_ = true;
  registered_buffers_.push_back({reinterpret_cast<std::uint64_t>(output), output_bytes});
  return Status::OK();
}

Status UmdkSession::Write(const std::vector<UmdkTransfer>& transfers, aclrtStream stream) {
  if (!connected_ || stream == nullptr || transfers.empty() || transfers.size() > kGatherMaxSpans)
    return Status::InvalidArgument(
        "UMDK WRITE requires a connected session, stream and transfer spans");
  const auto bytes = transfers.size() * sizeof(UmdkTransfer);
  if (bytes > transfers_capacity_) {
    void* allocation = nullptr;
    if (aclrtMalloc(&allocation, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
      return Status::Internal("failed to allocate UMDK WRITE descriptors");
    if (transfers_ != nullptr) (void)aclrtFree(transfers_);
    transfers_ = allocation;
    transfers_capacity_ = bytes;
  }
  if (aclrtMemcpy(transfers_, transfers_capacity_, transfers.data(), bytes,
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
    return Status::Internal("failed to upload UMDK WRITE descriptors");
  args_.transfer_spans = reinterpret_cast<std::uint64_t>(transfers_);
  args_.transfer_count = transfers.size();
  UmdkControlStatus status{};
  return Run(UmdkControlOp::kWriteGather, status, stream);
}

}  // namespace kvaf::aicpu_urma
