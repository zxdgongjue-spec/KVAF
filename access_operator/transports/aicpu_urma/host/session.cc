#include "aicpu_urma/session.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "aicpu_urma/kernel_loader.h"
#include "put_host_timing.h"

namespace kvaf::aicpu_urma {

Status UmdkSession::CompletePending(UmdkControlStatus& status) {
  if (aclrtSynchronizeStream(pending_) != ACL_SUCCESS)
    return Status::Internal("UMDK stream completion failed; resources retained");
  if (aclrtMemcpy(&status, sizeof(status), reinterpret_cast<void*>(args_.status), sizeof(status),
                  ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS || status.magic != kUmdkControlMagic)
    return Status::Internal("UMDK completion status unavailable; resources retained");
  if (args_.op == UmdkControlOp::kWrite && status.result == 0)
    write_->completion = {PutOutcome::Succeeded, Status::OK()};
  // A recovered prepare/destroy must update ownership before teardown continues.
  if (args_.op == UmdkControlOp::kPrepare) {
    args_.session = status.session;
    args_.owner_pid = status.owner_pid;
  } else if (args_.op == UmdkControlOp::kDestroy && status.result == 0) {
    args_.session = 0;
  }
  pending_ = nullptr;
  return Status::OK();
}

Status UmdkSession::Drain() {
  if (!pending_) return Status::OK();
  UmdkControlStatus status{};
  return CompletePending(status);
}

Status UmdkSession::Run(UmdkControlOp op, UmdkControlStatus& status, aclrtStream stream,
                        Launch launch_kernel, PutProviderTiming* timing) {
  PutHostTimingScope submit_timing(timing, &PutProviderTiming::submit_ns);
  if (failed_ && op != UmdkControlOp::kDestroy)
    return Status::InvalidState("UMDK session failed; shut down before reuse");
  if (stream == nullptr) stream = stream_;
  args_.op = op;
  auto* device_status = reinterpret_cast<void*>(args_.status);
  if (aclrtMemset(device_status, sizeof(status), 0, sizeof(status)) != ACL_SUCCESS)
    return Status::Internal("failed to clear UMDK setup status");
  if (op == UmdkControlOp::kWrite) {
    write_->attempted = true;
    // An error at the launch boundary cannot prove whether the command was
    // accepted. Retain its stream and original report before submitting it.
    pending_ = stream;
  }
  const auto launch = launch_kernel ? launch_kernel(&args_, stream)
                                    : LaunchUmdkKernel("KvafAicpuUmdkControlSessionsV5", &args_,
                                                       sizeof(args_), stream);
  if (launch != 0) {
    failed_ = true;
    return Status::Internal("UMDK kernel launch failed");
  }
  pending_ = stream;
  submit_timing.Finish();
  PutHostTimingScope completion_timing(timing, &PutProviderTiming::completion_ns);
  auto completed = CompletePending(status);
  if (!completed.ok()) {
    failed_ = true;
    return completed;
  }
  if (status.result != 0) {
    failed_ = true;
    std::fprintf(stderr,
                 "UMDK_SETUP_ERROR op=%u workers=%u "
                 "result=%d errno=%d stage=%s detail=%s\n",
                 static_cast<unsigned>(op), args_.worker_count, status.result,
                 status.error_number, status.stage, status.detail);
    return Status::Internal("UMDK setup/cleanup failed");
  }
  return Status::OK();
}

Status UmdkSession::Shutdown() {
  connected_ = false;  // A partial teardown must never be reused as a live session.
  auto drained = Drain();
  if (!drained.ok()) return drained;
  if (args_.session != 0) {
    UmdkControlStatus status{};
    auto result = Run(UmdkControlOp::kDestroy, status);
    if (!result.ok()) return result;
  }
  if (args_.status != 0) (void)aclrtFree(reinterpret_cast<void*>(args_.status));
  if (stream_ != nullptr) (void)aclrtDestroyStream(stream_);
  args_ = {};
  stream_ = nullptr;
  failed_ = false;
  registered_buffers_.clear();
  return Status::OK();
}

Status UmdkSession::ReleaseBuffers() {
  if (failed_ && args_.op == UmdkControlOp::kUnregisterBuffers && pending_) {
    UmdkControlStatus status{};
    auto completed = CompletePending(status);
    if (!completed.ok()) return completed;
    if (status.result != 0) return Status::CompletionUnknown("UMDK buffer unregister is unresolved");
    registered_buffers_.clear();
    failed_ = false;
  }
  auto drained = Drain();
  if (!drained.ok()) return drained;
  if (!connected_ || registered_buffers_.empty()) return Status::OK();
  UmdkControlStatus status{};
  auto released = Run(UmdkControlOp::kUnregisterBuffers, status);
  if (released.ok()) registered_buffers_.clear();
  return released;
}
Status UmdkSession::ReleaseBuffer(void* buffer, std::size_t bytes) {
  const auto address = reinterpret_cast<std::uint64_t>(buffer);
  const auto found = std::find(registered_buffers_.begin(), registered_buffers_.end(),
                              std::make_pair(address, bytes));
  if (found == registered_buffers_.end()) return Status::OK();
  if (failed_) {
    if (args_.op != UmdkControlOp::kUnregisterBuffer || !pending_ ||
        args_.local_buffer != address || args_.buffer_bytes != bytes)
      return Status::CompletionUnknown("UMDK buffer unregister is unresolved");
    UmdkControlStatus status{};
    auto s = CompletePending(status);
    if (!s.ok()) return s;
    if (status.result) return Status::CompletionUnknown("UMDK buffer unregister failed; close session");
    failed_ = false;
  } else {
    auto s = Drain();
    if (!s.ok()) return s;
    args_.local_buffer = address;
    args_.buffer_bytes = bytes;
    UmdkControlStatus status{};
    s = Run(UmdkControlOp::kUnregisterBuffer, status);
    if (!s.ok()) return s;
  }
  registered_buffers_.erase(found);
  return Status::OK();
}

Status UmdkSession::Load() {
  if (kernel_loaded_) return Status::OK();
  if (!config_.peer_link || config_.kernel_json.empty() || !config_.workers ||
      config_.workers > 8U ||
      (config_.queue_profile != UmdkQueueProfile::Gather && config_.queue_profile != UmdkQueueProfile::Put) ||
      std::all_of(config_.local_eid.begin(), config_.local_eid.end(), [](auto b) { return b == 0; }))
    return Status::InvalidArgument("UMDK requires PeerLink, kernel JSON, Host-selected EID and 1..8 workers");
  auto ready = PrepareUmdkKernel(config_.kernel_json,
      config_.queue_profile == UmdkQueueProfile::Put ? UmdkKernelKind::Put : UmdkKernelKind::Gather);
  if (ready.ok()) kernel_loaded_ = true;
  return ready;
}

Status UmdkSession::Ensure(std::uint64_t region_base, void* output, std::size_t output_bytes,
                           bool prepared_only) {
  if (failed_ || pending_)
    return Status::InvalidState("UMDK session failed; shut down before reuse");
  auto& link = *config_.peer_link;
  const auto workers = config_.workers;
  if (connected_ && args_.remote_address == region_base && args_.worker_count == workers) {
    const auto address = reinterpret_cast<std::uint64_t>(output);
    for (const auto& buffer : registered_buffers_)
      if (address >= buffer.first && address - buffer.first <= buffer.second &&
          output_bytes <= buffer.second - (address - buffer.first))
        return Status::OK();
    if (prepared_only)
      return Status::InvalidState("AICPU BatchGet output is not registered; warm up outside capture");
    args_.local_buffer = address;
    args_.buffer_bytes = output_bytes;
    UmdkControlStatus registered{};
    auto result = Run(UmdkControlOp::kRegisterBuffer, registered);
    if (result.ok()) registered_buffers_.push_back({address, output_bytes});
    return result;
  }
  if (prepared_only)
    return Status::InvalidState("AICPU BatchGet session is not prepared; warm up outside capture");
  auto result = Shutdown();
  if (!result.ok()) return result;
  peer::PeerRegion region;
  result = link.LookupRegion(region_base, &region);
  if (!result.ok()) return result;
  if (aclrtCreateStream(&stream_) != ACL_SUCCESS)
    return Status::Internal("failed to create UMDK setup stream");
  void* status_buffer = nullptr;
  if (aclrtMalloc(&status_buffer, sizeof(UmdkControlStatus), ACL_MEM_MALLOC_HUGE_FIRST) !=
      ACL_SUCCESS) {
    (void)Shutdown();
    return Status::Internal("failed to allocate UMDK status");
  }
  args_.status = reinterpret_cast<std::uint64_t>(status_buffer);
  args_.local_buffer = reinterpret_cast<std::uint64_t>(output);
  args_.buffer_bytes = output_bytes;
  args_.worker_count = workers;
  args_.queue_profile = config_.queue_profile;
  args_.token = region.token_value;
  args_.remote_address = region_base;
  std::copy(region.eid.begin(), region.eid.end(), args_.route_eid);
  std::copy(config_.local_eid.begin(), config_.local_eid.end(), args_.local_eid);
  UmdkControlStatus prepared{};
  result = Run(UmdkControlOp::kPrepare, prepared);
  if (!result.ok()) return result;
  if (!std::equal(config_.local_eid.begin(), config_.local_eid.end(), prepared.eid))
    return Status::Internal("AICPU context EID differs from the Host-selected EID");
  if (prepared.key_bytes != 28 || region.segment_key_bytes > sizeof(args_.segment))
    return Status::Internal("invalid UMDK peer descriptor");
  args_.segment_bytes = region.segment_key_bytes;
  std::copy_n(region.segment_key.begin(), args_.segment_bytes, args_.segment);
  const auto controls = UmdkControlQueueCount(config_.queue_profile, workers);
  const auto data = UmdkDataQueueCount(config_.queue_profile, workers);
  const auto prepared_index = data ? 0U : kUmdkControlQueueIndex;
  for (std::uint32_t queue = 0; queue < controls + data; ++queue) {
    const bool control = queue < controls;
    const auto index = control ? kUmdkControlQueueIndex + queue : queue - controls;
    args_.worker_index = index;
    auto descriptor = prepared;
    if (index != prepared_index) {
      result = Run(UmdkControlOp::kQueryWorker, descriptor);
      if (!result.ok()) return result;
    }
    std::array<std::uint8_t, peer::kQpKeyBytes> local_key{}, peer_key{};
    std::copy_n(descriptor.key, descriptor.key_bytes, local_key.begin());
    result = link.CreateChannel(region_base, local_key.data(), descriptor.key_bytes, descriptor.eid,
                                region.token_value, descriptor.tp, 1, &peer_key, &args_.key_bytes,
                                &args_.peer_tp, &args_.peer_psn, true);
    if (!result.ok()) return result;
    if (args_.key_bytes > sizeof(args_.peer_key))
      return Status::Internal("invalid UMDK Jetty key size");
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

Status UmdkSession::EnsureSession(std::uint64_t region_base, void* setup_buffer,
                                  std::size_t setup_bytes) {
  if (failed_ || pending_)
    return Status::InvalidState("UMDK session failed; shut down before reuse");
  if (connected_ && args_.remote_address == region_base && args_.worker_count == config_.workers)
    return Status::OK();
  return Ensure(region_base, setup_buffer, setup_bytes);
}

Status UmdkSession::ResolveWrite(PutCompletion* out) {
  if (!out) return Status::InvalidArgument("null WRITE completion output");
  if (write_->completion.outcome == PutOutcome::Succeeded) {
    // A WRITE may be complete while a later buffer-unregister command failed.
    // Keep the proven data result; AcknowledgePut retries registration cleanup.
    *out = write_->completion;
    return Status::OK();
  }
  if (!write_->attempted) {
    *out = {PutOutcome::Failed, Status::Internal("UMDK WRITE was not submitted")};
    return Status::OK();
  }
  if (args_.op != UmdkControlOp::kWrite || !pending_)
    return Status::CompletionUnknown("UMDK WRITE completion cannot be established");
  UmdkControlStatus status{};
  auto result = CompletePending(status);
  if (!result.ok()) return result;
  if (status.result != 0)
    return Status::CompletionUnknown("UMDK WRITE failed with potentially outstanding remote WRs");
  failed_ = false;
  *out = write_->completion;
  return Status::OK();
}

Status UmdkSession::WriteDescriptors(std::uint64_t descriptors, std::uint32_t count,
                                     aclrtStream stream, Launch launch, PutProviderTiming* timing) {
  if (failed_ || pending_)
    return Status::InvalidState("UMDK session failed; shut down before reuse");
  if (!connected_ || !stream || !descriptors || !count || count > kUmdkMaxTransfers)
    return Status::InvalidArgument("UMDK WRITE requires a connected session and descriptors");
  args_.transfer_spans = descriptors;
  args_.transfer_count = count;
  UmdkControlStatus status{};
  write_->attempted = false;
  write_->completion = {};
  auto result = Run(UmdkControlOp::kWrite, status, stream, launch, timing);
  if (!result.ok() && write_->completion.outcome != PutOutcome::Succeeded)
    write_->completion.result = result;
  return result;
}
}  // namespace kvaf::aicpu_urma
