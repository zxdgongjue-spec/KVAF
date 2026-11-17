#include "aicpu_access_operator.h"

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "gather_workspace.h"
#include "aicpu_batch_get/include/cpu_push_transport.h"
#include "aicpu_batch_put/include/kvaf_aicpu_batch_put.h"
#include "aicpu_urma/session.h"
namespace kvaf {
struct UmdkAccessOperator::Impl {
  explicit Impl(UmdkAccessOperatorConfig config)
      : valid_mode(config.batch_get_mode == "gather_read" ||
                   config.batch_get_mode == "cpu_push_scatter"),
        session({config.kernel_json, config.workers,
                 config.peer_link, config.local_eid}),
        put(session) {
    std::vector<std::uint32_t> widths = config.token_bytes_list;
    if (widths.empty()) widths.push_back(config.token_bytes);
    // One session serves every width; each transport gets its own slice of the
    // reserved remote gather workspace so staging never overlaps.
    std::uint64_t staging_offset = 0U;
    for (const std::uint32_t width : widths) {
      get.emplace(width, std::make_unique<aicpu_urma::BatchGetTransport>(
                             session, width, /*direct_addresses=*/false, staging_offset));
      staging_offset += GatherRemoteWorkspaceBytes(width, config.workers);
    }
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
    if (config.batch_get_mode == "cpu_push_scatter")
      push = std::make_unique<cpu_push::CpuPushTransport>(cpu_push::Config{
          config.peer_link, config.kernel_json, config.local_eid, config.token_bytes, config.workers});
#else
    valid_mode = config.batch_get_mode == "gather_read";
#endif
  }
  aicpu_urma::BatchGetTransport* GetTransport(std::uint32_t token_bytes) {
    const auto it = get.find(token_bytes);
    return it == get.end() ? nullptr : it->second.get();
  }
  bool valid_mode;
  aicpu_urma::UmdkSession session;
  std::map<std::uint32_t, std::unique_ptr<aicpu_urma::BatchGetTransport>> get;
  aicpu_urma::BatchPutTransport put;
  // Drain every width before Put reuses the session's write path.
  Status DrainTransports(bool check_status = true) {
    for (auto& entry : get) {
      const Status status = entry.second->Drain(check_status);
      if (!status.ok()) return status;
    }
    return Status::OK();
  }
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  std::unique_ptr<cpu_push::CpuPushTransport> push;
  Status DrainGet() { return push ? push->Drain() : DrainTransports(); }
#else
  Status DrainGet() { return DrainTransports(); }
#endif
  RemoteRegionDesc region;
  bool initialized{false};
  std::mutex mutex;
};
UmdkAccessOperator::UmdkAccessOperator(UmdkAccessOperatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
UmdkAccessOperator::~UmdkAccessOperator() { (void)Shutdown(); }
Status UmdkAccessOperator::Initialize(const RemoteRegionDesc& region) {
  if (!impl_->valid_mode) return Status::InvalidArgument("unknown AICPU BatchGet mode");
  if (impl_->initialized) return Status::InvalidState("AICPU operator already initialized");
  auto loaded = impl_->session.Load();
  if (!loaded.ok()) return loaded;
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (impl_->push) {
    auto ready = impl_->push->Initialize(region);
    if (!ready.ok()) return ready;
  } else
#endif
  {
    for (auto& entry : impl_->get) {
      auto ready = entry.second->Initialize(region);
      if (!ready.ok()) return ready;
    }
  }
  impl_->region = region;
  impl_->initialized = true;
  return Status::OK();
}
Status UmdkAccessOperator::Shutdown() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  // Push shutdown must stop the remote DMA producer even after a kernel failure.
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (impl_->push) {
    auto stopped = impl_->push->Shutdown();
    if (!stopped.ok()) return stopped;
  }
#endif
  // Gather teardown must still unregister buffers after a reported protocol
  // error; a failed stream synchronization continues to retain resources.
  auto drained = impl_->DrainTransports(false);
  if (!drained.ok()) return drained;
  auto closed = impl_->session.Shutdown();
  if (!closed.ok()) return closed;
  for (auto& entry : impl_->get) {
    auto released = entry.second->Shutdown();
    if (!released.ok()) return released;
  }
  auto put_released = impl_->put.Shutdown();
  if (!put_released.ok()) return put_released;
  impl_->initialized = false;
  return Status::OK();
}
Status UmdkAccessOperator::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) return Status::NotInitialized("AICPU operator is not initialized");
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (impl_->push) return impl_->push->Execute(exec);
#endif
  const auto token_bytes = static_cast<std::uint32_t>(
      exec.request ? exec.request->output_block_bytes : 0U);
  auto* transport = impl_->GetTransport(token_bytes);
  if (transport == nullptr)
    return Status::InvalidArgument("AICPU BatchGet has no transport for token_bytes=" +
                                   std::to_string(token_bytes));
  return transport->Execute(exec);
}
Status UmdkAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& exec) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) return Status::NotInitialized("AICPU operator is not initialized");
  auto drained = impl_->DrainGet();
  if (!drained.ok()) return drained;
  return impl_->put.Execute(impl_->region, exec);
}
Status UmdkAccessOperator::CheckBatchGetStatus(void* stream) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) return Status::NotInitialized("AICPU operator is not initialized");
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (impl_->push) return impl_->push->Check(stream);
#endif
  // A transport that never saw this stream answers kInvalidState. Keep the
  // first real failure instead of masking it with the generic message.
  Status first_error = Status::InvalidArgument("no AICPU BatchGet submitted on this stream");
  for (auto& entry : impl_->get) {
    const auto status = entry.second->Check(static_cast<aclrtStream>(stream));
    if (status.ok()) return status;
    if (status.code() != StatusCode::kInvalidState) first_error = status;
  }
  return first_error;
}
Status UmdkAccessOperator::SetBatchGetKernelProfiling(bool enabled) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) return Status::NotInitialized("AICPU operator is not initialized");
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (impl_->push) return impl_->push->SetKernelProfiling(enabled);
#endif
  for (auto& entry : impl_->get) {
    const auto status = entry.second->SetKernelProfiling(enabled);
    if (!status.ok()) return status;
  }
  return Status::OK();
}
Status UmdkAccessOperator::CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) return Status::NotInitialized("AICPU operator is not initialized");
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (impl_->push) return impl_->push->CopyKernelProfiles(profiles);
#endif
  for (auto& entry : impl_->get) {
    const auto status = entry.second->CopyKernelProfiles(profiles);
    if (!status.ok()) return status;
  }
  return Status::OK();
}
}  // namespace kvaf
