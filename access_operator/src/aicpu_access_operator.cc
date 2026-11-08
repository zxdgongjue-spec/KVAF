#include "aicpu_access_operator.h"

#include <mutex>

#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "aicpu_batch_get/include/cpu_push_transport.h"
#include "aicpu_batch_put/include/kvaf_aicpu_batch_put.h"
#include "aicpu_urma/session.h"
#include "aicpu_urma/kernel_loader.h"
namespace kvaf {
struct UmdkAccessOperator::Impl {
  explicit Impl(UmdkAccessOperatorConfig config)
      : valid_mode(config.batch_get_mode == "gather_read" ||
                   config.batch_get_mode == "cpu_push_scatter"),
        get_session({config.kernel_json, config.workers,
                 config.peer_link, config.local_eid}),
        put_session({config.kernel_json, config.workers, config.peer_link, config.local_eid}),
        get(get_session, config.token_bytes),
        put(put_session), enable_get(config.enable_get), managed_completion(config.managed_completion) {
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
    if (config.batch_get_mode == "cpu_push_scatter")
      push = std::make_unique<cpu_push::CpuPushTransport>(cpu_push::Config{
          config.peer_link, config.kernel_json, config.local_eid, config.token_bytes, config.workers});
#else
    valid_mode = config.batch_get_mode == "gather_read";
#endif
  }
  bool valid_mode;
  aicpu_urma::UmdkSession get_session, put_session;
  aicpu_urma::BatchGetTransport get;
  aicpu_urma::BatchPutTransport put;
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  std::unique_ptr<cpu_push::CpuPushTransport> push;
  Status DrainGet() { return push ? push->Drain() : get.Drain(); }
#else
  Status DrainGet() { return get.Drain(); }
#endif
  bool enable_get;
  bool managed_completion;
  RemoteRegionDesc region;
  bool initialized{false};
  void* warmup_buffer{nullptr};
  std::mutex get_mutex, put_mutex;
};
UmdkAccessOperator::UmdkAccessOperator(UmdkAccessOperatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
UmdkAccessOperator::~UmdkAccessOperator() { if (!Shutdown().ok()) (void)impl_.release(); }
Status UmdkAccessOperator::Initialize(const RemoteRegionDesc& region) {
  if (!impl_->valid_mode) return Status::InvalidArgument("unknown AICPU BatchGet mode");
  if (impl_->initialized) return Status::InvalidState("AICPU operator already initialized");
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (impl_->push) {
    auto checked = aicpu_urma::PrepareUmdkKernel(impl_->get_session.config().kernel_json, true);
    if (!checked.ok()) return checked;
  }
#endif
  auto loaded = impl_->enable_get ? impl_->get_session.Load() : impl_->put_session.Load();
  if (!loaded.ok()) return loaded;
  if (!impl_->enable_get) { impl_->region = region; impl_->initialized = true; return Status::OK(); }
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  auto ready = impl_->push ? impl_->push->Initialize(region) : impl_->get.Initialize(region);
#else
  auto ready = impl_->get.Initialize(region);
#endif
  if (!ready.ok()) return ready;
  impl_->region = region;
  impl_->initialized = true;
  return Status::OK();
}
Status UmdkAccessOperator::Warmup(const ExecutionWarmup& request) {
  if (!impl_->initialized) return Status::NotInitialized("AICPU provider not initialized");
  if (impl_->enable_get) {
    std::lock_guard<std::mutex> lock(impl_->get_mutex);
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
    if (impl_->push) return Status::OK();
#endif
    return impl_->get.Warmup();
  }
  std::lock_guard<std::mutex> lock(impl_->put_mutex);
  if (!impl_->warmup_buffer &&
      aclrtMalloc(&impl_->warmup_buffer, 64, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
    return Status::OutOfMemory("cannot allocate session setup buffer");
  auto s = impl_->put_session.Ensure(impl_->region.base, impl_->warmup_buffer, 64);
  return s.ok() ? impl_->put.Reserve(request.request_capacity ? request.request_capacity : 65536)
                : s;
}
Status UmdkAccessOperator::Shutdown() {
  std::scoped_lock lock(impl_->get_mutex, impl_->put_mutex);
  // Push shutdown must stop the remote DMA producer even after a kernel failure.
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (impl_->push) {
    auto stopped = impl_->push->Shutdown();
    if (!stopped.ok()) return stopped;
  }
#endif
  auto drained = impl_->DrainGet();
  if (!drained.ok()) return drained;
  auto closed = impl_->get_session.Shutdown();
  if (!closed.ok()) return closed;
  closed = impl_->put_session.Shutdown();
  if (!closed.ok()) return closed;
  auto released = impl_->get.Shutdown();
  if (!released.ok()) return released;
  auto put_released = impl_->put.Shutdown();
  if (!put_released.ok()) return put_released;
  if (impl_->warmup_buffer) {
    if (aclrtFree(impl_->warmup_buffer) != ACL_SUCCESS)
      return Status::Internal("cannot free session setup buffer");
    impl_->warmup_buffer = nullptr;
  }
  impl_->initialized = false;
  return Status::OK();
}
Status UmdkAccessOperator::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  std::lock_guard<std::mutex> lock(impl_->get_mutex);
  if (!impl_->initialized) return Status::NotInitialized("AICPU operator is not initialized");
  if (impl_->managed_completion && exec.execution.stream) {
    aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI graph = nullptr;
    if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(exec.execution.stream), &capture, &graph) != ACL_SUCCESS)
      return Status::Internal("cannot query graph capture state");
    if (capture == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)
      return Status::NotImplemented("per-call AICPU graph leases are not implemented");
  }
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (impl_->push) return impl_->push->Execute(exec);
#endif
  auto status = impl_->get.Execute(exec);
  if (status.ok() && exec.execution.synchronize && impl_->managed_completion)
    status = impl_->get_session.ReleaseBuffers();
  return status;
}
Status UmdkAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& exec) {
  std::lock_guard<std::mutex> lock(impl_->put_mutex);
  if (!impl_->initialized) return Status::NotInitialized("AICPU operator is not initialized");
  auto loaded = impl_->put_session.Load();
  if (!loaded.ok()) return loaded;
  auto status = impl_->put.Execute(impl_->region, exec);
  if (status.ok() && impl_->managed_completion) status = impl_->put_session.ReleaseBuffers();
  return status;
}
Status UmdkAccessOperator::DrainGet() {
  std::lock_guard<std::mutex> lock(impl_->get_mutex);
  auto status = impl_->DrainGet();
  if (status.ok() && impl_->managed_completion) status = impl_->get_session.ReleaseBuffers();
  return status;
}
}  // namespace kvaf
