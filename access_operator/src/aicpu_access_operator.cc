#include "aicpu_access_operator.h"

#include <mutex>

#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "aicpu_batch_get/include/cpu_push_transport.h"
#include "aicpu_batch_put/include/kvaf_aicpu_batch_put.h"
#include "aicpu_urma/session.h"
namespace kvaf {
struct UmdkAccessOperator::Impl {
  explicit Impl(UmdkAccessOperatorConfig config)
      : valid_mode(config.batch_get_mode == "gather_read" ||
                   config.batch_get_mode == "gather_read_direct" ||
                   config.batch_get_mode == "cpu_push_scatter"),
        session({config.kernel_json, config.workers,
                 config.peer_link, config.local_eid}),
        get(session, config.token_bytes, config.batch_get_mode == "gather_read_direct"),
        put(session) {
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
    if (config.batch_get_mode == "cpu_push_scatter")
      push = std::make_unique<cpu_push::CpuPushTransport>(cpu_push::Config{
          config.peer_link, config.kernel_json, config.local_eid, config.token_bytes, config.workers});
#else
    valid_mode = config.batch_get_mode == "gather_read" || config.batch_get_mode == "gather_read_direct";
#endif
  }
  bool valid_mode;
  aicpu_urma::UmdkSession session;
  aicpu_urma::BatchGetTransport get;
  aicpu_urma::BatchPutTransport put;
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  std::unique_ptr<cpu_push::CpuPushTransport> push;
  Status DrainGet() { return push ? push->Drain() : get.Drain(); }
#else
  Status DrainGet() { return get.Drain(); }
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
  auto ready = impl_->push ? impl_->push->Initialize(region) : impl_->get.Initialize(region);
#else
  auto ready = impl_->get.Initialize(region);
#endif
  if (!ready.ok()) return ready;
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
  auto drained = impl_->DrainGet();
  if (!drained.ok()) return drained;
  auto closed = impl_->session.Shutdown();
  if (!closed.ok()) return closed;
  auto released = impl_->get.Shutdown();
  if (!released.ok()) return released;
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
  return impl_->get.Execute(exec);
}
Status UmdkAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& exec) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) return Status::NotInitialized("AICPU operator is not initialized");
  auto drained = impl_->DrainGet();
  if (!drained.ok()) return drained;
  return impl_->put.Execute(impl_->region, exec);
}
}  // namespace kvaf
