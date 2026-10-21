#include "aicpu_access_operator.h"

#include <mutex>

#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "aicpu_batch_put/include/kvaf_aicpu_batch_put.h"
#include "umdk/session.h"
namespace kvaf {
struct UmdkAccessOperator::Impl {
  explicit Impl(UmdkAccessOperatorConfig config)
      : session({config.kernel_json, config.device, config.eid_index, config.workers,
                 config.peer_link}),
        get(session, config.token_bytes),
        put(session) {}
  aicpu_urma::UmdkSession session;
  aicpu_urma::BatchGetTransport get;
  aicpu_urma::BatchPutTransport put;
  RemoteRegionDesc region;
  bool initialized{false};
  std::mutex mutex;
};
UmdkAccessOperator::UmdkAccessOperator(UmdkAccessOperatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
UmdkAccessOperator::~UmdkAccessOperator() { (void)Shutdown(); }
Status UmdkAccessOperator::Initialize(const RemoteRegionDesc& region) {
  if (impl_->initialized) return Status::InvalidState("AICPU operator already initialized");
  auto loaded = impl_->session.Load();
  if (!loaded.ok()) return loaded;
  auto ready = impl_->get.Initialize(region);
  if (!ready.ok()) return ready;
  impl_->region = region;
  impl_->initialized = true;
  return Status::OK();
}
Status UmdkAccessOperator::Shutdown() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto drained = impl_->get.Drain();
  if (!drained.ok()) return drained;
  auto closed = impl_->session.Shutdown();
  if (!closed.ok()) return closed;
  auto released = impl_->get.Shutdown();
  if (!released.ok()) return released;
  impl_->put.Shutdown();
  impl_->initialized = false;
  return Status::OK();
}
Status UmdkAccessOperator::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) return Status::NotInitialized("AICPU operator is not initialized");
  return impl_->get.Execute(exec);
}
Status UmdkAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& exec) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) return Status::NotInitialized("AICPU operator is not initialized");
  auto drained = impl_->get.Drain();
  if (!drained.ok()) return drained;
  return impl_->put.Execute(impl_->region, exec);
}
}  // namespace kvaf
