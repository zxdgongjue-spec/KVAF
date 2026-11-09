#include "session_access_operator.h"

#include <cstdlib>
#include <iostream>

#define CHECK(c) do { if (!(c)) { std::cerr << "line " << __LINE__ << ": " #c "\n"; std::abort(); } } while (false)

namespace {
int remote_opens{}, remote_closes{}, local_initializes{}, local_warmups{}, local_closes{};
int local_owners{}, local_operations{};
bool fail_local_close{}, fail_remote_close{};
std::uint32_t warmed_capacity{};
}

namespace kvaf::peer {
struct PeerLink::ExecutionState { bool closed{false}; };
PeerLink::PeerLink(std::string address) : address_(std::move(address)) {}
std::shared_ptr<PeerLink> PeerLink::ExecutionSession(const execution::Request& request) {
  CHECK(request.put && request.transport == Transport::Urma);
  auto link = std::make_shared<PeerLink>(address_);
  link->parent_ = shared_from_this();
  link->execution_ = std::make_shared<ExecutionState>();
  return link;
}
Status PeerLink::PrepareExecution() { ++remote_opens; return Status::OK(); }
Status PeerLink::ResolveExecutionControl() { return Status::OK(); }
Status PeerLink::CloseExecution() {
  if (execution_->closed) return Status::OK();
  ++remote_closes;
  if (fail_remote_close) return Status::CompletionUnknown("remote session close unresolved");
  execution_->closed = true;
  return Status::OK();
}
}  // namespace kvaf::peer

namespace {
using namespace kvaf;
class LocalResources final : public ExecutionResourceOwner {
 public:
  LocalResources() { ++local_owners; }
  ~LocalResources() override { CHECK(local_operations == 0); --local_owners; }
  Status Initialize(const RemoteRegionDesc& region) override {
    CHECK(region.base == 4096);
    ++local_initializes;
    return Status::OK();
  }
  Status Warmup(const ExecutionWarmup& r) override {
    ++local_warmups;
    warmed_capacity = r.request_capacity;
    return Status::OK();
  }
  Status Close() override {
    if (closed_) return Status::OK();
    ++local_closes;
    if (fail_local_close) return Status::CompletionUnknown("local DMA still pending");
    closed_ = true;
    return Status::OK();
  }
 private:
  bool closed_{false};
};
class LocalOperation final : public AccessOperator {
 public:
  LocalOperation() : owner_(std::make_unique<LocalResources>()) { ++local_operations; }
  ~LocalOperation() override { CHECK(local_owners == 1); --local_operations; }
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override { return std::move(owner_); }
  Status Initialize(const RemoteRegionDesc&) override { std::abort(); }
  Status Warmup(const ExecutionWarmup&) override { std::abort(); }
  Status Shutdown() override { std::abort(); }
  Status ExecuteBatchPut(const BatchPutExecRequest&) override { return Status::OK(); }
  Status ExecuteBatchGet(const BatchGetExecRequest&) override { return Status::OK(); }
 private:
  std::unique_ptr<LocalResources> owner_;
};
}

int main() {
  using namespace kvaf;
  auto root = std::make_shared<peer::PeerLink>("fake:0");
  execution::Request request;
  request.put = true;
  request.transport = Transport::Urma;
  {
    SessionAccessOperator unused(root, request,
        [](std::shared_ptr<peer::PeerLink>) { std::abort(); return std::unique_ptr<AccessOperator>{}; }, 32);
    BatchGetRequest oversized;
    oversized.indices.bytes = 33 * sizeof(std::int32_t);
    std::uint32_t capacity{};
    CHECK(!unused.DescribeRequirements(oversized, &capacity).ok());
    auto unused_resources = unused.TakeResources();
    CHECK(unused_resources->Close().ok());
  }
  CHECK(remote_opens == 0 && remote_closes == 0 && local_owners == 0);
  auto facade = std::make_unique<SessionAccessOperator>(root, request,
      [](std::shared_ptr<peer::PeerLink>) { return std::make_unique<LocalOperation>(); }, 32);
  auto resources = facade->TakeResources();
  facade.reset(); // No initialization or teardown is delegated to the facade.
  CHECK(remote_opens == 0 && remote_closes == 0 && local_owners == 0);
  CHECK(resources->Initialize({4096, 1024, 64, DeviceKind::kCpu}).ok());
  CHECK(remote_opens == 1 && local_initializes == 1 && local_owners == 1);
  CHECK(!resources->Initialize({4096, 1024, 64, DeviceKind::kCpu}).ok());
  CHECK(remote_opens == 1 && local_initializes == 1);
  CHECK(resources->Warmup({}).ok() && warmed_capacity == 32);
  CHECK(!resources->Warmup({33, nullptr}).ok() && local_warmups == 1);
  fail_local_close = true;
  CHECK(!resources->Close().ok() && remote_closes == 0 && local_owners == 1);
  fail_local_close = false;
  fail_remote_close = true;
  CHECK(!resources->Close().ok() && remote_closes == 1 && local_owners == 1);
  const auto closed_locally = local_closes;
  fail_remote_close = false;
  CHECK(resources->Close().ok() && remote_closes == 2 && local_closes == closed_locally);
  CHECK(!resources->Warmup({}).ok());
  resources.reset();
  CHECK(local_operations == 0 && local_owners == 0 && remote_closes == 2);
  std::cout << "remote/local resource lifecycle is independent of operator adapters\n";
}
