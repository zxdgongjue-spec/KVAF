// Host-only regression for real Region ownership and CPU Pull grant/lease rules.
#include "cache_region_service.h"
#include "offload_service.h"
#include "remote_pull_protocol.h"
#include "remote_pull_service.h"
#include <cstdlib>
#include <iostream>
using namespace kvaf;
namespace ex = kvaf::execution;
namespace {
void Check(bool value, const char* message) {
  if (!value) { std::cerr << message << '\n'; std::abort(); }
}
struct State {
  alignas(64) char memory[128]{};
  unsigned preparations{}, closes{}, sources{}, reads{};
  bool fail_prepare{}, unknown_close{};
};
class Backend final : public offload::Backend {
 public:
  Backend(State& state, std::shared_ptr<void> lease) : state_(state), lease_(std::move(lease)) { ++state_.sources; }
  ~Backend() override { Check(Close().ok(), "test must resolve uncertain source before destroying it"); }
  Status Allocate(std::uint64_t, offload::MemoryDesc*) override { return Status::NotImplemented("unused"); }
  Status Open(const offload::ConnectionDesc&, const offload::MemoryDesc&, offload::ConnectionDesc*) override { return Status::OK(); }
  Status Read(const std::vector<offload::Span>&) override { ++state_.reads; return Status::OK(); }
  Status ReleaseSource() override { return Status::OK(); }
  Status Close() override {
    if (state_.unknown_close) return Status::CompletionUnknown("injected DMA uncertainty");
    if (lease_) { --state_.sources; lease_.reset(); }
    return Status::OK();
  }
  std::uint64_t Checksum(const std::vector<offload::Span>&) const override { return 0; }
 private:
  State& state_;
  std::shared_ptr<void> lease_;
};
class Allocation final : public ex::Allocation {
 public:
  explicit Allocation(State& state) : state_(state) {}
  Status Close() override {
    Check(!state_.sources, "DDR must outlive every CPU Pull source");
    ++state_.closes;
    return Status::OK();
  }
  Status PrepareCpuPull(std::uint32_t, std::uint32_t token, std::uint64_t bytes, peer::PeerRegion* out) override {
    ++state_.preparations;
    Check(token && token != 2, "CPU Pull grant must differ from ordinary remote MR token");
    Check(bytes == 64, "destination MR excludes auxiliary allocation bytes");
    if (state_.fail_prepare) return Status::Internal("injected partial MR setup");
    *out = descriptor; out->length = bytes; out->token_value = token;
    return Status::OK();
  }
  std::unique_ptr<offload::Backend> CreateCpuPullBackend(std::shared_ptr<void> lease) override {
    return std::make_unique<Backend>(state_, std::move(lease));
  }
 private:
  State& state_;
};
void Test(bool fail_prepare) {
  State state; state.fail_prepare = fail_prepare;
  ex::CacheRegionService service(ex::kUrma | ex::kCpuPull,
      [&](const ex::Request& q, std::unique_ptr<ex::Allocation>* out) {
        auto allocation = std::make_unique<Allocation>(state);
        allocation->descriptor.base = reinterpret_cast<std::uintptr_t>(state.memory);
        allocation->descriptor.length = q.logical_bytes + q.auxiliary_bytes;
        *out = std::move(allocation); return Status::OK();
      }, [](const ex::Request&) -> std::unique_ptr<ex::Session> { return {}; });
  ex::Request q;
  const auto capabilities = service.Handle(q);
  q.op = ex::Op::CreateRegion; q.client = 42; q.request = 1; q.epoch = capabilities.epoch;
  q.logical_bytes = 64; q.slot_size = 16; q.auxiliary_bytes = 64;
  const auto created = service.Handle(q);
  Check(created.status.ok(), "create true Region");
  q.op = ex::Op::PrepareCpuPullRegion; q.region = created.region; q.request = 2;
  auto bad = q; ++bad.client;
  Check(!service.Handle(bad).status.ok(), "different owner cannot mint grant");
  bad = q; bad.epoch = 0;
  Check(!service.Handle(bad).status.ok(), "missing epoch cannot mint grant");
  const auto prepared = service.Handle(q);
  const auto repeated = service.Handle(q);
  Check(state.preparations == 1, "preparation must execute once even after failure");
  Check(prepared.status.code() == repeated.status.code(), "retry retains preparation outcome");
  bad = q; ++bad.candidate;
  Check(!service.Handle(bad).status.ok(), "candidate cannot change after preparation");
  remote_pull::RegionInfo info;
  std::unique_ptr<offload::Backend> backend;
  if (!fail_prepare) {
    Check(prepared.status.ok() && prepared.binding.length == 64, "prepared logical DDR binding");
    Check(service.DescribeCpuPullRegion(created.region, &info).ok(), "describe prepared Region");
    auto spoof = info; ++spoof.token;
    Check(!service.AcquireCpuPullBackend(spoof), "wrong grant cannot acquire allocation");
    backend = service.AcquireCpuPullBackend(info);
    Check(bool(backend), "authorized source holds allocation lease");
  } else {
    Check(!service.DescribeCpuPullRegion(created.region, &info).ok(), "failed preparation is not attachable");
  }
  q.op = ex::Op::ReleaseRegion; q.request = 3;
  if (backend) {
    Check(service.Handle(q).status.code() == StatusCode::kBusy, "idle registered source blocks Region release");
    Check(!service.AcquireCpuPullBackend(info), "close intent stops new sources");
    Check(service.DescribeCpuPullRegion(created.region, &info).ok() && info.closing,
          "existing source remains describable for recovery after release request");
    state.unknown_close = true;
    Check(!backend->Close().ok() && state.closes == 0, "uncertain DMA retains allocation");
    Check(service.Handle(q).status.code() == StatusCode::kBusy, "uncertainty retains Region lease");
    state.unknown_close = false;
    Check(backend->Close().ok(), "source quiescence releases lease");
    backend.reset();
  }
  Check(service.Handle(q).status.ok() && state.closes == 1, "Region closes after source teardown");
  Check(service.Handle(q).status.ok() && state.closes == 1, "Region close is idempotent");
  Check(service.Shutdown().ok(), "service shutdown");
}
void TestRemoteShutdown() {
  namespace rp = kvaf::remote_pull;
  State state;
  ex::CacheRegionService regions(ex::kUrma | ex::kCpuPull,
      [&](const ex::Request& q, std::unique_ptr<ex::Allocation>* out) {
        auto allocation = std::make_unique<Allocation>(state);
        allocation->descriptor.base = reinterpret_cast<std::uintptr_t>(state.memory);
        allocation->descriptor.length = q.logical_bytes + q.auxiliary_bytes;
        *out = std::move(allocation); return Status::OK();
      }, [](const ex::Request&) -> std::unique_ptr<ex::Session> { return {}; });
  rp::Service service([&](std::uint64_t id, rp::RegionInfo* out) {
    return regions.DescribeCpuPullRegion(id, out);
  }, [&](const rp::RegionInfo& info) { return regions.AcquireCpuPullBackend(info); });
  ex::Request q;
  q.epoch = regions.Handle(q).epoch; q.client = 19; q.request = 1;
  q.op = ex::Op::CreateRegion; q.logical_bytes = 64; q.auxiliary_bytes = 64; q.slot_size = 16;
  q.region = regions.Handle(q).region;
  q.op = ex::Op::PrepareCpuPullRegion; q.request = 2;
  auto prepared = regions.Handle(q); Check(prepared.status.ok(), "prepare service grant");
  const auto dispatch = [&](rp::Command command, std::uint64_t task) {
    auto r = rp::MakeRequest(command);
    r.region = q.region; r.epoch = q.epoch; r.token = prepared.binding.token_value;
    r.source_id = 555; r.task = task; r.operation = task;
    r.source.bytes = 64; r.source.token = r.connection.token = r.token;
    r.timeout_ms = 1000;
    std::vector<rp::Span> spans;
    if (command == rp::Command::SubmitAndWait) spans = {{0, 0, 16}};
    r.count = spans.size(); r.header.bytes += spans.size() * sizeof(rp::Span);
    return service.Dispatch(r, spans);
  };
  Check(dispatch(rp::Command::Attach, 0).result == rp::Result::Ok, "source attaches to true Region");
  Check(dispatch(rp::Command::Bind, 0).result == rp::Result::Ok, "bind source");
  auto first = dispatch(rp::Command::SubmitAndWait, 1);
  Check(first.result == rp::Result::Ok && first.reserved && first.state == rp::State::Done,
        "single RPC returns accepted completion");
  Check(!service.Shutdown().ok(), "idle registered source keeps service alive for detach");
  q.op = ex::Op::ReleaseRegion;
  Check(regions.Handle(q).status.code() == StatusCode::kBusy, "registered source retains parent Region");
  Check(regions.Shutdown().code() == StatusCode::kBusy,
        "partial region shutdown retains active source and recovery access");
  auto duplicate = dispatch(rp::Command::SubmitAndWait, 1);
  Check(duplicate.reserved && duplicate.state == rp::State::Done && state.reads == 1,
        "closing must not turn previously accepted operation into rejection");
  auto rejected = dispatch(rp::Command::SubmitAndWait, 2);
  Check(!rejected.reserved && rejected.state == rp::State::Cancelled && state.reads == 1,
        "closing fences unseen submissions without DMA");
  Check(dispatch(rp::Command::Query, 2).state == rp::State::Cancelled,
        "recovery Query observes closing rejection fence");
  Check(dispatch(rp::Command::Detach, 0).result == rp::Result::Ok,
        "Detach is allowed after shutdown and Region release were requested");
  Check(service.Shutdown().ok() && regions.Handle(q).status.ok(), "child closes before allocation");
  Check(dispatch(rp::Command::Detach, 0).state == rp::State::Closed,
        "lost Detach reply remains recoverable after allocation release");
}
}
int main() { Test(false); Test(true); TestRemoteShutdown(); std::cout << "CPU Pull Region lease checks passed\n"; }
