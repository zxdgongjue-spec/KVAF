#include "hbm_exporter.h"
#include "offload_runtime.h"
#include "offload_client.h"
#include "peer_link.h"
#include "npu_eid_discovery.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "direct_hccp/target_channel.h"
#include <acl/acl.h>
#include <chrono>
#include <cstdio>
#include <random>
#include <sstream>
#include <thread>

namespace kvaf {
namespace {
namespace h = direct_hccp;
namespace p = offload;
using detail::Error;
using detail::EidHex;
using detail::Nonce;
using detail::ParseEid;
using detail::Exporter;

class PullOperator final : public AccessOperator, public DdrMemoryAllocator {
 public:
  explicit PullOperator(OffloadRuntimeOptions options) : options_(std::move(options)), region_id_(Nonce()) {}
  ~PullOperator() override {
    if (source_) (void)source_.release(); // Owner has already attempted explicit recovery/Shutdown.
  }
  std::string address;
  std::uint64_t task() const { return task_; }
  bool pending() const { return source_ != nullptr; }
  const p::Endpoint& cpu() const { return cpu_; }
  void StartupProbe(bool enabled) { startup_probe_ = enabled; }
  Status Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) override {
    auto r = p::MakeRequest(p::Command::Allocate);
    r.region = region_id_; r.bytes = request.ddr_bytes; r.slot_size = request.slot_size;
    r.candidate = options_.cpu_candidate;
    allocation_request_ = request; allocation_unknown_ = true;
    p::Response response{};
    auto status = p::Exchange(address, r, {}, &response);
    if (!status.ok()) return status;
    if (response.result == p::Result::InDoubt)
      return Status::InDoubt("CPU allocation/cleanup unresolved; candidate fallback stopped");
    if (response.result != p::Result::Ok) {
      allocation_unknown_ = false;
      return Error("allocate CPU Offload region");
    }
    if (!response.epoch) return Status::InDoubt("CPU allocation omitted service epoch");
    allocation_unknown_ = false;
    epoch_ = response.epoch; cpu_ = response.endpoint;
    if (!response.base || response.bytes != request.ddr_bytes) return Error("invalid CPU region allocation");
    region_ = {response.base, response.bytes, request.slot_size, DeviceKind::kCpu};
    *region = region_; return Status::OK();
  }
  Status Release(const RemoteRegionDesc&) override {
    if (pending()) return Status::InDoubt("source lease still active");
    if (allocation_unknown_) {
      // Recover the original Allocate identity after a lost ACK. Never make a
      // fresh allocation/attempt until the old region is accounted for.
      RemoteRegionDesc ignored{};
      const auto status = Allocate(allocation_request_, &ignored);
      if (allocation_unknown_) return status;
    }
    if (!epoch_) return Status::OK();
    p::Response response{}; auto status = Call(p::Command::Close, {}, &response);
    if (!status.ok()) return status;
    if (response.result != p::Result::Ok && response.result != p::Result::NotFound)
      return Status::InDoubt("CPU region close not confirmed");
    epoch_ = 0; return Status::OK();
  }
  Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
  Status Shutdown() override { return pending() ? Status::InDoubt("pending HBM export") : Status::OK(); }
  Status ExecuteBatchGet(const BatchGetExecRequest&) override { return Status::NotImplemented("Offload-only runtime"); }
  Status ExecuteBatchPut(const BatchPutExecRequest&) override { return Status::NotImplemented("use Offload"); }
  Status ExecuteOffload(const OffloadExecRequest& request) override {
    if (pending()) return Status::Busy("source is still leased");
    outcome_ = Error("Pull was not submitted");
    if (request.request->keys.size() > p::kMaxSpans) return Status::InvalidArgument("Offload supports at most 4096 spans");
    spans_.clear();
    for (std::size_t i = 0; i < request.put_desc->entries.size(); ++i) {
      const auto destination = request.put_desc->entries[i].address;
      if (destination < region_.base) return Error("destination below region");
      spans_.push_back({request.request->input_offsets[i], destination-region_.base, request.request->lengths[i]});
    }
    if (!p::ValidateSpans(spans_, request.request->input.bytes, region_.length)) return Error("invalid Pull ranges");
    ++task_; opened_ = false;
    source_ = std::make_unique<Exporter>();
    auto status = source_->Open(request.request->input, options_.npu_eid, cpu_);
    if (!status.ok()) { outcome_ = status; return Cleanup(); }
    opened_ = true;
    p::Response response{};
    status = Call(p::Command::Open, {}, &response);
    if (!status.ok()) return status;
    if (response.result != p::Result::Ok) return ResolveOffload();
    status = source_->Bind(response.endpoint);
    if (status.ok()) status = source_->Ready(request.execution.stream);
    if (!status.ok()) { outcome_ = status; return Cleanup(); }
    status = Call(startup_probe_ ? p::Command::Probe : p::Command::Submit, spans_, &response);
    if (!status.ok()) return status;
    return ResolveOffload();
  }
  Status ResolveOffload() override {
    if (!source_) return outcome_;
    if (!opened_) return Cleanup();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(options_.timeout_ms);
    for (;;) {
      p::Response response{}; auto status = Call(p::Command::Query, {}, &response);
      if (!status.ok()) return status;
      if (response.result != p::Result::Ok) return Status::InDoubt("cannot identify CPU task; retain source");
      if (response.state == p::State::Done) { outcome_ = Status::OK(); return Cleanup(); }
      if (response.state == p::State::Failed) {
        outcome_ = Error("CPU task failed; inspect CPU_PULL_OPEN_FAILED/CPU_PULL_TASK_END; region=" +
                         std::to_string(region_id_) + " task=" + std::to_string(task_));
        return Cleanup();
      }
      if (response.state == p::State::Open) return Cleanup();
      const auto identity = [this] { return " region=" + std::to_string(region_id_) + " task=" + std::to_string(task_); };
      if (response.state == p::State::InDoubt)
        return Status::InDoubt("CPU Pull transport quarantined; inspect CPU_PULL_TASK_END on CPU service;" +
                               identity() + "; retain source; resolve_offload does not resubmit READ");
      if (response.state != p::State::Running)
        return Status::InDoubt("unexpected CPU Pull state=" + std::to_string(static_cast<unsigned>(response.state)) +
                               identity() + "; retain source");
      if (std::chrono::steady_clock::now() >= deadline)
        return Status::InDoubt("CPU Pull is still running after timeout_ms=" + std::to_string(options_.timeout_ms) +
                               identity() + "; call resolve_offload with source retained");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  Status Checksum(std::uint64_t* checksum) {
    if (!checksum || !task_ || pending()) return Status::InvalidState("no completed Offload");
    p::Response response{}; auto status = Call(p::Command::Checksum, {}, &response);
    if (!status.ok()) return status;
    if (response.result != p::Result::Ok) return Error("CPU checksum unavailable");
    *checksum = response.checksum; return Status::OK();
  }
 private:
  Status Call(p::Command command, const std::vector<p::Span>& spans, p::Response* response) {
    auto r = p::MakeRequest(command);
    r.region = region_id_; r.epoch = epoch_; r.task = task_;
    if (source_) r.source = source_->descriptor;
    return p::Exchange(address, r, spans, response);
  }
  Status Cleanup() {
    if (opened_) {
      p::Response response{}; auto status = Call(p::Command::ReleaseSource, {}, &response);
      if (!status.ok()) return status;
      if (response.result != p::Result::Ok) return Status::InDoubt("CPU source release not confirmed");
      opened_ = false;
    }
    auto status = source_->Close();
    if (!status.ok()) return Status::InDoubt(status.message());
    source_.reset();
    if (outcome_.code() == StatusCode::kInDoubt) outcome_ = Error("source setup failed and is now quiescent");
    return outcome_;
  }
  OffloadRuntimeOptions options_;
  std::uint64_t region_id_, epoch_{}, task_{};
  p::Endpoint cpu_{};
  RemoteRegionDesc region_{};
  RuntimeInitRequest allocation_request_{};
  bool allocation_unknown_{};
  std::unique_ptr<Exporter> source_;
  bool opened_{}, startup_probe_{};
  Status outcome_;
  std::vector<p::Span> spans_;
};
}

namespace {
// Each candidate owns a fresh metadata manager and DDR pool. A failed probe
// must finish Shutdown before the next attempt can be constructed.
struct Attempt {
  OffloadRuntimeOptions options;
  std::unique_ptr<MetaManager> meta{CreateInMemoryMetaManager()};
  PullOperator transport;
  RuntimeCore core;
  void* probe{};
  bool ready{};
  explicit Attempt(OffloadRuntimeOptions o) : options(o), transport(std::move(o)), core(*meta, transport, transport) {}
  bool pending() const { return core.pending_offload() || transport.pending(); }
  Status Shutdown() {
    if (pending()) return Status::InDoubt("resolve pending Offload before shutdown");
    ready = false;
    if (core.initialized()) { auto s = core.Shutdown(); if (!s.ok()) return s; }
    // Also recover a lost Allocate/Close acknowledgement.
    auto s = transport.Release({}); if (!s.ok()) return s;
    if (probe) { if (aclrtFree(probe) != ACL_SUCCESS) return Error("free startup probe"); probe = nullptr; }
    return Status::OK();
  }
  Status Initialize(std::uint64_t bytes, std::uint64_t slot) {
    auto status = core.Initialize({bytes, slot, DeviceKind::kCpu});
    if (!status.ok()) return status;
    // Match the reference's first READ: a 64-byte control word at HBM + 64.
    const std::size_t length = std::min<std::uint64_t>(slot, 64);
    constexpr std::size_t allocation_bytes = 4096, source_offset = 64;
    aclrtStream stream{};
    if (aclrtMalloc(&probe, allocation_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
        aclrtMemset(probe, allocation_bytes, 0xa5, allocation_bytes) != ACL_SUCCESS ||
        aclrtCreateStream(&stream) != ACL_SUCCESS)
      return Error("allocate startup Pull probe");
    transport.StartupProbe(true);
    status = core.Offload({{0}, {probe, allocation_bytes, DeviceKind::kNpu}, {source_offset}, {length}}, {stream});
    transport.StartupProbe(false);
    const auto destroy = aclrtDestroyStream(stream);
    if (!status.ok()) return status;
    if (destroy != ACL_SUCCESS) return Error("destroy startup probe stream");
    std::uint64_t checksum{};
    status = transport.Checksum(&checksum);
    std::vector<std::uint8_t> expected(length, 0xa5);
    if (status.ok() && checksum != p::Hash(expected.data(), expected.size())) status = Error("startup CPU DDR checksum mismatch");
    if (!status.ok()) return status;
    status = core.BatchFree({{0}});
    if (!status.ok()) return status;
    if (aclrtFree(probe) != ACL_SUCCESS) return Error("free startup probe");
    probe = nullptr; ready = true;
    return Status::OK();
  }
};
}
struct OffloadRuntime::Impl {
  OffloadRuntimeOptions options;
  std::string address;
  std::unique_ptr<Attempt> attempt;
  explicit Impl(OffloadRuntimeOptions o) : options(std::move(o)), attempt(new Attempt(options)) {}
  Status Device() const {
    int32_t current = -1;
    return aclrtGetDevice(&current) == ACL_SUCCESS && current == static_cast<int32_t>(options.device_id)
        ? Status::OK() : Status::InvalidState("select the configured NPU device before using OffloadRuntime");
  }
};
OffloadRuntime::OffloadRuntime(OffloadRuntimeOptions options) : impl_(new Impl(std::move(options))) {}
OffloadRuntime::~OffloadRuntime() {
  if (!Shutdown().ok()) (void)impl_.release(); // Retain exports and any startup probe allocation.
}
Status OffloadRuntime::LinkTo(const LinkTarget& target) {
  if (impl_->attempt->core.initialized() || pending_offload()) return Status::InvalidState("already initialized or pending");
  if (target.address.empty()) return Status::InvalidArgument("empty CPU address");
  // A previously lost Allocate ACK must be resolved against its old service.
  auto status = impl_->attempt->Shutdown(); if (!status.ok()) return status;
  impl_->address = target.address; return Status::OK();
}
Status OffloadRuntime::Initialize(std::uint64_t bytes, std::uint64_t slot) {
  auto status = impl_->Device(); if (!status.ok()) return status;
  if (impl_->attempt->core.initialized() || pending_offload())
    return Status::InvalidState("already initialized or pending; resolve/shutdown before initializing again");
  if (!impl_->options.timeout_ms || impl_->address.empty() || !bytes || !slot || bytes % slot)
    return Status::InvalidArgument("initialize requires a link, timeout and valid DDR/slot sizes");
  status = impl_->attempt->Shutdown(); if (!status.ok()) return status;
  std::vector<npu_eid::NpuLocalEidCandidate> npu_candidates;
  if (impl_->options.npu_eid.empty()) {
    std::uint32_t physical{};
    std::string error;
    if (!h::GetCurrentNpuPhysicalDeviceId(&physical)) return Error("resolve NPU physical device");
    if (!npu_eid::DiscoverPhysicalNpuEids(physical, &npu_candidates, &error)) return Error(error);
  } else {
    npu_eid::NpuLocalEidCandidate candidate{};
    if (!ParseEid(impl_->options.npu_eid, candidate.eid.data()))
      return Status::InvalidArgument("npu_eid must contain a non-zero 32-digit hexadecimal EID");
    candidate.source = "explicit"; npu_candidates.push_back(candidate);
  }
  std::uint32_t count = 1;
  const bool auto_cpu = impl_->options.cpu_candidate == OffloadRuntimeOptions::kAutoCandidate;
  if (auto_cpu) {
    peer::PeerLink link(impl_->address);
    status = link.GetCandidateCount(&count);
    if (!status.ok()) return status;
    if (!count) return Error("CPU service reported no candidates");
  }
  std::size_t attempts = 0;
  for (std::uint32_t index = 0; index < count; ++index) {
    const auto cpu = auto_cpu ? index : impl_->options.cpu_candidate;
    for (const auto& candidate : npu_candidates) {
      auto options = impl_->options;
      options.npu_eid = npu_eid::EidToHex(candidate.eid); options.cpu_candidate = cpu;
      std::fprintf(stderr, "[KVAF_NPU] CPU_PULL_CANDIDATE_ATTEMPT attempt=%zu cpu_candidate=%u npu_eid=%s source=%s\n",
          ++attempts, cpu, options.npu_eid.c_str(), candidate.source.c_str());
      impl_->attempt = std::make_unique<Attempt>(options);
      impl_->attempt->transport.address = impl_->address;
      status = impl_->attempt->Initialize(bytes, slot);
      if (status.ok()) {
        std::fprintf(stderr, "[KVAF_NPU] CPU_PULL_CANDIDATE_SELECTED cpu_candidate=%u cpu_eid=%s npu_eid=%s probe=read_checksum_pass\n",
            cpu, EidHex(impl_->attempt->transport.cpu().eid).c_str(), options.npu_eid.c_str());
        return status;
      }
      // Unknown DMA/control results are not candidate failures. Retain this
      // exact attempt for explicit recovery, without trying another route.
      if (status.code() == StatusCode::kInDoubt || pending_offload()) {
        std::fprintf(stderr, "[KVAF_NPU] CPU_PULL_CANDIDATE_STOP cpu_candidate=%u npu_eid=%s detail=%s\n",
            cpu, options.npu_eid.c_str(), status.message().c_str());
        return Status::InDoubt(status.message() + "; candidate fallback stopped");
      }
      const auto cleanup = impl_->attempt->Shutdown();
      if (!cleanup.ok()) return Status::InDoubt("candidate cleanup not confirmed; fallback stopped: " + cleanup.message());
      std::fprintf(stderr, "[KVAF_NPU] CPU_PULL_CANDIDATE_REJECTED cpu_candidate=%u npu_eid=%s cleanup=complete detail=%s\n",
          cpu, options.npu_eid.c_str(), status.message().c_str());
    }
  }
  return Error("all " + std::to_string(attempts) + " CPU/NPU candidate pairs failed; last error: " + status.message());
}
Status OffloadRuntime::Offload(const BatchPutRequest& r, OffloadExecutionOptions e) {
  if (!initialized()) return Status::NotInitialized("OffloadRuntime startup probe has not passed");
  auto status = impl_->Device(); return status.ok() ? impl_->attempt->core.Offload(r, e) : status;
}
Status OffloadRuntime::ResolveOffload() {
  auto status = impl_->Device(); return status.ok() ? impl_->attempt->core.ResolveOffload() : status;
}
Status OffloadRuntime::Shutdown() { return impl_->attempt->Shutdown(); }
Status OffloadRuntime::LastChecksum(std::uint64_t* checksum) { return impl_->attempt->transport.Checksum(checksum); }
RuntimeCore& OffloadRuntime::core() { return impl_->attempt->core; }
bool OffloadRuntime::initialized() const { return impl_->attempt->ready; }
bool OffloadRuntime::pending_offload() const { return impl_->attempt->pending(); }
std::uint64_t OffloadRuntime::task_id() const { return impl_->attempt->transport.task(); }
std::uint32_t OffloadRuntime::device_id() const { return impl_->options.device_id; }
std::string OffloadRuntime::selected_npu_eid() const { return initialized() ? impl_->attempt->options.npu_eid : ""; }
std::uint32_t OffloadRuntime::selected_cpu_candidate() const {
  return initialized() ? impl_->attempt->options.cpu_candidate : OffloadRuntimeOptions::kAutoCandidate;
}
}
