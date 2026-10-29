#include "offload_runtime.h"
#include "offload_client.h"
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
Status Error(const std::string& text) { return Status::Internal("Offload: " + text); }
std::string EidHex(const std::uint8_t* eid) {
  constexpr char digits[] = "0123456789abcdef";
  std::string text;
  text.reserve(32);
  for (std::size_t i = 0; i < 16; ++i) {
    text.push_back(digits[eid[i] >> 4]);
    text.push_back(digits[eid[i] & 15]);
  }
  return text;
}
std::uint64_t Nonce() {
  std::random_device r;
  const auto n = (std::uint64_t(r()) << 32) ^ r();
  return n ? n : 1;
}
bool ParseEid(const std::string& text, std::uint8_t* eid) {
  std::string hex;
  for (char c : text) if (c != ':') hex.push_back(c);
  if (hex.size() != 32) return false;
  auto digit = [](char c) -> int {
    if (c >= '0' && c <= '9') return c-'0';
    if (c >= 'a' && c <= 'f') return c-'a'+10;
    if (c >= 'A' && c <= 'F') return c-'A'+10;
    return -1;
  };
  bool nonzero = false;
  for (std::size_t i = 0; i < 16; ++i) {
    const int a = digit(hex[2*i]), b = digit(hex[2*i+1]);
    if (a < 0 || b < 0) return false;
    eid[i] = static_cast<std::uint8_t>(a*16+b); nonzero |= eid[i] != 0;
  }
  return nonzero;
}

class Exporter {
 public:
  p::Endpoint descriptor{};
  ~Exporter() { if (!Close().ok()) std::fprintf(stderr, "OFFLOAD: quarantined HBM export\n"); }
  Status Open(const ConstBufferView& input, const std::string& eid, const p::Endpoint& cpu) {
    std::uint32_t physical{};
    if (!cpu.token) return Error("CPU Allocate omitted region token; rebuild both Offload peers");
    if (!ParseEid(eid, descriptor.eid)) return Status::InvalidArgument("npu_eid must contain 32 hex digits");
    if (!h::LoadDirectHccp(&api_)) return Error("load Direct HCCP API");
    if (!h::GetCurrentNpuPhysicalDeviceId(&physical)) return Error("resolve NPU physical device");
    if (!h::AcquireDirectHccpSession(physical, 18, &session_))
      return Error("initialize Direct HCCP session: physical=" + std::to_string(physical));
    h::CtxInitCfg cfg{}; cfg.mode = h::kOfflineHdc;
    h::CtxInitAttr attr{}; attr.phy_id = physical;
    std::memcpy(attr.ub.eid.raw, descriptor.eid, 16);
    const int context_ret = api_.ctx_init(&cfg, &attr, &context_);
    if (context_ret || !context_) {
      std::ostringstream message;
      message << "create HBM context: RaCtxInit ret=" << context_ret
              << " context_nonnull=" << (context_ != nullptr) << " mode=" << cfg.mode
              << " physical=" << physical << " eid_index=" << attr.ub.eid_index
              << " npu_eid=" << EidHex(descriptor.eid);
      return Error(message.str());
    }
    h::TokenId token{};
    if (api_.token_alloc(context_, &token, &token_) || !token_) return Error("allocate source token");
    descriptor.token = cpu.token;
    h::MrRegInfo mr{};
    mr.in.mem.addr = reinterpret_cast<std::uint64_t>(input.data); mr.in.mem.size = input.bytes;
    // Match cpu_push_host.cc's deployed HBM contract (0x29c1).
    // Offload still submits only READ operations.
    mr.in.ub.flags = h::MemRegFlags(true);
    mr.in.ub.token_value = descriptor.token; mr.in.ub.token_id_handle = token_;
    const int register_ret = api_.lmem_register(context_, &mr, &mr_);
    if (register_ret || !mr_) {
      std::ostringstream message;
      message << "register source HBM: RaCtxLmemRegister ret=" << register_ret
              << " handle_nonnull=" << (mr_ != nullptr) << " physical=" << physical
              << " addr=0x" << std::hex << mr.in.mem.addr << " flags=0x" << mr.in.ub.flags
              << std::dec << " bytes=" << input.bytes;
      return Error(message.str());
    }
    if (mr.out.key.size != 48)
      return Error("unsupported HBM segment ABI: key_bytes=" + std::to_string(mr.out.key.size) +
                   " expected=48 physical=" + std::to_string(physical));
    descriptor.segment_bytes = mr.out.key.size;
    std::memcpy(descriptor.segment, mr.out.key.value, mr.out.key.size);
    // urma_seg_t: packed UBVA {eid[16], uasid[4], va[8]}, aligned len at 32.
    std::uint64_t exported_bytes{};
    std::memcpy(&descriptor.base, descriptor.segment + 20, 8);
    std::memcpy(&exported_bytes, descriptor.segment + 32, 8);
    const bool zero_base = descriptor.base == 0;
    const bool short_segment = exported_bytes < input.bytes;
    const bool address_overflow = descriptor.base > UINT64_MAX-input.bytes;
    // Match aicpu-urma's CPU-push export: the configured EID selects a CTP
    // route, while the segment carries the provider's memory identity. Do not
    // require equality or replace either EID with the other one.
    if (zero_base || short_segment || address_overflow) {
      std::ostringstream message;
      message << "invalid exported HBM range: zero_base=" << zero_base
              << " short_segment=" << short_segment << " address_overflow=" << address_overflow
              << " physical=" << physical
              << " eid_index=" << attr.ub.eid_index << " key_bytes=" << unsigned(mr.out.key.size)
              << " requested_base=0x" << std::hex << reinterpret_cast<std::uint64_t>(input.data)
              << " exported_base=0x" << descriptor.base << std::dec
              << " requested_bytes=" << input.bytes << " exported_bytes=" << exported_bytes
              << " npu_eid=" << EidHex(descriptor.eid)
              << " segment_eid=" << EidHex(descriptor.segment);
      return Error(message.str());
    }
    descriptor.bytes = input.bytes;
    descriptor.psn = 1; // cpu_push_host.cc initializes npu_psn to 1.
    tp_cfg_.flag = h::CtpFlag(); tp_cfg_.transport_mode = h::kReliableMessage;
    std::memcpy(tp_cfg_.local_eid.raw, descriptor.eid, 16);
    std::memcpy(tp_cfg_.peer_eid.raw, cpu.eid, 16);
    if (api_.get_tp_info_async(context_, &tp_cfg_, &tp_info_, &tp_count_, &async_) || !async_)
      return Error("submit source CTP query");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (async_) {
      auto status = PollTp();
      if (status.code() != StatusCode::kBusy) {
        if (!status.ok()) return status;
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) return Status::InDoubt("source CTP query is pending");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Query CTP before creating the target channels, as in the reference.
    for (std::size_t index = 0; index < channels_.size(); ++index) {
      auto& channel = channels_[index];
      channel = std::make_unique<h::TargetChannel>(api_, context_);
      const auto status = channel->Initialize(token_, descriptor.token, 1024, 1024, 16384);
      if (!status.ok()) return status;
      const auto& key = channel->jetty_info().key;
      if (key.size != 32) return Error("unsupported NPU Jetty ABI");
      descriptor.key_bytes = key.size;
      std::memcpy(descriptor.key[index], key.value, key.size);
    }
    std::fprintf(stderr, "[KVAF_NPU] CPU_PULL_REFERENCE commit=8bc5a2a protocol=%u jettys=%u "
        "physical=%u npu_eid=%s cpu_eid=%s hbm_flags=0x%x psn=%u\n",
        p::kVersion, p::kChannelCount, physical, EidHex(descriptor.eid).c_str(), EidHex(cpu.eid).c_str(),
        mr.in.ub.flags, descriptor.psn);
    return Status::OK();
  }
  Status Bind(const p::Endpoint& cpu) {
    if (cpu.token != descriptor.token) return Error("CPU Jetty token differs from allocated region token");
    if (cpu.key_bytes != 28 || !cpu.tp || !cpu.psn) return Error("invalid CPU initiator");
    for (std::size_t index = 0; index < channels_.size(); ++index) {
      h::QpImportInfoT info{};
      if (!h::ImportPeerJetty(api_, context_, cpu.key[index], cpu.key_bytes,
          cpu.token, descriptor.tp, cpu.tp, descriptor.psn, cpu.psn, "cpu_pull",
          &info, channels_[index]->peer_jetty_output())) return Error("import CPU initiator");
    }
    return Status::OK();
  }
  Status Ready(void* stream) {
    if (!stream) return Status::InvalidArgument("Offload requires a producer ACL stream");
    aclrtEvent event{};
    if (aclrtCreateEvent(&event) != ACL_SUCCESS) return Error("create producer event");
    const auto record = aclrtRecordEvent(event, static_cast<aclrtStream>(stream));
    const auto wait = record == ACL_SUCCESS ? aclrtSynchronizeEvent(event) : record;
    const auto destroy = aclrtDestroyEvent(event);
    return wait == ACL_SUCCESS && destroy == ACL_SUCCESS ? Status::OK() : Error("wait producer event");
  }
  Status Close() {
    if (async_) { auto s = PollTp(); if (s.code() == StatusCode::kBusy) return Status::InDoubt("source CTP pending"); }
#define RELEASE(handle, operation) \
    if (handle) { if (api_.operation(context_, handle)) return Status::InDoubt(#operation); handle = nullptr; }
    for (auto& channel : channels_) if (channel) {
      const auto status = channel->Shutdown();
      if (!status.ok()) return Status::InDoubt(status.message());
      channel.reset();
    }
    RELEASE(mr_, lmem_unregister)
    RELEASE(token_, token_free)
#undef RELEASE
    if (context_) { if (api_.ctx_deinit(context_)) return Status::InDoubt("destroy source context"); context_ = nullptr; }
    if (session_) { h::ReleaseDirectHccpSession(session_); session_ = nullptr; }
    h::CloseDirectHccp(&api_);
    return Status::OK();
  }
 private:
  Status PollTp() {
    int result{}; const int rc = api_.get_async_req_result(async_, &result);
    if (rc == 128301) return Status::Busy("CTP pending");
    async_ = nullptr;
    if (rc || result || !tp_count_ || !tp_info_.tp_handle) return Error("source CTP query failed");
    descriptor.tp = tp_info_.tp_handle; return Status::OK();
  }
  h::DirectHccpApi api_{};
  h::DirectHccpSession* session_{};
  void *context_{}, *token_{}, *mr_{}, *async_{};
  std::array<std::unique_ptr<h::TargetChannel>, p::kChannelCount> channels_;
  h::GetTpCfg tp_cfg_{};
  h::TpInfo tp_info_{};
  std::uint32_t tp_count_{1};
};

class PullOperator final : public AccessOperator, public DdrMemoryAllocator {
 public:
  explicit PullOperator(OffloadRuntimeOptions options) : options_(std::move(options)), region_id_(Nonce()) {}
  ~PullOperator() override {
    if (source_) (void)source_.release(); // Owner has already attempted explicit recovery/Shutdown.
  }
  std::string address;
  std::uint64_t task() const { return task_; }
  bool pending() const { return source_ != nullptr; }
  void StartupProbe(bool enabled) { startup_probe_ = enabled; }
  Status Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) override {
    auto r = p::MakeRequest(p::Command::Allocate);
    r.region = region_id_; r.bytes = request.ddr_bytes; r.slot_size = request.slot_size;
    r.candidate = options_.cpu_candidate;
    p::Response response{};
    auto status = p::Exchange(address, r, {}, &response);
    if (!status.ok()) return status;
    if (response.result != p::Result::Ok || !response.epoch || !response.base || response.bytes != request.ddr_bytes)
      return Error("allocate CPU Offload region");
    epoch_ = response.epoch; cpu_ = response.endpoint;
    region_ = {response.base, response.bytes, request.slot_size, DeviceKind::kCpu};
    *region = region_; return Status::OK();
  }
  Status Release(const RemoteRegionDesc&) override {
    if (pending()) return Status::InDoubt("source lease still active");
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
      if (response.state == p::State::Failed || response.state == p::State::Open) return Cleanup();
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
  std::unique_ptr<Exporter> source_;
  bool opened_{}, startup_probe_{};
  Status outcome_;
  std::vector<p::Span> spans_;
};
}

struct OffloadRuntime::Impl {
  OffloadRuntimeOptions options;
  std::unique_ptr<MetaManager> meta{CreateInMemoryMetaManager()};
  PullOperator transport;
  RuntimeCore core;
  void* probe{};
  bool ready{};
  explicit Impl(OffloadRuntimeOptions o) : options(o), transport(std::move(o)), core(*meta, transport, transport) {}
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
  if (impl_->core.initialized()) return Status::InvalidState("already initialized");
  if (target.address.empty()) return Status::InvalidArgument("empty CPU address");
  impl_->transport.address = target.address; return Status::OK();
}
Status OffloadRuntime::Initialize(std::uint64_t bytes, std::uint64_t slot) {
  auto status = impl_->Device(); if (!status.ok()) return status;
  std::uint8_t eid[16]{};
  if (!ParseEid(impl_->options.npu_eid, eid) || !impl_->options.timeout_ms ||
      impl_->transport.address.empty() || !bytes || !slot || bytes % slot)
    return Status::InvalidArgument("initialize requires a valid EID, link, timeout and DDR/slot sizes");
  status = impl_->core.Initialize({bytes, slot, DeviceKind::kCpu});
  if (!status.ok()) return status;
  // Match the reference's first READ: a 64-byte control word at HBM + 64.
  // The real payload still uses the caller's Tensor without a staging copy.
  const std::size_t length = std::min<std::uint64_t>(slot, 64);
  constexpr std::size_t allocation_bytes = 4096, source_offset = 64;
  aclrtStream stream{};
  if (aclrtMalloc(&impl_->probe, allocation_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      aclrtMemset(impl_->probe, allocation_bytes, 0xa5, allocation_bytes) != ACL_SUCCESS || aclrtCreateStream(&stream) != ACL_SUCCESS)
    return Error("allocate startup Pull probe; call shutdown, then create a new OffloadRuntime");
  impl_->transport.StartupProbe(true);
  status = impl_->core.Offload({{0}, {impl_->probe, allocation_bytes, DeviceKind::kNpu}, {source_offset}, {length}}, {stream});
  impl_->transport.StartupProbe(false);
  aclrtDestroyStream(stream);
  if (!status.ok()) return status;
  std::uint64_t checksum{};
  status = impl_->transport.Checksum(&checksum);
  std::vector<std::uint8_t> expected(length, 0xa5);
  if (status.ok() && checksum != p::Hash(expected.data(), expected.size())) status = Error("startup CPU DDR checksum mismatch");
  (void)impl_->core.BatchFree({{0}});
  if (aclrtFree(impl_->probe) == ACL_SUCCESS) impl_->probe = nullptr;
  if (status.ok()) impl_->ready = true;
  return status;
}
Status OffloadRuntime::Offload(const BatchPutRequest& r, OffloadExecutionOptions e) {
  if (!impl_->ready) return Status::NotInitialized("OffloadRuntime startup probe has not passed");
  auto status = impl_->Device(); return status.ok() ? impl_->core.Offload(r, e) : status;
}
Status OffloadRuntime::ResolveOffload() {
  auto status = impl_->Device(); return status.ok() ? impl_->core.ResolveOffload() : status;
}
Status OffloadRuntime::Shutdown() {
  if (impl_->core.pending_offload() || impl_->transport.pending()) return Status::InDoubt("resolve pending Offload before shutdown");
  impl_->ready = false;
  if (impl_->core.initialized()) { auto s = impl_->core.Shutdown(); if (!s.ok()) return s; }
  // Also retry a lost Close acknowledgement even if RuntimeCore is already shut down.
  auto s = impl_->transport.Release({}); if (!s.ok()) return s;
  if (impl_->probe) { if (aclrtFree(impl_->probe) != ACL_SUCCESS) return Error("free startup probe"); impl_->probe = nullptr; }
  return Status::OK();
}
Status OffloadRuntime::LastChecksum(std::uint64_t* checksum) { return impl_->transport.Checksum(checksum); }
RuntimeCore& OffloadRuntime::core() { return impl_->core; }
bool OffloadRuntime::initialized() const { return impl_->ready; }
bool OffloadRuntime::pending_offload() const { return impl_->core.pending_offload() || impl_->transport.pending(); }
std::uint64_t OffloadRuntime::task_id() const { return impl_->transport.task(); }
std::uint32_t OffloadRuntime::device_id() const { return impl_->options.device_id; }
}
