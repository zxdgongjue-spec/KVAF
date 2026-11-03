#include "remote_offload.h"
#include "remote_pull_service.h"
#include "hbm_exporter.h"

namespace kvaf {
namespace rp = remote_pull;
namespace {
Status Error(const std::string& s) { return Status::Internal("Remote Offload: " + s); }
rp::Request Request(const OffloadRegionTicket& ticket, std::uint64_t id, std::uint64_t sequence, rp::Command command) {
  auto r = rp::MakeRequest(command); r.region = ticket.region_id; r.epoch = ticket.epoch;
  r.token = ticket.token; r.source_id = id; r.task = sequence; return r;
}
Status Result(const rp::Response& response, const char* operation) {
  if (response.result == rp::Result::Ok) return Status::OK();
  if (response.result == rp::Result::InDoubt || response.state == rp::State::InDoubt || response.state == rp::State::Running)
    return Status::InDoubt(std::string(operation) + ": CPU source/task unresolved; retain B HBM");
  return Error(std::string(operation) + " rejected; result=" + std::to_string(unsigned(response.result)) +
               " state=" + std::to_string(unsigned(response.state)));
}
}
struct CpuPullHbmSource::Impl {
  std::uint32_t device;
  std::string eid, selected;
  OffloadRegionTicket ticket;
  std::vector<ConstBufferView> inputs;
  std::vector<offload::MemoryDesc> memories;
  std::unique_ptr<detail::Exporter> exporter;
  std::uint64_t id{}, sequence{}, mr_id{1};
  bool attached{}, bound{}, frozen{};
  Status Device() const {
    int32_t current = -1;
    return aclrtGetDevice(&current) == ACL_SUCCESS && current == static_cast<int32_t>(device)
        ? Status::OK() : Status::InvalidState("select card B before operating its HBM source");
  }
  Status Call(rp::Command cmd, const std::vector<rp::Span>& spans, rp::Response* out, bool use_sequence=true) {
    auto r = Request(ticket, id, use_sequence ? sequence : 0, cmd);
    r.mr_id = mr_id;
    r.memory_count = static_cast<std::uint32_t>(inputs.size());
    if (cmd == rp::Command::Attach) r.connection = exporter->connection;
    if ((cmd == rp::Command::Attach || cmd == rp::Command::AddMemory) && mr_id && mr_id <= memories.size())
      r.source = memories[mr_id-1];
    return rp::Exchange(ticket.address, r, spans, out);
  }
  Status Shutdown() {
    if (frozen) return Status::InDoubt("B batch remains frozen; query/release or cancel it before shutdown");
    if (attached) {
      rp::Response response{}; auto status = Call(rp::Command::Detach, {}, &response, false);
      if (!status.ok()) return status;
      status = Result(response, "Detach");
      if (!status.ok() || response.state != rp::State::Closed)
        return Status::InDoubt("CPU detach not confirmed; retain B HBM registration");
      attached = false;
    }
    if (exporter) {
      auto status = Device(); if (!status.ok()) return status;
      status = exporter->Close(); if (!status.ok()) return status;
      exporter.reset();
    }
    bound = false; selected.clear(); inputs.clear(); memories.clear(); mr_id = 1; return Status::OK();
  }
};
CpuPullHbmSource::CpuPullHbmSource(std::uint32_t device, std::string eid) : impl_(new Impl{device, std::move(eid)}) {}
CpuPullHbmSource::~CpuPullHbmSource() { if (!Shutdown().ok()) (void)impl_.release(); }
Status CpuPullHbmSource::Register(const OffloadRegionTicket& ticket, ConstBufferView input, void* stream) {
  return Register(ticket, std::vector<ConstBufferView>{input}, stream);
}
Status CpuPullHbmSource::Register(const OffloadRegionTicket& ticket, const std::vector<ConstBufferView>& inputs, void* stream) {
  auto status = impl_->Device(); if (!status.ok()) return status;
  if (impl_->exporter || impl_->attached || impl_->frozen) return Status::InvalidState("source is already registered or unresolved");
  if (inputs.empty() || inputs.size() > rp::kMaxSourceMrs || !stream ||
      !ticket.region_id || !ticket.epoch || !ticket.token || ticket.address.empty())
    return Status::InvalidArgument("Register requires B HBM, producer stream and a shared DDR ticket");
  std::vector<std::pair<std::uintptr_t, std::size_t>> ranges;
  for (const auto& input : inputs) {
    const auto base = reinterpret_cast<std::uintptr_t>(input.data);
    if (!base || !input.bytes || input.device != DeviceKind::kNpu || input.bytes > UINT64_MAX-base)
      return Status::InvalidArgument("invalid CPU Pull tensor range");
    ranges.emplace_back(base, input.bytes);
  }
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t i = 1; i < ranges.size(); ++i)
    if (ranges[i].first < ranges[i-1].first+ranges[i-1].second)
      return Status::InvalidArgument("CPU Pull tensors must have distinct, non-overlapping ranges");
  impl_->ticket = ticket;
  std::vector<npu_eid::NpuLocalEidCandidate> candidates;
  if (impl_->eid.empty()) {
    std::uint32_t physical{}; std::string error;
    if (!direct_hccp::GetCurrentNpuPhysicalDeviceId(&physical)) return Error("resolve card B physical device");
    if (!npu_eid::DiscoverPhysicalNpuEids(physical, &candidates, &error)) return Error(error);
  } else {
    npu_eid::NpuLocalEidCandidate candidate{};
    if (!detail::ParseEid(impl_->eid, candidate.eid.data())) return Status::InvalidArgument("invalid B NPU EID");
    candidate.source = "explicit"; candidates.push_back(candidate);
  }
  offload::ConnectionDesc cpu{};
  cpu.token = ticket.token; std::memcpy(cpu.eid, ticket.eid.data(), 16);
  for (const auto& candidate : candidates) {
    impl_->inputs = inputs; impl_->id = detail::Nonce(); impl_->sequence = 0; impl_->mr_id = 1;
    impl_->exporter = std::make_unique<detail::Exporter>();
    const auto eid = npu_eid::EidToHex(candidate.eid);
    status = impl_->exporter->Open(inputs.front(), eid, cpu);
    if (status.ok()) {
      impl_->memories.push_back(impl_->exporter->descriptor);
      for (std::size_t i = 1; status.ok() && i < inputs.size(); ++i) {
        offload::MemoryDesc memory{};
        status = impl_->exporter->AddMemory(inputs[i], &memory);
        if (status.ok()) impl_->memories.push_back(memory);
      }
    }
    if (status.ok()) {
      impl_->attached = true; // The CPU may attach even when its reply is lost.
      rp::Response response{};
      status = impl_->Call(rp::Command::Attach, {}, &response, false);
      if (status.ok()) status = Result(response, "Attach");
      if (status.ok()) status = impl_->exporter->Bind(response.connection);
      for (std::size_t i = 1; status.ok() && i < inputs.size(); ++i) {
        impl_->mr_id = i+1;
        status = impl_->Call(rp::Command::AddMemory, {}, &response, false);
        if (status.ok()) status = Result(response, "AddMemory");
      }
      impl_->mr_id = 1;
      if (status.ok()) status = impl_->exporter->Ready(stream);
      if (status.ok()) {
        status = impl_->Call(rp::Command::Bind, {}, &response, false);
        if (status.ok()) status = Result(response, "Bind");
      }
      for (std::size_t i = 0; status.ok() && i < inputs.size(); ++i) {
        impl_->mr_id = i+1;
        const auto& input = inputs[i];
        const std::uint64_t offset = input.bytes >= 128 ? 64 : 0;
        const auto length = std::min<std::uint64_t>(64, input.bytes-offset);
        std::array<std::uint8_t, 64> expected{};
        if (aclrtMemcpy(expected.data(), expected.size(), static_cast<const std::uint8_t*>(input.data)+offset,
                        length, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) status = Error("copy B probe expectation");
        if (status.ok()) {
          status = impl_->Call(rp::Command::Probe, {{offset, 0, length}}, &response, false);
          if (status.ok()) status = Result(response, "Probe");
          if (status.ok() && response.checksum != offload::Hash(expected.data(), length)) status = Error("B probe checksum mismatch");
        }
      }
    }
    if (status.ok()) {
      impl_->bound = true; impl_->selected = eid;
      impl_->mr_id = 1;
      std::fprintf(stderr, "[KVAF_NPU] REMOTE_SOURCE_READY region=%llu source=%llu device=%u npu_eid=%s mrs=%zu jettys=4 probe=all_mrs_scratch_checksum_pass\n",
          (unsigned long long)ticket.region_id, (unsigned long long)impl_->id, impl_->device, eid.c_str(), inputs.size());
      return status;
    }
    if (status.code() == StatusCode::kInDoubt) return status;
    const auto cleanup = impl_->Shutdown();
    if (!cleanup.ok()) return Status::InDoubt("B candidate cleanup unresolved: " + cleanup.message());
    std::fprintf(stderr, "[KVAF_NPU] REMOTE_SOURCE_REJECTED npu_eid=%s detail=%s\n", eid.c_str(), status.message().c_str());
  }
  return status;
}
Status CpuPullHbmSource::GetDescriptor(CpuPullSourceDesc* out) const {
  if (!out || !impl_->bound) return Status::InvalidState("register CPU Pull HBM before exporting its descriptor");
  *out = {impl_->id, impl_->ticket.region_id, impl_->ticket.epoch, impl_->inputs[impl_->mr_id-1].bytes};
  return Status::OK();
}
Status CpuPullHbmSource::Ready(ReadyToken* out) {
  if (impl_->inputs.size() != 1) return Status::InvalidArgument("ready(tensor) must select one registered tensor");
  return Ready(impl_->inputs.front(), out);
}
Status CpuPullHbmSource::Ready(ConstBufferView input, ReadyToken* out) {
  auto status = impl_->Device(); if (!status.ok()) return status;
  CpuPullSourceDesc source;
  if (!out) return Status::InvalidArgument("missing ReadyToken output");
  if (!impl_->bound) return Status::InvalidState("register CPU Pull HBM before Ready");
  const auto it = std::find_if(impl_->inputs.begin(), impl_->inputs.end(), [&](ConstBufferView registered) {
    return registered.data == input.data && registered.bytes == input.bytes && registered.device == input.device;
  });
  if (it == impl_->inputs.end()) return Status::InvalidArgument("ready tensor is not registered in this source");
  const auto mr_id = static_cast<std::uint64_t>(it-impl_->inputs.begin()+1);
  if (impl_->frozen && impl_->mr_id != mr_id)
    return Status::Busy("previous batch is frozen; cannot select a different tensor");
  if (!impl_->frozen) {
    if (impl_->sequence == UINT64_MAX) return Status::InvalidState("source sequence exhausted");
    ++impl_->sequence; impl_->frozen = true; impl_->mr_id = mr_id;
  }
  status = GetDescriptor(&source); if (!status.ok()) return status;
  *out = {source, impl_->sequence, impl_->mr_id}; // Recoverable after a lost Ready ACK.
  rp::Response response{};
  status = impl_->Call(rp::Command::Ready, {}, &response);
  return status.ok() ? Result(response, "Ready") : status;
}
Status CpuPullHbmSource::ReleaseBatch(const ReadyToken& token) {
  CpuPullSourceDesc own; auto status = GetDescriptor(&own); if (!status.ok()) return status;
  const auto* source = std::get_if<CpuPullSourceDesc>(&token.source);
  if (!source || !SameSource(own, *source) || !token.sequence || token.sequence != impl_->sequence || token.mr_id != impl_->mr_id)
    return Status::InvalidArgument("stale CPU Pull batch release");
  if (!impl_->frozen) return Status::OK();
  rp::Response response{}; status = impl_->Call(rp::Command::Query, {}, &response);
  if (!status.ok()) return status;
  status = Result(response, "Query"); if (!status.ok()) return status;
  if (response.state != rp::State::Committed && response.state != rp::State::Cancelled)
    return Status::Busy("reader has not acknowledged this batch; keep HBM unchanged");
  impl_->frozen = false; return Status::OK();
}
Status CpuPullHbmSource::CancelBatch(const ReadyToken& token) {
  CpuPullSourceDesc own; auto status = GetDescriptor(&own); if (!status.ok()) return status;
  const auto* source = std::get_if<CpuPullSourceDesc>(&token.source);
  if (!source || !SameSource(own, *source) || !token.sequence || token.sequence != impl_->sequence || token.mr_id != impl_->mr_id)
    return Status::InvalidArgument("stale CPU Pull batch cancellation");
  if (!impl_->frozen) return Status::OK();
  rp::Response response{}; status = impl_->Call(rp::Command::Cancel, {}, &response);
  if (!status.ok()) return status;
  status = Result(response, "Cancel"); if (!status.ok()) return status;
  impl_->frozen = false; return Status::OK();
}
Status CpuPullHbmSource::Shutdown() { return impl_->Shutdown(); }
bool CpuPullHbmSource::frozen() const { return impl_->frozen; }
bool CpuPullHbmSource::registered() const { return impl_->bound; }
std::uint32_t CpuPullHbmSource::device_id() const { return impl_->device; }
std::string CpuPullHbmSource::selected_npu_eid() const { return impl_->selected; }
}
