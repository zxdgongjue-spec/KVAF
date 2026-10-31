#include "remote_offload.h"
#include "remote_pull_service.h"
#include "hbm_exporter.h"

namespace kvaf {
namespace rp = remote_pull;
namespace {
Status Error(const std::string& s) { return Status::Internal("Remote Offload: " + s); }
rp::Request Request(const OffloadRegionTicket& ticket, std::uint64_t id, std::uint64_t sequence, rp::Command command) {
  auto r = rp::MakeRequest(command); r.region = ticket.region.id; r.epoch = ticket.region.epoch;
  r.token = ticket.region.token; r.source_id = id; r.task = sequence; return r;
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
  ConstBufferView input{};
  std::unique_ptr<detail::Exporter> exporter;
  std::uint64_t id{}, sequence{};
  bool attached{}, bound{}, frozen{}, identity_only{};
  std::vector<rp::Span> ready;
  Status Device() const {
    int32_t current = -1;
    return aclrtGetDevice(&current) == ACL_SUCCESS && current == static_cast<int32_t>(device)
        ? Status::OK() : Status::InvalidState("select card B before operating its HBM source");
  }
  Status Call(rp::Command cmd, const std::vector<rp::Span>& spans, rp::Response* out, bool use_sequence=true) {
    auto r = Request(ticket, id, use_sequence ? sequence : 0, cmd);
    if (exporter) r.source = exporter->descriptor;
    if (cmd == rp::Command::Ready && identity_only) r.flags = 1;
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
    bound = false; selected.clear(); input = {}; return Status::OK();
  }
};
CpuPullHbmSource::CpuPullHbmSource(std::uint32_t device, std::string eid) : impl_(new Impl{device, std::move(eid)}) {}
CpuPullHbmSource::~CpuPullHbmSource() { if (!Shutdown().ok()) (void)impl_.release(); }
Status CpuPullHbmSource::Register(const OffloadRegionTicket& ticket, ConstBufferView input, void* stream) {
  auto status = impl_->Device(); if (!status.ok()) return status;
  if (impl_->exporter || impl_->attached || impl_->frozen) return Status::InvalidState("source is already registered or unresolved");
  if (!input.data || !input.bytes || input.device != DeviceKind::kNpu || !stream ||
      input.bytes > UINT64_MAX - reinterpret_cast<std::uintptr_t>(input.data) ||
      !ticket.region.id || !ticket.region.epoch || !ticket.region.token || ticket.address.empty())
    return Status::InvalidArgument("Register requires B HBM, producer stream and a shared DDR ticket");
  impl_->ticket = ticket; impl_->input = input;
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
  offload::Endpoint cpu{}; cpu.base = ticket.region.base; cpu.bytes = ticket.region.bytes;
  cpu.token = ticket.region.token; std::memcpy(cpu.eid, ticket.region.eid, 16);
  for (const auto& candidate : candidates) {
    impl_->input = input; impl_->id = detail::Nonce(); impl_->sequence = 0;
    impl_->exporter = std::make_unique<detail::Exporter>();
    const auto eid = npu_eid::EidToHex(candidate.eid);
    status = impl_->exporter->Open(input, eid, cpu);
    if (status.ok()) {
      impl_->attached = true; // The CPU may attach even when its reply is lost.
      rp::Response response{};
      status = impl_->Call(rp::Command::Attach, {}, &response, false);
      if (status.ok()) status = Result(response, "Attach");
      if (status.ok()) status = impl_->exporter->Bind(response.endpoint);
      if (status.ok()) status = impl_->exporter->Ready(stream);
      if (status.ok()) {
        status = impl_->Call(rp::Command::Bind, {}, &response, false);
        if (status.ok()) status = Result(response, "Bind");
      }
      if (status.ok()) {
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
      std::fprintf(stderr, "[KVAF_NPU] REMOTE_SOURCE_READY region=%llu source=%llu device=%u npu_eid=%s probe=scratch_checksum_pass\n",
          (unsigned long long)ticket.region.id, (unsigned long long)impl_->id, impl_->device, eid.c_str());
      return status;
    }
    if (status.code() == StatusCode::kInDoubt) return status;
    const auto cleanup = impl_->Shutdown();
    if (!cleanup.ok()) return Status::InDoubt("B candidate cleanup unresolved: " + cleanup.message());
    std::fprintf(stderr, "[KVAF_NPU] REMOTE_SOURCE_REJECTED npu_eid=%s detail=%s\n", eid.c_str(), status.message().c_str());
  }
  return status;
}
Status CpuPullHbmSource::Ready(const std::vector<std::uint64_t>& offsets, const std::vector<std::uint64_t>& lengths,
                        void* stream, HbmReadyBatch* batch) {
  auto status = impl_->Device(); if (!status.ok()) return status;
  if (!impl_->bound || !batch) return Status::InvalidState("register B HBM before Ready");
  if (offsets.empty() || offsets.size() != lengths.size() || offsets.size() > rp::kMaxSpans)
    return Status::InvalidArgument("invalid B batch ranges");
  std::vector<rp::Span> spans;
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    if (!offload::Contains(impl_->input.bytes, offsets[i], lengths[i])) return Status::InvalidArgument("B batch outside registered HBM");
    spans.push_back({offsets[i], 0, lengths[i]});
  }
  if (impl_->frozen) {
    if (impl_->identity_only || !rp::SameSpans(spans, impl_->ready)) return Status::Busy("previous B batch remains frozen");
  } else {
    status = impl_->exporter->Ready(stream); if (!status.ok()) return status;
    if (impl_->sequence == UINT64_MAX) return Status::InvalidState("B sequence exhausted");
    ++impl_->sequence; impl_->ready = spans; impl_->frozen = true; impl_->identity_only = false;
  }
  // Retrying a lost Ready ACK reuses the same sequence and never rereads HBM.
  rp::Response response{}; status = impl_->Call(rp::Command::Ready, spans, &response);
  if (!status.ok()) return status;
  status = Result(response, "Ready"); if (!status.ok()) return status;
  batch->source = {impl_->ticket.region.epoch, impl_->ticket.region.id, impl_->id,
                   impl_->sequence, impl_->input.bytes};
  batch->input_offsets = offsets; batch->lengths = lengths;
  return Status::OK();
}
Status CpuPullHbmSource::GetDescriptor(CpuPullSourceDesc* out) const {
  if (!out || !impl_->bound) return Status::InvalidState("register CPU Pull HBM before exporting its descriptor");
  *out = {impl_->id, impl_->input.bytes, impl_->ticket.region.id, impl_->ticket.region.epoch};
  return Status::OK();
}
Status CpuPullHbmSource::Ready(ReadyToken* out) {
  auto status = impl_->Device(); if (!status.ok()) return status;
  CpuPullSourceDesc source;
  if (!out) return Status::InvalidArgument("missing ReadyToken output");
  status = GetDescriptor(&source); if (!status.ok()) return status;
  if (impl_->frozen && !impl_->identity_only) return Status::Busy("legacy batch remains frozen");
  if (!impl_->frozen) {
    if (impl_->sequence == UINT64_MAX) return Status::InvalidState("source sequence exhausted");
    ++impl_->sequence; impl_->ready.clear(); impl_->identity_only = true; impl_->frozen = true;
  }
  *out = {source, impl_->sequence}; // Identity is recoverable even after a lost Ready ACK.
  rp::Response response{};
  status = impl_->Call(rp::Command::Ready, {}, &response);
  return status.ok() ? Result(response, "Ready") : status;
}
Status CpuPullHbmSource::ReleaseBatch(const ReadyToken& token) {
  CpuPullSourceDesc own; auto status = GetDescriptor(&own); if (!status.ok()) return status;
  const auto* source = std::get_if<CpuPullSourceDesc>(&token.source);
  if (!source || !SameSource(own, *source) || !token.sequence || token.sequence != impl_->sequence)
    return Status::InvalidArgument("stale CPU Pull batch release");
  return ReleaseBatch();
}
Status CpuPullHbmSource::CancelBatch(const ReadyToken& token) {
  CpuPullSourceDesc own; auto status = GetDescriptor(&own); if (!status.ok()) return status;
  const auto* source = std::get_if<CpuPullSourceDesc>(&token.source);
  if (!source || !SameSource(own, *source) || !token.sequence || token.sequence != impl_->sequence)
    return Status::InvalidArgument("stale CPU Pull batch cancellation");
  return CancelBatch();
}
Status CpuPullHbmSource::ReleaseBatch() {
  if (!impl_->frozen) return Status::OK();
  rp::Response response{}; auto status = impl_->Call(rp::Command::Query, {}, &response);
  if (!status.ok()) return status;
  status = Result(response, "Query"); if (!status.ok()) return status;
  if (response.state != rp::State::Committed && response.state != rp::State::Cancelled)
    return Status::Busy("A has not acknowledged this B batch; keep HBM unchanged");
  impl_->frozen = false; return Status::OK();
}
Status CpuPullHbmSource::CancelBatch() {
  if (!impl_->frozen) return Status::OK();
  rp::Response response{}; auto status = impl_->Call(rp::Command::Cancel, {}, &response);
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
