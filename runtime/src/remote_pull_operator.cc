#include "remote_pull_operator.h"
#include "remote_pull_service.h"

namespace kvaf {
namespace rp = remote_pull;
namespace {
rp::Request Request(const OffloadRegionTicket& ticket, const CpuPullRequest& current, rp::Command cmd) {
  auto r = rp::MakeRequest(cmd); r.epoch = ticket.epoch; r.region = ticket.region_id; r.token = ticket.token;
  r.source_id = current.source.source_id; r.task = current.sequence; r.mr_id = current.mr_id; return r;
}

}
RemotePullOperator::RemotePullOperator(AccessOperator* aiv, std::shared_ptr<peer::PeerLink> link)
    : aiv_(aiv), link_(std::move(link)) {}
Status RemotePullOperator::Initialize(const RemoteRegionDesc& region) {
  peer::PeerRegion peer{};
  auto status = link_->LookupRegion(region.base, &peer); if (!status.ok()) return status;
  auto r = rp::MakeRequest(rp::Command::Describe); r.region = peer.session_id; r.token = peer.token_value;
  rp::Response response{}; status = channel_.Exchange(link_->address(), r, {}, &response);
  if (!status.ok()) return status;
  if (response.result != rp::Result::Ok || !response.pool.epoch || response.pool.id != peer.session_id ||
      response.pool.base != region.base || response.pool.bytes != region.length || response.pool.slot_size != region.slot_size)
    return Status::Internal("CPU shared DDR descriptor mismatch; update both peers");
  region_ = response.pool;
  ticket_ = {link_->address(), region_.id, region_.epoch, region_.token, {}};
  std::copy(std::begin(region_.eid), std::end(region_.eid), ticket_.eid.begin());
  return aiv_ ? aiv_->Initialize(region) : Status::OK();
}
Status RemotePullOperator::CanShutdown() const {
  if (active_) return Status::InDoubt("resolve remote source Offload before shutdown");
  if (ticket_.region_id) {
    auto r = rp::MakeRequest(rp::Command::Describe); r.region = ticket_.region_id;
    r.token = ticket_.token;
    rp::Response response{}; auto status = channel_.Exchange(ticket_.address, r, {}, &response);
    if (!status.ok()) return status;
    if (response.result != rp::Result::Ok || response.pool.epoch != ticket_.epoch)
      return Status::InDoubt("shared DDR identity unavailable during shutdown");
    if (response.reserved) return Status::Busy("detach B HBM sources before A shutdown");
  }
  return Status::OK();
}
Status RemotePullOperator::Shutdown() {
  auto status = CanShutdown(); if (!status.ok()) return status;
  status = aiv_ ? aiv_->Shutdown() : Status::OK();
  if (status.ok()) channel_.Close();
  return status;
}
Status RemotePullOperator::ExecuteBatchGet(const BatchGetExecRequest& r) {
  if (active_) return Status::Busy("remote source Offload pending");
  return aiv_ ? aiv_->ExecuteBatchGet(r) : Status::NotImplemented("BatchGet is not enabled");
}
Status RemotePullOperator::ExecuteBatchPut(const BatchPutExecRequest& r) {
  if (active_) return Status::Busy("remote source Offload pending");
  return aiv_ ? aiv_->ExecuteBatchPut(r) : Status::NotImplemented("AIV BatchPut is not enabled");
}
Status RemotePullOperator::CompleteOffload(bool committed) {
  if (!active_) return Status::OK();
  auto r = Request(ticket_, current_, rp::Command::Complete); r.flags = committed ? 1 : 0;
  rp::Response response{}; auto status = channel_.Exchange(ticket_.address, r, {}, &response);
  if (!status.ok()) return status;
  if (response.result != rp::Result::Ok) return Status::InDoubt("CPU did not acknowledge A's metadata decision");
  committed_ = committed;
  active_ = false; return Status::OK();
}
Status RemotePullOperator::LastChecksum(std::uint64_t* checksum) const {
  if (!checksum || active_ || !current_.sequence) return Status::InvalidState("no completed remote Offload checksum");
  if (!committed_) return Status::InvalidState("last CPU Pull was not committed");
  rp::Response response{};
  auto status = channel_.Exchange(ticket_.address, Request(ticket_, current_, rp::Command::Checksum), {}, &response);
  if (!status.ok()) return status;
  if (response.result != rp::Result::Ok || response.state != rp::State::Committed)
    return Status::InvalidState("remote Offload checksum unavailable; verify before next Ready or source detach, with DDR slots retained");
  *checksum = response.checksum; return Status::OK();
}
Status RemotePullOperator::ValidateCpuSource(const CpuPullSourceDesc& source) {
  if (!source.source_id || !source.mr_length || source.region_id != ticket_.region_id ||
      source.region_epoch != ticket_.epoch)
    return Status::InvalidArgument("CPU source is not bound to this DDR pool");
  return Status::OK();
}
Status RemotePullOperator::BuildSpans(const CpuPullRequest& request, std::vector<rp::Span>* output, bool prepared) const {
  if (active_) return Status::Busy("previous CPU Pull is unresolved");
  const auto& source = request.source;
  if (source.region_id != ticket_.region_id || source.region_epoch != ticket_.epoch ||
      !request.sequence || !source.source_id || request.copies.source.size() != request.copies.lengths.size() ||
      request.copies.destination.size() != request.copies.lengths.size())
    return Status::InvalidArgument("invalid CPU Pull request");
  auto& spans = *output;
  spans.reserve(request.copies.lengths.size());
  for (std::size_t i = 0; i < request.copies.lengths.size(); ++i)
    spans.push_back({request.copies.source[i], request.copies.destination[i], request.copies.lengths[i]});
  if (!prepared && !offload::ValidateSpans(spans, source.mr_length, region_.bytes))
    return Status::InvalidArgument("invalid CPU Pull source/destination ranges");
  return Status::OK();
}
Status RemotePullOperator::SubmitCpuPull(const CpuPullRequest& request) {
  std::vector<rp::Span> spans;
  auto status = BuildSpans(request, &spans);
  if (!status.ok()) return status;
  const auto& source = request.source;
  rp::Response response{};
  status = channel_.Exchange(ticket_.address, Request(ticket_, request, rp::Command::Query), {}, &response);
  if (!status.ok()) return Status::Internal("CPU Pull pre-submit query failed; no READ submitted: "+status.message());
  if (response.result != rp::Result::Ok || response.state != rp::State::Ready || response.memory.bytes != source.mr_length)
    return Status::InvalidState("CPU Pull batch is not Ready");
  current_ = request; active_ = true; committed_ = false;
  status = channel_.Exchange(ticket_.address, Request(ticket_, current_, rp::Command::Submit), spans, &response);
  if (!status.ok()) return status;
  // Even a rejected Submit is resolved by Query before slots are released.
  if (response.result != rp::Result::Ok) return Status::InDoubt("CPU Pull submit rejected; query the original task before aborting");
  return Status::OK();
}
Status RemotePullOperator::SubmitCpuPullAndWait(const CpuPullRequest& request,
                                                std::uint32_t timeout_ms, CpuPullState* state) {
  return SubmitCpuPullAndWait(request, timeout_ms, state, nullptr);
}
Status RemotePullOperator::SubmitCpuPullAndWait(const CpuPullRequest& request, std::uint32_t timeout_ms,
                                                CpuPullState* state, CpuPullSubmitTiming* profile) {
  return SubmitCpuPullAndWaitImpl(request, timeout_ms, state, profile, false);
}
Status RemotePullOperator::SubmitPreparedCpuPullAndWait(const CpuPullRequest& request, std::uint32_t timeout_ms,
                                                       CpuPullState* state, CpuPullSubmitTiming* profile) {
  return SubmitCpuPullAndWaitImpl(request, timeout_ms, state, profile, true);
}
Status RemotePullOperator::SubmitCpuPullAndWaitImpl(const CpuPullRequest& request, std::uint32_t timeout_ms,
                                                   CpuPullState* state, CpuPullSubmitTiming* profile, bool prepared) {
  if (!state) return Status::InvalidArgument("missing CPU Pull result");
  std::vector<rp::Span> spans;
  Status status;
  { CpuPullStageTimer timer(profile ? &profile->span_prepare_us : nullptr);
    status = BuildSpans(request, &spans, prepared); }
  if (!status.ok()) return status;
  auto r = Request(ticket_, request, rp::Command::SubmitAndWait);
  r.timeout_ms = timeout_ms;
  if (profile) r.flags |= rp::kSubmitProfile;
  r.source.bytes = request.source.mr_length;
  { CpuPullStageTimer timer(profile ? &profile->request_save_us : nullptr);
    current_ = request; active_ = true; committed_ = false; }
  rp::Response response{};
  status = channel_.Exchange(ticket_.address, r, spans, &response, profile);
  if (!status.ok()) return status; // Never replay a possibly accepted READ.
  CpuPullStageTimer response_timer(profile ? &profile->response_check_us : nullptr);
  if (profile) {
    profile->server = rp::ServerTiming(response);
    profile->server_available = profile->server.magic == rp::kTimingMagic;
  }
  if (response.reserved > 1) return Status::InDoubt("invalid CPU Pull acceptance response");
  if (!response.reserved) {
    if (response.result == rp::Result::Ok) return Status::InDoubt("CPU omitted Pull acceptance state");
    // The CPU definitively rejected this submission before starting a READ.
    active_ = false;
    return Status::InvalidState("CPU rejected synchronous Pull before READ acceptance");
  }
  if (response.result != rp::Result::Ok &&
      !(response.result == rp::Result::Failed && response.state == rp::State::Failed))
    return Status::InDoubt("CPU Pull request conflicts with an existing task; retain slots");
  switch (response.state) {
    case rp::State::Done: *state = CpuPullState::Done; return Status::OK();
    case rp::State::Failed:
      *state = CpuPullState::Failed;
      return Status::Internal("CPU Pull READ failed");
    default:
      *state = CpuPullState::InDoubt;
      return Status::InDoubt("CPU Pull unresolved; retain source and query the original task");
  }
}
Status RemotePullOperator::QueryCpuPull(CpuPullState* state) {
  if (!active_ || !state) return Status::InvalidState("no active CPU Pull task");
  rp::Response response{};
  auto status = channel_.Exchange(ticket_.address, Request(ticket_, current_, rp::Command::Query), {}, &response);
  if (!status.ok()) return status;
  if (response.result != rp::Result::Ok) return Status::InDoubt("cannot identify CPU Pull task");
  switch (response.state) {
    case rp::State::Done: *state = CpuPullState::Done; break;
    case rp::State::Running: *state = CpuPullState::Running; break;
    case rp::State::Ready: case rp::State::Failed: case rp::State::Cancelled: *state = CpuPullState::Failed; break;
    default: *state = CpuPullState::InDoubt; break;
  }
  return Status::OK();
}
}
