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
  if (pending()) return Status::InDoubt("resolve remote source Offload before shutdown");
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
  if (pending()) return Status::Busy("remote source Offload pending");
  return aiv_ ? aiv_->ExecuteBatchGet(r) : Status::NotImplemented("BatchGet is not enabled");
}
Status RemotePullOperator::ExecuteBatchPut(const BatchPutExecRequest& r) {
  if (pending()) return Status::Busy("remote source Offload pending");
  return aiv_ ? aiv_->ExecuteBatchPut(r) : Status::NotImplemented("AIV BatchPut is not enabled");
}
Status RemotePullOperator::CompleteOffload(bool committed) {
  if (!active_) return Status::OK();
  // RuntimeCore has finished the original Put's local metadata decision.
  committed_ = committed;
  finished_ = true;
  active_ = false; return Status::OK();
}
Status RemotePullOperator::LastChecksum(std::uint64_t* checksum) const {
  if (!checksum || pending() || !current_.sequence) return Status::InvalidState("no completed remote Offload checksum");
  if (!committed_) return Status::InvalidState("last CPU Pull was not committed");
  rp::Response response{};
  auto r = Request(ticket_, current_, rp::Command::Checksum);
  r.operation = operation_;
  auto status = channel_.Exchange(ticket_.address, r, {}, &response);
  if (!status.ok()) return status;
  if (response.result != rp::Result::Ok || response.state != rp::State::Done)
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
  if (pending()) return Status::Busy("previous CPU Pull is unresolved");
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
  CpuPullState state;
  return SubmitImpl(request, rp::Command::Submit, 0, &state, nullptr, false);
}
Status RemotePullOperator::SubmitCpuPullAndWait(const CpuPullRequest& request,
                                                std::uint32_t timeout_ms, CpuPullState* state) {
  return SubmitCpuPullAndWait(request, timeout_ms, state, nullptr);
}
Status RemotePullOperator::SubmitCpuPullAndWait(const CpuPullRequest& request, std::uint32_t timeout_ms,
                                                CpuPullState* state, CpuPullSubmitTiming* profile) {
  return SubmitImpl(request, rp::Command::SubmitAndWait, timeout_ms, state, profile, false);
}
Status RemotePullOperator::SubmitPreparedCpuPullAndWait(const CpuPullRequest& request, std::uint32_t timeout_ms,
                                                       CpuPullState* state, CpuPullSubmitTiming* profile) {
  return SubmitImpl(request, rp::Command::SubmitAndWait, timeout_ms, state, profile, true);
}
Status RemotePullOperator::SubmitImpl(const CpuPullRequest& request, rp::Command command, std::uint32_t timeout_ms,
                                      CpuPullState* state, CpuPullSubmitTiming* profile, bool prepared) {
  if (!state) return Status::InvalidArgument("missing CPU Pull result");
  std::vector<rp::Span> spans;
  Status status;
  { CpuPullStageTimer timer(profile ? &profile->span_prepare_us : nullptr);
    status = BuildSpans(request, &spans, prepared); }
  if (!status.ok()) return status;
  if (operation_ == UINT64_MAX) return Status::InvalidState("CPU Pull operation IDs exhausted");
  auto r = Request(ticket_, request, command);
  r.operation = ++operation_;
  r.timeout_ms = timeout_ms;
  if (profile) r.flags |= rp::kSubmitProfile;
  r.source.bytes = request.source.mr_length;
  { CpuPullStageTimer timer(profile ? &profile->request_save_us : nullptr);
    current_ = request; active_ = true; finished_ = committed_ = false; cancelling_ = false; }
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
    return Status::InvalidState("CPU rejected Pull before READ acceptance");
  }
  if (response.result != rp::Result::Ok &&
      !(response.result == rp::Result::Failed && response.state == rp::State::Failed))
    return Status::InDoubt("CPU Pull request conflicts with an existing task; retain slots");
  switch (response.state) {
    case rp::State::Running:
      if (command == rp::Command::Submit) { *state = CpuPullState::Running; return Status::OK(); }
      return Status::InDoubt("CPU Pull still running; retain original Put");
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
  return InspectTask(rp::Command::Query, state);
}
Status RemotePullOperator::InspectTask(rp::Command command, CpuPullState* state) {
  if (!active_ || !state || current_.copies.source.empty()) return Status::InvalidState("no submitted CPU Pull task");
  rp::Response response{};
  auto r = Request(ticket_, current_, command);
  r.operation = operation_;
  auto status = channel_.Exchange(ticket_.address, r, {}, &response);
  if (!status.ok()) return status;
  if (response.result != rp::Result::Ok) return Status::InDoubt("cannot identify original CPU Pull task");
  if (response.state == rp::State::NotSeen && command == rp::Command::Query) {
    // Query may overtake the original submit. Fence it before exposing failure,
    // and retain the abort decision even if Cancel loses the race or its ACK.
    cancelling_ = true;
    return InspectTask(rp::Command::Cancel, state);
  }
  switch (response.state) {
    case rp::State::Done: *state = cancelling_ ? CpuPullState::Failed : CpuPullState::Done; break;
    case rp::State::Running: *state = CpuPullState::Running; break;
    case rp::State::Failed: case rp::State::Cancelled: *state = CpuPullState::Failed; break;
    default: *state = CpuPullState::InDoubt; break;
  }
  return Status::OK();
}
Status RemotePullOperator::CancelCpuPull(const ReadyToken& ready) {
  const auto* source = std::get_if<CpuPullSourceDesc>(&ready.source);
  if (!source || !ready.sequence || !ready.mr_id || ready.mr_id > rp::kMaxSourceMrs)
    return Status::InvalidArgument("invalid CPU Pull cancellation token");
  auto status = ValidateCpuSource(*source);
  if (!status.ok()) return status;
  const auto& tracked = pending_cancel_ ? *pending_cancel_ : current_;
  const bool same = SameSource(tracked.source, *source) && tracked.sequence == ready.sequence &&
                    tracked.mr_id == ready.mr_id;
  if (!pending() && finished_ && same)
    return committed_ ? Status::InvalidState("cannot cancel an already committed Put") : Status::OK();
  if (pending() && !same)
    return Status::Busy("another CPU Pull is pending");
  if (active_) {
    cancelling_ = true;
    CpuPullState state;
    status = InspectTask(rp::Command::Cancel, &state);
    if (!status.ok()) return status;
    return state == CpuPullState::Failed ? Status::OK() : Status::InDoubt("READ unresolved; retain original Put");
  }
  // Track the fence across a lost reply without replacing the last completed batch.
  pending_cancel_ = CpuPullRequest{*source, ready.sequence, {}, ready.mr_id};
  auto r = Request(ticket_, *pending_cancel_, rp::Command::Cancel);
  r.source.bytes = source->mr_length;
  rp::Response response{};
  status = channel_.Exchange(ticket_.address, r, {}, &response);
  if (!status.ok()) return status;
  if (response.result == rp::Result::Ok && response.state == rp::State::Cancelled) {
    current_ = std::move(*pending_cancel_);
    finished_ = true;
    committed_ = false;
  } else {
    status = Status::InvalidState("unused CPU Pull cancellation rejected");
  }
  pending_cancel_.reset();
  return status;
}

}
