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
RemotePullOperator::RemotePullOperator(std::shared_ptr<peer::PeerLink> link, peer::PeerCandidateInfo candidate)
    : candidate_(std::move(candidate)), link_(std::move(link)) {}
Status RemotePullOperator::Initialize(const RemoteRegionDesc& region) {
  if (closed_) return Status::InvalidState("CPU Pull executor is closed");
  if (ticket_.region_id) {
    return region_.base == region.base && region_.bytes == region.length && region_.slot_size == region.slot_size
        ? Status::OK() : Status::InvalidState("CPU Pull destination already bound");
  }
  peer::PeerRegion peer{};
  auto status = link_->PrepareCpuPullRegion(region.base, candidate_.candidate_id, &peer);
  if (!status.ok()) return status;
  if (!peer.region_id || !peer.epoch || peer.length != region.length)
    return Status::CompletionUnknown("CPU Pull destination identity mismatch");
  if (peer.eid != candidate_.eid)
    return Status::InvalidState("prepared CPU Pull EID differs from the advertised candidate");
  region_.id = peer.region_id; region_.epoch = peer.epoch;
  region_.base = region.base; region_.bytes = region.length; region_.slot_size = region.slot_size;
  region_.candidate = candidate_.candidate_id; region_.token = peer.token_value;
  std::copy(peer.eid.begin(), peer.eid.end(), region_.eid);
  ticket_ = {link_->address(), region_.id, region_.epoch, region_.token, peer.eid};
  return Status::OK();
}
Status RemotePullOperator::CanShutdown() const {
  if (closed_) return Status::OK();
  if (pending()) return Status::InDoubt("resolve remote source Offload before shutdown");
  if (ticket_.region_id) {
    auto r = rp::MakeRequest(rp::Command::Describe); r.region = ticket_.region_id;
    r.token = ticket_.token; r.epoch = ticket_.epoch;
    rp::Response response{}; auto status = channel_.Exchange(ticket_.address, r, {}, &response);
    if (!status.ok()) return status;
    if (response.result != rp::Result::Ok || response.pool.epoch != ticket_.epoch)
      return Status::InDoubt("shared DDR identity unavailable during shutdown");
    if (response.reserved) return Status::Busy("detach B HBM sources before A shutdown");
  }
  return Status::OK();
}
Status RemotePullOperator::Shutdown() {
  auto status = CanShutdown();
  if (status.ok()) {
    channel_.Close();
    // A later DDR Release may succeed remotely but lose its ACK. Repeating
    // Shutdown must not Describe a Region that is already gone.
    closed_ = true;
  }
  return status;
}
Status RemotePullOperator::CompleteOffload(bool committed) {
  if (!active_) return Status::OK();
  // RuntimeCore has finished the original Put's local metadata decision.
  committed_ = committed;
  finished_ = true;
  active_ = false; return Status::OK();
}
Status RemotePullOperator::LastChecksum(std::uint64_t* checksum) const {
  if (closed_) return Status::InvalidState("CPU Pull executor is closed");
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
  if (closed_) return Status::InvalidState("CPU Pull executor is closed");
  if (!source.source_id || !source.mr_length || source.region_id != ticket_.region_id ||
      source.region_epoch != ticket_.epoch)
    return Status::InvalidArgument("CPU source is not bound to this DDR pool");
  return Status::OK();
}
Status RemotePullOperator::BuildSpans(const CpuPullRequest& request, std::vector<rp::Span>* output, bool prepared) const {
  if (closed_) return Status::InvalidState("CPU Pull executor is closed");
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
