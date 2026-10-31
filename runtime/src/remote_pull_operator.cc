#include "remote_pull_operator.h"
#include "remote_pull_service.h"
#include "meta_manager.h"
#include <chrono>
#include <thread>

namespace kvaf {
namespace rp = remote_pull;
namespace {
rp::Request Request(const OffloadRegionTicket& ticket, const RemoteOffloadRequest& current, rp::Command cmd) {
  auto r = rp::MakeRequest(cmd); r.epoch = ticket.region.epoch; r.region = ticket.region.id; r.token = ticket.region.token;
  r.source_id = current.source.source_id; r.task = current.source.sequence; return r;
}
bool Same(const RemoteOffloadRequest& a, const RemoteOffloadRequest& b) {
  return a.source.region_id == b.source.region_id && a.source.region_epoch == b.source.region_epoch &&
      a.source.source_id == b.source.source_id && a.source.sequence == b.source.sequence &&
      a.source.source_bytes == b.source.source_bytes && a.keys == b.keys &&
      a.input_offsets == b.input_offsets && a.lengths == b.lengths;
}
}
RemotePullOperator::RemotePullOperator(AccessOperator* aiv, std::shared_ptr<peer::PeerLink> link, std::uint32_t timeout)
    : aiv_(aiv), link_(std::move(link)), timeout_ms_(timeout) {}
Status RemotePullOperator::Initialize(const RemoteRegionDesc& region) {
  peer::PeerRegion peer{};
  auto status = link_->LookupRegion(region.base, &peer); if (!status.ok()) return status;
  auto r = rp::MakeRequest(rp::Command::Describe); r.region = peer.session_id; r.token = peer.token_value;
  rp::Response response{}; status = rp::Exchange(link_->address(), r, {}, &response);
  if (!status.ok()) return status;
  if (response.result != rp::Result::Ok || !response.pool.epoch || response.pool.id != peer.session_id ||
      response.pool.base != region.base || response.pool.bytes != region.length || response.pool.slot_size != region.slot_size)
    return Status::Internal("CPU shared DDR descriptor mismatch; update both peers");
  ticket_ = {link_->address(), response.pool};
  return aiv_ ? aiv_->Initialize(region) : Status::OK();
}
Status RemotePullOperator::CanShutdown() const {
  if (active_) return Status::InDoubt("resolve remote source Offload before shutdown");
  if (ticket_.region.id) {
    auto r = rp::MakeRequest(rp::Command::Describe); r.region = ticket_.region.id;
    r.token = ticket_.region.token;
    rp::Response response{}; auto status = rp::Exchange(ticket_.address, r, {}, &response);
    if (!status.ok()) return status;
    if (response.result != rp::Result::Ok || response.pool.epoch != ticket_.region.epoch)
      return Status::InDoubt("shared DDR identity unavailable during shutdown");
    if (response.reserved) return Status::Busy("detach B HBM sources before A shutdown");
  }
  return Status::OK();
}
Status RemotePullOperator::Shutdown() {
  auto status = CanShutdown(); if (!status.ok()) return status;
  return aiv_ ? aiv_->Shutdown() : Status::OK();
}
Status RemotePullOperator::ExecuteBatchGet(const BatchGetExecRequest& r) {
  if (active_) return Status::Busy("remote source Offload pending");
  return aiv_ ? aiv_->ExecuteBatchGet(r) : Status::NotImplemented("BatchGet is not enabled");
}
Status RemotePullOperator::ExecuteBatchPut(const BatchPutExecRequest& r) {
  if (active_) return Status::Busy("remote source Offload pending");
  return aiv_ ? aiv_->ExecuteBatchPut(r) : Status::NotImplemented("AIV BatchPut is not enabled");
}
Status RemotePullOperator::Validate(const RemoteOffloadRequest& request, bool* duplicate) {
  *duplicate = false;
  if (active_) return Status::Busy("resolve previous remote Offload");
  if (!request.source.source_id || !request.source.sequence || request.source.region_epoch != ticket_.region.epoch ||
      request.source.region_id != ticket_.region.id || request.keys.size() > rp::kMaxSpans)
    return Status::InvalidArgument("B source does not belong to this shared DDR region");
  const auto old = records_.find(request.source.source_id);
  if (old != records_.end() && request.source.sequence <= old->second.request.source.sequence) {
    if (!Same(request, old->second.request)) return Status::InvalidArgument("stale/conflicting B Ready message");
    current_ = old->second.request;
    *duplicate = true; return old->second.result;
  }
  rp::Response response{};
  auto status = rp::Exchange(ticket_.address, Request(ticket_, request, rp::Command::Query), {}, &response);
  if (!status.ok()) return status;
  if (response.result != rp::Result::Ok || response.state != rp::State::Ready ||
      response.endpoint.bytes != request.source.source_bytes)
    return Status::InvalidState("B batch is not ready for this sequence");
  return Status::OK();
}
Status RemotePullOperator::ExecuteRemoteOffload(const RemoteOffloadExecRequest& exec) {
  if (active_) return Status::Busy("remote source task pending");
  current_ = *exec.request;
  std::vector<rp::Span> spans;
  for (std::size_t i = 0; i < current_.keys.size(); ++i) {
    const auto address = exec.put_desc->entries[i].address;
    if (address < ticket_.region.base) return Status::InvalidArgument("destination below shared DDR");
    spans.push_back({current_.input_offsets[i], address-ticket_.region.base, current_.lengths[i]});
  }
  if (!offload::ValidateSpans(spans, current_.source.source_bytes, ticket_.region.bytes))
    return Status::InvalidArgument("invalid shared DDR spans");
  active_ = true;
  rp::Response response{};
  auto status = rp::Exchange(ticket_.address, Request(ticket_, current_, rp::Command::Submit), spans, &response);
  if (!status.ok()) return status;
  // Query determines whether a rejected/lost submission can still access B.
  return ResolveOffload();
}
Status RemotePullOperator::ResolveOffload() {
  if (!active_) return Status::InvalidState("no remote source task");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);
  for (;;) {
    rp::Response response{};
    auto status = rp::Exchange(ticket_.address, Request(ticket_, current_, rp::Command::Query), {}, &response);
    if (!status.ok()) return status;
    if (response.result != rp::Result::Ok) return Status::InDoubt("cannot identify remote source task");
    if (response.state == rp::State::Done) return Status::OK();
    if (response.state == rp::State::Ready || response.state == rp::State::Failed || response.state == rp::State::Cancelled)
      return Status::Internal("remote READ failed or was not submitted");
    if (response.state != rp::State::Running || std::chrono::steady_clock::now() >= deadline)
      return Status::InDoubt("remote source task unresolved; retain B HBM and shared DDR slots");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
Status RemotePullOperator::CompleteOffload(bool committed) {
  if (!active_) return Status::OK();
  auto r = Request(ticket_, current_, rp::Command::Complete); r.flags = committed ? 1 : 0;
  rp::Response response{}; auto status = rp::Exchange(ticket_.address, r, {}, &response);
  if (!status.ok()) return status;
  if (response.result != rp::Result::Ok) return Status::InDoubt("CPU did not acknowledge A's metadata decision");
  records_[current_.source.source_id] = {current_, committed ? Status::OK() : Status::Internal("B batch was aborted")};
  active_ = false; return Status::OK();
}
Status RemotePullOperator::LastChecksum(std::uint64_t* checksum) const {
  if (!checksum || active_ || !current_.source.sequence) return Status::InvalidState("no completed remote Offload checksum");
  const auto record = records_.find(current_.source.source_id);
  if (record == records_.end() || !record->second.result.ok()) return Status::InvalidState("last Offload was not committed");
  rp::Response response{};
  auto status = rp::Exchange(ticket_.address, Request(ticket_, current_, rp::Command::Checksum), {}, &response);
  if (!status.ok()) return status;
  if (response.result != rp::Result::Ok || response.state != rp::State::Committed)
    return Status::InvalidState("remote Offload checksum unavailable; verify before next Ready or source detach, with DDR slots retained");
  *checksum = response.checksum; return Status::OK();
}
Status RemotePullOperator::ValidateCpuSource(const CpuPullSourceDesc& source) {
  if (!source.source_id || !source.source_bytes || source.region_id != ticket_.region.id ||
      source.region_epoch != ticket_.region.epoch)
    return Status::InvalidArgument("CPU source is not bound to this DDR pool");
  RemoteOffloadRequest current;
  current.source = {source.region_epoch, source.region_id, source.source_id, 0, source.source_bytes};
  rp::Response response{};
  auto status = rp::Exchange(ticket_.address, Request(ticket_, current, rp::Command::InspectSource), {}, &response);
  if (!status.ok()) return status;
  if (response.result != rp::Result::Ok || response.endpoint.bytes != source.source_bytes ||
      response.state == rp::State::Connecting || response.state == rp::State::Failed ||
      response.state == rp::State::InDoubt || response.state == rp::State::Closed)
    return Status::InvalidState("CPU source is missing, stale or unavailable");
  return Status::OK();
}
Status RemotePullOperator::SubmitCpuPull(const CpuPullRequest& request) {
  if (active_) return Status::Busy("previous CPU Pull is unresolved");
  RemoteOffloadRequest current;
  const auto& source = request.source;
  current.source = {source.region_epoch, source.region_id, source.source_id, request.sequence, source.source_bytes};
  current.input_offsets = request.copies.source; current.lengths = request.copies.lengths;
  current.keys = request.keys;
  if (source.region_id != ticket_.region.id || source.region_epoch != ticket_.region.epoch ||
      !request.sequence || !source.source_id || request.copies.source.size() != request.copies.lengths.size() ||
      request.copies.destination.size() != request.copies.lengths.size())
    return Status::InvalidArgument("invalid CPU Pull request");
  std::vector<rp::Span> spans;
  for (std::size_t i = 0; i < request.copies.lengths.size(); ++i)
    spans.push_back({request.copies.source[i], request.copies.destination[i], request.copies.lengths[i]});
  if (!offload::ValidateSpans(spans, source.source_bytes, ticket_.region.bytes))
    return Status::InvalidArgument("invalid CPU Pull source/destination ranges");
  rp::Response response{};
  auto status = rp::Exchange(ticket_.address, Request(ticket_, current, rp::Command::Query), {}, &response);
  if (!status.ok()) return Status::Internal("CPU Pull pre-submit query failed; no READ submitted: "+status.message());
  if (response.result != rp::Result::Ok || response.state != rp::State::Ready || response.endpoint.bytes != source.source_bytes)
    return Status::InvalidState("CPU Pull batch is not Ready");
  current_ = std::move(current); active_ = true;
  status = rp::Exchange(ticket_.address, Request(ticket_, current_, rp::Command::Submit), spans, &response);
  if (!status.ok()) return status;
  // Even a rejected Submit is resolved by Query before slots are released.
  if (response.result != rp::Result::Ok) return Status::InDoubt("CPU Pull submit rejected; query the original task before aborting");
  return Status::OK();
}
Status RemotePullOperator::QueryCpuPull(CpuPullState* state) {
  if (!active_ || !state) return Status::InvalidState("no active CPU Pull task");
  rp::Response response{};
  auto status = rp::Exchange(ticket_.address, Request(ticket_, current_, rp::Command::Query), {}, &response);
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
