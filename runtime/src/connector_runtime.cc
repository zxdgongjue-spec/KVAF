#include "runtime.h"
#include <algorithm>
#include <chrono>
#include <thread>

namespace kvaf {
struct ConnectorPutState {
  std::shared_ptr<const int> owner;
  PutContext context;
  PutAccessDesc allocation;
  RemoteRegionDesc region;
  CpuPullRequest request;
  CpuPullState state{CpuPullState::Prepared};
  bool submitted{}, metadata_finished{}, decision_set{}, commit{};
};
std::uint64_t PutHandle::id() const { return state_ ? state_->context.id : 0; }
const std::vector<PutAllocation>& PutHandle::entries() const {
  static const std::vector<PutAllocation> empty;
  return state_ ? state_->allocation.entries : empty;
}
CpuPullState PutHandle::state() const { return state_ ? state_->state : CpuPullState::Aborted; }
bool PutHandle::submitted() const { return state_ && state_->submitted; }

Status RuntimeCore::AttachCpuSource(const CpuPullSourceDesc& source) {
  if (!initialized_ || closing_) return Status::InvalidState("initialize Runtime before attaching CPU source");
  if (!source.source_id || !source.source_bytes || !source.region_id || !source.region_epoch ||
      source.mr_count() > 4096 || source.bytes_for(1) != source.source_bytes ||
      std::any_of(source.mr_bytes.begin(), source.mr_bytes.end(), [](auto bytes) { return !bytes; }))
    return Status::InvalidArgument("invalid CPU source descriptor");
  const auto existing = cpu_sources_.find(source.source_id);
  if (existing != cpu_sources_.end() && !SameSource(existing->second, source))
    return Status::InvalidArgument("CPU source identity conflicts with an attached source");
  auto status = access_operator_.ValidateCpuSource(source);
  if (status.ok()) cpu_sources_[source.source_id] = source;
  return status;
}
Status RuntimeCore::PreparePut(const std::vector<KeyId>& keys,
                               const std::vector<std::uint64_t>& lengths, PutHandle* out) {
  if (!initialized_ || closing_) return Status::InvalidState("initialize Runtime before PreparePut");
  if (!out || keys.empty() || keys.size() != lengths.size()) return Status::InvalidArgument("invalid PreparePut arrays");
  if (pending_offload()) return Status::Busy("finish the pending Put/Offload before PreparePut");
  auto state = std::make_shared<ConnectorPutState>();
  state->owner = connector_owner_; state->region = remote_region_;
  auto status = meta_manager_.PreparePut(keys, lengths, &state->context, &state->allocation);
  if (!status.ok()) return status;
  connector_put_ = state; out->state_ = state;
  return Status::OK();
}
Status RuntimeCore::SubmitCpuPull(const ReadyToken& ready,
                                  const PutHandle& put, Uint64ArrayView src_offsets, Uint64ArrayView dst_offsets,
                                  Uint64ArrayView lengths) {
  const auto* desc = std::get_if<CpuPullSourceDesc>(&ready.source);
  if (!initialized_ || closing_ || !put.state_ || put.state_->owner != connector_owner_ ||
      !desc || !ready.sequence || !desc->bytes_for(ready.mr_id))
    return Status::InvalidArgument("CPU source, ReadyToken or put belongs to another Runtime/batch");
  const auto source = cpu_sources_.find(desc->source_id);
  if (source == cpu_sources_.end() || !SameSource(source->second, *desc))
    return Status::InvalidArgument("CPU Pull source is not attached to this Runtime or its identity changed");
  auto& p = *put.state_;
  CopyArrays copies;
  auto status = SnapshotCopyArrays(src_offsets, dst_offsets, lengths, &copies);
  if (!status.ok()) return status;
  if (p.submitted) {
    if (!SameSource(p.request.source, *desc) || p.request.sequence != ready.sequence || p.request.mr_id != ready.mr_id ||
        copies.source != p.request.copies.source || copies.destination != p.request.copies.destination ||
        copies.lengths != p.request.copies.lengths)
      return Status::InvalidArgument("duplicate CPU Pull changed its request");
    // Never resubmit, including after a lost response. The original put owns recovery.
    return Status::OK();
  }
  if (connector_put_ != put.state_ || p.state != CpuPullState::Prepared)
    return Status::InvalidState("put is no longer prepared");
  std::vector<std::pair<std::uint64_t, std::uint64_t>> destinations, reservations;
  for (std::size_t i = 0; i < copies.source.size(); ++i) {
    const auto n = copies.lengths[i];
    const auto source_bytes = desc->bytes_for(ready.mr_id);
    if (copies.source[i] > source_bytes || n > source_bytes-copies.source[i] ||
        copies.destination[i] > p.region.length || n > p.region.length-copies.destination[i])
      return Status::InvalidArgument("CPU Pull range outside registered source/DDR");
    destinations.emplace_back(copies.destination[i], n);
  }
  for (const auto& entry : p.allocation.entries) {
    if (entry.address < p.region.base || entry.address-p.region.base > p.region.length ||
        entry.reserved_length > p.region.length-(entry.address-p.region.base))
      return Status::InvalidState("invalid PreparePut allocation");
    reservations.emplace_back(entry.address-p.region.base, entry.reserved_length);
  }
  std::sort(destinations.begin(), destinations.end());
  std::sort(reservations.begin(), reservations.end());
  std::size_t next = 0;
  for (auto reservation : reservations) {
    auto cursor = reservation.first;
    const auto end = cursor + reservation.second;
    while (next < destinations.size() && cursor < end) {
      const auto segment = destinations[next++];
      if (segment.first != cursor || segment.second > end-cursor)
        return Status::InvalidArgument("CPU Pull must exactly cover this put without gaps, overlaps or crossing slots");
      cursor += segment.second;
    }
    if (cursor != end) return Status::InvalidArgument("CPU Pull leaves unwritten Put bytes");
  }
  if (next != destinations.size()) return Status::InvalidArgument("CPU Pull targets bytes outside this put");
  p.request = {*desc, ready.sequence, std::move(copies), ready.mr_id};
  p.submitted = true; p.state = CpuPullState::Submitted;
  // Record the request on the caller's existing put before the first RPC.
  try { status = access_operator_.SubmitCpuPull(p.request); }
  catch (const std::exception& e) { status = Status::InDoubt(std::string("CPU Pull submit exception: ")+e.what()); }
  if (!status.ok()) p.state = status.code() == StatusCode::kInDoubt ? CpuPullState::InDoubt : CpuPullState::Failed;
  return status;
}
Status RuntimeCore::QueryCpuPull(const PutHandle& put, CpuPullState* out) {
  if (!out || !put.state_ || put.state_->owner != connector_owner_)
    return Status::InvalidArgument("CPU Pull put belongs to another Runtime");
  auto& p = *put.state_;
  if (p.state == CpuPullState::Submitted || p.state == CpuPullState::Running || p.state == CpuPullState::InDoubt) {
    if (connector_put_ != put.state_) return Status::InvalidState("CPU Pull put is no longer active");
    CpuPullState state = CpuPullState::InDoubt;
    Status status;
    try { status = access_operator_.QueryCpuPull(&state); }
    catch (const std::exception& e) { status = Status::InDoubt(std::string("CPU Pull query exception: ")+e.what()); }
    if (!status.ok()) { p.state = CpuPullState::InDoubt; *out = p.state; return status; }
    p.state = state;
  }
  *out = p.state; return Status::OK();
}
Status RuntimeCore::WaitCpuPull(const PutHandle& put, std::uint32_t timeout_ms, CpuPullState* out) {
  if (!out) return Status::InvalidArgument("missing CPU Pull result");
  const auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
  for (;;) {
    auto status = QueryCpuPull(put, out); if (!status.ok()) return status;
    if (*out == CpuPullState::Prepared)
      return Status::InvalidState("submit CPU Pull before waiting on this put");
    if (*out == CpuPullState::Done || *out == CpuPullState::Committed) return Status::OK();
    if (*out == CpuPullState::Failed || *out == CpuPullState::Aborted)
      return Status::Internal("CPU Pull failed/cancelled without committing");
    if (*out == CpuPullState::InDoubt || std::chrono::steady_clock::now() >= deadline)
      return Status::InDoubt("CPU Pull unresolved; retain put and source, query the original put");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
Status RuntimeCore::FinishConnectorPut(const PutHandle& put, bool commit) {
  if (!put.state_ || put.state_->owner != connector_owner_) return Status::InvalidArgument("foreign Put handle");
  auto& p = *put.state_;
  if (p.state == CpuPullState::Committed || p.state == CpuPullState::Aborted)
    return (p.state == CpuPullState::Committed) == commit ? Status::OK() : Status::InvalidState("Put already finished with another decision");
  if (connector_put_ != put.state_) return Status::InvalidState("Put is not active");
  if (p.decision_set && p.commit != commit) return Status::InvalidState("retry the original Put decision");
  if (!p.metadata_finished) {
    if (p.submitted) {
      CpuPullState state;
      auto status = QueryCpuPull(put, &state); if (!status.ok()) return status;
      if (state != CpuPullState::Done && state != CpuPullState::Failed)
        return Status::InDoubt("cannot finish Put while DMA may still be active");
    }
    if (commit && (!p.submitted || p.state != CpuPullState::Done))
      return Status::InvalidState("CommitPut requires successful CPU Pull completion");
    p.decision_set = true; p.commit = commit;
    auto status = commit ? meta_manager_.CommitPut(p.context) : meta_manager_.AbortPut(p.context);
    if (!status.ok()) return Status::InDoubt("Put metadata decision pending: "+status.message());
    p.metadata_finished = true;
  }
  if (p.submitted) {
    auto status = access_operator_.CompleteOffload(commit);
    if (!status.ok()) return Status::InDoubt("Put completion acknowledgement pending: "+status.message());
  }
  p.state = commit ? CpuPullState::Committed : CpuPullState::Aborted;
  connector_put_.reset();
  return Status::OK();
}
Status RuntimeCore::CommitPut(const PutHandle& put) { return FinishConnectorPut(put, true); }
Status RuntimeCore::AbortPut(const PutHandle& put) { return FinishConnectorPut(put, false); }

}  // namespace kvaf
