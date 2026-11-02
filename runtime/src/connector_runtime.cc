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
CpuPullTask PutHandle::task() const {
  CpuPullTask result;
  if (state_ && state_->submitted) result.state_ = state_;
  return result;
}
std::uint64_t CpuPullTask::id() const { return state_ ? state_->context.id : 0; }

Status RuntimeCore::AttachCpuSource(const CpuPullSourceDesc& source) {
  if (!initialized_ || closing_) return Status::InvalidState("initialize Runtime before attaching CPU source");
  if (!source.source_id || !source.source_bytes || !source.region_id || !source.region_epoch)
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
                                  const PutHandle& put, Uint64ArrayView src_offsets, Uint64ArrayView dst_addrs,
                                  Uint64ArrayView lengths, CpuPullTask* out) {
  const auto* desc = std::get_if<CpuPullSourceDesc>(&ready.source);
  if (!initialized_ || closing_ || !out || !put.state_ || put.state_->owner != connector_owner_ ||
      !desc || !ready.sequence)
    return Status::InvalidArgument("CPU source, ReadyToken or put belongs to another Runtime/batch");
  const auto source = cpu_sources_.find(desc->source_id);
  if (source == cpu_sources_.end() || !SameSource(source->second, *desc))
    return Status::InvalidArgument("CPU Pull source is not attached to this Runtime or its identity changed");
  auto& p = *put.state_;
  CopyArrays copies;
  auto status = SnapshotCopyArrays(src_offsets, dst_addrs, lengths, &copies);
  if (!status.ok()) return status;
  // Normalize only our owned snapshot, before duplicate comparison. Application
  // arrays stay absolute; service-side spans retain their existing offset ABI.
  for (std::size_t i = 0; i < copies.destination.size(); ++i) {
    auto& address = copies.destination[i];
    if (address < p.region.base || address-p.region.base > p.region.length ||
        copies.lengths[i] > p.region.length-(address-p.region.base))
      return Status::InvalidArgument("CPU Pull destination address outside this Put's DDR region");
    address -= p.region.base;
  }
  if (p.submitted) {
    if (!SameSource(p.request.source, *desc) || p.request.sequence != ready.sequence ||
        copies.source != p.request.copies.source || copies.destination != p.request.copies.destination ||
        copies.lengths != p.request.copies.lengths)
      return Status::InvalidArgument("duplicate CPU Pull changed its request");
    out->state_ = put.state_; // Never resubmit, including after a lost response.
    return Status::OK();
  }
  if (connector_put_ != put.state_ || p.state != CpuPullState::Prepared)
    return Status::InvalidState("put is no longer prepared");
  std::vector<std::pair<std::uint64_t, std::uint64_t>> destinations, reservations;
  for (std::size_t i = 0; i < copies.source.size(); ++i) {
    const auto n = copies.lengths[i];
    if (copies.source[i] > desc->source_bytes || n > desc->source_bytes-copies.source[i] ||
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
  p.request = {*desc, ready.sequence, std::move(copies)};
  p.submitted = true; p.state = CpuPullState::Submitted;
  out->state_ = put.state_; // Recovery handle exists before the first RPC.
  try { status = access_operator_.SubmitCpuPull(p.request); }
  catch (const std::exception& e) { status = Status::InDoubt(std::string("CPU Pull submit exception: ")+e.what()); }
  if (!status.ok()) p.state = status.code() == StatusCode::kInDoubt ? CpuPullState::InDoubt : CpuPullState::Failed;
  return status;
}
Status RuntimeCore::QueryCpuPull(const CpuPullTask& task, CpuPullState* out) {
  if (!out || !task.state_ || task.state_->owner != connector_owner_)
    return Status::InvalidArgument("CPU Pull task belongs to another Runtime");
  auto& p = *task.state_;
  if (p.state == CpuPullState::Submitted || p.state == CpuPullState::Running || p.state == CpuPullState::InDoubt) {
    if (connector_put_ != task.state_) return Status::InvalidState("CPU Pull task is no longer active");
    CpuPullState state = CpuPullState::InDoubt;
    Status status;
    try { status = access_operator_.QueryCpuPull(&state); }
    catch (const std::exception& e) { status = Status::InDoubt(std::string("CPU Pull query exception: ")+e.what()); }
    if (!status.ok()) { p.state = CpuPullState::InDoubt; *out = p.state; return status; }
    p.state = state;
  }
  *out = p.state; return Status::OK();
}
Status RuntimeCore::WaitCpuPull(const CpuPullTask& task, std::uint32_t timeout_ms, CpuPullState* out) {
  if (!out) return Status::InvalidArgument("missing CPU Pull result");
  const auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
  for (;;) {
    auto status = QueryCpuPull(task, out); if (!status.ok()) return status;
    if (*out == CpuPullState::Done || *out == CpuPullState::Committed) return Status::OK();
    if (*out == CpuPullState::Failed || *out == CpuPullState::Aborted)
      return Status::Internal("CPU Pull failed/cancelled without committing");
    if (*out == CpuPullState::InDoubt || std::chrono::steady_clock::now() >= deadline)
      return Status::InDoubt("CPU Pull unresolved; retain put and source, query the original task");
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
      auto status = QueryCpuPull(put.task(), &state); if (!status.ok()) return status;
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
