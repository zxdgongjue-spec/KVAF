#include "runtime.h"
#include <algorithm>
#include <chrono>
#include <thread>

namespace kvaf {
std::uint64_t PutHandle::id() const { return state_ ? state_->transaction.id : 0; }
const std::vector<PutAllocation>& PutHandle::entries() const {
  static const std::vector<PutAllocation> empty;
  return state_ ? state_->allocation.entries : empty;
}
CpuPullState PutHandle::state() const { return state_ ? state_->state.load() : CpuPullState::Aborted; }
bool PutHandle::submitted() const { return state_ && state_->submitted.load(); }

bool PutHandle::aborting() const { return state_ && state_->abort_requested.load(); }

Status RuntimeCore::PreparePut(const std::vector<KeyId>& keys,
                               const std::vector<std::uint64_t>& lengths, PutHandle* out) {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!initialized_ || closing_) return Status::InvalidState("initialize Runtime before PreparePut");
  if (!out || keys.empty() || keys.size() != lengths.size()) return Status::InvalidArgument("invalid PreparePut arrays");
  if (!cpu_pull_) return Status::NotImplemented("prepare CPU Pull resources before reserving slots");
  if (pending_put_ || cpu_pull_->pending())
    return Status::Busy("finish the pending Put/Offload before PreparePut");
  auto state = std::make_shared<ConnectorPutState>();
  state->cpu = true;
  state->owner = connector_owner_; state->region = remote_region_;
  auto status = meta_manager_.PreparePut(keys, lengths, &state->transaction, &state->allocation);
  if (!status.ok()) return status;
  pending_put_ = state; out->state_ = state;
  return Status::OK();
}
Status RuntimeCore::SubmitCpuPull(const ReadyToken& ready,
                                  const PutHandle& put, Uint64ArrayView src_offsets, Uint64ArrayView dst_offsets,
                                  Uint64ArrayView lengths) {
  return SubmitCpuPullImpl(ready, put, src_offsets, dst_offsets, lengths, nullptr, nullptr);
}
Status RuntimeCore::SubmitCpuPullAndWait(const ReadyToken& ready, const PutHandle& put,
                                        Uint64ArrayView src_offsets, Uint64ArrayView dst_offsets,
                                        Uint64ArrayView lengths, std::uint32_t timeout_ms, CpuPullSubmitTiming* profile) {
  if (profile) *profile = {};
  return SubmitCpuPullImpl(ready, put, src_offsets, dst_offsets, lengths, &timeout_ms, profile);
}
Status RuntimeCore::SubmitCpuPullImpl(const ReadyToken& ready, const PutHandle& put,
                                     Uint64ArrayView src_offsets, Uint64ArrayView dst_offsets,
                                     Uint64ArrayView lengths, const std::uint32_t* timeout_ms, CpuPullSubmitTiming* profile) {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  const auto* desc = std::get_if<CpuPullSourceDesc>(&ready.source);
  if (!initialized_ || closing_ || !cpu_pull_ || !put.state_ || put.state_->owner != connector_owner_ ||
      !desc || !ready.sequence || !ready.mr_id || ready.mr_id > 4096 || !desc->mr_length)
    return Status::InvalidArgument("CPU source, ReadyToken or put belongs to another Runtime/batch");
  // ReadyToken carries the source identity. This validates only local metadata;
  // the CPU checks the selected MR and batch before accepting the actual READ.
  auto status = cpu_pull_->ValidateCpuSource(*desc);
  if (!status.ok()) return status;
  auto& p = *put.state_;
  CopyArrays copies;
  {
    CpuPullStageTimer timer(profile ? &profile->snapshot_us : nullptr);
    status = SnapshotCopyArrays(src_offsets, dst_offsets, lengths, &copies);
  }
  if (!status.ok()) return status;
  if (p.submitted) {
    if (!SameSource(p.request.source, *desc) || p.request.sequence != ready.sequence || p.request.mr_id != ready.mr_id ||
        copies.source != p.request.copies.source || copies.destination != p.request.copies.destination ||
        copies.lengths != p.request.copies.lengths)
      return Status::InvalidArgument("duplicate CPU Pull changed its request");
    // Never resubmit, including after a lost response. The original put owns recovery.
    return Status::OK();
  }
  if (pending_put_ != put.state_ || p.state != CpuPullState::Prepared)
    return Status::InvalidState("put is no longer prepared");
  {
    CpuPullStageTimer timer(profile ? &profile->cover_check_us : nullptr);
    const auto& entries = p.allocation.entries;
    // PreparePut owns distinct DSA slots. Matching each whole entry proves
    // exact, non-overlapping coverage in O(n), regardless of key/address order.
    bool whole_entries = copies.source.size() == entries.size();
    for (const auto& entry : entries) {
      if (entry.address < p.region.base || entry.address-p.region.base > p.region.length ||
          entry.reserved_length > p.region.length-(entry.address-p.region.base))
        return Status::InvalidState("invalid PreparePut allocation");
    }
    for (std::size_t i = 0; i < copies.source.size(); ++i) {
      const auto n = copies.lengths[i];
      if (copies.source[i] > desc->mr_length || n > desc->mr_length-copies.source[i] ||
          copies.destination[i] > p.region.length || n > p.region.length-copies.destination[i])
        return Status::InvalidArgument("CPU Pull range outside registered source/DDR");
      if (whole_entries && (copies.destination[i] != entries[i].address-p.region.base ||
                            n != entries[i].reserved_length)) whole_entries = false;
    }
    if (!whole_entries) {
      // Explicit requests may split or reorder entries. Keep the general
      // coverage check, including gaps, overlaps and crossing-slot rejection.
      std::vector<std::pair<std::uint64_t, std::uint64_t>> destinations, reservations;
      destinations.reserve(copies.destination.size());
      reservations.reserve(entries.size());
      for (std::size_t i = 0; i < copies.destination.size(); ++i)
        destinations.emplace_back(copies.destination[i], copies.lengths[i]);
      for (const auto& entry : entries)
        reservations.emplace_back(entry.address-p.region.base, entry.reserved_length);
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
    }
  }
  {
    CpuPullStageTimer timer(profile ? &profile->request_save_us : nullptr);
    p.request = {*desc, ready.sequence, std::move(copies), ready.mr_id};
  }
  p.submitted = true; p.state = CpuPullState::Submitted;
  // Record the request on the caller's existing put before the first RPC.
  CpuPullState submitted_state = CpuPullState::Submitted;
  try {
    status = timeout_ms ? cpu_pull_->SubmitPreparedCpuPullAndWait(p.request, *timeout_ms, &submitted_state, profile)
                        : cpu_pull_->SubmitCpuPull(p.request);
  }
  catch (const std::exception& e) { status = Status::InDoubt(std::string("CPU Pull submit exception: ")+e.what()); }
  p.state = status.ok() ? submitted_state
      : status.code() == StatusCode::kInDoubt ? CpuPullState::InDoubt : CpuPullState::Failed;
  return status;
}
Status RuntimeCore::QueryCpuPullLocked(const PutHandle& put, CpuPullState* out) {
  if (!out || !put.state_ || !put.state_->cpu || put.state_->owner != connector_owner_)
    return Status::InvalidArgument("CPU Pull put belongs to another Runtime");
  auto& p = *put.state_;
  if (p.state == CpuPullState::Submitted || p.state == CpuPullState::Running || p.state == CpuPullState::InDoubt) {
    if (pending_put_ != put.state_) return Status::InvalidState("CPU Pull put is no longer active");
    CpuPullState state = CpuPullState::InDoubt;
    Status status;
    try { status = cpu_pull_->QueryCpuPull(&state); }
    catch (const std::exception& e) { status = Status::InDoubt(std::string("CPU Pull query exception: ")+e.what()); }
    if (!status.ok()) { p.state = CpuPullState::InDoubt; *out = p.state; return status; }
    p.state = state;
  }
  *out = p.state; return Status::OK();
}
Status RuntimeCore::QueryCpuPull(const PutHandle& put, CpuPullState* out) {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  return QueryCpuPullLocked(put, out);
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
Status RuntimeCore::FinishConnectorPutLocked(const PutHandle& put, bool commit) {
  if (!put.state_ || !put.state_->cpu || put.state_->owner != connector_owner_) return Status::InvalidArgument("foreign Put handle");
  auto& p = *put.state_;
  if (p.state == CpuPullState::Committed || p.state == CpuPullState::Aborted)
    return (p.state == CpuPullState::Committed) == commit ? Status::OK() : Status::InvalidState("Put already finished with another decision");
  if (pending_put_ != put.state_) return Status::InvalidState("Put is not active");
  if (p.decision && *p.decision != commit) return Status::InvalidState("retry the original Put decision");
  // An abort is irrevocable even when querying the original DMA loses its reply.
  // Keep the slots pinned until completion, but never later turn this into Commit.
  if (!commit) { p.decision = false; p.abort_requested = true; }
  if (!p.metadata_finalized) {
    if (p.submitted) {
      CpuPullState state;
      auto status = QueryCpuPullLocked(put, &state);
      if (!status.ok()) return status;
      if (state != CpuPullState::Done && state != CpuPullState::Failed)
        return Status::CompletionUnknown("cannot finalize while DMA may still be active");
    }
    if (commit && (!p.submitted || p.state != CpuPullState::Done))
      return Status::InvalidState("CommitPut requires successful CPU Pull completion");
    p.decision = commit;
    p.completion = p.state == CpuPullState::Done
        ? PutCompletion{PutOutcome::Succeeded, Status::OK()}
        : PutCompletion{PutOutcome::Failed, Status::Internal("CPU Pull aborted")};
  }
  return FinalizePutLocked();
}
Status RuntimeCore::CancelCpuPull(const ReadyToken& ready) {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!initialized_) return Status::NotInitialized("initialize DDR before cancelling CPU Pull");
  if (!cpu_pull_) return Status::NotImplemented("CPU Pull resources unavailable");
  if (!pending_put_) return cpu_pull_->CancelCpuPull(ready);
  if (!pending_put_->cpu) return Status::Busy("ordinary Put owns the write slot");
  auto& p = *pending_put_;
  if (p.submitted) {
    const auto* source = std::get_if<CpuPullSourceDesc>(&ready.source);
    if (!source || !SameSource(*source, p.request.source) || ready.sequence != p.request.sequence ||
        ready.mr_id != p.request.mr_id) return Status::InvalidArgument("cancel token does not match the pending Put");
  } else {
    return Status::Busy("abort the unsubmitted Put before cancelling an unused token");
  }
  if (p.decision && *p.decision) return Status::InvalidState("cannot reverse a commit decision");
  p.decision = false;
  p.abort_requested = true;
  if (p.state != CpuPullState::Failed && p.state != CpuPullState::Done) {
    auto status = cpu_pull_->CancelCpuPull(ready);
    if (!status.ok()) return status;
    p.state = CpuPullState::Failed;
  }
  PutHandle put; put.state_ = pending_put_;
  return FinishConnectorPutLocked(put, false);
}
Status RuntimeCore::CommitPut(const PutHandle& put) {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  return FinishConnectorPutLocked(put, true);
}
Status RuntimeCore::AbortPut(const PutHandle& put) {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  return FinishConnectorPutLocked(put, false);
}

Status RuntimeCore::SetCpuPullExecutor(CpuPullExecutor* executor) {
  std::unique_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!initialized_ || closing_) return Status::InvalidState("DDR is not accepting CPU Pull setup");
  if (pending_put_ || (cpu_pull_ && cpu_pull_ != executor))
    return Status::Busy("cannot replace the write executor");
  cpu_pull_ = executor;
  return Status::OK();
}
bool RuntimeCore::pending_offload() const {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  return (pending_put_ && pending_put_->cpu) || (cpu_pull_ && cpu_pull_->pending());
}
Status RuntimeCore::LastOffloadChecksum(std::uint64_t* value) const {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!cpu_pull_) return Status::NotImplemented("CPU Pull resources unavailable");
  return cpu_pull_->LastChecksum(value);
}
Status RuntimeCore::BatchCpuPull(const std::vector<KeyId>& keys, const ReadyToken& ready,
                                 Uint64ArrayView offsets, const std::vector<std::uint64_t>& lengths,
                                 std::uint32_t timeout, CpuPullProfile* profile) {
  if (profile) *profile = {};
  CpuPullStageTimer total(profile ? &profile->total_us : nullptr);
  {
    CpuPullStageTimer timer(profile ? &profile->validation_us : nullptr);
    if (keys.empty() || keys.size() > 4096 || offsets.size != keys.size() || lengths.size() != keys.size() || !offsets.data)
      return Status::InvalidArgument("CPU Pull keys, source offsets and lengths must match in 1..4096");
  }
  PutHandle handle;
  Status status;
  {
    CpuPullStageTimer timer(profile ? &profile->prepare_put_us : nullptr);
    status = PreparePut(keys, lengths, &handle);
  }
  if (!status.ok()) return status;
  {
    std::lock_guard<std::mutex> lock(put_mutex_);
    batch_pull_put_ = handle;
  }
  try {
    std::vector<std::uint64_t> destinations;
    {
      CpuPullStageTimer timer(profile ? &profile->dst_offsets_us : nullptr);
      destinations.reserve(handle.entries().size());
      for (const auto& entry : handle.entries()) destinations.push_back(entry.address-handle.state_->region.base);
    }
    {
      CpuPullStageTimer timer(profile ? &profile->submit_wait_cpu_pull_us : nullptr);
      status = SubmitCpuPullAndWait(ready, handle, offsets, destinations, lengths, timeout,
                                   profile ? &profile->submit : nullptr);
    }
  } catch (const std::exception& e) {
    status = handle.submitted() ? Status::CompletionUnknown(e.what()) : Status::Internal(e.what());
  }
  if (!status.ok()) {
    if (status.code() == StatusCode::kCompletionUnknown) return status;
    auto aborted = AbortPut(handle);
    if (!aborted.ok()) return aborted;
    return status;
  }
  CpuPullStageTimer timer(profile ? &profile->commit_put_us : nullptr);
  return CommitPut(handle);
}
Status RuntimeCore::ResolveCpuPull(std::uint32_t timeout) {
  PutHandle handle;
  {
    std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
    std::lock_guard<std::mutex> lock(put_mutex_);
    handle = batch_pull_put_;
    if (!handle.state_) return Status::InvalidState("no convenient CPU Pull task to resolve");
    if (pending_put_ && pending_put_ != handle.state_)
      return Status::InvalidState("resolve the current explicit Put using its original handle");
    if (handle.state_->state == CpuPullState::Committed) return Status::OK();
    if (handle.state_->state == CpuPullState::Aborted) return Status::Internal("CPU Pull was aborted");
    if (pending_put_ != handle.state_) return Status::InvalidState("original CPU Pull is not active");
  }
  CpuPullState state = CpuPullState::InDoubt;
  auto status = WaitCpuPull(handle, timeout, &state);
  if (state != CpuPullState::Done && state != CpuPullState::Failed) return status;
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (handle.aborting() || state == CpuPullState::Failed) {
    const auto aborted = FinishConnectorPutLocked(handle, false);
    return aborted.ok() ? Status::Internal("CPU Pull was aborted") : aborted;
  }
  return FinishConnectorPutLocked(handle, true);
}

}  // namespace kvaf
