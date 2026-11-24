// Host-only fault injection against the real RuntimeCore and in-memory metadata.
// The CPU Pull executor is independent from the ordinary AccessOperator.
#include "runtime.h"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
using namespace kvaf;
namespace {
void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void Ok(Status status) { if (!status.ok()) throw std::runtime_error(status.message()); }
bool Unknown(const Status& status) { return status.code() == StatusCode::kCompletionUnknown; }

class Metadata final : public MetaManager {
 public:
  std::unique_ptr<MetaManager> real{CreateInMemoryMetaManager()};
  unsigned commits{}, aborts{};
  bool fail_commit{}, fail_abort{};
  Status RegisterRemoteRegion(const RemoteRegionDesc& r) override { return real->RegisterRemoteRegion(r); }
  Status PreparePut(const std::vector<KeyId>& k, const std::vector<std::uint64_t>& n,
                    PutContext* c, PutAccessDesc* d) override { return real->PreparePut(k,n,c,d); }
  Status CommitPut(const PutContext& c) override {
    ++commits;
    if (fail_commit) { fail_commit = false; return Status::Internal("injected metadata commit failure"); }
    return real->CommitPut(c);
  }
  Status AbortPut(const PutContext& c) override {
    ++aborts;
    if (fail_abort) { fail_abort = false; return Status::Internal("injected metadata abort failure"); }
    return real->AbortPut(c);
  }
  Status LockKeys(const std::vector<KeyId>& k, MetaManagerLockMode m, LockContext* c) override { return real->LockKeys(k,m,c); }
  Status BatchLookup(const std::vector<KeyId>& k, const LockContext& c, FlatAccessDesc* d) override { return real->BatchLookup(k,c,d); }
  Status UnlockKeys(const LockContext& c) override { return real->UnlockKeys(c); }
  Status BatchFree(const std::vector<KeyId>& k) override { return real->BatchFree(k); }
  bool Visible(KeyId key) {
    LockContext lock;
    auto s = real->LockKeys({key}, MetaManagerLockMode::kStableReadAllowAppend, &lock);
    if (s.ok()) Ok(real->UnlockKeys(lock));
    return s.ok();
  }
};
class Allocator final : public DdrMemoryAllocator {
 public:
  std::unique_ptr<DdrMemoryAllocator> real{CreateHostDdrMemoryAllocator()};
  unsigned releases{};
  bool lose_release_reply{}, released{};
  Status Allocate(const RuntimeInitRequest& q, RemoteRegionDesc* r) override { return real->Allocate(q,r); }
  Status Release(const RemoteRegionDesc& r) override {
    ++releases;
    if (!released) { auto s=real->Release(r); if(!s.ok()) return s; released=true; }
    if (lose_release_reply) { lose_release_reply=false; return Status::CompletionUnknown("lost DDR release reply"); }
    return Status::OK();
  }
};
class Ordinary final : public AccessOperator {
 public:
  unsigned puts{}, gets{}, acks{};
  bool unknown{};
  PutCompletion resolved{PutOutcome::Succeeded, Status::OK()};
  Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
  Status Shutdown() override { return Status::OK(); }
  Status ExecuteBatchPut(const BatchPutExecRequest& in) override {
    ++puts;
    for (std::size_t i=0;i<in.put_desc->entries.size();++i) {
      const auto& entry=in.put_desc->entries[i];
      std::memcpy(reinterpret_cast<void*>(entry.address),
          static_cast<const char*>(in.request->input.data)+in.request->input_offsets[i], entry.reserved_length);
    }
    return unknown ? Status::CompletionUnknown("lost ordinary Put completion") : Status::OK();
  }
  Status ResolvePut(PutCompletion* out) override { *out=resolved; return Status::OK(); }
  Status AcknowledgePut() override { ++acks; return Status::OK(); }
  Status ExecuteBatchGet(const BatchGetExecRequest& in) override {
    ++gets;
    Check(in.host_access_table.size==1 && in.host_access_table.valid[0], "Get must see a committed pinned slot");
    std::memcpy(in.request->output.data, reinterpret_cast<void*>(in.host_access_table.addresses[0]), 16);
    return Status::OK();
  }
};
class Pull final : public CpuPullExecutor {
 public:
  RemoteRegionDesc region;
  std::array<std::uint8_t,64> source{};
  unsigned submits{}, queries{}, cancellations{}, completions{}, closes{};
  bool active{}, rejected{}, lose_submit_reply{}, fail_ack{}, closed{};
  CpuPullState result{CpuPullState::Done};
  bool last_committed{};
  CpuPullRequest request;
  std::mutex mutex;
  std::condition_variable condition;
  bool block_submit{}, entered{}, unblock{};
  Pull() { source.fill(0xA5); }
  Status CanShutdown() const override { return active ? Status::CompletionUnknown("Pull active") : Status::OK(); }
  Status Shutdown() override { auto s=CanShutdown(); if(s.ok() && !closed) { ++closes; closed=true; } return s; }
  Status ValidateCpuSource(const CpuPullSourceDesc& d) override {
    return d.source_id==31 && d.region_id==41 && d.region_epoch==51 && d.mr_length==source.size()
        ? Status::OK() : Status::InvalidArgument("wrong source identity");
  }
  void Copy() {
    for (std::size_t i=0;i<request.copies.lengths.size();++i)
      std::memcpy(reinterpret_cast<void*>(region.base+request.copies.destination[i]),
                  source.data()+request.copies.source[i],request.copies.lengths[i]);
  }
  Status Submit(const CpuPullRequest& in, CpuPullState* out) {
    ++submits; request=in;
    if (rejected) return Status::InvalidState("definitively rejected before READ");
    active=true;
    if (block_submit) {
      std::unique_lock<std::mutex> lock(mutex); entered=true; condition.notify_all();
      condition.wait(lock,[&]{return unblock;});
    }
    if (result==CpuPullState::Done) Copy();
    *out=result;
    return lose_submit_reply || result==CpuPullState::Running || result==CpuPullState::InDoubt
        ? Status::CompletionUnknown("completion unavailable") : Status::OK();
  }
  Status SubmitCpuPull(const CpuPullRequest& in) override { CpuPullState out; return Submit(in,&out); }
  Status SubmitPreparedCpuPullAndWait(const CpuPullRequest& in, std::uint32_t,
                                      CpuPullState* out, CpuPullSubmitTiming*) override { return Submit(in,out); }
  Status QueryCpuPull(CpuPullState* out) override {
    ++queries; *out=result; if(result==CpuPullState::Done) Copy(); return Status::OK();
  }
  Status CancelCpuPull(const ReadyToken&) override {
    ++cancellations;
    if (result==CpuPullState::Running || result==CpuPullState::InDoubt)
      return Status::CompletionUnknown("cancel waits for DMA quiescence");
    result=CpuPullState::Failed; return Status::OK();
  }
  Status CompleteOffload(bool committed) override {
    ++completions;
    if (fail_ack) { fail_ack=false; return Status::Internal("injected completion acknowledgement failure"); }
    last_committed=committed; active=false; return Status::OK();
  }
  Status LastChecksum(std::uint64_t*) const override { return Status::NotImplemented("not diagnostic"); }
  bool pending() const override { return active; }
};
struct Fixture {
  Metadata meta; Ordinary ordinary; Allocator allocator; Pull pull;
  RuntimeCore core{meta,ordinary,allocator};
  std::array<std::uint8_t,16> input{};
  ReadyToken ready{CpuPullSourceDesc{31,41,51,64},1,1};
  std::vector<std::uint64_t> source{0}, lengths{16};
  Fixture() {
    input.fill(0x3C); Ok(core.Initialize({256,16,DeviceKind::kCpu}));
    pull.region=core.remote_region(); Ok(core.SetCpuPullExecutor(&pull));
  }
  BatchPutRequest Put(KeyId key) { return {{key},{input.data(),input.size(),DeviceKind::kCpu},{0},{16}}; }
  PutHandle Prepare(KeyId key) { PutHandle handle; Ok(core.PreparePut({key},lengths,&handle)); return handle; }
  std::vector<std::uint64_t> Destination(const PutHandle& handle) {
    return {handle.entries()[0].address-core.remote_region().base};
  }
  Status Submit(const PutHandle& handle) {
    return core.SubmitCpuPullAndWait(ready,handle,source,Destination(handle),lengths,10);
  }
  bool Get(KeyId key) {
    AccessTableHandle table; Ok(core.PrepareAccessTable({{key}},&table));
    std::int32_t index=0, offset=0, count=1;
    std::array<std::uint8_t,16> output{};
    BatchGetRequest q; q.access_table=table;
    q.indices={&index,sizeof(index),DeviceKind::kCpu}; q.offsets={&offset,sizeof(offset),DeviceKind::kCpu};
    q.output_indices=q.indices; q.miss_count={&count,sizeof(count),DeviceKind::kCpu};
    q.output={output.data(),output.size(),DeviceKind::kCpu}; q.output_block_bytes=16;
    auto status=core.BatchGet(q); Ok(core.ReleaseAccessTable(table)); Ok(status);
    return output==input;
  }
  void Close() { Ok(core.Shutdown()); Check(allocator.released,"DDR release after resources close"); }
};

void SharedWriterAndIndependentGet() {
  Fixture f; Ok(f.core.BatchPut(f.Put(0)));
  f.pull.result=CpuPullState::Running;
  auto cpu=f.Prepare(1); Check(Unknown(f.Submit(cpu)),"CPU Pull retains unknown completion");
  Check(!f.meta.Visible(1),"unresolved CPU slot is invisible");
  Check(f.core.BatchPut(f.Put(2)).code()==StatusCode::kBusy,"CPU Pull excludes ordinary Put");
  PutHandle other; Check(f.core.PreparePut({2},f.lengths,&other).code()==StatusCode::kBusy,"CPU Pull excludes another CPU reservation");
  Check(f.Get(0),"Get of other slot progresses during CPU Pull");
  Check(f.core.RecoverPut({Transport::Urma,Executor::Aiv}).code()==StatusCode::kBusy,"ordinary recovery cannot consume CPU transaction");
  f.pull.result=CpuPullState::Done; CpuPullState state;
  Ok(f.core.QueryCpuPull(cpu,&state)); Ok(f.core.CommitPut(cpu));
  Check(f.meta.Visible(1) && f.pull.submits==1,"completion commits without replay");
  f.ordinary.unknown=true; Check(Unknown(f.core.BatchPut(f.Put(2))),"ordinary Put retains unknown completion");
  Check(f.core.PreparePut({3},f.lengths,&other).code()==StatusCode::kBusy,"ordinary Put excludes CPU reservation");
  Ok(f.core.RecoverPut({Transport::Urma,Executor::Aiv}));
  Check(f.ordinary.puts==2 && f.meta.Visible(2),"ordinary recovery never resubmits"); f.Close();
}
void ConcurrentGetDuringSingleRpc() {
  Fixture f; Ok(f.core.BatchPut(f.Put(0))); auto handle=f.Prepare(1); f.pull.block_submit=true;
  auto submit=std::async(std::launch::async,[&]{return f.Submit(handle);});
  {
    std::unique_lock<std::mutex> lock(f.pull.mutex);
    Check(f.pull.condition.wait_for(lock,std::chrono::seconds(2),[&]{return f.pull.entered;}),"submit entered fake RPC");
  }
  auto get=std::async(std::launch::async,[&]{return f.Get(0);});
  const bool independent=get.wait_for(std::chrono::seconds(1))==std::future_status::ready;
  { std::lock_guard<std::mutex> lock(f.pull.mutex); f.pull.unblock=true; } f.pull.condition.notify_all();
  Ok(submit.get()); Check(get.get() && independent,"Get must not wait for CPU write/RPC mutex");
  Ok(f.core.CommitPut(handle)); Check(f.pull.submits==1 && f.pull.queries==0,"successful path uses one submit-and-wait call"); f.Close();
}
void RejectionAndImmutableSnapshot() {
  Fixture f; f.pull.rejected=true;
  Check(!f.core.BatchCpuPull({0},f.ready,f.source,f.lengths,10).ok(),"definitive rejection surfaces");
  Check(!f.core.HasPendingPut() && !f.meta.Visible(0) && f.meta.aborts==1,"rejection aborts reserved metadata");
  f.pull.rejected=false; auto handle=f.Prepare(0); Ok(f.Submit(handle));
  auto changed=f.source; changed[0]=1;
  Check(!f.core.SubmitCpuPullAndWait(f.ready,handle,changed,f.Destination(handle),f.lengths,10).ok(),"duplicate cannot change source offsets");
  Ok(f.Submit(handle)); Check(f.pull.submits==2,"identical duplicate does not send another RPC");
  Ok(f.core.CommitPut(handle)); f.Close();
}
void MetadataAndAcknowledgementRetries() {
  Fixture f; auto handle=f.Prepare(0); Ok(f.Submit(handle)); f.meta.fail_commit=true;
  Check(!f.core.CommitPut(handle).ok(),"metadata commit failure retained");
  Check(!f.core.AbortPut(handle).ok(),"commit decision cannot flip to abort");
  Check(f.core.HasPendingPut() && !f.meta.Visible(0),"metadata failure retains write ownership");
  f.pull.fail_ack=true;
  Check(!f.core.CommitPut(handle).ok() && f.meta.Visible(0),"ack failure occurs after metadata publish");
  const auto commits=f.meta.commits; Ok(f.core.CommitPut(handle));
  Check(f.meta.commits==commits && f.pull.completions==2 && !f.core.HasPendingPut(),"ack retry must not repeat metadata commit");
  f.Close();
}
void UnknownRecoveryAndStickyAbort(bool cancel) {
  Fixture f; f.pull.result=CpuPullState::Running;
  Check(Unknown(f.core.BatchCpuPull({0},f.ready,f.source,f.lengths,10)),"convenient Pull retains unknown task");
  // Explicit Abort and Cancel both choose abort before DMA has settled.
  // Use an explicit handle for Abort because BatchCpuPull intentionally hides it.
  if (cancel) {
    Check(Unknown(f.core.CancelCpuPull(f.ready)),"Cancel cannot retire active DMA");
    f.pull.result=CpuPullState::Done;
    Check(!f.core.ResolveCpuPull(10).ok(),"recovery preserves cancellation despite eventual READ success");
    Check(!f.meta.Visible(0) && !f.pull.last_committed && f.meta.aborts==1,"cancellation never publishes completed bytes");
    f.Close(); return;
  }
  f.pull.result=CpuPullState::Done; Ok(f.core.ResolveCpuPull(10));
  auto handle=f.Prepare(1); f.pull.result=CpuPullState::Running; ++f.ready.sequence;
  Check(Unknown(f.Submit(handle)),"explicit Pull unresolved");
  Check(Unknown(f.core.AbortPut(handle)),"Abort waits for DMA");
  Check(handle.aborting(),"Abort decision must be sticky before Query/DMA completion");
  Check(!f.core.CommitPut(handle).ok(),"sticky abort cannot turn into commit");
  f.pull.result=CpuPullState::Done; Ok(f.core.AbortPut(handle));
  Check(!f.meta.Visible(1) && !f.pull.last_committed,"abort after successful READ rolls metadata back"); f.Close();
}
void ClosingAllowsOriginalRecovery() {
  Fixture f; f.pull.result=CpuPullState::Running;
  Check(Unknown(f.core.BatchCpuPull({0},f.ready,f.source,f.lengths,10)),"Pull running before close");
  Check(!f.core.Shutdown().ok() && !f.allocator.released,"close retains unresolved write and DDR");
  Check(!f.core.BatchPut(f.Put(1)).ok(),"closing refuses new ordinary write");
  PutHandle handle; Check(!f.core.PreparePut({1},f.lengths,&handle).ok(),"closing refuses new CPU reservation");
  f.pull.result=CpuPullState::Done; Ok(f.core.ResolveCpuPull(10));
  Check(f.meta.Visible(0) && f.pull.submits==1,"original CPU task recovers while closing");
  f.allocator.lose_release_reply=true;
  Check(Unknown(f.core.Shutdown()) && f.allocator.released,"release can succeed with lost reply");
  Ok(f.core.Shutdown()); Check(f.pull.closes==1 && f.allocator.releases==2,"closing retries only retained ownership, CPU close is idempotent");
}
void ClosingAbortsUnsubmittedReservation() {
  Fixture f; auto handle=f.Prepare(0); f.Close();
  Check(handle.state()==CpuPullState::Aborted && f.meta.aborts==1 && f.pull.submits==0,
        "closing safely aborts a never-submitted reservation");
}
void UnusedCancelDoesNotReserveDdr() {
  Fixture f;
  // Model a lost cancellation reply for an unused Ready token: executor has a
  // pending fence, but no accepted DDR transfer or Core metadata transaction.
  f.pull.active=true;
  Ok(f.core.BatchPut(f.Put(0)));
  PutHandle handle;
  Check(f.core.PreparePut({1},f.lengths,&handle).code()==StatusCode::kBusy,
        "unresolved unused-token fence must reject new CPU preparation before reserving slots");
  Check(!f.core.HasPendingPut() && !f.meta.Visible(1),"unused cancellation does not own DDR write slots");
  f.pull.active=false; handle=f.Prepare(1); Ok(f.core.AbortPut(handle)); f.Close();
}
}
int main() {
  try {
    SharedWriterAndIndependentGet(); ConcurrentGetDuringSingleRpc(); RejectionAndImmutableSnapshot();
    MetadataAndAcknowledgementRetries(); UnknownRecoveryAndStickyAbort(true); UnknownRecoveryAndStickyAbort(false);
    ClosingAllowsOriginalRecovery(); ClosingAbortsUnsubmittedReservation(); UnusedCancelDoesNotReserveDdr();
    std::cout<<"CPU Pull write transaction checks passed\n";
  } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
