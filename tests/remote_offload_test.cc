// Real NpuRuntime/Core/HBM exporter/control protocol; simulated ACL/HCCP and
// CPU DMA. Hardware permissions and cache visibility require the two-card test.
#include "offload_fake_server.h"
#include "npu_runtime.h"
#include "remote_pull_operator.h"
#include "meta_manager.h"
#include <chrono>
#include <sys/wait.h>

void Ok(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message()); }
void NoResources(){Require(leases==0&&registrations==0&&sessions==0&&contexts==0&&jetties==0&&cqs==0&&peer_imports==0,"all registrations and leases released");}
void Lifecycle(){
  SharedServer server;
  kvaf::NpuRuntimeOptions options;options.enable_cpu_offload=true;options.cpu_candidate=1;
  kvaf::NpuRuntime a(options);Ok(a.LinkTo({server.address}));Ok(a.Initialize(1024,256));
  kvaf::OffloadRegionTicket ticket;Ok(a.GetOffloadRegion(&ticket));
  std::vector<std::uint8_t> hbm(4096);for(unsigned i=0;i<hbm.size();++i)hbm[i]=i*3+7;
  aclrtSetDevice(7);kvaf::CpuPullHbmSource b(7,"00000000000000000000000000000001");
  Ok(b.Register(ticket,{hbm.data(),hbm.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)));
  Require(std::all_of(server.pool.begin(),server.pool.end(),[](auto c){return c==0xA5;}),"probe preserves shared DDR");
  kvaf::CpuPullSourceDesc desc;Ok(b.GetDescriptor(&desc));Ok(a.AttachCpuSource(desc));
  kvaf::ReadyToken ready;Ok(b.Ready(&ready));
  Require(!b.ReleaseBatch(ready).ok()&&!a.Shutdown().ok(),"source lease blocks premature release");
  kvaf::PutHandle put;Ok(a.PreparePut({1,3},{64,128},&put));
  std::vector<std::uint64_t> src{16,512},dst{put.entries()[0].address-ticket.region.base,put.entries()[1].address-ticket.region.base},sizes{64,128};
  const int before=payload_reads, hashes=checksum_calls;
  Ok(a.SubmitCpuPull(ready,put,src,dst,sizes));kvaf::CpuPullState state;
  Ok(a.WaitCpuPull(put,1000,&state));Ok(a.CommitPut(put));
  Require(payload_reads==before+1&&checksum_calls==hashes,"transfer and commit exclude checksum");
  std::uint64_t hash{},corrupted{};Ok(a.LastOffloadChecksum(&hash));
  Require(hash==p::Hash(hbm.data()+512,128,p::Hash(hbm.data()+16,64)),"DDR checksum");
  server.pool[256]^=0x80;Ok(a.LastOffloadChecksum(&corrupted));Require(corrupted!=hash,"diagnostic scans current DDR");server.pool[256]^=0x80;
  server.drop=unsigned(rp::Command::Checksum);Require(!a.LastOffloadChecksum(&corrupted).ok()&&!a.pending_offload(),"lost diagnostic ACK is not pending DMA");
  Ok(a.LastOffloadChecksum(&corrupted));Require(corrupted==hash&&payload_reads==before+1,"diagnostic retry never replays READ");
  fail_checksum=true;Require(!a.LastOffloadChecksum(&corrupted).ok()&&!a.pending_offload(),"diagnostic failure leaves commit intact");fail_checksum=false;
  Ok(b.ReleaseBatch(ready));Ok(b.Ready(&ready));Require(!a.LastOffloadChecksum(&hash).ok(),"next generation invalidates old checksum");Ok(b.CancelBatch(ready));
  server.drop=unsigned(rp::Command::Detach);Require(!b.Shutdown().ok(),"lost detach ACK retains exporter");Ok(b.Shutdown());NoResources();
  server.fail_release=true;Require(!a.Shutdown().ok(),"DDR release can be retried");Ok(a.Shutdown());
}
// The dispatcher routes both data planes through one Core/MetaManager/pool.
struct PoolAllocator : kvaf::DdrMemoryAllocator {
  std::shared_ptr<kvaf::peer::PeerLink> link;
  explicit PoolAllocator(std::shared_ptr<kvaf::peer::PeerLink> p):link(std::move(p)){}
  Status Allocate(const kvaf::RuntimeInitRequest& r,kvaf::RemoteRegionDesc* out)override{
    kvaf::peer::PeerRegion region;auto s=link->AllocateRegion(r.ddr_bytes,r.slot_size,0,&region);
    if(s.ok())*out={region.base,region.length,r.slot_size,kvaf::DeviceKind::kCpu};return s;
  }
  Status Release(const kvaf::RemoteRegionDesc& r)override{return link->ReleaseRegion(r.base);}
};
struct FakeAiv : kvaf::AccessOperator {
  int puts{};
  Status Initialize(const kvaf::RemoteRegionDesc&)override{return Status::OK();}
  Status Shutdown()override{return Status::OK();}
  Status ExecuteBatchGet(const kvaf::BatchGetExecRequest&)override{return Status::NotImplemented("model");}
  Status ExecuteBatchPut(const kvaf::BatchPutExecRequest& r)override{
    ++puts;for(std::size_t i=0;i<r.request->keys.size();++i)
      std::memcpy(reinterpret_cast<void*>(r.put_desc->entries[i].address),
        static_cast<const std::uint8_t*>(r.request->input.data)+r.request->input_offsets[i],r.request->lengths[i]);
    return Status::OK();
  }
};
void SharedMetadata(){
  SharedServer server;auto link=std::make_shared<kvaf::peer::PeerLink>(server.address);
  PoolAllocator allocator(link);FakeAiv aiv;kvaf::RemotePullOperator op(&aiv,link);
  auto meta=kvaf::CreateInMemoryMetaManager();kvaf::RuntimeCore core(*meta,op,allocator);Ok(core.Initialize({1024,256,kvaf::DeviceKind::kCpu}));
  std::vector<std::uint8_t> a_data(64,0x23),b_data(1024,0x67);
  Ok(core.BatchPut({{0},{a_data.data(),a_data.size(),kvaf::DeviceKind::kNpu},{0},{64}}));
  aclrtSetDevice(7);kvaf::CpuPullHbmSource b(7,"00000000000000000000000000000001");
  Ok(b.Register(op.ticket(),{b_data.data(),b_data.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)));
  Require(!std::memcmp(server.pool.data(),a_data.data(),64),"B registration probe preserves A BatchPut");
  kvaf::ReadyToken ready;Ok(b.Ready(&ready));
  Ok(core.AttachCpuSource(std::get<kvaf::CpuPullSourceDesc>(ready.source)));
  kvaf::PutHandle put;Ok(core.PreparePut({1},{64},&put));
  std::vector<std::uint64_t> src{64},dst{put.entries()[0].address-core.remote_region().base},sizes{64};
  Ok(core.SubmitCpuPull(ready,put,src,dst,sizes));kvaf::CpuPullState state;
  Ok(core.WaitCpuPull(put,1000,&state));Ok(core.CommitPut(put));
  Require(aiv.puts==1&&!std::memcmp(server.pool.data()+256,b_data.data()+64,64),"Pull bypasses AIV and shares DDR");
  const int copied=payload_reads;
  Require(!core.BatchPut({{1},{a_data.data(),a_data.size(),kvaf::DeviceKind::kNpu},{0},{64}}).ok(),"same MetaManager prevents duplicate key across data planes");
  Require(aiv.puts==1&&payload_reads==copied,"conflicting key never reaches either data plane");
  kvaf::AccessTableHandle table;Ok(core.PrepareAccessTable({{0,1}},&table));
  kvaf::AccessTableView view;Ok(core.GetAccessTableView(table,&view));
  Require(view.valid[0]&&view.valid[1]&&view.addresses[0]==server.info.base&&view.addresses[1]==server.info.base+256,"one access table sees both writes");
  Require(!core.Shutdown().ok(),"busy B blocks shutdown before unlocking tables");Ok(core.GetAccessTableView(table,&view));
  Ok(core.ReleaseAccessTable(table));Ok(b.ReleaseBatch(ready));Ok(b.Shutdown());Ok(core.Shutdown());NoResources();
}
int producer_syncs=0;
extern "C" aclError aclrtSynchronizeStream(aclrtStream){++producer_syncs;return 0;}
void UnifiedBatchCpuPull() {
  EidFixture fixture;
  const std::string eid1="00000000000000000000000000000001",eid5="00000000000000000000000000000005";
  EidFixture::Write(fixture.paths.rootinfo,EidFixture::Clos(0,eid1)+EidFixture::Clos(0,eid5));
  const char* prior=getenv("KVAF_HCCL_ROOTINFO_PATH");const bool had_prior=prior;const std::string saved=prior?prior:"";
  setenv("KVAF_HCCL_ROOTINFO_PATH",fixture.paths.rootinfo.c_str(),1);auto_mode=true;failed_route=1;
  SharedServer server;kvaf::NpuRuntimeOptions options;options.enable_cpu_offload=true;options.offload_timeout_ms=1000;
  kvaf::NpuRuntime reader(options);Ok(reader.LinkTo({server.address}));Ok(reader.Initialize(1024,256));
  kvaf::OffloadRegionTicket ticket;Ok(reader.GetOffloadRegion(&ticket));
  std::vector<std::uint8_t> first(512,0x13),second(4096);
  for(std::size_t i=0;i<second.size();++i)second[i]=(i*17+3)%251;
  std::vector<kvaf::ConstBufferView> views{{first.data(),first.size(),kvaf::DeviceKind::kNpu},
                                         {second.data(),second.size(),kvaf::DeviceKind::kNpu}};
  aclrtSetDevice(0);kvaf::CpuPullHbmSource source(0);
  Ok(source.Register(ticket,views,reinterpret_cast<void*>(1)));
  kvaf::CpuPullSourceDesc desc;Ok(source.GetDescriptor(&desc));Ok(reader.AttachCpuSource(desc));
  kvaf::ReadyToken ready;Ok(source.Ready(views[1],&ready));
  std::vector<std::uint64_t> offsets{32,600},lengths{64,32};
  const int reads=payload_reads,probes=scratch_reads,syncs=producer_syncs;
  auto wrong=ready;wrong.mr_id=3;
  Require(!reader.BatchCpuPull({3,1},wrong,offsets,lengths).ok(),"reject missing MR");
  wrong=ready;wrong.source=kvaf::NpuPullSourceDesc{13,4096,7};
  Require(!reader.BatchCpuPull({3,1},wrong,offsets,lengths).ok(),"reject NPU Pull Ready");
  Require(!reader.BatchCpuPull({3,1},ready,offsets,{64}).ok(),"reject unequal arrays");
  Require(!reader.BatchCpuPull({3,1},ready,std::vector<std::uint64_t>{UINT64_MAX,600},lengths).ok(),"reject overflowing source");
  Require(!reader.BatchCpuPull({3,1},ready,offsets,{0,32}).ok(),"reject empty span");
  wrong=ready;std::get<kvaf::CpuPullSourceDesc>(wrong.source).source_id++;
  Require(!reader.BatchCpuPull({3,1},wrong,offsets,lengths).ok()&&!reader.pending_offload(),
          "unattached source rolls back the internal Put");
  Require(payload_reads==reads,"invalid convenience calls do not submit READ");
  kvaf::CpuPullProfile profile;profile.total_us=1e12;
  Ok(reader.BatchCpuPull({3,1},ready,offsets,lengths,1000,&profile));
  const auto stage_sum=profile.validation_us+profile.prepare_put_us+profile.dst_offsets_us+
      profile.submit_cpu_pull_us+profile.wait_cpu_pull_us+profile.commit_put_us;
  Require(profile.prepare_put_us>0&&profile.submit_cpu_pull_us>0&&profile.wait_cpu_pull_us>0&&
          profile.commit_put_us>0&&profile.total_us>=stage_sum&&profile.total_us<1e12,
          "optional wrapper profile resets output and nests Host stage durations in native total");
  Require(!reader.pending_offload()&&source.frozen(),"success commits but never releases the producer batch");
  Require(!std::memcmp(server.pool.data()+768,second.data()+32,64)&&
          !std::memcmp(server.pool.data()+256,second.data()+600,32),"key order selects the correct destinations");
  kvaf::AccessTableHandle table;Ok(reader.core().PrepareAccessTable({{3,1}},&table));kvaf::AccessTableView view;
  Ok(reader.core().GetAccessTableView(table,&view));
  Require(view.valid[0]&&view.valid[1]&&view.lengths[0]==64&&view.lengths[1]==32,"success includes metadata commit");
  Ok(reader.core().ReleaseAccessTable(table));
  Require(!reader.BatchCpuPull({3,1},ready,offsets,lengths).ok()&&!reader.pending_offload()&&payload_reads==reads+1,
          "new call with occupied keys fails without replay");
  Require(!reader.ResolveCpuPull().ok(),"no hidden batch remains after success");
  Ok(source.ReleaseBatch(ready));Ok(reader.core().BatchFree({{3,1}}));

  // A successful submit followed by timeout retains the original owned arrays.
  Ok(source.Ready(views[1],&ready));hold_read=true;
  Require(reader.BatchCpuPull({3,1},ready,offsets,lengths,1).code()==kvaf::StatusCode::kInDoubt&&reader.pending_offload(),
          "timeout retains internal transaction");
  offsets[0]=70;lengths[0]=16;
  Require(!reader.BatchCpuPull({3,1},ready,offsets,lengths).ok(),"cannot start a second hidden transaction");
  kvaf::PutHandle explicit_put;
  Require(!reader.PreparePut({0},{8},&explicit_put).ok()&&!reader.core().BatchFree({{3,1}}).ok(),"pending DMA protects DDR");
  Require(!reader.ResolveCpuPull(0).ok()&&!source.ReleaseBatch(ready).ok(),"no premature completion or producer reuse");
  hold_read=false;Ok(reader.ResolveCpuPull(1000));
  Require(payload_reads==reads+2&&!std::memcmp(server.pool.data()+768,second.data()+32,64),"recovery uses snapshot without replay");
  offsets[0]=32;lengths[0]=64;
  Ok(source.ReleaseBatch(ready));Ok(reader.core().BatchFree({{3,1}}));

  Ok(source.Ready(views[1],&ready));server.drop=unsigned(rp::Command::Submit);
  Require(reader.BatchCpuPull({3,1},ready,offsets,lengths).code()==kvaf::StatusCode::kInDoubt,"lost Submit retains hidden Put");
  Ok(reader.ResolveCpuPull(1000));Require(payload_reads==reads+3,"lost Submit resolves without another READ");
  Ok(source.ReleaseBatch(ready));Ok(reader.core().BatchFree({{3,1}}));

  // Lost pre-submit query is a definite failure; Ready stays producer-owned.
  Ok(source.Ready(views[1],&ready));server.drop=unsigned(rp::Command::Query);
  Require(!reader.BatchCpuPull({3,1},ready,offsets,lengths).ok()&&!reader.pending_offload()&&payload_reads==reads+3,
          "pre-submit failure aborts slots without DMA");
  Ok(source.CancelBatch(ready));
  Ok(source.Ready(views[1],&ready));Ok(source.CancelBatch(ready));
  Require(!reader.BatchCpuPull({3,1},ready,offsets,lengths).ok()&&!reader.pending_offload(),"cancelled Ready rolls back Put");

  // Terminal DMA failure must not publish keys or report a successful resolve.
  Ok(source.Ready(views[1],&ready));fail_read=true;
  Require(!reader.BatchCpuPull({3,1},ready,offsets,lengths,1000).ok()&&!reader.pending_offload(),"terminal failure safely aborts");
  fail_read=false;Ok(source.ReleaseBatch(ready));
  Require(reader.core().PrepareAccessTable({{3,1}},&table).code()==kvaf::StatusCode::kNotFound,
          "aborted data is not published");

  // Failed abort acknowledgement retains the abort decision, never commits.
  Ok(source.Ready(views[1],&ready));fail_read=true;server.drop=unsigned(rp::Command::Complete);
  Require(!reader.BatchCpuPull({3,1},ready,offsets,lengths,1000).ok()&&reader.pending_offload(),"lost abort ACK stays pending");
  Require(!reader.ResolveCpuPull(1000).ok()&&!reader.pending_offload(),"resolved failure still reports failure");
  fail_read=false;Ok(source.ReleaseBatch(ready));

  Ok(source.Ready(views[1],&ready));server.drop=unsigned(rp::Command::Complete);
  Require(!reader.BatchCpuPull({3,1},ready,offsets,lengths,1000).ok()&&reader.pending_offload(),"lost commit ACK retains decision");
  const int committed=payload_reads;Ok(reader.ResolveCpuPull(1000));
  Require(payload_reads==committed,"commit recovery never replays READ");
  Ok(source.ReleaseBatch(ready));Ok(reader.core().BatchFree({{3,1}}));

  // The generic recovery entry must not consume an application's explicit Put.
  Ok(reader.PreparePut({0},{8},&explicit_put));
  Require(!reader.ResolveCpuPull().ok()&&reader.pending_offload(),"explicit transactions require their original handle");
  Ok(reader.AbortPut(explicit_put));
  Ok(source.Ready(views[1],&ready));hold_read=true;server.drop=unsigned(rp::Command::Submit);
  Require(!reader.BatchCpuPull({3,1},ready,offsets,lengths).ok(),"retain for shutdown");
  Require(!reader.Shutdown().ok()&&reader.pending_offload(),"shutdown cannot free active DMA");
  hold_read=false;
  for(int i=0;i<1000&&reader.pending_offload();++i) {
    Require(!reader.Shutdown().ok(),"producer registration still blocks DDR release");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  Require(!reader.pending_offload()&&source.frozen(),"shutdown resolves original transaction but never releases source");
  Ok(source.ReleaseBatch(ready));Ok(source.Shutdown());Ok(reader.Shutdown());NoResources();
  Require(producer_syncs==syncs&&scratch_reads==probes,"batch calls never synchronize producer or rebuild registrations");
  auto_mode=false;failed_route=0;
  if(had_prior)setenv("KVAF_HCCL_ROOTINFO_PATH",saved.c_str(),1);else unsetenv("KVAF_HCCL_ROOTINFO_PATH");
}
void InitializationRecovery() {
  SharedServer server;kvaf::NpuRuntimeOptions options;options.enable_cpu_offload=true;
  kvaf::NpuRuntime a(options);Ok(a.LinkTo({server.address}));
  server.drop=std::uint16_t(rp::Command::Describe);server.fail_release=true;
  Require(!a.Initialize(1024,256).ok()&&a.initialized()&&!a.ddr_initialized(),"lost description and failed cleanup preserve failed initialization");
  Require(!a.Initialize(1024,256).ok()&&server.pool_allocations==1,"reinitialization cannot discard an unresolved allocation");
  Ok(a.Shutdown());Require(server.pool.empty(),"shutdown retries failed initialization cleanup");
  Ok(a.Initialize(1024,256));Ok(a.Shutdown());NoResources();
}
void ConnectorApi(){
  SharedServer server;kvaf::NpuRuntimeOptions options;options.enable_cpu_offload=true;
  kvaf::NpuRuntime reader(options);Ok(reader.LinkTo({server.address}));Ok(reader.Initialize(1024,256));
  kvaf::OffloadRegionTicket ticket;Ok(reader.GetOffloadRegion(&ticket));
  std::vector<std::uint8_t> hbm(4096);for(size_t i=0;i<hbm.size();++i)hbm[i]=(i*17+3)%251;
  aclrtSetDevice(7);kvaf::CpuPullHbmSource producer(7,"00000000000000000000000000000001");
  Ok(producer.Register(ticket,{hbm.data(),hbm.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)));
  kvaf::CpuPullSourceDesc desc;Ok(producer.GetDescriptor(&desc));
  Ok(reader.AttachCpuSource(desc));
  Ok(reader.AttachCpuSource(desc)); // Rebinding the same identity is idempotent.
  auto bad_desc=desc;bad_desc.region_epoch++;
  Require(!reader.AttachCpuSource(bad_desc).ok(),"stale source descriptor rejected");
  kvaf::ReadyToken ready;
  server.drop=unsigned(rp::Command::Ready);
  Require(!producer.Ready(&ready).ok()&&producer.frozen(),"lost identity-only Ready keeps batch frozen");
  const auto sequence=ready.sequence;Ok(producer.Ready(&ready));Require(ready.sequence==sequence,"Ready retry reuses sequence");
  kvaf::PutHandle put;Ok(reader.PreparePut({1,3},{64,32},&put));
  kvaf::CpuPullState state;
  const auto original_put=put;
  Require(!put.submitted() && put.state()==kvaf::CpuPullState::Prepared,"prepared put has no submission");
  Ok(reader.QueryCpuPull(put,&state));Require(state==kvaf::CpuPullState::Prepared,"query prepared put without a remote request");
  Require(reader.WaitCpuPull(put,0,&state).code()==kvaf::StatusCode::kInvalidState,
          "waiting before submit fails immediately instead of timing out");
  kvaf::PutHandle empty;
  Require(!empty.submitted() && !reader.QueryCpuPull(empty,&state).ok() && !reader.WaitCpuPull(empty,0,&state).ok(),
          "empty put cannot query or wait on another transaction");
  Require(put.entries()[0].address==ticket.region.base+256&&put.entries()[0].reserved_length==64,"public Put uses MetaManager slots and effective length");
  kvaf::PutHandle second;Require(!reader.PreparePut({2},{16},&second).ok(),"one active transaction");
  Require(!reader.CommitPut(put).ok(),"cannot commit untransferred Put");
  // Three segments, two keys: reordered source and destination plus slot-internal split.
  std::vector<std::uint64_t> src{200,7,100},dst{put.entries()[1].address-ticket.region.base,put.entries()[0].address-ticket.region.base+32,put.entries()[0].address-ticket.region.base},sizes{32,32,32};
  const int reads=payload_reads;
  for(auto offset:{ticket.region.bytes-16,ticket.region.bytes,
                   UINT64_MAX,put.entries()[1].address}) {
    auto outside=dst;outside[0]=offset;
    Require(!reader.SubmitCpuPull(ready,put,src,outside,sizes).ok(),
            "DDR offsets reject overrun, overflow and old absolute addresses");
  }
  for(unsigned field=0;field<4;++field) {
    auto stale=ready;auto& identity=std::get<kvaf::CpuPullSourceDesc>(stale.source);
    if(field==0)++identity.source_id;
    if(field==1)++identity.source_bytes;
    if(field==2)++identity.region_id;
    if(field==3)++identity.region_epoch;
    Require(!reader.SubmitCpuPull(stale,put,src,dst,sizes).ok(),"Ready must match an attached CPU source");
  }
  auto invalid=dst;invalid[0]=0;
  Require(!reader.SubmitCpuPull(ready,put,src,invalid,sizes).ok(),"cannot target another slot in same pool");
  invalid=dst;invalid[1]=put.entries()[0].address-ticket.region.base+24;
  Require(!reader.SubmitCpuPull(ready,put,src,invalid,sizes).ok(),"overlapping or incomplete reservation rejected");
  auto short_sizes=sizes;short_sizes[1]=16;
  Require(!reader.SubmitCpuPull(ready,put,src,dst,short_sizes).ok(),"cannot commit unwritten holes");
  Require(!reader.SubmitCpuPull(ready,put,src,{dst.data(),2},sizes).ok(),"unequal arrays rejected");
  auto overflow=src;overflow[0]=UINT64_MAX;
  Require(!reader.SubmitCpuPull(ready,put,overflow,dst,sizes).ok(),"overflow rejected");
  Require(payload_reads==reads&&!put.submitted(),"invalid requests never create a remote task");
  {
    auto meta=kvaf::CreateInMemoryMetaManager();auto allocator=kvaf::CreateHostDdrMemoryAllocator();FakeAiv op;
    kvaf::RuntimeCore foreign(*meta,op,*allocator);Ok(foreign.Initialize({1024,256,kvaf::DeviceKind::kCpu}));
    kvaf::PutHandle foreign_put;Ok(foreign.PreparePut({1},{64},&foreign_put));
    Require(!reader.QueryCpuPull(foreign_put,&state).ok() && !reader.WaitCpuPull(foreign_put,0,&state).ok(),
            "foreign put cannot query or wait in this Runtime");
    Require(!reader.SubmitCpuPull(ready,foreign_put,src,dst,sizes).ok(),"foreign Runtime put rejected");
    Require(!foreign.CommitPut(put).ok()&&!foreign.AbortPut(put).ok(),"foreign Runtime cannot finish owned Put");
    Ok(foreign.AbortPut(foreign_put));Ok(foreign.Shutdown());
  }
  hold_read=true;server.drop=unsigned(rp::Command::Submit);
  auto status=reader.SubmitCpuPull(ready,put,src,dst,sizes);
  Require(status.code()==kvaf::StatusCode::kInDoubt&&put.submitted(),"lost Submit keeps request and recovery on the original put");
  Require(original_put.submitted() && original_put.state()==kvaf::CpuPullState::InDoubt,
          "the handle available before Submit retains the recorded request");
  Require(!reader.AbortPut(put).ok()&&!reader.CommitPut(put).ok(),"in-flight DMA forbids abort and commit");
  Require(!reader.core().BatchFree({{1,3}}).ok()&&!reader.Shutdown().ok(),"pending Put protects DDR and shutdown");
  Require(reader.WaitCpuPull(put,1,&state).code()==kvaf::StatusCode::kInDoubt,"timeout retains original put");
  hold_read=false;Ok(reader.WaitCpuPull(original_put,1000,&state));Require(state==kvaf::CpuPullState::Done,"wait stops at DMA completion");
  Require(put.submitted() && put.state()==state,"wait refreshes the shared put state");
  Require(!producer.ReleaseBatch(ready).ok(),"producer waits for metadata acknowledgement");
  Ok(reader.SubmitCpuPull(ready,put,src,dst,sizes));
  Require(dst[0]==put.entries()[1].address-ticket.region.base && dst[1]==put.entries()[0].address-ticket.region.base+32,
          "submission preserves application offset arrays");
  auto changed=src;changed[0]++;
  Require(!reader.SubmitCpuPull(ready,put,changed,dst,sizes).ok(),"conflicting retry rejected");
  auto changed_dst=dst;std::swap(changed_dst[1],changed_dst[2]);
  Require(!reader.SubmitCpuPull(ready,put,src,changed_dst,sizes).ok(),"duplicate destination changes rejected");
  Require(payload_reads==reads+1,"duplicate does not issue a second READ");
  Require(!std::memcmp(server.pool.data()+256,hbm.data()+100,32)&&!std::memcmp(server.pool.data()+288,hbm.data()+7,32)&&
          !std::memcmp(server.pool.data()+768,hbm.data()+200,32),"split/reordered CopyArrays copied correct data");
  server.drop=unsigned(rp::Command::Complete);
  Require(!reader.CommitPut(put).ok()&&reader.pending_offload(),"lost commit acknowledgement retains transaction");
  Require(!reader.AbortPut(put).ok(),"cannot reverse committed metadata");
  Ok(reader.CommitPut(put));Ok(reader.CommitPut(put));Ok(producer.ReleaseBatch(ready));
  Ok(reader.QueryCpuPull(put,&state));Require(state==kvaf::CpuPullState::Committed&&payload_reads==reads+1,"completed handle remains queryable without replay");
  kvaf::AccessTableHandle table;Ok(reader.core().PrepareAccessTable({{1,3}},&table));
  kvaf::AccessTableView view;Ok(reader.core().GetAccessTableView(table,&view));Require(view.valid[0]&&view.lengths[0]==64,"commit publishes effective lengths");
  Ok(reader.core().ReleaseAccessTable(table));Ok(reader.core().BatchFree({{1,3}}));
  // An old handle cannot act on a new reservation of the same DSA slots.
  const auto old=ready;Ok(producer.Ready(&ready));
  Require(!producer.ReleaseBatch(old).ok()&&!producer.CancelBatch(old).ok(),"stale release/cancel cannot affect new batch");
  Ok(reader.PreparePut({1},{64},&second));
  Ok(reader.CommitPut(put));Require(reader.pending_offload(),"old commit cannot finish a newer Put");
  Ok(reader.WaitCpuPull(put,0,&state));Require(state==kvaf::CpuPullState::Committed && !second.submitted(),
          "waiting on completed put cannot query or change a new reservation");
  Ok(reader.AbortPut(second));Ok(producer.CancelBatch(ready));
  Ok(reader.QueryCpuPull(second,&state));Require(state==kvaf::CpuPullState::Aborted && !second.submitted(),
          "unsubmitted abort remains queryable on the original put");
  Require(!reader.WaitCpuPull(second,0,&state).ok(),"aborted put is not a successful READ");
  // Completed DMA may be explicitly discarded without publishing its key.
  Ok(producer.Ready(&ready));Ok(reader.PreparePut({0},{32},&second));
  std::vector<std::uint64_t> one{0},destination{second.entries()[0].address-ticket.region.base},length{32};
  Ok(reader.SubmitCpuPull(ready,second,one,destination,length));
  Ok(reader.WaitCpuPull(second,1000,&state));Ok(reader.AbortPut(second));Ok(producer.ReleaseBatch(ready));
  // A recorded request may fail before CPU accepts it; submitted is not success.
  Ok(producer.Ready(&ready));Ok(reader.PreparePut({0},{32},&second));
  destination={second.entries()[0].address-ticket.region.base};Ok(producer.CancelBatch(ready));
  const int before_rejection=payload_reads;
  Require(!reader.SubmitCpuPull(ready,second,one,destination,length).ok() && second.submitted(),
          "pre-submit remote rejection leaves the recorded request on put");
  Ok(reader.QueryCpuPull(second,&state));
  Require(state==kvaf::CpuPullState::Failed && payload_reads==before_rejection,
          "submitted does not imply CPU acceptance or DMA completion");
  Require(!reader.WaitCpuPull(second,0,&state).ok(),"rejected request does not wait or report success");
  Ok(reader.AbortPut(second));
  // One Runtime chooses between attached producers using only Ready.source.
  std::vector<std::uint8_t> other_hbm(4096,0x77);
  kvaf::CpuPullHbmSource other(7,"00000000000000000000000000000001");
  Ok(other.Register(ticket,{other_hbm.data(),other_hbm.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)));
  kvaf::CpuPullSourceDesc other_desc;Ok(other.GetDescriptor(&other_desc));Ok(reader.AttachCpuSource(other_desc));
  for(auto* source:{&other,&producer}) {
    Ok(source->Ready(&ready));Ok(reader.PreparePut({0},{32},&second));destination={second.entries()[0].address-ticket.region.base};
    Ok(reader.SubmitCpuPull(ready,second,one,destination,length));Ok(reader.WaitCpuPull(second,1000,&state));
    Require(!std::memcmp(server.pool.data(),source==&other?other_hbm.data():hbm.data(),32),
            "Ready identity routes CPU Pull to the correct producer");
    Ok(reader.CommitPut(second));Ok(source->ReleaseBatch(ready));Ok(reader.core().BatchFree({{0}}));
  }
  Ok(other.Shutdown());
  Ok(producer.Shutdown());Ok(reader.Shutdown());NoResources();
}
void ConnectorQuarantine(bool batch){
  // An unresolved CQ must retain all resources. Isolate intentional retention.
  const auto child=fork();Require(child>=0,"fork quarantine test");
  if(child==0){try{
    SharedServer server;kvaf::NpuRuntimeOptions options;options.enable_cpu_offload=true;
    kvaf::NpuRuntime reader(options);Ok(reader.LinkTo({server.address}));Ok(reader.Initialize(1024,256));
    kvaf::OffloadRegionTicket ticket;Ok(reader.GetOffloadRegion(&ticket));
    std::vector<std::uint8_t> data(4096,7);aclrtSetDevice(7);
    std::vector<std::uint8_t> second(4096,0x9a);
    std::vector<kvaf::ConstBufferView> sources{{data.data(),data.size(),kvaf::DeviceKind::kNpu},
                                             {second.data(),second.size(),kvaf::DeviceKind::kNpu}};
    kvaf::CpuPullHbmSource producer(7,"00000000000000000000000000000001");
    Ok(producer.Register(ticket,sources,reinterpret_cast<void*>(1)));
    kvaf::CpuPullSourceDesc desc;Ok(producer.GetDescriptor(&desc));
    Ok(reader.AttachCpuSource(desc));kvaf::ReadyToken ready;Ok(producer.Ready(sources[1],&ready));
    kvaf::PutHandle put;std::vector<std::uint64_t> offsets{0},destinations,sizes{32};
    uncertain_read=true;
    if(batch) {
      Require(reader.BatchCpuPull({0},ready,offsets,sizes,1000).code()==kvaf::StatusCode::kInDoubt,"wrapper retains quarantined task");
    } else {
      Ok(reader.PreparePut({0},{32},&put));destinations={put.entries()[0].address-ticket.region.base};
      Ok(reader.SubmitCpuPull(ready,put,offsets,destinations,sizes));
      kvaf::CpuPullState state;
      Require(reader.WaitCpuPull(put,1000,&state).code()==kvaf::StatusCode::kInDoubt,"CQ uncertainty quarantines task");
    }
    const int reads=payload_reads;
    kvaf::ReadyToken other;
    Require(!producer.Ready(sources[0],&other).ok()&&registrations==2&&jetties==4,
            "uncertain batch retains all MRs and prevents switching source");
    if(batch) Require(!reader.ResolveCpuPull(0).ok(),"resolve cannot turn quarantine into completion");
    else Require(!reader.CommitPut(put).ok()&&!reader.AbortPut(put).ok(),"uncertain DMA forbids commit and abort");
    Require(!reader.Shutdown().ok(),"uncertain DMA retains DDR");
    Require(!producer.CancelBatch(ready).ok()&&!producer.ReleaseBatch(ready).ok()&&!producer.Shutdown().ok(),"uncertain DMA retains producer registration");
    if(batch) Require(!reader.BatchCpuPull({0},ready,offsets,sizes).ok(),"pending wrapper cannot resubmit");
    else Ok(reader.SubmitCpuPull(ready,put,offsets,destinations,sizes));
    Require(payload_reads==reads&&reader.pending_offload(),"quarantined duplicate never retries READ");
    _exit(0);
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';_exit(1);}}
  int status{};Require(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"Connector quarantine checks");
}
void MultipleMemorySource() {
  SharedServer server;kvaf::NpuRuntimeOptions options;options.enable_cpu_offload=true;
  kvaf::NpuRuntime reader(options);Ok(reader.LinkTo({server.address}));Ok(reader.Initialize(1024,256));
  kvaf::OffloadRegionTicket ticket;Ok(reader.GetOffloadRegion(&ticket));
  std::vector<std::vector<std::uint8_t>> tensors(78);
  std::vector<kvaf::ConstBufferView> views;
  for(std::size_t i=0;i<tensors.size();++i) {
    tensors[i].resize(512+i*64,std::uint8_t(i+1));
    views.push_back({tensors[i].data(),tensors[i].size(),kvaf::DeviceKind::kNpu});
  }
  aclrtSetDevice(7);kvaf::CpuPullHbmSource source(7,"00000000000000000000000000000001");
  Require(!source.Register(ticket,std::vector<kvaf::ConstBufferView>{},reinterpret_cast<void*>(1)).ok(),"empty MR list rejected");
  Require(!source.Register(ticket,std::vector<kvaf::ConstBufferView>{views[0],views[0]},reinterpret_cast<void*>(1)).ok(),"duplicate tensor rejected");
  fail_memory_at=37;
  Require(!source.Register(ticket,views,reinterpret_cast<void*>(1)).ok(),"partial HBM registration fails atomically");
  fail_memory_at=0;NoResources();
  // CPU may import an additional MR even when its reply is lost. The producer
  // keeps the entire exporter until Detach confirms retirement.
  server.drop=unsigned(rp::Command::AddMemory);
  Require(source.Register(ticket,views,reinterpret_cast<void*>(1)).code()==kvaf::StatusCode::kInDoubt,
          "lost AddMemory response retains partial registration");
  Require(registrations==78&&leases==1&&!source.registered(),"partial table is not publicly usable");
  Ok(source.Shutdown());NoResources();
  const auto probes=scratch_reads.load();
  Ok(source.Register(ticket,views,reinterpret_cast<void*>(1)));
  Require(registrations==78&&contexts==1&&sessions==1&&jetties==4&&peer_imports==4&&leases==1,
          "78 MRs share one exporter, backend and four Jettys");
  Require(scratch_reads==probes+78,"probe every MR before publishing source");
  Require(std::all_of(server.pool.begin(),server.pool.end(),[](auto x){return x==0xA5;}),"all probes preserve pool");
  kvaf::CpuPullSourceDesc desc;Ok(source.GetDescriptor(&desc));
  Require(desc.mr_count()==78&&desc.bytes_for(78)==tensors.back().size(),"descriptor retains individual sizes");
  auto wrong=desc;wrong.mr_bytes[50]++;Require(!reader.AttachCpuSource(wrong).ok(),"attach checks every MR");
  Ok(reader.AttachCpuSource(desc));
  kvaf::ReadyToken ready;
  Require(!source.Ready(&ready).ok(),"multi-MR ready must select tensor");
  auto unregistered=views[1];unregistered.bytes--;
  Require(!source.Ready(unregistered,&ready).ok(),"ready rejects unregistered views");
  const auto reads=payload_reads.load();
  for(std::size_t n=0;n<80;++n) {
    const auto index=n%78;
    std::fill(tensors[index].begin(),tensors[index].end(),std::uint8_t(n+11));
    if(n==0) {
      server.drop=unsigned(rp::Command::Ready);
      Require(!source.Ready(views[index],&ready).ok()&&source.frozen(),"lost Ready retains selected MR");
      kvaf::ReadyToken other;
      Require(!source.Ready(views[1],&other).ok(),"lost Ready cannot switch MR");
    }
    Ok(source.Ready(views[index],&ready));
    Require(ready.mr_id==index+1,"Ready selects tensor MR");
    auto stale=ready;stale.mr_id=index==0?2:1;
    Require(!source.ReleaseBatch(stale).ok()&&!source.CancelBatch(stale).ok(),"producer acknowledgement binds MR");
    auto forged=rp::MakeRequest(rp::Command::Ready);forged.region=ticket.region.id;forged.epoch=ticket.region.epoch;
    forged.token=ticket.region.token;forged.source_id=desc.source_id;forged.task=ready.sequence;forged.mr_id=stale.mr_id;
    rp::Response response;Ok(rp::Exchange(ticket.address,forged,{},&response));
    Require(response.result!=rp::Result::Ok,"server rejects same sequence with changed MR");
    if (n % 2) {
      std::vector<std::uint64_t> src{7,views[index].bytes-32},lengths{64,32};
      auto outside=src;outside[1]++;
      Require(!reader.BatchCpuPull({1,2},ready,outside,lengths).ok()&&!reader.pending_offload(),"wrapper bounds use selected tensor");
      Ok(reader.BatchCpuPull({1,2},ready,src,lengths,1000));
    } else {
      kvaf::PutHandle put;Ok(reader.PreparePut({1,2},{64,32},&put));
      std::vector<std::uint64_t> src{7,300},dst{put.entries()[0].address-ticket.region.base,put.entries()[1].address-ticket.region.base},lengths{64,32};
      auto outside=src;outside[0]=views[index].bytes-32;
      Require(!reader.SubmitCpuPull(ready,put,outside,dst,lengths).ok(),"bounds use selected tensor");
      if(n==0)server.drop=unsigned(rp::Command::Submit);
      auto submitted=reader.SubmitCpuPull(ready,put,src,dst,lengths);
      Require(submitted.ok()||(n==0&&submitted.code()==kvaf::StatusCode::kInDoubt),"submit or retain lost response");
      Require(!reader.SubmitCpuPull(stale,put,src,dst,lengths).ok(),"duplicate cannot change MR");
      Ok(reader.SubmitCpuPull(ready,put,src,dst,lengths));
      kvaf::CpuPullState state;Ok(reader.WaitCpuPull(put,1000,&state));Ok(reader.CommitPut(put));
    }
    Require(!std::memcmp(server.pool.data()+256,tensors[index].data()+7,64)&&
            !std::memcmp(server.pool.data()+512,tensors[index].data()+300,32),"batch reads only selected tensor");
    Ok(source.ReleaseBatch(ready));Ok(reader.core().BatchFree({{1,2}}));
  }
  Require(payload_reads==reads+80&&registrations==78&&contexts==1&&jetties==4&&scratch_reads==probes+78,
          "MR switching never registers, rebuilds links, probes or replays DMA");
  Ok(source.Shutdown());Ok(reader.Shutdown());NoResources();
}
int main(){try{Lifecycle();SharedMetadata();UnifiedBatchCpuPull();InitializationRecovery();MultipleMemorySource();ConnectorApi();ConnectorQuarantine(false);ConnectorQuarantine(true);std::cout<<"remote offload lifecycle passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
