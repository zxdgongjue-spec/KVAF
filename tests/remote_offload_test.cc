// Real NpuRuntime/Core/HBM exporter/control protocol; simulated ACL/HCCP and
// CPU DMA. Hardware permissions and cache visibility require the two-card test.
#include "offload_fake_transport.h"
#include "npu_runtime.h"
#include "remote_pull_service.h"
#include "remote_pull_operator.h"
#include "meta_manager.h"
#include <chrono>
#include <sys/wait.h>

namespace rp = kvaf::remote_pull;
using kvaf::Status;
void Ok(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message()); }
std::atomic<int> payload_reads{0}, scratch_reads{0}, checksum_calls{0}, leases{0};
std::atomic<bool> hold_read{false}, uncertain_read{false}, fail_checksum{false};
struct SharedBackend final : p::Backend {
  std::vector<std::uint8_t>& pool;
  p::Endpoint source{};
  bool closed{};
  explicit SharedBackend(std::vector<std::uint8_t>& v):pool(v){++leases;}
  Status Allocate(std::uint64_t,p::Endpoint*) override { throw std::runtime_error("B must not allocate DDR"); }
  Status Open(const p::Endpoint& s,p::Endpoint* cpu) override {
    source=s; cpu->key_bytes=28;cpu->tp=1;cpu->psn=2;
    for(unsigned i=0;i<4;++i){cpu->key[i][15]=4;cpu->key[i][20]=10+i;}
    return Status::OK();
  }
  Status ProbeToScratch(std::uint64_t offset,std::uint64_t bytes,std::uint64_t* hash) override {
    ++scratch_reads;
    if(source.eid[15]==failed_route) return Status::Internal("retired candidate probe");
    *hash=p::Hash(reinterpret_cast<const std::uint8_t*>(source.base+offset),bytes);
    return Status::OK();
  }
  Status Read(const std::vector<p::Span>& spans) override {
    ++payload_reads;
    while(hold_read)std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if(uncertain_read)return Status::InDoubt("simulated unresolved CQ");
    for(auto span:spans)std::memcpy(pool.data()+span.destination,reinterpret_cast<void*>(source.base+span.source),span.bytes);
    return Status::OK();
  }
  std::uint64_t Checksum(const std::vector<p::Span>& spans)const override {
    ++checksum_calls;
    if(fail_checksum)throw std::runtime_error("simulated diagnostic failure");
    std::uint64_t h=14695981039346656037ULL;
    for(auto span:spans)h=p::Hash(pool.data()+span.destination,span.bytes,h);return h;
  }
  Status ReleaseSource()override{return Status::OK();}
  Status Close()override{if(!closed){--leases;closed=true;}return Status::OK();}
};
class SharedServer {
 public:
  std::vector<std::uint8_t> pool;
  rp::RegionInfo info{};
  rp::Service service{[this](std::uint64_t id,rp::RegionInfo* out){
    if(id!=info.id||pool.empty())return Status::NotFound("region");*out=info;return Status::OK();
  },[this](const rp::RegionInfo&){return std::make_unique<SharedBackend>(pool);}};
  std::atomic<std::uint16_t> drop{0};
  std::atomic<int> pool_allocations{0};
  std::atomic<bool> stop{false}, fail_release{false};
  std::string address;
  int listener{-1};std::thread thread;
  template<class T> void Reply(int fd,kvaf::peer::Command cmd,T response){
    response.header=kvaf::peer::MakeHeader(cmd,sizeof(response));response.header.command|=0x8000;
    p::WriteBytes(fd,&response,sizeof(response));
  }
  template<class T> void ReadTail(int fd,T* request){
    Require(p::ReadBytes(fd,reinterpret_cast<char*>(request)+sizeof(request->header),sizeof(T)-sizeof(request->header)),"control tail");
  }
  void Peer(int fd,kvaf::peer::MessageHeader header){
    namespace peer=kvaf::peer;
    const auto cmd=static_cast<peer::Command>(header.command);
    if(cmd==peer::Command::kGetCandidateCount){peer::CandidateCountResponse r{};r.candidate_count=2;r.device_count=2;Reply(fd,cmd,r);return;}
    if(cmd==peer::Command::kAllocateRegion){
      peer::AllocateRegionRequest req{};req.header=header;ReadTail(fd,&req);
      Require(pool.empty(),"only A owns one DDR allocation");++pool_allocations;
      pool.resize(req.ddr_bytes,0xA5);info={42,1,reinterpret_cast<std::uint64_t>(pool.data()),req.ddr_bytes,req.slot_size,req.candidate_id,kRegionToken,0,{}};info.eid[15]=2;
      peer::AllocateRegionResponse r{};r.session_id=1;r.peer_base=info.base;r.peer_length=info.bytes;r.token_id=1;r.token_value=kRegionToken;r.segment_key_bytes=48;r.peer_eid[15]=2;
      Reply(fd,cmd,r);return;
    }
    if(cmd==peer::Command::kReleaseRegion){
      peer::ReleaseRegionRequest req{};req.header=header;ReadTail(fd,&req);
      peer::StatusResponse r{};
      if(fail_release.exchange(false)||service.RegionBusy(req.session_id)||leases)r.result=int(peer::Result::kResourceError);
      else pool.clear();Reply(fd,cmd,r);return;
    }
    throw std::runtime_error("unexpected KVCP command");
  }
  SharedServer(){
    listener=socket(AF_INET,SOCK_STREAM,0);Require(listener>=0,"socket");
    sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    Require(!bind(listener,reinterpret_cast<sockaddr*>(&addr),sizeof(addr)),"bind loopback");
    socklen_t size=sizeof(addr);Require(!getsockname(listener,reinterpret_cast<sockaddr*>(&addr),&size),"getsockname");
    address="127.0.0.1:"+std::to_string(ntohs(addr.sin_port));Require(!listen(listener,16),"listen");
    thread=std::thread([this]{while(!stop){
      const int fd=accept(listener,nullptr,nullptr);if(fd<0)break;p::SetSocketTimeout(fd);
      rp::Request req{};
      if(!p::ReadBytes(fd,&req.header,sizeof(req.header))){close(fd);continue;}
      if(req.header.magic==kvaf::peer::kProtocolMagic)Peer(fd,req.header);
      else{
        ReadTail(fd,&req);Require(req.count<=rp::kMaxSpans,"bounded spans");std::vector<rp::Span> spans(req.count);
        Require(p::ReadBytes(fd,spans.data(),spans.size()*sizeof(rp::Span)),"read spans");
        const auto response=service.Dispatch(req,spans);auto expected=req.header.command;
        if(!drop.compare_exchange_strong(expected,0))p::WriteBytes(fd,&response,sizeof(response));
      }
      close(fd);
    }});
  }
  ~SharedServer(){stop=true;shutdown(listener,SHUT_RDWR);close(listener);if(thread.joinable())thread.join();}
};
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
  kvaf::CpuPullSourceDesc desc;Ok(b.GetDescriptor(&desc));kvaf::CpuPullSourceHandle source;Ok(a.AttachCpuSource(desc,&source));
  kvaf::ReadyToken ready;Ok(b.Ready(&ready));
  Require(!b.ReleaseBatch(ready).ok()&&!a.Shutdown().ok(),"source lease blocks premature release");
  kvaf::PutHandle put;Ok(a.PreparePut({1,3},{64,128},&put));
  std::vector<std::uint64_t> src{16,512},dst{256,768},sizes{64,128};kvaf::CpuPullTask task;
  const int before=payload_reads, hashes=checksum_calls;
  Ok(a.SubmitCpuPull(source,ready,put,src,dst,sizes,&task));kvaf::CpuPullState state;
  Ok(a.WaitCpuPull(task,1000,&state));Ok(a.CommitPut(put));
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
  kvaf::CpuPullSourceHandle source;Ok(core.AttachCpuSource(std::get<kvaf::CpuPullSourceDesc>(ready.source),&source));
  kvaf::PutHandle put;Ok(core.PreparePut({1},{64},&put));kvaf::CpuPullTask task;
  std::vector<std::uint64_t> src{64},dst{256},sizes{64};
  Ok(core.SubmitCpuPull(source,ready,put,src,dst,sizes,&task));kvaf::CpuPullState state;
  Ok(core.WaitCpuPull(task,1000,&state));Ok(core.CommitPut(put));
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
void LocalAndFallback(){
  EidFixture fixture;
  const std::string eid1="00000000000000000000000000000001",eid5="00000000000000000000000000000005";
  EidFixture::Write(fixture.paths.rootinfo,EidFixture::Clos(0,eid1)+EidFixture::Clos(0,eid5));
  const char* prior=getenv("KVAF_HCCL_ROOTINFO_PATH");const bool had_prior=prior;const std::string saved=prior?prior:"";
  setenv("KVAF_HCCL_ROOTINFO_PATH",fixture.paths.rootinfo.c_str(),1);auto_mode=true;failed_route=1;
  SharedServer server;kvaf::NpuRuntimeOptions options;options.enable_cpu_offload=true;options.offload_timeout_ms=2;
  kvaf::NpuRuntime a(options);Ok(a.LinkTo({server.address}));Ok(a.Initialize(1024,256));
  std::vector<std::uint8_t> hbm(4096,0x43);kvaf::BatchPutRequest req{{2},{hbm.data(),hbm.size(),kvaf::DeviceKind::kNpu},{32},{64}};
  Ok(a.BatchCpuPull(req,{reinterpret_cast<void*>(1)}));
  Require(server.pool_allocations==1&&leases==1&&producer_syncs==1,"local Pull shares pool, synchronizes producer and caches source");
  Require(!std::memcmp(server.pool.data()+512,hbm.data()+32,64),"local Pull payload");
  kvaf::AccessTableHandle table;Ok(a.core().PrepareAccessTable({{2}},&table));kvaf::AccessTableView view;Ok(a.core().GetAccessTableView(table,&view));
  Require(view.valid[0],"synchronous success includes metadata commit");Ok(a.core().ReleaseAccessTable(table));
  const int reads=payload_reads,probes=scratch_reads;
  Require(!a.BatchCpuPull(req,{reinterpret_cast<void*>(1)}).ok()&&!a.pending_offload()&&payload_reads==reads,"duplicate occupied key fails before transfer");
  Ok(a.core().BatchFree({{2}}));hbm[32]=0x88;Ok(a.BatchCpuPull(req,{reinterpret_cast<void*>(1)}));
  Require(server.pool[512]==0x88&&payload_reads==reads+1&&scratch_reads==probes,"each call is a new batch and reuses registration");
  Ok(a.core().BatchFree({{2}}));
  server.drop=unsigned(rp::Command::Ready);
  Require(!a.BatchCpuPull(req,{reinterpret_cast<void*>(1)}).ok()&&!a.pending_offload()&&payload_reads==reads+1,"lost Ready aborts safely before READ");
  // Metadata commit has happened; retry only its acknowledgement on shutdown.
  server.drop=unsigned(rp::Command::Complete);
  Require(!a.BatchCpuPull(req,{reinterpret_cast<void*>(1)}).ok()&&a.pending_offload(),"lost commit ACK retains the decision");
  const int committed=payload_reads;Ok(a.Shutdown());Require(payload_reads==committed,"shutdown never replays READ");NoResources();
  Ok(a.Initialize(1024,256));
  hold_read=true;server.drop=unsigned(rp::Command::Submit);
  Require(!a.BatchCpuPull(req,{reinterpret_cast<void*>(1)}).ok()&&a.pending_offload(),"unresolved READ retains source and slots");
  Require(!a.BatchCpuPull(req,{reinterpret_cast<void*>(1)}).ok(),"repeated calls do not implicitly resume a pending operation");
  Require(!a.Shutdown().ok()&&!a.core().BatchFree({{2}}).ok(),"in-flight DMA blocks cleanup and slot reuse");
  hold_read=false;
  Status close;
  for(int i=0;i<1000;++i){close=a.Shutdown();if(close.ok())break;std::this_thread::sleep_for(std::chrono::milliseconds(1));}
  Ok(close);NoResources();
  // A different input replaces the cached registration after the old one retires.
  Ok(a.Initialize(1024,256));Ok(a.BatchCpuPull(req,{reinterpret_cast<void*>(1)}));Ok(a.core().BatchFree({{2}}));
  std::vector<std::uint8_t> replacement(8192,0x94);req.input={replacement.data(),replacement.size(),kvaf::DeviceKind::kNpu};
  Ok(a.BatchCpuPull(req,{reinterpret_cast<void*>(1)}));Require(server.pool[512]==0x94&&leases==1,"source replacement retires previous registration");
  Ok(a.Shutdown());NoResources();auto_mode=false;failed_route=0;
  if(had_prior)setenv("KVAF_HCCL_ROOTINFO_PATH",saved.c_str(),1);else unsetenv("KVAF_HCCL_ROOTINFO_PATH");
}
void InitializationRecovery() {
  SharedServer server;kvaf::NpuRuntimeOptions options;options.enable_cpu_offload=true;
  kvaf::NpuRuntime a(options);Ok(a.LinkTo({server.address}));
  server.drop=std::uint16_t(rp::Command::Describe);server.fail_release=true;
  Require(!a.Initialize(1024,256).ok()&&!a.initialized(),"lost description and failed cleanup preserve failed initialization");
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
  kvaf::CpuPullSourceHandle source;Ok(reader.AttachCpuSource(desc,&source));
  auto bad_desc=desc;bad_desc.region_epoch++;kvaf::CpuPullSourceHandle rejected;
  Require(!reader.AttachCpuSource(bad_desc,&rejected).ok(),"stale source descriptor rejected");
  kvaf::ReadyToken ready;
  server.drop=unsigned(rp::Command::Ready);
  Require(!producer.Ready(&ready).ok()&&producer.frozen(),"lost identity-only Ready keeps batch frozen");
  const auto sequence=ready.sequence;Ok(producer.Ready(&ready));Require(ready.sequence==sequence,"Ready retry reuses sequence");
  kvaf::PutHandle put;Ok(reader.PreparePut({1,3},{64,32},&put));
  Require(put.entries()[0].address==ticket.region.base+256&&put.entries()[0].reserved_length==64,"public Put uses MetaManager slots and effective length");
  kvaf::PutHandle second;Require(!reader.PreparePut({2},{16},&second).ok(),"one active transaction");
  Require(!reader.CommitPut(put).ok(),"cannot commit untransferred Put");
  // Three segments, two keys: reordered source and destination plus slot-internal split.
  std::vector<std::uint64_t> src{200,7,100},dst{768,288,256},sizes{32,32,32};
  kvaf::CpuPullTask task;const int reads=payload_reads;
  auto invalid=dst;invalid[0]=0;
  Require(!reader.SubmitCpuPull(source,ready,put,src,invalid,sizes,&task).ok(),"cannot target another slot in same pool");
  invalid=dst;invalid[1]=280;
  Require(!reader.SubmitCpuPull(source,ready,put,src,invalid,sizes,&task).ok(),"overlapping or incomplete reservation rejected");
  auto short_sizes=sizes;short_sizes[1]=16;
  Require(!reader.SubmitCpuPull(source,ready,put,src,dst,short_sizes,&task).ok(),"cannot commit unwritten holes");
  Require(!reader.SubmitCpuPull(source,ready,put,src,{dst.data(),2},sizes,&task).ok(),"unequal arrays rejected");
  auto overflow=src;overflow[0]=UINT64_MAX;
  Require(!reader.SubmitCpuPull(source,ready,put,overflow,dst,sizes,&task).ok(),"overflow rejected");
  Require(payload_reads==reads&&!put.task().valid(),"invalid requests never create a remote task");
  {
    auto meta=kvaf::CreateInMemoryMetaManager();auto allocator=kvaf::CreateHostDdrMemoryAllocator();FakeAiv op;
    kvaf::RuntimeCore foreign(*meta,op,*allocator);Ok(foreign.Initialize({1024,256,kvaf::DeviceKind::kCpu}));
    kvaf::PutHandle foreign_put;Ok(foreign.PreparePut({1},{64},&foreign_put));
    Require(!reader.SubmitCpuPull(source,ready,foreign_put,src,dst,sizes,&task).ok(),"foreign Runtime put rejected");
    Require(!foreign.CommitPut(put).ok()&&!foreign.AbortPut(put).ok(),"foreign Runtime cannot finish owned Put");
    Ok(foreign.AbortPut(foreign_put));Ok(foreign.Shutdown());
  }
  hold_read=true;server.drop=unsigned(rp::Command::Submit);
  auto status=reader.SubmitCpuPull(source,ready,put,src,dst,sizes,&task);
  Require(status.code()==kvaf::StatusCode::kInDoubt&&put.task().valid(),"lost Submit exposes original task on put");
  task=put.task();kvaf::CpuPullState state;
  Require(!reader.AbortPut(put).ok()&&!reader.CommitPut(put).ok(),"in-flight DMA forbids abort and commit");
  Require(!reader.core().BatchFree({{1,3}}).ok()&&!reader.Shutdown().ok(),"pending Put protects DDR and shutdown");
  Require(reader.WaitCpuPull(task,1,&state).code()==kvaf::StatusCode::kInDoubt,"timeout retains original task");
  hold_read=false;Ok(reader.WaitCpuPull(task,1000,&state));Require(state==kvaf::CpuPullState::Done,"wait stops at DMA completion");
  Require(!producer.ReleaseBatch(ready).ok(),"producer waits for metadata acknowledgement");
  Ok(reader.SubmitCpuPull(source,ready,put,src,dst,sizes,&task));
  auto changed=src;changed[0]++;
  Require(!reader.SubmitCpuPull(source,ready,put,changed,dst,sizes,&task).ok(),"conflicting retry rejected");
  Require(payload_reads==reads+1,"duplicate does not issue a second READ");
  Require(!std::memcmp(server.pool.data()+256,hbm.data()+100,32)&&!std::memcmp(server.pool.data()+288,hbm.data()+7,32)&&
          !std::memcmp(server.pool.data()+768,hbm.data()+200,32),"split/reordered CopyArrays copied correct data");
  server.drop=unsigned(rp::Command::Complete);
  Require(!reader.CommitPut(put).ok()&&reader.pending_offload(),"lost commit acknowledgement retains transaction");
  Require(!reader.AbortPut(put).ok(),"cannot reverse committed metadata");
  Ok(reader.CommitPut(put));Ok(reader.CommitPut(put));Ok(producer.ReleaseBatch(ready));
  Ok(reader.QueryCpuPull(task,&state));Require(state==kvaf::CpuPullState::Committed&&payload_reads==reads+1,"completed handle remains queryable without replay");
  kvaf::AccessTableHandle table;Ok(reader.core().PrepareAccessTable({{1,3}},&table));
  kvaf::AccessTableView view;Ok(reader.core().GetAccessTableView(table,&view));Require(view.valid[0]&&view.lengths[0]==64,"commit publishes effective lengths");
  Ok(reader.core().ReleaseAccessTable(table));Ok(reader.core().BatchFree({{1,3}}));
  // An old handle cannot act on a new reservation of the same DSA slots.
  const auto old=ready;Ok(producer.Ready(&ready));
  Require(!producer.ReleaseBatch(old).ok()&&!producer.CancelBatch(old).ok(),"stale release/cancel cannot affect new batch");
  Ok(reader.PreparePut({1},{64},&second));
  Ok(reader.CommitPut(put));Require(reader.pending_offload(),"old commit cannot finish a newer Put");
  Ok(reader.AbortPut(second));Ok(producer.CancelBatch(ready));
  // Completed DMA may be explicitly discarded without publishing its key.
  Ok(producer.Ready(&ready));Ok(reader.PreparePut({0},{32},&second));
  std::vector<std::uint64_t> one{0},length{32};
  Ok(reader.SubmitCpuPull(source,ready,second,one,one,length,&task));
  Ok(reader.WaitCpuPull(task,1000,&state));Ok(reader.AbortPut(second));Ok(producer.ReleaseBatch(ready));
  Ok(producer.Shutdown());Ok(reader.Shutdown());NoResources();
}
void ConnectorQuarantine(){
  // An unresolved CQ must retain all resources. Isolate intentional retention.
  const auto child=fork();Require(child>=0,"fork quarantine test");
  if(child==0){try{
    SharedServer server;kvaf::NpuRuntimeOptions options;options.enable_cpu_offload=true;
    kvaf::NpuRuntime reader(options);Ok(reader.LinkTo({server.address}));Ok(reader.Initialize(1024,256));
    kvaf::OffloadRegionTicket ticket;Ok(reader.GetOffloadRegion(&ticket));
    std::vector<std::uint8_t> data(4096,7);aclrtSetDevice(7);
    kvaf::CpuPullHbmSource producer(7,"00000000000000000000000000000001");
    Ok(producer.Register(ticket,{data.data(),data.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)));
    kvaf::CpuPullSourceDesc desc;Ok(producer.GetDescriptor(&desc));kvaf::CpuPullSourceHandle source;
    Ok(reader.AttachCpuSource(desc,&source));kvaf::ReadyToken ready;Ok(producer.Ready(&ready));
    kvaf::PutHandle put;Ok(reader.PreparePut({0},{32},&put));std::vector<std::uint64_t> offsets{0},sizes{32};
    uncertain_read=true;kvaf::CpuPullTask task;Ok(reader.SubmitCpuPull(source,ready,put,offsets,offsets,sizes,&task));
    kvaf::CpuPullState state;
    Require(reader.WaitCpuPull(task,1000,&state).code()==kvaf::StatusCode::kInDoubt,"CQ uncertainty quarantines task");
    const int reads=payload_reads;
    Require(!reader.CommitPut(put).ok()&&!reader.AbortPut(put).ok()&&!reader.Shutdown().ok(),"uncertain DMA retains DDR");
    Require(!producer.CancelBatch(ready).ok()&&!producer.ReleaseBatch(ready).ok()&&!producer.Shutdown().ok(),"uncertain DMA retains producer registration");
    Ok(reader.SubmitCpuPull(source,ready,put,offsets,offsets,sizes,&task));
    Require(payload_reads==reads&&reader.pending_offload(),"quarantined duplicate never retries READ");
    _exit(0);
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';_exit(1);}}
  int status{};Require(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"Connector quarantine checks");
}
int main(){try{Lifecycle();SharedMetadata();LocalAndFallback();InitializationRecovery();ConnectorApi();ConnectorQuarantine();std::cout<<"remote offload lifecycle passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
