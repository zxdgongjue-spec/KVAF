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
std::atomic<int> payload_reads{0}, scratch_reads{0}, leases{0};
std::atomic<bool> hold_read{false}, uncertain_read{false};
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
  kvaf::NpuRuntimeOptions options;options.device_id=0;options.enable_cpu_offload=true;options.offload_timeout_ms=20;options.cpu_candidate=1;
  kvaf::NpuRuntime a(options);Ok(a.LinkTo({server.address}));Ok(a.Initialize(1024,256));
  kvaf::OffloadRegionTicket ticket;Ok(a.GetOffloadRegion(&ticket));Require(ticket.region.candidate==1,"pinned shared pool");
  std::vector<std::uint8_t> hbm(4096);for(unsigned i=0;i<hbm.size();++i)hbm[i]=i*3+7;
  aclrtSetDevice(7);kvaf::HbmSource b(7,"00000000000000000000000000000001");
  Ok(b.Register(ticket,{hbm.data(),hbm.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)));
  Require(server.pool_allocations==1&&leases==1&&scratch_reads==1,"source borrows A pool and probes scratch");
  Require(std::all_of(server.pool.begin(),server.pool.end(),[](auto c){return c==0xA5;}),"probe cannot overwrite shared slot zero");
  kvaf::HbmReadyBatch batch;Ok(b.Ready({16,512},{64,128},reinterpret_cast<void*>(1),&batch));
  Require(!b.ReleaseBatch().ok()&&b.frozen(),"B retained before A submits");
  Require(!a.Shutdown().ok()&&a.initialized(),"A cannot release shared DDR while B registered");
  aclrtSetDevice(0);const int before=payload_reads;Ok(a.OffloadFrom({1,3},batch));
  Require(payload_reads==before+1&&server.pool_allocations==1,"one direct CPU READ batch, one pool");
  Require(!std::memcmp(server.pool.data()+256,hbm.data()+16,64)&&!std::memcmp(server.pool.data()+768,hbm.data()+512,128),"CPU uses B offsets and PreparePut DDR destinations");
  std::uint64_t hash{};Ok(a.LastOffloadChecksum(&hash));Require(hash==p::Hash(hbm.data()+512,128,p::Hash(hbm.data()+16,64)),"shared DDR contents hash");
  Ok(a.OffloadFrom({1,3},batch));Require(payload_reads==before+1,"duplicate Ready notification does not redo PreparePut/READ");
  auto conflict=batch;conflict.input_offsets[0]++;Require(!a.OffloadFrom({1,3},conflict).ok(),"conflicting duplicate rejected");
  kvaf::AccessTableHandle table;Ok(a.core().PrepareAccessTable({{1,3}},&table));Ok(a.core().ReleaseAccessTable(table));
  Ok(a.core().BatchFree({{1,3}}));
  aclrtSetDevice(7);Ok(b.ReleaseBatch());Require(!b.frozen(),"only committed batch can be reused");
  Ok(b.Ready({32},{32},reinterpret_cast<void*>(1),&batch));
  auto stale=batch;stale.source.region_epoch++;Require(!a.OffloadFrom({2},stale).ok(),"old service epoch rejected before PreparePut");
  server.drop=std::uint16_t(rp::Command::Submit);
  aclrtSetDevice(0);Require(a.OffloadFrom({2},batch).code()==kvaf::StatusCode::kInDoubt&&a.pending_offload(),"lost Submit ACK retains task");
  const int submitted=payload_reads;
  Require(!a.core().BatchFree({{2}}).ok()&&!a.core().PrepareAccessTable({{2}},&table).ok(),"pending slots hidden/protected");
  Ok(a.ResolveOffload());Require(payload_reads==submitted&&!a.pending_offload(),"Resolve queries, never resubmits READ");
  aclrtSetDevice(7);Ok(b.ReleaseBatch());Ok(a.core().BatchFree({{2}}));
  Ok(b.Ready({0},{16},reinterpret_cast<void*>(1),&batch));
  server.drop=std::uint16_t(rp::Command::Complete);
  aclrtSetDevice(0);Require(a.OffloadFrom({0},batch).code()==kvaf::StatusCode::kInDoubt&&a.pending_offload(),"lost metadata ACK keeps pending state");
  const int complete_reads=payload_reads;Ok(a.ResolveOffload());Require(payload_reads==complete_reads,"commit ACK retry does not copy again");
  aclrtSetDevice(7);Ok(b.ReleaseBatch());Ok(a.core().BatchFree({{0}}));
  // Cancel before submission is safe; a stale A notification cannot read B again.
  Ok(b.Ready({0},{16},reinterpret_cast<void*>(1),&batch));Ok(b.CancelBatch());
  Require(!a.OffloadFrom({0},batch).ok(),"cancelled source not submitted");
  // Timeout remains recoverable while the same worker owns the READ.
  Ok(b.Ready({0},{16},reinterpret_cast<void*>(1),&batch));hold_read=true;
  aclrtSetDevice(0);Require(a.OffloadFrom({0},batch).code()==kvaf::StatusCode::kInDoubt,"READ deadline retains operation");
  Require(!b.CancelBatch().ok()&&!b.Shutdown().ok(),"B cannot unregister or cancel an active READ");
  hold_read=false;Ok(a.ResolveOffload());aclrtSetDevice(7);Ok(b.ReleaseBatch());
  server.drop=std::uint16_t(rp::Command::Detach);Require(!b.Shutdown().ok(),"lost Detach reply retains exporter");
  Ok(b.Shutdown());NoResources();aclrtSetDevice(0);server.fail_release=true;
  Require(!a.Shutdown().ok(),"failed DDR release retains runtime for shutdown retry");
  Require(!a.core().PrepareAccessTable({{0}},&table).ok(),"partially closed runtime cannot resume data operations");
  Ok(a.Shutdown());Require(server.pool.empty(),"shared allocation released after detach");
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
  PoolAllocator allocator(link);FakeAiv aiv;kvaf::RemotePullOperator op(&aiv,link,100);
  auto meta=kvaf::CreateInMemoryMetaManager();kvaf::RuntimeCore core(*meta,op,allocator);Ok(core.Initialize({1024,256,kvaf::DeviceKind::kCpu}));
  std::vector<std::uint8_t> a_data(64,0x23),b_data(1024,0x67);
  Ok(core.BatchPut({{0},{a_data.data(),a_data.size(),kvaf::DeviceKind::kNpu},{0},{64}}));
  aclrtSetDevice(7);kvaf::HbmSource b(7,"00000000000000000000000000000001");
  Ok(b.Register(op.ticket(),{b_data.data(),b_data.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)));
  Require(!std::memcmp(server.pool.data(),a_data.data(),64),"B registration probe preserves A BatchPut");
  kvaf::HbmReadyBatch batch;Ok(b.Ready({64},{64},reinterpret_cast<void*>(1),&batch));
  kvaf::RemoteOffloadRequest request{batch.source,{1},batch.input_offsets,batch.lengths};bool duplicate=false;
  Ok(op.Validate(request,&duplicate));Ok(core.OffloadFrom(request));
  Require(aiv.puts==1&&!std::memcmp(server.pool.data()+256,b_data.data()+64,64),"Pull bypasses AIV and shares DDR");
  const int copied=payload_reads;
  Require(!core.BatchPut({{1},{a_data.data(),a_data.size(),kvaf::DeviceKind::kNpu},{0},{64}}).ok(),"same MetaManager prevents duplicate key across data planes");
  Require(aiv.puts==1&&payload_reads==copied,"conflicting key never reaches either data plane");
  kvaf::AccessTableHandle table;Ok(core.PrepareAccessTable({{0,1}},&table));
  kvaf::AccessTableView view;Ok(core.GetAccessTableView(table,&view));
  Require(view.valid[0]&&view.valid[1]&&view.addresses[0]==server.info.base&&view.addresses[1]==server.info.base+256,"one access table sees both writes");
  Require(!core.Shutdown().ok(),"busy B blocks shutdown before unlocking tables");Ok(core.GetAccessTableView(table,&view));
  Ok(core.ReleaseAccessTable(table));Ok(b.ReleaseBatch());Ok(b.Shutdown());Ok(core.Shutdown());NoResources();
}
void LocalAndFallback(){
  EidFixture fixture;
  const std::string eid1="00000000000000000000000000000001",eid5="00000000000000000000000000000005";
  EidFixture::Write(fixture.paths.rootinfo,EidFixture::Clos(0,eid1)+EidFixture::Clos(0,eid5));
  const char* prior=getenv("KVAF_HCCL_ROOTINFO_PATH");const bool had_prior=prior;const std::string saved=prior?prior:"";
  setenv("KVAF_HCCL_ROOTINFO_PATH",fixture.paths.rootinfo.c_str(),1);auto_mode=true;failed_route=1;
  SharedServer server;kvaf::NpuRuntimeOptions options;options.enable_cpu_offload=true;
  kvaf::NpuRuntime a(options);Ok(a.LinkTo({server.address}));Ok(a.Initialize(1024,256));
  std::vector<std::uint8_t> hbm(4096,0x43);kvaf::BatchPutRequest req{{2},{hbm.data(),hbm.size(),kvaf::DeviceKind::kNpu},{32},{64}};
  Ok(a.Offload(req,{reinterpret_cast<void*>(1)}));
  Require(server.pool_allocations==1&&leases==1,"local offload and candidate fallback keep A pool");
  Require(!std::memcmp(server.pool.data()+512,hbm.data()+32,64),"local source offload uses same DDR");
  Ok(a.core().BatchFree({{2}}));const int probes=scratch_reads;
  server.drop=std::uint16_t(rp::Command::Submit);
  Require(!a.Offload(req,{reinterpret_cast<void*>(1)}).ok()&&a.pending_offload(),"local lost Submit retains source");
  Require(!a.ReleaseOffloadSource().ok(),"cannot unregister pending local source");Ok(a.ResolveOffload());
  Require(!a.pending_offload()&&scratch_reads==probes,"local registration reused across batches");
  Ok(a.ReleaseOffloadSource());Ok(a.Shutdown());NoResources();
  auto_mode=false;failed_route=0;
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
int main(){try{Lifecycle();SharedMetadata();LocalAndFallback();InitializationRecovery();std::cout<<"remote offload lifecycle passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
