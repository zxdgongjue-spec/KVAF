// Executes the real CANN-host runtime and TCP protocol against fake ACL/HCCP
// and a host-memory backend. This tests orchestration, not hardware support.
#include "offload_runtime.h"
#include "npu_eid_fixture.h"
#include <mutex>
#include "offload_service.h"
#include "offload_client.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include <acl/acl.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace p = kvaf::offload;
constexpr std::uint32_t kRegionToken = 0x13579bdf;
std::atomic<int> registrations{0}, events{0}, event_waits{0}, reads{0};
std::atomic<int> sessions{0}, contexts{0}, probe_allocations{0};
std::atomic<int> cqs{0}, jetties{0}, peer_imports{0};
int context_error = 0;
int fail_jetty_at = 0, fail_peer_at = 0;
bool null_context = false;
bool auto_mode = false;
std::atomic<int> failed_route{0}, failed_cpu{-1}, corrupt_route{0};
std::atomic<int> pools{0}, allocations{0}, candidate_queries{0};
std::mutex attempts_mutex;
std::vector<std::pair<unsigned, unsigned>> attempted_pairs;
bool Route(unsigned eid) { return eid == 1 || (auto_mode && eid == 5); }
enum class BadSegment { None, ZeroBase, Short, Overflow, Abi };
BadSegment bad_segment = BadSegment::None;
void Require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
extern "C" {
aclError aclrtGetDevice(int32_t* d){*d=0;return 0;}
aclError aclrtMalloc(void** p,std::size_t n,int){*p=std::malloc(n);if(*p)++probe_allocations;return *p?0:1;}
aclError aclrtFree(void* p){if(p)--probe_allocations;std::free(p);return 0;}
aclError aclrtMemset(void* p,std::size_t,int value,std::size_t n){std::memset(p,value,n);return 0;}
aclError aclrtCreateStream(aclrtStream* p){*p=new int;return 0;}
aclError aclrtDestroyStream(aclrtStream p){delete static_cast<int*>(p);return 0;}
aclError aclrtCreateEvent(aclrtEvent* p){++events;*p=new int;return 0;}
aclError aclrtRecordEvent(aclrtEvent,aclrtStream){return 0;}
aclError aclrtSynchronizeEvent(aclrtEvent){++event_waits;return 0;}
aclError aclrtDestroyEvent(aclrtEvent p){--events;delete static_cast<int*>(p);return 0;}
}
namespace kvaf::direct_hccp {
struct DirectHccpSession {};
bool AcquireDirectHccpSession(std::uint32_t,std::uint32_t,DirectHccpSession** s){++sessions;*s=new DirectHccpSession;return true;}
void ReleaseDirectHccpSession(DirectHccpSession* s){--sessions;delete s;}
bool GetCurrentNpuPhysicalDeviceId(std::uint32_t* d){*d=0;return true;}
int Destroy(void* p){delete static_cast<int*>(p);return 0;}
int DestroyWithContext(void*,void* p){return Destroy(p);}
bool LoadDirectHccp(DirectHccpApi* a){
  a->ctx_init=[](CtxInitCfg*,CtxInitAttr* attr,void** p){
    Require(Route(attr->ub.eid.raw[15]),"context uses configured route EID");
    if(context_error || null_context){*p=nullptr;return context_error;}
    ++contexts;*p=new int;return 0;
  };
  a->ctx_deinit=[](void* p){--contexts;return Destroy(p);};
  a->token_alloc=[](void*,TokenId*,void** p){*p=new int;return 0;};
  a->token_free=DestroyWithContext;
  a->lmem_register=[](void*,MrRegInfo* mr,void** p){
    Require(mr->in.ub.token_value==kRegionToken,"HBM registration uses CPU region token");
    Require(mr->in.ub.flags==0x29c1,"reference HBM registration flags");
    ++registrations;*p=new int;mr->out.key.size=48;
    // Reproduce the hardware case: route EID 1, exported memory EID 3.
    std::memset(mr->out.key.value,0,48);mr->out.key.value[15]=3;
    auto base=mr->in.mem.addr,bytes=mr->in.mem.size;
    if(bad_segment==BadSegment::ZeroBase)base=0;
    if(bad_segment==BadSegment::Short)bytes-=1;
    if(bad_segment==BadSegment::Overflow)base=UINT64_MAX;
    if(bad_segment==BadSegment::Abi)mr->out.key.size=52;
    std::memcpy(mr->out.key.value+20,&base,8);
    std::memcpy(mr->out.key.value+32,&bytes,8);return 0;
  };
  a->lmem_unregister=[](void*,void* p){--registrations;return Destroy(p);};
  a->cq_create=[](void*,CqInfo* info,void** p){
    Require(info->in.depth==16384&&info->in.ub.mode==kJfcNormal,"reference NPU CQ");
    ++cqs;*p=new int;return 0;
  };
  a->cq_destroy=[](void*,void* p){--cqs;return Destroy(p);};
  a->qp_create=[](void*,QpCreateAttr* attr,QpCreateInfo* info,void** p){
    Require(attr->ub.token_value==kRegionToken,"NPU Jetty uses CPU region token");
    Require(attr->sq_depth==1024&&attr->rq_depth==1024&&attr->ub.priority==2&&
            attr->ub.rnr_retry==7&&attr->ub.err_timeout==16&&attr->ub.jfs_flag==JfsFlagErrorSuspend(),
            "reference NPU queue configuration");
    info->key.size=32;info->key.value[15]=3;info->key.value[20]=++jetties;*p=new int;
    return jetties==fail_jetty_at?-51:0;
  };
  a->qp_destroy=[](void* p){--jetties;return Destroy(p);};
  a->qp_unimport=[](void*,void* p){--peer_imports;return Destroy(p);};
  a->get_tp_info_async=[](void*,GetTpCfg* cfg,TpInfo* info,std::uint32_t* count,void** p){
    Require(Route(cfg->local_eid.raw[15])&&cfg->peer_eid.raw[15]==2,"CTP uses route EIDs, not memory EIDs");
    Require(jetties==0&&cqs==0,"reference queries CTP before creating NPU queues");
    info->tp_handle=1;*count=1;*p=new int;return 0;
  };
  a->get_async_req_result=[](void* p,int* result){*result=0;return Destroy(p);};
  return true;
}
void CloseDirectHccp(DirectHccpApi* a){*a={};}
bool ImportPeerJetty(const DirectHccpApi&,void*,const std::uint8_t* key,std::uint32_t,std::uint32_t token,
    std::uint64_t,std::uint64_t,std::uint32_t tx,std::uint32_t rx,const char*,QpImportInfoT*,void** p){
  Require(key[15]==4,"CPU Jetty identity is not rewritten with CPU route EID");
  Require(token==kRegionToken,"NPU imports CPU Jetty with region token");
  Require(tx==1&&rx==2,"reference NPU/CPU PSNs");
  Require(key[20]==10+peer_imports,"all four CPU keys imported in order");
  ++peer_imports;
  *p=new int;return peer_imports!=fail_peer_at;
}
}
struct Backend final:p::Backend {
  unsigned candidate;
  bool allocated{};
  explicit Backend(unsigned id):candidate(id){}
  std::vector<std::uint8_t> memory;
  std::thread::id opener;
  p::Endpoint source{};
  kvaf::Status Allocate(std::uint64_t bytes,p::Endpoint* e) override {
    Require(!allocated,"one allocation per backend");
    ++allocations; ++pools; allocated=true;
    memory.resize(bytes);e->base=reinterpret_cast<std::uint64_t>(memory.data());e->bytes=bytes;
    e->eid[15]=2;e->token=kRegionToken;return kvaf::Status::OK();
  }
  kvaf::Status Open(const p::Endpoint& s,p::Endpoint* e) override {
    Require(Route(s.eid[15])&&s.segment[15]==3,"wire preserves route and memory identities");
    Require(jetties==4&&cqs==4&&s.psn==1,"four reference NPU channels before Open");
    for(std::size_t i=0;i<p::kChannelCount;++i){
      Require(s.key[i][15]==3&&s.key[i][20]==i+1,"wire preserves all four NPU Jetty keys");
      e->key[i][15]=4;e->key[i][20]=10+i;
    }
    Require(s.token==kRegionToken,"Open echoes allocated region token");
    {
      std::lock_guard<std::mutex> lock(attempts_mutex);
      attempted_pairs.emplace_back(candidate,s.eid[15]);
    }
    opener=std::this_thread::get_id();
    source=s;e->key_bytes=28;e->tp=1;e->psn=2;return kvaf::Status::OK();
  }
  kvaf::Status Read(const std::vector<p::Span>& spans) override {
    const int previous=reads.fetch_add(1);
    Require(event_waits>previous,"producer event must complete before READ");
    Require(registrations>0,"source MR must remain registered");
    Require(peer_imports==4,"bind every channel before READ");
    if(spans.size()==1&&spans[0].source==64&&spans[0].bytes<=64){
      Require(source.bytes==4096&&std::this_thread::get_id()==opener,"bounded startup READ on service thread");
    }else{
      Require(std::this_thread::get_id()!=opener,"user payload READ on worker thread");
    }
    for(auto s:spans)std::memcpy(memory.data()+s.destination,reinterpret_cast<void*>(source.base+s.source),s.bytes);
    return kvaf::Status::OK();
  }
  kvaf::Status Probe(const std::vector<p::Span>& spans) override {
    auto status=Read(spans);
    if (int(candidate)==failed_cpu || source.eid[15]==failed_route)
      return kvaf::Status::Internal("injected completed-and-retired probe failure");
    if (source.eid[15]==corrupt_route) memory[spans[0].destination]^=1;
    return status;
  }
  kvaf::Status ReleaseSource() override{return kvaf::Status::OK();}
  kvaf::Status Close() override{if(allocated){--pools;allocated=false;}return kvaf::Status::OK();}
  std::uint64_t Checksum(const std::vector<p::Span>& spans)const override{
    std::uint64_t hash=14695981039346656037ULL;for(auto s:spans)hash=p::Hash(memory.data()+s.destination,s.bytes,hash);return hash;
  }
};
class Server {
 public:
  p::Service service{[](std::uint32_t id){return std::make_unique<Backend>(id);}};
  int listener=-1;
  std::thread thread;
  std::atomic<bool> stop{false};
  std::atomic<std::uint16_t> drop{0};
  std::atomic<std::uint16_t> clear_token{0};
  std::atomic<p::State> query_state{p::State::Empty};
  std::uint16_t port{};
  Server(){
    listener=socket(AF_INET,SOCK_STREAM,0);Require(listener>=0,"socket");
    sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    Require(!bind(listener,reinterpret_cast<sockaddr*>(&address),sizeof(address)),"bind loopback");
    socklen_t size=sizeof(address);Require(!getsockname(listener,reinterpret_cast<sockaddr*>(&address),&size),"getsockname");
    port=ntohs(address.sin_port);Require(!listen(listener,16),"listen");
    thread=std::thread([this]{
      while(!stop){
        int fd=accept(listener,nullptr,nullptr);if(fd<0)break;
        p::SetSocketTimeout(fd);p::Request request{};
        if(!p::ReadBytes(fd,&request.header,sizeof(request.header))){close(fd);continue;}
        if(request.header.magic==kvaf::peer::kProtocolMagic){
          Require(request.header.command==static_cast<std::uint16_t>(kvaf::peer::Command::kGetCandidateCount),"candidate discovery RPC");
          ++candidate_queries;
          kvaf::peer::CandidateCountResponse response{};
          response.header=kvaf::peer::MakeHeader(kvaf::peer::Command::kGetCandidateCount,sizeof(response));
          response.header.command|=0x8000U;
          response.candidate_count=2;response.device_count=2;
          p::WriteBytes(fd,&response,sizeof(response));close(fd);continue;
        }
        if(p::ReadBytes(fd,reinterpret_cast<char*>(&request)+sizeof(request.header),sizeof(request)-sizeof(request.header))&&request.count<=p::kMaxSpans){
          std::vector<p::Span> spans(request.count);
          if(p::ReadBytes(fd,spans.data(),spans.size()*sizeof(p::Span))){
            auto response=service.Dispatch(request,spans);
            if(request.header.command==clear_token.load())response.endpoint.token=0;
            const auto state=query_state.load();
            if(request.header.command==static_cast<std::uint16_t>(p::Command::Query)&&state!=p::State::Empty)
              response.state=state;
            auto expected=request.header.command;
            if(!drop.compare_exchange_strong(expected,0))p::WriteBytes(fd,&response,sizeof(response));
          }
        }
        close(fd);
      }
    });
  }
  ~Server(){stop=true;shutdown(listener,SHUT_RDWR);close(listener);if(thread.joinable())thread.join();}
};
void StartupFailureTest(Server& server) {
  auto fail=[&](const char* diagnostic,bool physical=true){
    kvaf::OffloadRuntime runtime({0,"00000000000000000000000000000001",0,1000});
    Require(runtime.LinkTo({"127.0.0.1:"+std::to_string(server.port)}).ok(),"link failed startup");
    const auto status=runtime.Initialize(1024,256);
    Require(!status.ok()&&status.message().find(diagnostic)!=std::string::npos,"startup preserves failure detail");
    if(physical)Require(status.message().find("physical=0")!=std::string::npos,"startup identifies physical device");
    Require(!runtime.initialized()&&!runtime.pending_offload(),"setup failure is safe, not ready or pending");
    Require(reads==0&&registrations==0&&contexts==0&&sessions==0&&cqs==0&&jetties==0&&peer_imports==0,
            "setup failure cleans source before any READ");
    Require(runtime.Shutdown().ok()&&probe_allocations==0,"failed startup releases probe and CPU pool");
  };
  context_error=-17;fail("RaCtxInit ret=-17");context_error=0;
  null_context=true;fail("RaCtxInit ret=0 context_nonnull=0");null_context=false;
  bad_segment=BadSegment::ZeroBase;fail("zero_base=1");
  bad_segment=BadSegment::Short;fail("short_segment=1");
  bad_segment=BadSegment::Overflow;fail("address_overflow=1");
  bad_segment=BadSegment::Abi;fail("key_bytes=52 expected=48");
  bad_segment=BadSegment::None;
  for(int index=1;index<=int(p::kChannelCount);++index){
    fail_jetty_at=index;fail("NPU target Jetty creation failed",false);fail_jetty_at=0;
    fail_peer_at=index;fail("import CPU initiator",false);fail_peer_at=0;
  }
  server.clear_token=static_cast<std::uint16_t>(p::Command::Allocate);
  fail("CPU Allocate omitted region token",false);
  server.clear_token=static_cast<std::uint16_t>(p::Command::Open);
  fail("CPU Jetty token differs",false);
  server.clear_token=0;
}
void LostProbeResponseTest(Server& server){
  kvaf::OffloadRuntime runtime({0,"00000000000000000000000000000001",0,1000});
  Require(runtime.LinkTo({"127.0.0.1:"+std::to_string(server.port)}).ok(),"link probe recovery");
  const int before=reads;
  server.drop=static_cast<std::uint16_t>(p::Command::Probe);
  Require(runtime.Initialize(1024,256).code()==kvaf::StatusCode::kInDoubt,"lost probe ACK is pending");
  Require(!runtime.initialized()&&runtime.pending_offload()&&probe_allocations==1&&registrations==1,
          "pending initialization retains its internal HBM probe");
  Require(runtime.Shutdown().code()==kvaf::StatusCode::kInDoubt,"cannot free pending initialization");
  Require(runtime.ResolveOffload().ok()&&reads==before+1&&registrations==0,"probe resolution never repeats READ");
  Require(!runtime.initialized(),"resolving transfer alone does not bypass startup checksum");
  Require(runtime.Shutdown().ok()&&probe_allocations==0,"recovered probe released on shutdown");
  kvaf::OffloadRuntime fresh({0,"00000000000000000000000000000001",0,1000});
  Require(fresh.LinkTo({"127.0.0.1:"+std::to_string(server.port)}).ok(),"link fresh runtime");
  Require(fresh.Initialize(1024,256).ok()&&fresh.initialized(),"fresh runtime verifies probe checksum");
  Require(fresh.Shutdown().ok(),"close fresh runtime");
}
void AutomaticCandidatesTest(Server& server) {
  EidFixture fixture;
  const std::string eid1="00000000000000000000000000000001", eid5="00000000000000000000000000000005";
  EidFixture::Write(fixture.paths.rootinfo,EidFixture::Clos(0,eid1)+EidFixture::Clos(0,eid5));
  const auto* prior=std::getenv("KVAF_HCCL_ROOTINFO_PATH");
  const bool had_prior=prior!=nullptr;
  const std::string saved=prior?prior:"";
  setenv("KVAF_HCCL_ROOTINFO_PATH",fixture.paths.rootinfo.c_str(),1);
  auto_mode=true;
  const auto link=[&](kvaf::OffloadRuntime& r){Require(r.LinkTo({"127.0.0.1:"+std::to_string(server.port)}).ok(),"link automatic candidates");};
  const auto clean=[] {
    Require(pools==0&&probe_allocations==0&&registrations==0&&contexts==0&&sessions==0&&cqs==0&&jetties==0&&peer_imports==0,
            "every rejected attempt fully releases both peers");
  };
  const auto clear_attempts=[] {std::lock_guard<std::mutex> lock(attempts_mutex);attempted_pairs.clear();};
  const auto expect_attempts=[](const std::vector<std::pair<unsigned,unsigned>>& expected){
    std::lock_guard<std::mutex> lock(attempts_mutex);
    Require(attempted_pairs==expected,"CPU outer/NPU inner candidate ordering and explicit pinning");
  };
  clean();clear_attempts();failed_cpu=0;failed_route=1;
  kvaf::OffloadRuntime automatic({});link(automatic);
  Require(automatic.selected_cpu_candidate()==kvaf::OffloadRuntimeOptions::kAutoCandidate&&automatic.selected_npu_eid().empty(),"selection unavailable before probe");
  Require(automatic.Initialize(1024,256).ok(),"fallback across both NPU and CPU candidates");
  expect_attempts({{0,1},{0,5},{1,1},{1,5}});
  Require(automatic.selected_cpu_candidate()==1&&automatic.selected_npu_eid()==eid5&&pools==1,"selected pair and sole retained pool");
  std::uint8_t payload[16]={1,2,3,4};aclrtStream stream{};aclrtCreateStream(&stream);
  Require(automatic.Offload({{0},{payload,sizeof(payload),kvaf::DeviceKind::kNpu},{0},{sizeof(payload)}},{stream}).ok(),"selected route reused for real payload and probe slot is free");
  std::uint64_t hash{};Require(automatic.LastChecksum(&hash).ok()&&hash==p::Hash(payload,sizeof(payload)),"selected payload checksum");
  aclrtDestroyStream(stream);
  Require(automatic.Shutdown().ok(),"close selected route");clean();
  failed_cpu=-1;clear_attempts();
  kvaf::OffloadRuntime pinned_cpu({0,"",0,1000});link(pinned_cpu);
  const auto queries=candidate_queries.load();
  Require(pinned_cpu.Initialize(1024,256).ok(),"auto NPU with pinned CPU");expect_attempts({{0,1},{0,5}});
  Require(candidate_queries==queries&&pinned_cpu.selected_cpu_candidate()==0,"pinned CPU skips discovery");
  Require(pinned_cpu.Shutdown().ok(),"close pinned CPU");clean();
  failed_route=0;failed_cpu=0;clear_attempts();
  kvaf::OffloadRuntime pinned_eid({0,eid5});link(pinned_eid);
  Require(pinned_eid.Initialize(1024,256).ok(),"auto CPU with pinned NPU EID");expect_attempts({{0,5},{1,5}});
  Require(pinned_eid.Shutdown().ok(),"close pinned NPU");clean();
  failed_cpu=-1;failed_route=1;clear_attempts();
  kvaf::OffloadRuntime explicit_bad({0,eid1,0,1000});link(explicit_bad);
  Require(!explicit_bad.Initialize(1024,256).ok()&&!explicit_bad.pending_offload(),"explicit bad route fails without fallback");expect_attempts({{0,1}});clean();
  failed_route=0;corrupt_route=1;clear_attempts();
  kvaf::OffloadRuntime checksum_fallback({0,"",0,1000});link(checksum_fallback);
  Require(checksum_fallback.Initialize(1024,256).ok()&&checksum_fallback.selected_npu_eid()==eid5,"CQ success alone cannot select a corrupt probe");expect_attempts({{0,1},{0,5}});
  Require(checksum_fallback.Shutdown().ok(),"close checksum selection");clean();corrupt_route=0;
  context_error=-17;const auto before=allocations.load();
  kvaf::OffloadRuntime all_bad({});link(all_bad);
  auto status=all_bad.Initialize(1024,256);
  Require(!status.ok()&&status.message().find("all 4 CPU/NPU candidate pairs failed")!=std::string::npos&&allocations==before+4,"bounded exhaustion reports failure count");
  Require(!all_bad.initialized()&&all_bad.selected_npu_eid().empty(),"no false selection on exhaustion");clean();context_error=0;
  // Lost control ACKs must stop iteration, preserving the original identity.
  for(auto command:{p::Command::Allocate,p::Command::Probe,p::Command::Close}) {
    kvaf::OffloadRuntime unresolved({0,"",0,1000});link(unresolved);
    const auto before=allocations.load(), read_before=reads.load();
    if(command==p::Command::Close) failed_route=1;
    server.drop=static_cast<std::uint16_t>(command);
    status=unresolved.Initialize(1024,256);
    Require(status.code()==kvaf::StatusCode::kInDoubt&&allocations==before+1&&!unresolved.initialized(),"unknown response/cleanup stops candidate fallback");
    if(command==p::Command::Probe){
      Require(unresolved.pending_offload()&&probe_allocations==1&&registrations==1,"pending probe source retained");
      Require(unresolved.ResolveOffload().ok()&&reads==read_before+1,"resolve original probe without resubmit");
    }
    Require(unresolved.Shutdown().ok(),"recover same region after lost ACK");
    Require(allocations==before+1,"Allocate recovery never creates a second region");
    clean();failed_route=0;
  }
  // A reported pending/quarantined state must not try the next EID either.
  kvaf::OffloadRuntime quarantined({0,"",0,1000});link(quarantined);
  server.query_state=p::State::InDoubt;const auto read_before=reads.load();
  Require(quarantined.Initialize(1024,256).code()==kvaf::StatusCode::kInDoubt&&quarantined.pending_offload()&&reads==read_before+1,"quarantine stops selection");
  server.query_state=p::State::Empty;
  Require(quarantined.ResolveOffload().ok()&&quarantined.Shutdown().ok(),"recover original task only");clean();
  auto_mode=false;
  if(had_prior)setenv("KVAF_HCCL_ROOTINFO_PATH",saved.c_str(),1);else unsetenv("KVAF_HCCL_ROOTINFO_PATH");
}
int main(){try{
  Server server;
  StartupFailureTest(server);
  kvaf::OffloadRuntime runtime({0,"00000000000000000000000000000001",0,1000});
  Require(runtime.LinkTo({"127.0.0.1:"+std::to_string(server.port)}).ok(),"link");
  Require(runtime.Initialize(1024,256).ok()&&runtime.initialized(),"startup CPU Pull probe");
  Require(registrations==0&&events==0,"startup releases source resources");
  std::uint8_t source[256];for(int i=0;i<256;++i)source[i]=i;
  kvaf::BatchPutRequest request{{1,3},{source,256,kvaf::DeviceKind::kNpu},{7,150},{71,99}};
  aclrtStream stream{};aclrtCreateStream(&stream);
  Require(runtime.Offload(request,{stream}).ok(),"offload succeeds");
  auto expected=p::Hash(source+7,71);expected=p::Hash(source+150,99,expected);
  std::uint64_t hash{};Require(runtime.LastChecksum(&hash).ok()&&hash==expected,"CPU checksum");
  Require(registrations==0,"completed source unregistered");
  Require(runtime.core().BatchFree({{1,3}}).ok(),"free keys");
  server.drop=static_cast<std::uint16_t>(p::Command::Submit);
  Require(runtime.Offload(request,{stream}).code()==kvaf::StatusCode::kInDoubt,"lost submit ack becomes unknown");
  Require(runtime.pending_offload()&&registrations==1,"unknown retains MR");
  Require(runtime.Shutdown().code()==kvaf::StatusCode::kInDoubt,"unknown blocks shutdown");
  Require(runtime.ResolveOffload().ok()&&!runtime.pending_offload()&&registrations==0,"query recovers without resubmitting");
  Require(reads==3,"startup plus two user transfers, no duplicate");
  Require(runtime.core().BatchFree({{1,3}}).ok(),"free again");
  server.drop=static_cast<std::uint16_t>(p::Command::ReleaseSource);
  Require(runtime.Offload(request,{stream}).code()==kvaf::StatusCode::kInDoubt&&registrations==1,"lost release ack retains HBM");
  Require(runtime.ResolveOffload().ok()&&registrations==0,"release retry idempotent");
  Require(runtime.core().BatchFree({{1,3}}).ok(),"free before open failure");
  server.drop=static_cast<std::uint16_t>(p::Command::Open);
  Require(runtime.Offload(request,{stream}).code()==kvaf::StatusCode::kInDoubt,"lost open ack");
  Require(runtime.pending_offload()&&registrations==1,"lost open retains source");
  Require(!runtime.ResolveOffload().ok()&&!runtime.pending_offload()&&registrations==0,"unsubmitted task aborts after release confirmation");
  Require(reads==4,"lost open never submits DMA");
  Require(runtime.Offload(request,{stream}).ok(),"aborted slots reusable after lost open");
  for(auto state:{p::State::InDoubt,p::State::Running}){
    Require(runtime.core().BatchFree({{1,3}}).ok(),"free before remote status scenario");
    const auto before=reads.load();server.query_state=state;
    const auto status=runtime.Offload(request,{stream});
    const char* diagnostic=state==p::State::InDoubt?"transport quarantined":"still running after timeout_ms=1000";
    Require(status.code()==kvaf::StatusCode::kInDoubt&&status.message().find(diagnostic)!=std::string::npos,
            "quarantine differs from client wait timeout");
    Require(status.message().find("region=")!=std::string::npos&&status.message().find("task=")!=std::string::npos,
            "pending error identifies CPU task");
    Require(runtime.pending_offload()&&registrations==1,"pending remote state retains source");
    server.query_state=p::State::Empty;
    Require(runtime.ResolveOffload().ok()&&reads==before+1&&registrations==0,"query resolution never resubmits READ");
  }
  server.drop=static_cast<std::uint16_t>(p::Command::Close);
  Require(runtime.Shutdown().code()==kvaf::StatusCode::kInDoubt,"lost close ack");
  Require(runtime.Shutdown().ok(),"close retry");
  Require(events==0&&registrations==0&&contexts==0&&sessions==0&&probe_allocations==0&&cqs==0&&jetties==0&&peer_imports==0,
          "no source leaks on recovered paths");
  aclrtDestroyStream(stream);
  LostProbeResponseTest(server);
  AutomaticCandidatesTest(server);
  std::cout<<"Offload runtime/TCP recovery tests passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
