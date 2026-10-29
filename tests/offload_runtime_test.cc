// Executes the real CANN-host runtime and TCP protocol against fake ACL/HCCP
// and a host-memory backend. This tests orchestration, not hardware support.
#include "offload_runtime.h"
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
std::atomic<int> registrations{0}, events{0}, event_waits{0}, reads{0};
void Require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
extern "C" {
aclError aclrtGetDevice(int32_t* d){*d=0;return 0;}
aclError aclrtMalloc(void** p,std::size_t n,int){*p=std::malloc(n);return *p?0:1;}
aclError aclrtFree(void* p){std::free(p);return 0;}
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
bool AcquireDirectHccpSession(std::uint32_t,std::uint32_t,DirectHccpSession** s){*s=new DirectHccpSession;return true;}
void ReleaseDirectHccpSession(DirectHccpSession* s){delete s;}
bool GetCurrentNpuPhysicalDeviceId(std::uint32_t* d){*d=0;return true;}
int Destroy(void* p){delete static_cast<int*>(p);return 0;}
int DestroyWithContext(void*,void* p){return Destroy(p);}
bool LoadDirectHccp(DirectHccpApi* a){
  a->ctx_init=[](CtxInitCfg*,CtxInitAttr*,void** p){*p=new int;return 0;};
  a->ctx_deinit=Destroy;
  a->token_alloc=[](void*,TokenId*,void** p){*p=new int;return 0;};
  a->token_free=DestroyWithContext;
  a->lmem_register=[](void*,MrRegInfo* mr,void** p){
    ++registrations;*p=new int;mr->out.key.size=48;
    std::memset(mr->out.key.value,0,48);mr->out.key.value[15]=1;
    std::memcpy(mr->out.key.value+20,&mr->in.mem.addr,8);
    std::memcpy(mr->out.key.value+32,&mr->in.mem.size,8);return 0;
  };
  a->lmem_unregister=[](void*,void* p){--registrations;return Destroy(p);};
  a->cq_create=[](void*,CqInfo*,void** p){*p=new int;return 0;};a->cq_destroy=DestroyWithContext;
  a->qp_create=[](void*,QpCreateAttr*,QpCreateInfo* info,void** p){info->key.size=32;*p=new int;return 0;};
  a->qp_destroy=Destroy;a->qp_unimport=DestroyWithContext;
  a->get_tp_info_async=[](void*,GetTpCfg*,TpInfo* info,std::uint32_t* count,void** p){info->tp_handle=1;*count=1;*p=new int;return 0;};
  a->get_async_req_result=[](void* p,int* result){*result=0;return Destroy(p);};
  return true;
}
void CloseDirectHccp(DirectHccpApi* a){*a={};}
bool ImportPeerJetty(const DirectHccpApi&,void*,const std::uint8_t*,std::uint32_t,std::uint32_t,
    std::uint64_t,std::uint64_t,std::uint32_t,std::uint32_t,const char*,QpImportInfoT*,void** p){*p=new int;return true;}
}
struct Backend final:p::Backend {
  std::vector<std::uint8_t> memory;
  p::Endpoint source{};
  kvaf::Status Allocate(std::uint64_t bytes,p::Endpoint* e) override {
    memory.resize(bytes);e->base=reinterpret_cast<std::uint64_t>(memory.data());e->bytes=bytes;e->eid[15]=2;return kvaf::Status::OK();
  }
  kvaf::Status Open(const p::Endpoint& s,p::Endpoint* e) override {source=s;e->key_bytes=28;e->tp=1;e->psn=1;return kvaf::Status::OK();}
  kvaf::Status Read(const std::vector<p::Span>& spans) override {
    const int previous=reads.fetch_add(1);
    Require(event_waits>previous,"producer event must complete before READ");
    Require(registrations>0,"source MR must remain registered");
    for(auto s:spans)std::memcpy(memory.data()+s.destination,reinterpret_cast<void*>(source.base+s.source),s.bytes);
    return kvaf::Status::OK();
  }
  kvaf::Status ReleaseSource() override{return kvaf::Status::OK();}
  kvaf::Status Close() override{return kvaf::Status::OK();}
  std::uint64_t Checksum(const std::vector<p::Span>& spans)const override{
    std::uint64_t hash=14695981039346656037ULL;for(auto s:spans)hash=p::Hash(memory.data()+s.destination,s.bytes,hash);return hash;
  }
};
class Server {
 public:
  p::Service service{[](std::uint32_t){return std::make_unique<Backend>();}};
  int listener=-1;
  std::thread thread;
  std::atomic<bool> stop{false};
  std::atomic<std::uint16_t> drop{0};
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
        if(p::ReadBytes(fd,&request,sizeof(request))&&request.count<=p::kMaxSpans){
          std::vector<p::Span> spans(request.count);
          if(p::ReadBytes(fd,spans.data(),spans.size()*sizeof(p::Span))){
            auto response=service.Dispatch(request,spans);
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
int main(){try{
  Server server;
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
  server.drop=static_cast<std::uint16_t>(p::Command::Close);
  Require(runtime.Shutdown().code()==kvaf::StatusCode::kInDoubt,"lost close ack");
  Require(runtime.Shutdown().ok(),"close retry");
  Require(events==0&&registrations==0,"no source leaks on recovered paths");
  aclrtDestroyStream(stream);
  std::cout<<"Offload runtime/TCP recovery tests passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
