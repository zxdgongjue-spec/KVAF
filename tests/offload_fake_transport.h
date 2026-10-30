#pragma once
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
int fake_device=0;
enum class BadSegment { None, ZeroBase, Short, Overflow, Abi };
BadSegment bad_segment = BadSegment::None;
void Require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
extern "C" {
aclError aclrtGetDevice(int32_t* d){*d=fake_device;return 0;}
aclError aclrtSetDevice(int32_t d){fake_device=d;return 0;}
aclError aclrtMemcpy(void* dst,std::size_t capacity,const void* src,std::size_t bytes,int){
  if(bytes>capacity)return 1;std::memcpy(dst,src,bytes);return 0;
}
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
bool GetCurrentNpuPhysicalDeviceId(std::uint32_t* d){*d=fake_device;return true;}
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
