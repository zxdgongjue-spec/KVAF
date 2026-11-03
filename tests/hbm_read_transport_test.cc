// Exercise the production Host cache and import/teardown code with Host-backed
// ACL allocations. No NPU or network is needed; queue addresses are sentinels.
#include "peer_link.h"
#include "../access_operator/demo/aiv_batch_get/host/batch_get_host.cpp"
#include <stdexcept>
#include <iostream>

namespace {
int device_allocations{}, local_mrs{}, tokens{}, remote_mrs{}, import_calls{}, fail_import{}, fail_unimport{}, fail_copy{};
void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
class Link final:public kvaf::peer::PeerLink {
 public:
  Link():PeerLink("unused"){}
  kvaf::Status LookupRegion(uint64_t base,kvaf::peer::PeerRegion* r) const override {
    const auto id=base/0x10000;
    *r={};r->base=base;r->length=4096+id;r->token_value=37;r->token_id=(100+id)<<8;
    r->eid[15]=1;r->has_memory_eid=true;r->memory_eid[0]=id;r->memory_eid[15]=3;
    r->segment_key_bytes=48;r->segment_key[0]=id;
    return kvaf::Status::OK();
  }
};
KvafBatchGetTransport* Transport(Link& link) {
  auto* t=new KvafBatchGetTransport;
  t->peer_link=&link;t->paths.resize(1);t->paths[0].device_context=new int;
  std::array<uint8_t,hccp::kEidBytes> route{};
  Check(hccp::LoadPeerMemory(&link,0x10000,&t->peer_memory,&route),"load first MR");
  t->remote_memories.emplace(1,RemoteMemoryRegistration{t->peer_memory,0x10000,4097,{},true});
  t->paths[0].device_peer_memory_import=new int;++remote_mrs;
  t->worker_count=4;
  for(unsigned i=0;i<4;++i) {
    auto& e=t->entity_templates[i];e.magic=kDirectEntityMagic;e.version=kDirectEntityVersion;
    e.remote_host_ddr_address=0x10000;e.remote_host_ddr_size=4097;
    e.remote_token_id=t->peer_memory.token_id;e.remote_token_value=37;
    const auto eid=ReverseEidForAivWqe(t->peer_memory.wqe_eid);std::memcpy(e.remote_eid,eid.data(),16);
    e.sq_va=0x100+i;e.cq_va=0x200+i;e.tp_id=0x300+i;e.jetty_id=0x400+i;
  }
  t->api.token_alloc=[](void*,hccp::TokenId* token,void** out){*out=new int;++tokens;token->token_id=0x5500;return 0;};
  t->api.token_free=[](void*,void* p){--tokens;delete static_cast<int*>(p);return 0;};
  t->api.lmem_register=[](void*,hccp::MrRegInfo* info,void** out){
    Check(!(info->in.mem.addr%4096),"aligned output registration");
    *out=new int;++local_mrs;info->out.key.size=48;return 0;};
  t->api.lmem_unregister=[](void*,void* p){--local_mrs;delete static_cast<int*>(p);return 0;};
  t->api.rmem_import=[](void*,hccp::MrImportInfo*,void** out){
    *out=new int;++remote_mrs;return ++import_calls==fail_import?-1:0;};
  t->api.rmem_unimport=[](void*,void* p){if(fail_unimport){--fail_unimport;return -1;}--remote_mrs;delete static_cast<int*>(p);return 0;};
  t->api.ctx_deinit=[](void* p){delete static_cast<int*>(p);return 0;};
  return t;
}
}
extern "C" aclError aclrtGetDevice(int32_t* d){*d=0;return 0;}
extern "C" aclError aclrtMalloc(void** p,size_t n,int){*p=std::malloc(n);if(*p)++device_allocations;return *p?0:1;}
extern "C" aclError aclrtFree(void* p){if(p)--device_allocations;std::free(p);return 0;}
extern "C" aclError aclrtMemcpy(void* d,size_t capacity,const void* s,size_t n,int){if(fail_copy){--fail_copy;return 1;}if(n>capacity)return 1;std::memcpy(d,s,n);return 0;}
int main(){try{
  Link link;auto* t=Transport(link);
  for(uint64_t mr=2;mr<=78;++mr)Check(!KvafBatchGetTransportAddRemoteMemory(t,mr,mr*0x10000,4096+mr),"add MR to same context");
  Check(remote_mrs==78 && t->paths.size()==1 && t->worker_count==4,"78 imports share one context/four queues");
  std::vector<uint8_t> output(4096);
  std::map<uint64_t,uint64_t> cached;
  for(unsigned round=0;round<3;++round)for(uint64_t mr=1;mr<=78;++mr) {
    KvafBatchGetLocalBuffer* local=nullptr;uint64_t pointer{};
    Check(!KvafBatchGetLocalBufferAcquireForRemote(t,mr,output.data(),output.size(),&local,&pointer),"acquire selected MR descriptor");
    auto* entities=reinterpret_cast<KvafBatchGetDirectUrmaEntity*>(pointer);
    for(unsigned i=0;i<4;++i) {
      const auto& e=entities[i];
      Check(e.remote_host_ddr_address==mr*0x10000 && e.remote_host_ddr_size==4096+mr && e.remote_token_id==100+mr,
            "switch complete remote identity, not only base");
      Check(e.remote_token_value==37 && e.remote_eid[15]==mr && e.remote_eid[0]==3,"memory EID uses original WQE encoding");
      Check(e.local_hbm_address==reinterpret_cast<uint64_t>(output.data()) && e.local_token_id==0x55,"one local registration");
      Check(e.sq_va==0x100+i && e.cq_va==0x200+i && e.tp_id==0x300+i && e.jetty_id==0x400+i,"queue addresses unchanged across MR views");
    }
    if(round)Check(cached.at(mr)==pointer,"cached view is immutable and reused");else cached[mr]=pointer;
    KvafBatchGetLocalBufferRelease(local);
  }
  Check(local_mrs==1 && tokens==1 && device_allocations==78,"78 lightweight views; one output MR");
  fail_unimport=1;Check(KvafBatchGetTransportClose(&t)!=0 && t,"partial teardown retained");
  Check(!KvafBatchGetTransportClose(&t) && !t,"retry teardown");
  Check(!local_mrs && !tokens && !device_allocations && !remote_mrs,"all resources freed");
  t=Transport(link);fail_import=import_calls+1;
  Check(KvafBatchGetTransportAddRemoteMemory(t,2,0x20000,4098)!=0 && remote_mrs==2,"partial import handle retained");
  KvafBatchGetLocalBuffer* local=nullptr;uint64_t pointer{};
  Check(KvafBatchGetLocalBufferAcquireForRemote(t,2,output.data(),output.size(),&local,&pointer)!=0,"partial MR cannot be selected");
  Check(!KvafBatchGetTransportClose(&t) && !remote_mrs,"partial import cleanup");
  t=Transport(link);fail_import=0;
  Check(!KvafBatchGetTransportAddRemoteMemory(t,2,0x20000,4098),"import for upload-failure test");
  Check(!KvafBatchGetLocalBufferAcquireForRemote(t,1,output.data(),output.size(),&local,&pointer),"pre-register output");
  KvafBatchGetLocalBufferRelease(local);local=nullptr;fail_copy=1;
  Check(KvafBatchGetLocalBufferAcquireForRemote(t,2,output.data(),output.size(),&local,&pointer)!=0 && !local && !pointer,
        "partial descriptor upload is never returned to kernel");
  Check(local_mrs==1 && device_allocations==2,"partial descriptor remains owned");
  Check(!KvafBatchGetLocalBufferAcquireForRemote(t,2,output.data(),output.size(),&local,&pointer),"retry unfinished descriptor upload");
  Check(device_allocations==2,"retry reuses allocation");KvafBatchGetLocalBufferRelease(local);
  Check(!KvafBatchGetTransportClose(&t) && !local_mrs && !tokens && !device_allocations && !remote_mrs,"upload failure cleanup complete");
  t=Transport(link);
  Check(!KvafBatchGetTransportAddRemoteMemory(t,2,0x20000,4098),"import for legacy destroy");
  KvafBatchGetTransportDestroy(t);t=nullptr;
  Check(!remote_mrs && !local_mrs && !device_allocations,"legacy destroy also retires multi-MR imports");
  std::cout<<"NPU multi-MR production transport cache: identity, output reuse and teardown passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
