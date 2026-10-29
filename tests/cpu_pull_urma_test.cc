// The real CPU READ backend linked against a fake URMA provider. Requires the
// public 24.12 headers; verifies descriptors, splitting and CQ correlation.
#include "cpu_pull_urma.h"
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace p = kvaf::offload;
void Require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
int posts{}, post_calls{}, local_mrs{}, imported_mrs{}, queues{};
enum class Fault { None, Post, Poll, CqStatus, UserContext };
Fault fault = Fault::None;
std::uint64_t pending{};
urma_seg_t expected_segment{};
std::uint8_t expected_jetty_key[32]{};
struct TpQuery { std::uint32_t flag, mode; std::uint8_t local[16], remote[16]; };
#ifdef KVAF_CPU_PULL_EXTENDED_SGE
struct Sge {std::uint64_t addr;std::uint32_t len;urma_target_seg_t* tseg;void* user_tseg;};
#else
using Sge = urma_sge_t;
#endif
extern "C" {
urma_status_t urma_query_device(urma_device_t*,urma_device_attr_t* a){
  a->dev_cap.max_jfs_sge=1;a->dev_cap.max_jfs_rsge=1;a->dev_cap.max_msg_size=4096;return URMA_SUCCESS;
}
urma_token_id_t* urma_alloc_token_id(urma_context_t* c){auto* t=new urma_token_id_t{};t->token_id=7;t->urma_ctx=c;return t;}
urma_status_t urma_free_token_id(urma_token_id_t* t){delete t;return URMA_SUCCESS;}
urma_target_seg_t* urma_register_seg(urma_context_t*,urma_seg_cfg_t* cfg){
  Require(cfg->flag.bs.access==URMA_ACCESS_LOCAL_WRITE&&!cfg->flag.bs.non_pin,"DDR must be pinned LOCAL_WRITE");
  auto* mr=new urma_target_seg_t{};mr->seg.ubva.va=cfg->va;mr->seg.len=cfg->len;++local_mrs;return mr;
}
urma_status_t urma_unregister_seg(urma_target_seg_t* mr){--local_mrs;delete mr;return URMA_SUCCESS;}
urma_target_seg_t* urma_import_seg(urma_context_t*,urma_seg_t* seg,urma_token_t*,std::uint64_t,urma_import_seg_flag_t flags){
  Require(flags.bs.access==URMA_ACCESS_REMOTE_READ,"source must be READ-only");
  Require(!std::memcmp(seg,&expected_segment,sizeof(*seg)),"import exported segment verbatim, including memory EID");
  auto* mr=new urma_target_seg_t{};mr->seg=*seg;++imported_mrs;return mr;
}
urma_status_t urma_unimport_seg(urma_target_seg_t* mr){--imported_mrs;delete mr;return URMA_SUCCESS;}
urma_jfc_t* urma_create_jfc(urma_context_t*,urma_jfc_cfg_t*){++queues;return new urma_jfc_t{};}
urma_status_t urma_delete_jfc(urma_jfc_t* q){--queues;delete q;return URMA_SUCCESS;}
urma_jfr_t* urma_create_jfr(urma_context_t*,urma_jfr_cfg_t*){++queues;return new urma_jfr_t{};}
urma_status_t urma_delete_jfr(urma_jfr_t* q){--queues;delete q;return URMA_SUCCESS;}
urma_jetty_t* urma_create_jetty(urma_context_t*,urma_jetty_cfg_t* cfg){
  Require(cfg->jfs_cfg.max_rsge==1,"initiator remote SGE limit");
  auto* q=new urma_jetty_t{};q->jetty_cfg=*cfg;++queues;return q;
}
urma_status_t urma_delete_jetty(urma_jetty_t* q){--queues;delete q;return URMA_SUCCESS;}
urma_status_t urma_unimport_jetty(urma_target_jetty_t* q){delete q;return URMA_SUCCESS;}
int urma_get_tp_list(void*,void* input,std::uint32_t* count,void* info){
  const auto* cfg=static_cast<TpQuery*>(input);
  Require(cfg->local[15]==1&&cfg->remote[15]==2,"CTP uses selected route EIDs, not segment EID");
  *count=1;*static_cast<std::uint64_t*>(info)=9;return 0;
}
urma_target_jetty_t* urma_import_jetty_ex(void*,void* remote,void*,void*){
  Require(!std::memcmp(remote,expected_jetty_key,28),"import provider Jetty identity verbatim");
  return new urma_target_jetty_t{};
}
urma_status_t urma_post_jetty_send_wr(urma_jetty_t*,urma_jfs_wr_t* wr,urma_jfs_wr_t** bad){
  ++post_calls;
  if(fault==Fault::Post){*bad=wr;errno=EINVAL;return URMA_EINVAL;}
  Require(!pending&&wr->opcode==URMA_OPC_READ&&wr->flag.bs.complete_enable,"one signalled READ at a time");
  Require(wr->rw.src.num_sge==1&&wr->rw.dst.num_sge==1,"one source and destination");
  const auto& source=*reinterpret_cast<Sge*>(wr->rw.src.sge);
  const auto& destination=*reinterpret_cast<Sge*>(wr->rw.dst.sge);
  Require(source.len==destination.len&&source.len<=4096,"respect provider message limit");
  Require(p::Contains(destination.tseg->seg.len,destination.addr-destination.tseg->seg.ubva.va,destination.len),"destination range");
  Require(p::Contains(source.tseg->seg.len,source.addr-source.tseg->seg.ubva.va,source.len),"source range");
  std::memcpy(reinterpret_cast<void*>(destination.addr),reinterpret_cast<void*>(source.addr),source.len);
  pending=wr->user_ctx;++posts;return URMA_SUCCESS;
}
int urma_poll_jfc(urma_jfc_t*,int,urma_cr_t* cr){
  Require(pending!=0,"poll follows post");
  if(fault==Fault::Poll){errno=EIO;return -1;}
  cr->status=fault==Fault::CqStatus?URMA_CR_REM_ACCESS_ABORT_ERR:URMA_CR_SUCCESS;
  cr->user_ctx=fault==Fault::UserContext?pending+1:pending;
  cr->local_id=17;cr->tpn=23;pending=0;return 1;
}
}
void ReadTest(bool distinct_eids){
  urma_device_t device{};urma_context_t context{};context.dev=&device;context.eid.raw[15]=1;
  auto backend=p::CreateUrmaBackend(&context);
  p::Endpoint cpu{};Require(backend->Allocate(32768,&cpu).ok(),"allocate");
  std::vector<std::uint8_t> memory(20000);for(std::size_t i=0;i<memory.size();++i)memory[i]=i%251;
  p::Endpoint source{};source.base=reinterpret_cast<std::uint64_t>(memory.data());source.bytes=memory.size();
  source.tp=7;source.psn=1;source.token=2;source.key_bytes=32;source.segment_bytes=48;source.eid[15]=2;
  urma_seg_t segment{};segment.ubva.va=source.base;segment.len=source.bytes;
  segment.ubva.eid.raw[15]=distinct_eids?3:2;
  source.key[15]=segment.ubva.eid.raw[15];
  std::memcpy(source.segment,&segment,sizeof(segment));auto mode=URMA_TM_RM;std::memcpy(source.key+24,&mode,4);
  expected_segment=segment;std::memcpy(expected_jetty_key,source.key,sizeof(expected_jetty_key));
  auto invalid=source;invalid.base=segment.ubva.va-1;
  Require(!backend->Open(invalid,&cpu).ok(),"reject source before exported range");
  invalid=source;invalid.bytes=segment.len+1;
  Require(!backend->Open(invalid,&cpu).ok(),"reject source extending past exported range");
  invalid=source;invalid.base=UINT64_MAX;invalid.bytes=2;
  Require(!backend->Open(invalid,&cpu).ok(),"reject source address overflow");
  Require(imported_mrs==0&&queues==0,"range errors rejected before resource creation");
  Require(backend->Open(source,&cpu).ok(),"open source");
  std::vector<p::Span> spans{{7,64,10001},{15000,16000,123}};
  const int before=posts;
  const auto result=backend->Read(spans);
  if(fault!=Fault::None){
    Require(result.code()==kvaf::StatusCode::kInDoubt,"READ failure remains in doubt");
    const char* stage=fault==Fault::Post?"stage=post_read":fault==Fault::Poll?"stage=poll_cq":"stage=cq_completion";
    Require(result.message().find(stage)!=std::string::npos,"failure identifies READ stage");
    Require(result.message().find("sge_bytes="+std::to_string(sizeof(Sge)))!=std::string::npos,"failure identifies active SGE ABI");
    if(fault==Fault::Post)
      Require(result.message().find("errno="+std::to_string(EINVAL))!=std::string::npos&&
              result.message().find("bad_wr_is_current=1")!=std::string::npos,"post error keeps errno and bad WR");
    if(fault==Fault::CqStatus||fault==Fault::UserContext)
      Require(result.message().find("cq_status=")!=std::string::npos&&
              result.message().find("cq_user_ctx=")!=std::string::npos&&
              result.message().find("local_id=17 tpn=23")!=std::string::npos,"CQ diagnostics retain completion details");
    const int calls=post_calls;
    Require(backend->Read(spans).message()==result.message()&&post_calls==calls,"quarantined backend never resubmits READ");
    Require(backend->ReleaseSource().message()==result.message()&&backend->Close().message()==result.message(),
            "cleanup preserves original failure");
    Require(local_mrs==1&&imported_mrs==1&&queues==3,"quarantined resources remain alive");
    // This child models process exit with deliberately quarantined resources.
    // Do not destruct/free them, or report their required retention as a leak.
    _exit(0);
  }
  Require(result.ok()&&posts==before+4,"3+1 WR split");
  Require(!std::memcmp(reinterpret_cast<void*>(cpu.base+64),memory.data()+7,10001),"READ bytes");
  const auto* data=reinterpret_cast<const std::uint8_t*>(cpu.base);
  Require(data[63]==0&&data[10065]==0&&data[15999]==0&&data[16123]==0,"guard bytes unchanged");
  auto expected=p::Hash(memory.data()+7,10001);expected=p::Hash(memory.data()+15000,123,expected);
  Require(backend->Checksum(spans)==expected,"checksum source order");
  Require(backend->ReleaseSource().ok()&&imported_mrs==0&&queues==0&&local_mrs==1,"release source before pool");
  Require(backend->Close().ok()&&local_mrs==0,"release pool");
}
int main(){try{
  ReadTest(false);
  ReadTest(true);
  for(auto injected:{Fault::Post,Fault::Poll,Fault::CqStatus,Fault::UserContext}){
    const auto child=fork();Require(child>=0,"fork failure scenario");
    if(child==0){fault=injected;try{ReadTest(true);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';}_exit(1);}
    int status{};Require(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"READ failure scenario passed");
  }
  std::cout<<"CPU URMA READ provider model passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
