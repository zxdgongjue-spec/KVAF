// The real CPU READ backend linked against a fake URMA provider. Requires the
// public 24.12 headers; verifies descriptors, splitting and CQ correlation.
#include "cpu_pull_urma.h"
#include "urma_transport.h"
#include "urma_device_abi.h"
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace p = kvaf::offload;
void Require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
int posts{}, post_calls{}, local_mrs{}, imported_mrs{}, queues{}, peers{}, imports{};
int fail_create = -1, fail_import = -1;
bool fail_delete = false;
bool ctp_queried{};
std::uint32_t expected_token{};
enum class Fault { None, Post, PartialPost, Poll, CqStatus, LocalAccess, UserContext, Receive, Multiple, InteriorError, AckTimeout };
Fault fault = Fault::None;
int ack_timeout_queue=-1, corrupt_queue=-1;
std::array<int,p::kChannelCount> posts_by_queue{};
enum class BadLocalSegment { None, ZeroBase, Short, NoWrite, NonPin };
BadLocalSegment bad_local_segment = BadLocalSegment::None;
std::uint32_t registered_token{};
struct PendingQueue {
  std::vector<urma_jfs_wr_t*> wrs;
  std::uint64_t tail{}, last_context{};
  std::size_t index{};
  int delay{};
};
std::map<urma_jfc_t*, PendingQueue> pending;
int cqes{}, required_wave_queues{}, fail_post_call{-1};
bool check_wave{};
std::uint32_t returned_depth=8192;
urma_seg_t expected_segment{};
std::uint8_t expected_jetty_key[p::kChannelCount][64]{};
struct TpQuery { std::uint32_t flag, mode; std::uint8_t local[16], remote[16]; };
using Sge = kvaf::urma::Sge;
extern "C" {
urma_device_t** urma_get_device_list(int*){return nullptr;}
void urma_free_device_list(urma_device_t**){}
urma_eid_info_t* urma_get_eid_list(urma_device_t*,std::uint32_t*){return nullptr;}
void urma_free_eid_list(urma_eid_info_t*){}
urma_status_t urma_query_device(urma_device_t*,urma_device_attr_t* a){
  auto* expanded=reinterpret_cast<kvaf::urma::UbDeviceAttributes*>(a);
  *expanded={};expanded->dev_cap.max_jfs_sge=13;expanded->dev_cap.max_jfs_rsge=1;
  expanded->dev_cap.max_msg_size=4096;expanded->reserved_jetty_id_max=123;return URMA_SUCCESS;
}
urma_token_id_t* urma_alloc_token_id(urma_context_t* c){auto* t=new urma_token_id_t{};t->token_id=7;t->urma_ctx=c;return t;}
urma_status_t urma_free_token_id(urma_token_id_t* t){delete t;return URMA_SUCCESS;}
urma_target_seg_t* urma_register_seg(urma_context_t*,urma_seg_cfg_t* cfg){
  Require(cfg->flag.bs.access==URMA_ACCESS_LOCAL_WRITE&&!cfg->flag.bs.non_pin,"DDR must be pinned LOCAL_WRITE");
  Require(cfg->token_value.token&&cfg->token_id&&cfg->flag.bs.token_id_valid,"explicit DDR token");
  Require(imported_mrs>=1,"reference imports HBM before receiving DDR registration");
  Require(cfg->token_value.token==expected_token,"DDR uses allocated region token");
  registered_token=cfg->token_value.token;
  auto* mr=new urma_target_seg_t{};mr->seg.ubva.va=cfg->va;mr->seg.len=cfg->len;
  mr->seg.attr.value=cfg->flag.value;mr->seg.token_id=cfg->token_id->token_id;
  if(bad_local_segment==BadLocalSegment::ZeroBase)mr->seg.ubva.va=0;
  if(bad_local_segment==BadLocalSegment::Short)--mr->seg.len;
  if(bad_local_segment==BadLocalSegment::NoWrite)mr->seg.attr.bs.access=0;
  if(bad_local_segment==BadLocalSegment::NonPin)mr->seg.attr.bs.non_pin=1;
  ++local_mrs;return mr;
}
urma_status_t urma_unregister_seg(urma_target_seg_t* mr){--local_mrs;delete mr;return URMA_SUCCESS;}
urma_target_seg_t* urma_import_seg(urma_context_t*,urma_seg_t* seg,urma_token_t* token,std::uint64_t,urma_import_seg_flag_t flags){
  Require(flags.bs.access==(URMA_ACCESS_REMOTE_READ|URMA_ACCESS_REMOTE_WRITE),"reference HBM import access");
  Require(token->token==expected_token,"import HBM with region token");
  Require(!std::memcmp(seg,&expected_segment,sizeof(*seg)),"import exported segment verbatim, including memory EID");
  auto* mr=new urma_target_seg_t{};mr->seg=*seg;++imported_mrs;return mr;
}
urma_status_t urma_unimport_seg(urma_target_seg_t* mr){--imported_mrs;delete mr;return URMA_SUCCESS;}
urma_jfc_t* urma_create_jfc(urma_context_t*,urma_jfc_cfg_t* cfg){
  Require(ctp_queried&&cfg->depth==16384&&!cfg->flag.bs.lock_free,"reference CQ config and CTP order");
  auto* q=new urma_jfc_t{};pending.emplace(q,PendingQueue{});++queues;return q;
}
urma_status_t urma_delete_jfc(urma_jfc_t* q){pending.erase(q);--queues;delete q;return URMA_SUCCESS;}
urma_jfr_t* urma_create_jfr(urma_context_t*,urma_jfr_cfg_t* cfg){
  Require(cfg->token_value.token==registered_token,"CPU RQ shares DDR region token");
  Require(cfg->depth==8192&&cfg->max_sge==1&&cfg->trans_mode==URMA_TM_RM,"reference RQ config");
  ++queues;return new urma_jfr_t{};
}
urma_status_t urma_delete_jfr(urma_jfr_t* q){--queues;delete q;return URMA_SUCCESS;}
urma_jetty_t* urma_create_jetty(urma_context_t*,urma_jetty_cfg_t* cfg){
  Require(cfg->jfs_cfg.max_rsge==1&&cfg->jfs_cfg.max_sge==13&&cfg->jfs_cfg.depth==8192,"reference SQ limits");
  Require(!cfg->jfs_cfg.flag.bs.error_suspend&&!cfg->jfs_cfg.flag.bs.lock_free,"reference CPU queue flags");
  Require(cfg->jfs_cfg.priority==15&&cfg->jfs_cfg.rnr_retry==7&&cfg->jfs_cfg.err_timeout==17,"reference 24.12 retry settings");
  if(imports==fail_create)return nullptr;
  auto* q=new urma_jetty_t{};q->jetty_cfg=*cfg;q->jetty_cfg.jfs_cfg.depth=returned_depth;
  q->jetty_id.id=10+imports%p::kChannelCount;
  pending.at(cfg->jfs_cfg.jfc).index=imports%p::kChannelCount;
  ++queues;return q;
}
urma_status_t urma_delete_jetty(urma_jetty_t* q){if(fail_delete)return URMA_EINVAL;--queues;delete q;return URMA_SUCCESS;}
urma_status_t urma_unimport_jetty(urma_target_jetty_t* q){--peers;delete q;return URMA_SUCCESS;}
int urma_get_tp_list(void*,void* input,std::uint32_t* count,void* info){
  const auto* cfg=static_cast<TpQuery*>(input);
  Require(cfg->local[15]==1&&cfg->remote[15]==2,"CTP uses selected route EIDs, not segment EID");
  Require(local_mrs==1&&imported_mrs>=1&&queues%12==0,"CTP follows MR setup and precedes this group of queues");
  ctp_queried=true;*count=1;*static_cast<std::uint64_t*>(info)=9;return 0;
}
urma_target_jetty_t* urma_import_jetty_ex(void*,void* remote,void* token,void* active){
  Require(!std::memcmp(remote,expected_jetty_key[imports%p::kChannelCount],28),"import provider Jetty identity verbatim");
  Require(static_cast<urma_token_t*>(token)->token==registered_token,"import Jetty with region token");
  const auto* tp=static_cast<kvaf::urma::ActiveTp*>(active);
  Require(tp->tp_handle==9&&tp->peer_tp_handle==7&&tp->tx_psn&&tp->rx_psn==1,"reference CTP PSN mapping");
  if(imports==fail_import)return nullptr;
  ++peers;++imports;return new urma_target_jetty_t{};
}
urma_status_t urma_post_jetty_send_wr(urma_jetty_t* jetty,urma_jfs_wr_t* wr,urma_jfs_wr_t** bad){
  ++post_calls;
  if(fault==Fault::Post){*bad=wr;errno=EINVAL;return URMA_EINVAL;}
  auto& q=pending.at(jetty->jetty_cfg.jfs_cfg.jfc);
  std::size_t chain=0;
  for(auto* current=wr;current;current=current->next){
    if(fault==Fault::PartialPost&&post_calls==fail_post_call&&chain==1){
      *bad=current;errno=EIO;return URMA_EINVAL;
    }
    Require(++chain<=32,"doorbell chain bounded to 32 WRs");
    Require(!q.tail&&current->opcode==URMA_OPC_READ,"new wave cannot overwrite pending tail");
    Require(current->flag.bs.comp_order,"READ completion order required");
    Require(current->rw.src.num_sge==1&&current->rw.dst.num_sge==1,"one remote and local SGE; no remote gather");
    Require(current->user_ctx>q.last_context,"completion IDs never reused across waves/cache hits");
    q.last_context=current->user_ctx;
    const auto& source=*reinterpret_cast<Sge*>(current->rw.src.sge);
    const auto& destination=*reinterpret_cast<Sge*>(current->rw.dst.sge);
    Require(!source.user_tseg&&!destination.user_tseg,"reference 32-byte SGE tail is null");
    Require(source.len==destination.len&&source.len<=4096,"respect provider message limit");
    Require(p::Contains(destination.tseg->seg.len,destination.addr-destination.tseg->seg.ubva.va,destination.len),"destination range");
    Require(p::Contains(source.tseg->seg.len,source.addr-source.tseg->seg.ubva.va,source.len),"source range");
    if(current->flag.bs.complete_enable){
      Require(!current->next&&current->flag.bs.place_order==URMA_STRONG_ORDER,"only wave tail signals completion");
      Require(q.wrs.empty()||current->flag.bs.fence,"multi-READ tail fences preceding READs");
      q.tail=current->user_ctx;
    }else{
      Require(current->flag.bs.place_order==URMA_RELAX_ORDER&&!current->flag.bs.fence,"interior READs can overlap");
    }
    q.wrs.push_back(current);++posts;++posts_by_queue[q.index];
    Require(q.wrs.size()<=std::min(1024U,returned_depth),"bounded in-flight wave and returned SQ depth");
  }
  return URMA_SUCCESS;
}
int urma_poll_jfc(urma_jfc_t* cq,int capacity,urma_cr_t* cr){
  Require(capacity==8,"reference CQ poll batch");
  auto& q=pending.at(cq);
  Require(q.tail!=0,"poll follows tail post");
  if(check_wave&&required_wave_queues){
    int active=0;for(auto& item:pending)active+=item.second.tail!=0;
    Require(active==required_wave_queues,"all Jettys posted before polling any CQ");
    check_wave=false;
  }
  if(q.delay){--q.delay;return 0;} // Other CQs must progress while this one is delayed.
  if(fault==Fault::Poll){errno=EIO;return -1;}
  cr->status=fault==Fault::CqStatus?URMA_CR_REM_ACCESS_ABORT_ERR:URMA_CR_SUCCESS;
  if(fault==Fault::AckTimeout&&int(q.index)==ack_timeout_queue)cr->status=URMA_CR_ACK_TIMEOUT_ERR;
  if(fault==Fault::LocalAccess||fault==Fault::Receive||fault==Fault::Multiple||fault==Fault::InteriorError)cr->status=URMA_CR_LOC_ACCESS_ERR;
  cr->flag.bs.s_r=fault==Fault::Receive;
  cr->user_ctx=fault==Fault::UserContext?q.tail+1:fault==Fault::InteriorError?q.wrs.front()->user_ctx:q.tail;
  cr->local_id=17;cr->tpn=23;
  // DMA is deferred until completion, exercising descriptor lifetime and
  // ensuring later waves/cache reuse cannot mutate in-flight addresses.
  if(cr->status==URMA_CR_SUCCESS)for(const auto* wr:q.wrs){
    const auto& source=*reinterpret_cast<const Sge*>(wr->rw.src.sge);
    const auto& destination=*reinterpret_cast<const Sge*>(wr->rw.dst.sge);
    std::memcpy(reinterpret_cast<void*>(destination.addr),reinterpret_cast<void*>(source.addr),source.len);
    if(int(q.index)==corrupt_queue)*reinterpret_cast<std::uint8_t*>(destination.addr)^=1;
  }
  q.tail=0;q.wrs.clear();++cqes;
  check_wave=std::all_of(pending.begin(),pending.end(),[](const auto& item){return item.second.wrs.empty();});
  if(fault==Fault::Multiple)cr[1]=cr[0];
  return fault==Fault::Multiple?2:1;
}
}
struct TestSource { p::MemoryDesc memory; p::ConnectionDesc connection; };
kvaf::Status Open(p::Backend& backend, const TestSource& source) {
  p::ConnectionDesc local{};return backend.Open(source.connection,source.memory,&local);
}
TestSource Source(std::vector<std::uint8_t>& memory,std::uint32_t token,bool distinct_eids){
  TestSource source{};source.memory.base=reinterpret_cast<std::uint64_t>(memory.data());source.memory.bytes=memory.size();
  source.connection.tp=7;source.connection.psn=1;source.memory.token=source.connection.token=token;source.connection.key_bytes=32;source.memory.segment_bytes=48;source.connection.eid[15]=2;
  urma_seg_t segment{};segment.ubva.va=source.memory.base;segment.len=source.memory.bytes;
  segment.ubva.eid.raw[15]=distinct_eids?3:2;
  std::memcpy(source.memory.segment,&segment,sizeof(segment));auto mode=URMA_TM_RM;for(std::size_t i=0;i<p::kChannelCount;++i){
    source.connection.key[i][15]=segment.ubva.eid.raw[15];source.connection.key[i][20]=i+1;
    std::memcpy(source.connection.key[i]+24,&mode,4);
  }
  expected_token=token;ctp_queried=false;imports=0;
  expected_segment=segment;std::memcpy(expected_jetty_key,source.connection.key,sizeof(expected_jetty_key));
  return source;
}
void ReadTest(bool distinct_eids){
  urma_device_t device{};urma_context_t context{};context.dev=&device;context.eid.raw[15]=1;
  auto backend=p::CreateUrmaBackend(&context);
  p::MemoryDesc cpu{};Require(backend->Allocate(32768,&cpu).ok(),"allocate");
  Require(cpu.token&&local_mrs==0,"Allocate reserves DDR; registration waits for HBM import");
  std::vector<std::uint8_t> memory(20000);for(std::size_t i=0;i<memory.size();++i)memory[i]=i%251;
  auto source=Source(memory,cpu.token,distinct_eids);
  auto invalid=source;invalid.memory.base=source.memory.base-1;
  Require(!Open(*backend,invalid).ok(),"reject source before exported range");
  invalid=source;invalid.memory.bytes=source.memory.bytes+1;
  Require(!Open(*backend,invalid).ok(),"reject source extending past exported range");
  invalid=source;invalid.memory.base=UINT64_MAX;invalid.memory.bytes=2;
  Require(!Open(*backend,invalid).ok(),"reject source address overflow");
  invalid=source;invalid.memory.token=source.memory.token==1?2:1;
  Require(Open(*backend,invalid).message().find("source token differs")!=std::string::npos,"reject mismatched region token");
  invalid=source;invalid.memory.token=0;
  Require(!Open(*backend,invalid).ok(),"reject missing token");
  Require(imported_mrs==0&&queues==0,"range/token errors rejected before resource creation");
  Require(Open(*backend,source).ok()&&queues==12&&peers==4,"open four reference channels");
  Require(cpu.token==registered_token,"DDR registration shares region token");
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
    if(fault==Fault::CqStatus||fault==Fault::LocalAccess||fault==Fault::UserContext)
      Require(result.message().find("cq_status=")!=std::string::npos&&
              result.message().find("cq_user_ctx=")!=std::string::npos&&
              result.message().find("local_id=17 tpn=23")!=std::string::npos,"CQ diagnostics retain completion details");
    if(fault==Fault::LocalAccess)
      Require(result.message().find("cq_status_name=URMA_CR_LOC_ACCESS_ERR")!=std::string::npos,
              "local access failure is decoded");
    Require(result.message().find("dst_mr_access=1 dst_mr_token_id=7")!=std::string::npos&&
            result.message().find("dst_mr_bytes=32768")!=std::string::npos&&
            result.message().find("src_mr_bytes=20000")!=std::string::npos,"failure retains registered MR details");
    const int calls=post_calls;
    Require(backend->Read(spans).message()==result.message()&&post_calls==calls,"quarantined backend never resubmits READ");
    Require(backend->ReleaseSource().message()==result.message()&&backend->Close().message()==result.message(),
            "cleanup preserves original failure");
    Require(local_mrs==1&&imported_mrs==1&&queues==12&&peers==4,"quarantined resources remain alive");
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
  Require(backend->ReleaseSource().ok()&&imported_mrs==0&&queues==0&&peers==0&&local_mrs==1,"release source before pool");
  Require(backend->Close().ok()&&local_mrs==0,"release pool");
}
void ProfileSummaryTest(){
  const char* names[]={"KVAF_CPU_PULL_PROFILE","KVAF_CPU_PULL_PROFILE_WARMUP","KVAF_CPU_PULL_PROFILE_EACH"};
  std::string saved[3];bool existed[3]{};
  for(int i=0;i<3;++i)if(const auto* value=std::getenv(names[i])){existed[i]=true;saved[i]=value;}
  setenv(names[0],"1",1);setenv(names[1],"1",1);setenv(names[2],"0",1);
  FILE* output=std::tmpfile();Require(output,"capture profile output");
  std::fflush(stderr);const int old_stderr=dup(STDERR_FILENO);Require(old_stderr>=0,"save stderr");
  Require(dup2(fileno(output),STDERR_FILENO)>=0,"redirect profile output");
  urma_context_t context{};context.eid.raw[15]=1;
  auto backend=p::CreateUrmaBackend(&context);p::MemoryDesc cpu{};
  Require(backend->Allocate(4096,&cpu).ok(),"profile pool");
  std::vector<std::uint8_t> hbm(4096,0x75);
  Require(Open(*backend,Source(hbm,cpu.token,true)).ok(),"profile source");
  std::uint64_t checksum{};
  Require(backend->ProbeToScratch(64,64,&checksum).ok(),"profile excludes startup probe");
  for(int i=0;i<3;++i){
    Require(backend->Read({{64,128,64}}).ok(),"profile payload READ");
    Require(backend->Checksum({{64,128,64}})==p::Hash(hbm.data()+64,64),"profile excludes checksum");
  }
  Require(backend->Close().ok()&&backend->Close().ok(),"profile closes idempotently");
  std::fflush(stderr);Require(dup2(old_stderr,STDERR_FILENO)>=0,"restore stderr");close(old_stderr);
  std::rewind(output);std::string log;char bytes[1024];
  while(const auto n=std::fread(bytes,1,sizeof(bytes),output))log.append(bytes,n);
  std::fclose(output);
  for(int i=0;i<3;++i){if(existed[i])setenv(names[i],saved[i].c_str(),1);else unsetenv(names[i]);}
  Require(log.find("attempts=3 samples=3 failed=0")!=std::string::npos,"summary includes all payload batches only");
  Require(log.find("attempts=2 samples=2 failed=0")!=std::string::npos,"summary excludes configured warmup");
  Require(log.find("CPU_PULL_PROFILE mode=batch")==std::string::npos,"per-batch profile output disabled");
  std::size_t headers=0,pos=0;
  while((pos=log.find("CPU_PULL_SUMMARY profile_id=",pos))!=std::string::npos){++headers;++pos;}
  Require(headers==2&&log.find("reason=source_release")!=std::string::npos,"one all/measured summary on close");
}
void RegistrationFailureTest(){
  urma_context_t context{};
  for(auto invalid:{BadLocalSegment::ZeroBase,BadLocalSegment::Short,BadLocalSegment::NoWrite,BadLocalSegment::NonPin}){
    bad_local_segment=invalid;
    auto backend=p::CreateUrmaBackend(&context);p::MemoryDesc cpu{};
    Require(backend->Allocate(4096,&cpu).ok(),"allocate invalid-registration pool");
    std::vector<std::uint8_t> memory(4096);
    const auto source=Source(memory,cpu.token,true);
    const auto status=Open(*backend,source);
    Require(status.message().find("invalid registered DDR")!=std::string::npos,"reject malformed DDR registration");
    Require(imported_mrs==1&&queues==0,"invalid DDR never linked or used for READ");
    Require(backend->ReleaseSource().ok(),"release failed source setup");
    Require(!Open(*backend,source).ok(),"retry never accepts previously invalid registered DDR");
    Require(backend->Close().ok()&&local_mrs==0,"failed registration cleans up");
  }
  bad_local_segment=BadLocalSegment::None;
}
void PartialChannelFailureTest(){
  urma_context_t context{};context.eid.raw[15]=1;
  for(bool importing:{false,true})for(int index=0;index<int(p::kChannelCount);++index){
    auto backend=p::CreateUrmaBackend(&context);p::MemoryDesc cpu{};
    Require(backend->Allocate(4096,&cpu).ok(),"allocate partial setup pool");
    std::vector<std::uint8_t> memory(4096,0x5a);
    auto source=Source(memory,cpu.token,true);
    (importing?fail_import:fail_create)=index;
    const int before=post_calls;
    Require(!Open(*backend,source).ok()&&post_calls==before,"partial channel setup never submits READ");
    Require(backend->ReleaseSource().ok()&&queues==0&&peers==0&&imported_mrs==0&&local_mrs==1,
            "partial four-channel setup cleans up and retains DDR pool");
    fail_import=fail_create=-1;
    source=Source(memory,cpu.token,true);
    Require(Open(*backend,source).ok(),"retry uses retained DDR registration");
    Require(backend->Read({{64,0,64}}).ok(),"retry can read after partial setup cleanup");
    Require(backend->Close().ok()&&queues==0&&peers==0&&local_mrs==0&&imported_mrs==0,"retry closes cleanly");
  }
}
void ProbeFailureTest(Fault injected, bool teardown_failure=false, int queue=0) {
  fault=injected;fail_delete=teardown_failure;
  ack_timeout_queue=queue;
  urma_context_t context{};context.eid.raw[15]=1;
  auto backend=p::CreateUrmaBackend(&context);p::MemoryDesc cpu{};
  Require(backend->Allocate(4096,&cpu).ok(),"probe pool");
  std::vector<std::uint8_t> memory(4096,0xa5);
  const auto source=Source(memory,cpu.token,true);
  Require(Open(*backend,source).ok(),"probe link");
  const auto before=posts;
  const auto result=backend->Probe({{64,0,64}});
  const bool retired=!teardown_failure&&(injected==Fault::LocalAccess||injected==Fault::CqStatus||injected==Fault::AckTimeout);
  Require(!result.ok(),"probe error cannot select candidate");
  if(injected==Fault::AckTimeout) {
    Require(posts==before+queue+1,"probe stops at the failing Jetty");
    Require(result.message().find("queue="+std::to_string(queue))!=std::string::npos&&
            result.message().find("cq_status_name=URMA_CR_ACK_TIMEOUT_ERR")!=std::string::npos,
            "ACK timeout identifies the failing Jetty and decoded status");
  }
  if(retired) {
    Require(result.code()==kvaf::StatusCode::kInternal&&result.message().find("probe queues and source retired")!=std::string::npos,
            "matching terminal CQ plus successful destruction permits fallback");
    Require(queues==0&&peers==0&&imported_mrs==0&&local_mrs==1,"retire transport before allowing fallback");
    Require(backend->ReleaseSource().ok()&&backend->Close().ok()&&local_mrs==0,"host can release and close rejected candidate");
    fault=Fault::None;
  } else {
    Require(result.code()==kvaf::StatusCode::kInDoubt,"unknown completion or failed teardown forbids fallback");
    const auto calls=post_calls;
    Require(!backend->Probe({{64,0,64}}).ok()&&post_calls==calls,"quarantined probe never retries READ");
    Require(!backend->ReleaseSource().ok()&&!backend->Close().ok(),"unresolved probe retains resources");
    Require(queues==12&&peers==4&&imported_mrs==1&&local_mrs==1,"unknown probe retains all leases");
    _exit(0);
  }
}
void AllJettyScratchTest() {
  urma_context_t context{};context.eid.raw[15]=1;
  auto pool=std::make_shared<std::vector<std::uint8_t>>(4096,0x3c);
  auto backend=p::CreateBorrowedUrmaBackend(&context,pool->data(),pool->size(),123,pool);
  std::vector<std::uint8_t> hbm(4096,0x5a);
  Require(Open(*backend,Source(hbm,123,true)).ok(),"open all-Jetty probe");
  const auto before=posts_by_queue;
  required_wave_queues=1;check_wave=true;
  std::uint64_t checksum{};
  Require(backend->ProbeToScratch(64,64,&checksum).ok()&&checksum==p::Hash(hbm.data()+64,64),"all-Jetty scratch checksum");
  for(std::size_t i=0;i<p::kChannelCount;++i)
    Require(posts_by_queue[i]==before[i]+1,"each Jetty must perform a startup READ");
  Require(std::all_of(pool->begin(),pool->end(),[](auto b){return b==0x3c;}),"all Jetty probes preserve business DDR");
  Require(local_mrs==1,"scratch registration is released after all Jettys complete");
  corrupt_queue=1;
  const auto calls=post_calls;
  const auto result=backend->ProbeToScratch(64,64,&checksum);
  Require(!result.ok()&&result.message().find("checksum differs across Jettys: queue=1")!=std::string::npos,
          "earlier Jetty checksum failure cannot be masked by the final Jetty");
  Require(post_calls==calls+2&&local_mrs==1,"checksum failure stops probing and releases scratch");
  corrupt_queue=-1;required_wave_queues=0;
  Require(backend->Close().ok()&&queues==0&&peers==0&&imported_mrs==0&&local_mrs==0,"all-Jetty probe cleanup");
}
void BorrowedPoolTest() {
  urma_context_t context{};context.eid.raw[15]=1;
  auto pool=std::make_shared<std::vector<std::uint8_t>>(8192,0x5a);
  auto backend=p::CreateBorrowedUrmaBackend(&context,pool->data(),pool->size(),123,pool);
  Require(pool.use_count()==2,"borrowed backend holds shared DDR lease");
  std::vector<std::uint8_t> hbm(4096,0x91);
  auto source=Source(hbm,123,true);p::MemoryDesc cpu{};cpu.token=123;
  Require(Open(*backend,source).ok()&&local_mrs==1,"borrowed pool registered locally");
  std::uint64_t hash{};
  Require(backend->ProbeToScratch(64,64,&hash).ok()&&hash==p::Hash(hbm.data()+64,64),"scratch probe checksum");
  Require(local_mrs==1&&std::all_of(pool->begin(),pool->end(),[](auto c){return c==0x5a;}),"scratch MR released; all DDR slots unchanged");
  Require(backend->Read({{32,256,128}}).ok()&&!std::memcmp(pool->data()+256,hbm.data()+32,128),"direct READ into borrowed DDR");
  Require(backend->Close().ok()&&pool.use_count()==1&&local_mrs==0,"close releases registration and lease, not pool allocation");
  (*pool)[0]=7;Require((*pool)[0]==7,"A still owns DDR after B detaches");
}
void MultipleMrTransportTest() {
  urma_context_t context{};context.eid.raw[15]=1;
  auto pool=std::make_shared<std::vector<std::uint8_t>>(32768,0xa5);
  const auto base=reinterpret_cast<std::uint64_t>(pool->data());
  constexpr std::uint32_t token=1337;
  auto destination=p::CreateUrmaDestination(&context,pool->data(),pool->size(),token);
  auto backend=p::CreateBorrowedUrmaBackend(&context,pool->data(),pool->size(),token,pool,destination);
  std::vector<std::vector<std::uint8_t>> memories(78);
  for(std::size_t i=0;i<memories.size();++i)memories[i].resize(20000+i*64,std::uint8_t(i+1));
  auto source=Source(memories[0],token,true);p::MemoryDesc cpu{};cpu.base=base;cpu.bytes=pool->size();cpu.token=token;
  Require(Open(*backend,source).ok(),"open shared MR group");
  for(std::size_t i=1;i<memories.size();++i) {
    auto mr=source;mr.memory.base=reinterpret_cast<std::uint64_t>(memories[i].data());mr.memory.bytes=memories[i].size();
    urma_seg_t segment{};std::memcpy(&segment,mr.memory.segment,sizeof(segment));
    segment.ubva.va=mr.memory.base;segment.len=mr.memory.bytes;segment.token_id=0x10000+i*256;
    std::memcpy(mr.memory.segment,&segment,sizeof(segment));expected_segment=segment;
    Require(backend->AddSource(i+1,mr.memory).ok(),"import additional HBM MR");
    Require(backend->AddSource(i+1,mr.memory).ok(),"same MR import is idempotent");
    ++mr.memory.bytes;Require(!backend->AddSource(i+1,mr.memory).ok(),"MR identity cannot be replaced");
  }
  Require(imported_mrs==78&&local_mrs==1&&queues==12&&peers==4&&imports==4,
          "78 imports share four Jettys and one DDR destination MR");
  Require(!p::CloseUrmaDestination(destination).ok(),"DDR registration cannot close while borrowed");
  const std::vector<p::Span> spans{{7,64,10001},{15000,16000,123}};
  for(std::size_t n=0;n<80;++n) {
    const auto index=n%78;Require(backend->SelectSource(index+1).ok(),"select registered MR");
    std::uint64_t hash{};
    Require(backend->ProbeToScratch(64,64,&hash).ok()&&hash==p::Hash(memories[index].data()+64,64),"probe selected MR");
    Require(local_mrs==1,"scratch is separate and retired");
    for(int repeat=0;repeat<2;++repeat) {
      memories[index][7]=std::uint8_t(n+repeat+100);
      Require(backend->Read(spans).ok(),"read switched MR and reuse plan");
      Require(!std::memcmp(pool->data()+64,memories[index].data()+7,10001)&&
              !std::memcmp(pool->data()+16000,memories[index].data()+15000,123),"WQE source base and tseg match selected MR");
    }
  }
  Require(!backend->SelectSource(0).ok()&&!backend->SelectSource(79).ok(),"unknown MR rejected");
  Require(backend->Close().ok()&&local_mrs==1&&imported_mrs==0&&queues==0&&peers==0,
          "source teardown preserves pool-owned DDR registration");
  // A later source/candidate borrows the same pinned registration, even though
  // it has its own peer link. Never allocate another receiving MR for the pool.
  backend=p::CreateBorrowedUrmaBackend(&context,pool->data(),pool->size(),token,pool,destination);
  Require(Open(*backend,Source(memories[1],token,true)).ok()&&local_mrs==1,"another source reuses pool MR");
  Require(backend->Read(spans).ok()&&backend->Close().ok(),"second source reads and detaches");
  Require(p::CloseUrmaDestination(destination).ok()&&local_mrs==0,"DDR owner finally unregisters MR");
  destination.reset();Require(pool.use_count()==1,"all source leases released");
}
void BulkReadTest(std::size_t count, std::uint32_t depth) {
  returned_depth=depth;
  urma_context_t context{};context.eid.raw[15]=1;
  auto backend=p::CreateUrmaBackend(&context);p::MemoryDesc cpu{};
  Require(backend->Allocate((count+1)*4096,&cpu).ok(),"bulk pool");
  std::vector<std::uint8_t> hbm((count+1)*4099);
  auto source=Source(hbm,cpu.token,true);
  Require(Open(*backend,source).ok(),"bulk link");
  std::vector<p::Span> spans;
  for(std::size_t i=0;i<count;++i)
    spans.push_back({(count-1-i)*4099+7,i*4096+3,65+i%1900});
  required_wave_queues=4;
  for(unsigned round=0;round<3;++round){
    for(std::size_t i=0;i<hbm.size();++i)hbm[i]=(i*17+round*31)%251;
    std::memset(reinterpret_cast<void*>(cpu.base),0xa5,cpu.bytes);
    if(round==2)for(auto& span:spans){span.source+=7;span.destination+=1;--span.bytes;}
    const int before_posts=posts,before_calls=post_calls,before_cqes=cqes;
    check_wave=true;
    for(auto& item:pending)item.second.delay=item.second.index==0?3:0;
    Require(backend->Read(spans).ok(),"bulk READ completes");
    std::size_t calls=0,tails=0;
    const std::size_t channels=4,limit=std::min(1024U,depth);
    for(std::size_t remaining=count;remaining;){
      const auto n=std::min(remaining,channels*limit);
      for(std::size_t q=0;q<channels;++q){
        const auto wrs=n/channels+(q<n%channels);
        calls+=(wrs+31)/32;tails+=wrs!=0;
      }
      remaining-=n;
    }
    Require(posts-before_posts==int(count)&&post_calls-before_calls==int(calls)&&cqes-before_cqes==int(tails),
            "one WR per discontiguous source; 32-WR chains; one CQE per active queue per wave");
    auto* actual=reinterpret_cast<const std::uint8_t*>(cpu.base);
    std::uint64_t hash=14695981039346656037ULL;
    for(const auto& span:spans){
      Require(!std::memcmp(actual+span.destination,hbm.data()+span.source,span.bytes),"bulk/cache/scattered bytes");
      Require(actual[span.destination-1]==0xa5&&actual[span.destination+span.bytes]==0xa5,"bulk preserves padding");
      hash=p::Hash(hbm.data()+span.source,span.bytes,hash);
    }
    Require(backend->Checksum(spans)==hash,"checksum waits for ALL queue tails including delayed queue zero");
    auto invalid=spans;invalid.back().destination=invalid.front().destination;
    const int calls_before=post_calls;
    Require(!backend->Read(invalid).ok()&&post_calls==calls_before,"overlapping destinations rejected before parallel DMA");
  }
  required_wave_queues=0;
  Require(backend->Close().ok()&&local_mrs==0&&queues==0&&pending.empty(),"bulk cleanup");
  returned_depth=8192;
}
void BulkFailureTest(Fault injected) {
  urma_context_t context{};context.eid.raw[15]=1;
  auto backend=p::CreateUrmaBackend(&context);p::MemoryDesc cpu{};
  Require(backend->Allocate(512*128,&cpu).ok(),"fault pool");
  std::vector<std::uint8_t> hbm(512*256,0x93);
  auto source=Source(hbm,cpu.token,true);
  Require(Open(*backend,source).ok(),"fault link");
  std::vector<p::Span> spans;
  for(unsigned i=0;i<512;++i)spans.push_back({i*256U,i*128U,64});
  fault=injected;
  // Queue zero has four complete chains posted; queue one's first chain
  // accepts a prefix and fails, while queue zero may still perform DMA.
  fail_post_call=post_calls+5;
  const auto result=backend->Read(spans);
  Require(result.code()==kvaf::StatusCode::kInDoubt,"bulk error quarantines whole transfer");
  if(injected==Fault::PartialPost){
    Require(result.message().find("queue=1")!=std::string::npos&&
            result.message().find("wr_index=1")!=std::string::npos&&
            result.message().find("bad_wr_is_current=1")!=std::string::npos,"partial-chain failure identifies actual WR and queue");
  }
  if(injected==Fault::InteriorError)
    Require(result.message().find("wr_index=0")!=std::string::npos,"unsignalled error CQE identifies interior READ");
  const auto calls=post_calls;
  Require(!backend->Read(spans).ok()&&post_calls==calls&&!backend->Close().ok(),"no retransmit or resource reuse after multi-queue error");
  Require(queues==12&&peers==4&&local_mrs==1&&imported_mrs==1,"all queue/MR resources retained");
  // The backend itself may be destroyed by an exception path: in-flight
  // descriptor storage and transport resources must still remain valid.
  backend.reset();
  std::size_t retained=0;
  for(const auto& item:pending)for(const auto* wr:item.second.wrs){
    Require(wr->opcode==URMA_OPC_READ&&wr->rw.src.num_sge==1,"quarantined descriptor remains alive");
    const auto* src=reinterpret_cast<const Sge*>(wr->rw.src.sge);
    const auto* dst=reinterpret_cast<const Sge*>(wr->rw.dst.sge);
    Require(src->len==64&&dst->len==64,"quarantined SGE remains alive");++retained;
  }
  Require(retained!=0,"failure scenario retains pending DMA on other queues");
  _exit(0);
}
int main(){try{
  AllJettyScratchTest();
  ProfileSummaryTest();
  BorrowedPoolTest();
  MultipleMrTransportTest();
  RegistrationFailureTest();
  PartialChannelFailureTest();
  ReadTest(false);
  ReadTest(true);
  BulkReadTest(257,8192); // Single wave reused, then changed offsets/lengths.
  BulkReadTest(4096,8192); // Full protocol limit, 1024 WRs per queue.
  BulkReadTest(512,64); // Returned SQ depth forces waves and storage reuse.
  BulkReadTest(33,8192); // Uneven distribution across four queues.
  ProbeFailureTest(Fault::LocalAccess);
  ProbeFailureTest(Fault::CqStatus);
  for(int queue=0;queue<int(p::kChannelCount);++queue)ProbeFailureTest(Fault::AckTimeout,false,queue);
  {
    const auto child=fork();Require(child>=0,"fork later Jetty teardown failure");
    if(child==0){try{ProbeFailureTest(Fault::AckTimeout,true,1);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';}_exit(1);}
    int status{};Require(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"later Jetty failure retains resources when teardown fails");
  }
  for(auto injected:{Fault::Post,Fault::Poll,Fault::UserContext,Fault::Receive,Fault::Multiple,Fault::LocalAccess}){
    const auto child=fork();Require(child>=0,"fork probe scenario");
    if(child==0){try{ProbeFailureTest(injected,injected==Fault::LocalAccess);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';}_exit(1);}
    int status{};Require(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"probe quarantine scenario passed");
  }
  for(auto injected:{Fault::Post,Fault::Poll,Fault::CqStatus,Fault::LocalAccess,Fault::UserContext}){
    const auto child=fork();Require(child>=0,"fork failure scenario");
    if(child==0){fault=injected;try{ReadTest(true);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';}_exit(1);}
    int status{};Require(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"READ failure scenario passed");
  }
  for(auto injected:{Fault::PartialPost,Fault::Poll,Fault::CqStatus,Fault::LocalAccess,
                     Fault::UserContext,Fault::Receive,Fault::Multiple,Fault::InteriorError}){
    const auto child=fork();Require(child>=0,"fork bulk failure scenario");
    if(child==0){try{BulkFailureTest(injected);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';}_exit(1);}
    int status{};Require(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"bulk failure scenario passed");
  }
  std::cout<<"CPU URMA READ provider model passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
