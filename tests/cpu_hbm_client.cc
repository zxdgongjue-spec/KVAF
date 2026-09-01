#include "hbm_rw_protocol.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <sys/file.h>

#include <urma_api.h>

namespace {
using namespace kvaf::direct_hccp;
using namespace kvaf::hbm_test;

struct ImportFlag { std::uint64_t value, stag, dtag; };
struct RemoteJettyEx { urma_jetty_id_t jetty_id; std::uint32_t trans_mode, policy, type, reserved; ImportFlag flag; std::uint32_t tp_type, reserved2; };
struct TpQuery { std::uint32_t flag, trans_mode; std::uint8_t local_eid[16], peer_eid[16]; };
struct LocalTpInfo { std::uint64_t tp_handle; };
struct ActiveTp { std::uint64_t tp_handle, peer_tp_handle, tag; std::uint32_t tx_psn, rx_psn; std::uint64_t reserved; };
using GetTpList = int (*)(urma_context_t*, TpQuery*, std::uint32_t*, LocalTpInfo*);
using ImportJettyEx = urma_target_jetty_t* (*)(urma_context_t*, RemoteJettyEx*, urma_token_t*, ActiveTp*);

bool ReadAll(int fd, void* data, std::size_t bytes) { auto* p = static_cast<std::uint8_t*>(data); for (std::size_t n=0;n<bytes;) { ssize_t r=recv(fd,p+n,bytes-n,0); if(r<=0)return false;n+=static_cast<std::size_t>(r);} return true; }
bool WriteAll(int fd, const void* data, std::size_t bytes) { const auto* p=static_cast<const std::uint8_t*>(data); for(std::size_t n=0;n<bytes;){ssize_t r=send(fd,p+n,bytes-n,MSG_NOSIGNAL);if(r<=0)return false;n+=static_cast<std::size_t>(r);}return true; }
int Connect(const std::string& host, std::uint16_t port) {
  addrinfo hints{}; hints.ai_socktype=SOCK_STREAM; hints.ai_family=AF_UNSPEC; addrinfo* list=nullptr;
  if(getaddrinfo(host.c_str(),std::to_string(port).c_str(),&hints,&list)!=0)return -1; int fd=-1;
  for(auto* a=list;a;a=a->ai_next){fd=socket(a->ai_family,a->ai_socktype,a->ai_protocol);if(fd>=0&&connect(fd,a->ai_addr,a->ai_addrlen)==0)break;if(fd>=0)close(fd);fd=-1;} freeaddrinfo(list); return fd;
}
int EidIndex(urma_device_t* device, const std::array<std::uint8_t,16>& wanted) {
  std::uint32_t count=0; auto* list=urma_get_eid_list(device,&count); if(!list)return -1; int result=-1;
  for(std::uint32_t i=0;i<count;++i) if(std::memcmp(list[i].eid.raw,wanted.data(),16)==0){result=static_cast<int>(list[i].eid_index);break;} urma_free_eid_list(list); return result;
}
bool PairInfoLocalEid(std::uint32_t physical, std::array<std::uint8_t,16>* eid) {
  if (eid == nullptr || access("/proc/asdrv_ub/dev_id", W_OK) != 0 || access("/proc/asdrv_ub/pair_info", R_OK) != 0) return false;
  const int lock_fd = open("/tmp/kvaf_hbm_eid.lock", O_CREAT | O_RDWR, 0600);
  if (lock_fd < 0 || flock(lock_fd, LOCK_EX) != 0) return false;
  bool found = false;
  { std::ofstream selector("/proc/asdrv_ub/dev_id");
    if (selector.is_open()) { selector << physical << '\n'; selector.close(); std::ifstream input("/proc/asdrv_ub/pair_info"); std::string line;
      while (std::getline(input,line)) { const std::size_t p=line.find("local_eid:"); if(p==std::string::npos)continue; std::string hex;for(unsigned char c:line.substr(p+10U))if(std::isxdigit(c))hex.push_back(static_cast<char>(c)); if(hex.size()!=32)continue; auto h=[](char c){return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1;};for(int i=0;i<16;++i){int x=h(hex[2*i]),y=h(hex[2*i+1]);if(x<0||y<0){hex.clear();break;}(*eid)[i]=static_cast<std::uint8_t>((x<<4)|y);}found=hex.size()==32; if(found)break; }
    }
  }
  (void)flock(lock_fd, LOCK_UN); close(lock_fd); return found;
}
struct State { urma_context_t* ctx=nullptr; urma_jfc_t* jfc=nullptr; urma_jfr_t* jfr=nullptr; urma_token_id_t* token=nullptr; urma_target_seg_t* local_seg=nullptr; urma_target_seg_t* remote_seg=nullptr; urma_jetty_t* jetty=nullptr; urma_target_jetty_t* remote_jetty=nullptr; std::uint64_t remote_va=0; std::vector<std::uint8_t> host; };
void Cleanup(State* s) { if(s->remote_jetty)urma_unimport_jetty(s->remote_jetty); if(s->jetty)urma_delete_jetty(s->jetty); if(s->jfr)urma_delete_jfr(s->jfr); if(s->remote_seg)urma_unimport_seg(s->remote_seg); if(s->local_seg)urma_unregister_seg(s->local_seg); if(s->token)urma_free_token_id(s->token); if(s->jfc)urma_delete_jfc(s->jfc); if(s->ctx)urma_delete_context(s->ctx); }
bool Transfer(State* s, bool write, std::uint64_t offset, std::size_t bytes) {
  urma_sge_t local{}; local.addr=reinterpret_cast<std::uint64_t>(s->host.data()); local.len=static_cast<std::uint32_t>(bytes); local.tseg=s->local_seg;
  urma_sge_t remote{}; remote.addr=s->remote_va+offset; remote.len=static_cast<std::uint32_t>(bytes); remote.tseg=s->remote_seg;
  urma_jfs_wr_t wr{}; wr.opcode=write?URMA_OPC_WRITE:URMA_OPC_READ; wr.flag.bs.complete_enable=1; wr.tjetty=s->remote_jetty; wr.user_ctx=0xCAFE; if(write){wr.rw.src={&local,1};wr.rw.dst={&remote,1};}else{wr.rw.src={&remote,1};wr.rw.dst={&local,1};}
  urma_jfs_wr_t* bad=nullptr; if(urma_post_jetty_send_wr(s->jetty,&wr,&bad)!=URMA_SUCCESS)return false;
  for(int i=0;i<500000;++i){urma_cr_t cr{};int n=urma_poll_jfc(s->jfc,1,&cr);if(n<0)return false;if(n==1)return cr.status==URMA_CR_SUCCESS&&cr.user_ctx==0xCAFE;}
  return false;
}
}

int main(int argc,char** argv){
  if(argc<2)return 2; std::string host=argv[1]; std::uint16_t port=29888; std::uint64_t bytes=4096; std::array<std::uint8_t,16> wanted{}; bool have=false; std::uint32_t eid_index=0; std::uint32_t physical=0;
  int arg_start=2; if(arg_start<argc && argv[arg_start][0]!='-'){port=static_cast<std::uint16_t>(std::strtoul(argv[arg_start++],nullptr,0));}
  for(int i=arg_start;i<argc;++i){std::string a=argv[i];if(i+1>=argc)return 2;if(a=="--bytes")bytes=std::strtoull(argv[++i],nullptr,0);else if(a=="--device-id")physical=std::strtoul(argv[++i],nullptr,0);else if(a=="--host-eid"){const char*t=argv[++i];if(std::strlen(t)!=32)return 2;for(int j=0;j<16;++j){auto h=[](char c){return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1;};int x=h(t[2*j]),y=h(t[2*j+1]);if(x<0||y<0)return 2;wanted[j]=static_cast<std::uint8_t>((x<<4)|y);}have=true;}else if(a=="--eid-index")eid_index=std::strtoul(argv[++i],nullptr,0);else return 2;}
  urma_init_attr_t init{}; if(bytes==0||urma_init(&init)!=URMA_SUCCESS)return 1; if(!have && !PairInfoLocalEid(physical,&wanted)){std::fprintf(stderr,"[HBM_TEST][CPU] cannot auto-discover local_eid from pair_info\n");return 1;} int fd=Connect(host,port); if(fd<0)return 1; Hello hello{kMagic,kVersion,0,bytes,0,eid_index}; if(!WriteAll(fd,&hello,sizeof(hello)))return 1; Export ex{}; if(!ReadAll(fd,&ex,sizeof(ex))||ex.magic!=kMagic){close(fd);return 1;}
  urma_eid_t host_eid{}; std::memcpy(host_eid.raw,wanted.data(),wanted.size()); urma_device_t* dev=urma_get_device_by_eid(host_eid,URMA_TRANSPORT_UB); if(!dev){std::fprintf(stderr,"[HBM_TEST][CPU] no UB device for selected local_eid\n");close(fd);return 1;} int idx=EidIndex(dev,wanted); if(idx<0)idx=static_cast<int>(eid_index); State s; s.host.resize(bytes); s.ctx=urma_create_context(dev,idx); if(!s.ctx)return 1; urma_jfc_cfg_t jc{};jc.depth=1024;s.jfc=urma_create_jfc(s.ctx,&jc); if(!s.jfc)return 1;
  s.token=urma_alloc_token_id(s.ctx); urma_seg_cfg_t lc{};lc.va=reinterpret_cast<std::uint64_t>(s.host.data());lc.len=bytes;lc.token_id=s.token;lc.token_value.token=0xCAFE;lc.flag.bs.token_policy=URMA_TOKEN_PLAIN_TEXT;lc.flag.bs.access=URMA_ACCESS_REMOTE_READ|URMA_ACCESS_REMOTE_WRITE|URMA_ACCESS_REMOTE_ATOMIC;lc.flag.bs.non_pin=1;lc.flag.bs.token_id_valid=1;s.local_seg=urma_register_seg(s.ctx,&lc); if(!s.local_seg)return 1;
  urma_seg_t remote_desc{};if(ex.segment_key_bytes<sizeof(remote_desc)||ex.segment_key_bytes>sizeof(ex.segment_key)||ex.hbm_bytes!=bytes){std::fprintf(stderr,"[HBM_TEST][CPU] invalid NPU segment descriptor\n");return 1;}std::memcpy(&remote_desc,ex.segment_key,sizeof(remote_desc));s.remote_va=remote_desc.ubva.va;urma_token_t rt{};rt.token=ex.token_value;urma_import_seg_flag_t rif{};rif.bs.access=URMA_ACCESS_REMOTE_READ|URMA_ACCESS_REMOTE_WRITE;rif.bs.mapping=URMA_SEG_NOMAP;s.remote_seg=urma_import_seg(s.ctx,&remote_desc,&rt,0,rif);if(!s.remote_seg)return 1;
  urma_jfs_cfg_t jfs{};jfs.depth=1024;jfs.trans_mode=URMA_TM_RM;jfs.max_sge=1;jfs.jfc=s.jfc;urma_jfr_cfg_t jfrc{};jfrc.depth=1024;jfrc.trans_mode=URMA_TM_RM;jfrc.max_sge=1;jfrc.jfc=s.jfc;jfrc.token_value.token=ex.token_value;s.jfr=urma_create_jfr(s.ctx,&jfrc);urma_jetty_cfg_t cfg{};cfg.flag.bs.share_jfr=1;cfg.jfs_cfg=jfs;cfg.shared.jfr=s.jfr;cfg.shared.jfc=s.jfc;s.jetty=urma_create_jetty(s.ctx,&cfg);if(!s.jetty)return 1;
  auto get_tp=reinterpret_cast<GetTpList>(dlsym(RTLD_DEFAULT,"urma_get_tp_list"));auto import_ex=reinterpret_cast<ImportJettyEx>(dlsym(RTLD_DEFAULT,"urma_import_jetty_ex"));if(!get_tp||!import_ex)return 1;TpQuery q{};q.flag=1;q.trans_mode=URMA_TM_RM;std::memcpy(q.local_eid,s.ctx->eid.raw,16);std::memcpy(q.peer_eid,ex.peer_eid,16);LocalTpInfo tp{};std::uint32_t count=1;if(get_tp(s.ctx,&q,&count,&tp)!=0||!tp.tp_handle)return 1;
  ChannelRequest cr{};cr.magic=kMagic;cr.version=kVersion;cr.local_tp=tp.tp_handle;cr.local_psn=1;cr.local_key_bytes=sizeof(urma_jetty_id_t)+sizeof(urma_transport_mode_t);std::memcpy(cr.local_eid,s.ctx->eid.raw,16);std::memcpy(cr.local_key,&s.jetty->jetty_id,sizeof(s.jetty->jetty_id));auto mode=s.jetty->jetty_cfg.jfs_cfg.trans_mode;std::memcpy(cr.local_key+sizeof(s.jetty->jetty_id),&mode,sizeof(mode));if(!WriteAll(fd,&cr,sizeof(cr)))return 1;ChannelResponse resp{};if(!ReadAll(fd,&resp,sizeof(resp))||resp.status!=0)return 1;
  RemoteJettyEx rj{};std::memcpy(&rj.jetty_id,resp.peer_key,sizeof(rj.jetty_id));std::memcpy(&rj.trans_mode,resp.peer_key+sizeof(rj.jetty_id),sizeof(rj.trans_mode));rj.policy=0;rj.type=URMA_JETTY; rj.flag.value=1;rj.tp_type=1;ActiveTp active{tp.tp_handle,resp.peer_tp,0,1,resp.peer_psn,0};urma_token_t pt{};pt.token=ex.token_value;s.remote_jetty=import_ex(s.ctx,&rj,&pt,&active);if(!s.remote_jetty)return 1;
  for(std::size_t i=0;i<bytes;++i)s.host[i]=static_cast<std::uint8_t>(i*13+7);bool ok=Transfer(&s,true,0,bytes);std::vector<std::uint8_t> expected=s.host;std::fill(s.host.begin(),s.host.end(),0);ok=ok&&Transfer(&s,false,0,bytes)&&s.host==expected;std::printf("[HBM_TEST][CPU] %s bytes=%llu\n",ok?"PASS":"FAIL",static_cast<unsigned long long>(bytes));close(fd);Cleanup(&s);urma_uninit();return ok?0:1;
}
