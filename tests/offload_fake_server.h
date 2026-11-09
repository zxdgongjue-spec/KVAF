#pragma once
#include "offload_fake_transport.h"
#include "remote_pull_service.h"
#include <map>
#include <set>
#include "runtime_service_loop.h"
namespace rp = kvaf::remote_pull;
using kvaf::Status;
std::atomic<int> payload_reads{0}, scratch_reads{0}, checksum_calls{0}, leases{0};
std::atomic<int> read_delay_ms{0};
std::atomic<bool> hold_read{false}, uncertain_read{false}, fail_read{false}, fail_checksum{false}, fail_select{false};
struct SharedBackend final : p::Backend {
  std::vector<std::uint8_t>& pool;
  p::MemoryDesc source{};
  p::ConnectionDesc connection{};
  std::map<std::uint64_t,p::MemoryDesc> memories;
  bool closed{};
  explicit SharedBackend(std::vector<std::uint8_t>& v):pool(v){++leases;}
  Status Allocate(std::uint64_t,p::MemoryDesc*) override { throw std::runtime_error("B must not allocate DDR"); }
  Status Open(const p::ConnectionDesc& peer,const p::MemoryDesc& s,p::ConnectionDesc* cpu) override {
    connection=peer; source=s; memories[1]=s; cpu->key_bytes=28;cpu->tp=1;cpu->psn=2;
    for(unsigned i=0;i<4;++i){cpu->key[i][15]=4;cpu->key[i][20]=10+i;}
    return Status::OK();
  }
  Status AddSource(std::uint64_t id,const p::MemoryDesc& s) override { memories[id]=s;return Status::OK(); }
  Status SelectSource(std::uint64_t id) override {
    if(fail_select)return Status::Internal("simulated MR selection failure");
    auto found=memories.find(id);if(found==memories.end())return Status::InvalidArgument("missing MR");
    source=found->second;return Status::OK();
  }
  Status ProbeToScratch(std::uint64_t offset,std::uint64_t bytes,std::uint64_t* hash) override {
    ++scratch_reads;
    if(connection.eid[15]==failed_route) return Status::Internal("retired candidate probe");
    *hash=p::Hash(reinterpret_cast<const std::uint8_t*>(source.base+offset),bytes);
    return Status::OK();
  }
  Status Read(const std::vector<p::Span>& spans) override {
    ++payload_reads;
    if(const int delay=read_delay_ms.load())std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    while(hold_read)std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if(uncertain_read)return Status::InDoubt("simulated unresolved CQ");
    if(fail_read)return Status::Internal("simulated terminal READ failure");
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
  std::mutex control_mutex;
  rp::Service service{[this](std::uint64_t id,rp::RegionInfo* out){
    if(id!=info.id||pool.empty())return Status::NotFound("region");*out=info;return Status::OK();
  },[this](const rp::RegionInfo&){return std::make_unique<SharedBackend>(pool);},&control_mutex};
  std::atomic<std::uint16_t> drop{0};
  std::atomic<int> pool_allocations{0}, inspect_calls{0}, query_calls{0}, submit_calls{0}, sync_submit_calls{0}, connection_count{0};
  std::atomic<bool> stop{false}, fail_release{false}, disconnect_submit{false}, omit_timing{false};
  std::string address;
  int listener{-1};std::thread thread;
  std::set<std::uint16_t> ports;
  bool real_parser;
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
  explicit SharedServer(bool use_real_parser=false):real_parser(use_real_parser){
    listener=socket(AF_INET,SOCK_STREAM,0);Require(listener>=0,"socket");
    sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    Require(!bind(listener,reinterpret_cast<sockaddr*>(&addr),sizeof(addr)),"bind loopback");
    socklen_t size=sizeof(addr);Require(!getsockname(listener,reinterpret_cast<sockaddr*>(&addr),&size),"getsockname");
    address="127.0.0.1:"+std::to_string(ntohs(addr.sin_port));Require(!listen(listener,16),"listen");
    thread=std::thread([this]{kvaf::detail::ServeControlConnections(listener,[this](int fd){
      sockaddr_in peer{};socklen_t bytes=sizeof(peer);
      Require(!getpeername(fd,reinterpret_cast<sockaddr*>(&peer),&bytes),"peer address");
      {std::lock_guard<std::mutex> guard(control_mutex);
       if(ports.insert(peer.sin_port).second)++connection_count;}
      rp::Request req{};
      if(!p::ReadBytes(fd,&req.header,sizeof(req.header)))return Status::Internal("closed");
      if(req.header.magic==kvaf::peer::kProtocolMagic){std::lock_guard<std::mutex> guard(control_mutex);Peer(fd,req.header);}
      else{
        if(real_parser){service.Serve(fd,req.header);return Status::OK();}
        Require(p::ReadBytes(fd,reinterpret_cast<char*>(&req)+sizeof(req.header),rp::RequestBytes(req.header.command)-sizeof(req.header)),"read control body");Require(req.count<=rp::kMaxSpans,"bounded spans");std::vector<rp::Span> spans(req.count);
        Require(p::ReadBytes(fd,spans.data(),spans.size()*sizeof(rp::Span)),"read spans");
        if(req.header.command==unsigned(rp::Command::AddMemory)) {
          Require(req.header.bytes==offsetof(rp::Request,connection),"CPU extra MR frame excludes connection tail");
          Require(!req.connection.tp&&!req.connection.key_bytes,"connection was not transmitted with extra MR");
        }
        if(req.header.command==unsigned(rp::Command::InspectSource))++inspect_calls;
        if(req.header.command==unsigned(rp::Command::Query))++query_calls;
        if(req.header.command==unsigned(rp::Command::Submit))++submit_calls;
        if(req.header.command==unsigned(rp::Command::SubmitAndWait))++sync_submit_calls;
        if(req.header.command==unsigned(rp::Command::SubmitAndWait)&&disconnect_submit.exchange(false))
          shutdown(fd,SHUT_RDWR);
        auto response=service.Dispatch(req,spans);auto expected=req.header.command;
        if(omit_timing&&req.header.command==unsigned(rp::Command::SubmitAndWait))response.memory={};
        if(drop.compare_exchange_strong(expected,0))return Status::Internal("dropped reply");
        if(!p::WriteBytes(fd,&response,rp::ResponseBytes(req.header.command)))return Status::Internal("write reply");
      }
      return Status::OK();
    },[this]{return !stop.load();});});
  }
  ~SharedServer(){stop=true;shutdown(listener,SHUT_RDWR);if(thread.joinable())thread.join();close(listener);}
};
