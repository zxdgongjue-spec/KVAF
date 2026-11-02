#pragma once
#include "offload_fake_transport.h"
#include "remote_pull_service.h"
namespace rp = kvaf::remote_pull;
using kvaf::Status;
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
