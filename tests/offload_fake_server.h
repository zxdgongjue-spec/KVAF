#pragma once
#include "offload_fake_transport.h"
#include "remote_pull_service.h"
#include <map>
#include <set>
#include "runtime_service_loop.h"
#include "cache_region_service.h"
namespace rp = kvaf::remote_pull;
using kvaf::Status;
std::atomic<int> payload_reads{0}, scratch_reads{0}, checksum_calls{0}, leases{0};
std::atomic<int> read_delay_ms{0};
std::atomic<int> read_worker_instances{0}, active_read_workers{0};
struct ReadWorkerLifetime {
  ReadWorkerLifetime() { ++read_worker_instances; ++active_read_workers; }
  ~ReadWorkerLifetime() { --active_read_workers; }
};
std::atomic<bool> hold_read{false}, uncertain_read{false}, fail_read{false}, fail_checksum{false}, fail_select{false};
struct SharedBackend final : p::Backend {
  std::vector<std::uint8_t>& pool;
  p::MemoryDesc source{};
  p::ConnectionDesc connection{};
  std::map<std::uint64_t,p::MemoryDesc> memories;
  bool closed{};
  std::shared_ptr<void> region_lease;
  explicit SharedBackend(std::vector<std::uint8_t>& v, std::shared_ptr<void> lease = {})
      :pool(v),region_lease(std::move(lease)){++leases;}
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
    // A thread-local lifetime distinguishes reuse from a new thread that was
    // assigned the same OS thread ID, and verifies workers exit on Detach.
    static thread_local ReadWorkerLifetime lifetime;
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
  Status Close()override{if(!closed){--leases;closed=true;region_lease.reset();}return Status::OK();}
};
class SharedServer {
 public:
  std::vector<std::uint8_t> pool;
  rp::RegionInfo info{};
  std::mutex control_mutex;
  // Exercise the actual main Region state machine, grants and allocation leases.
  class Allocation final : public kvaf::execution::Allocation {
   public:
    Allocation(SharedServer& owner, const kvaf::execution::Request& q):owner_(owner) {
      Require(owner.pool.empty(),"only A owns one DDR allocation");
      ++owner.pool_allocations;
      owner.pool.resize(q.logical_bytes+q.auxiliary_bytes,0xA5);
      descriptor.base=reinterpret_cast<std::uint64_t>(owner.pool.data());
      descriptor.length=owner.pool.size();
      owner.info={q.epoch,0,descriptor.base,q.logical_bytes,q.slot_size,0,0,0,{}};
    }
    Status Close() override {
      if(owner_.fail_release.exchange(false)) return Status::CompletionUnknown("injected region release failure");
      owner_.pool.clear(); return Status::OK();
    }
    Status PrepareCpuPull(std::uint32_t candidate,std::uint32_t token,std::uint64_t bytes,
                          kvaf::peer::PeerRegion* out) override {
      ++owner_.pull_preparations;
      Require(bytes==owner_.info.bytes && token && token!=2,"logical DDR and independent CPU grant");
      owner_.info.candidate=candidate; owner_.info.token=token;
      owner_.info.eid[0]=candidate; owner_.info.eid[15]=2;
      *out=descriptor; out->length=bytes; out->token_value=token;
      std::copy_n(owner_.info.eid, out->eid.size(), out->eid.begin());
      if (owner_.stale_pull_eid) ++out->eid[15];
      return Status::OK();
    }
    std::unique_ptr<p::Backend> CreateCpuPullBackend(std::shared_ptr<void> lease) override {
      return std::make_unique<SharedBackend>(owner_.pool,std::move(lease));
    }
   private:
    SharedServer& owner_;
  };
  kvaf::execution::CacheRegionService regions{kvaf::execution::kUrma|kvaf::execution::kCpuPull,
    [this](const kvaf::execution::Request& q,std::unique_ptr<kvaf::execution::Allocation>* out) {
      *out=std::make_unique<Allocation>(*this,q); return Status::OK();
    }, [](const kvaf::execution::Request&)->std::unique_ptr<kvaf::execution::Session>{return {};}};
  rp::Service service{[this](std::uint64_t id,rp::RegionInfo* out){
    return regions.DescribeCpuPullRegion(id,out);
  },[this](const rp::RegionInfo& region){return regions.AcquireCpuPullBackend(region);},&control_mutex};
  std::atomic<std::uint16_t> drop{0};
  std::atomic<std::uint32_t> drop_execution{0};
  bool stale_pull_eid{false};
  std::atomic<int> pull_preparations{0}, describe_calls{0};
  std::atomic<int> control_calls{0}, pool_allocations{0}, inspect_calls{0}, query_calls{0}, submit_calls{0}, sync_submit_calls{0}, connection_count{0};
  std::atomic<bool> stop{false}, fail_release{false}, disconnect_submit{false}, lose_before_dispatch{false}, omit_timing{false};
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
  static void OkDecode(Status status) { if(!status.ok())throw std::runtime_error(status.message()); }
  void Peer(int fd,kvaf::peer::MessageHeader header){
    namespace peer=kvaf::peer;
    const auto cmd=static_cast<peer::Command>(header.command);
    if(cmd==peer::Command::kGetCacheCandidateCount){peer::CandidateCountResponse r{};r.candidate_count=2;r.device_count=2;Reply(fd,cmd,r);return;}
    if(cmd==peer::Command::kGetCandidateInfo){
      peer::CandidateEidRequest req{};
      Require(header.bytes == sizeof(req), "CPU Pull queries full candidate EIDs");
      Require(p::ReadBytes(fd, reinterpret_cast<char*>(&req)+sizeof(header), sizeof(req)-sizeof(header)), "read candidate query");
      peer::CandidateEidResponse record{};
      auto& r = record.info;
      r.header = peer::MakeHeader(cmd, sizeof(record));
      r.header.command = peer::ResponseCommand(cmd);
      r.candidate_id = r.device_id = req.info.candidate_id;
      std::snprintf(r.device_name,sizeof(r.device_name),"fake-%u",r.candidate_id);
      record.eid[0] = r.candidate_id;
      record.eid[15] = 2;
      Require(p::WriteBytes(fd, &record, sizeof(record)), "write candidate EID");
      return;
    }
    if(header.command==kvaf::execution::kCommand){
      Require(header.bytes>=sizeof(header)&&header.bytes<=16384,"bounded execution frame");
      std::vector<std::uint8_t> bytes(header.bytes-sizeof(header));
      Require(p::ReadBytes(fd,bytes.data(),bytes.size()),"execution body");
      kvaf::execution::Request q;OkDecode(kvaf::execution::Decode(bytes,&q));
      auto response=regions.Handle(q);
      if(q.op==kvaf::execution::Op::CreateRegion&&response.status.ok())info.id=response.region;
      auto expected=static_cast<std::uint32_t>(q.op);
      if(expected && drop_execution.compare_exchange_strong(expected,0)) {
        shutdown(fd,SHUT_RDWR);return;
      }
      bytes=kvaf::execution::Encode(response);
      auto reply=peer::MakeHeader(cmd,sizeof(header)+bytes.size());reply.command|=0x8000;
      Require(p::WriteBytes(fd,&reply,sizeof(reply))&&p::WriteBytes(fd,bytes.data(),bytes.size()),"execution reply");
      return;
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
        ++control_calls;
        if(req.header.command==unsigned(rp::Command::Describe))++describe_calls;
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
        if(req.header.command==unsigned(rp::Command::SubmitAndWait)&&lose_before_dispatch.exchange(false))
          return Status::Internal("lost request before admission");
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
