#include "hbm_read.h"
#include "hbm_read_protocol.h"
#include "hbm_exporter.h"
#include <algorithm>
#include <atomic>
#include <future>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <set>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace kvaf {
namespace hr=hbm_read;
struct NpuPullHbmSource::Impl {
  explicit Impl(std::uint32_t d,std::string e):device(d),eid(std::move(e)){}
  std::uint32_t device;
  std::string eid,selected;
  mutable std::mutex mutex;
  std::vector<ConstBufferView> inputs;
  std::vector<offload::Endpoint> memories;
  std::vector<std::uint64_t> checksums;
  HbmReadTicket ticket{};
  std::vector<npu_eid::NpuLocalEidCandidate> candidates;
  std::unique_ptr<detail::Exporter> exporter;
  std::array<offload::Endpoint,offload::kChannelCount> peers{};
  std::array<bool,offload::kChannelCount> bound{};
  std::set<std::uint64_t> retired;
  std::uint64_t client{},sequence{},completed{},mr_id{1},completed_mr{};
  std::uint32_t candidate{};
  offload::Endpoint initiator{};
  Status open_result;
  hr::State state{hr::State::Registered};
  bool confirmed{},frozen{},closing{};
  int listener{-1};
  std::uint16_t port{};
  std::thread thread;
  std::atomic<bool> stop{false};

  Status Device() const {
    int current=-1;
    return aclrtGetDevice(&current)==ACL_SUCCESS && current==int(device)?Status::OK():
        Status::InvalidState("select NpuPullHbmSource device before this operation");
  }
  Status Handle(const hr::Request& r,const std::vector<hr::Span>& spans,hr::Response* response) {
    if(closing) return Status::Busy("HBM source is shutting down");
    if(r.source_id!=ticket.source_id || r.epoch!=ticket.epoch || r.token!=ticket.token || inputs.empty())
      return Status::NotFound("HBM ticket is stale or belongs to another source");
    auto command=static_cast<hr::Command>(r.header.command);
    if(command==hr::Command::Discover) {
      response->candidates=candidates.size();
      if(r.candidate>=candidates.size()) return Status::NotFound("HBM EID candidate not found");
      std::memcpy(response->endpoint.eid,candidates[r.candidate].eid.data(),16);
      return Status::OK();
    }
    if(!r.client) return Status::InvalidArgument("missing HBM reader identity");
    if(command==hr::Command::Close && retired.count(r.client)) return Status::OK();
    if(command==hr::Command::Open) {
      if(retired.count(r.client)) return Status::InvalidState("reader connection already retired");
      if(client && client!=r.client) return Status::Busy("HBM already has a reader");
      if(client) {
        if(candidate!=r.candidate || std::memcmp(initiator.eid,r.endpoint.eid,16))
          return Status::InvalidArgument("reader retry changed its EID pair");
        if(open_result.ok()) { response->endpoint=memories.front(); response->checksum=checksums.front(); response->mr_count=inputs.size(); }
        return open_result;
      }
      if(frozen || r.candidate>=candidates.size()) return Status::Busy("cannot attach while HBM is frozen");
      if(retired.size()>=256) return Status::Busy("HBM connection history full; recreate source");
      client=r.client; candidate=r.candidate; initiator=r.endpoint; initiator.token=ticket.token;
      exporter=std::make_unique<detail::Exporter>();
      selected=npu_eid::EidToHex(candidates[candidate].eid);
      memories.resize(inputs.size()); checksums.resize(inputs.size());
      open_result=exporter->Open(inputs.front(),selected,initiator,true);
      if(open_result.ok()) memories.front()=exporter->descriptor;
      for(std::size_t i=1;open_result.ok() && i<inputs.size();++i)
        open_result=exporter->AddMemory(inputs[i],&memories[i]);
      for(std::size_t i=0;open_result.ok() && i<inputs.size();++i) {
        const auto& input=inputs[i];
        std::array<std::uint8_t,64> expected{};
        auto n=std::min<std::size_t>(expected.size(),input.bytes);
        if(aclrtMemcpy(expected.data(),expected.size(),input.data,n,ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS)
          open_result=Status::Internal("copy HBM READ probe expectation");
        checksums[i]=offload::Hash(expected.data(),n);
      }
      if(open_result.ok()) {response->endpoint=memories.front();response->checksum=checksums.front();response->mr_count=inputs.size();}
      return open_result;
    }
    if(client!=r.client || !exporter) return Status::NotFound("HBM reader connection not found");
    if(command==hr::Command::Close) {
      if(frozen) return Status::Busy("release/cancel frozen HBM before detach");
      auto status=exporter->Close(); if(!status.ok()) return status;
      exporter.reset(); memories.clear(); checksums.clear(); retired.insert(client); client=0; bound.fill(false); peers={};
      confirmed=false; selected.clear(); state=hr::State::Registered; return Status::OK();
    }
    if(!open_result.ok()) return open_result;
    if(command==hr::Command::Memory) {
      if(!r.mr_id || r.mr_id>memories.size()) return Status::InvalidArgument("unknown HBM MR");
      response->endpoint=memories[r.mr_id-1];response->checksum=checksums[r.mr_id-1];
      response->mr_count=memories.size();return Status::OK();
    }
    if(command==hr::Command::Channel) {
      if(confirmed || r.channel>=bound.size() || std::memcmp(r.endpoint.eid,initiator.eid,16))
        return Status::InvalidArgument("invalid HBM target channel");
      if(bound[r.channel]) {
        if(std::memcmp(&peers[r.channel],&r.endpoint,sizeof(r.endpoint)))
          return Status::InvalidArgument("channel retry changed descriptor");
      } else {
        // Preserve a partial failed import: do not retry into the same handle.
        if(peers[r.channel].key_bytes) return Status::InDoubt("partial target import; detach before retry");
        peers[r.channel]=r.endpoint;
        auto s=exporter->BindNpuChannel(r.channel,r.endpoint); if(!s.ok()) return s;
        bound[r.channel]=true;
      }
      response->endpoint=exporter->NpuChannelDescriptor(r.channel); return Status::OK();
    }
    if(command==hr::Command::Confirm) {
      if(!bound[0]) return Status::InvalidState("HBM has no bound channel");
      confirmed=true; if(state==hr::State::Registered) state=hr::State::Connected;
      return Status::OK();
    }
    if((command==hr::Command::Complete || command==hr::Command::Cancel) &&
        r.sequence && r.sequence==completed)
      return r.mr_id==completed_mr ? Status::OK() : Status::InvalidArgument("completed HBM batch MR mismatch");
    if(!confirmed || !frozen || r.sequence!=sequence || r.mr_id!=mr_id) return Status::InvalidState("HBM batch is stale or not ready");
    if(command==hr::Command::AcquireRaw) {
      if(!spans.empty()) return Status::InvalidArgument("NPU Pull control carries identity only");
      if(state==hr::State::Reading) return Status::OK();
      if(state!=hr::State::Ready) return Status::InvalidState("HBM batch already consumed");
      state=hr::State::Reading; return Status::OK();
    }
    if(command==hr::Command::Complete || command==hr::Command::Cancel) {
      if(state==hr::State::Done) return Status::OK();
      if(state!=hr::State::Reading && !(command==hr::Command::Cancel && state==hr::State::Ready)) return Status::InvalidState("HBM batch was not acquired");
      // Cancel is only sent by a reader that has not launched any READ.
      state=hr::State::Done; completed=sequence; completed_mr=mr_id; return Status::OK();
    }
    return Status::InvalidArgument("unknown HBM command");
  }
  void Serve(std::promise<Status> startup) {
    if(aclrtSetDevice(device)!=ACL_SUCCESS) { startup.set_value(Status::Internal("select B device in HBM service"));return; }
    startup.set_value(Status::OK());
    std::vector<int> clients;
    std::vector<pollfd> polls;
    while(!stop.load()) {
      polls.clear();polls.push_back({listener,POLLIN,0});
      for(int fd:clients)polls.push_back({fd,POLLIN,0});
      if(poll(polls.data(),polls.size(),100)<=0)continue;
      // Process existing clients before accepting another. Idle persistent
      // connections must not block discovery, channel setup or other readers.
      for(std::size_t i=clients.size();i>0;--i) {
        if(!polls[i].revents)continue;
        int fd=clients[i-1];bool keep=false;
        hr::Request r{};hr::Response response{};
        if((polls[i].revents&POLLIN) && offload::ReadBytes(fd,&r,sizeof(r)) &&
            r.header.magic==hr::kMagic && r.header.version==hr::kVersion &&
            r.count<=offload::kMaxSpans && r.header.bytes==sizeof(r)+r.count*sizeof(hr::Span)) {
          std::vector<hr::Span> spans(r.count);
          if(offload::ReadBytes(fd,spans.data(),spans.size()*sizeof(hr::Span))) {
            response.header={hr::kMagic,hr::kVersion,std::uint16_t(r.header.command|0x8000U),sizeof(response)};
            response.source_id=r.source_id;response.epoch=r.epoch;response.client=r.client;response.sequence=r.sequence;
            response.mr_id=r.mr_id;
            response.reserved=1U; // Persistent exchanges supported.
            Status s;
            {std::lock_guard<std::mutex> lock(mutex);s=Handle(r,spans,&response);response.state=unsigned(state);}
            response.code=unsigned(s.code());std::snprintf(response.detail,sizeof(response.detail),"%s",s.message().c_str());
            keep=offload::WriteBytes(fd,&response,sizeof(response));
          }
        }
        if(!keep){close(fd);clients.erase(clients.begin()+i-1);}
      }
      if(polls[0].revents&POLLIN) {
        int fd=accept(listener,nullptr,nullptr);if(fd<0)continue;
        if(clients.size()>=64){close(fd);continue;}
        offload::SetSocketTimeout(fd);
        int yes=1;setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&yes,sizeof(yes));
        clients.push_back(fd);
      }
    }
    for(int fd:clients)close(fd);
  }
};
NpuPullHbmSource::NpuPullHbmSource(std::uint32_t d,std::string e):impl_(new Impl(d,std::move(e))){}
NpuPullHbmSource::~NpuPullHbmSource(){if(!Shutdown().ok()) (void)impl_.release();}
Status NpuPullHbmSource::Register(ConstBufferView input,void* stream,HbmReadTicket* out) {
  return Register(std::vector<ConstBufferView>{input},stream,out);
}
Status NpuPullHbmSource::Register(const std::vector<ConstBufferView>& inputs,void* stream,HbmReadTicket* out) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto s=impl_->Device(); if(!s.ok()) return s;
  if(!impl_->inputs.empty()) return Status::InvalidState("HBM source already registered");
  if(!out || inputs.empty() || inputs.size()>4096) return Status::InvalidArgument("HBM requires 1..4096 tensors");
  std::vector<std::pair<std::uintptr_t,std::uintptr_t>> ranges;
  for(const auto& input:inputs) {
    const auto base=reinterpret_cast<std::uintptr_t>(input.data);
    if(!input.valid() || input.device!=DeviceKind::kNpu || input.bytes>UINT64_MAX-base)
      return Status::InvalidArgument("invalid HBM source buffer");
    ranges.emplace_back(base,base+input.bytes);
  }
  std::sort(ranges.begin(),ranges.end());
  for(std::size_t i=1;i<ranges.size();++i) if(ranges[i].first<ranges[i-1].second)
    return Status::InvalidArgument("HBM tensors must not overlap");
  s=detail::Exporter::Ready(stream); if(!s.ok()) return s;
  impl_->candidates.clear();
  if(impl_->eid.empty()) {
    std::uint32_t physical{};std::string error;
    if(!direct_hccp::GetCurrentNpuPhysicalDeviceId(&physical)) return Status::Internal("resolve B physical device");
    if(!npu_eid::DiscoverPhysicalNpuEids(physical,&impl_->candidates,&error)) return Status::Internal(error);
  } else {
    npu_eid::NpuLocalEidCandidate c{};
    if(!detail::ParseEid(impl_->eid,c.eid.data())) return Status::InvalidArgument("invalid B EID");
    impl_->candidates.push_back(c);
  }
  impl_->inputs=inputs;impl_->closing=false;impl_->sequence=impl_->completed=impl_->completed_mr=0;impl_->mr_id=1;
  impl_->ticket={detail::Nonce(),detail::Nonce(),inputs.front().bytes,std::uint32_t(detail::Nonce())|1U};
  if(inputs.size()>1) for(const auto& input:inputs) impl_->ticket.mr_bytes.push_back(input.bytes);
  *out=impl_->ticket; return Status::OK();
}
Status NpuPullHbmSource::Listen(const std::string& address) {
  if(impl_->listener>=0) return Status::InvalidState("HBM source already listening");
  if(impl_->inputs.empty()) return Status::InvalidState("register HBM before listen");
  auto s=hr::Listen(address,&impl_->listener,&impl_->port);if(!s.ok())return s;
  impl_->stop=false;std::promise<Status> promise;auto future=promise.get_future();
  impl_->thread=std::thread([p=impl_.get(),promise=std::move(promise)]()mutable{p->Serve(std::move(promise));});
  s=future.get();
  if(!s.ok()){impl_->thread.join();close(impl_->listener);impl_->listener=-1;}
  return s;
}
Status NpuPullHbmSource::Ready(ReadyToken* out) { return Ready(ConstBufferView{},out); }
Status NpuPullHbmSource::Ready(ConstBufferView tensor,ReadyToken* out) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto status=impl_->Device();if(!status.ok())return status;
  if(!out || !impl_->confirmed) return Status::InvalidState("attach reader before Ready");
  std::uint64_t mr=1;
  if(!tensor.data && !tensor.bytes) {
    if(impl_->inputs.size()!=1) return Status::InvalidArgument("select a registered tensor for multi-MR Ready");
  } else {
    auto it=std::find_if(impl_->inputs.begin(),impl_->inputs.end(),[&](const auto& input){
      return input.data==tensor.data && input.bytes==tensor.bytes && input.device==tensor.device;});
    if(it==impl_->inputs.end()) return Status::InvalidArgument("tensor is not registered in this HBM source");
    mr=std::distance(impl_->inputs.begin(),it)+1;
  }
  if(impl_->frozen && impl_->mr_id!=mr) return Status::Busy("release current batch before selecting another MR");
  if(!impl_->frozen) {
    impl_->mr_id=mr;
    if(impl_->sequence==UINT64_MAX) return Status::InvalidState("HBM sequence exhausted");
    ++impl_->sequence;impl_->frozen=true;impl_->state=hr::State::Ready;
  }
  *out={impl_->ticket.source(),impl_->sequence,impl_->mr_id};
  return Status::OK();
}
Status NpuPullHbmSource::FinishBatch(const ReadyToken& token,bool cancel){
  std::lock_guard<std::mutex> lock(impl_->mutex);
  {
    const auto* source=std::get_if<NpuPullSourceDesc>(&token.source);
    if(!source || !impl_->ticket.matches(*source) || !token.sequence || token.sequence!=impl_->sequence || token.mr_id!=impl_->mr_id)
      return Status::InvalidArgument("stale NPU Pull batch identity");
  }
  if(!impl_->frozen)return Status::OK();
  if(impl_->state!=(cancel?hr::State::Ready:hr::State::Done))
    return Status::Busy("HBM batch cannot be released/cancelled in its current state");
  impl_->frozen=false;impl_->state=hr::State::Connected;return Status::OK();
}
Status NpuPullHbmSource::ReleaseBatch(const ReadyToken& t){return FinishBatch(t,false);}
Status NpuPullHbmSource::CancelBatch(const ReadyToken& t){return FinishBatch(t,true);}
Status NpuPullHbmSource::Shutdown(){
  {std::lock_guard<std::mutex> lock(impl_->mutex);
   if(impl_->client || impl_->frozen)return Status::Busy("detach A and release HBM batch before shutdown");
   impl_->closing=true;}
  impl_->stop=true;if(impl_->thread.joinable())impl_->thread.join();
  if(impl_->listener>=0){close(impl_->listener);impl_->listener=-1;}
  impl_->inputs.clear();impl_->retired.clear();impl_->ticket={};return Status::OK();
}
std::uint16_t NpuPullHbmSource::port()const{return impl_->port;}
bool NpuPullHbmSource::frozen()const{std::lock_guard<std::mutex>l(impl_->mutex);return impl_->frozen || (impl_->client && !impl_->confirmed);}
bool NpuPullHbmSource::connected()const{std::lock_guard<std::mutex>l(impl_->mutex);return impl_->confirmed;}
std::uint32_t NpuPullHbmSource::device_id()const{return impl_->device;}
std::string NpuPullHbmSource::selected_npu_eid()const{std::lock_guard<std::mutex>l(impl_->mutex);return impl_->selected;}
}
