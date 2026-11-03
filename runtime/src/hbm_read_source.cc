#include "hbm_read.h"
#include "hbm_read_protocol.h"
#include "hbm_exporter.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
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
  HbmReceiverTicket receiver;
  std::mutex socket_mutex;
  int socket{-1};
  bool admission_sent{}, peer_retired{};
  std::thread thread;
  std::atomic<bool> stop{false};

  Status Device() const {
    int current=-1;
    return aclrtGetDevice(&current)==ACL_SUCCESS && current==int(device)?Status::OK():
        Status::InvalidState("select NpuPullHbmSource device before this operation");
  }
  Status Handle(const hr::Request& r,hr::Response* response) {
    if(closing && r.header.command!=unsigned(hr::Command::Close) && r.header.command!=unsigned(hr::Command::Retire)) return Status::Busy("HBM source is shutting down");
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
    if(command==hr::Command::Retire && retired.count(r.client)) {peer_retired=true;return Status::OK();}
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
      peer_retired=false;client=r.client; candidate=r.candidate; initiator=r.endpoint; initiator.token=ticket.token;
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
    if(command==hr::Command::Complete || command==hr::Command::Cancel) {
      if(state==hr::State::Done) return Status::OK();
      if(state!=hr::State::Ready) return Status::InvalidState("HBM batch is not published");
      // Reader serializes READ/cancel and records this sequence before replying.
      // Producer cannot revoke a published token without reader confirmation.
      state=hr::State::Done; completed=sequence; completed_mr=mr_id; return Status::OK();
    }
    return Status::InvalidArgument("unknown HBM command");
  }
  void Serve(std::promise<Status> startup) {
    bool reported=false, first=true;
    const auto startup_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    auto report=[&](const Status& status){if(!reported){startup.set_value(status);reported=true;}};
    if(aclrtSetDevice(device)!=ACL_SUCCESS) {report(Status::Internal("select producer device in HBM control"));return;}
    while(!stop.load()) {
      int fd=hr::Connect(receiver.control_address);
      if(fd>=0) {
        {std::lock_guard<std::mutex> lock(socket_mutex);socket=fd;}
        auto status=hr::SendHello(fd,first?hr::Admission::Join:hr::Admission::Resume,receiver,ticket);
        if(status.ok()) {std::lock_guard<std::mutex> lock(mutex);admission_sent=true;first=false;}
        while(status.ok() && !stop.load()) {
          pollfd p{fd,POLLIN,0};const int n=poll(&p,1,100);
          if(n==0) {
            // A silent receiver must not keep register blocked forever. Keep
            // serving the same connection and retaining storage: a timeout is
            // not evidence that the peer did not register or start its probe.
            if(std::chrono::steady_clock::now()>=startup_deadline)
              report(Status::InDoubt("HBM registration timed out; retain source and retry shutdown"));
            continue;
          }
          peer::MessageHeader h{};
          if(n<0 || !(p.revents&POLLIN) || !offload::ReadBytes(fd,&h,sizeof(h)))break;
          if(h.command==unsigned(hr::Admission::Result)) {
            status=hr::ReadResult(fd,h);
            if(!status.ok()) {
              std::lock_guard<std::mutex> lock(mutex);
              if(status.code()==StatusCode::kNotFound && !client && !retired.empty()){peer_retired=true;stop=true;}
              if(status.code()!=StatusCode::kInDoubt && !client && retired.empty()) {
                // Definitive rejection before any exporter existed (for example
                // a stale receiver epoch). There is no remote lease to retain.
                admission_sent=false;stop=true;
              }
              report(status);break;
            }
            report(status);continue;
          }
          hr::Request r{};r.header=h;
          if(h.magic!=hr::kMagic || h.version!=hr::kVersion || h.bytes!=sizeof(r) ||
             !offload::ReadBytes(fd,reinterpret_cast<char*>(&r)+sizeof(h),sizeof(r)-sizeof(h)) || r.count)break;
          hr::Response response{};
          response.header={hr::kMagic,hr::kVersion,std::uint16_t(h.command|0x8000U),sizeof(response)};
          response.source_id=r.source_id;response.epoch=r.epoch;response.client=r.client;
          response.sequence=r.sequence;response.mr_id=r.mr_id;
          Status result;
          {std::lock_guard<std::mutex> lock(mutex);result=Handle(r,&response);response.state=unsigned(state);}
          response.code=unsigned(result.code());std::snprintf(response.detail,sizeof(response.detail),"%s",result.message().c_str());
          if(!offload::WriteBytes(fd,&response,sizeof(response)))break;
        }
        {std::lock_guard<std::mutex> lock(socket_mutex);socket=-1;close(fd);}
      }
      report(Status::InDoubt("HBM registration acknowledgement unavailable; retain source and retry shutdown"));
      {std::lock_guard<std::mutex> lock(mutex);if(first)break;}
      // Only TCP reconnects. Source, peer, MR and published batch identities
      // remain unchanged. The receiver refuses a different instance epoch.
      for(unsigned i=0;i<10 && !stop.load();++i)std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    report(Status::InDoubt("HBM registration interrupted"));
  }

};
NpuPullHbmSource::NpuPullHbmSource(std::uint32_t d,std::string e):impl_(new Impl(d,std::move(e))){}
NpuPullHbmSource::~NpuPullHbmSource(){if(!Shutdown().ok()) (void)impl_.release();}
Status NpuPullHbmSource::Register(ConstBufferView input,const HbmReceiverTicket& receiver,void* stream) {
  return Register(std::vector<ConstBufferView>{input},receiver,stream);
}
Status NpuPullHbmSource::Register(const std::vector<ConstBufferView>& inputs,const HbmReceiverTicket& receiver,void* stream) {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  auto s=impl_->Device(); if(!s.ok()) return s;
  if(!impl_->inputs.empty()) return Status::InvalidState("HBM source already registered");
  if(!receiver.valid())return Status::InvalidArgument("invalid HBM receiver ticket");
  if(inputs.empty() || inputs.size()>4096) return Status::InvalidArgument("HBM requires 1..4096 tensors");
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
  impl_->receiver=receiver;impl_->admission_sent=false;impl_->peer_retired=false;impl_->stop=false;
  std::promise<Status> promise;auto future=promise.get_future();
  impl_->thread=std::thread([p=impl_.get(),promise=std::move(promise)]()mutable{p->Serve(std::move(promise));});
  lock.unlock();
  return future.get();
}
Status NpuPullHbmSource::Ready(ReadyToken* out) { return Ready(ConstBufferView{},out); }
Status NpuPullHbmSource::Ready(ConstBufferView tensor,ReadyToken* out) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto status=impl_->Device();if(!status.ok())return status;
  if(!out || !impl_->confirmed || impl_->closing) return Status::InvalidState("complete registration before Ready; source must not be closing");
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
Status NpuPullHbmSource::FinishBatch(const ReadyToken& token){
  std::lock_guard<std::mutex> lock(impl_->mutex);
  {
    const auto* source=std::get_if<NpuPullSourceDesc>(&token.source);
    if(!source || !impl_->ticket.matches(*source) || !token.sequence || token.sequence!=impl_->sequence || token.mr_id!=impl_->mr_id)
      return Status::InvalidArgument("stale NPU Pull batch identity");
  }
  if(!impl_->frozen)return Status::OK();
  if(impl_->state!=hr::State::Done)
    return Status::Busy("published HBM cannot be revoked locally; reader must complete or cancel_raw_addr first");
  impl_->frozen=false;impl_->state=hr::State::Connected;return Status::OK();
}
Status NpuPullHbmSource::ReleaseBatch(const ReadyToken& t){return FinishBatch(t);}
Status NpuPullHbmSource::CancelBatch(const ReadyToken& t){return FinishBatch(t);}
Status NpuPullHbmSource::Shutdown(){
  bool detach=false;
  {std::lock_guard<std::mutex> lock(impl_->mutex);
   if(impl_->frozen)return Status::Busy("release the completed/cancelled HBM batch before shutdown");
   impl_->closing=true;
   detach=impl_->admission_sent && !impl_->peer_retired;}
  if(detach) {
    auto status=hr::Leave(impl_->receiver,impl_->ticket);
    // A reader which already closed the exporter may have shut its listener.
    // Otherwise EOF/timeout never proves remote DMA is quiescent.
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(!status.ok() && !impl_->peer_retired)return status;
    if(impl_->client)return Status::InDoubt("HBM reader did not close the exporter");
  }
  impl_->stop=true;
  {std::lock_guard<std::mutex> lock(impl_->socket_mutex);if(impl_->socket>=0)shutdown(impl_->socket,SHUT_RDWR);}
  if(impl_->thread.joinable())impl_->thread.join();
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->inputs.clear();impl_->retired.clear();impl_->ticket={};impl_->admission_sent=false;return Status::OK();
}
bool NpuPullHbmSource::frozen()const{std::lock_guard<std::mutex>l(impl_->mutex);return impl_->frozen || (impl_->client && !impl_->confirmed);}
bool NpuPullHbmSource::connected()const{std::lock_guard<std::mutex>l(impl_->mutex);return impl_->confirmed;}
std::uint32_t NpuPullHbmSource::device_id()const{return impl_->device;}
std::string NpuPullHbmSource::selected_npu_eid()const{std::lock_guard<std::mutex>l(impl_->mutex);return impl_->selected;}
}
