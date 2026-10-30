#include "hbm_read.h"
#include "hbm_read_protocol.h"
#include "hbm_exporter.h"
#include <atomic>
#include <future>
#include <mutex>
#include <poll.h>
#include <set>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace kvaf {
namespace hr=hbm_read;
struct HbmReadSource::Impl {
  explicit Impl(std::uint32_t d,std::string e):device(d),eid(std::move(e)){}
  std::uint32_t device;
  std::string eid,selected;
  mutable std::mutex mutex;
  ConstBufferView input{};
  HbmReadTicket ticket{};
  std::vector<npu_eid::NpuLocalEidCandidate> candidates;
  std::unique_ptr<detail::Exporter> exporter;
  std::array<offload::Endpoint,offload::kChannelCount> peers{};
  std::array<bool,offload::kChannelCount> bound{};
  std::set<std::uint64_t> retired;
  std::uint64_t client{},sequence{},checksum{},completed{};
  std::uint32_t candidate{};
  offload::Endpoint initiator{};
  Status open_result;
  hr::State state{hr::State::Registered};
  bool confirmed{},frozen{},closing{};
  HbmReadBatch batch;
  std::vector<hr::Span> submitted;
  int listener{-1};
  std::uint16_t port{};
  std::thread thread;
  std::atomic<bool> stop{false};

  Status Device() const {
    int current=-1;
    return aclrtGetDevice(&current)==ACL_SUCCESS && current==int(device)?Status::OK():
        Status::InvalidState("select HbmReadSource device before this operation");
  }
  Status Handle(const hr::Request& r,const std::vector<hr::Span>& spans,hr::Response* response) {
    if(closing) return Status::Busy("HBM source is shutting down");
    if(r.source_id!=ticket.source_id || r.epoch!=ticket.epoch || r.token!=ticket.token || !input.valid())
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
        if(open_result.ok()) { response->endpoint=exporter->descriptor; response->checksum=checksum; }
        return open_result;
      }
      if(frozen || r.candidate>=candidates.size()) return Status::Busy("cannot attach while HBM is frozen");
      if(retired.size()>=256) return Status::Busy("HBM connection history full; recreate source");
      client=r.client; candidate=r.candidate; initiator=r.endpoint; initiator.token=ticket.token;
      exporter=std::make_unique<detail::Exporter>();
      selected=npu_eid::EidToHex(candidates[candidate].eid);
      open_result=exporter->Open(input,selected,initiator,true);
      if(open_result.ok()) {
        std::array<std::uint8_t,64> expected{};
        auto n=std::min<std::size_t>(expected.size(),input.bytes);
        if(aclrtMemcpy(expected.data(),expected.size(),input.data,n,ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS)
          open_result=Status::Internal("copy HBM READ probe expectation");
        checksum=offload::Hash(expected.data(),n);
      }
      if(open_result.ok()) {response->endpoint=exporter->descriptor;response->checksum=checksum;}
      return open_result;
    }
    if(client!=r.client || !exporter) return Status::NotFound("HBM reader connection not found");
    if(command==hr::Command::Close) {
      if(frozen) return Status::Busy("release/cancel frozen HBM before detach");
      auto status=exporter->Close(); if(!status.ok()) return status;
      exporter.reset(); retired.insert(client); client=0; bound.fill(false); peers={};
      confirmed=false; selected.clear(); state=hr::State::Registered; return Status::OK();
    }
    if(!open_result.ok()) return open_result;
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
        r.sequence && r.sequence==completed) return Status::OK();
    if(!confirmed || !frozen || r.sequence!=sequence) return Status::InvalidState("HBM batch is stale or not ready");
    if(command==hr::Command::Acquire) {
      if(spans.size()!=batch.lengths.size()) return Status::InvalidArgument("HBM span count mismatch");
      for(size_t i=0;i<spans.size();++i)
        if(spans[i].source!=batch.source_addresses[i] || spans[i].bytes!=batch.lengths[i])
          return Status::InvalidArgument("HBM request differs from published ranges");
      if(state==hr::State::Reading) {
        if(submitted.size()!=spans.size() || std::memcmp(submitted.data(),spans.data(),spans.size()*sizeof(hr::Span)))
          return Status::InvalidArgument("READ retry changed destination");
        return Status::OK();
      }
      if(state!=hr::State::Ready) return Status::InvalidState("HBM batch already consumed");
      submitted=spans; state=hr::State::Reading; return Status::OK();
    }
    if(command==hr::Command::Complete || command==hr::Command::Cancel) {
      if(state==hr::State::Done) return Status::OK();
      if(state!=hr::State::Reading && !(command==hr::Command::Cancel && state==hr::State::Ready)) return Status::InvalidState("HBM batch was not acquired");
      // Cancel is only sent by a reader that has not launched any READ.
      state=hr::State::Done; completed=sequence; return Status::OK();
    }
    return Status::InvalidArgument("unknown HBM command");
  }
  void Serve(std::promise<Status> startup) {
    if(aclrtSetDevice(device)!=ACL_SUCCESS) { startup.set_value(Status::Internal("select B device in HBM service"));return; }
    startup.set_value(Status::OK());
    while(!stop.load()) {
      pollfd p{listener,POLLIN,0}; if(poll(&p,1,100)<=0) continue;
      int fd=accept(listener,nullptr,nullptr); if(fd<0) continue;
      offload::SetSocketTimeout(fd);
      hr::Request r{}; hr::Response response{};
      if(offload::ReadBytes(fd,&r,sizeof(r)) && r.header.magic==hr::kMagic && r.header.version==hr::kVersion &&
          r.count<=offload::kMaxSpans && r.header.bytes==sizeof(r)+r.count*sizeof(hr::Span)) {
        std::vector<hr::Span> spans(r.count);
        if(offload::ReadBytes(fd,spans.data(),spans.size()*sizeof(hr::Span))) {
          response.header={hr::kMagic,hr::kVersion,std::uint16_t(r.header.command|0x8000U),sizeof(response)};
          response.source_id=r.source_id; response.epoch=r.epoch; response.client=r.client; response.sequence=r.sequence;
          Status s;
          { std::lock_guard<std::mutex> lock(mutex); s=Handle(r,spans,&response); response.state=unsigned(state); }
          response.code=unsigned(s.code()); std::snprintf(response.detail,sizeof(response.detail),"%s",s.message().c_str());
          offload::WriteBytes(fd,&response,sizeof(response));
        }
      }
      close(fd);
    }
  }
};
HbmReadSource::HbmReadSource(std::uint32_t d,std::string e):impl_(new Impl(d,std::move(e))){}
HbmReadSource::~HbmReadSource(){if(!Shutdown().ok()) (void)impl_.release();}
Status HbmReadSource::Register(ConstBufferView input,void* stream,HbmReadTicket* out) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto s=impl_->Device(); if(!s.ok()) return s;
  if(impl_->input.valid()) return Status::InvalidState("HBM source already registered");
  auto base=reinterpret_cast<std::uintptr_t>(input.data);
  if(!out || !input.valid() || input.device!=DeviceKind::kNpu || input.bytes>UINT64_MAX-base)
    return Status::InvalidArgument("invalid HBM source buffer");
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
  impl_->input=input;impl_->closing=false;impl_->sequence=impl_->completed=0;
  impl_->ticket={detail::Nonce(),detail::Nonce(),input.bytes,std::uint32_t(detail::Nonce())|1U};
  *out=impl_->ticket; return Status::OK();
}
Status HbmReadSource::Listen(const std::string& address) {
  if(impl_->listener>=0) return Status::InvalidState("HBM source already listening");
  if(!impl_->input.valid()) return Status::InvalidState("register HBM before listen");
  auto s=hr::Listen(address,&impl_->listener,&impl_->port);if(!s.ok())return s;
  impl_->stop=false;std::promise<Status> promise;auto future=promise.get_future();
  impl_->thread=std::thread([p=impl_.get(),promise=std::move(promise)]()mutable{p->Serve(std::move(promise));});
  s=future.get();
  if(!s.ok()){impl_->thread.join();close(impl_->listener);impl_->listener=-1;}
  return s;
}
Status HbmReadSource::Ready(const std::vector<std::uint64_t>& offsets,const std::vector<std::uint64_t>& lengths,
                           void* stream,HbmReadBatch* out) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto s=impl_->Device();if(!s.ok())return s;
  if(!out || !impl_->confirmed || impl_->frozen) return Status::Busy("attach reader and release previous batch before Ready");
  if(offsets.empty() || offsets.size()!=lengths.size() || offsets.size()>offload::kMaxSpans || impl_->sequence==UINT64_MAX)
    return Status::InvalidArgument("invalid raw HBM ranges");
  for(size_t i=0;i<offsets.size();++i)
    if(!offload::Contains(impl_->input.bytes,offsets[i],lengths[i]))return Status::InvalidArgument("source outside registered HBM");
  s=detail::Exporter::Ready(stream);if(!s.ok())return s;
  auto& b=impl_->batch;b={impl_->ticket.source_id,impl_->ticket.epoch,++impl_->sequence,{},lengths};
  for(auto offset:offsets)b.source_addresses.push_back(impl_->exporter->descriptor.base+offset);
  impl_->frozen=true;impl_->state=hr::State::Ready;impl_->submitted.clear();*out=b;return Status::OK();
}
Status HbmReadSource::ReleaseBatch(){
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if(!impl_->frozen)return Status::OK();
  if(impl_->state!=hr::State::Done)return Status::Busy("A has not acknowledged READ completion; retain HBM");
  impl_->frozen=false;impl_->state=hr::State::Connected;return Status::OK();
}
Status HbmReadSource::CancelBatch(){
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if(!impl_->frozen)return Status::OK();
  if(impl_->state!=hr::State::Ready)return Status::Busy("READ acquired; cannot cancel producer buffer");
  impl_->frozen=false;impl_->state=hr::State::Connected;return Status::OK();
}
Status HbmReadSource::Shutdown(){
  {std::lock_guard<std::mutex> lock(impl_->mutex);
   if(impl_->client || impl_->frozen)return Status::Busy("detach A and release HBM batch before shutdown");
   impl_->closing=true;}
  impl_->stop=true;if(impl_->thread.joinable())impl_->thread.join();
  if(impl_->listener>=0){close(impl_->listener);impl_->listener=-1;}
  impl_->input={};impl_->retired.clear();impl_->ticket={};return Status::OK();
}
std::uint16_t HbmReadSource::port()const{return impl_->port;}
bool HbmReadSource::frozen()const{std::lock_guard<std::mutex>l(impl_->mutex);return impl_->frozen || (impl_->client && !impl_->confirmed);}
bool HbmReadSource::connected()const{std::lock_guard<std::mutex>l(impl_->mutex);return impl_->confirmed;}
std::uint32_t HbmReadSource::device_id()const{return impl_->device;}
std::string HbmReadSource::selected_npu_eid()const{std::lock_guard<std::mutex>l(impl_->mutex);return impl_->selected;}
}
