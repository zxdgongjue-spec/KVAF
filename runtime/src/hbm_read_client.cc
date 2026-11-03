#include "hbm_read_client.h"
#include "hbm_read_protocol.h"
#include "hbm_exporter.h"
#include "kvaf_aiv_batch_get_raw_addr.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <unistd.h>
#include <atomic>
#include <future>
#include <thread>
#include <mutex>
#include <poll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

namespace kvaf {
namespace hr=hbm_read;
namespace {
using ProfileClock = std::chrono::steady_clock;
struct ProfileTimer {
  double* target;
  ProfileClock::time_point begin;
  explicit ProfileTimer(double* out):target(out),begin(out?ProfileClock::now():ProfileClock::time_point{}){}
  ~ProfileTimer(){if(target)*target+=std::chrono::duration<double,std::micro>(ProfileClock::now()-begin).count();}
};
template<class F> auto Timed(double* target,F operation){ProfileTimer timer(target);return operation();}
struct DeviceBuffer {
  void* data{}; std::size_t bytes{};
  Status Allocate(std::size_t n) {
    if(data) return Status::InvalidState("raw READ buffer already allocated");
    if(aclrtMalloc(&data,n,ACL_MEM_MALLOC_NORMAL_ONLY)!=ACL_SUCCESS || !data)
      return Status::OutOfMemory("allocate raw READ metadata");
    bytes=n;return Status::OK();
  }
  Status Free() {
    if(data && aclrtFree(data)!=ACL_SUCCESS) return Status::InDoubt("free raw READ device buffer");
    data=nullptr;bytes=0;return Status::OK();
  }
};
class ReadLink final:public peer::PeerLink {
 public:
  ReadLink(std::shared_ptr<hr::ControlChannel> channel,HbmReadTicket t):PeerLink("reverse-hbm"),control(std::move(channel)),ticket(t),client(detail::Nonce()){}
  std::shared_ptr<hr::ControlChannel> control;
  HbmReadTicket ticket;
  std::uint64_t client;
  std::vector<offload::MemoryDesc> memories;
  std::vector<std::uint64_t> checksums;
  offload::ConnectionDesc connection{};
  std::uint32_t channel{};
  bool opened{};
  Status Call(hr::Command command,hr::Response* response,std::uint64_t sequence=0,std::uint64_t mr_id=1) {
    auto r=hr::Make(command,ticket,client);r.sequence=sequence;r.mr_id=mr_id;return control->Exchange(r,response);
  }
  Status LookupRegion(std::uint64_t base,peer::PeerRegion* out)const override {
    auto it=std::find_if(memories.begin(),memories.end(),[&](const auto& e){return e.base==base;});
    if(!out || it==memories.end() || it->segment_bytes!=48) return Status::InvalidArgument("invalid imported B HBM");
    const auto& endpoint=*it;
    *out={};out->base=endpoint.base;out->length=endpoint.bytes;out->token_value=endpoint.token;
    std::memcpy(&out->token_id,endpoint.segment+44,4);
    out->segment_key_bytes=endpoint.segment_bytes;std::memcpy(out->segment_key.data(),endpoint.segment,endpoint.segment_bytes);
    std::memcpy(out->eid.data(),connection.eid,16);
    out->has_memory_eid=true;std::memcpy(out->memory_eid.data(),endpoint.segment,16);
    return Status::OK();
  }
  Status CreateChannel(std::uint64_t base,const std::uint8_t* key,std::uint32_t key_bytes,
      const std::uint8_t* local_eid,std::uint32_t token,std::uint64_t tp,std::uint32_t psn,
      std::array<std::uint8_t,peer::kQpKeyBytes>* peer_key,std::uint32_t* peer_bytes,
      std::uint64_t* peer_tp,std::uint32_t* peer_psn)override {
    if(memories.empty() || base!=memories.front().base || channel>=offload::kChannelCount || key_bytes!=32 || token!=ticket.token)
      return Status::InvalidArgument("invalid NPU raw READ channel");
    auto r=hr::Make(hr::Command::Channel,ticket,client);r.channel=channel;
    r.connection.tp=tp;r.connection.psn=psn;r.connection.token=token;r.connection.key_bytes=key_bytes;
    std::memcpy(r.connection.eid,local_eid,16);std::memcpy(r.connection.key[0],key,key_bytes);
    hr::Response response{};auto s=control->Exchange(r,&response);if(!s.ok())return s;
    if(response.connection.key_bytes!=32 || response.connection.token!=token) return Status::Internal("invalid B target descriptor");
    *peer_bytes=response.connection.key_bytes;std::memcpy(peer_key->data(),response.connection.key[channel],*peer_bytes);
    *peer_tp=response.connection.tp;*peer_psn=response.connection.psn;++channel;return Status::OK();
  }
};
struct Engine {
  KvafBatchGetTransport* transport{};
  KvafBatchGetLocalBuffer* local{};
  DeviceBuffer descriptors,statuses,queues,scratch;
  std::vector<KvafRawReadCoreStatus> host_statuses;
  std::uint32_t cores,jettys,budget;
  std::uint64_t mr_id{1};
  bool uncertain{},kernel_finished{},cache_output{};
  aclrtStream probe_stream{};
  aclrtEvent profile_start{},profile_end{};
  Engine(std::uint32_t c,std::uint32_t j,std::uint32_t b):cores(c),jettys(j),budget(b){}
  Status Cleanup() {
    if(uncertain)return Status::InDoubt("raw READ transport quarantined; source and output must remain alive");
    if(cache_output) {
      KvafBatchGetLocalBufferRelease(local);local=nullptr;
    } else if(KvafBatchGetLocalBufferDiscard(&local))return Status::InDoubt("raw READ output unregister pending");
    return descriptors.Free();
  }
  Status QuiesceProbe() {
    if (!uncertain) return Status::OK();
    if (!kernel_finished || KvafBatchGetTransportQuiesce(transport))
      return Status::InDoubt("probe transport could not be quiesced; retain B source and A scratch");
    uncertain=false;
    return Status::OK();
  }
  Status Close() {
    auto s=Cleanup();if(!s.ok())return s;
    for(auto* event:{&profile_start,&profile_end})if(*event) {
      if(aclrtDestroyEvent(*event)!=ACL_SUCCESS)return Status::InDoubt("destroy raw READ profiling event");
      *event=nullptr;
    }
    if(KvafBatchGetTransportClose(&transport))return Status::InDoubt("raw READ transport teardown pending");
    s=statuses.Free();if(!s.ok())return s;
    s=queues.Free();if(!s.ok())return s;
    if(probe_stream) {
      if(aclrtDestroyStream(probe_stream)!=ACL_SUCCESS)return Status::InDoubt("destroy raw probe stream");
      probe_stream=nullptr;
    }
    return scratch.Free();
  }
  Status RunProbe(const std::vector<hr::Span>& spans,MutableBufferView output,void* stream) {
    if(!transport || !stream || uncertain || local || descriptors.data)
      return Status::InvalidState("raw READ engine is not idle");
    if(spans.empty() || spans.size()>KVAF_RAW_MAX_WQES)
      return Status::InvalidArgument("invalid raw probe request count");
    std::vector<std::uint64_t> src,dst;std::vector<std::uint32_t> sizes;
    for(const auto& span:spans) {
      if(!span.bytes || span.bytes>UINT32_MAX)
        return Status::InvalidArgument("raw probe length must fit one uint32 WQE");
      src.push_back(span.source);dst.push_back(span.destination);
      sizes.push_back(static_cast<std::uint32_t>(span.bytes));
    }
    const std::size_t n=src.size();
    auto s=descriptors.Allocate(n*20);if(!s.ok())return s;
    auto* d=static_cast<std::uint8_t*>(descriptors.data);
    if(aclrtMemcpy(d,n*20,src.data(),n*8,ACL_MEMCPY_HOST_TO_DEVICE)!=ACL_SUCCESS ||
        aclrtMemcpy(d+n*8,n*12,dst.data(),n*8,ACL_MEMCPY_HOST_TO_DEVICE)!=ACL_SUCCESS ||
        aclrtMemcpy(d+n*16,n*4,sizes.data(),n*4,ACL_MEMCPY_HOST_TO_DEVICE)!=ACL_SUCCESS)
      return Status::Internal("prepare raw READ descriptors/status");
    KvafExecuteBatchGetRawAddrKernelArgs args{};
    args.source_addresses=reinterpret_cast<std::uint64_t>(d);
    args.destination_addresses=reinterpret_cast<std::uint64_t>(d+n*8);
    args.lengths=reinterpret_cast<std::uint64_t>(d+n*16);
    args.request_count=n;
    return RunKernel(args,output,stream);
  }
  Status RunDevice(NpuInt64ArrayView src,NpuInt64ArrayView dst,NpuInt64ArrayView lengths,
                   MutableBufferView output,void* stream,HbmReadProfile* profile,bool validate_ranges) {
    if(!transport || !stream || uncertain || local || descriptors.data)
      return Status::InvalidState("raw READ engine is not idle");
    KvafExecuteBatchGetRawAddrKernelArgs args{};
    args.source_addresses=reinterpret_cast<std::uint64_t>(src.data);
    args.destination_addresses=reinterpret_cast<std::uint64_t>(dst.data);
    args.lengths=reinterpret_cast<std::uint64_t>(lengths.data);
    args.request_count=src.size;args.input_mode=KVAF_RAW_NPU_OFFSETS;
    args.validate_ranges=validate_ranges;
    cache_output=true;
    return RunKernel(args,output,stream,profile);
  }
  Status RunKernel(KvafExecuteBatchGetRawAddrKernelArgs args,MutableBufferView output,void* stream,
                   HbmReadProfile* profile=nullptr) {
    auto s=Timed(profile?&profile->buffers_us:nullptr,[&]() -> Status {
    if(!statuses.data) {
      auto allocated=statuses.Allocate(cores*sizeof(KvafRawReadCoreStatus));if(!allocated.ok())return allocated;
    }
    if(!queues.data) {
      auto allocated=queues.Allocate(cores*jettys*sizeof(KvafBatchGetQueueState));if(!allocated.ok())return allocated;
      if(aclrtMemset(queues.data,queues.bytes,0,queues.bytes)!=ACL_SUCCESS)return Status::Internal("clear raw READ queues");
    }
    // Completion/cancellation decisions must not observe a previous batch's
    // success record if this launch fails to write a status.
    if(aclrtMemset(statuses.data,statuses.bytes,0,statuses.bytes)!=ACL_SUCCESS)
      return Status::Internal("clear raw READ status");
    host_statuses.resize(cores);
    if(profile)for(auto* event:{&profile_start,&profile_end})
      if(!*event && aclrtCreateEventWithFlag(event,ACL_EVENT_TIME_LINE)!=ACL_SUCCESS)
        return Status::Internal("create raw READ timing event");
    return Status::OK();
    });
    if(!s.ok())return s;
    std::uint64_t entity{};
    if(Timed(profile?&profile->register_output_us:nullptr,[&]{
        return KvafBatchGetLocalBufferAcquireForRemote(transport,mr_id,output.data,output.bytes,&local,&entity);}))
      return Status::Internal("register A raw READ output");
    args.output=reinterpret_cast<std::uint64_t>(output.data);
    args.core_status=reinterpret_cast<std::uint64_t>(statuses.data);args.queue_state=reinterpret_cast<std::uint64_t>(queues.data);
    args.channel_entity=entity;args.block_count=cores;args.jettys_per_core=jettys;args.batch_get_max_inflight_total=budget;
    kernel_finished=false;
    uncertain=true; // Includes launch errors: never assume the device saw no work.
    if(profile)profile->event_error=Timed(&profile->event_record_us,[&]{
      return aclrtRecordEvent(profile_start,static_cast<aclrtStream>(stream));});
    if(Timed(profile?&profile->launch_us:nullptr,[&]{
        return KvafExecuteBatchGetRawAddrLaunch(&args,static_cast<aclrtStream>(stream));}))
      return Status::InDoubt("raw READ launch/stream completion uncertain");
    if(profile && !profile->event_error)profile->event_error=Timed(&profile->event_record_us,[&]{
      return aclrtRecordEvent(profile_end,static_cast<aclrtStream>(stream));});
    if(Timed(profile?&profile->synchronize_us:nullptr,[&]{
        return aclrtSynchronizeStream(static_cast<aclrtStream>(stream));})!=ACL_SUCCESS)
      return Status::InDoubt("raw READ launch/stream completion uncertain");
    kernel_finished=true;
    if(profile && !profile->event_error) {
      float ms{};
      profile->event_error=Timed(&profile->event_query_us,[&]{return aclrtEventElapsedTime(&ms,profile_start,profile_end);});
      profile->kernel_event_valid=!profile->event_error && std::isfinite(ms) && ms>=0;
      if(profile->kernel_event_valid)profile->kernel_event_us=static_cast<double>(ms)*1000;
    }
    ProfileTimer status_timer(profile?&profile->status_us:nullptr);
    auto& host=host_statuses;
    if(aclrtMemcpy(host.data(),statuses.bytes,statuses.data,statuses.bytes,ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS)
      return Status::InDoubt("cannot inspect raw READ completion");
    // Cancellation is safe only when EVERY core reports the same full-batch
    // validation failure and explicitly reports zero posted/processed WQEs.
    const auto& first=host.front().execution;
    bool rejected=args.input_mode==KVAF_RAW_NPU_OFFSETS &&
        (first.result==1 || first.result==4 || first.result==8);
    for(const auto& record:host)
      rejected=rejected && record.execution.magic==KVAF_AIV_BATCH_GET_STATUS_MAGIC &&
          record.execution.result==first.result && record.execution.failed_request_index==first.failed_request_index &&
          !record.total_wqes && !record.posted_wqes && !record.execution.processed_request_count;
    if(rejected) {
      uncertain=false;
      return Status::InvalidArgument("NPU raw READ metadata rejected before DMA: result="+std::to_string(first.result)+
                                     " span="+std::to_string(first.failed_request_index));
    }
    const auto n=host.front().total_wqes;
    if(!n || n!=args.request_count)
      return Status::InDoubt("invalid raw READ final WQE count");
    for(std::uint32_t c=0;c<cores;++c) {
      const auto& record=host[c];const auto& status=record.execution;
      const auto expected=n*(c+1)/cores-n*c/cores;
      if(status.magic!=KVAF_AIV_BATCH_GET_STATUS_MAGIC || status.result || record.total_wqes!=n ||
          status.processed_request_count!=expected || record.posted_wqes!=expected)
        return Status::InDoubt("raw READ CQ/kernel failure: core="+std::to_string(c)+" result="+std::to_string(status.result)+
                              " failed_request="+std::to_string(status.failed_request_index));
    }
    uncertain=false;return Status::OK();
  }
};
enum class Pending { None, Dma, Cleanup, Complete, Cancel };
struct Connection {
  std::unique_ptr<ReadLink> link;
  std::string local_eid;
  Engine engine;
  Pending pending{Pending::None};
  std::uint64_t sequence{},last_sequence{},mr_id{1};
  bool ready{};
  Status result;
  Connection(std::shared_ptr<hr::ControlChannel> a,const HbmReadTicket& t,const std::string& eid,std::uint32_t c,std::uint32_t j,std::uint32_t b):
      link(new ReadLink(a,t)),local_eid(eid),engine(c,j,b){}
  Status Resolve(HbmReadProfile* profile=nullptr) {
    if(pending==Pending::None)return Status::InvalidState("no pending raw READ");
    if(pending==Pending::Dma || engine.uncertain)return Status::InDoubt("raw READ DMA unresolved; retain A output and B source; no READ replay");
    auto s=Timed(profile?&profile->cleanup_us:nullptr,[&]{return engine.Cleanup();});if(!s.ok())return s;
    if(pending==Pending::Cleanup)pending=result.ok()?Pending::Complete:Pending::Cancel;
    hr::Response response{};
    s=Timed(profile?&profile->complete_us:nullptr,[&]{
      return link->Call(pending==Pending::Complete?hr::Command::Complete:hr::Command::Cancel,&response,sequence,mr_id);});
    if(!s.ok())return Status::InDoubt("raw READ acknowledgement pending: "+s.message());
    pending=Pending::None;last_sequence=sequence;return result;
  }
};
}
struct HbmReader::Impl {
  std::uint32_t device,cores,jettys,budget;
  std::string kernel;
  mutable std::recursive_mutex mutex;
  HbmReceiverTicket receiver;
  std::string local_eid;
  bool accepting{true};
  int listener{-1};
  std::atomic<bool> stop{false};
  std::thread thread;
  // Registration identity, never an application-maintained local connection ID.
  using SourceKey = std::pair<std::uint64_t, std::uint64_t>;
  std::map<SourceKey,std::unique_ptr<Connection>> connections;
  Status Device()const {
    int current=-1;return aclrtGetDevice(&current)==ACL_SUCCESS && current==int(device)?Status::OK():
        Status::InvalidState("select A device before HBM READ operation");
  }
};
HbmReader::HbmReader(std::uint32_t d,std::uint32_t c,std::uint32_t j,std::uint32_t b,std::string k):
    impl_(new Impl{d,c,j,b,std::move(k)}){}
HbmReader::~HbmReader(){if(!Shutdown().ok()){
  impl_->stop=true;if(impl_->thread.joinable())impl_->thread.join();
  (void)impl_.release(); // Retain unresolved hardware, but never a thread capturing a destroyed owner.
}}
Status HbmReader::Listen(const std::string& address,const std::string& advertised,const std::string& eid,HbmReceiverTicket* out) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  auto status=impl_->Device();if(!status.ok())return status;
  if(!out || impl_->listener>=0)return Status::InvalidState("HBM receiver already listening or missing result");
  if(!impl_->cores || !impl_->jettys || impl_->cores*std::uint64_t(impl_->jettys)>offload::kChannelCount ||
     (impl_->budget && impl_->budget<impl_->cores*2) || impl_->kernel.empty())
    return Status::InvalidArgument("configure raw kernel and 1..4 total Jettys before listen_hbm");
  if(!eid.empty()) {std::uint8_t parsed[16];if(!detail::ParseEid(eid,parsed))return Status::InvalidArgument("invalid receiver fullmesh EID");}
  if(KvafExecuteBatchGetRawAddrLoadKernel(impl_->kernel.c_str()))return Status::Internal("load batch_get_raw_addr kernel");
  auto endpoint=advertised.empty()?address:advertised;
  const auto split=endpoint.rfind(':');
  if(split==std::string::npos || endpoint.substr(0,split)=="0.0.0.0")
    return Status::InvalidArgument("listen_hbm requires a routable advertise_address when binding 0.0.0.0");
  std::uint16_t port{};status=hr::Listen(address,&impl_->listener,&port);if(!status.ok())return status;
  if(endpoint.substr(split+1)=="0")endpoint=endpoint.substr(0,split+1)+std::to_string(port);
  impl_->receiver={endpoint,detail::Nonce(),detail::Nonce()};impl_->local_eid=eid;impl_->stop=false;impl_->accepting=true;
  // Device initialization errors are returned before the listener is published.
  std::promise<Status> promise;auto future=promise.get_future();
  impl_->thread=std::thread([this,promise=std::move(promise)]()mutable{
    auto s=aclrtSetDevice(impl_->device)==ACL_SUCCESS?Status::OK():Status::Internal("select receiver control device");
    promise.set_value(s);if(s.ok())Serve();
  });
  status=future.get();
  if(!status.ok()){impl_->thread.join();close(impl_->listener);impl_->listener=-1;return status;}
  *out=impl_->receiver;return Status::OK();
}
void HbmReader::Serve() {
  if(aclrtSetDevice(impl_->device)!=ACL_SUCCESS)return;
  while(!impl_->stop.load()) {
    pollfd p{impl_->listener,POLLIN,0};if(poll(&p,1,50)<=0)continue;
    int fd=accept(impl_->listener,nullptr,nullptr);if(fd<0)continue;
    offload::SetSocketTimeout(fd);int yes=1;setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&yes,sizeof(yes));
    hr::Hello h{};HbmReadTicket ticket;auto status=hr::ReadHello(fd,&h,&ticket);
    {std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
      if(status.ok() && (h.receiver_id!=impl_->receiver.receiver_id || h.receiver_epoch!=impl_->receiver.receiver_epoch))
        status=Status::InvalidState("stale HBM receiver ticket; reinitialize the source");
      if(status.ok() && !impl_->accepting && h.header.command==unsigned(hr::Admission::Join))
        status=Status::Busy("HBM receiver is shutting down; only recovery/detach is allowed");
      if(status.ok()) {
        const Impl::SourceKey key{ticket.source_id,ticket.epoch};
        auto it=impl_->connections.find(key);
        if(h.header.command==unsigned(hr::Admission::Leave))status=Detach(ticket);
        else if(it!=impl_->connections.end()) {
          auto& c=*it->second;
          if(!c.link->ticket.same_registration(ticket))
            status=Status::InvalidArgument("HBM reconnect changed source manifest/token");
          else {
            int copy=dup(fd);
            status=copy<0?Status::Internal("duplicate HBM socket"):c.link->control->Replace(copy);
            if(!status.ok() && copy>=0)close(copy);
          }
        } else if(h.header.command==unsigned(hr::Admission::Resume))status=Status::NotFound("HBM receiver session retired; do not recreate or replay READ");
        else {
          int copy=dup(fd);
          status=copy<0?Status::Internal("duplicate HBM socket"):Attach(std::make_shared<hr::ControlChannel>(copy),ticket,impl_->local_eid);
        }
      }
      hr::SendResult(fd,status);
    }
    close(fd);
  }
}
Status HbmReader::Cancel(const ReadyToken& ready) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  auto status=impl_->Device();if(!status.ok())return status;
  const auto* source=std::get_if<NpuPullSourceDesc>(&ready.source);
  if(!source)return Status::InvalidArgument("cancel_raw_addr requires an NPU Pull token");
  auto it=impl_->connections.find({source->source_id,source->source_epoch});
  if(it==impl_->connections.end())return Status::NotFound("NPU Pull source is not connected");
  auto& c=*it->second;
  if(!c.link->ticket.matches(*source) || !c.link->ticket.bytes_for(ready.mr_id) || !ready.sequence)
    return Status::InvalidArgument("invalid NPU Pull cancellation identity");
  if(c.pending!=Pending::None)return Status::Busy("resolve the existing raw READ/cancellation first");
  if(ready.sequence==c.last_sequence && ready.mr_id==c.mr_id)return Status::OK();
  if(ready.sequence<=c.last_sequence)return Status::InvalidState("NPU Pull batch already consumed");
  c.sequence=ready.sequence;c.mr_id=ready.mr_id;c.pending=Pending::Cancel;c.result=Status::OK();
  // Mark pending before contacting P. Even a lost reply must fence a late READ.
  return c.Resolve();
}
Status HbmReader::Attach(std::shared_ptr<hr::ControlChannel> control,const HbmReadTicket& ticket,const std::string& local_eid) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  auto s=impl_->Device();if(!s.ok())return s;
  if(!control || !ticket.valid() ||
      !impl_->cores || !impl_->jettys || impl_->cores*std::uint64_t(impl_->jettys)>offload::kChannelCount ||
      (impl_->budget && impl_->budget<impl_->cores*2) || impl_->kernel.empty())
    return Status::InvalidArgument("configure raw kernel and 1..4 total Jettys for HBM READ");
  const Impl::SourceKey key{ticket.source_id,ticket.epoch};
  const auto existing=impl_->connections.find(key);
  if(existing!=impl_->connections.end()) {
    const auto& c=*existing->second;
    if(!c.link->ticket.same_registration(ticket) ||
        c.local_eid!=local_eid)
      return Status::InvalidArgument("HBM source is already attached with different metadata or route");
    return c.ready ? Status::OK() : Status::InDoubt("HBM attach/close incomplete; retry producer shutdown");
  }
  if(KvafExecuteBatchGetRawAddrLoadKernel(impl_->kernel.c_str()))return Status::Internal("load batch_get_raw_addr kernel");
  std::vector<npu_eid::NpuLocalEidCandidate> candidates;
  if(local_eid.empty()) {
    std::uint32_t physical{};std::string error;
    if(!direct_hccp::GetCurrentNpuPhysicalDeviceId(&physical))return Status::Internal("resolve A physical device");
    if(!npu_eid::DiscoverPhysicalNpuEids(physical,&candidates,&error))return Status::Internal(error);
  } else { npu_eid::NpuLocalEidCandidate c{};if(!detail::ParseEid(local_eid,c.eid.data()))return Status::InvalidArgument("invalid A EID");candidates.push_back(c); }
  auto discover=hr::Make(hr::Command::Discover,ticket,0);hr::Response info{};
  s=control->Exchange(discover,&info);if(!s.ok())return s;
  if(!info.candidates || info.candidates>256)return Status::Internal("invalid B EID candidate count");
  for(std::uint32_t b=0;b<info.candidates;++b)for(const auto& a:candidates) {
    auto owned=std::make_unique<Connection>(control,ticket,local_eid,impl_->cores,impl_->jettys,impl_->budget);
    auto& c=*owned;impl_->connections.emplace(key,std::move(owned));
    auto request=hr::Make(hr::Command::Open,ticket,c.link->client);request.candidate=b;
    std::memcpy(request.connection.eid,a.eid.data(),16);
    hr::Response response{};c.link->opened=true;
    s=control->Exchange(request,&response);
    if(s.ok()) {
      const auto count=ticket.mr_count();
      c.link->connection=response.connection;
      const auto& connection=c.link->connection;
      if(response.mr_count!=count || connection.key_bytes!=32 || connection.token!=ticket.token ||
          !connection.tp || !connection.psn) s=Status::Internal("invalid B HBM connection or MR count");
      for(std::uint64_t mr=1;s.ok() && mr<=count;++mr) {
        if(mr>1) s=c.link->Call(hr::Command::Memory,&response,0,mr);
        if(!s.ok()) break;
        const auto& e=response.memory;
        if(response.mr_count!=count || e.token!=ticket.token || e.bytes!=ticket.bytes_for(mr) ||
            e.segment_bytes!=48 || !e.base || e.bytes>UINT64_MAX-e.base) {
          s=Status::Internal("invalid B HBM export descriptor");break;
        }
        for(const auto& prior:c.link->memories) if(e.base==prior.base)
          s=Status::Internal("ambiguous exported HBM base");
        c.link->memories.push_back(e);c.link->checksums.push_back(response.checksum);
      }
      if(s.ok() && KvafBatchGetTransportCreateWithPeerLink(reinterpret_cast<void*>(c.link->memories.front().base),c.link->memories.front().bytes,
          impl_->cores,impl_->jettys,c.link.get(),a.eid.data(),&c.engine.transport))
        s=Status::Internal("create A-B raw READ transport");
      for(std::uint64_t mr=2;s.ok() && mr<=count;++mr) {
        const auto& memory=c.link->memories[mr-1];
        if(KvafBatchGetTransportAddRemoteMemory(c.engine.transport,mr,memory.base,memory.bytes))
          s=Status::Internal("import B HBM MR "+std::to_string(mr));
      }
    }
    if(s.ok()) {
      const auto rows=impl_->cores*(128U*(impl_->jettys-1)+1U);
      s=c.engine.scratch.Allocate(rows*64U);
      auto& stream=c.engine.probe_stream;
      if(s.ok() && aclrtCreateStream(&stream)!=ACL_SUCCESS)s=Status::Internal("create raw READ probe stream");
      for(std::size_t mr=0;s.ok() && mr<c.link->memories.size();++mr) {
        c.engine.mr_id=mr+1;
        const auto& memory=c.link->memories[mr];
        const auto bytes=std::min<std::uint64_t>(memory.bytes,64);
        auto base=reinterpret_cast<std::uint64_t>(c.engine.scratch.data);
        std::vector<hr::Span> probe;
        for(std::uint32_t i=0;i<rows;++i)probe.push_back({memory.base,base+i*bytes,bytes});
        s=c.engine.RunProbe(probe,{c.engine.scratch.data,c.engine.scratch.bytes,DeviceKind::kNpu},stream);
        if(s.ok()) {
          std::vector<std::uint8_t> result(rows*bytes);
          if(aclrtMemcpy(result.data(),result.size(),c.engine.scratch.data,result.size(),ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS)
            s=Status::Internal("copy raw READ probe output");
          for(std::uint32_t i=0;s.ok() && i<rows;++i)
            if(offload::Hash(result.data()+i*bytes,bytes)!=c.link->checksums[mr])s=Status::Internal("raw READ probe checksum mismatch");
        }
        if(s.ok()) s=c.engine.Cleanup();
      }
      if(c.engine.uncertain) {
        const auto quiesced=c.engine.QuiesceProbe();
        if(!quiesced.ok()) {
          c.pending=Pending::Dma;
          return Status::InDoubt("HBM attach probe quarantined; source="+std::to_string(ticket.source_id)+"; retain producer source; "+s.message()+"; "+quiesced.message());
        }
        s=Status::Internal("startup probe READ failed; transport quiesced: "+s.message());
      }
      if(stream) {
        if(aclrtDestroyStream(stream)==ACL_SUCCESS)stream=nullptr;
        else if(s.ok())s=Status::Internal("destroy raw READ probe stream");
      }
      if(s.ok())s=c.engine.Cleanup();
      if(s.ok())s=c.engine.scratch.Free();
      if(s.ok())s=c.link->Call(hr::Command::Confirm,&response);
    }
    if(s.ok()) {
      c.ready=true;
      std::fprintf(stderr,"[KVAF_NPU] HBM_READ_READY source=%llu epoch=%llu A_eid=%s B_eid=%s jettys=%u mrs=%zu probe=all_mrs_all_lanes_pass\n",
          (unsigned long long)ticket.source_id,(unsigned long long)ticket.epoch,npu_eid::EidToHex(a.eid).c_str(),detail::EidHex(c.link->connection.eid).c_str(),impl_->cores*impl_->jettys,ticket.mr_count());
      return s;
    }
    std::fprintf(stderr,"[KVAF_NPU] HBM_READ_CANDIDATE_REJECTED source=%llu A_eid=%s B_candidate=%u detail=%s\n",
        (unsigned long long)ticket.source_id,npu_eid::EidToHex(a.eid).c_str(),b,s.message().c_str());
    auto cleanup=Detach(ticket);
    if(!cleanup.ok())return Status::InDoubt("HBM attach cleanup pending; retry producer shutdown; "+cleanup.message());
  }
  return s;
}
Status HbmReader::Read(const ReadyToken& ready,NpuInt64ArrayView src,NpuInt64ArrayView dst,
                       NpuInt64ArrayView lengths,MutableBufferView output,void* stream,HbmReadProfile* profile,
                       bool validate_ranges) {
  if(profile)*profile={};
  ProfileTimer total_timer(profile?&profile->total_us:nullptr);
  const auto validation_begin=profile?ProfileClock::now():ProfileClock::time_point{};
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  auto s=impl_->Device();if(!s.ok())return s;
  const auto* source=std::get_if<NpuPullSourceDesc>(&ready.source);
  if(!source)return Status::InvalidArgument("raw READ requires an NPU Pull ReadyToken");
  auto it=impl_->connections.find({source->source_id,source->source_epoch});
  if(it==impl_->connections.end())return Status::NotFound("NPU Pull source is not attached");
  auto& c=*it->second;
  if(!c.ready || c.pending!=Pending::None)return Status::Busy("raw READ region not ready or has unresolved work");
  const auto& ticket=c.link->ticket;
  if(!ticket.matches(*source) || !ticket.bytes_for(ready.mr_id) ||
      !ready.sequence || ready.sequence<=c.last_sequence)
    return Status::InvalidArgument("NPU Pull ReadyToken belongs to another source or consumed batch");
  const auto base=reinterpret_cast<std::uint64_t>(output.data);
  if(!output.valid() || output.device!=DeviceKind::kNpu || !stream || output.bytes>UINT64_MAX-base ||
      !src.data || !dst.data || !lengths.data || !src.size || src.size>offload::kMaxSpans ||
      src.size!=dst.size || src.size!=lengths.size ||
      reinterpret_cast<std::uintptr_t>(src.data)%8 || reinterpret_cast<std::uintptr_t>(dst.data)%8 ||
      reinterpret_cast<std::uintptr_t>(lengths.data)%8)
    return Status::InvalidArgument("raw READ requires equally sized aligned NPU int64 arrays and NPU output");
  // READ must never overwrite metadata that another core is still consuming.
  // Reject overlap with the entire output allocation before acquiring the batch.
  for(const auto view:{src,dst,lengths}) {
    const auto address=reinterpret_cast<std::uint64_t>(view.data);
    const std::uint64_t bytes=view.size*sizeof(std::int64_t);
    if(address>UINT64_MAX-bytes || (address<=base ? base-address<bytes : address-base<output.bytes))
      return Status::InvalidArgument("NPU raw READ metadata must not overlap output");
  }
  if(!c.link->control->connected())return Status::InDoubt("HBM control offline; wait for producer reconnect before READ");
  c.sequence=ready.sequence;c.mr_id=ready.mr_id;c.engine.mr_id=ready.mr_id;c.result=Status::OK();
  if(profile)profile->validation_us=std::chrono::duration<double,std::micro>(ProfileClock::now()-validation_begin).count();
  c.pending=Pending::Dma;
  s=c.engine.RunDevice(src,dst,lengths,output,stream,profile,validate_ranges);
  if(c.engine.uncertain)return s;
  c.result=s;c.pending=Pending::Cleanup;
  return c.Resolve(profile);
}
Status HbmReader::Resolve(const ReadyToken& ready){
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  auto s=impl_->Device();if(!s.ok())return s;
  const auto* source=std::get_if<NpuPullSourceDesc>(&ready.source);
  if(!source)return Status::InvalidArgument("raw READ recovery requires an NPU Pull ReadyToken");
  auto it=impl_->connections.find({source->source_id,source->source_epoch});
  if(it==impl_->connections.end())return Status::NotFound("NPU Pull source is not attached");
  const auto& ticket=it->second->link->ticket;
  if(!ticket.matches(*source) || !ticket.bytes_for(ready.mr_id) ||
      !ready.sequence || ready.sequence!=it->second->sequence || ready.mr_id!=it->second->mr_id)
    return Status::InvalidArgument("raw READ recovery token does not match the pending batch");
  return it->second->Resolve();
}
Status HbmReader::Detach(const HbmReadTicket& ticket){
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  auto s=impl_->Device();if(!s.ok())return s;
  if(!ticket.valid())
    return Status::InvalidArgument("invalid HBM export ticket");
  auto it=impl_->connections.find({ticket.source_id,ticket.epoch});
  if(it==impl_->connections.end())return Status::OK(); // Also valid after a fully cleaned failed Attach.
  auto& c=*it->second;
  if(!c.link->ticket.same_registration(ticket))
    return Status::InvalidArgument("HBM detach ticket does not match the attached source");
  if(c.pending!=Pending::None)return Status::InDoubt("resolve raw READ before detach");
  // Close may destroy queues before a later MR unregister fails. Retrying
  // teardown is safe; admitting another READ on that transport is not.
  c.ready=false;
  s=c.engine.Close();if(!s.ok())return s;
  if(c.link->opened){hr::Response response{};s=c.link->Call(hr::Command::Close,&response);
    if(!s.ok() && s.code()!=StatusCode::kNotFound)return s;
    // Both hardware sides are retired. This final notification lets P finish
    // shutdown even if D subsequently closes its listener. Lost notification
    // is recovered by Leave/Resume -> NotFound, never a new hardware attach.
    c.link->Call(hr::Command::Retire,&response);}
  impl_->connections.erase(it);return Status::OK();
}
Status HbmReader::Shutdown(){
  {std::lock_guard<std::recursive_mutex> lock(impl_->mutex);impl_->accepting=false;}
  impl_->stop=true;
  if(impl_->thread.joinable())impl_->thread.join();
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  while(!impl_->connections.empty()){const auto ticket=impl_->connections.begin()->second->link->ticket;auto s=Detach(ticket);if(!s.ok()){impl_->stop=false;impl_->thread=std::thread([this]{Serve();});return s;}}
  if(impl_->listener>=0){close(impl_->listener);impl_->listener=-1;}
  impl_->receiver={};return Status::OK();}
bool HbmReader::pending()const{std::lock_guard<std::recursive_mutex> lock(impl_->mutex);for(auto& item:impl_->connections)if(item.second->pending!=Pending::None)return true;return false;}
}
