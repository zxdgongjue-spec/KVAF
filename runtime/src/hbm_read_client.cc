#include "hbm_read_client.h"
#include "hbm_read_protocol.h"
#include "hbm_exporter.h"
#include "kvaf_aiv_batch_get_raw_addr.h"
#include "hbm_read_diagnostics.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <deque>
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
    if(data && aclrtFree(data)!=ACL_SUCCESS) return Status::CompletionUnknown("free raw READ device buffer");
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
  std::uint32_t cores,jettys,budget;
  std::uint64_t mr_id{1};
  bool statuses_initialized{};
  bool uncertain{},kernel_finished{},cache_output{};
  bool diagnostics{};
  aclrtStream probe_stream{};
  aclrtEvent profile_start{},profile_end{};
  Engine(std::uint32_t c,std::uint32_t j,std::uint32_t b):cores(c),jettys(j),budget(b){
    const auto* debug=std::getenv("KVAF_HBM_RAW_READ_DIAGNOSTICS");
    diagnostics=debug && debug[0]=='1' && debug[1]=='\0';
  }
  Status Cleanup() {
    if(uncertain)return Status::CompletionUnknown("raw READ transport quarantined; source and output must remain alive");
    if(cache_output) {
      KvafBatchGetLocalBufferRelease(local);local=nullptr;
    } else if(KvafBatchGetLocalBufferDiscard(&local))return Status::CompletionUnknown("raw READ output unregister pending");
    return descriptors.Free();
  }
  Status QuiesceProbe() {
    if (!uncertain) return Status::OK();
    if (!kernel_finished || KvafBatchGetTransportQuiesce(transport))
      return Status::CompletionUnknown("probe transport could not be quiesced; retain B source and A scratch");
    uncertain=false;
    return Status::OK();
  }
  Status Close() {
    auto s=Cleanup();if(!s.ok())return s;
    for(auto* event:{&profile_start,&profile_end})if(*event) {
      if(aclrtDestroyEvent(*event)!=ACL_SUCCESS)return Status::CompletionUnknown("destroy raw READ profiling event");
      *event=nullptr;
    }
    if(KvafBatchGetTransportClose(transport))return Status::CompletionUnknown("raw READ transport teardown pending");
    transport=nullptr;
    s=statuses.Free();if(!s.ok())return s;
    statuses_initialized=false;
    s=queues.Free();if(!s.ok())return s;
    if(probe_stream) {
      if(aclrtDestroyStream(probe_stream)!=ACL_SUCCESS)return Status::CompletionUnknown("destroy raw probe stream");
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
                   MutableBufferView output,void* stream,HbmReadProfile* profile,bool validate_ranges,bool synchronize) {
    if(!transport || !stream || uncertain || local || descriptors.data)
      return Status::InvalidState("raw READ engine is not idle");
    KvafExecuteBatchGetRawAddrKernelArgs args{};
    args.source_addresses=reinterpret_cast<std::uint64_t>(src.data);
    args.destination_addresses=reinterpret_cast<std::uint64_t>(dst.data);
    args.lengths=reinterpret_cast<std::uint64_t>(lengths.data);
    args.request_count=src.size;args.input_mode=KVAF_RAW_NPU_OFFSETS;
    args.validate_ranges=validate_ranges;
    cache_output=true;
    return RunKernel(args,output,stream,profile,synchronize);
  }
  Status Prepare() {
    if(!statuses.data) {
      auto allocated=statuses.Allocate(cores*sizeof(KvafRawReadCoreStatus));if(!allocated.ok())return allocated;
    }
    // Device diagnostics are initialized once, like ordinary BatchGet. The
    // Host does not copy or interpret them on the synchronous READ path.
    if(!statuses_initialized) {
      if(aclrtMemset(statuses.data,statuses.bytes,0,statuses.bytes)!=ACL_SUCCESS)
        return Status::Internal("initialize raw READ status");
      statuses_initialized=true;
    }
    if(!queues.data) {
      auto allocated=queues.Allocate(cores*jettys*sizeof(KvafBatchGetQueueState));if(!allocated.ok())return allocated;
      if(aclrtMemset(queues.data,queues.bytes,0,queues.bytes)!=ACL_SUCCESS)return Status::Internal("clear raw READ queues");
    }
    return Status::OK();
  }
  Status RunKernel(KvafExecuteBatchGetRawAddrKernelArgs args,MutableBufferView output,void* stream,
                   HbmReadProfile* profile=nullptr,bool synchronize=true) {
    auto s=Timed(profile?&profile->buffers_us:nullptr,[&]() -> Status {
    auto prepared=Prepare();if(!prepared.ok())return prepared;
    if(profile && synchronize)for(auto* event:{&profile_start,&profile_end})
      if(!*event && aclrtCreateEventWithFlag(event,ACL_EVENT_TIME_LINE)!=ACL_SUCCESS)
        return Status::Internal("create raw READ timing event");
    return Status::OK();
    });
    if(!s.ok())return s;
    std::uint64_t entity{};
    if(Timed(profile?&profile->register_output_us:nullptr,[&]{
        return KvafBatchGetLocalBufferAcquireForRemote(transport,mr_id,output.data,output.bytes,&local,&entity);}))
      { if(local)cache_output=false;
        return local?Status::CompletionUnknown("raw output registration cleanup pending"):
                     Status::Internal("register A raw READ output"); }
    args.output=reinterpret_cast<std::uint64_t>(output.data);
    args.core_status=reinterpret_cast<std::uint64_t>(statuses.data);args.queue_state=reinterpret_cast<std::uint64_t>(queues.data);
    args.channel_entity=entity;args.block_count=cores;args.jettys_per_core=jettys;args.batch_get_max_inflight_total=budget;
    if(diagnostics && aclrtMemset(statuses.data,statuses.bytes,0,statuses.bytes)!=ACL_SUCCESS)
      return Status::Internal("clear raw READ diagnostic status");
    kernel_finished=false;
    uncertain=true; // Includes launch errors: never assume the device saw no work.
    if(profile && synchronize)profile->event_error=Timed(&profile->event_record_us,[&]{
      return aclrtRecordEvent(profile_start,static_cast<aclrtStream>(stream));});
    if(Timed(profile?&profile->launch_us:nullptr,[&]{
        return KvafExecuteBatchGetRawAddrLaunch(&args,static_cast<aclrtStream>(stream));}))
      return Status::CompletionUnknown("raw READ launch/stream completion uncertain");
    if(profile && synchronize && !profile->event_error)profile->event_error=Timed(&profile->event_record_us,[&]{
      return aclrtRecordEvent(profile_end,static_cast<aclrtStream>(stream));});
    // The default path only enqueues work, including during graph capture.
    // Synchronization and result readiness belong to the caller.
    if(synchronize || diagnostics) {
      if(Timed(profile?&profile->synchronize_us:nullptr,[&]{
          return aclrtSynchronizeStream(static_cast<aclrtStream>(stream));})!=ACL_SUCCESS)
        return Status::CompletionUnknown("raw READ launch/stream completion uncertain");
      kernel_finished=true;
    }
    if(diagnostics) {
      std::vector<KvafRawReadCoreStatus> status(cores);
      std::vector<KvafBatchGetQueueState> queue(cores*jettys);
      std::vector<KvafBatchGetDirectUrmaEntity> entities(cores*jettys);
      if(aclrtMemcpy(status.data(),statuses.bytes,statuses.data,statuses.bytes,ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS ||
          aclrtMemcpy(queue.data(),queues.bytes,queues.data,queues.bytes,ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS ||
          aclrtMemcpy(entities.data(),entities.size()*sizeof(entities.front()),reinterpret_cast<void*>(entity),
              entities.size()*sizeof(entities.front()),ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS)
        return Status::CompletionUnknown("raw READ diagnostic snapshot failed; retain source and output");
      std::fprintf(stderr,"%s",raw_read_debug::Format(args,status.data(),queue.data(),entities.data()).c_str());
      if(!raw_read_debug::AllRequestsCompleted(args,status.data()))
        return Status::CompletionUnknown("raw READ kernel diagnostic failed; inspect HBM_READ_CORE_DIAGNOSTIC; retain source and output");
    }
    if(profile && synchronize && !profile->event_error) {
      float ms{};
      profile->event_error=Timed(&profile->event_query_us,[&]{return aclrtEventElapsedTime(&ms,profile_start,profile_end);});
      profile->kernel_event_valid=!profile->event_error && std::isfinite(ms) && ms>=0;
      if(profile->kernel_event_valid)profile->kernel_event_us=static_cast<double>(ms)*1000;
    }
    uncertain=false;return Status::OK();
  }
};
struct RawInvocation {
  std::uint64_t id{}, sequence{}, mr{};
  RawReadLifetime lifetime;
};
struct RawGraphNode {
  KvafExecuteBatchGetRawAddrKernelArgs args{};
  DeviceBuffer status;
  KvafBatchGetLocalBuffer* partial_local{};
  std::uint64_t mr{};
  MutableBufferView output;
  void* stream{};
  RawReadLifetime lifetime;
  bool captured{};
};
struct Connection {
  std::unique_ptr<ReadLink> link;
  std::string local_eid;
  Engine engine;
  std::uint64_t cancelled_sequence{},cancelled_mr{1};
  std::uint64_t failed_sequence{},failed_mr{1};
  bool ready{}, draining{}, closing{};
  std::mutex mutex;
  void* stream{};
  std::uint64_t next_invocation{1};
  std::deque<RawInvocation> invocations;
  std::map<std::pair<std::uintptr_t,std::uint64_t>,std::shared_ptr<void>> outputs;
  std::map<std::uint64_t,std::unique_ptr<RawGraphNode>> graphs;
  Connection(std::shared_ptr<hr::ControlChannel> a,const HbmReadTicket& t,const std::string& eid,std::uint32_t c,std::uint32_t j,std::uint32_t b):
      link(new ReadLink(a,t)),local_eid(eid),engine(c,j,b){}
};
}
struct HbmReader::Impl {
  std::uint32_t device,cores,jettys,budget,max_inflight,max_outputs;
  bool diagnostics{};
  std::atomic<std::uint64_t> next_graph{1};
  std::string kernel;
  mutable std::recursive_mutex mutex;
  HbmReceiverTicket receiver;
  std::string local_eid;
  std::atomic<bool> accepting{true};
  int listener{-1};
  std::atomic<bool> stop{false};
  std::thread thread;
  // Registration identity, never an application-maintained local connection ID.
  using SourceKey = std::pair<std::uint64_t, std::uint64_t>;
  std::map<SourceKey,std::shared_ptr<Connection>> connections;
  std::shared_ptr<Connection> Find(const ReadyToken& ready) const {
    const auto* source=std::get_if<NpuPullSourceDesc>(&ready.source);
    if(!source)return {};
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto it=connections.find({source->source_id,source->source_epoch});
    return it==connections.end()?nullptr:it->second;
  }
  Status Device()const {
    int current=-1;return aclrtGetDevice(&current)==ACL_SUCCESS && current==int(device)?Status::OK():
        Status::InvalidState("select A device before HBM READ operation");
  }
};
HbmReader::HbmReader(std::uint32_t d,std::uint32_t c,std::uint32_t j,std::uint32_t b,std::string k,
    std::uint32_t max_inflight,std::uint32_t max_outputs,bool diagnostics):
    impl_(new Impl{}) {
  impl_->device=d;impl_->cores=c;impl_->jettys=j;impl_->budget=b;impl_->kernel=std::move(k);
  impl_->max_inflight=max_inflight;impl_->max_outputs=max_outputs;impl_->diagnostics=diagnostics;
}
HbmReader::~HbmReader(){if(!Shutdown().ok()){
  impl_->stop=true;if(impl_->thread.joinable())impl_->thread.join();
  (void)impl_.release(); // Retain unresolved hardware, but never a thread capturing a destroyed owner.
}}
Status HbmReader::Listen(const std::string& address,const std::string& advertised,const std::string& eid,HbmReceiverTicket* out) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  auto status=impl_->Device();if(!status.ok())return status;
  if(!out || impl_->listener>=0)return Status::InvalidState("HBM receiver already listening or missing result");
  if(!impl_->cores || !impl_->jettys || impl_->cores*std::uint64_t(impl_->jettys)>offload::kChannelCount ||
     (impl_->budget && impl_->budget<impl_->cores*2) || !impl_->max_inflight || !impl_->max_outputs || impl_->kernel.empty())
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
    std::shared_ptr<Connection> existing;
    {
      std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
      if(status.ok() && (h.receiver_id!=impl_->receiver.receiver_id || h.receiver_epoch!=impl_->receiver.receiver_epoch))
        status=Status::InvalidState("stale HBM receiver ticket; reinitialize the source");
      if(status.ok() && !impl_->accepting && h.header.command==unsigned(hr::Admission::Join))
        status=Status::Busy("HBM receiver is shutting down; only recovery/detach is allowed");
      auto it=impl_->connections.find({ticket.source_id,ticket.epoch});
      if(it!=impl_->connections.end())existing=it->second;
    }
    if(status.ok()) {
      if(h.header.command==unsigned(hr::Admission::Leave))status=Detach(ticket);
      else if(existing) {
        std::lock_guard<std::mutex> lock(existing->mutex);
        if(!existing->link->ticket.same_registration(ticket))
          status=Status::InvalidArgument("HBM reconnect changed source manifest/token");
        else {
          int copy=dup(fd);
          status=copy<0?Status::Internal("duplicate HBM socket"):existing->link->control->Replace(copy);
          if(!status.ok() && copy>=0)close(copy);
        }
      } else if(h.header.command==unsigned(hr::Admission::Resume))
        status=Status::NotFound("HBM receiver session retired; do not recreate or replay READ");
      else {
        int copy=dup(fd);
        status=copy<0?Status::Internal("duplicate HBM socket"):Attach(std::make_shared<hr::ControlChannel>(copy),ticket,impl_->local_eid);
      }
    }
    hr::SendResult(fd,status);
    close(fd);
  }
}
Status HbmReader::Cancel(const ReadyToken& ready) {
  auto status=impl_->Device();if(!status.ok())return status;
  const auto* source=std::get_if<NpuPullSourceDesc>(&ready.source);
  if(!source)return Status::InvalidArgument("cancel_raw_addr requires an NPU Pull token");
  auto connection=impl_->Find(ready);
  if(!connection)return Status::NotFound("NPU Pull source is not connected");
  std::lock_guard<std::mutex> lock(connection->mutex);
  auto& c=*connection;
  if(c.closing || c.draining)return Status::Busy("NPU Pull connection is draining/closing");
  if(!c.link->ticket.matches(*source) || !c.link->ticket.bytes_for(ready.mr_id) || !ready.sequence)
    return Status::InvalidArgument("invalid NPU Pull cancellation identity");
  if(c.engine.uncertain)return Status::Busy("raw READ submission/completion is unresolved");
  if(ready.sequence==c.cancelled_sequence && ready.mr_id==c.cancelled_mr)return Status::OK();
  if(ready.sequence<=c.cancelled_sequence)return Status::InvalidState("NPU Pull batch already cancelled");
  // Fence a skipped batch locally. The application coordinates source reuse;
  // this operation neither notifies the producer nor changes its state.
  // Already enqueued work and captured graphs remain the caller's responsibility.
  c.cancelled_sequence=ready.sequence;c.cancelled_mr=ready.mr_id;
  return Status::OK();
}
Status HbmReader::Attach(std::shared_ptr<hr::ControlChannel> control,const HbmReadTicket& ticket,const std::string& local_eid) {
  auto s=impl_->Device();if(!s.ok())return s;
  if(!control || !ticket.valid() ||
      !impl_->cores || !impl_->jettys || impl_->cores*std::uint64_t(impl_->jettys)>offload::kChannelCount ||
      (impl_->budget && impl_->budget<impl_->cores*2) || !impl_->max_inflight || !impl_->max_outputs || impl_->kernel.empty())
    return Status::InvalidArgument("configure raw kernel and 1..4 total Jettys for HBM READ");
  const Impl::SourceKey key{ticket.source_id,ticket.epoch};
  std::shared_ptr<Connection> existing;
  {std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    auto it=impl_->connections.find(key);if(it!=impl_->connections.end())existing=it->second;}
  if(existing) {
    std::lock_guard<std::mutex> lock(existing->mutex);
    const auto& c=*existing;
    if(!c.link->ticket.same_registration(ticket) || c.local_eid!=local_eid)
      return Status::InvalidArgument("HBM source is already attached with different metadata or route");
    return c.ready ? Status::OK() : Status::CompletionUnknown("HBM attach/close incomplete; retry producer shutdown");
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
    auto owned=std::make_shared<Connection>(control,ticket,local_eid,impl_->cores,impl_->jettys,impl_->budget);
    auto& c=*owned;
    c.engine.diagnostics=c.engine.diagnostics || impl_->diagnostics;
    {std::lock_guard<std::recursive_mutex> lock(impl_->mutex);impl_->connections.emplace(key,owned);}
    std::unique_lock<std::mutex> connection_lock(c.mutex);
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
        // A zero-filled producer and untouched zero-filled scratch must not
        // make a failed/no-op READ appear successful. Choose a poison whose
        // checksum differs from this MR's expectation, before EVERY MR probe.
        std::array<std::uint8_t,64> poison{};
        unsigned fill=0;
        for(;fill<256;++fill) {
          poison.fill(static_cast<std::uint8_t>(fill));
          if(offload::Hash(poison.data(),bytes)!=c.link->checksums[mr])break;
        }
        if(fill==256 || aclrtMemset(c.engine.scratch.data,c.engine.scratch.bytes,
            fill,c.engine.scratch.bytes)!=ACL_SUCCESS) {
          s=Status::Internal("prefill raw READ probe scratch");break;
        }
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
      // Failed data probes still need checked queue retirement. Stream success
      // alone no longer exposes a device CQ error through a Host status copy.
      if(!s.ok() && c.engine.kernel_finished)c.engine.uncertain=true;
      if(c.engine.uncertain) {
        const auto quiesced=c.engine.QuiesceProbe();
        if(!quiesced.ok()) {
          return Status::CompletionUnknown("HBM attach probe quarantined; source="+std::to_string(ticket.source_id)+"; retain producer source; "+s.message()+"; "+quiesced.message());
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
    connection_lock.unlock();
    auto cleanup=Detach(ticket);
    if(!cleanup.ok())return Status::CompletionUnknown("HBM attach cleanup pending; retry producer shutdown; "+cleanup.message());
  }
  return s;
}
Status HbmReader::Read(const ReadyToken& ready,NpuInt64ArrayView src,NpuInt64ArrayView dst,
                       NpuInt64ArrayView lengths,MutableBufferView output,void* stream,HbmReadProfile* profile,
                       bool validate_ranges,bool synchronize,std::uint64_t graph_lease,RawReadLifetime lifetime) {
  if(profile)*profile={};
  ProfileTimer total_timer(profile?&profile->total_us:nullptr);
  const auto validation_begin=profile?ProfileClock::now():ProfileClock::time_point{};
  auto s=impl_->Device();if(!s.ok())return s;
  const auto* source=std::get_if<NpuPullSourceDesc>(&ready.source);
  if(!source)return Status::InvalidArgument("raw READ requires an NPU Pull ReadyToken");
  auto connection=impl_->Find(ready);
  if(!connection)return Status::NotFound("NPU Pull source is not attached");
  std::lock_guard<std::mutex> lock(connection->mutex);
  auto& c=*connection;
  if(!impl_->accepting || !c.ready || c.engine.uncertain || c.closing || c.draining)
    return Status::Busy("raw READ connection not ready or has unresolved work");
  const auto& ticket=c.link->ticket;
  if(!ticket.matches(*source) || !ticket.bytes_for(ready.mr_id) ||
      !ready.sequence || ready.sequence<=c.cancelled_sequence)
    return Status::InvalidArgument("NPU Pull ReadyToken belongs to another source or cancelled batch");
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
  if(c.stream && c.stream!=stream)
    return Status::Busy("DrainRawAddr before switching the source connection stream");
  if(graph_lease) {
    auto found=c.graphs.find(graph_lease);
    if(found==c.graphs.end())return Status::InvalidArgument("raw graph lease belongs to another connection");
    auto& node=*found->second;
    if(node.captured)return Status::InvalidState("raw graph lease already captured; prepare a separate node");
    if(synchronize || c.engine.diagnostics || node.mr!=ready.mr_id || node.stream!=stream ||
       node.output.data!=output.data || node.output.bytes!=output.bytes ||
       node.args.source_addresses!=reinterpret_cast<std::uint64_t>(src.data) ||
       node.args.destination_addresses!=reinterpret_cast<std::uint64_t>(dst.data) ||
       node.args.lengths!=reinterpret_cast<std::uint64_t>(lengths.data) || node.args.request_count!=src.size)
      return Status::InvalidArgument("raw graph request differs from prepared node");
    node.args.validate_ranges=validate_ranges;
    node.captured=true;
    if(KvafExecuteBatchGetRawAddrLaunch(&node.args,static_cast<aclrtStream>(stream))) {
      c.engine.uncertain=true;c.failed_sequence=ready.sequence;c.failed_mr=ready.mr_id;
      return Status::CompletionUnknown("raw graph launch uncertain; retain lease");
    }
    return Status::OK();
  }
  if(c.invocations.size()>=impl_->max_inflight)
    return Status::Busy("raw invocation budget exhausted; DrainRawAddr");
  const auto output_key=std::make_pair(reinterpret_cast<std::uintptr_t>(output.data),output.bytes);
  if(!c.outputs.count(output_key)) {
    if(c.outputs.size()>=impl_->max_outputs)return Status::Busy("raw output binding budget exhausted");
    c.outputs.emplace(output_key,lifetime.output);
  }
  c.stream=stream;
  c.invocations.push_back({c.next_invocation++,ready.sequence,ready.mr_id,std::move(lifetime)});
  c.engine.mr_id=ready.mr_id;
  if(profile)profile->validation_us=std::chrono::duration<double,std::micro>(ProfileClock::now()-validation_begin).count();
  s=c.engine.RunDevice(src,dst,lengths,output,stream,profile,validate_ranges,synchronize);
  if(c.engine.uncertain) {
    c.failed_sequence=ready.sequence;c.failed_mr=ready.mr_id;
    return s;
  }
  // Release the temporary reference, retaining cached registrations for queued
  // work and graph replay exactly as ordinary BatchGet does.
  const auto cleanup=Timed(profile?&profile->cleanup_us:nullptr,[&]{return c.engine.Cleanup();});
  if(s.ok() && cleanup.ok() && (synchronize || c.engine.diagnostics))c.invocations.clear();
  return s.ok()?cleanup:s;
}
Status HbmReader::Drain(const ReadyToken& ready) {
  auto status=impl_->Device();if(!status.ok())return status;
  auto connection=impl_->Find(ready);
  if(!connection)return Status::NotFound("NPU Pull source is not attached");
  void* stream{};std::uint64_t watermark{};
  {
    std::lock_guard<std::mutex> lock(connection->mutex);
    auto& c=*connection;
    if(!ready.sequence || !c.link->ticket.bytes_for(ready.mr_id))return Status::InvalidArgument("invalid raw Drain identity");
    if(c.draining || c.closing)return Status::Busy("raw connection is draining/closing");
    if(c.engine.uncertain)return Status::CompletionUnknown("resolve raw failure; Drain does not discard quarantine");
    c.draining=true;stream=c.stream;
    watermark=c.next_invocation-1;
  }
  const auto result=stream?aclrtSynchronizeStream(static_cast<aclrtStream>(stream)):ACL_SUCCESS;
  std::lock_guard<std::mutex> lock(connection->mutex);
  auto& c=*connection;c.draining=false;
  if(result!=ACL_SUCCESS) {
    c.engine.uncertain=true;
    if(!c.invocations.empty()) {c.failed_sequence=c.invocations.back().sequence;c.failed_mr=c.invocations.back().mr;}
    return Status::CompletionUnknown("raw Drain completion uncertain; retain invocations");
  }
  auto cleanup=c.engine.Cleanup();
  if(!cleanup.ok())return cleanup;
  while(!c.invocations.empty() && c.invocations.front().id<=watermark)c.invocations.pop_front();
  if(c.graphs.empty())c.stream=nullptr;
  return Status::OK();
}
Status HbmReader::BeginGraph(const ReadyToken& ready,NpuInt64ArrayView src,NpuInt64ArrayView dst,
    NpuInt64ArrayView lengths,MutableBufferView output,void* stream,std::uint64_t* lease,RawReadLifetime lifetime) {
  if(!lease || *lease || !stream || !output.valid() || output.device!=DeviceKind::kNpu ||
     !src.data || !dst.data || !lengths.data || !src.size || src.size>KVAF_RAW_MAX_SPANS ||
     src.size!=dst.size || src.size!=lengths.size)
    return Status::InvalidArgument("invalid raw graph node");
  const auto base=reinterpret_cast<std::uint64_t>(output.data);
  if(output.bytes>UINT64_MAX-base)return Status::InvalidArgument("raw graph output overflow");
  for(const auto view:{src,dst,lengths}) {
    const auto address=reinterpret_cast<std::uint64_t>(view.data);
    const auto bytes=view.size*sizeof(std::int64_t);
    if(address%8 || address>UINT64_MAX-bytes ||
       (address<=base?base-address<bytes:address-base<output.bytes))
      return Status::InvalidArgument("raw graph metadata overlap/alignment");
  }
  auto status=impl_->Device();if(!status.ok())return status;
  auto connection=impl_->Find(ready);
  if(!connection)return Status::NotFound("NPU Pull source is not attached");
  std::lock_guard<std::mutex> lock(connection->mutex);
  auto& c=*connection;
  if(c.engine.diagnostics)return Status::InvalidArgument("raw diagnostics are eager-only");
  if(!impl_->accepting || !c.ready || c.engine.uncertain || c.draining || c.closing || !c.invocations.empty())
    return Status::Busy("Drain eager work before preparing raw graph");
  if(!ready.sequence || ready.sequence<=c.cancelled_sequence || !c.link->ticket.bytes_for(ready.mr_id))
    return Status::InvalidArgument("invalid raw graph Ready identity");
  if(c.stream && c.stream!=stream)return Status::Busy("raw graph connection stream differs");
  const auto key=std::make_pair(reinterpret_cast<std::uintptr_t>(output.data),output.bytes);
  if(!c.outputs.count(key)) {
    if(c.outputs.size()>=impl_->max_outputs)return Status::Busy("raw output binding budget exhausted");
    c.outputs.emplace(key,lifetime.output);
  }
  // Publish ownership before any allocation can fail; EndGraph retries cleanup.
  const auto id=impl_->next_graph.fetch_add(1);
  auto node=std::make_unique<RawGraphNode>();node->mr=ready.mr_id;
  node->output=output;node->stream=stream;node->lifetime=std::move(lifetime);
  auto& n=*node;c.graphs.emplace(id,std::move(node));*lease=id;c.stream=stream;
  status=c.engine.Prepare();if(!status.ok())return status;
  status=n.status.Allocate(impl_->cores*sizeof(KvafRawReadCoreStatus));if(!status.ok())return status;
  if(aclrtMemset(n.status.data,n.status.bytes,0,n.status.bytes)!=ACL_SUCCESS)
    return Status::Internal("initialize raw graph status");
  std::uint64_t entity{};
  if(KvafBatchGetLocalBufferAcquireForRemote(c.engine.transport,ready.mr_id,output.data,output.bytes,&n.partial_local,&entity))
    return Status::Internal("prepare raw graph output binding; retain lease");
  KvafBatchGetLocalBufferRelease(n.partial_local);n.partial_local=nullptr;
  auto& args=n.args;
  args.source_addresses=reinterpret_cast<std::uint64_t>(src.data);
  args.destination_addresses=reinterpret_cast<std::uint64_t>(dst.data);
  args.lengths=reinterpret_cast<std::uint64_t>(lengths.data);args.output=base;
  args.core_status=reinterpret_cast<std::uint64_t>(n.status.data);
  args.channel_entity=entity;args.queue_state=reinterpret_cast<std::uint64_t>(c.engine.queues.data);
  args.request_count=src.size;args.block_count=impl_->cores;args.jettys_per_core=impl_->jettys;
  args.batch_get_max_inflight_total=impl_->budget;args.input_mode=KVAF_RAW_NPU_OFFSETS;
  return Status::OK();
}
Status HbmReader::EndGraph(std::uint64_t lease) {
  auto status=impl_->Device();if(!status.ok())return status;
  std::vector<std::shared_ptr<Connection>> connections;
  {std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    for(const auto& entry:impl_->connections)connections.push_back(entry.second);}
  for(auto& connection:connections) {
    std::lock_guard<std::mutex> lock(connection->mutex);
    auto& c=*connection;auto it=c.graphs.find(lease);
    if(it==c.graphs.end())continue;
    if(c.engine.uncertain)return Status::CompletionUnknown("raw graph quarantined; retain lease");
    // Caller must have stopped replay and destroyed the framework graph.
    if(aclrtSynchronizeStream(static_cast<aclrtStream>(it->second->stream))!=ACL_SUCCESS)
      return Status::CompletionUnknown("raw graph stream completion uncertain; retain lease");
    if(it->second->partial_local && KvafBatchGetLocalBufferDiscard(&it->second->partial_local))
      return Status::CompletionUnknown("raw graph partial output registration cleanup pending");
    status=it->second->status.Free();if(!status.ok())return status;
    c.graphs.erase(it);
    // Synchronization also covers preceding eager work on this same stream.
    c.invocations.clear();
    if(c.graphs.empty())c.stream=nullptr;
    return Status::OK();
  }
  return Status::NotFound("raw graph lease not found");
}
bool HbmReader::inflight() const {
  std::vector<std::shared_ptr<Connection>> connections;
  {std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    for(const auto& entry:impl_->connections)connections.push_back(entry.second);}
  for(auto& c:connections){std::lock_guard<std::mutex> lock(c->mutex);if(!c->invocations.empty() || c->draining)return true;}
  return false;
}
Status HbmReader::Resolve(const ReadyToken& ready){
  auto s=impl_->Device();if(!s.ok())return s;
  const auto* source=std::get_if<NpuPullSourceDesc>(&ready.source);
  if(!source)return Status::InvalidArgument("raw READ recovery requires an NPU Pull ReadyToken");
  auto connection=impl_->Find(ready);
  if(!connection)return Status::NotFound("NPU Pull source is not attached");
  std::lock_guard<std::mutex> lock(connection->mutex);
  const auto& ticket=connection->link->ticket;
  if(!ticket.matches(*source) || !ticket.bytes_for(ready.mr_id) ||
      !ready.sequence || ready.sequence!=connection->failed_sequence || ready.mr_id!=connection->failed_mr)
    return Status::InvalidArgument("raw READ recovery token does not match the pending batch");
  return connection->engine.uncertain
      ?Status::CompletionUnknown("raw READ submission/completion unresolved; retain resources; no READ replay")
      :Status::InvalidState("no unresolved raw READ");
}
Status HbmReader::Detach(const HbmReadTicket& ticket){
  auto s=impl_->Device();if(!s.ok())return s;
  if(!ticket.valid())
    return Status::InvalidArgument("invalid HBM export ticket");
  const Impl::SourceKey key{ticket.source_id,ticket.epoch};
  std::shared_ptr<Connection> connection;
  {std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    auto it=impl_->connections.find(key);
    if(it==impl_->connections.end())return Status::OK();
    connection=it->second;}
  auto& c=*connection;
  std::unique_lock<std::mutex> connection_lock(c.mutex);
  if(c.draining || !c.invocations.empty() || !c.graphs.empty())
    return Status::Busy("Drain raw work and end graph leases before detach");
  if(!c.link->ticket.same_registration(ticket))
    return Status::InvalidArgument("HBM detach ticket does not match the attached source");
  c.closing=true;
  if(c.engine.uncertain)return Status::CompletionUnknown("raw READ submission/completion unresolved; retain resources before detach");
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
  connection_lock.unlock();
  {std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    auto it=impl_->connections.find(key);
    if(it!=impl_->connections.end() && it->second==connection)impl_->connections.erase(it);}
  return Status::OK();
}
Status HbmReader::Shutdown(){
  impl_->accepting=false;impl_->stop=true;
  if(impl_->thread.joinable())impl_->thread.join();
  std::vector<std::shared_ptr<Connection>> connections;
  {std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    for(const auto& item:impl_->connections)connections.push_back(item.second);}
  for(const auto& c:connections) {
    ReadyToken ready;bool drain=false;
    {std::lock_guard<std::mutex> lock(c->mutex);
      if(!c->invocations.empty() && c->graphs.empty() && !c->engine.uncertain) {
        ready={c->link->ticket.source(),c->invocations.back().sequence,c->invocations.back().mr};drain=true;
      }}
    auto status=drain?Drain(ready):Status::OK();
    if(status.ok())status=Detach(c->link->ticket);
    if(!status.ok()) {
      if(impl_->listener>=0){impl_->stop=false;impl_->thread=std::thread([this]{Serve();});}
      return status;
    }
  }
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if(impl_->listener>=0){close(impl_->listener);impl_->listener=-1;}
  impl_->receiver={};return Status::OK();
}
bool HbmReader::pending()const{
  std::vector<std::shared_ptr<Connection>> connections;
  {std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    for(const auto& item:impl_->connections)connections.push_back(item.second);}
  for(const auto& c:connections){std::lock_guard<std::mutex> lock(c->mutex);if(c->engine.uncertain)return true;}
  return false;
}
}
