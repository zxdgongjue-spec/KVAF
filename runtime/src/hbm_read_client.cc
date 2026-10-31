#include "hbm_read_client.h"
#include "hbm_read_protocol.h"
#include "hbm_exporter.h"
#include "kvaf_aiv_batch_get_raw_addr.h"
#include <algorithm>
#include <map>

namespace kvaf {
namespace hr=hbm_read;
namespace {
constexpr std::size_t kMaxWqes=1U<<20;
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
  ReadLink(std::string address,HbmReadTicket t):PeerLink(std::move(address)),ticket(t),client(detail::Nonce()){}
  HbmReadTicket ticket;
  std::uint64_t client,checksum{};
  offload::Endpoint endpoint{};
  std::uint32_t channel{};
  bool opened{};
  Status Call(hr::Command command,const std::vector<hr::Span>& spans,hr::Response* response,std::uint64_t sequence=0) {
    auto r=hr::Make(command,ticket,client);r.sequence=sequence;return hr::Exchange(address(),r,spans,response);
  }
  Status LookupRegion(std::uint64_t base,peer::PeerRegion* out)const override {
    if(!out || base!=endpoint.base || endpoint.segment_bytes!=48) return Status::InvalidArgument("invalid imported B HBM");
    *out={};out->base=endpoint.base;out->length=endpoint.bytes;out->token_value=endpoint.token;
    std::memcpy(&out->token_id,endpoint.segment+44,4);
    out->segment_key_bytes=endpoint.segment_bytes;std::memcpy(out->segment_key.data(),endpoint.segment,endpoint.segment_bytes);
    std::memcpy(out->eid.data(),endpoint.eid,16);
    out->has_memory_eid=true;std::memcpy(out->memory_eid.data(),endpoint.segment,16);
    return Status::OK();
  }
  Status CreateChannel(std::uint64_t base,const std::uint8_t* key,std::uint32_t key_bytes,
      const std::uint8_t* local_eid,std::uint32_t token,std::uint64_t tp,std::uint32_t psn,
      std::array<std::uint8_t,peer::kQpKeyBytes>* peer_key,std::uint32_t* peer_bytes,
      std::uint64_t* peer_tp,std::uint32_t* peer_psn)override {
    if(base!=endpoint.base || channel>=offload::kChannelCount || key_bytes!=32 || token!=ticket.token)
      return Status::InvalidArgument("invalid NPU raw READ channel");
    auto r=hr::Make(hr::Command::Channel,ticket,client);r.channel=channel;
    r.endpoint.tp=tp;r.endpoint.psn=psn;r.endpoint.token=token;r.endpoint.key_bytes=key_bytes;
    std::memcpy(r.endpoint.eid,local_eid,16);std::memcpy(r.endpoint.key[0],key,key_bytes);
    hr::Response response{};auto s=hr::Exchange(address(),r,{},&response);if(!s.ok())return s;
    if(response.endpoint.key_bytes!=32 || response.endpoint.token!=token) return Status::Internal("invalid B target descriptor");
    *peer_bytes=response.endpoint.key_bytes;std::memcpy(peer_key->data(),response.endpoint.key[channel],*peer_bytes);
    *peer_tp=response.endpoint.tp;*peer_psn=response.endpoint.psn;++channel;return Status::OK();
  }
};
struct Engine {
  KvafBatchGetTransport* transport{};
  KvafBatchGetLocalBuffer* local{};
  DeviceBuffer descriptors,statuses,queues,scratch;
  std::uint32_t cores,jettys,budget;
  bool uncertain{},kernel_finished{};
  aclrtStream probe_stream{};
  Engine(std::uint32_t c,std::uint32_t j,std::uint32_t b):cores(c),jettys(j),budget(b){}
  Status Cleanup() {
    if(uncertain)return Status::InDoubt("raw READ transport quarantined; source and output must remain alive");
    if(KvafBatchGetLocalBufferDiscard(&local))return Status::InDoubt("raw READ output unregister pending");
    auto s=descriptors.Free();if(!s.ok())return s;
    return statuses.Free();
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
    if(KvafBatchGetTransportClose(&transport))return Status::InDoubt("raw READ transport teardown pending");
    s=queues.Free();if(!s.ok())return s;
    if(probe_stream) {
      if(aclrtDestroyStream(probe_stream)!=ACL_SUCCESS)return Status::InDoubt("destroy raw probe stream");
      probe_stream=nullptr;
    }
    return scratch.Free();
  }
  Status Run(const std::vector<hr::Span>& spans,MutableBufferView output,void* stream) {
    if(!transport || !stream || uncertain || local || descriptors.data || statuses.data)
      return Status::InvalidState("raw READ engine is not idle");
    std::vector<std::uint64_t> src,dst;std::vector<std::uint32_t> sizes;
    for(auto span:spans) {
      while(span.bytes) {
        if(src.size()==kMaxWqes)return Status::InvalidArgument("raw READ exceeds 1048576 WQEs");
        auto n=std::min<std::uint64_t>(span.bytes,65536);
        src.push_back(span.source);dst.push_back(span.destination);sizes.push_back(n);
        span.source+=n;span.destination+=n;span.bytes-=n;
      }
    }
    if(src.empty())return Status::InvalidArgument("empty raw READ");
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
                   MutableBufferView output,void* stream) {
    if(!transport || !stream || uncertain || local || descriptors.data || statuses.data)
      return Status::InvalidState("raw READ engine is not idle");
    KvafExecuteBatchGetRawAddrKernelArgs args{};
    args.source_addresses=reinterpret_cast<std::uint64_t>(src.data);
    args.destination_addresses=reinterpret_cast<std::uint64_t>(dst.data);
    args.lengths=reinterpret_cast<std::uint64_t>(lengths.data);
    args.request_count=src.size;args.input_mode=KVAF_RAW_NPU_OFFSETS;
    return RunKernel(args,output,stream);
  }
  Status RunKernel(KvafExecuteBatchGetRawAddrKernelArgs args,MutableBufferView output,void* stream) {
    auto s=statuses.Allocate(cores*sizeof(KvafRawReadCoreStatus));if(!s.ok())return s;
    if(!queues.data) {
      s=queues.Allocate(cores*jettys*sizeof(KvafBatchGetQueueState));if(!s.ok())return s;
      if(aclrtMemset(queues.data,queues.bytes,0,queues.bytes)!=ACL_SUCCESS)return Status::Internal("clear raw READ queues");
    }
    if(aclrtMemset(statuses.data,statuses.bytes,0,statuses.bytes)!=ACL_SUCCESS)
      return Status::Internal("clear raw READ status");
    std::uint64_t entity{};
    if(KvafBatchGetLocalBufferAcquire(transport,output.data,output.bytes,&local,&entity))
      return Status::Internal("register A raw READ output");
    args.output=reinterpret_cast<std::uint64_t>(output.data);
    args.core_status=reinterpret_cast<std::uint64_t>(statuses.data);args.queue_state=reinterpret_cast<std::uint64_t>(queues.data);
    args.channel_entity=entity;args.block_count=cores;args.jettys_per_core=jettys;args.batch_get_max_inflight_total=budget;
    kernel_finished=false;
    uncertain=true; // Includes launch errors: never assume the device saw no work.
    if(KvafExecuteBatchGetRawAddrLaunch(&args,static_cast<aclrtStream>(stream)) ||
       aclrtSynchronizeStream(static_cast<aclrtStream>(stream))!=ACL_SUCCESS)
      return Status::InDoubt("raw READ launch/stream completion uncertain");
    kernel_finished=true;
    std::vector<KvafRawReadCoreStatus> host(cores);
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
    if(!n || n>kMaxWqes || (args.input_mode==KVAF_RAW_NPU_OFFSETS ? n<args.request_count : n!=args.request_count))
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
enum class Pending { None, Acquire, Dma, Cleanup, Complete, Cancel };
struct Connection {
  std::unique_ptr<ReadLink> link;
  Engine engine;
  Pending pending{Pending::None};
  std::uint64_t sequence{},last_sequence{};
  bool ready{};
  Status result;
  Connection(const std::string& a,const HbmReadTicket& t,std::uint32_t c,std::uint32_t j,std::uint32_t b):
      link(new ReadLink(a,t)),engine(c,j,b){}
  Status Resolve() {
    if(pending==Pending::None)return Status::InvalidState("no pending raw READ");
    if(pending==Pending::Dma || engine.uncertain)return Status::InDoubt("raw READ DMA unresolved; retain A output and B source; no READ replay");
    if(pending==Pending::Acquire) {result=Status::Internal("raw READ cancelled before DMA after uncertain Acquire");pending=Pending::Cancel;}
    auto s=engine.Cleanup();if(!s.ok())return s;
    if(pending==Pending::Cleanup)pending=result.ok()?Pending::Complete:Pending::Cancel;
    hr::Response response{};
    s=link->Call(pending==Pending::Complete?hr::Command::Complete:hr::Command::Cancel,{},&response,sequence);
    if(!s.ok())return Status::InDoubt("raw READ acknowledgement pending: "+s.message());
    pending=Pending::None;last_sequence=sequence;return result;
  }
};
}
struct HbmReader::Impl {
  std::uint32_t device,cores,jettys,budget;
  std::string kernel;
  LinkId next{1};
  std::map<LinkId,std::unique_ptr<Connection>> connections;
  Status Device()const {
    int current=-1;return aclrtGetDevice(&current)==ACL_SUCCESS && current==int(device)?Status::OK():
        Status::InvalidState("select A device before HBM READ operation");
  }
};
HbmReader::HbmReader(std::uint32_t d,std::uint32_t c,std::uint32_t j,std::uint32_t b,std::string k):
    impl_(new Impl{d,c,j,b,std::move(k)}){}
HbmReader::~HbmReader(){if(!Shutdown().ok())(void)impl_.release();}
Status HbmReader::Attach(const std::string& address,const HbmReadTicket& ticket,const std::string& local_eid,LinkId* out) {
  auto s=impl_->Device();if(!s.ok())return s;
  if(!out || !ticket.source_id || !ticket.epoch || !ticket.token || !ticket.bytes ||
      !impl_->cores || !impl_->jettys || impl_->cores*std::uint64_t(impl_->jettys)>offload::kChannelCount ||
      (impl_->budget && impl_->budget<impl_->cores*2) || impl_->kernel.empty())
    return Status::InvalidArgument("configure raw kernel and 1..4 total Jettys for HBM READ");
  if(KvafExecuteBatchGetRawAddrLoadKernel(impl_->kernel.c_str()))return Status::Internal("load batch_get_raw_addr kernel");
  std::vector<npu_eid::NpuLocalEidCandidate> candidates;
  if(local_eid.empty()) {
    std::uint32_t physical{};std::string error;
    if(!direct_hccp::GetCurrentNpuPhysicalDeviceId(&physical))return Status::Internal("resolve A physical device");
    if(!npu_eid::DiscoverPhysicalNpuEids(physical,&candidates,&error))return Status::Internal(error);
  } else { npu_eid::NpuLocalEidCandidate c{};if(!detail::ParseEid(local_eid,c.eid.data()))return Status::InvalidArgument("invalid A EID");candidates.push_back(c); }
  auto discover=hr::Make(hr::Command::Discover,ticket,0);hr::Response info{};
  s=hr::Exchange(address,discover,{},&info);if(!s.ok())return s;
  if(!info.candidates || info.candidates>256)return Status::Internal("invalid B EID candidate count");
  for(std::uint32_t b=0;b<info.candidates;++b)for(const auto& a:candidates) {
    *out=impl_->next++;
    auto owned=std::make_unique<Connection>(address,ticket,impl_->cores,impl_->jettys,impl_->budget);
    auto& c=*owned;impl_->connections.emplace(*out,std::move(owned));
    auto request=hr::Make(hr::Command::Open,ticket,c.link->client);request.candidate=b;
    std::memcpy(request.endpoint.eid,a.eid.data(),16);
    hr::Response response{};c.link->opened=true;
    s=hr::Exchange(address,request,{},&response);
    if(s.ok()) {
      c.link->endpoint=response.endpoint;c.link->checksum=response.checksum;
      auto& e=c.link->endpoint;
      if(e.token!=ticket.token || e.bytes!=ticket.bytes || e.segment_bytes!=48 || e.key_bytes!=32 ||
          !e.base || e.bytes>UINT64_MAX-e.base || !e.tp || !e.psn)
        s=Status::Internal("invalid B HBM export descriptor");
      else if(KvafBatchGetTransportCreateWithPeerLink(reinterpret_cast<void*>(e.base),e.bytes,
          impl_->cores,impl_->jettys,c.link.get(),a.eid.data(),&c.engine.transport))
        s=Status::Internal("create A-B raw READ transport");
    }
    if(s.ok()) {
      const auto bytes=std::min<std::uint64_t>(ticket.bytes,64);
      const auto rows=impl_->cores*(128U*(impl_->jettys-1)+1U);
      s=c.engine.scratch.Allocate(rows*bytes);
      auto& stream=c.engine.probe_stream;
      if(s.ok() && aclrtCreateStream(&stream)!=ACL_SUCCESS)s=Status::Internal("create raw READ probe stream");
      if(s.ok()) {
        auto base=reinterpret_cast<std::uint64_t>(c.engine.scratch.data);
        std::vector<hr::Span> probe;
        for(std::uint32_t i=0;i<rows;++i)probe.push_back({c.link->endpoint.base,base+i*bytes,bytes});
        s=c.engine.Run(probe,{c.engine.scratch.data,c.engine.scratch.bytes,DeviceKind::kNpu},stream);
        if(s.ok()) {
          std::vector<std::uint8_t> result(rows*bytes);
          if(aclrtMemcpy(result.data(),result.size(),c.engine.scratch.data,result.size(),ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS)
            s=Status::Internal("copy raw READ probe output");
          for(std::uint32_t i=0;s.ok() && i<rows;++i)
            if(offload::Hash(result.data()+i*bytes,bytes)!=c.link->checksum)s=Status::Internal("raw READ probe checksum mismatch");
        }
      }
      if(c.engine.uncertain) {
        const auto quiesced=c.engine.QuiesceProbe();
        if(!quiesced.ok()) {
          c.pending=Pending::Dma;
          return Status::InDoubt("HBM attach probe quarantined; region="+std::to_string(*out)+"; "+s.message()+"; "+quiesced.message());
        }
        s=Status::Internal("startup probe READ failed; transport quiesced: "+s.message());
      }
      if(stream) {
        if(aclrtDestroyStream(stream)==ACL_SUCCESS)stream=nullptr;
        else if(s.ok())s=Status::Internal("destroy raw READ probe stream");
      }
      if(s.ok())s=c.engine.Cleanup();
      if(s.ok())s=c.engine.scratch.Free();
      if(s.ok())s=c.link->Call(hr::Command::Confirm,{},&response);
    }
    if(s.ok()) {
      c.ready=true;
      std::fprintf(stderr,"[KVAF_NPU] HBM_READ_READY region=%llu A_eid=%s B_eid=%s jettys=%u probe=all_lanes_pass\n",
          (unsigned long long)*out,npu_eid::EidToHex(a.eid).c_str(),detail::EidHex(c.link->endpoint.eid).c_str(),impl_->cores*impl_->jettys);
      return s;
    }
    std::fprintf(stderr,"[KVAF_NPU] HBM_READ_CANDIDATE_REJECTED region=%llu A_eid=%s B_candidate=%u detail=%s\n",
        (unsigned long long)*out,npu_eid::EidToHex(a.eid).c_str(),b,s.message().c_str());
    auto cleanup=Detach(*out);
    if(!cleanup.ok())return Status::InDoubt("HBM attach cleanup pending; region="+std::to_string(*out)+"; "+cleanup.message());
  }
  return s;
}
Status HbmReader::Read(LinkId id,const HbmReadBatch& b,MutableBufferView output,
                       const std::vector<std::uint64_t>& offsets,void* stream,bool identity_only) {
  auto s=impl_->Device();if(!s.ok())return s;
  auto it=impl_->connections.find(id);if(it==impl_->connections.end())return Status::NotFound("HBM reader region not found");
  auto& c=*it->second;
  if(!c.ready || c.pending!=Pending::None)return Status::Busy("raw READ region not ready or has unresolved work");
  auto base=reinterpret_cast<std::uint64_t>(output.data);
  if(!output.valid() || output.device!=DeviceKind::kNpu || !stream || output.bytes>UINT64_MAX-base ||
      b.source_id!=c.link->ticket.source_id || b.epoch!=c.link->ticket.epoch || !b.sequence || b.sequence<=c.last_sequence ||
      b.lengths.empty() || b.lengths.size()>offload::kMaxSpans || b.source_addresses.size()!=b.lengths.size() || offsets.size()!=b.lengths.size())
    return Status::InvalidArgument("invalid raw READ identity, sequence or output");
  std::vector<hr::Span> spans,relative;
  std::size_t wqes=0;
  for(size_t i=0;i<offsets.size();++i) {
    auto src=b.source_addresses[i],n=b.lengths[i];
    if(src<c.link->endpoint.base || !offload::Contains(c.link->endpoint.bytes,src-c.link->endpoint.base,n) ||
       !offload::Contains(output.bytes,offsets[i],n))return Status::InvalidArgument("raw READ outside source/destination registration");
    auto count=n/65536+(n%65536!=0);if(count>kMaxWqes-wqes)return Status::InvalidArgument("too many raw READ WQEs");wqes+=count;
    spans.push_back({src,base+offsets[i],n});relative.push_back({src-c.link->endpoint.base,offsets[i],n});
  }
  if(!offload::ValidateSpans(relative,c.link->endpoint.bytes,output.bytes))return Status::InvalidArgument("raw READ destinations overlap");
  c.sequence=b.sequence;c.pending=Pending::Acquire;c.result=Status::OK();
  hr::Response response{};s=c.link->Call(identity_only?hr::Command::AcquireRaw:hr::Command::Acquire,spans,&response,b.sequence);
  if(!s.ok()) {
    if(s.code()!=StatusCode::kInDoubt)c.pending=Pending::None;
    return s;
  }
  c.pending=Pending::Dma;s=c.engine.Run(spans,output,stream);
  if(c.engine.uncertain)return s;
  c.result=s;c.pending=Pending::Cleanup;
  return c.Resolve();
}
Status HbmReader::Read(LinkId id,const ReadyToken& ready,NpuInt64ArrayView src,NpuInt64ArrayView dst,
                       NpuInt64ArrayView lengths,MutableBufferView output,void* stream) {
  auto s=impl_->Device();if(!s.ok())return s;
  auto it=impl_->connections.find(id);
  if(it==impl_->connections.end())return Status::NotFound("HBM reader link not found");
  auto& c=*it->second;
  if(!c.ready || c.pending!=Pending::None)return Status::Busy("raw READ region not ready or has unresolved work");
  const auto* source=std::get_if<NpuPullSourceDesc>(&ready.source);
  const auto& ticket=c.link->ticket;
  if(!source || !SameSource(*source,{ticket.source_id,ticket.bytes,ticket.epoch}) ||
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
  c.sequence=ready.sequence;c.pending=Pending::Acquire;c.result=Status::OK();
  hr::Response response{};
  // The source service leases the identity/batch, not a Host copy of the
  // device metadata. Each READ core validates the complete metadata before posting DMA.
  s=c.link->Call(hr::Command::AcquireRaw,{},&response,ready.sequence);
  if(!s.ok()) {if(s.code()!=StatusCode::kInDoubt)c.pending=Pending::None;return s;}
  c.pending=Pending::Dma;
  s=c.engine.RunDevice(src,dst,lengths,output,stream);
  if(c.engine.uncertain)return s;
  c.result=s;c.pending=Pending::Cleanup;
  return c.Resolve();
}
Status HbmReader::Resolve(LinkId id){auto s=impl_->Device();if(!s.ok())return s;auto it=impl_->connections.find(id);
  return it==impl_->connections.end()?Status::NotFound("HBM region not found"):it->second->Resolve();}
Status HbmReader::Detach(LinkId id){
  auto s=impl_->Device();if(!s.ok())return s;
  auto it=impl_->connections.find(id);if(it==impl_->connections.end())return Status::NotFound("HBM region not found");
  auto& c=*it->second;
  if(c.pending!=Pending::None)return Status::InDoubt("resolve raw READ before detach");
  s=c.engine.Close();if(!s.ok())return s;
  c.ready=false;
  if(c.link->opened){hr::Response response{};s=c.link->Call(hr::Command::Close,{},&response);
    if(!s.ok() && s.code()!=StatusCode::kNotFound)return s;}
  impl_->connections.erase(it);return Status::OK();
}
Status HbmReader::Shutdown(){while(!impl_->connections.empty()){auto s=Detach(impl_->connections.begin()->first);if(!s.ok())return s;}return Status::OK();}
bool HbmReader::pending()const{for(auto& item:impl_->connections)if(item.second->pending!=Pending::None)return true;return false;}
}
