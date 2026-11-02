#include "peer_link.h"
#include "offload_fake_transport.h"
#include "hbm_read_client.h"
#include "hbm_read_protocol.h"
#include "kvaf_aiv_batch_get_raw_addr.h"
#include <poll.h>
#include "aiv_model/kernel_operator.h"
#define __attribute__(...)
#include "../access_operator/demo/aiv_batch_get/kernel/execute_batch_get_raw_addr_aiv.asc"
#undef __attribute__
namespace aiv_model {
void Poll(void*) { std::abort(); }
void Doorbell(uint32_t, uint32_t*) { std::abort(); }
}

using kvaf::Status;
namespace hr=kvaf::hbm_read;
std::atomic<int> raw_calls{0}, raw_wqes{0}, local_buffers{0}, raw_transports{0}, local_acquires{0};
int unregister_failures=0;
bool fail_probe_cq=false;
int quiesces=0, raw_syncs=0;
extern "C" aclError aclrtSynchronizeStream(aclrtStream){++raw_syncs;return 0;}
struct KvafBatchGetLocalBuffer {
  std::uint64_t base,bytes,source_base,source_bytes;
  KvafBatchGetTransport* transport;unsigned references;
};
struct KvafBatchGetTransport {
  kvaf::peer::PeerLink* link;std::uint64_t base,bytes;
  std::vector<std::unique_ptr<KvafBatchGetLocalBuffer>> buffers;
};
extern "C" int32_t KvafExecuteBatchGetRawAddrLoadKernel(const char*){return 0;}
extern "C" int32_t KvafBatchGetTransportCreateWithPeerLink(void* base,std::uint64_t bytes,std::uint32_t cores,
    std::uint32_t jettys,void* peer,const std::uint8_t* eid,KvafBatchGetTransport** out){
  auto* link=static_cast<kvaf::peer::PeerLink*>(peer);kvaf::peer::PeerRegion region;
  Require(link->LookupRegion(reinterpret_cast<std::uint64_t>(base),&region).ok(),"lookup B region");
  Require(region.length==bytes && region.has_memory_eid && region.memory_eid[15]==3 && region.eid[15]==1,
          "preserve memory identity separately from route");
  Require(region.token_id==0x110000,"preserve raw NPU token id");
  for(unsigned i=0;i<cores*jettys;++i){
    std::uint8_t key[64]{};key[15]=4;key[20]=10+i;
    std::array<std::uint8_t,64> remote{};std::uint32_t remote_bytes{},psn{};std::uint64_t tp{};
    Require(link->CreateChannel(region.base,key,32,eid,region.token_value,2+i,2,&remote,&remote_bytes,&tp,&psn).ok(),"bind NPU-to-NPU channel");
    Require(remote_bytes==32 && remote[15]==3 && psn==1 && tp==i+1,"NPU target raw descriptor");
  }
  ++raw_transports;*out=new KvafBatchGetTransport{link,reinterpret_cast<std::uint64_t>(base),bytes};return 0;
}
extern "C" int32_t KvafBatchGetLocalBufferAcquire(KvafBatchGetTransport* transport,void* output,std::uint64_t bytes,
    KvafBatchGetLocalBuffer** out,std::uint64_t* entity){
  const auto base=reinterpret_cast<uint64_t>(output);
  for(auto& b:transport->buffers)if(b->base==base && b->bytes==bytes){
    ++b->references;*out=b.get();*entity=reinterpret_cast<uint64_t>(*out);return 0;}
  ++local_buffers;++local_acquires;
  transport->buffers.emplace_back(new KvafBatchGetLocalBuffer{base,bytes,transport->base,transport->bytes,transport,1});
  *out=transport->buffers.back().get();*entity=reinterpret_cast<uint64_t>(*out);return 0;
}
extern "C" void KvafBatchGetLocalBufferRelease(KvafBatchGetLocalBuffer* p){
  if(p){Require(p->references>0,"release live output registration");--p->references;}}
extern "C" int32_t KvafBatchGetLocalBufferDiscard(KvafBatchGetLocalBuffer** p){
  if(!*p)return 0;if(unregister_failures){--unregister_failures;return -1;}
  Require((*p)->references<=1,"discard output with no outstanding users");
  auto& buffers=(*p)->transport->buffers;
  auto it=std::find_if(buffers.begin(),buffers.end(),[&](const auto& b){return b.get()==*p;});
  Require(it!=buffers.end(),"discard cached registration");buffers.erase(it);--local_buffers;*p=nullptr;return 0;}
extern "C" int32_t KvafBatchGetTransportQuiesce(KvafBatchGetTransport*){++quiesces;return 0;}
extern "C" int32_t KvafBatchGetTransportClose(KvafBatchGetTransport** p){
  if(*p){
    for(const auto& b:(*p)->buffers)Require(!b->references,"close only after READ references released");
    while(!(*p)->buffers.empty()){auto* b=(*p)->buffers.back().get();if(KvafBatchGetLocalBufferDiscard(&b))return -1;}
    --raw_transports;delete *p;*p=nullptr;
  }return 0;
}
extern "C" int32_t KvafExecuteBatchGetRawAddrLaunch(const KvafExecuteBatchGetRawAddrKernelArgs* a,aclrtStream){
  ++raw_calls;auto* src=reinterpret_cast<uint64_t*>(a->source_addresses);auto* dst=reinterpret_cast<uint64_t*>(a->destination_addresses);
  auto* sizes=reinterpret_cast<uint32_t*>(a->lengths);auto* local=reinterpret_cast<KvafBatchGetLocalBuffer*>(a->channel_entity);
  uint32_t count=a->request_count;uint64_t result=0,failed=UINT64_MAX;
  if(a->input_mode==KVAF_RAW_NPU_OFFSETS)
    ValidateAndCountRaw(src,dst,reinterpret_cast<const int64_t*>(sizes),a->request_count,
                        local->source_bytes,local->bytes,&count,&result,&failed,a->validate_ranges);
  if(!result)for(unsigned i=0;i<a->request_count;++i){
    const uint64_t n=a->input_mode==KVAF_RAW_NPU_OFFSETS?reinterpret_cast<const int64_t*>(sizes)[i]:sizes[i];
    const uint64_t source=a->input_mode==KVAF_RAW_NPU_OFFSETS?local->source_base+src[i]:src[i];
    const uint64_t destination=a->input_mode==KVAF_RAW_NPU_OFFSETS?local->base+dst[i]:dst[i];
    Require(n && destination>=local->base && n<=local->bytes && destination-local->base<=local->bytes-n,"destination registration");
    std::memcpy(reinterpret_cast<void*>(destination),reinterpret_cast<void*>(source),n);
    raw_wqes+=n/65536+(n%65536!=0);
  }
  auto* statuses=reinterpret_cast<KvafRawReadCoreStatus*>(a->core_status);
  for(unsigned c=0;c<a->block_count;++c){auto& status=statuses[c];
    status.execution.magic=KVAF_AIV_BATCH_GET_STATUS_MAGIC;
    status.execution.result=fail_probe_cq?7:result;status.execution.failed_request_index=failed;
    status.execution.processed_request_count=uint64_t(count)*(c+1)/a->block_count-uint64_t(count)*c/a->block_count;
    status.total_wqes=count;status.posted_wqes=status.execution.processed_request_count;
  }
  fail_probe_cq=false;return 0;
}
class Proxy {
 public:
  std::string upstream,address;int fd=-1;std::atomic<bool> stop{false};std::atomic<unsigned> drop{0};std::thread thread;
  explicit Proxy(std::string target):upstream(std::move(target)){
    uint16_t port;Require(hr::Listen("127.0.0.1:0",&fd,&port).ok(),"listen proxy");address="127.0.0.1:"+std::to_string(port);
    thread=std::thread([this]{while(!stop){pollfd p{fd,POLLIN,0};if(poll(&p,1,20)<=0)continue;int client=accept(fd,nullptr,nullptr);if(client<0)continue;
      kvaf::offload::SetSocketTimeout(client);hr::Request r{};
      if(kvaf::offload::ReadBytes(client,&r,sizeof(r))){std::vector<hr::Span> spans(r.count);
        if(r.header.command==unsigned(hr::Command::AcquireRaw))Require(r.count==0,"NPU metadata is absent from the control message");
        if(kvaf::offload::ReadBytes(client,spans.data(),spans.size()*sizeof(hr::Span))){
          int server=hr::Connect(upstream);Require(server>=0,"connect source");hr::Response response{};
          Require(kvaf::offload::WriteBytes(server,&r,sizeof(r))&&kvaf::offload::WriteBytes(server,spans.data(),spans.size()*sizeof(hr::Span))&&
            kvaf::offload::ReadBytes(server,&response,sizeof(response)),"proxy exchange");close(server);
          // This proxy deliberately models a legacy one-request connection.
          response.reserved=0;
          if(drop.load()==r.header.command)drop=0;else kvaf::offload::WriteBytes(client,&response,sizeof(response));
        }}close(client);
    }});
  }
  ~Proxy(){stop=true;thread.join();close(fd);}
};
void Check(const Status& s){if(!s.ok())throw std::runtime_error(s.message());}
int main(){try{
  raw_mode=true;fake_device=0;
  constexpr size_t source_bytes=400128;
  std::vector<uint8_t> storage(source_bytes+8192,0xee),output(300000,0xa5);
  auto* source=reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(storage.data())+4095U)&~uintptr_t(4095U))+0xc00;
  for(size_t i=0;i<source_bytes;++i)source[i]=(i*17+3)%251;
  kvaf::NpuPullHbmSource b(0,"00000000000000000000000000000001");kvaf::HbmReadTicket ticket;
  Check(b.Register({source,source_bytes,kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1),&ticket));Check(b.Listen("127.0.0.1:0"));
  Proxy proxy("127.0.0.1:"+std::to_string(b.port()));
  {kvaf::HbmReader failed(0,2,2,2048,"fake.o");kvaf::LinkId rejected{};
    fail_probe_cq=true;
    Require(!failed.Attach(proxy.address,ticket,"00000000000000000000000000000002",&rejected).ok(),"probe CQ failure rejected");
    Require(quiesces==1 && !failed.pending() && !b.connected() && !registrations && !raw_transports && !local_buffers,
            "failed probe quiesces all Jettys before retiring either side");
    Check(failed.Shutdown());raw_calls=0;}
  kvaf::HbmReader a(0,2,2,2048,"fake.o");kvaf::LinkId id;
  Check(a.Attach(proxy.address,ticket,"00000000000000000000000000000002",&id));
  Require(registered_base==reinterpret_cast<uintptr_t>(source)-0xc00 && registered_bytes==source_bytes+0xc00,
          "register aligned range covering the unaligned Tensor");
  Require(b.connected() && raw_calls==1,"all-lane probe completed");
  { kvaf::HbmReader other(0,2,2,2048,"fake.o");kvaf::LinkId rejected{};
    Require(!other.Attach(proxy.address,ticket,"00000000000000000000000000000002",&rejected).ok(),"reject second reader");
    Check(other.Shutdown()); }
  Require(!b.Shutdown().ok(),"cannot unregister connected source");
  auto out=kvaf::MutableBufferView{output.data(),output.size(),kvaf::DeviceKind::kNpu};
  int before=0;
  // Connector chooses its own source ranges after identity-only Ready.
  kvaf::ReadyToken token;Check(b.Ready(&token));
  std::vector<std::int64_t> sources{9,90001},destinations{7,5000},lengths{67,70001};
  auto read=[&](const kvaf::ReadyToken& t,const std::vector<std::int64_t>& src,
                const std::vector<std::int64_t>& dst,const std::vector<std::int64_t>& sizes) {
    const auto calls=raw_calls.load(),syncs=raw_syncs;
    const auto status=a.Read(id,t,{src.data(),src.size()},{dst.data(),dst.size()},
                             {sizes.data(),sizes.size()},out,reinterpret_cast<void*>(1),nullptr,true);
    Require(raw_calls-calls<=1 && raw_syncs-syncs==raw_calls-calls,"single fused launch and final synchronization");
    return status;
  };
  const auto saved=token;auto bad=token;std::get<kvaf::NpuPullSourceDesc>(bad.source).source_epoch++;
  before=raw_calls;const auto before_wqes=raw_wqes.load();
  Require(!read(bad,sources,destinations,lengths).ok(),"raw-array stale identity rejected");
  Require(!a.Read(id,token,{reinterpret_cast<const int64_t*>(output.data()),sources.size()},
                   {destinations.data(),destinations.size()},{lengths.data(),lengths.size()},out,
                   reinterpret_cast<void*>(1)).ok() && raw_calls==before && !a.pending(),
          "metadata/output alias is rejected before Acquire");
  auto invalid_destination=destinations;invalid_destination[1]=output.size();
  Require(!read(token,sources,invalid_destination,lengths).ok(),"opt-in destination bounds enforced");
  Require(!a.pending(),"NPU validation failure cancels without unresolved DMA");
  Check(b.ReleaseBatch(token));Check(b.Ready(&token));
  Require(!read(token,sources,{destinations[0]},lengths).ok(),"raw-array count mismatch rejected");
  auto overrun=sources;overrun[1]=source_bytes;
  Require(!read(token,overrun,destinations,lengths).ok(),"raw-array source bounds enforced");
  Check(b.ReleaseBatch(token));Check(b.Ready(&token));
  Require(raw_calls==before+2 && raw_wqes==before_wqes,"fused validation rejects all invalid ranges before posting DMA");
  const int cached_acquires=local_acquires,allocated=acl_allocation_calls;
  kvaf::HbmReadProfile profile;
  Check(a.Read(id,token,{sources.data(),sources.size()},{destinations.data(),destinations.size()},
               {lengths.data(),lengths.size()},out,reinterpret_cast<void*>(1),&profile));
  const double host_phases=profile.validation_us+profile.acquire_us+profile.buffers_us+profile.register_output_us+
      profile.event_record_us+profile.launch_us+profile.synchronize_us+profile.event_query_us+
      profile.status_us+profile.cleanup_us+profile.complete_us;
  Require(profile.kernel_event_valid && profile.kernel_event_us==125.0 && !profile.event_error,
          "device event milliseconds converted to microseconds");
  Require(host_phases<=profile.total_us && profile.total_us>0 && events==2,
          "Host stages fit total; timing events retained for reuse");
  Check(b.ReleaseBatch(token));Check(b.Ready(&token));
  fail_timing_query=true;
  Check(a.Read(id,token,{sources.data(),sources.size()},{destinations.data(),destinations.size()},
               {lengths.data(),lengths.size()},out,reinterpret_cast<void*>(1),&profile));
  Require(!profile.kernel_event_valid && profile.event_error==123 && !a.pending(),
          "timing query failure does not change successful DMA result");
  Require(local_acquires==cached_acquires && local_buffers==1,"output registration reused across batches");
  Require(acl_allocation_calls==allocated,"status and queue allocations reused across batches");
  fail_timing_query=false;
  Check(b.ReleaseBatch(token));Check(b.Ready(&token));
  const int queries=timing_queries;
  before=raw_calls;
  proxy.drop=unsigned(hr::Command::Complete);
  Require(!read(token,sources,destinations,lengths).ok()&&a.pending(),"new raw API retains lost Complete");
  Check(a.Resolve(id));Require(raw_calls==before+1,"new raw API resolve does not replay DMA");
  Require(timing_queries==queries,"unprofiled READ and recovery do not query timing events");
  Require(!std::memcmp(output.data()+7,source+9,67)&&!std::memcmp(output.data()+5000,source+90001,70001),"raw-array offsets are relative to logical source/output");
  Check(b.ReleaseBatch(token));Check(b.Ready(&token));
  Require(!b.ReleaseBatch(saved).ok()&&!b.CancelBatch(saved).ok(),"explicit source token rejects delayed notifications");
  proxy.drop=unsigned(hr::Command::AcquireRaw);before=raw_calls;
  Require(!read(token,sources,destinations,lengths).ok()&&a.pending(),"lost new Acquire response retains batch");
  Require(!b.CancelBatch(token).ok(),"cannot cancel an acquired identity-only batch");
  Require(!a.Resolve(id).ok()&&!a.pending()&&raw_calls==before,"unknown Acquire is cancelled without READ replay");
  Require(!read(token,sources,destinations,lengths).ok(),"consumed sequence cannot replay");
  Check(b.ReleaseBatch(token));Check(b.Ready(&token));
  before=raw_calls;
  Check(read(token,sources,destinations,lengths));Check(b.ReleaseBatch(token));
  Require(raw_calls==before+1 && local_acquires==cached_acquires && local_buffers==1,
          "batch cleanup releases a reference without unregistering");
  // Different output sizes/addresses get separate cache entries and old ones
  // remain reusable, matching BatchGet's transport cache.
  --out.bytes;Check(b.Ready(&token));Check(read(token,sources,destinations,lengths));Check(b.ReleaseBatch(token));
  ++out.bytes;Check(b.Ready(&token));Check(read(token,sources,destinations,lengths));Check(b.ReleaseBatch(token));
  Require(local_acquires==cached_acquires+1 && local_buffers==2,"cache key includes output length");
  std::vector<uint8_t> second_output(output.size());auto original_out=out;
  out.data=second_output.data();Check(b.Ready(&token));Check(read(token,sources,destinations,lengths));Check(b.ReleaseBatch(token));
  out=original_out;
  Require(local_acquires==cached_acquires+2 && local_buffers==3,"cache key includes output address");
  Check(b.Ready(&token));Check(b.CancelBatch(token));Require(!read(token,sources,destinations,lengths).ok(),"cancelled batch cannot be read");
  unregister_failures=1;Require(!a.Detach(id).ok() && local_buffers==3,"failed detach preserves cached registrations");
  Check(b.Ready(&token));before=raw_calls;
  Require(!read(token,sources,destinations,lengths).ok() && raw_calls==before,"partially closed transport rejects further READs");
  Check(b.CancelBatch(token));
  proxy.drop=unsigned(hr::Command::Close);Require(!a.Detach(id).ok(),"lost detach reply");Check(a.Detach(id));
  // Exercise real persistent connections and legacy one-shot calls together.
  int control=-1;hr::Response response{};auto discover=hr::Make(hr::Command::Discover,ticket,99);
  const auto address="127.0.0.1:"+std::to_string(b.port());
  {kvaf::HbmReader direct(0,2,2,2048,"fake.o");kvaf::LinkId direct_id;
    Check(direct.Attach(address,ticket,"00000000000000000000000000000002",&direct_id));
    for(unsigned i=0;i<3;++i) {
      Check(b.Ready(&token));
      Check(direct.Read(direct_id,token,{sources.data(),sources.size()},{destinations.data(),destinations.size()},
                        {lengths.data(),lengths.size()},out,reinterpret_cast<void*>(1)));
      Check(b.ReleaseBatch(token));
    }
    Check(direct.Detach(direct_id));
  }
  Check(hr::Exchange(address,discover,{},&response,&control));Require(control>=0 && response.reserved==1,"persistent capability negotiated");
  const int connected=control;
  Check(hr::Exchange(address,discover,{},&response)); // idle cached fd does not monopolize the server
  Check(hr::Exchange(address,discover,{},&response,&control));Require(control==connected,"persistent control fd reused");
  shutdown(control,SHUT_RDWR);
  Require(!hr::Exchange(address,discover,{},&response,&control).ok() && control==-1,"uncertain socket discarded without automatic replay");
  Check(hr::Exchange(address,discover,{},&response,&control));
  // Shutdown must finish even while an idle persistent client is connected.
  Check(b.Shutdown());close(control);
  Require(!registrations && !contexts && !sessions && !cqs && !jetties && !peer_imports && !probe_allocations && !raw_transports && !local_buffers && !events,"all resources and profiling events retired");
  std::cout<<"HBM raw READ: real TCP/source/client, split spans, leases, lost ACKs and cleanup passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
