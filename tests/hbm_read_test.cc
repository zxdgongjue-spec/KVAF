#include "peer_link.h"
#include "offload_fake_transport.h"
#include "hbm_read_client.h"
#include "hbm_read_protocol.h"
#include "kvaf_aiv_batch_get_raw_addr.h"
#include "npu_runtime.h"
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
#include "offload_fake_server.h"
#endif
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
    ValidateRawMetadata(src,dst,reinterpret_cast<const int64_t*>(sizes),a->request_count,
                        local->source_bytes,local->bytes,&result,&failed,a->validate_ranges);
  if(result)count=0;
  if(!result)for(unsigned i=0;i<a->request_count;++i){
    const uint64_t n=a->input_mode==KVAF_RAW_NPU_OFFSETS?reinterpret_cast<const int64_t*>(sizes)[i]:sizes[i];
    const uint64_t source=a->input_mode==KVAF_RAW_NPU_OFFSETS?local->source_base+src[i]:src[i];
    const uint64_t destination=a->input_mode==KVAF_RAW_NPU_OFFSETS?local->base+dst[i]:dst[i];
    Require(n && destination>=local->base && n<=local->bytes && destination-local->base<=local->bytes-n,"destination registration");
    std::memcpy(reinterpret_cast<void*>(destination),reinterpret_cast<void*>(source),n);
    ++raw_wqes;
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
void RuntimeInitialization() {
  raw_mode=true;fake_device=0;
  std::vector<uint8_t> source(4096,0x67), output(256,0);
  kvaf::NpuPullHbmSource b(0,"00000000000000000000000000000001");
  kvaf::HbmReadTicket ticket;
  Check(b.Register({source.data(),source.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1),&ticket));
  Check(b.Listen("127.0.0.1:0"));
  const auto address="127.0.0.1:"+std::to_string(b.port());
  const auto eid="00000000000000000000000000000002";
  kvaf::NpuRuntimeOptions options;options.batch_get_raw_addr_kernel_object_path="fake.o";
  auto read=[&](kvaf::NpuRuntime& a) {
    kvaf::ReadyToken ready;Check(b.Ready(&ready));
    int64_t src=11,dst=7,n=64;output.assign(output.size(),0);
    Check(a.BatchGetRawAddr(ready,{&src,1},{&dst,1},{&n,1},
          {output.data(),output.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)));
    Require(!std::memcmp(output.data()+dst,source.data()+src,n),"Runtime raw payload survives DDR lifecycle");
    Check(b.ReleaseBatch(ready));
  };
  {
    kvaf::NpuRuntime a(options);
    Require(!a.initialized()&&!a.ddr_initialized(),"construction does not initialize DDR or device");
    Require(!a.InitializeDdr(4096,256).ok(),"DDR requires base initialization");
    Require(!a.AttachHbm(address,ticket,eid).ok(),"attach requires base initialization");
    Check(a.Initialize());Check(a.Initialize());
    Require(a.initialized()&&!a.ddr_initialized(),"raw-only base initialization without CPU service");
    kvaf::PutHandle put;
    Require(!a.PreparePut({1},{64},&put).ok(),"base initialization cannot satisfy DDR operations");
    bool rejected=false;try{(void)a.core();}catch(const std::logic_error&){rejected=true;}
    Require(rejected,"raw-only Runtime exposes no DDR metadata core");
    Check(a.AttachHbm(address,ticket,eid));read(a);
    const auto transports=raw_transports.load(), buffers=local_buffers.load();
    Require(!a.Initialize(4096,256).ok(),"combined initializer still requires CPU target");
    Check(a.ShutdownDdr());Check(a.ShutdownDdr());
    Require(a.initialized()&&!a.ddr_initialized()&&b.connected()&&raw_transports==transports&&local_buffers==buffers,
            "DDR-only cleanup preserves raw connections and output MR cache");
    read(a);
    unregister_failures=1;
    Require(!a.Shutdown().ok()&&a.initialized(),"failed raw cleanup preserves base state");
    Require(!a.Initialize().ok()&&!a.AttachHbm(address,ticket,eid).ok(),"unfinished global shutdown blocks initialization");
    Check(a.Shutdown());Require(!a.initialized()&&!a.ddr_initialized()&&!b.connected(),"global shutdown closes raw-only Runtime");
    Check(a.Initialize());Check(a.AttachHbm(address,ticket,eid));read(a);Check(a.DetachHbm(ticket));Check(a.Shutdown());
  }
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  options.enable_cpu_offload=true;
  for(bool ddr_first:{false,true}) {
    SharedServer cpu;kvaf::NpuRuntime a(options);
    if(ddr_first) {Check(a.LinkTo({cpu.address}));Check(a.Initialize(4096,256));}
    else Check(a.Initialize());
    Check(a.AttachHbm(address,ticket,eid));read(a);
    const auto transports=raw_transports.load(), buffers=local_buffers.load();
    if(!ddr_first) {
      Check(a.LinkTo({cpu.address}));Require(cpu.pool_allocations==0,"link_to only configures DDR target");
      cpu.drop=uint16_t(rp::Command::Describe);cpu.fail_release=true;
      Require(!a.InitializeDdr(4096,256).ok()&&!a.ddr_initialized()&&a.initialized(),"failed DDR init preserves base");
      Require(!a.InitializeDdr(4096,256).ok()&&cpu.pool_allocations==1,"unresolved DDR cannot be replaced");
      read(a);Check(a.ShutdownDdr());Check(a.InitializeDdr(4096,256));
    }
    Require(a.ddr_initialized(),"both initialization orders support DDR");
    Require(!a.InitializeDdr(4096,256).ok(),"double DDR init rejected");
    kvaf::PutHandle old;Check(a.PreparePut({1},{64},&old));Check(a.AbortPut(old));
    cpu.fail_release=true;
    Require(!a.ShutdownDdr().ok(),"lost DDR release retains cleanup state");
    read(a);Check(a.ShutdownDdr());
    Require(a.initialized()&&!a.ddr_initialized()&&cpu.pool.empty()&&raw_transports==transports&&local_buffers==buffers,
            "DDR teardown does not unregister raw output or tear down its TP");
    read(a);Check(a.InitializeDdr(4096,256));
    kvaf::CpuPullState state;
    Require(!a.QueryCpuPull(old,&state).ok(),"old PutHandle cannot access a reinitialized DDR core");
    Check(a.ReleaseLink(1));Require(a.initialized()&&!a.ddr_initialized()&&b.connected(),"ReleaseLink affects DDR only");
    read(a);Check(a.LinkTo({cpu.address}));Check(a.InitializeDdr(4096,256));
    Check(a.DetachHbm(ticket));Require(a.ddr_initialized(),"HBM detach preserves DDR");
    Check(a.AttachHbm(address,ticket,eid));read(a);Check(a.Shutdown());
    Require(!a.initialized()&&!a.ddr_initialized()&&cpu.pool.empty()&&!b.connected(),"mixed shutdown releases both modules");
  }
#endif
  Check(b.Shutdown());
  Require(!raw_transports&&!local_buffers&&!registrations&&!sessions,"Runtime lifecycle released raw resources");
  raw_calls=0;raw_wqes=0;quiesces=0;
}
int main(){try{
  RuntimeInitialization();
  raw_mode=true;fake_device=0;
  constexpr size_t source_bytes=400128;
  std::vector<uint8_t> storage(source_bytes+8192,0xee),output(300000,0xa5);
  auto* source=reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(storage.data())+4095U)&~uintptr_t(4095U))+0xc00;
  for(size_t i=0;i<source_bytes;++i)source[i]=(i*17+3)%251;
  kvaf::NpuPullHbmSource b(0,"00000000000000000000000000000001");kvaf::HbmReadTicket ticket;
  Check(b.Register({source,source_bytes,kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1),&ticket));Check(b.Listen("127.0.0.1:0"));
  Proxy proxy("127.0.0.1:"+std::to_string(b.port()));
  {kvaf::HbmReader failed(0,2,2,2048,"fake.o");
    fail_probe_cq=true;
    Require(!failed.Attach(proxy.address,ticket,"00000000000000000000000000000002").ok(),"probe CQ failure rejected");
    Require(quiesces==1 && !failed.pending() && !b.connected() && !registrations && !raw_transports && !local_buffers,
            "failed probe quiesces all Jettys before retiring either side");
    Check(failed.Shutdown());raw_calls=0;}
  {kvaf::HbmReader failed(0,2,2,2048,"fake.o");
    fail_probe_cq=true;proxy.drop=unsigned(hr::Command::Close);
    Require(!failed.Attach(proxy.address,ticket,"00000000000000000000000000000002").ok(),"failed attach retains lost cleanup reply");
    Check(failed.Detach(ticket));Check(failed.Detach(ticket));Check(failed.Shutdown());
    Require(!b.connected() && !raw_transports && !registrations,"original export ticket recovers failed attachment");
    raw_calls=0;}
  kvaf::HbmReader a(0,2,2,2048,"fake.o");
  Check(a.Attach(proxy.address,ticket,"00000000000000000000000000000002"));
  Require(registered_base==reinterpret_cast<uintptr_t>(source)-0xc00 && registered_bytes==source_bytes+0xc00,
          "register aligned range covering the unaligned Tensor");
  Require(b.connected() && raw_calls==1,"all-lane probe completed");
  Check(a.Attach(proxy.address,ticket,"00000000000000000000000000000002"));
  Require(raw_calls==1 && raw_transports==1,"repeated attach reuses the saved connection");
  auto wrong_ticket=ticket;wrong_ticket.token++;
  Require(!a.Attach(proxy.address,wrong_ticket,"00000000000000000000000000000002").ok(),"conflicting ticket rejected");
  Require(!a.Detach(wrong_ticket).ok() && b.connected(),"wrong ticket cannot detach a saved source");
  Require(!a.Attach(proxy.address,ticket,"00000000000000000000000000000004").ok(),"conflicting route rejected");
  { kvaf::HbmReader other(0,2,2,2048,"fake.o");
    Require(!other.Attach(proxy.address,ticket,"00000000000000000000000000000002").ok(),"reject second reader");
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
    const auto status=a.Read(t,{src.data(),src.size()},{dst.data(),dst.size()},
                             {sizes.data(),sizes.size()},out,reinterpret_cast<void*>(1),nullptr,true);
    Require(raw_calls-calls<=1 && raw_syncs-syncs==raw_calls-calls,"single fused launch and final synchronization");
    return status;
  };
  const auto saved=token;auto bad=token;std::get<kvaf::NpuPullSourceDesc>(bad.source).source_epoch++;
  before=raw_calls;const auto before_wqes=raw_wqes.load();
  Require(!read(bad,sources,destinations,lengths).ok(),"raw-array stale identity rejected");
  bad=token;std::get<kvaf::NpuPullSourceDesc>(bad.source).source_bytes++;
  Require(!read(bad,sources,destinations,lengths).ok(),"raw-array source size mismatch rejected");
  bad=token;bad.source=kvaf::CpuPullSourceDesc{};
  Require(!read(bad,sources,destinations,lengths).ok(),"raw READ rejects CPU Ready tokens");
  Require(!a.Read(token,{reinterpret_cast<const int64_t*>(output.data()),sources.size()},
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
  Check(a.Read(token,{sources.data(),sources.size()},{destinations.data(),destinations.size()},
               {lengths.data(),lengths.size()},out,reinterpret_cast<void*>(1),&profile));
  Require(raw_wqes-before_wqes==sources.size(),"one WQE per request, including lengths above 64 KiB");
  const double host_phases=profile.validation_us+profile.acquire_us+profile.buffers_us+profile.register_output_us+
      profile.event_record_us+profile.launch_us+profile.synchronize_us+profile.event_query_us+
      profile.status_us+profile.cleanup_us+profile.complete_us;
  Require(profile.kernel_event_valid && profile.kernel_event_us==125.0 && !profile.event_error,
          "device event milliseconds converted to microseconds");
  Require(host_phases<=profile.total_us && profile.total_us>0 && events==2,
          "Host stages fit total; timing events retained for reuse");
  Check(b.ReleaseBatch(token));Check(b.Ready(&token));
  fail_timing_query=true;
  Check(a.Read(token,{sources.data(),sources.size()},{destinations.data(),destinations.size()},
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
  auto wrong_batch=token;wrong_batch.sequence++;
  Require(!a.Resolve(wrong_batch).ok() && a.pending(),"wrong sequence cannot resolve another batch");
  Require(!a.Resolve(saved).ok() && a.pending(),"stale Ready cannot resolve current batch");
  Check(a.Resolve(token));Require(raw_calls==before+1,"new raw API resolve does not replay DMA");
  Require(timing_queries==queries,"unprofiled READ and recovery do not query timing events");
  Require(!std::memcmp(output.data()+7,source+9,67)&&!std::memcmp(output.data()+5000,source+90001,70001),"raw-array offsets are relative to logical source/output");
  Check(b.ReleaseBatch(token));Check(b.Ready(&token));
  Require(!b.ReleaseBatch(saved).ok()&&!b.CancelBatch(saved).ok(),"explicit source token rejects delayed notifications");
  proxy.drop=unsigned(hr::Command::AcquireRaw);before=raw_calls;
  Require(!read(token,sources,destinations,lengths).ok()&&a.pending(),"lost new Acquire response retains batch");
  Require(!b.CancelBatch(token).ok(),"cannot cancel an acquired identity-only batch");
  Require(!a.Resolve(token).ok()&&!a.pending()&&raw_calls==before,"unknown Acquire is cancelled without READ replay");
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
  unregister_failures=1;Require(!a.Detach(ticket).ok() && local_buffers==3,"failed detach preserves cached registrations");
  Check(b.Ready(&token));before=raw_calls;
  Require(!read(token,sources,destinations,lengths).ok() && raw_calls==before,"partially closed transport rejects further READs");
  Check(b.CancelBatch(token));
  proxy.drop=unsigned(hr::Command::Close);Require(!a.Detach(ticket).ok(),"lost detach reply");Check(a.Detach(ticket));
  Check(a.Detach(ticket));
  // Exercise real persistent connections and legacy one-shot calls together.
  int control=-1;hr::Response response{};auto discover=hr::Make(hr::Command::Discover,ticket,99);
  const auto address="127.0.0.1:"+std::to_string(b.port());
  {kvaf::HbmReader direct(0,2,2,2048,"fake.o");
    Check(direct.Attach(address,ticket,"00000000000000000000000000000002"));
    std::vector<uint8_t> other_storage(source_bytes,0x77);
    kvaf::NpuPullHbmSource other(0,"00000000000000000000000000000001");kvaf::HbmReadTicket other_ticket;
    Check(other.Register({other_storage.data(),other_storage.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1),&other_ticket));
    Check(other.Listen("127.0.0.1:0"));
    Check(direct.Attach("127.0.0.1:"+std::to_string(other.port()),other_ticket,"00000000000000000000000000000002"));
    kvaf::ReadyToken other_ready;Check(other.Ready(&other_ready));
    Check(direct.Read(other_ready,{sources.data(),sources.size()},{destinations.data(),destinations.size()},
                      {lengths.data(),lengths.size()},out,reinterpret_cast<void*>(1)));
    Require(!std::memcmp(output.data()+7,other_storage.data()+9,67),"Ready selects second saved NPU source");
    Check(other.ReleaseBatch(other_ready));Check(direct.Detach(other_ticket));Check(other.Shutdown());
    for(unsigned i=0;i<3;++i) {
      Check(b.Ready(&token));
      Check(direct.Read(token,{sources.data(),sources.size()},{destinations.data(),destinations.size()},
                        {lengths.data(),lengths.size()},out,reinterpret_cast<void*>(1)));
      Check(b.ReleaseBatch(token));
      Require(!std::memcmp(output.data()+7,source+9,67),"detaching one source preserves the other connection");
    }
    Check(direct.Detach(ticket));
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
  std::cout<<"HBM raw READ: real TCP/source/client, one WQE per span, leases, lost ACKs and cleanup passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
