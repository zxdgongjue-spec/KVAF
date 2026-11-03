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
#include <map>
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
  std::map<uint64_t,std::pair<uint64_t,uint64_t>> memories;
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
  ++raw_transports;*out=new KvafBatchGetTransport{link,reinterpret_cast<std::uint64_t>(base),bytes};
  (*out)->memories[1]={reinterpret_cast<uint64_t>(base),bytes};return 0;
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
extern "C" int32_t KvafBatchGetTransportAddRemoteMemory(KvafBatchGetTransport* t,uint64_t mr,uint64_t base,uint64_t bytes){
  kvaf::peer::PeerRegion region;Require(t->link->LookupRegion(base,&region).ok() && region.length==bytes,"lookup additional MR");
  t->memories[mr]={base,bytes};return 0;
}
extern "C" int32_t KvafBatchGetLocalBufferAcquireForRemote(KvafBatchGetTransport* t,uint64_t mr,void* output,uint64_t bytes,
    KvafBatchGetLocalBuffer** out,uint64_t* entity){
  if(!t->memories.count(mr))return -1;
  auto s=KvafBatchGetLocalBufferAcquire(t,output,bytes,out,entity);
  if(!s){(*out)->source_base=t->memories.at(mr).first;(*out)->source_bytes=t->memories.at(mr).second;}
  return s;
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
// P connects to this proxy as if it were D. Inspect real framed traffic and
// drop selected acknowledgements after their side effects have happened.
class Proxy {
 public:
  kvaf::HbmReceiverTicket receiver;
  int fd=-1;std::atomic<bool> stop{false};std::atomic<unsigned> drop{0},requests{0};
  std::thread thread;std::mutex mutex;std::vector<int> sockets;std::vector<std::thread> workers;
  explicit Proxy(kvaf::HbmReceiverTicket target):receiver(target) {
    uint16_t port;Require(hr::Listen("127.0.0.1:0",&fd,&port).ok(),"listen reverse proxy");
    receiver.control_address="127.0.0.1:"+std::to_string(port);
    thread=std::thread([this,target]{while(!stop){pollfd p{fd,POLLIN,0};if(poll(&p,1,20)<=0)continue;
      int client=accept(fd,nullptr,nullptr);if(client<0)continue;
      int server=hr::Connect(target.control_address);Require(server>=0,"connect receiver");
      {std::lock_guard<std::mutex> l(mutex);sockets.push_back(client);sockets.push_back(server);}
      workers.emplace_back([this,client,server,target]{
        kvaf::offload::SetSocketTimeout(client);hr::Hello hello{};kvaf::HbmReadTicket ticket;
        if(hr::ReadHello(client,&hello,&ticket).ok() && hr::SendHello(server,static_cast<hr::Admission>(hello.header.command),target,ticket).ok()) {
          while(!stop) {
            pollfd p{server,POLLIN,0};if(poll(&p,1,20)<=0)continue;
            kvaf::peer::MessageHeader h{};if(!kvaf::offload::ReadBytes(server,&h,sizeof(h)))break;
            if(h.command==unsigned(hr::Admission::Result)) {
              hr::HelloResult result{};result.header=h;
              if(!kvaf::offload::ReadBytes(server,reinterpret_cast<char*>(&result)+sizeof(h),sizeof(result)-sizeof(h)))break;
              if(drop==h.command){drop=0;break;}
              if(!kvaf::offload::WriteBytes(client,&result,sizeof(result)))break;
              continue;
            }
            hr::Request request{};request.header=h;
            Require(h.bytes==sizeof(request) && h.command!=9,"no AcquireRaw or metadata on control path");
            if(!kvaf::offload::ReadBytes(server,reinterpret_cast<char*>(&request)+sizeof(h),sizeof(request)-sizeof(h)))break;
            ++requests;hr::Response response{};
            if(!kvaf::offload::WriteBytes(client,&request,sizeof(request)) || !kvaf::offload::ReadBytes(client,&response,sizeof(response)))break;
            if(drop==h.command){drop=0;break;}
            if(!kvaf::offload::WriteBytes(server,&response,sizeof(response)))break;
          }
        }
        std::lock_guard<std::mutex> l(mutex);
        for(auto socket:{client,server}){sockets.erase(std::remove(sockets.begin(),sockets.end(),socket),sockets.end());close(socket);}
      });
    }});
  }
  ~Proxy(){stop=true;thread.join();{std::lock_guard<std::mutex> l(mutex);for(int s:sockets)shutdown(s,SHUT_RDWR);}
    for(auto& w:workers)w.join();close(fd);}
};
void Check(const Status& s){if(!s.ok())throw std::runtime_error(s.message());}
template<class F> Status Retry(F f) {
  Status s=Status::InDoubt("not attempted");
  for(unsigned i=0;i<100;++i){s=f();if(s.ok())return s;std::this_thread::sleep_for(std::chrono::milliseconds(20));}
  return s;
}
const char* producer_eid="00000000000000000000000000000001";
const char* reader_eid="00000000000000000000000000000002";
void RuntimeInitialization() {
  raw_mode=true;fake_device=0;
  std::vector<uint8_t> source(4096,0x67),output(256);
  kvaf::NpuRuntimeOptions options;options.batch_get_raw_addr_kernel_object_path="fake.o";
  auto exercise=[&](bool with_ddr,bool ddr_first) {
    kvaf::NpuRuntime a(options);kvaf::HbmReceiverTicket receiver;
    Require(!a.ListenHbm("127.0.0.1:0","",reader_eid,&receiver).ok(),"listen requires initialize");
    Check(a.Initialize());Check(a.Initialize());
    Require(a.initialized()&&!a.ddr_initialized(),"raw-only initialization");
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
    SharedServer cpu;
    if(with_ddr){Check(a.LinkTo({cpu.address}));if(ddr_first)Check(a.InitializeDdr(4096,256));}
#endif
    Check(a.ListenHbm("127.0.0.1:0","",reader_eid,&receiver));
    kvaf::NpuPullHbmSource b(0,producer_eid);
    Check(b.Register({source.data(),source.size(),kvaf::DeviceKind::kNpu},receiver,reinterpret_cast<void*>(1)));
    auto read=[&] {
      kvaf::ReadyToken ready;Check(b.Ready(&ready));int64_t src=11,dst=7,n=64;
      Check(a.BatchGetRawAddr(ready,{&src,1},{&dst,1},{&n,1},{output.data(),output.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)));
      Require(!std::memcmp(output.data()+dst,source.data()+src,n),"raw payload survives DDR lifecycle");Check(b.ReleaseBatch(ready));
    };
    read();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
    if(with_ddr && !ddr_first)Check(a.InitializeDdr(4096,256));
    if(with_ddr){Check(a.ShutdownDdr());read();Check(a.InitializeDdr(4096,256));Check(a.ReleaseLink(1));}
#endif
    Check(a.ShutdownDdr());read();
    unregister_failures=1;
    Require(!a.Shutdown().ok()&&a.initialized(),"failed teardown retains runtime");
    Require(!a.Initialize().ok(),"closing runtime cannot reinitialize");
    Check(a.Shutdown());Check(b.Shutdown());
    Require(!a.initialized()&&!a.ddr_initialized(),"global teardown complete");
    Check(a.Initialize());Check(a.ListenHbm("127.0.0.1:0","",reader_eid,&receiver));Check(a.Shutdown());
  };
  exercise(false,false);
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  options.enable_cpu_offload=true;exercise(true,true);exercise(true,false);
#endif
  Require(!raw_transports&&!local_buffers&&!registrations&&!sessions,"runtime resources released");raw_calls=0;
}
void ReceiverTopology() {
  raw_mode=true;fake_device=0;std::vector<uint8_t> storage(8192,0x71),output(128);
  kvaf::ConstBufferView tensor{storage.data(),storage.size(),kvaf::DeviceKind::kNpu};
  kvaf::HbmReader d1(0,2,2,2048,"fake.o"),d2(0,2,2,2048,"fake.o");
  kvaf::HbmReceiverTicket r1,r2;
  Require(!d1.Listen("0.0.0.0:0","",reader_eid,&r1).ok(),"wildcard requires advertised address");
  Check(d1.Listen("0.0.0.0:0","127.0.0.1:0",reader_eid,&r1));
  Check(d2.Listen("127.0.0.1:0","",reader_eid,&r2));
  kvaf::NpuPullHbmSource p1(0,producer_eid),p2(0,producer_eid),rejected(0,producer_eid);
  auto stale=r1;++stale.receiver_epoch;
  Require(!rejected.Register(tensor,stale,reinterpret_cast<void*>(1)).ok(),"old D instance rejected");Check(rejected.Shutdown());
  Check(p1.Register(tensor,r1,reinterpret_cast<void*>(1)));Check(p2.Register(tensor,r2,reinterpret_cast<void*>(1)));
  Require(registrations==2&&raw_transports==2,"same P tensor separately connected to two D runtimes");
  int64_t src=9,dst=3,n=64;kvaf::ReadyToken t1,t2;Check(p1.Ready(&t1));Check(p2.Ready(&t2));
  Require(!d1.Read(t2,{&src,1},{&dst,1},{&n,1},{output.data(),output.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)).ok(),"other D source identity rejected");
  Check(d1.Cancel(t1));Check(p1.ReleaseBatch(t1));Check(p1.Shutdown());
  Check(d2.Read(t2,{&src,1},{&dst,1},{&n,1},{output.data(),output.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)));
  Require(!std::memcmp(output.data()+dst,storage.data()+src,n),"second D unaffected by first D teardown");
  Check(p2.ReleaseBatch(t2));Check(p2.Shutdown());Check(d1.Shutdown());Check(d2.Shutdown());
  Require(!registrations&&!raw_transports&&!local_buffers,"topology resources released");
}
void MultiMemory() {
  raw_mode=true;fake_device=0;
  std::vector<std::vector<uint8_t>> storage;
  std::vector<kvaf::ConstBufferView> tensors;
  for(unsigned i=0;i<78;++i)storage.emplace_back(4096+i*16,static_cast<uint8_t>(i+1));
  for(auto& t:storage)tensors.push_back({t.data()+3,t.size()-3,kvaf::DeviceKind::kNpu});
  kvaf::HbmReader a(0,2,2,2048,"fake.o");kvaf::HbmReceiverTicket receiver;
  Check(a.Listen("127.0.0.1:0","",reader_eid,&receiver));Proxy proxy(receiver);
  kvaf::NpuPullHbmSource b(0,producer_eid);
  auto duplicate=tensors;duplicate.push_back(tensors.front());
  Require(!b.Register(duplicate,proxy.receiver,reinterpret_cast<void*>(1)).ok(),"reject overlapping tensors");
  fail_memory_at=37;
  Require(!b.Register(tensors,proxy.receiver,reinterpret_cast<void*>(1)).ok(),"partial registration fails atomically");
  fail_memory_at=0;Check(Retry([&]{return b.Shutdown();}));
  Require(!registrations&&!contexts&&!jetties&&!raw_transports,"partial registration unwound");
  const auto probes=raw_calls.load();
  Check(b.Register(tensors,proxy.receiver,reinterpret_cast<void*>(1)));
  Require(raw_calls==probes+78&&registrations==78&&contexts==1&&jetties==4&&raw_transports==1,"78 MRs share four Jettys and one connection");
  kvaf::ReadyToken ready;
  Require(!b.Ready(&ready).ok(),"multi-MR ready needs tensor");
  Require(!b.Ready({storage[0].data(),storage[0].size(),kvaf::DeviceKind::kNpu},&ready).ok(),"exact tensor selection");
  std::vector<uint8_t> output(128);int64_t src=7,dst=11,n=64;
  for(unsigned i=0;i<156;++i) {
    const auto mr=i%78;Check(b.Ready(tensors[mr],&ready));
    Require(ready.mr_id==mr+1,"ready selects MR");
    Require(!b.Ready(tensors[(mr+1)%78],&ready).ok(),"cannot change frozen MR");
    auto bad=ready;bad.mr_id=79;
    Require(!a.Read(bad,{&src,1},{&dst,1},{&n,1},{output.data(),output.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)).ok(),"unknown MR rejected");
    Check(a.Read(ready,{&src,1},{&dst,1},{&n,1},{output.data(),output.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1)));
    Require(!std::memcmp(output.data()+dst,static_cast<const uint8_t*>(tensors[mr].data)+src,n),"selected MR payload");
    Check(b.ReleaseBatch(ready));
  }
  Check(b.Shutdown());Check(a.Shutdown());Require(!registrations&&!raw_transports&&!local_buffers,"multi-MR cleanup");
}
int main(){try{
  ReceiverTopology();
  MultiMemory();
  RuntimeInitialization();
  raw_mode=true;fake_device=0;
  constexpr size_t source_bytes=400128;
  std::vector<uint8_t> storage(source_bytes+8192,0xee),output(300000,0xa5);
  auto* source=reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(storage.data())+4095U)&~uintptr_t(4095U))+0xc00;
  for(size_t i=0;i<source_bytes;++i)source[i]=(i*17+3)%251;
  kvaf::HbmReader a(0,2,2,2048,"fake.o");kvaf::HbmReceiverTicket receiver;
  Check(a.Listen("127.0.0.1:0","",reader_eid,&receiver));Proxy proxy(receiver);
  kvaf::NpuPullHbmSource b(0,producer_eid);
  fail_probe_cq=true;
  Require(!b.Register({source,source_bytes,kvaf::DeviceKind::kNpu},proxy.receiver,reinterpret_cast<void*>(1)).ok(),"failed startup probe");
  Check(Retry([&]{return b.Shutdown();}));Require(!registrations&&!raw_transports&&!local_buffers,"failed probe cleanup");
  raw_calls=0;
  Check(b.Register({source,source_bytes,kvaf::DeviceKind::kNpu},proxy.receiver,reinterpret_cast<void*>(1)));
  Require(registered_base==reinterpret_cast<uintptr_t>(source)-0xc00&&registered_bytes==source_bytes+0xc00,"aligned registration covers logical tensor");
  Require(b.connected()&&raw_calls==1,"register returns after probe and confirmation");
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
  const double host_phases=profile.validation_us+profile.buffers_us+profile.register_output_us+
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
  Check(Retry([&]{return a.Resolve(token);}));Require(raw_calls==before+1,"new raw API resolve does not replay DMA");
  Require(timing_queries==queries,"unprofiled READ and recovery do not query timing events");
  Require(!std::memcmp(output.data()+7,source+9,67)&&!std::memcmp(output.data()+5000,source+90001,70001),"raw-array offsets are relative to logical source/output");
  Check(b.ReleaseBatch(token));Check(b.Ready(&token));
  Require(!b.ReleaseBatch(saved).ok()&&!b.CancelBatch(saved).ok(),"explicit source token rejects delayed notifications");
  before=raw_calls;
  Require(!b.CancelBatch(token).ok(),"published source cannot cancel unilaterally");
  proxy.drop=unsigned(hr::Command::Cancel);
  Require(!a.Cancel(token).ok()&&a.pending(),"lost cancel response retains fence");
  Require(!read(token,sources,destinations,lengths).ok()&&raw_calls==before,"pending cancellation blocks delayed READ");
  Check(Retry([&]{return a.Resolve(token);}));
  Check(a.Cancel(token));
  Check(b.ReleaseBatch(token));
  Require(!read(token,sources,destinations,lengths).ok()&&raw_calls==before,"cancelled sequence cannot replay");
  Check(b.Ready(&token));
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
  Check(b.Ready(&token));Check(a.Cancel(token));Check(b.CancelBatch(token));
  Require(!read(token,sources,destinations,lengths).ok(),"cancel fence survives producer release");
  // A second producer shares the same D listener and has independent resources.
  std::vector<uint8_t> other_storage(source_bytes,0x77);
  kvaf::NpuPullHbmSource other(0,producer_eid);
  Check(other.Register({other_storage.data(),other_storage.size(),kvaf::DeviceKind::kNpu},receiver,reinterpret_cast<void*>(1)));
  kvaf::ReadyToken other_ready;Check(other.Ready(&other_ready));Check(read(other_ready,sources,destinations,lengths));
  Require(!std::memcmp(output.data()+7,other_storage.data()+9,67),"Ready selects other source");
  Check(other.ReleaseBatch(other_ready));Check(other.Shutdown());
  Check(b.Ready(&token));const auto requests=proxy.requests.load();Check(read(token,sources,destinations,lengths));
  Require(proxy.requests==requests+1,"only Complete RPC per successful READ, no AcquireRaw");Check(b.ReleaseBatch(token));
  unregister_failures=1;Require(!b.Shutdown().ok(),"failed detach preserves source");
  Require(!b.Ready(&token).ok(),"closing source cannot publish");
  proxy.drop=unsigned(hr::Command::Close);Require(!b.Shutdown().ok(),"lost Close response retains teardown");
  Check(Retry([&]{return b.Shutdown();}));Check(a.Shutdown());
  Require(!registrations && !contexts && !sessions && !cqs && !jetties && !peer_imports && !probe_allocations && !raw_transports && !local_buffers && !events,"all resources and profiling events retired");
  std::cout<<"HBM raw READ: real TCP/source/client, one WQE per span, leases, lost ACKs and cleanup passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
