#include "peer_link.h"
#include "offload_fake_transport.h"
#include "hbm_read_client.h"
#include "hbm_read_protocol.h"
#include "kvaf_aiv_batch_get_raw_addr.h"
#include <poll.h>

using kvaf::Status;
namespace hr=kvaf::hbm_read;
std::atomic<int> raw_calls{0}, raw_wqes{0}, local_buffers{0}, raw_transports{0};
int unregister_failures=0;
bool fail_probe_cq=false;
int quiesces=0;
extern "C" aclError aclrtSynchronizeStream(aclrtStream){return 0;}
struct KvafBatchGetTransport {kvaf::peer::PeerLink* link;};
struct KvafBatchGetLocalBuffer {std::uint64_t base,bytes;};
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
  ++raw_transports;*out=new KvafBatchGetTransport{link};return 0;
}
extern "C" int32_t KvafBatchGetLocalBufferAcquire(KvafBatchGetTransport*,void* output,std::uint64_t bytes,
    KvafBatchGetLocalBuffer** out,std::uint64_t* entity){++local_buffers;*out=new KvafBatchGetLocalBuffer{reinterpret_cast<uint64_t>(output),bytes};*entity=reinterpret_cast<uint64_t>(*out);return 0;}
extern "C" int32_t KvafBatchGetLocalBufferDiscard(KvafBatchGetLocalBuffer** p){
  if(!*p)return 0;if(unregister_failures){--unregister_failures;return -1;}--local_buffers;delete *p;*p=nullptr;return 0;}
extern "C" int32_t KvafBatchGetTransportQuiesce(KvafBatchGetTransport*){++quiesces;return 0;}
extern "C" int32_t KvafBatchGetTransportClose(KvafBatchGetTransport** p){if(*p){--raw_transports;delete *p;*p=nullptr;}return 0;}
extern "C" int32_t KvafExecuteBatchGetRawAddrLaunch(const KvafExecuteBatchGetRawAddrKernelArgs* a,aclrtStream){
  ++raw_calls;auto* src=reinterpret_cast<uint64_t*>(a->source_addresses);auto* dst=reinterpret_cast<uint64_t*>(a->destination_addresses);
  auto* sizes=reinterpret_cast<uint32_t*>(a->lengths);auto* local=reinterpret_cast<KvafBatchGetLocalBuffer*>(a->channel_entity);
  for(unsigned i=0;i<a->request_count;++i){Require(sizes[i] && sizes[i]<=65536,"READ chunk <=64 KiB");
    Require(dst[i]>=local->base && dst[i]-local->base<=local->bytes-sizes[i],"destination registration");
    std::memcpy(reinterpret_cast<void*>(dst[i]),reinterpret_cast<void*>(src[i]),sizes[i]);++raw_wqes;}
  auto* statuses=reinterpret_cast<KvafBatchGetCoreStatus*>(a->core_status);
  for(unsigned c=0;c<a->block_count;++c){statuses[c].magic=KVAF_AIV_BATCH_GET_STATUS_MAGIC;
    statuses[c].result=fail_probe_cq?7:0;statuses[c].processed_request_count=uint64_t(a->request_count)*(c+1)/a->block_count-uint64_t(a->request_count)*c/a->block_count;}
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
        if(kvaf::offload::ReadBytes(client,spans.data(),spans.size()*sizeof(hr::Span))){
          int server=hr::Connect(upstream);Require(server>=0,"connect source");hr::Response response{};
          Require(kvaf::offload::WriteBytes(server,&r,sizeof(r))&&kvaf::offload::WriteBytes(server,spans.data(),spans.size()*sizeof(hr::Span))&&
            kvaf::offload::ReadBytes(server,&response,sizeof(response)),"proxy exchange");close(server);
          if(drop.load()==r.header.command)drop=0;else kvaf::offload::WriteBytes(client,&response,sizeof(response));
        }}close(client);
    }});
  }
  ~Proxy(){stop=true;thread.join();close(fd);}
};
void Check(const Status& s){if(!s.ok())throw std::runtime_error(s.message());}
int main(){try{
  raw_mode=true;fake_device=0;
  std::vector<uint8_t> source(400128),output(300000,0xa5);
  for(size_t i=0;i<source.size();++i)source[i]=(i*17+3)%251;
  kvaf::HbmReadSource b(0,"00000000000000000000000000000001");kvaf::HbmReadTicket ticket;
  Check(b.Register({source.data(),source.size(),kvaf::DeviceKind::kNpu},reinterpret_cast<void*>(1),&ticket));Check(b.Listen("127.0.0.1:0"));
  Proxy proxy("127.0.0.1:"+std::to_string(b.port()));
  {kvaf::HbmReader failed(0,2,2,2048,"fake.o");kvaf::LinkId rejected{};
    fail_probe_cq=true;
    Require(!failed.Attach(proxy.address,ticket,"00000000000000000000000000000002",&rejected).ok(),"probe CQ failure rejected");
    Require(quiesces==1 && !failed.pending() && !b.connected() && !registrations && !raw_transports && !local_buffers,
            "failed probe quiesces all Jettys before retiring either side");
    Check(failed.Shutdown());raw_calls=0;}
  kvaf::HbmReader a(0,2,2,2048,"fake.o");kvaf::LinkId id;
  Check(a.Attach(proxy.address,ticket,"00000000000000000000000000000002",&id));
  Require(b.connected() && raw_calls==1,"all-lane probe completed");
  { kvaf::HbmReader other(0,2,2,2048,"fake.o");kvaf::LinkId rejected{};
    Require(!other.Attach(proxy.address,ticket,"00000000000000000000000000000002",&rejected).ok(),"reject second reader");
    Check(other.Shutdown()); }
  Require(!b.Shutdown().ok(),"cannot unregister connected source");
  auto ready=[&]{kvaf::HbmReadBatch batch;Check(b.Ready({7,100003,250019},{1,656,100001},reinterpret_cast<void*>(1),&batch));return batch;};
  auto out=kvaf::MutableBufferView{output.data(),output.size(),kvaf::DeviceKind::kNpu};
  auto batch=ready();auto before=raw_calls.load();
  Require(!a.Read(id,batch,out,{0,0,0},reinterpret_cast<void*>(1)).ok(),"overlap rejected before Acquire");
  auto stale=batch;stale.epoch++;Require(!a.Read(id,stale,out,{3,101,2000},reinterpret_cast<void*>(1)).ok(),"stale epoch rejected");
  auto corrupt=batch;corrupt.source_addresses[0]++;
  Require(!a.Read(id,corrupt,out,{3,101,2000},reinterpret_cast<void*>(1)).ok(),"B verifies published address");
  Require(raw_calls==before && !a.pending(),"invalid request posts nothing");
  Check(a.Read(id,batch,out,{3,101,2000},reinterpret_cast<void*>(1)));
  Require(raw_calls==before+1 && raw_wqes>=4,"variable lengths split into READs");
  Require(output[3]==source[7] && std::memcmp(output.data()+101,source.data()+100003,656)==0 &&
          std::memcmp(output.data()+2000,source.data()+250019,100001)==0 && output[2]==0xa5 && output[102001]==0xa5,"payload and padding");
  Require(!a.Read(id,batch,out,{3,101,2000},reinterpret_cast<void*>(1)).ok(),"consumed sequence cannot replay");Check(b.ReleaseBatch());
  batch=ready();proxy.drop=unsigned(hr::Command::Complete);before=raw_calls;
  Require(a.Read(id,batch,out,{3,101,2000},reinterpret_cast<void*>(1)).code()==kvaf::StatusCode::kInDoubt && a.pending(),"lost Complete ACK retains task");
  Check(b.ReleaseBatch()); // Even an early producer release must leave an ACK tombstone.
  Check(a.Resolve(id));Require(raw_calls==before+1 && !a.pending(),"resolve ACK does not replay READ");
  batch=ready();proxy.drop=unsigned(hr::Command::Acquire);before=raw_calls;
  Require(!a.Read(id,batch,out,{3,101,2000},reinterpret_cast<void*>(1)).ok() && a.pending(),"lost Acquire response");
  Require(!b.CancelBatch().ok() && !b.ReleaseBatch().ok(),"producer cannot reuse acquired source");
  Require(!a.Resolve(id).ok() && !a.pending() && raw_calls==before,"resolve cancels before DMA, reports no output");Check(b.ReleaseBatch());
  batch=ready();unregister_failures=1;
  Require(!a.Read(id,batch,out,{3,101,2000},reinterpret_cast<void*>(1)).ok() && a.pending(),"unregister failure retains output");
  Check(a.Resolve(id));Check(b.ReleaseBatch());
  batch=ready();Check(b.CancelBatch());Require(!a.Read(id,batch,out,{3,101,2000},reinterpret_cast<void*>(1)).ok(),"cancelled Ready cannot be read");
  proxy.drop=unsigned(hr::Command::Close);Require(!a.Detach(id).ok(),"lost detach reply");Check(a.Detach(id));Check(b.Shutdown());
  Require(!registrations && !contexts && !sessions && !cqs && !jetties && !peer_imports && !probe_allocations && !raw_transports && !local_buffers,"all resources retired");
  std::cout<<"HBM raw READ: real TCP/source/client, split spans, leases, lost ACKs and cleanup passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
