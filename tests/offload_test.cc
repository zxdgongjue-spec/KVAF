#include "runtime.h"
#include "offload_service.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace kvaf;
namespace p = kvaf::offload;
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

struct Operator : AccessOperator {
  Status result;
  int calls{};
  RemoteRegionDesc region;
  Status Initialize(const RemoteRegionDesc& r) override { region = r; return Status::OK(); }
  Status Shutdown() override { return Status::OK(); }
  Status ExecuteBatchGet(const BatchGetExecRequest&) override { return Status::OK(); }
  Status ExecuteBatchPut(const BatchPutExecRequest&) override { return Status::OK(); }
  Status ExecuteOffload(const OffloadExecRequest& r) override {
    ++calls;
    if (result.ok()) for (std::size_t i=0; i<r.put_desc->entries.size(); ++i)
      std::memcpy(reinterpret_cast<void*>(r.put_desc->entries[i].address),
                  static_cast<const char*>(r.request->input.data)+r.request->input_offsets[i], r.request->lengths[i]);
    return result;
  }
  Status ResolveOffload() override { return result; }
};

void TransactionTest() {
  auto meta = CreateInMemoryMetaManager();
  auto allocator = CreateHostDdrMemoryAllocator();
  Operator op;
  RuntimeCore core(*meta, op, *allocator);
  Require(core.Initialize({256,64,DeviceKind::kCpu}).ok(), "initialize");
  std::uint8_t source[128];
  for (int i=0;i<128;++i) source[i] = i;
  BatchPutRequest request{{1,3}, {source,128,DeviceKind::kNpu}, {7,80}, {33,17}};
  Require(core.Offload(request).ok(), "successful offload");
  Require(!std::memcmp(reinterpret_cast<void*>(op.region.base+64), source+7,33), "slot 1 bytes");
  Require(!std::memcmp(reinterpret_cast<void*>(op.region.base+192), source+80,17), "slot 3 bytes");
  LockContext lock; FlatAccessDesc lookup;
  Require(meta->LockKeys({1,3},MetaManagerLockMode::kStableReadAllowAppend,&lock).ok(),"lock committed");
  Require(meta->BatchLookup({1,3},lock,&lookup).ok() && lookup.valid[0] && lookup.valid[1],"committed metadata");
  Require(meta->UnlockKeys(lock).ok(),"unlock");
  Require(core.Offload(request).code()==StatusCode::kAlreadyExists && op.calls==1,"duplicate key rejected before transport");
  Require(core.BatchFree({{1,3}}).ok(),"free");
  op.result=Status::Internal("no DMA submitted");
  Require(!core.Offload(request).ok() && !core.pending_offload(),"safe abort");
  op.result=Status::InDoubt("response lost");
  Require(core.Offload(request).code()==StatusCode::kInDoubt && core.pending_offload(),"pending transaction retained");
  Require(core.Shutdown().code()==StatusCode::kInDoubt,"shutdown cannot free pending pool");
  PutContext duplicate; PutAccessDesc desc;
  Require(meta->PreparePut(request.keys,request.lengths,&duplicate,&desc).code()==StatusCode::kAlreadyExists,"pending slots cannot be reused");
  Require(meta->LockKeys({1,3},MetaManagerLockMode::kStableReadAllowAppend,&lock).code()==StatusCode::kNotFound,"pending data invisible");
  Require(core.ResolveOffload().code()==StatusCode::kInDoubt,"unknown stays pending");
  op.result=Status::OK();
  Require(core.ResolveOffload().ok() && !core.pending_offload(),"recovery commits once");
  Require(core.BatchFree({{1,3}}).ok(),"free resolved");
  op.result=Status::InDoubt("partial write");
  Require(core.Offload(request).code()==StatusCode::kInDoubt,"second pending");
  op.result=Status::Internal("DMA drained after failure");
  Require(!core.ResolveOffload().ok() && !core.pending_offload(),"recovery aborts safely");
  request.input_offsets[0]=127;
  const int before=op.calls;
  Require(core.Offload(request).code()==StatusCode::kInvalidArgument && op.calls==before,"range rejected before reserve");
  request.input_offsets[0]=7;
  op.result=Status::OK();
  Require(core.Offload(request).ok(),"aborted slots reusable");
  Require(core.Shutdown().ok(),"clean shutdown");
}

struct Fake : p::Backend {
  std::vector<std::uint8_t> memory;
  std::atomic<int> reads{0};
  std::atomic<bool> go{false};
  bool fail_unknown{};
  bool leased{};
  p::Endpoint source{};
  Status Allocate(std::uint64_t bytes,p::Endpoint* endpoint) override {
    memory.resize(bytes); endpoint->base=reinterpret_cast<std::uint64_t>(memory.data()); endpoint->bytes=bytes;
    return Status::OK();
  }
  Status Open(const p::Endpoint& src,p::Endpoint*) override {source=src;leased=true;return Status::OK();}
  Status Read(const std::vector<p::Span>& spans) override {
    ++reads;
    while(!go) std::this_thread::yield();
    if(fail_unknown) return Status::InDoubt("DMA unknown");
    for(auto span:spans) std::memcpy(memory.data()+span.destination,reinterpret_cast<void*>(source.base+span.source),span.bytes);
    return Status::OK();
  }
  Status ReleaseSource() override {leased=false;return Status::OK();}
  Status Close() override {return leased?Status::InDoubt("leased"):Status::OK();}
  std::uint64_t Checksum(const std::vector<p::Span>& spans) const override {
    std::uint64_t hash=14695981039346656037ULL;
    for(auto span:spans) hash=p::Hash(memory.data()+span.destination,span.bytes,hash);
    return hash;
  }
};

void ServiceTest() {
  Fake* fake{};
  p::Service service([&](std::uint32_t){auto b=std::make_unique<Fake>();fake=b.get();return b;});
  auto request=p::MakeRequest(p::Command::Allocate);
  request.region=19;request.bytes=256;request.slot_size=64;
  auto response=service.Dispatch(request,{});
  Require(response.result==p::Result::Ok,"allocate");
  Require(service.Dispatch(request,{}).base==response.base,"allocation retry same pool");
  const auto epoch=response.epoch;
  std::uint8_t source[128];for(int i=0;i<128;++i)source[i]=i;
  auto call=[&](p::Command command,std::vector<p::Span> spans=std::vector<p::Span>{}){
    request.header.command=static_cast<std::uint16_t>(command);
    request.count=spans.size();request.header.bytes=sizeof(request)+spans.size()*sizeof(p::Span);
    return service.Dispatch(request,spans);
  };
  request.epoch=epoch;request.task=1;request.source.base=reinterpret_cast<std::uint64_t>(source);request.source.bytes=128;
  Require(call(p::Command::Open).result==p::Result::Ok,"open source");
  Require(call(p::Command::Open).result==p::Result::Ok,"idempotent open");
  Require(call(p::Command::Submit,{{0,0,65}}).result==p::Result::Invalid,"reject slot spill");
  Require(call(p::Command::Submit,{{0,0,32},{32,0,32}}).result==p::Result::Invalid,"reject overlap");
  std::vector<p::Span> spans{{7,64,33},{80,192,17}};
  Require(call(p::Command::Submit,spans).result==p::Result::Ok,"submit");
  Require(call(p::Command::Submit,spans).result==p::Result::Ok,"lost submit response retry");
  Require(call(p::Command::Submit,{{0,0,1}}).result==p::Result::Invalid,"task payload conflict");
  Require(call(p::Command::ReleaseSource).result==p::Result::Busy && service.Busy(),"retain source during DMA");
  Require(call(p::Command::Close).result==p::Result::Busy,"cannot close active pool");
  fake->go=true;
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  do {response=call(p::Command::Query);std::this_thread::yield();}
  while(response.state==p::State::Running && std::chrono::steady_clock::now()<deadline);
  Require(response.state==p::State::Done && fake->reads==1,"one execution, completed");
  auto hash=p::Hash(source+7,33);hash=p::Hash(source+80,17,hash);
  Require(call(p::Command::Checksum).checksum==hash,"correct data and source order");
  Require(call(p::Command::ReleaseSource).result==p::Result::Ok,"release source");
  Require(call(p::Command::ReleaseSource).result==p::Result::Ok,"release ack lost retry");
  Require(call(p::Command::Submit,spans).result==p::Result::Invalid,"late submit after release rejected");
  request.epoch=epoch+1;
  Require(call(p::Command::Query).result==p::Result::NotFound,"stale server epoch");
  request.epoch=epoch;request.task=2;
  Require(call(p::Command::Open).result==p::Result::Ok,"next task new generation");
  Require(call(p::Command::ReleaseSource).result==p::Result::Ok,"cancel before submit");
  Require(call(p::Command::Close).result==p::Result::Ok,"close pool");
  Require(call(p::Command::Close).result==p::Result::NotFound,"close ack lost retry");
  Require(service.Shutdown().ok(),"shutdown drained service");
}
int main(){try{TransactionTest();ServiceTest();std::cout<<"Offload transaction/service tests passed\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
