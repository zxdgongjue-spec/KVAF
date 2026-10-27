#include "runtime.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace {
using namespace kvaf;
void Check(bool value, const char* what) { if (!value) throw std::runtime_error(what); }
void Ok(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message()); }
void Code(const Status& s, StatusCode code) { Check(s.code() == code, s.message().c_str()); }

struct Trace {
  bool fail_init{false}, fail_put{false}, fail_get{false}, quiescent{true}, fail_stop{false};
  int init{0}, puts{0}, gets{0}, stops{0};
  RemoteRegionDesc region;
};
class Operator final : public AccessOperator {
 public:
  explicit Operator(std::shared_ptr<Trace> trace) : t(std::move(trace)) {}
  Status Initialize(const RemoteRegionDesc& region) override {
    ++t->init; t->region = region;
    if (t->fail_init) return Status::Internal("injected Initialize failure");
    return Status::OK();
  }
  Status Shutdown() override {
    ++t->stops;
    if (t->fail_stop) { t->fail_stop = false; return Status::Internal("injected stop failure"); }
    return Status::OK();
  }
  bool FailedPutIsQuiescent() const override { return t->quiescent; }
  Status ExecuteBatchPut(const BatchPutExecRequest& e) override {
    ++t->puts;
    if (t->fail_put) { t->fail_put = false; return Status::Internal("injected Put failure"); }
    for (std::size_t i = 0; i < e.put_desc->entries.size(); ++i) {
      const auto& target = e.put_desc->entries[i];
      Check(target.address >= t->region.base && target.reserved_length <= t->region.length &&
          target.address - t->region.base <= t->region.length - target.reserved_length, "cross-region write");
      std::memcpy(reinterpret_cast<void*>(target.address),
          static_cast<const std::uint8_t*>(e.request->input.data) + e.request->input_offsets[i],
          static_cast<std::size_t>(target.reserved_length));
    }
    return Status::OK();
  }
  Status ExecuteBatchGet(const BatchGetExecRequest& e) override {
    ++t->gets;
    if (t->fail_get) { t->fail_get = false; return Status::Internal("injected device Get failure"); }
    const auto& r = *e.request;
    const auto count = *static_cast<const std::int32_t*>(r.miss_count.data);
    const auto* indices = static_cast<const std::int32_t*>(r.indices.data);
    const auto* offsets = static_cast<const std::int32_t*>(r.offsets.data);
    const auto* rows = static_cast<const std::int32_t*>(r.output_indices.data);
    for (int i = 0; i < count; ++i) {
      if (indices[i] < 0 || std::size_t(indices[i]) >= e.host_access_table.size || offsets[i] < 0 ||
          std::uint64_t(offsets[i]) + r.output_block_bytes > e.host_access_table.lengths[indices[i]] ||
          rows[i] < 0 || std::uint64_t(rows[i]) >= r.output.bytes / r.output_block_bytes)
        return Status::InvalidArgument("invalid Get row");
    }
    for (int i = 0; i < count; ++i) {
      const auto source = e.host_access_table.addresses[indices[i]] + offsets[i];
      std::memcpy(static_cast<std::uint8_t*>(r.output.data) + rows[i] * r.output_block_bytes,
                  reinterpret_cast<const void*>(source), std::size_t(r.output_block_bytes));
    }
    return Status::OK();
  }
  std::shared_ptr<Trace> t;
};

class Allocator final : public DdrMemoryAllocator {
 public:
  std::unique_ptr<DdrMemoryAllocator> host = CreateHostDdrMemoryAllocator();
  std::unordered_map<Address, RemoteRegionDesc> live;
  bool fail_allocate{false}, fail_release{false}, invalid{false};
  int allocations{0}, releases{0};
  Status Allocate(const RuntimeInitRequest& r, RemoteRegionDesc* out) override {
    if (fail_allocate) { fail_allocate = false; return Status::OutOfMemory("injected allocation failure"); }
    auto s = host->Allocate(r, out);
    if (s.ok()) { ++allocations; live.emplace(out->base, *out); if (invalid) --out->length; }
    return s;
  }
  Status Release(const RemoteRegionDesc& r) override {
    if (fail_release) { fail_release = false; return Status::Internal("injected release failure"); }
    auto s = host->Release(r);
    if (s.ok()) { ++releases; live.erase(r.base); }
    return s;
  }
};
struct Fixture {
  std::unique_ptr<MetaManager> meta = CreateInMemoryMetaManager();
  Allocator allocator;
  Operator legacy{std::make_shared<Trace>()};
  RuntimeCore core{*meta, legacy, allocator};
  std::vector<std::shared_ptr<Trace>> traces;
  bool fail_next_init{false}, fail_next_stop{false};
  Fixture() {
    Ok(core.InitializePools([this] {
      auto t = std::make_shared<Trace>(); t->fail_init = fail_next_init; fail_next_init = false;
      t->fail_stop = fail_next_stop; fail_next_stop = false;
      traces.push_back(t); return std::make_unique<Operator>(t);
    }));
  }
  ~Fixture() { (void)core.Shutdown(); }
  PoolHandle Pool(const char* name = "pool", std::uint64_t count = 4,
                  std::uint32_t token = 8, std::uint32_t blocks = 2) {
    PoolHandle p; Ok(core.CreatePool(PoolSpec{name, std::uint64_t{token} * blocks, count, token, blocks}, &p)); return p;
  }
  LayerHandle Layer(PoolHandle pool, std::uint64_t layer = 1, std::uint64_t ns = 0) {
    LayerHandle l; Ok(core.BindLayer(ns, layer, pool, &l)); return l;
  }
  PoolInfo Info(PoolHandle p) { PoolInfo i; Ok(core.GetPoolInfo(p, &i)); return i; }
  Status Put(LayerHandle l, const std::vector<KeyId>& keys, std::uint8_t seed = 19) {
    LayerLayout layout; auto s = core.GetLayerLayout(l, &layout); if (!s.ok()) return s;
    const auto block = std::size_t(layout.spec.token_bytes) * layout.spec.tokens_per_block;
    std::vector<std::uint8_t> input(block * keys.size());
    std::vector<std::uint64_t> offsets, lengths;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      offsets.push_back(i * block); lengths.push_back(block);
      for (std::size_t j = 0; j < block; ++j) input[i * block + j] = std::uint8_t(seed + i * 17 + j * 7);
    }
    return core.BatchPut(BatchPutRequest{keys, {input.data(), input.size(), DeviceKind::kCpu}, offsets, lengths, l});
  }
  AccessTableHandle Table(LayerHandle l, std::vector<KeyId> keys) {
    AccessTableHandle h; Ok(core.PrepareAccessTable({std::move(keys), l}, &h)); return h;
  }
  AccessTableView View(AccessTableHandle h) { AccessTableView v; Ok(core.GetAccessTableView(h, &v)); return v; }
  Status Get(AccessTableHandle h, std::uint32_t token, std::vector<std::uint8_t>& out,
             std::int32_t offset = 0, LayerHandle expected = {}) {
    std::int32_t index = 0, row = 0, count = 1;
    BatchGetRequest r{h, {&index,4,DeviceKind::kCpu}, {&offset,4,DeviceKind::kCpu},
      {&count,4,DeviceKind::kCpu}, {&row,4,DeviceKind::kCpu},
      {out.data(),out.size(),DeviceKind::kCpu}, token, 0, {}, expected};
    return core.BatchGet(r);
  }
};

void HeterogeneousReadWrite() {
  Fixture f;
  auto a = f.Pool("a", 4, 1024, 128), b = f.Pool("b", 4, 656, 128);
  auto l1 = f.Layer(a, 1), l2 = f.Layer(a, 2), l5 = f.Layer(b, 5);
  Ok(f.Put(l1, {999999999999ULL}, 11)); Ok(f.Put(l2, {999999999999ULL}, 21)); Ok(f.Put(l5, {999999999999ULL}, 31));
  auto t1=f.Table(l1,{999999999999ULL}), t2=f.Table(l2,{999999999999ULL}), t5=f.Table(l5,{999999999999ULL});
  Check(f.View(t1).addresses[0] != f.View(t2).addresses[0], "same-pool layers alias");
  Check(f.Info(a).region.base != f.Info(b).region.base, "pools share region");
  for (auto pair : {std::make_pair(t1,1024U), std::make_pair(t2,1024U), std::make_pair(t5,656U)}) {
    std::vector<std::uint8_t> out(pair.second, 0);
    Ok(f.Get(pair.first,pair.second,out,127*std::int32_t(pair.second)));
    const auto seed = pair.first.opaque == t1.opaque ? 11 : pair.first.opaque == t2.opaque ? 21 : 31;
    for (std::size_t j=0;j<out.size();++j)
      Check(out[j] == std::uint8_t(seed+(127*pair.second+j)*7), "token read mismatch");
  }
  Code(f.core.BatchFree({{999999999999ULL},l1}),StatusCode::kBusy);
  for (auto t : {t1,t2,t5}) Ok(f.core.ReleaseAccessTable(t));
  for (auto l : {l1,l2,l5}) Ok(f.core.BatchFree({{999999999999ULL},l}));
  Check(f.traces.size()==2 && f.traces[0]->init==1 && f.traces[1]->init==1, "repeated operator init");
}
void CapacityAtomicityAndReuse() {
  Fixture f; auto a=f.Pool("a",2), b=f.Pool("b",2); auto x=f.Layer(a), y=f.Layer(b,2);
  Ok(f.Put(x,{1,2})); Code(f.Put(x,{3}),StatusCode::kOutOfMemory); Ok(f.Put(y,{3}));
  Code(f.core.BatchFree({{1,77},x}),StatusCode::kNotFound);
  auto t=f.Table(x,{1}); const auto address=f.View(t).addresses[0];
  Code(f.core.BatchFree({{2,1},x}),StatusCode::kBusy);
  AccessTableHandle missing;
  Code(f.core.PrepareAccessTable({{2,77},x},&missing),StatusCode::kNotFound);
  Ok(f.core.ReleaseAccessTable(t)); Ok(f.core.BatchFree({{1},x}));
  Ok(f.Put(x,{3})); t=f.Table(x,{3}); Check(f.View(t).addresses[0]==address,"slot not reused");
  Ok(f.core.ReleaseAccessTable(t)); Check(f.Info(a).committed_slots==2,"partial Free changed metadata");
}
void NamespacesAndHandles() {
  Fixture f, other; auto p=f.Pool(); auto l=f.Layer(p,1,7), n=f.Layer(p,1,8);
  Ok(f.Put(l,{42})); Ok(f.Put(n,{42}));
  LayerHandle found; Ok(f.core.GetLayer(7,1,&found)); Check(found.id==l.id,"layer lookup");
  Code(f.core.BindLayer(7,1,p,&found),StatusCode::kAlreadyExists);
  Code(other.core.BindLayer(0,1,p,&found),StatusCode::kNotFound);
  auto t=f.Table(l,{42}); std::vector<std::uint8_t> out(8);
  Code(f.Get(t,8,out,0,n),StatusCode::kInvalidArgument);
  Code(other.Get(t,8,out),StatusCode::kNotFound);
  Code(f.core.Shutdown(),StatusCode::kBusy);
  Ok(f.core.ReleaseAccessTable(t)); Code(f.core.ReleaseAccessTable(t),StatusCode::kNotFound);
  Code(f.core.UnbindLayer(l),StatusCode::kBusy);
  Ok(f.core.BatchFree({{42},l})); Ok(f.core.UnbindLayer(l));
  Code(f.Put(l,{42}),StatusCode::kNotFound);
  auto fresh=f.Layer(p,1,7); Check(fresh.id != l.id,"reused layer generation");
  Code(f.core.DestroyPool(p),StatusCode::kBusy);
}
void ConfigurationAndInput() {
  Fixture f; PoolHandle p;
  for (auto s : {PoolSpec{"",16,2,8,2},PoolSpec{"x",0,2,8,2},PoolSpec{"x",16,0,8,2},
      PoolSpec{"x",15,2,8,2},PoolSpec{"x",16,2,0,2},
      PoolSpec{"x",std::numeric_limits<std::uint64_t>::max(),2,8,2}})
    Code(f.core.CreatePool(s,&p),StatusCode::kInvalidArgument);
  Check(f.allocator.allocations==0,"invalid configuration allocated memory");
  p=f.Pool(); auto l=f.Layer(p);
  Code(f.core.CreatePool({"pool",16,2,8,2},&p),StatusCode::kAlreadyExists);
  Code(f.Put(l,{1,1}),StatusCode::kInvalidArgument);
  std::uint8_t input[16]{};
  Code(f.core.BatchPut({{1},{input,16,DeviceKind::kCpu},{0},{15},l}),StatusCode::kInvalidArgument);
  Code(f.core.BatchPut({{1},{input,16,DeviceKind::kCpu},{1},{16},l}),StatusCode::kInvalidArgument);
  Check(f.Info(p).free_slots==4,"invalid input reserved slots");
  Ok(f.Put(l,{1})); auto t=f.Table(l,{1}); std::vector<std::uint8_t> out(8);
  Code(f.Get(t,4,out),StatusCode::kInvalidArgument);
  Code(f.Get(t,8,out,9),StatusCode::kInvalidArgument);
  Ok(f.core.ReleaseAccessTable(t));
}
void FailuresAndQuarantine() {
  Fixture f; auto p=f.Pool("a"), q=f.Pool("b"); auto l=f.Layer(p), m=f.Layer(q,2);
  f.traces[0]->fail_put=true;
  Code(f.Put(l,{1}),StatusCode::kInternal);
  Check(f.Info(p).free_slots==4,"quiescent failure leaked slot");
  Ok(f.Put(l,{1}));
  f.traces[0]->quiescent=false; f.traces[0]->fail_put=true;
  Code(f.Put(l,{2}),StatusCode::kInternal);
  auto info=f.Info(p); Check(info.quarantined && info.quarantined_slots==1,"uncertain DMA not quarantined");
  Code(f.Put(l,{3}),StatusCode::kInvalidState); Ok(f.Put(m,{3}));
  f.traces[0]->fail_stop=true;
  Code(f.core.Shutdown(),StatusCode::kInternal);
  Check(f.allocator.live.size()==1 && f.core.initialized(),"failed stop released region");
  Ok(f.core.Shutdown()); Check(f.allocator.live.empty(),"retry leaked region");
}
void InitializationAndReleaseFailures() {
  Fixture f; PoolHandle p;
  f.allocator.fail_allocate=true;
  Code(f.core.CreatePool({"x",16,2,8,2},&p),StatusCode::kOutOfMemory);
  f.fail_next_init=true;
  Code(f.core.CreatePool({"x",16,2,8,2},&p),StatusCode::kInternal);
  Check(f.allocator.live.empty(),"failed Initialize leaked region");
  f.allocator.invalid=true;
  Code(f.core.CreatePool({"x",16,2,8,2},&p),StatusCode::kInvalidArgument);
  Check(f.allocator.live.empty(),"invalid region leaked"); f.allocator.invalid=false;
  p=f.Pool("x");
  f.allocator.fail_release=true; Code(f.core.DestroyPool(p),StatusCode::kInternal);
  Check(f.allocator.live.size()==1,"release failure lost allocation");
  Ok(f.core.DestroyPool(p)); Code(f.core.GetPoolInfo(p,nullptr),StatusCode::kNotFound);
  auto again=f.Pool("x"); Check(again.id!=p.id,"reused pool ID");
  f.fail_next_init=true; f.fail_next_stop=true;
  const auto before = f.allocator.live.size();
  Code(f.core.CreatePool({"retained",16,2,8,2},&p),StatusCode::kInternal);
  Check(f.allocator.live.size()==before+1,"failed initialization teardown released region");
  Ok(f.core.Shutdown()); Check(f.allocator.live.empty(),"initialization cleanup retry leaked");
}
void MetadataTransactionVisibility() {
  Allocator allocator;
  PoolManager slots(allocator, [] { return std::make_unique<Operator>(std::make_shared<Trace>()); });
  auto meta=CreateInMemoryMetaManager();
  PoolHandle pool; Ok(slots.Create({"meta",16,4,8,2},&pool));
  LayerHandle layer; Ok(slots.Bind(0,1,pool,&layer));
  LayerLayout layout; Ok(slots.Resolve(layer,&layout));
  PutContext put; PutAccessDesc desc; LockContext read;
  Ok(meta->pools().PreparePut(slots,layout,{71,72},{16,16},&put,&desc));
  Code(meta->pools().LockKeys(layout,{71},&read),StatusCode::kNotFound);
  PutContext duplicate; PutAccessDesc ignored;
  Code(meta->pools().PreparePut(slots,layout,{71},{16},&duplicate,&ignored),StatusCode::kAlreadyExists);
  Ok(meta->pools().AbortPut(slots,put));
  Code(meta->pools().CommitPut(slots,put),StatusCode::kNotFound);
  Check(meta->pools().CountPool(pool)==0,"Abort left keys");
  Ok(meta->pools().PreparePut(slots,layout,{71},{16},&put,&desc));
  Ok(meta->pools().CommitPut(slots,put));
  Ok(meta->pools().LockKeys(layout,{71,71},&read));
  FlatAccessDesc view;
  Code(meta->pools().Lookup({71},read,&view),StatusCode::kInvalidArgument);
  Ok(meta->pools().Lookup({71,71},read,&view));
  Check(view.addresses.size()==2 && view.addresses[0]==view.addresses[1],"duplicate snapshot mismatch");
  Code(meta->pools().ErasePool(pool),StatusCode::kBusy);
  Code(meta->pools().Free(slots,layout,{71}),StatusCode::kBusy);
  Ok(meta->pools().UnlockKeys(read));
  Code(meta->pools().UnlockKeys(read),StatusCode::kNotFound);
  const auto slot=(desc.entries[0].address-layout.region.base)/layout.spec.slot_size;
  SlotLease lease{pool,slot,desc.entries[0].address};
  Code(slots.Release({lease,lease}),StatusCode::kInvalidState);
  PoolInfo info; Ok(slots.Info(pool,&info)); Check(info.committed_slots==1,"invalid lease batch changed slots");
  Ok(meta->pools().Free(slots,layout,{71})); Ok(slots.Unbind(layer)); Ok(slots.Destroy(pool));
  Check(allocator.live.empty(),"metadata test leaked");
}
void FailedDeviceGetIsolation() {
  Fixture f; auto p=f.Pool("a"), q=f.Pool("b"); auto l=f.Layer(p), m=f.Layer(q,2);
  Ok(f.Put(l,{1})); auto table=f.Table(l,{1}); auto view=f.View(table);
  std::int32_t zero=0, count=1;
  std::uint8_t out[8]{};
  // Simulated device descriptors are never dereferenced: the injected backend
  // returns before executing IO. This tests Core error ownership, not hardware.
  BatchGetRequest request{table,{&zero,4,DeviceKind::kNpu},{&zero,4,DeviceKind::kNpu},
    {&count,4,DeviceKind::kNpu},{&zero,4,DeviceKind::kNpu},{out,8,DeviceKind::kNpu},8,2,
    {{view.addresses,8,DeviceKind::kNpu},{view.valid,1,DeviceKind::kNpu}}};
  f.traces[0]->fail_get=true;
  Code(f.core.BatchGet(request),StatusCode::kInternal);
  Check(f.Info(p).quarantined,"device Get failure not isolated");
  Code(f.Put(l,{2}),StatusCode::kInvalidState);
  Ok(f.Put(m,{2}));
  Ok(f.core.ReleaseAccessTable(table));
}
void PinAcrossDirectoryGrowth() {
  Fixture f; auto p=f.Pool("big",2048); auto l=f.Layer(p);
  Ok(f.Put(l,{9})); auto t=f.Table(l,{9,9});
  std::vector<KeyId> keys; for(KeyId i=100;i<1100;++i)keys.push_back(i);
  Ok(f.Put(l,keys)); std::vector<std::uint8_t> out(8); Ok(f.Get(t,8,out));
  Code(f.core.BatchFree({{9},l}),StatusCode::kBusy);
  Ok(f.core.ReleaseAccessTable(t)); Ok(f.core.BatchFree({{9},l}));
  auto n=f.Layer(p,2); Ok(f.Put(n,{9}));
}
void IndependentDestroyAndRestart() {
  Fixture f; auto a=f.Pool("a"),b=f.Pool("b"); auto l=f.Layer(a),m=f.Layer(b,2);
  Ok(f.Put(l,{0})); Ok(f.Put(m,{0}));
  Ok(f.core.BatchFree({{0},l})); Ok(f.core.UnbindLayer(l)); Ok(f.core.DestroyPool(a));
  auto t=f.Table(m,{0}); std::vector<std::uint8_t> out(8); Ok(f.Get(t,8,out));
  Ok(f.core.ReleaseAccessTable(t)); Ok(f.core.Shutdown()); Check(f.allocator.live.empty(),"shutdown leaked");
  Ok(f.core.InitializePools([&] { return std::make_unique<Operator>(std::make_shared<Trace>()); }));
  LayerLayout stale; Code(f.core.GetLayerLayout(m,&stale),StatusCode::kNotFound);
  auto c=f.Pool("a"); Check(c.owner!=a.owner,"restart reused owner");
}
}

int main() {
  const std::pair<const char*,std::function<void()>> tests[] = {
    {"heterogeneous read/write",HeterogeneousReadWrite}, {"capacity/atomicity/reuse",CapacityAtomicityAndReuse},
    {"namespace/handle isolation",NamespacesAndHandles}, {"configuration/input",ConfigurationAndInput},
    {"failure/quarantine",FailuresAndQuarantine}, {"initialization/release failure",InitializationAndReleaseFailures},
    {"pin across directory rehash",PinAcrossDirectoryGrowth}, {"independent destroy/restart",IndependentDestroyAndRestart},
    {"metadata transaction visibility",MetadataTransactionVisibility},
    {"failed device Get isolation",FailedDeviceGetIsolation}};
  for (const auto& test : tests) {
    try { test.second(); std::cout << "PASS " << test.first << '\n'; }
    catch (const std::exception& e) { std::cerr << "FAIL " << test.first << ": " << e.what() << '\n'; return 1; }
  }
}
