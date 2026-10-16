// CPU shim for the complete current Put kernel; injected transport reads DDR
// PI at doorbell time and may delay, drop or fail each terminal completion.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>
#include "batch_put_internal.h"
#define __aicore__
#define __global__
#define __gm__
#define __ubuf__
#define __simt_vf__
#define LAUNCH_BOUND(value)
struct Channel {
  KvafBatchPutDirectUrmaEntity *entity;
  KvafBatchPutQueueState *queue;
  uint64_t sq_head, cq_head, pending_pi, durable_pi;
  uint32_t sq_db=0, cq_db=0, arrival=1, probes=0, pending_word=0, pending_slot=0;
  uint32_t cq_error=0;
  uint64_t requests=0, doorbells=0;
  bool pending_cq=false;
  std::vector<uint32_t> sq, cq;
};
std::vector<Channel> *channels;
uint32_t current_core;
uint64_t ticks=0, tick_step=1;
alignas(64) uint8_t ub[128*1024];
struct Dma { uint8_t *address; std::vector<uint8_t> bytes; };
std::vector<Dma> dma;
namespace AscendC {
enum { PIPE_S, PIPE_V, PIPE_MTE3 };
using event_t=unsigned;
struct ThreadIndex { uint32_t x; } threadIdx;
uint32_t events[3][3][2];
uint32_t GetBlockIdx() { return current_core; }
uint64_t GetSystemCycle() { return ticks; }
uint64_t get_imm(uint32_t offset) { return reinterpret_cast<uint64_t>(ub+offset); }
void set_flag(unsigned from,unsigned to,event_t id) { assert(++events[from][to][id]==1); }
void wait_flag(unsigned from,unsigned to,event_t id) {
  assert(events[from][to][id]--==1);
  if(from==PIPE_MTE3) {
    for(auto& copy:dma) std::memcpy(copy.address,copy.bytes.data(),copy.bytes.size());
    dma.clear();
  }
}
void set_loop_size_ubtoout(uint64_t) {}
void set_loop1_stride_ubtoout(uint64_t) {}
void set_loop2_stride_ubtoout(uint64_t) {}
void set_atomic_none() {}
void copy_ubuf_to_gm_align_v2(float *dst,float *src,uint64_t config,uint64_t) {
  auto *begin=reinterpret_cast<uint8_t*>(src);
  dma.push_back({reinterpret_cast<uint8_t*>(dst),{begin,begin+(config>>25U)}});
}
template<class T> T ReadGmByPassDCache(T *address) {
  if constexpr(std::is_same<T,uint32_t>::value) {
    for(auto& c:*channels) {
      const uint64_t pos=reinterpret_cast<uint64_t>(address);
      const uint64_t begin=reinterpret_cast<uint64_t>(c.cq.data());
      if(pos>=begin && pos<begin+c.cq.size()*4 && (pos-begin)%64==0) {
        ++c.probes; ticks+=tick_step;
        if(c.pending_cq && c.arrival && c.probes>=c.arrival) {
          c.cq[c.pending_slot*16]=c.pending_word;
          c.pending_cq=false;
        }
      }
    }
  }
  return *address;
}
template<class T> void WriteGmByPassDCache(T *address,T value) {
  *address=value;
  if constexpr(std::is_same<T,uint64_t>::value) {
    for(auto& c:*channels)
      if(address==&c.queue->sq_producer_index) c.pending_pi=value;
  }
}
enum class MemDsbT { DDR };
template<MemDsbT> void DataSyncBarrier() {
  assert(dma.empty());
  for(auto& c:*channels) c.durable_pi=c.pending_pi;
}
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template<class T> struct GlobalTensor {
  T *pointer;
  void SetGlobalBuffer(T *value) { pointer=value; }
  T& operator[](uint64_t i) { return pointer[i]; }
};
template<class T,CacheLine,DcciDst> void DataCacheCleanAndInvalid(T& value) {
  // Queue/CQ cache-line writeback would invalidate this coherence model.
  const uint64_t line=reinterpret_cast<uint64_t>(&value);
  for(const auto& c:*channels) {
    const uint64_t q=reinterpret_cast<uint64_t>(c.queue)&~63ULL;
    assert(line!=q);
    const uint64_t cq=reinterpret_cast<uint64_t>(c.cq.data());
    assert(line<cq || line>=cq+c.cq.size()*4);
  }
}
namespace Simt {
struct Dim3 { uint32_t x,y,z; };
template<auto Function,class...Args> void VF_CALL(Dim3 dim,Args...args) {
  for(threadIdx.x=0;threadIdx.x<dim.x;++threadIdx.x) Function(args...);
}
}
void st_dev(uint32_t value,uint32_t *address,int) {
  for(auto& c:*channels) {
    if(address==&c.cq_db) { c.cq_db=value; return; }
    if(address!=&c.sq_db) continue;
    assert(dma.empty() && c.durable_pi==value);
    auto *e=c.entity;
    for(;c.sq_head<value;++c.sq_head) {
      auto *w=c.sq.data()+(c.sq_head%e->sq_depth)*16;
      assert((w[0]>>31U)==(1U-((c.sq_head/e->sq_depth)&1U)));
      assert(w[1]==0x300U && w[2]==((e->tp_id&0xffffffU)|0x1000000U));
      assert(w[3]==(e->remote_token_id&0xfffffU) && w[8]==e->remote_token_value);
      assert(w[13]==e->local_token_id && w[12]>0);
      const uint64_t remote=w[10]|(uint64_t(w[11])<<32U);
      const uint64_t local=w[14]|(uint64_t(w[15])<<32U);
      assert(local>=e->local_hbm_address && local+w[12]<=e->local_hbm_address+e->local_hbm_size);
      assert(remote>=e->remote_host_ddr_address && remote+w[12]<=e->remote_host_ddr_address+e->remote_host_ddr_size);
      std::memcpy(reinterpret_cast<void*>(remote),reinterpret_cast<void*>(local),w[12]);
      ++c.requests;
      if(w[0]&(1U<<21U)) {
        assert(c.sq_head+1==value && !c.pending_cq);
        c.pending_slot=c.cq_head%e->cq_depth;
        c.pending_word=uint32_t((1U-((c.cq_head/e->cq_depth)&1U))<<2U)|c.cq_error;
        c.pending_cq=true; c.probes=0; ++c.cq_head;
      }
    }
    c.sq_db=value; ++c.doorbells; return;
  }
  assert(false);
}
}
// ACTUAL_PRODUCTION_KERNEL
void Check(uint32_t rows,uint32_t workers,uint32_t block_bytes,int failure,bool fresh=false) {
  const uint32_t jettys=2, depth=8192, cq_depth=64;
  std::vector<uint8_t> source(uint64_t(rows)*block_bytes),output(source.size());
  for(uint64_t i=0;i<source.size();++i) source[i]=(i*7+i/block_bytes)%251;
  std::vector<uint64_t> offsets(rows),lengths(rows,block_bytes),destinations(rows);
  for(uint32_t i=0;i<rows;++i) {
    offsets[i]=uint64_t(i)*block_bytes;
    destinations[i]=reinterpret_cast<uint64_t>(output.data()+uint64_t(rows-1-i)*block_bytes);
  }
  std::vector<KvafBatchPutDirectUrmaEntity> entities(workers*jettys);
  std::vector<KvafBatchPutQueueState> queues(entities.size());
  std::vector<KvafBatchPutCoreStatus> status(workers);
  std::vector<Channel> transport(entities.size()); channels=&transport;
  for(uint32_t i=0;i<transport.size();++i) {
    auto& e=entities[i]; auto& q=queues[i]; auto& c=transport[i];
    c.entity=&e; c.queue=&q; c.sq_head=c.pending_pi=c.durable_pi=fresh?0:8190; c.cq_head=fresh?0:63;
    c.sq.resize(depth*16); c.cq.resize(cq_depth*16);
    if(!fresh) for(uint32_t old=0;old<63;++old) c.cq[old*16]=4U;
    e.magic=0x4b564155U; e.version=1; e.local_hbm_address=reinterpret_cast<uint64_t>(source.data());
    e.local_hbm_size=source.size(); e.remote_host_ddr_address=reinterpret_cast<uint64_t>(output.data());
    e.remote_host_ddr_size=output.size(); e.local_token_id=0x123; e.local_token_value=1;
    e.remote_token_id=0x456; e.remote_token_value=2; e.tp_id=17;
    e.sq_va=reinterpret_cast<uint64_t>(c.sq.data()); e.sq_db_va=reinterpret_cast<uint64_t>(&c.sq_db);
    e.sq_depth=depth; e.wqe_size=64; e.cq_va=reinterpret_cast<uint64_t>(c.cq.data());
    e.cq_db_va=reinterpret_cast<uint64_t>(&c.cq_db); e.cq_depth=cq_depth; e.cqe_size=64;
    if(!fresh) {
      q.magic=0x4b564151U; q.initialized=1; q.sq_producer_index=8190; q.cq_consumer_index=63;
    }
  }
  if(failure==1) { transport[0].arrival=0; tick_step=10000000; }
  else if(failure==2) { transport[0].cq_error=2U<<24U; tick_step=1; }
  else { transport[0].arrival=20003; tick_step=1; }
  KvafExecuteBatchPutLaunchArgs args{};
  args.request={reinterpret_cast<uint64_t>(source.data()),reinterpret_cast<uint64_t>(offsets.data()),
    reinterpret_cast<uint64_t>(lengths.data()),reinterpret_cast<uint64_t>(destinations.data()),
    reinterpret_cast<uint64_t>(status.data()),source.size(),rows,workers};
  args.channel_entity=reinterpret_cast<uint64_t>(entities.data());
  args.queue_state=reinterpret_cast<uint64_t>(queues.data()); args.jettys_per_core=jettys;
  for(uint32_t repeat=0;repeat<(failure?1U:3U);++repeat) {
    std::fill(output.begin(),output.end(),0);
    for(current_core=0;current_core<workers;++current_core) {
      std::memset(AscendC::events,0,sizeof(AscendC::events));
      KvafExecuteBatchPutAivKernel(args);
      const uint64_t expected=failure && current_core==0 ? (failure==1?7U:8U) : 0U;
      assert(status[current_core].result==expected);
      assert(status[current_core].processed_request_count==
        uint64_t(rows)*(current_core+1)/workers-uint64_t(rows)*current_core/workers);
      assert(dma.empty());
      for(auto& from:AscendC::events) for(auto& to:from) for(auto count:to) assert(count==0);
    }
    for(uint32_t i=0;i<rows;++i)
      assert(std::memcmp(source.data()+uint64_t(i)*block_bytes,
        output.data()+uint64_t(rows-1-i)*block_bytes,block_bytes)==0);
  }
  if(failure) {
    const uint64_t error=failure==1?7U:8U;
    assert(queues[0].last_error==error && queues[1].last_error==error);
    const uint64_t before=transport[0].doorbells+transport[1].doorbells;
    current_core=0; KvafExecuteBatchPutAivKernel(args);
    assert(status[0].result==KVAF_BATCH_PUT_CORE_INVALID_QUEUE);
    assert(queues[0].last_error==error && queues[1].last_error==error);
    assert(transport[0].doorbells+transport[1].doorbells==before);
  } else {
    uint64_t total=0;
    for(const auto& c:transport) { total+=c.requests; assert(c.queue->last_error==0); }
    assert(total==uint64_t(rows)*3);
  }
}
int main() {
  for(uint32_t fresh_case=0;fresh_case<6;++fresh_case) Check(128,24,83968,0,true);
  Check(128,24,83968,0); // Actual first-seed shape: 5/6 WRITE WQEs per worker.
  Check(1536,24,656,0);  // Both Jettys participate and CQ/SQ rings wrap.
  Check(6144,24,656,0);  // Ping-pong banks, multiple posts, terminal fences.
  Check(128,24,83968,1); // Missing CQ poisons and refuses another DMA.
  Check(128,24,83968,2); // Error CQ is consumed once, then remains poisoned.
  std::puts("BATCH_PUT_ACTUAL_KERNEL_CPU_PASS scenarios=11 workers=24 fresh_case_cycles=6 first_seed_rows=128 replay_wrap_late_poison=pass");
}
