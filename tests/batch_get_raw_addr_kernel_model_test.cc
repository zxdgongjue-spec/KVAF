#include "kernel_operator.h"
#include "batch_get_abi.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <vector>

// The Ascend metadata section is not a native Mach-O/ELF section. All device
// code otherwise comes directly from the production source, without a copy.
#define __attribute__(...)
#include "../access_operator/demo/aiv_batch_get/kernel/execute_batch_get_raw_addr_aiv.asc"
#undef __attribute__

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

namespace aiv_model {
struct Queue {
  std::vector<uint32_t> sq = std::vector<uint32_t>(32768 * 16);
  alignas(64) std::array<uint32_t, 8 * 16> cq{};
  uint32_t sq_db = 0, cq_db = 0;
  uint64_t producer = 0, completed = 0, cq_producer = 0, cq_consumer = 0;
  std::deque<uint64_t> checkpoints;
  bool timeout = false, error = false;
};
std::vector<Queue> hardware;
uint64_t doorbells = 0;

void Doorbell(uint32_t value, uint32_t* address) {
  for (auto& queue : hardware) {
    if (address == &queue.cq_db) {
      CHECK(queue.cq_consumer < queue.cq_producer);
      ++queue.cq_consumer;
      CHECK(value == (queue.cq_consumer & 0x003fffffU));
      queue.cq_db = value;
      return;
    }
    if (address != &queue.sq_db) continue;
    ++doorbells;
    const uint32_t added = value - queue.sq_db;
    CHECK(added > 0 && added <= 128);
    CHECK(queue.producer + added - queue.completed < 32768);
    for (uint32_t i = 0; i < added; ++i) {
      const uint64_t pi = queue.producer++;
      const uint32_t* wqe = queue.sq.data() + (pi % 32768) * 16;
      CHECK((wqe[0] >> 31U) == 1U - ((pi / 32768) & 1U));
      const bool signal = (wqe[0] & (1U << 21U)) != 0;
      CHECK(((wqe[0] >> 16U) & 3U) == (signal ? 2U : 1U));
      if (signal) queue.checkpoints.push_back(queue.producer);
    }
    queue.sq_db = value;
    return;
  }
  CHECK(false);
}

void Poll(void* address) {
  const auto probe = reinterpret_cast<uintptr_t>(address);
  for (auto& queue : hardware) {
    const auto begin = reinterpret_cast<uintptr_t>(queue.cq.data());
    if (probe < begin || probe >= begin + sizeof(queue.cq)) continue;
    // Deliver a checkpoint only when polled. This forces the initial two
    // groups to coexist and leaves the younger group outstanding during refill.
    if (queue.timeout || queue.checkpoints.empty() ||
        queue.cq_consumer != queue.cq_producer) return;
    const uint64_t end = queue.checkpoints.front();
    queue.checkpoints.pop_front();
    while (queue.completed < end) {
      const uint32_t* wqe = queue.sq.data() + (queue.completed++ % 32768) * 16;
      const uint64_t source = wqe[10] | (uint64_t(wqe[11]) << 32U);
      const uint64_t destination = wqe[14] | (uint64_t(wqe[15]) << 32U);
      std::memcpy(reinterpret_cast<void*>(destination),
                  reinterpret_cast<void*>(source), wqe[12]);
    }
    const uint32_t owner = 1U - ((queue.cq_producer / 8) & 1U);
    queue.cq[(queue.cq_producer % 8) * 16] =
        (owner << 2U) | (queue.error ? 1U << 24U : 0U);
    ++queue.cq_producer;
    return;
  }
  CHECK(false);
}
}  // namespace aiv_model


struct Fixture {
  uint32_t capacity, cores, jettys;
  std::vector<uint8_t> remote=std::vector<uint8_t>(131072);
  std::vector<uint8_t> output;
  std::vector<uint64_t> src,dst;
  std::vector<uint32_t> lengths;
  std::vector<KvafBatchGetQueueState> queues;
  std::vector<KvafBatchGetDirectUrmaEntity> entities;
  std::vector<KvafBatchGetCoreStatus> statuses;
  KvafExecuteBatchGetRawAddrKernelArgs args{};
  Fixture(uint32_t n,uint32_t c,uint32_t j,uint32_t budget):capacity(n),cores(c),jettys(j),output(n*40),
      src(n),dst(n),lengths(n),queues(c*j),entities(c*j),statuses(c) {
    aiv_model::hardware.clear();aiv_model::hardware.resize(c*j);
    for(uint32_t i=0;i<remote.size();++i)remote[i]=(i*17+3)%251;
    for(uint32_t i=0;i<n;++i){src[i]=reinterpret_cast<uint64_t>(remote.data())+(i*7)%10000;
      dst[i]=reinterpret_cast<uint64_t>(output.data())+(n-1-i)*40+1;lengths[i]=1+i%31;}
    for(uint32_t i=0;i<c*j;++i){auto& e=entities[i];auto& h=aiv_model::hardware[i];
      e.magic=0x4b564155U;e.version=1;e.remote_host_ddr_address=reinterpret_cast<uint64_t>(remote.data());
      e.remote_host_ddr_size=remote.size();e.local_hbm_address=reinterpret_cast<uint64_t>(output.data());e.local_hbm_size=output.size();e.local_token_id=1;
      e.sq_va=reinterpret_cast<uint64_t>(h.sq.data());e.sq_db_va=reinterpret_cast<uint64_t>(&h.sq_db);e.sq_depth=32768;e.wqe_size=64;
      e.cq_va=reinterpret_cast<uint64_t>(h.cq.data());e.cq_db_va=reinterpret_cast<uint64_t>(&h.cq_db);e.cq_depth=8;e.cqe_size=64;
    }
    for(auto& s:statuses)s.profile.magic=KVAF_AIV_BATCH_GET_PROFILE_MAGIC;
    args.source_addresses=reinterpret_cast<uint64_t>(src.data());args.destination_addresses=reinterpret_cast<uint64_t>(dst.data());
    args.lengths=reinterpret_cast<uint64_t>(lengths.data());args.output=reinterpret_cast<uint64_t>(output.data());
    args.core_status=reinterpret_cast<uint64_t>(statuses.data());args.channel_entity=reinterpret_cast<uint64_t>(entities.data());
    args.queue_state=reinterpret_cast<uint64_t>(queues.data());args.block_count=c;args.jettys_per_core=j;args.batch_get_max_inflight_total=budget;
  }
  void Run(uint32_t n,uint64_t expected=0){
    args.request_count=n;std::fill(output.begin(),output.end(),0xa5);
    for(uint32_t c=0;c<cores;++c){aiv_model::core=c;KvafExecuteBatchGetRawAddrAivKernelV1(args);
      CHECK(statuses[c].magic==KVAF_AIV_BATCH_GET_STATUS_MAGIC);CHECK(statuses[c].result==expected);
      if(!expected)CHECK(statuses[c].profile.submitted_group_count==statuses[c].profile.completed_group_count);
    }
    if(expected)return;
    for(uint32_t i=0;i<capacity;++i)for(uint32_t b=0;b<40;++b){
      auto want=i<n && b>=1 && b<=lengths[i]?reinterpret_cast<uint8_t*>(src[i])[b-1]:0xa5;
      CHECK(output[(capacity-1-i)*40+b]==want);
    }
    for(auto& h:aiv_model::hardware){CHECK(h.producer==h.completed);CHECK(h.cq_producer==h.cq_consumer);CHECK(h.checkpoints.empty());}
    for(auto& from:model_events)for(auto& to:from)for(auto e:to)CHECK(!e);
  }
};
int main(){
  for(auto shape:{std::array<uint32_t,3>{1,1,2048},{2,2,2048},{1,4,2049},{1,1,2},{1,4,0}}){
    Fixture f(8193,shape[0],shape[1],shape[2]);
    for(int repeat=0;repeat<2;++repeat)for(uint32_t n:{1U,127U,128U,129U,2048U,2049U,4096U,8193U})f.Run(n);
  }
  {Fixture f(65537,1,1,2048);f.Run(65537);f.Run(1);}
  {Fixture f(2,1,1,2048);f.src[1]=UINT64_MAX;auto before=aiv_model::doorbells;f.Run(2,kInvalidLookup);CHECK(before==aiv_model::doorbells);}
  {Fixture f(2,1,1,2048);f.dst[1]=UINT64_MAX;f.Run(2,kInvalidOutput);}
  {Fixture f(2,1,1,2048);f.lengths[1]=0;f.Run(2,kInvalidLookup);}
  {Fixture f(2,1,1,2048);f.lengths[1]=65537;f.Run(2,kInvalidLookup);}
  for(bool timeout:{false,true}){Fixture f(8193,1,2,2048);aiv_model::hardware[1].timeout=timeout;aiv_model::hardware[1].error=!timeout;
    f.Run(8193,timeout?kCqTimeout:kCqError);CHECK(f.queues[0].poisoned&&f.queues[1].poisoned);
    auto before=aiv_model::doorbells;f.Run(1,kInvalidQueue);CHECK(before==aiv_model::doorbells);}
  std::cout<<"Raw-address kernel model: variable spans, padding, windows, ring wrap and CQ faults passed\n";
}
