"""Execute actual Full/Dense CQ helpers with separate DDR/AIV cache views.

This models the documented non-coherent Scalar cache, not HCCP bandwidth or
hardware instruction timing. A write-back of CQ must never undo a bypass SQ
commit on the same 64-byte queue line. No production source is rewritten.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "access_operator/demo/aiv_batch_get"


def function(source, name):
    start = source.index("__aicore__ inline", source.rfind("\n", 0, source.index(name + "(")))
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


SDK = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "batch_get_abi.h"
#include "batch_get_compact_prefix.h"
#define __aicore__
#define __gm__
#define __ubuf__
unsigned lane=0;
struct Cache {
  std::array<uint64_t,6> ddr{};
  std::array<uint64_t,6> copy[2]{};
  bool valid[2]{}, dirty[2]{};
  uint64_t Read(unsigned i) {
    if(!valid[lane]) {copy[lane]=ddr;valid[lane]=true;} return copy[lane][i];
  }
  void Write(unsigned i,uint64_t v) {Read(i);copy[lane][i]=v;dirty[lane]=true;}
  void Clean() {if(dirty[lane]) ddr=copy[lane];dirty[lane]=valid[lane]=false;}
};
template<class T> struct Word {
  Cache* cache;unsigned index;
  operator T() const {return static_cast<T>(cache->Read(index));}
  Word& operator=(T v) {cache->Write(index,v);return *this;}
  T operator++() {T v=T(*this)+1;*this=v;return v;}
};
struct ModelQueue {
  Cache cache;
  Word<uint32_t> magic{&cache,0},initialized{&cache,1};
  Word<uint64_t> sq_producer_index{&cache,2},cq_consumer_index{&cache,3},
    completed_batch_count{&cache,4},last_error{&cache,5};
};
#define KvafBatchGetQueueState ModelQueue
ModelQueue* live_queue;
template<class T> T ReadGmByPassDCache(Word<T>* p) {return p->cache->ddr[p->index];}
template<class T> void WriteGmByPassDCache(Word<T>* p,T value) {p->cache->ddr[p->index]=value;}
template<class T> T ReadGmByPassDCache(T* p) {return *p;}
template<class T> void WriteGmByPassDCache(T* p,T value) {*p=value;}
namespace AscendC {
enum class MemDsbT {DDR};
template<MemDsbT> void DataSyncBarrier() {}
uint64_t GetSystemCycle() {return 1;}
}
using namespace AscendC;
void CleanCacheLines(uint8_t* p,uint64_t) {
  if(p==reinterpret_cast<uint8_t*>(live_queue)) live_queue->cache.Clean();
}
void st_dev(uint32_t value,uint32_t* address,int) {*address=value;}
alignas(64) uint8_t ub[160*1024];
uintptr_t get_imm(uint32_t offset) {return reinterpret_cast<uintptr_t>(ub+offset);}
constexpr uint32_t kEntityMagic=0x4b564155U,kEntityVersion=1,kQueueMagic=0x4b564151U,
  kTransferBytes=656,kWqeBytes=64,kCqeProbeBytes=32,kCqConsumerMask=0x003fffffU,
  kMaxCqPolls=20,kWqesPerPost=32,kActiveRequestCapacity=16384,kDenseRequestCapacity=256,
  kActiveTableIndicesUbOffset=4096,kActiveRequestIdsUbOffset=69632;
enum ExecuteResult:uint64_t {kSuccess=0,kInvalidArgs=1,kInvalidEntity=2,kInvalidQueue=3,
  kInvalidLookup=4,kQueueFull=5,kCqTimeout=6,kCqError=7,kInvalidOutput=8};
template<class... T> void WriteProfile(T...) {}
void ConfigureUbToGmCopy() {}
uint32_t LoadBatchGetCompactPrefix(const int32_t*,uint32_t begin,uint32_t end,uint32_t*,uint32_t*) {return end-begin;}
uint32_t CompactBatchGetRequests(const uint8_t*,const int32_t*,uint32_t begin,uint32_t end,uint32_t*,uint32_t*) {return end-begin;}
uint64_t hardware_sq=0,hardware_cq=0;
bool inject_error=false,inject_timeout=false;
void PostActiveRequests(const uint32_t*,const uint32_t*,const int32_t*,const int32_t*,
    const uint64_t*,uint32_t count,uint32_t,uint32_t,const KvafBatchGetDirectUrmaEntity* e,
    ModelQueue* q,uint32_t,uint64_t*,uint64_t*) {
  if(!count) return;
  // The real CommitPostScalarStage uses precisely this bypass publication.
  const auto head=ReadGmByPassDCache<uint64_t>(&q->sq_producer_index);
  assert(head==hardware_sq && "SQ rewound before a new hardware Post");
  hardware_sq+=count;
  WriteGmByPassDCache<uint64_t>(&q->sq_producer_index,hardware_sq);
  if(inject_timeout) return;
  auto* cqe=reinterpret_cast<uint32_t*>(e->cq_va+hardware_cq%e->cq_depth*e->cqe_size);
  *cqe=((1U-((hardware_cq/e->cq_depth)&1U))<<2U)|(inject_error?1U<<24U:0U);
  ++hardware_cq;
}
void ValidateDenseBatchGetDescriptors(const uint64_t*,const uint8_t*,const int32_t*,const int32_t*,
  const int32_t*,uint32_t,uint32_t,uint32_t,uint32_t,const KvafBatchGetDirectUrmaEntity*,uint64_t* result,uint64_t*) {
  *result=0;
}
void PrepareDenseBatchGetOutputForInboundDma(const int32_t*,uint32_t,uint64_t,uint32_t) {}
void PostDenseBatchGetRequests(const int32_t*,const int32_t*,const int32_t*,const uint64_t*,
  uint32_t count,uint32_t block_bytes,uint32_t jettys,const KvafBatchGetDirectUrmaEntity* e,
  ModelQueue* q,uint32_t profile,uint64_t* doorbell,uint64_t* last) {
  PostActiveRequests(nullptr,nullptr,nullptr,nullptr,nullptr,count,block_bytes,jettys,e,q,profile,doorbell,last);
}
"""

MAIN = r"""
int main() {
  for(uint32_t count:{0U,17U,256U,600U,8191U}) {
    ModelQueue q;live_queue=&q;lane=0;
    KvafBatchGetCoreStatus status{};
    std::array<uint32_t,32> cq{};uint32_t db=0;
    KvafBatchGetDirectUrmaEntity entity{};
    entity.magic=kEntityMagic;entity.version=kEntityVersion;
    entity.local_hbm_address=4096;entity.local_hbm_size=8192*656;entity.local_token_id=1;
    entity.remote_host_ddr_address=4096;entity.remote_host_ddr_size=8192*656;
    entity.sq_va=1;entity.sq_db_va=1;entity.sq_depth=8192;entity.wqe_size=64;
    entity.cq_va=uintptr_t(cq.data());entity.cq_db_va=uintptr_t(&db);entity.cq_depth=4;entity.cqe_size=32;
    int32_t active=count;
    KvafExecuteBatchGetKernelArgs a{};
    a.block_count=a.jettys_per_core=1;a.address_table=a.valid_table=a.indices=a.offsets=a.output_indices=1;
    a.mask=uintptr_t(&active);a.output=entity.local_hbm_address;a.core_status=uintptr_t(&status);
    a.channel_entity=uintptr_t(&entity);a.queue_state=uintptr_t(&q);a.table_capacity=8192;
    a.request_capacity=a.output_rows=8192;a.block_bytes=656;a.selection_mode=KVAF_BATCH_GET_SELECTION_COMPACT_COUNT;
    hardware_sq=hardware_cq=0;
    for(unsigned epoch=0;epoch<12;++epoch) {
      // Execute the actual dense state machine before the terminal Full Get.
      lane=epoch%2;
      a.request_capacity=256;
      KvafDenseBatchGetTransaction transaction;
      BeginKvafDenseBatchGetPostOnly(a,18,0,&transaction);
      assert(transaction.state==kDenseBatchGetInFlight);
      FinishKvafDenseBatchGet(a,&transaction);
      assert(transaction.result==0 && transaction.finalized==1);
      a.request_capacity=8192;
      // Terminal SEND can land on the other AIV, with a different cache view.
      lane=1-lane;q.cache.Clean();
      RunKvafFullBatchGetCore(a,0);
      assert(status.result==0);
      q.cache.Clean(); // Exact Deep terminal release before the next MIX.
      assert(ReadGmByPassDCache<uint64_t>(&q.sq_producer_index)==hardware_sq && "Full Get flush rewound bypass SQ PI");
      assert(ReadGmByPassDCache<uint64_t>(&q.cq_consumer_index)==hardware_cq);
      assert(status.sq_producer_index==hardware_sq);
      assert(status.cq_consumer_index==hardware_cq);
      assert(status.completed_batch_count==(count?2:1)*(epoch+1));
    }
    // Error/timeout must remain errors; no fabricated completion/reset.
    if(count) for(unsigned kind=0;kind<2;++kind) {
      inject_error=kind==0;inject_timeout=kind==1;
      const auto completed=ReadGmByPassDCache<uint64_t>(&q.completed_batch_count);
      RunKvafFullBatchGetCore(a,0);q.cache.Clean();
      assert(status.result==(kind==0?kCqError:kCqTimeout));
      assert(ReadGmByPassDCache<uint64_t>(&q.sq_producer_index)==hardware_sq);
      assert(ReadGmByPassDCache<uint64_t>(&q.completed_batch_count)==completed);
      inject_error=inject_timeout=false;
    }
  }
  // Standalone native Full Get: first-use initialization and changes to an
  // empty prefix must not reset persistent SQ/CQ positions on later calls.
  ModelQueue q;live_queue=&q;lane=0;hardware_sq=hardware_cq=0;
  KvafBatchGetCoreStatus status{};std::array<uint32_t,32> cq{};uint32_t db=0;
  KvafBatchGetDirectUrmaEntity entity{};
  entity.magic=kEntityMagic;entity.version=kEntityVersion;
  entity.local_hbm_address=4096;entity.local_hbm_size=8192*656;entity.local_token_id=1;
  entity.remote_host_ddr_address=4096;entity.remote_host_ddr_size=8192*656;
  entity.sq_va=entity.sq_db_va=1;entity.sq_depth=8192;entity.wqe_size=64;
  entity.cq_va=uintptr_t(cq.data());entity.cq_db_va=uintptr_t(&db);entity.cq_depth=4;entity.cqe_size=32;
  int32_t active=0;KvafExecuteBatchGetKernelArgs a{};
  a.block_count=a.jettys_per_core=1;a.address_table=a.valid_table=a.indices=a.offsets=a.output_indices=1;
  a.mask=uintptr_t(&active);a.output=entity.local_hbm_address;a.core_status=uintptr_t(&status);
  a.channel_entity=uintptr_t(&entity);a.queue_state=uintptr_t(&q);a.table_capacity=8192;
  a.request_capacity=a.output_rows=8192;a.block_bytes=656;a.selection_mode=KVAF_BATCH_GET_SELECTION_COMPACT_COUNT;
  for(unsigned epoch=0;epoch<24;++epoch) {
    lane=epoch%2;active=epoch%3?33:0;
    RunKvafFullBatchGetCore(a,0);q.cache.Clean();
    assert(status.result==0 && status.sq_producer_index==hardware_sq);
    assert(status.cq_consumer_index==hardware_cq);
    assert(ReadGmByPassDCache<uint64_t>(&q.sq_producer_index)==hardware_sq);
    assert(ReadGmByPassDCache<uint64_t>(&q.cq_consumer_index)==hardware_cq);
  }
  std::puts("KVAF_MIXED_QUEUE_CACHE_PASS mixed_epochs=60 native_epochs=24 cq_wrap=1 sq_wrap=1 errors_retained=1 hardware=0");
}
"""


def test_actual_full_get_preserves_dense_bypass_queue_across_epochs(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = (DEMO / "kernel/execute_batch_get_aiv.asc").read_text()
    program = SDK + "\n".join(function(source, name) for name in (
        "WriteStatus", "ActiveRequestsForJetty", "WaitForCompletion", "RunKvafFullBatchGetCore",
    ))
    program += source[source.index("enum DenseBatchGetTransactionState"):source.index("// Begin performs")]
    program += "\n".join(function(source, name) for name in (
        "BeginKvafDenseBatchGetPostOnly", "TryRetireKvafDenseBatchGet", "FinishKvafDenseBatchGet",
    )) + MAIN
    executable = tmp_path / "queue_replay"
    built = subprocess.run([compiler, "-std=c++17", "-O0", "-fsanitize=address,undefined",
                            "-I", str(DEMO / "include"), "-x", "c++", "-", "-o", str(executable)],
                           input=program, text=True, capture_output=True)
    assert built.returncode == 0, built.stderr
    run = subprocess.run([str(executable)], text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert "KVAF_MIXED_QUEUE_CACHE_PASS" in run.stdout


def test_full_post_loads_the_same_durable_pi_that_commit_publishes():
    source = (DEMO / "kernel/execute_batch_get_aiv.asc").read_text()
    post = function(source, "PostActiveRequests")
    commit = function(source, "CommitPostScalarStage")
    assert "ReadGmByPassDCache<uint64_t>(\n        &queues[jetty].sq_producer_index)" in post
    assert "current_heads[jetty] = queues[jetty].sq_producer_index" not in post
    assert "WriteGmByPassDCache<uint64_t>(&queue->sq_producer_index" in commit
