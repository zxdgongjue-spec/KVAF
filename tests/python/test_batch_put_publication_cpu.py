"""Execute scalar Put publication against durable-memory and corruption models.

This checks the actual device helpers, including every WQE word, doorbell
suppression on bad SQ contents, and owner publication across ring wrap.
It does not assert that an NPU fabric failure is resolved.
"""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "access_operator/demo/aiv_batch_put"


def _between(text, begin, end):
    first = text.index(begin)
    return text[first:text.index(end, first)]


def test_actual_put_sq_and_producer_publication(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    kernel = (DEMO / "kernel/execute_batch_put_aiv.asc").read_text()
    helpers = _between(kernel, "__aicore__ inline void CleanCacheLines(",
                       "__aicore__ inline uint32_t BatchPutRequestsForJetty(")
    assert "VF_CALL" not in kernel
    assert "PIPE_MTE3" not in kernel
    source = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>
#include "batch_put_internal.h"
#define __aicore__
#define __gm__
enum pipe_t { PIPE_ALL };
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
enum class MemDsbT { DDR };
template<typename T> struct GlobalTensor {
  T *base;
  void SetGlobalBuffer(T *p) { base=p; }
  T &operator[](uint64_t offset) { return base[offset]; }
};
constexpr uint32_t kWqeBytes=64, kWqesPerPost=32, kWriteOpcode=3;
constexpr uint64_t kCacheLineBytes=64, kSuccess=0;
constexpr uint32_t depth=64;
alignas(64) uint32_t sq[2][depth][16];
alignas(64) KvafBatchPutQueueState queues[2];
uint32_t durable[2][depth][16], doorbell[2];
uint64_t old_heads[2], expected_heads[2];
uint64_t offsets[96], lengths[96], addresses[96];
KvafBatchPutDirectUrmaEntity entities[2];
unsigned calls, barriers, flushes, synced_flushes;
int inject_word, inject_pass;
bool injected;
std::map<uintptr_t,unsigned> line_passes;
std::array<uint32_t,16> expected[2][depth];
void check_payload(uint32_t jetty,uint32_t ring,bool owner_valid) {
  for(uint32_t word=0;word<16;++word)
    assert(durable[jetty][ring][word]==
        (expected[jetty][ring][word]^((word==0 && !owner_valid)?0x80000000U:0U)));
}
template<pipe_t> void PipeBarrier() { ++barriers; }
template<typename T, CacheLine, DcciDst>
void DataCacheCleanAndInvalid(T &line) {
  auto ptr=reinterpret_cast<uintptr_t>(&line);
  for(uint32_t jetty=0;jetty<2;++jetty) {
    auto lo=reinterpret_cast<uintptr_t>(sq[jetty]);
    auto hi=lo+sizeof(sq[jetty]);
    if(ptr>=lo && ptr<hi) {
      const uint32_t ring=(ptr-lo)/64;
      auto pass=++line_passes[ptr];
      // A valid owner can be written only after every payload in this
      // doorbell batch has reached the durable memory image.
      if(pass==2 && !injected) {
        uint64_t stop=expected_heads[jetty];
        if(stop>old_heads[jetty]+32) stop=old_heads[jetty]+32;
        for(uint64_t head=old_heads[jetty];head<stop;++head) {
          const uint32_t slot=head%depth;
          for(uint32_t word=1;word<16;++word)
            assert(durable[jetty][slot][word]==expected[jetty][slot][word]);
        }
      }
      if(!injected && inject_word>=0 && pass==static_cast<unsigned>(inject_pass)) {
        sq[jetty][ring][inject_word]^=0x100U;
        injected=true;
      }
      std::memcpy(durable[jetty][ring],sq[jetty][ring],64);
      ++flushes;
      return;
    }
    if(ptr==reinterpret_cast<uintptr_t>(&queues[jetty])) {
      ++flushes;
      return;
    }
  }
  assert(false && "DCCI outside SQ or owned queue");
}
template<MemDsbT> void DataSyncBarrier() { synced_flushes=flushes; }
void st_dev(uint32_t value,uint32_t *address,int) {
  uint32_t jetty=address==&doorbell[0]?0:1;
  assert(address==&doorbell[jetty] && value<=expected_heads[jetty]);
  assert(flushes==synced_flushes && barriers!=0);
  for(uint64_t head=old_heads[jetty];head<value;++head)
    check_payload(jetty,head%depth,true);
  assert(queues[jetty].sq_producer_index==value);
  *address=value;
  old_heads[jetty]=value;
  ++calls;
}
''' + helpers + r'''
void run(uint32_t head,uint32_t count,uint32_t jettys,int bad_word=-1,int bad_pass=1) {
  std::memset(sq,0,sizeof(sq)); std::memset(durable,0,sizeof(durable));
  std::memset(queues,0,sizeof(queues)); std::memset(entities,0,sizeof(entities));
  calls=barriers=flushes=synced_flushes=0; inject_word=bad_word;
  inject_pass=bad_pass; injected=false; line_passes.clear();
  for(uint32_t j=0;j<2;++j) {
    old_heads[j]=expected_heads[j]=head; queues[j].sq_producer_index=head;
    queues[j].cq_consumer_index=17; queues[j].completed_batch_count=8;
    entities[j].local_hbm_address=0x100000000ULL;
    entities[j].sq_va=reinterpret_cast<uint64_t>(sq[j]);
    entities[j].sq_db_va=reinterpret_cast<uint64_t>(&doorbell[j]);
    entities[j].sq_depth=depth; entities[j].tp_id=0x123456;
    entities[j].remote_token_id=0x1357; entities[j].remote_token_value=0xabcd;
    entities[j].local_token_id=0x2468;
    for(uint32_t k=0;k<16;++k) entities[j].remote_eid[k]=k;
    doorbell[j]=head;
  }
  for(uint32_t i=0;i<count;++i) {
    offsets[i]=i*83968ULL; lengths[i]=83968; addresses[i]=0x200000000ULL+i*83968ULL;
    const uint32_t j=(i/32)%jettys;
    const uint64_t pi=expected_heads[j]++;
    const uint32_t post=i/32,total=(count+31)/32;
    const bool completion=post+jettys>=total && ((i+1)%32==0 || i+1==count);
    BuildBatchPutWqeWords(&entities[j],offsets[i],lengths[i],addresses[i],pi,
                         completion,expected[j][pi%depth].data());
    const auto &w=expected[j][pi%depth];
    assert((w[0]&0xffffU)==pi%depth);
    assert(((w[0]>>31)&1U)==(1U-((pi/depth)&1U)));
    assert(w[1]==0x300 && w[2]==0x1123456 && w[3]==0x1357);
    assert(w[8]==0xabcd && w[9]==0 && w[12]==83968 && w[13]==0x2468);
    assert((uint64_t(w[15])<<32|w[14])==entities[j].local_hbm_address+offsets[i]);
    assert((uint64_t(w[11])<<32|w[10])==addresses[i]);
  }
  KvafBatchPutCoreStatus status{}; uint64_t last=0, failed=UINT64_MAX;
  const uint64_t result=PostBatchPutRequests(offsets,lengths,addresses,0,count,
                                             jettys,entities,queues,&status,&last,&failed);
  if(bad_word>=0) {
    assert(failed!=UINT64_MAX);
    assert(injected && result==KVAF_BATCH_PUT_CORE_SQ_PUBLICATION_ERROR);
    assert(calls==0 && doorbell[0]==head && queues[0].sq_producer_index==head);
    assert(status.reserved[KVAF_BATCH_PUT_SQ_DIAG_MISMATCH_WORD]!=0);
  } else {
    assert(result==0 && calls==(count+31)/32);
    for(uint32_t j=0;j<jettys;++j) {
      assert(old_heads[j]==expected_heads[j] && queues[j].sq_producer_index==expected_heads[j]);
      assert(queues[j].cq_consumer_index==17 && queues[j].completed_batch_count==8);
      if(expected_heads[j]!=head)
        assert(status.reserved[j*4+KVAF_BATCH_PUT_SQ_DIAG_PUBLISHED_PI]==expected_heads[j]);
    }
  }
  assert(status.reserved[KVAF_BATCH_PUT_SQ_DIAGNOSTIC_MAGIC_WORD]==KVAF_BATCH_PUT_SQ_DIAGNOSTIC_MAGIC);
}
int main() {
  for(uint32_t j:{1U,2U}) for(uint32_t head:{0U,57U,128U})
    for(uint32_t count:{1U,5U,8U,31U,32U,33U,63U}) run(head,count,j);
  for(int word=0;word<16;++word) for(int pass=1;pass<=2;++pass) run(57,8,2,word,pass);
  assert(sizeof(KvafBatchPutQueueState)==64 && sizeof(KvafBatchPutCoreStatus)==384);
  std::puts("BATCH_PUT_SCALAR_PUBLICATION_CPU_PASS valid=42 corrupted=32 ring_wrap=1");
}
'''
    cpp = tmp_path / "put_publication.cpp"
    cpp.write_text(source)
    executable = tmp_path / "put_publication"
    subprocess.run([compiler, "-std=c++17", "-O2", "-fsanitize=address,undefined",
                    "-I", str(DEMO / "include"), str(cpp), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True,
                            text=True)
    assert "BATCH_PUT_SCALAR_PUBLICATION_CPU_PASS valid=42 corrupted=32" in result.stdout
