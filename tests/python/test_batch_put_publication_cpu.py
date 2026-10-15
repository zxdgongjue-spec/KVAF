"""Exercise actual Put publication helpers against a delayed DMA CPU model.

The model checks queue/head/doorbell ordering and ring copies. It cannot prove
that the publication change resolves a fabric or NPU failure.
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
    structs = _between(kernel, "struct PreparedPost {", "__aicore__ inline void CleanCacheLines(")
    clean = _between(kernel, "__aicore__ inline void CleanCacheLines(", "__aicore__ inline uint32_t WqeControl(")
    copy = _between(kernel, "__aicore__ inline void CopyUbToGm(", "__simt_vf__")
    publish = _between(kernel, "__aicore__ inline void CopyPreparedPostToSq(", "__aicore__ inline void PostBatchPutRequests(")
    source = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "batch_put_internal.h"
#define __aicore__
#define __gm__
#define __ubuf__
enum pipe_t { PIPE_V, PIPE_MTE3, PIPE_S, PIPE_ALL };
using event_t = unsigned;
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
enum class MemDsbT { DDR };
template<typename T> struct GlobalTensor {
  T *base;
  void SetGlobalBuffer(T *p) { base=p; }
  T &operator[](uint64_t offset) { return base[offset]; }
};
constexpr uint32_t kWqeBytes = 64U;
constexpr uint32_t kWqeBankBytes = 2048U;
constexpr uint64_t kCacheLineBytes = 64U;
alignas(64) uint8_t ub[4096];
struct DmaCopy { uint8_t *destination; std::vector<uint8_t> data; };
std::vector<DmaCopy> pending_dma;
KvafBatchPutQueueState *current_queue;
uint32_t *current_doorbell;
uint64_t durable_head, clean_sequence, sync_sequence;
uint32_t doorbells;
std::array<std::array<std::array<unsigned,2>,4>,4> events;
void *get_imm(uint32_t offset) { return ub+offset; }
void wait_flag(pipe_t from, pipe_t to, event_t id) {
  assert(events[from][to][id] != 0);
  --events[from][to][id];
}
void set_flag(pipe_t from, pipe_t to, event_t id) {
  ++events[from][to][id];
}
template<pipe_t pipe> void PipeBarrier() {
  static_assert(pipe == PIPE_ALL);
  for (auto &copy : pending_dma)
    std::memcpy(copy.destination, copy.data.data(), copy.data.size());
  pending_dma.clear();
}
void copy_ubuf_to_gm_align_v2(float *dst, float *src,
                             uint64_t config0, uint64_t) {
  const auto *bytes = reinterpret_cast<uint8_t *>(src);
  pending_dma.push_back({reinterpret_cast<uint8_t *>(dst),
                        std::vector<uint8_t>(bytes,bytes+(config0>>25U))});
}
template<typename T, CacheLine, DcciDst>
void DataCacheCleanAndInvalid(T &line) {
  // Each AIV owns a complete 64-byte queue line. Model its writeback,
  // including neighbors, so a PI-only overwrite cannot silently pass.
  assert(reinterpret_cast<uint8_t *>(&line) ==
         reinterpret_cast<uint8_t *>(current_queue));
  durable_head=current_queue->sq_producer_index;
  ++clean_sequence;
}
template<MemDsbT> void DataSyncBarrier() {
  assert(pending_dma.empty());
  sync_sequence=clean_sequence;
}
void st_dev(uint32_t value, uint32_t *address, int) {
  assert(address==current_doorbell);
  assert(pending_dma.empty());
  assert(clean_sequence==doorbells+1);
  assert(sync_sequence==clean_sequence);
  assert(value==static_cast<uint32_t>(durable_head));
  *address=value;
  ++doorbells;
}
''' + structs + clean + copy + publish + r'''
void run(uint32_t bank, uint32_t head, uint32_t count, bool release_bank) {
  constexpr uint32_t depth=64;
  alignas(64) uint8_t sq[depth*kWqeBytes]={};
  alignas(64) KvafBatchPutQueueState queue{};
  uint32_t db=0;
  current_queue=&queue; current_doorbell=&db;
  queue.magic=0x4b564151U; queue.initialized=1;
  queue.sq_producer_index=head; queue.cq_consumer_index=17;
  queue.completed_batch_count=8; queue.last_error=0;
  for(uint32_t i=0;i<3;++i) queue.reserved[i]=0xfeed0000U+i;
  durable_head=head; clean_sequence=0; sync_sequence=0; doorbells=0;
  pending_dma.clear(); events={};
  for(uint32_t i=0;i<kWqeBankBytes;++i)
    ub[bank*kWqeBankBytes+i]=static_cast<uint8_t>((i*13U+bank+1U)&255U);
  set_flag(PIPE_V,PIPE_MTE3,bank);
  PreparedPost prepared{
    reinterpret_cast<uint64_t>(sq),reinterpret_cast<uint64_t>(&db),
    reinterpret_cast<uint64_t>(&queue),head,head+count,depth,count,bank};
  PendingPost pending{};
  PreparePostMte3Stage(&prepared,release_bank,&pending);
  // SQ data must be published before its producer is committed.
  assert(pending_dma.empty());
  for(uint32_t i=0;i<count*kWqeBytes;++i)
    assert(sq[((head%depth)*kWqeBytes+i)%(depth*kWqeBytes)]==
           ub[bank*kWqeBankBytes+i]);
  assert(queue.sq_producer_index==head && doorbells==0);
  CommitPostScalarStage(&pending);
  assert(db==head+count && queue.sq_producer_index==head+count);
  assert(doorbells==1 && durable_head==head+count);
  assert(queue.magic==0x4b564151U && queue.initialized==1);
  assert(queue.cq_consumer_index==17 && queue.completed_batch_count==8);
  assert(queue.last_error==0);
  for(uint32_t i=0;i<3;++i) assert(queue.reserved[i]==0xfeed0000U+i);
  if(release_bank) wait_flag(PIPE_MTE3,PIPE_V,bank);
  for(auto &from:events) for(auto &to:from) for(auto count:to) assert(count==0);
}
int main() {
  for(uint32_t bank=0;bank<2;++bank)
    for(uint32_t count : {1U,5U,6U,8U,31U,32U}) {
      run(bank,0,count,false);
      run(bank,57,count,true);
      run(bank,128,count,true);
    }
  assert(sizeof(KvafBatchPutQueueState)==64);
  std::puts("BATCH_PUT_PUBLICATION_CPU_PASS cases=36 ring_wrap=1 ABI=64");
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
    assert "BATCH_PUT_PUBLICATION_CPU_PASS cases=36" in result.stdout
