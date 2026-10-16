"""Execute the actual device CQ wait body with deterministic CPU CQ/clock shims.

This validates control flow and bounded waiting, not fabric or NPU performance.
The function and host diagnostic formatter are extracted from current sources.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "access_operator/demo/aiv_batch_put"


def _body(text, begin, end):
    return text[text.index(begin) : text.index(end, text.index(begin))]


def test_actual_cq_wait_and_host_diagnostics(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    kernel = (DEMO / "kernel/execute_batch_put_aiv.asc").read_text()
    host = (DEMO / "host/batch_put_host.cpp").read_text()
    wait = _body(
        kernel, "__aicore__ inline uint64_t WaitForCompletion(", "\n}  // namespace"
    )
    formatter = _body(
        host,
        "static void LogBatchPutCompletionDiagnostics(",
        '\nextern "C" int32_t KvafExecuteBatchPutSync(',
    )
    source = (
        r"""
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>
#include "batch_put_internal.h"
#define __aicore__
#define __gm__
constexpr uint32_t kMaxCqPolls = 20000U;
constexpr uint32_t kMaxSlowCqPolls = 1000000U;
constexpr uint64_t kSlowCqDeadlineTicks = 100000000ULL;
constexpr uint32_t kCqeProbeBytes = 32U;
constexpr uint32_t kCqConsumerMask = 0x003fffffU;
constexpr uint64_t kSuccess = KVAF_BATCH_PUT_CORE_SUCCESS;
constexpr uint64_t kCqError = KVAF_BATCH_PUT_CORE_CQ_ERROR;
constexpr uint64_t kCqTimeout = KVAF_BATCH_PUT_CORE_CQ_TIMEOUT;
uint64_t ticks, tick_step, arrival;
uint32_t reads, arrival_word, doorbells;
namespace AscendC { uint64_t GetSystemCycle() { return ticks; } }
// Direct device reads see transport DDR without touching an older L1 line.
template<class T> T ReadGmByPassDCache(T *address) {
  if constexpr (std::is_same<T, uint32_t>::value) {
    ++reads; ticks += tick_step;
    if (arrival != 0U && reads >= arrival) *address = arrival_word;
  }
  return *address;
}
template<class T> void WriteGmByPassDCache(T *address, T value) { *address = value; }
enum class MemDsbT { DDR };
template<MemDsbT> void DataSyncBarrier() {}
void CleanCacheLines(uint8_t *, uint64_t) { assert(false && "CQ probe must not write back stale L1"); }
void st_dev(uint32_t value, uint32_t *address, int) {
  *address = value;
  ++doorbells;
}
"""
        + wait
        + formatter
        + r"""
void configure(uint64_t at, uint32_t word, uint64_t step = 1U) {
  ticks = 0; tick_step = step; arrival = at; reads = 0;
  arrival_word = word; doorbells = 0;
}
uint64_t *record(KvafBatchPutCoreStatus &s, uint32_t jetty) {
  return s.reserved + KVAF_BATCH_PUT_CQ_DIAGNOSTIC_BASE_WORD +
      jetty * KVAF_BATCH_PUT_CQ_DIAGNOSTIC_WORDS_PER_JETTY;
}
int main() {
  uint32_t cq[16] = {}, db = 0;
  KvafBatchPutDirectUrmaEntity entity{};
  entity.cq_va = reinterpret_cast<uint64_t>(cq);
  entity.cq_db_va = reinterpret_cast<uint64_t>(&db);
  entity.cq_depth = 16; entity.cqe_size = 4;
  KvafBatchPutQueueState q{};
  KvafBatchPutCoreStatus s{};
  for (uint32_t i=0; i<8; ++i) s.reserved[i]=0xfeed0000+i;
  configure(3, 4);
  assert(WaitForCompletion(&entity,&q,&s,0,32)==kSuccess);
  auto *r=record(s,0);
  assert(reads==3 && q.cq_consumer_index==1 && db==1 && doorbells==1);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_POLLS]==3);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_FAST_BUDGET_EXCEEDED]==0);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_TIMEOUT_REASON]==KVAF_BATCH_PUT_CQ_TIMEOUT_NONE);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_ELAPSED_TICKS]==3);
  // Normal completion does not emit host queue logs.
  LogBatchPutCompletionDiagnostics(&s,1,2);
  std::puts("FAST_PATH_SILENT_PASS");

  q={}; cq[0]=0; s={}; configure(20003,4);
  assert(WaitForCompletion(&entity,&q,&s,1,1)==kSuccess);
  r=record(s,1);
  assert(reads==20003 && q.cq_consumer_index==1 && doorbells==1);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_FAST_BUDGET_EXCEEDED]==1);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_POLLS]==20003);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_SLOW_ELAPSED_TICKS]==3);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_TIMEOUT_REASON]==KVAF_BATCH_PUT_CQ_TIMEOUT_NONE);
  LogBatchPutCompletionDiagnostics(&s,1,2);

  q={}; cq[0]=0; s={}; configure(0,0,10000000);
  assert(WaitForCompletion(&entity,&q,&s,1,1)==kCqTimeout);
  r=record(s,1);
  assert(reads==20010 && q.cq_consumer_index==0 && doorbells==0);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_TIMEOUT_REASON]==KVAF_BATCH_PUT_CQ_TIMEOUT_DEVICE_DEADLINE);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_SLOW_ELAPSED_TICKS]==kSlowCqDeadlineTicks);
  LogBatchPutCompletionDiagnostics(&s,1,2);

  q={}; cq[0]=0; s={}; configure(0,0,0);
  assert(WaitForCompletion(&entity,&q,&s,0,32)==kCqTimeout);
  r=record(s,0);
  assert(reads==kMaxCqPolls+kMaxSlowCqPolls && doorbells==0);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_TIMEOUT_REASON]==KVAF_BATCH_PUT_CQ_TIMEOUT_POLL_FALLBACK);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_ELAPSED_TICKS]==0);
  LogBatchPutCompletionDiagnostics(&s,1,2);

  // Real CQ error still consumes exactly one CQE and is never retried.
  q={}; cq[0]=0; s={}; configure(1,4U|(2U<<24U)|(3U<<16U));
  assert(WaitForCompletion(&entity,&q,&s,1,1)==(kCqError|(2U<<8U)|(3U<<16U)));
  assert(reads==1 && q.cq_consumer_index==1 && doorbells==1);
  r=record(s,1);
  assert(r[KVAF_BATCH_PUT_CQ_DIAG_RESULT]==kCqError);
  LogBatchPutCompletionDiagnostics(&s,1,2);

  // Valid owner after the ring wraps is zero, not always one.
  q={}; q.cq_consumer_index=16; cq[0]=4; s={}; configure(2,0);
  assert(WaitForCompletion(&entity,&q,&s,0,32)==kSuccess);
  assert(reads==2 && q.cq_consumer_index==17 && db==17);
  assert(record(s,0)[KVAF_BATCH_PUT_CQ_DIAG_EXPECTED_OWNER]==1);

  // Reproduce two-probe counters: jetty0 succeeds, jetty1 times out.
  s={}; for(uint32_t i=0;i<8;++i) s.reserved[i]=0xfeed0000+i;
  q={}; q.sq_producer_index=33; q.cq_consumer_index=1;
  cq[1]=0; configure(1,4);
  assert(WaitForCompletion(&entity,&q,&s,0,32)==kSuccess);
  assert(q.cq_consumer_index==2);
  KvafBatchPutQueueState q1{}; q1.sq_producer_index=1;
  cq[0]=0; configure(0,0,10000000);
  assert(WaitForCompletion(&entity,&q1,&s,1,1)==kCqTimeout);
  assert(record(s,0)[KVAF_BATCH_PUT_CQ_DIAG_CQ_AFTER]==2);
  assert(record(s,1)[KVAF_BATCH_PUT_CQ_DIAG_CQ_AFTER]==0);
  assert(record(s,0)[KVAF_BATCH_PUT_CQ_DIAG_ASSIGNED_ROWS]==32);
  assert(record(s,1)[KVAF_BATCH_PUT_CQ_DIAG_ASSIGNED_ROWS]==1);
  for(uint32_t i=0;i<8;++i) assert(s.reserved[i]==0xfeed0000+i);
  LogBatchPutCompletionDiagnostics(&s,1,2);
  assert(sizeof(s)==384 && sizeof(KvafExecuteBatchPutKernelArgs)==56);
  std::puts("BATCH_PUT_CQ_CPU_PASS cases=7 ABI=384 SGE_0_7_preserved=1");
}
"""
    )
    cpp = tmp_path / "cq_wait.cpp"
    cpp.write_text(source)
    executable = tmp_path / "cq_wait"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I",
            str(DEMO / "include"),
            str(cpp),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=20
    )
    (tmp_path / "cq.stdout.log").write_text(result.stdout)
    (tmp_path / "cq.stderr.log").write_text(result.stderr)
    assert "BATCH_PUT_CQ_CPU_PASS cases=7" in result.stdout
    assert "late_success=1 timeout_reason=none" in result.stderr
    assert "timeout_reason=device_deadline" in result.stderr
    assert "timeout_reason=poll_fallback" in result.stderr
    assert "core=0 jetty=1 result=8" in result.stderr
    assert "core=0 jetty=0 result=0 rows=32" in result.stderr
    assert "core=0 jetty=1 result=7 rows=1" in result.stderr
    assert result.stderr.startswith(
        "[KVAF_AIV] BATCH_PUT_CQ_WAIT_DETAILS core=0 jetty=1 result=0"
    )
    assert "cqe_valid=0" in result.stderr
    assert "cqe_valid=1" in result.stderr
    assert "eager_put_cq_not_Get_performance" in result.stderr
    # A dirty scalar-L1 line left from an earlier allocation is independent
    # of a fresh NIC DDR write. Reconstruct the previous clean/cached probe:
    # its writeback destroys the injected completion; bypass preserves it.
    cached_probe = "ReadCachedCq(reinterpret_cast<__gm__ uint32_t *>(cqe_address))"
    cached_wait = wait.replace(
        "ReadGmByPassDCache<uint32_t>(reinterpret_cast<__gm__ uint32_t *>(cqe_address))",
        cached_probe,
    )
    assert cached_wait.count(cached_probe) == 2
    stale_l1 = r"""
uint32_t ReadCachedCq(uint32_t *address) {
  ++reads; ticks += tick_step;
  if (arrival != 0U && reads >= arrival) *address = arrival_word; // NIC DDR write.
  *address = 0U; // DCCI writes the older dirty L1 line back before its GM read.
  return *address;
}
"""
    cached_source = source.replace(wait, stale_l1 + cached_wait)
    cached_cpp = tmp_path / "cq_cached_probe.cpp"
    cached_cpp.write_text(cached_source)
    cached_executable = tmp_path / "cq_cached_probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(DEMO / "include"),
            str(cached_cpp),
            "-o",
            str(cached_executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cached = subprocess.run(
        [str(cached_executable)], capture_output=True, text=True, timeout=20
    )
    (tmp_path / "old_dcci_probe.failure.log").write_text(cached.stderr)
    assert cached.returncode != 0 and "WaitForCompletion" in cached.stderr


def test_does_not_change_registration_or_get_contract():
    kernel = (DEMO / "kernel/execute_batch_put_aiv.asc").read_text()
    host = (DEMO / "host/batch_put_host.cpp").read_text()
    assert "constexpr uint32_t kMaxCqPolls = 20000U;" in kernel
    assert "bool non_pin = true;" in host
    assert 'std::strcmp(token_source, "process") == 0' in host
    assert "KVAF_BATCH_PUT_CQ_DIAGNOSTIC_BASE_WORD +" in kernel
    assert "BATCH_PUT_CQ_WAIT_DETAILS core=%u jetty=%u" in host
