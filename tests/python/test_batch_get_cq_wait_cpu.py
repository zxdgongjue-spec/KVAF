"""Execute the shared device CQ wait body with deterministic CPU CQ arrivals.

The current function, result constants and real ABI are used directly. This
checks completion control flow, not DMA, cache behavior or NPU performance.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "access_operator/demo/aiv_batch_get"


def test_actual_shared_cq_wait_with_real_abi(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    kernel = (DEMO / "include/kvaf_batch_get_device.h").read_text()
    begin = kernel.index("__aicore__ inline uint64_t WaitForCompletion(")
    wait = kernel[begin:kernel.index("\n// One ordinary device call", begin)]
    constants = "\n".join(
        line for line in kernel.splitlines()
        if line.startswith(("constexpr uint32_t kMaxCqPolls =",
                            "constexpr uint32_t kCqeProbeBytes =",
                            "constexpr uint32_t kCqConsumerMask ="))
    )
    begin = kernel.index("enum ExecuteResult : uint64_t {")
    results = kernel[begin:kernel.index("\nstruct PreparedPost", begin)]
    code = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include "batch_get_abi.h"
#define __aicore__
#define __gm__
''' + constants + results + r'''
uint32_t reads, arrival, word, doorbells;
uint8_t *expected_probe;
void CleanCacheLines(uint8_t *address, uint64_t bytes) {
  assert(address == expected_probe && bytes == kCqeProbeBytes);
  if (++reads == arrival) *reinterpret_cast<uint32_t *>(address) = word;
}
void st_dev(uint32_t value, uint32_t *address, int) {
  *address = value;
  ++doorbells;
}
''' + wait + r'''
void configure(uint32_t at, uint32_t cq_word, uint32_t *probe) {
  reads = doorbells = 0;
  arrival = at;
  word = cq_word;
  expected_probe = reinterpret_cast<uint8_t *>(probe);
}
int main() {
  // Independent, correctly sized CQE slots, rather than a one-word fake ABI.
  alignas(64) uint32_t cq[16][16] = {}, other_cq[16][16] = {};
  uint32_t db = 123, other_db = 456, polls = 0;
  KvafBatchGetDirectUrmaEntity entity{};
  entity.cq_va = reinterpret_cast<uint64_t>(cq);
  entity.cq_db_va = reinterpret_cast<uint64_t>(&db);
  entity.cq_depth = 16;
  entity.cqe_size = sizeof(cq[0]);
  KvafBatchGetQueueState q{};
  q.sq_producer_index = 160;

  configure(3, 4, cq[0]);
  assert(WaitForCompletion(&entity, &q, &polls) == kSuccess);
  assert(reads == 3 && polls == 3 && q.cq_consumer_index == 1);
  assert(db == 1 && doorbells == 1 && q.sq_producer_index == 160);

  // The final permitted arrival succeeds; a later arrival is not consumed.
  q.cq_consumer_index = 0; cq[0][0] = 0;
  configure(kMaxCqPolls, 4, cq[0]);
  assert(WaitForCompletion(&entity, &q, &polls) == kSuccess);
  assert(polls == kMaxCqPolls && q.cq_consumer_index == 1 && doorbells == 1);
  q.cq_consumer_index = 0; cq[0][0] = 0; db = 777;
  configure(kMaxCqPolls + 1, 4, cq[0]);
  assert(WaitForCompletion(&entity, &q, &polls) == kCqTimeout);
  assert(reads == kMaxCqPolls && polls == kMaxCqPolls);
  assert(q.cq_consumer_index == 0 && db == 777 && doorbells == 0);

  // Absent/old-owner CQEs never advance or acknowledge the consumer.
  configure(0, 0, cq[0]);
  assert(WaitForCompletion(&entity, &q, &polls) == kCqTimeout);
  assert(q.cq_consumer_index == 0 && db == 777 && doorbells == 0);

  // Status and sub-status errors each consume exactly one terminal CQE.
  for (uint32_t error : {2U << 24U, 3U << 16U,
                         (2U << 24U) | (3U << 16U)}) {
    q.cq_consumer_index = 0; cq[0][0] = 0;
    configure(1, 4U | error, cq[0]);
    assert(WaitForCompletion(&entity, &q, &polls) == kCqError);
    assert(reads == 1 && polls == 1 && q.cq_consumer_index == 1);
    assert(db == 1 && doorbells == 1);
  }

  // After wrapping the CQ ring, owner zero is valid and owner one is old.
  q.cq_consumer_index = 16; cq[0][0] = 4;
  configure(2, 0, cq[0]);
  assert(WaitForCompletion(&entity, &q, &polls) == kSuccess);
  assert(reads == 2 && q.cq_consumer_index == 17 && db == 17);

  // A nonzero CQ slot is selected independently of owner polarity.
  q.cq_consumer_index = 5; cq[5][0] = 0;
  configure(2, 4, cq[5]);
  assert(WaitForCompletion(&entity, &q, &polls) == kSuccess);
  assert(reads == 2 && q.cq_consumer_index == 6 && db == 6);

  // Consumer state keeps the full index; its doorbell uses the protocol mask.
  q.cq_consumer_index = kCqConsumerMask;
  cq[15][0] = 4; configure(1, 0, cq[15]);
  assert(WaitForCompletion(&entity, &q, &polls) == kSuccess);
  assert(q.cq_consumer_index == uint64_t(kCqConsumerMask) + 1);
  assert(db == 0 && doorbells == 1);

  // Each posted Jetty needs its own CQ fence. Completing one cannot satisfy
  // another; these are the 160/147 request counts of BS24-miss15 legacy Get.
  KvafBatchGetDirectUrmaEntity other = entity;
  other.cq_va = reinterpret_cast<uint64_t>(other_cq);
  other.cq_db_va = reinterpret_cast<uint64_t>(&other_db);
  KvafBatchGetQueueState other_q{};
  other_q.sq_producer_index = 147;
  q.cq_consumer_index = 0; cq[0][0] = 0;
  configure(1, 4, cq[0]);
  assert(WaitForCompletion(&entity, &q, &polls) == kSuccess);
  configure(0, 0, other_cq[0]);
  assert(WaitForCompletion(&other, &other_q, &polls) == kCqTimeout);
  assert(q.cq_consumer_index == 1 && other_q.cq_consumer_index == 0);
  assert(q.sq_producer_index == 160 && other_q.sq_producer_index == 147);
  assert(db == 1 && other_db == 456);
  assert(q.last_error == 0 && other_q.last_error == 0);
  static_assert(sizeof(KvafBatchGetQueueState) == 64);
  static_assert(sizeof(KvafBatchGetDirectUrmaEntity) == 192);
  static_assert(sizeof(KvafBatchGetCoreStatus) == 384);
  static_assert(sizeof(KvafExecuteBatchGetKernelArgs) == 112);
  std::puts("SHARED_GET_CQ_CONTROL_FLOW_PASS ABI=64/192/384/112");
}
'''
    source = tmp_path / "get_cq_wait.cc"
    source.write_text(code)
    executable = tmp_path / "get_cq_wait"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-fsanitize=address,undefined",
         "-fno-omit-frame-pointer", "-I", str(DEMO / "include"),
         str(source), "-o", str(executable)],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=20,
    )
    assert result.stdout == "SHARED_GET_CQ_CONTROL_FLOW_PASS ABI=64/192/384/112\n"
    assert result.stderr == ""
