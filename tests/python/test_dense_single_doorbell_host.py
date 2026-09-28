"""Compile and execute the production dense-submit control flow on the host.

The source under test is extracted from execute_batch_get_aiv.asc, not copied.
Only device operations are replaced by deterministic event/SQ stubs. This
checks submission ordering, ring wrap and the single-CQE contract; it is not
a substitute for CANN instruction timing or an HCCP transfer test.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_dense_single_doorbell_control_flow(tmp_path: Path) -> None:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (
        Path(__file__).resolve().parents[2]
        / "access_operator/demo/aiv_batch_get/kernel/execute_batch_get_aiv.asc"
    ).read_text()
    begin = source.index(
        "__aicore__ inline void PostDenseBatchGetRequestsSingleDoorbell("
    )
    end = source.index("// Preserve the proven one-warp path", begin)
    production = source[begin:end]
    cpp = tmp_path / "dense_submit.cc"
    cpp.write_text(
        r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

#define __aicore__
#define __gm__
constexpr uint32_t kWqeBankCount = 2U;
constexpr uint32_t kWqesPerPost = 32U;
enum Pipe { PIPE_V, PIPE_MTE3, PIPE_S };
using event_t = int;
enum class MemDsbT { DDR };

struct KvafBatchGetDirectUrmaEntity {
  uint64_t local_hbm_address = 0;
  uint64_t sq_depth = 512;
  uint64_t sq_db_va = 0;
};
struct KvafBatchGetQueueState { uint64_t sq_producer_index = 0; };
struct PreparedPost {
  uint32_t event_id = 0, offset = 0, count = 0;
  uint64_t head = 0;
  bool completion_on_last = false;
};
struct Wqe { uint64_t pi = UINT64_MAX; bool signaled = false; };
struct State {
  std::array<PreparedPost, 2> banks{};
  std::array<bool, 2> vector_ready{}, bank_released{};
  std::vector<Wqe> sq;
  uint32_t prepared_rows = 0, copied_rows = 0;
  uint32_t pi_publications = 0, barriers = 0, doorbells = 0;
  bool final_mte3_to_scalar = false;
  explicit State(uint32_t depth) : sq(depth) {}
};
State* state = nullptr;

template<typename T> T ReadGmByPassDCache(const T* ptr) { return *ptr; }
template<typename T> void WriteGmByPassDCache(T* ptr, T value) {
  assert(state->copied_rows == state->prepared_rows);
  assert(state->final_mte3_to_scalar);
  assert(state->pi_publications++ == 0);
  *ptr = value;
}
template<MemDsbT> void DataSyncBarrier() {
  assert(state->pi_publications == 1);
  ++state->barriers;
}
void st_dev(uint32_t value, uint32_t* address, int) {
  assert(state->barriers == 1 && state->doorbells++ == 0);
  *address = value;
}
uint32_t WqeControl(bool completion, uint32_t owner) {
  return (completion ? 0x100U : 0U) | owner;
}
namespace AscendC { uint64_t GetSystemCycle() { return 123U; } }

void set_flag(Pipe from, Pipe to, event_t event) {
  assert(event >= 0 && event < 2);
  if (from == PIPE_MTE3 && to == PIPE_V) {
    state->bank_released[event] = true;
  } else if (from == PIPE_MTE3 && to == PIPE_S) {
    assert(state->copied_rows == state->prepared_rows);
    state->final_mte3_to_scalar = true;
  } else {
    assert(false);
  }
}
void wait_flag(Pipe from, Pipe to, event_t event) {
  assert(event >= 0 && event < 2);
  if (from == PIPE_V && to == PIPE_MTE3) {
    assert(state->vector_ready[event]);
    state->vector_ready[event] = false;
  } else if (from == PIPE_MTE3 && to == PIPE_S) {
    assert(state->final_mte3_to_scalar);
  } else if (from == PIPE_MTE3 && to == PIPE_V) {
    assert(state->bank_released[event]);
    state->bank_released[event] = false;
  } else {
    assert(false);
  }
}
void PrepareDensePostVectorStage(
    const int32_t* indices, const int32_t* output_indices,
    const int32_t* offsets, const uint64_t* address_table,
    uint32_t request_offset, uint32_t block_bytes, uint64_t hbm_address,
    uint64_t current_head, uint32_t wqe_count,
    uint32_t completion_on_last_wqe, uint32_t bank_id,
    bool wait_for_bank_reuse,
    const KvafBatchGetDirectUrmaEntity* entity,
    KvafBatchGetQueueState* queue, PreparedPost* prepared) {
  (void)indices; (void)output_indices; (void)offsets;
  (void)address_table; (void)block_bytes; (void)hbm_address;
  (void)entity; (void)queue;
  assert(bank_id < 2 && wqe_count > 0 && wqe_count <= 32);
  if (wait_for_bank_reuse) {
    wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(bank_id));
  }
  *prepared = {bank_id, request_offset, wqe_count, current_head,
               completion_on_last_wqe != 0};
  state->banks[bank_id] = *prepared;
  state->prepared_rows += wqe_count;
  state->vector_ready[bank_id] = true;
}
void CopyDensePreparedPostToSq(const PreparedPost* prepared) {
  assert(prepared->event_id < 2);
  assert(state->banks[prepared->event_id].head == prepared->head);
  for (uint32_t i = 0; i < prepared->count; ++i) {
    const uint64_t pi = prepared->head + i;
    auto& wqe = state->sq[pi % state->sq.size()];
    wqe = {pi, prepared->completion_on_last && i + 1 == prepared->count};
    ++state->copied_rows;
  }
}
"""
        + production
        + r"""
int main() {
  const int32_t descriptors[256] = {};
  const uint64_t addresses[256] = {};
  for (uint32_t count : {0U, 1U, 31U, 32U, 33U, 63U, 64U,
                         65U, 127U, 128U, 129U, 255U, 256U}) {
    for (uint64_t head : {0ULL, 1ULL, 31ULL, 511ULL, 512ULL, 1023ULL}) {
      State observed(512U); state = &observed;
      uint32_t doorbell_register = UINT32_MAX;
      KvafBatchGetDirectUrmaEntity entity;
      entity.sq_db_va = reinterpret_cast<uint64_t>(&doorbell_register);
      KvafBatchGetQueueState queue{head};
      uint64_t first_doorbell_cycle = 0, last_wqe = 0;
      PostDenseBatchGetRequestsSingleDoorbell(
          descriptors, descriptors, descriptors, addresses, count, 656U,
          &entity, &queue, 1U, &first_doorbell_cycle, &last_wqe);
      if (count == 0U) {
        assert(queue.sq_producer_index == head);
        assert(observed.doorbells == 0 && observed.pi_publications == 0);
        continue;
      }
      assert(observed.prepared_rows == count);
      assert(observed.copied_rows == count);
      assert(observed.pi_publications == 1);
      assert(observed.barriers == 1 && observed.doorbells == 1);
      assert(queue.sq_producer_index == head + count);
      assert(doorbell_register == static_cast<uint32_t>(head + count));
      assert(first_doorbell_cycle == 123U);
      uint32_t completion_count = 0;
      for (uint64_t pi = head; pi < head + count; ++pi) {
        const auto& wqe = observed.sq[pi % entity.sq_depth];
        assert(wqe.pi == pi);
        completion_count += wqe.signaled;
        assert(wqe.signaled == (pi + 1U == head + count));
      }
      assert(completion_count == 1U);
      const uint64_t last_pi = head + count - 1U;
      const uint32_t owner =
          1U - static_cast<uint32_t>((last_pi / entity.sq_depth) & 1U);
      assert(last_wqe == WqeControl(true, owner));
    }
  }
}
"""
    )
    executable = tmp_path / "dense_submit"
    subprocess.run(
        [compiler, "-std=c++17", "-O1", "-fsanitize=address,undefined",
         str(cpp), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
