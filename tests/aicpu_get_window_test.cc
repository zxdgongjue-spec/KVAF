#include "gather_workspace.h"
#include "cpu_push_get_layout.h"
#include <cstdlib>
#include <iostream>

#define CHECK(value) do { if (!(value)) { std::cerr << "line " << __LINE__ << ": " #value "\n"; std::abort(); } } while (false)
using namespace kvaf;

int main() {
  CHECK(kGatherDefaultReadWaveWrs == 4096);
  for (uint32_t limit : {1U, 31U, 32U, 64U, 96U, 128U, 256U, 512U, 4096U})
    CHECK(ValidGatherReadWaveWrs(limit));
  CHECK(!ValidGatherReadWaveWrs(0) && !ValidGatherReadWaveWrs(4097) &&
        !ValidGatherReadWaveWrs(UINT32_MAX));
  for (uint32_t workers : {1U, 3U, 5U, 8U}) {
    for (uint32_t token : {1U, 7U, 656U, 4096U}) {
      CHECK(GatherWindowCapacity(token, workers, 0) == 65536);
      CHECK(cpu_push::ScatterWindowCapacity(token, workers, 0) == 65536);
      for (uint32_t rows : {4096U, 8192U, 65536U, 69632U, 131072U}) {
        const auto bytes = GatherRemoteWorkspaceBytes(token, workers, rows);
        CHECK(GatherWindowCapacity(token, workers, bytes) == rows);
        CHECK(GatherWindowCapacity(token, workers, bytes - 1U) == rows - 4096);
        cpu_push::Layout layout;
        CHECK(cpu_push::MakeLayout(token, workers, &layout, rows));
        CHECK(cpu_push::ScatterWindowCapacity(token, workers, layout.bytes) == rows);
        CHECK(cpu_push::ScatterWindowCapacity(token, workers, layout.bytes - 1U) == rows - 4096);
      }
      for (uint64_t alignment : {1ULL, 1ULL << 16, 1ULL << 21, 1ULL << 24,
                                 1ULL << 25, 1ULL << 29, 1ULL << 30, 1ULL << 34}) {
        uint64_t budget{};
        CHECK(GetWindowBytes(GatherRemoteWorkspaceBytes(token, workers, 65536), alignment, &budget));
        const auto rows = GatherWindowCapacity(token, workers, budget, alignment);
        CHECK(rows && rows % 4096 == 0 && ValidGatherWindow(token, workers, rows));
        uint64_t used{}, next{};
        CHECK(GetWindowBytes(GatherRemoteWorkspaceBytes(token, workers, rows), alignment, &used));
        CHECK(used <= budget);
        CHECK(GetWindowBytes(GatherRemoteWorkspaceBytes(token, workers, rows + 4096), alignment, &next));
        CHECK(!ValidGatherWindow(token, workers, rows + 4096) || next > budget);
      }
    }
  }
  CHECK(GatherWindowCapacity(656, 5, 1) == 0);
  CHECK(GatherWindowCapacity(656, 5, (1ULL << 30) - 1, 1ULL << 30) == 0);
  CHECK(cpu_push::ScatterWindowCapacity(656, 5, 1) == 0);
  CHECK(GatherWindowCapacity(0, 5, UINT64_MAX) == 0);
  CHECK(GatherWindowCapacity(1, 0, UINT64_MAX) == 0);
  CHECK(cpu_push::ScatterWindowCapacity(UINT32_MAX, 5, UINT64_MAX) == 0);
  CHECK(GatherWindowCapacity(1, 8, UINT64_MAX) == uint32_t(INT32_MAX / 4096 * 4096));
  CHECK(cpu_push::ScatterWindowCapacity(1, 8, UINT64_MAX) == uint32_t(INT32_MAX / 4096 * 4096));
  uint64_t rounded{};
  CHECK(!GetWindowBytes(UINT64_MAX, 4096, &rounded));
  uint32_t previous = 0;
  for (uint32_t worker = 0; worker < 8; ++worker) {
    const auto range = cpu_push::WorkerRange(INT32_MAX, 1, worker, 8);
    CHECK(range.begin == previous && range.end >= range.begin);
    previous = range.end;
  }
  CHECK(previous == INT32_MAX);
  std::cout << "AICPU workspace budgets: exact maximal 4096-row windows, page rounding and overflow passed\n";
}
