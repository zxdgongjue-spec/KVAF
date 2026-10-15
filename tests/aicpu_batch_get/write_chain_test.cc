#include "pipeline.h"
#include "cpu_urma/write_chain_schedule.h"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {
using namespace kvaf::aicpu_test;
void Check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void ProgressAndLayout() {
  Config c; c.rows = 4096; c.row_bytes = 656; c.scatter_threads = 4;
  c.pipeline_chunk_rows = 256; c.pipeline_jettys = 4;
  Layout l{}; Check(MakeLayout(c, &l), "chain layout rejected");
  Check(ChainReadyOffset(c, l, 0) >= RegisteredOutputOffset(c, l) + l.output_bytes + kGuardBytes,
        "progress markers overlap output/guards");
  for (std::uint32_t w = 0; w < 4; ++w)
    for (std::uint32_t q = 0; q < 4; ++q)
      Check(ChainReadyOffset(c, l, w, q) % 8 == 0 && ChainReadyOffset(c, l, w, q) + 8 <= l.region_bytes,
            "marker alignment/bounds");
  std::array<std::uint64_t, 4> tokens{};
  auto read = [&](std::uint32_t q) { return tokens[q]; };
  for (std::uint32_t q = 0; q < 3; ++q) tokens[q] = PipelineProgressToken(7, 4);
  Check(ChainReadyProgress(c, 0, 7, read) == 0, "released while one queue has not placed data");
  tokens[3] = PipelineProgressToken(6, 4);
  Check(ChainReadyProgress(c, 0, 7, read) == 0, "reused stale sequence");
  tokens[3] = PipelineProgressToken(7, 1);
  Check(ChainReadyProgress(c, 0, 7, read) == PipelineProgressToken(7, 1), "minimum prefix incorrect");
  tokens[3] = PipelineProgressToken(7, 4);
  Check(ChainReadyProgress(c, 0, 7, read) == PipelineProgressToken(7, 4), "lost cumulative prefix");
  tokens[3] = PipelineProgressToken(8, 1); tokens[0] = 0;
  Check(ChainReadyProgress(c, 0, 7, read) >> 32 == 8, "stale queue hid future sequence");
  tokens.fill(PipelineProgressToken(7, 4)); tokens[2] = PipelineProgressToken(7, 5);
  Check(ChainReadyProgress(c, 0, 7, read) >> 32 == 8, "out-of-range progress accepted");
  tokens[2] = PipelineProgressToken(7, 0);
  Check(ChainReadyProgress(c, 0, 7, read) >> 32 == 8, "zero current progress accepted");
  c.pipeline_jettys = 0; Check(!MakeLayout(c, &l), "zero progress queues accepted");
  c.pipeline_jettys = 17; Check(!MakeLayout(c, &l), "too many progress queues accepted");
  c.pipeline_jettys = 4; c.pipeline_chunk_rows = 257; Check(!MakeLayout(c, &l), "unaligned chunk accepted");

}
void PostingFairnessAndFailure() {
  using namespace kvaf::cpu_urma_detail;
  using Post = std::tuple<std::size_t, std::size_t, std::size_t>;
  std::array<std::size_t, 4> prefix{21, 22, 23, 24}, count{84, 88, 92, 96};
  std::vector<Post> posted;
  auto p = [&](std::size_t q) { return prefix[q]; };
  auto n = [&](std::size_t q) { return count[q]; };
  Check(DispatchWriteChain(4, p, n, [&](auto q, auto begin, auto end) {
    ForEachChainPost(begin, end, 32, [&](auto a, auto b) { posted.emplace_back(q, a, b); });
    return true;
  }), "chain post failed");
  for (std::size_t q = 0; q < 4; ++q) {
    Check(posted[q] == Post(q, 0, prefix[q]), "suffix posted before all first prefixes");
    std::size_t next = 0;
    for (auto [queue, begin, end] : posted) if (queue == q) {
      Check(begin == next && end > begin && end - begin <= 32, "post duplicated/skipped WR or doorbell bound");
      next = end;
    }
    Check(next == count[q], "queue tail never posted");
  }
  posted.clear();
  Check(!DispatchWriteChain(4, p, n, [&](auto q, auto begin, auto end) {
    posted.emplace_back(q, begin, end); return q != 1;
  }) && posted.size() == 2, "partial post failure allowed more descriptors to escape");
}
}
int main() {
  try { ProgressAndLayout(); PostingFairnessAndFailure(); std::puts("write chain: PASS"); return 0; }
  catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
