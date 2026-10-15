#include "metrics.h"
#include "pipeline.h"
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace {
using namespace kvaf::aicpu_test;
void LayoutAndCoverage() {
  for (auto rows : {1, 3, 16, 67, 4096}) for (auto bytes : {8, 65, 656, 4096}) for (auto threads : {1, 4, 32}) {
    Config c; c.rows = rows; c.row_bytes = bytes; c.scatter_threads = threads;
    Layout l{}; Require(MakeLayout(c, &l), "valid layout rejected");
    Require(l.region_bytes <= kMaxAllocation && RegisteredOutputOffset(c, l) >= kStagingOffset + l.payload_bytes,
            "registered ranges overlap/overflow");
    Require(ChainReadyOffset(c, l, 0) >= RegisteredOutputOffset(c, l) + l.output_bytes + kGuardBytes,
            "markers overlap output guards");
    std::vector<unsigned> counts(rows);
    ForEachPipelineChunk(c, [&](auto worker, auto, RowRange r) {
      const auto range = ScatterRange(c, worker, ActiveThreads(c));
      Require(r.begin >= range.begin && r.end <= range.end && r.begin < r.end && r.begin * bytes % 64 == 0,
              "chunk crossed worker/cache ownership");
      for (auto row = r.begin; row < r.end; ++row) ++counts[row];
    });
    for (auto count : counts) Require(count == 1, "missing/duplicated row");
  }
  Config c; Layout l{};
  Require(MakeLayout(c, &l) && c.rows == 4096 && c.row_bytes == 656 && ActiveThreads(c) == 4 &&
      c.pipeline_chunk_rows == 1024 && c.pipeline_jettys == 4 && !c.payload_cache_maintenance,
      "optimized defaults changed");
  c.rows = UINT64_MAX; Require(!MakeLayout(c, &l), "oversized row count accepted");
  c = {}; c.row_bytes = UINT32_MAX; Require(!MakeLayout(c, &l), "allocation overflow accepted");
  c = {}; c.pipeline_chunk_rows = 1023; Require(!MakeLayout(c, &l), "unaligned chunk accepted");
  c = {}; c.verify_payload = 1; Require(!MakeLayout(c, &l), "payload check without per-round observer");
  c.verify_each_round = 1; Require(MakeLayout(c, &l), "payload validation rejected");
  c = {}; c.payload_cache_maintenance = 2; Require(!MakeLayout(c, &l), "invalid cache policy accepted");
  c = {}; Require(!ShouldVerifyOutput(c, 1) && ShouldVerifyOutput(c, c.warmup + c.iterations), "final observer schedule");
}
void Timings() {
  Config c; c.warmup = 0; c.iterations = 2;
  KernelResult result{}; result.status = Result::kOk; result.completed_rounds = 2;
  result.counter_frequency = 1000000000; result.kernel_revision = kKernelRevision;
  result.requested_threads = c.scatter_threads; result.active_threads = ActiveThreads(c);
  std::vector<Sample> samples{{100, 20, 80, 10, 0}, {200, 30, 170, 10, 40}};
  const auto report = Summarize(c, result, samples);
  Require(report.mean_ns == 150 && report.p50_ns == 100 && report.p99_ns == 200 &&
      report.wait_ns == 25 && report.scatter_ns == 125 && report.prepare_ns == 10 && report.verify_ns == 20,
      "timing boundaries/percentiles changed");
  Require(report.gbps == double(c.rows * c.row_bytes) / 150, "bandwidth numerator/units");
  samples[0].scatter_ticks = 79;
  bool rejected = false; try { (void)Summarize(c, result, samples); } catch (...) { rejected = true; }
  Require(rejected, "inconsistent sample accepted");
}
struct ChunkIO {
  std::vector<std::uint64_t> tokens;
  std::vector<RowRange> copied;
  std::size_t reads{}; std::uint64_t clock{}, cleaned{};
  Result Status() { return Result::kPending; }
  std::uint64_t Now() { return ++clock; }
  std::uint64_t ReadRangeDone(std::uint32_t) { const auto i = reads++; return tokens[std::min(i, tokens.size()-1)]; }
  void RangeReady(std::uint32_t, std::uint64_t) {}
  void CopySpan(std::uint32_t, RowRange r, std::uint64_t) { copied.push_back(r); }
  void CleanRange(std::uint32_t, std::uint64_t) { ++cleaned; }
  void Relax() {}
};
void ChunkProtocol() {
  Config c; c.pipeline_chunk_rows = 256;
  ChunkIO merged; merged.tokens = {PipelineProgressToken(3, 4)};
  Require(RunPipelineChunks(c, 0, 3, 0, 100, merged) == Result::kOk && merged.copied.size() == 1 &&
      merged.copied[0].begin == 0 && merged.copied[0].end == 1024 && merged.cleaned == 1, "prefix merge lost data");
  ChunkIO incremental; incremental.tokens = {PipelineProgressToken(3, 1), PipelineProgressToken(3, 4)};
  Require(RunPipelineChunks(c, 0, 3, 0, 100, incremental) == Result::kOk && incremental.copied.size() == 2 &&
      incremental.copied[1].begin == 256 && incremental.cleaned == 1, "incremental copy repeated/skipped data");
  for (auto token : {PipelineProgressToken(4, 1), PipelineProgressToken(3, 0), PipelineProgressToken(3, 5)}) {
    ChunkIO bad; bad.tokens = {token};
    Require(RunPipelineChunks(c, 0, 3, 0, 100, bad) == Result::kSequenceError && bad.copied.empty(), "invalid progress accepted");
  }
  ChunkIO stale; stale.tokens = {PipelineProgressToken(2, 4)};
  Require(RunPipelineChunks(c, 0, 3, 0, 3, stale) == Result::kTimeout && stale.copied.empty(), "stale progress released data");
  ChunkIO regress; regress.tokens = {PipelineProgressToken(3, 2), PipelineProgressToken(3, 1)};
  Require(RunPipelineChunks(c, 0, 3, 0, 100, regress) == Result::kSequenceError && !regress.cleaned, "regressed progress accepted");
}
}
int main() {
  try { LayoutAndCoverage(); Timings(); ChunkProtocol(); std::puts("ordered pipeline protocol: PASS"); return 0; }
  catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
