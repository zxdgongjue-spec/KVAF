#include "metrics.h"
#include "pipeline.h"
#include "scatter_copy.h"
#include "output_readback.h"
#include "cpu_urma/indexed_async_write_tracker.h"
#include "cpu_urma/contiguous_write_spans.h"
#include "cpu_urma/write_chain_schedule.h"
#include <array>
#include <cstdio>
#include <deque>

namespace {
using namespace kvaf::aicpu_test;
using namespace kvaf::cpu_urma_detail;
using Tracker = IndexedAsyncWriteTracker;
void Bounds() {
  Config c; Layout l{};
  Require(MakeLayout(c, &l), "default layout invalid");
  c.rows = UINT64_MAX; Require(!MakeLayout(c, &l), "overflow accepted");
  c = {}; c.row_bytes = 7; Require(!MakeLayout(c, &l), "undersized tag accepted");
  c = {}; c.warmup = UINT64_MAX; Require(!MakeLayout(c, &l), "round overflow accepted");
  c = {}; c.iterations = 0; Require(!MakeLayout(c, &l), "zero samples accepted");
  c = {}; c.timeout_ms = 0; Require(!MakeLayout(c, &l), "unbounded wait accepted");
  c = {}; c.magic = 0; Require(!MakeLayout(c, &l), "invalid ABI accepted");
  c = {}; c.pipeline_chunk_rows = 1023; Require(!MakeLayout(c, &l), "unaligned chunk accepted");
  c = {}; c.pipeline_jettys = 17; Require(!MakeLayout(c, &l), "too many progress queues accepted");
  c = {}; c.payload_cache_maintenance = 2; Require(!MakeLayout(c, &l), "invalid cache policy accepted");
  c = {}; c.verify_payload = 1; Require(!MakeLayout(c, &l), "payload check without per-round observer");
}
void DeviceSelection() {
  Require(SelectDevice(4, false, 0) == 4, "CPU device-id was ignored");
  Require(SelectDevice(4, true, 4) == 4, "matching pinned device rejected");
  bool rejected = false;
  try { (void)SelectDevice(4, true, 0); } catch (...) { rejected = true; }
  Require(rejected, "mismatched NPU devices accepted");
  rejected = false;
  try { (void)SelectDevice(UINT32_MAX, false, 0); } catch (...) { rejected = true; }
  Require(rejected, "overflowing device-id accepted");

}
void RangesAndMetrics() {
  for (auto rows : {1, 67, 4096}) for (auto bytes : {8, 65, 656, 4096}) for (auto workers : {1, 4, 32}) {
    Config c; c.rows = rows; c.row_bytes = bytes; c.scatter_threads = workers;
    Layout l{}; Require(MakeLayout(c, &l), "valid layout rejected");
    Require(ChainReadyOffset(c, l, 0) >= RegisteredOutputOffset(c, l) + l.output_bytes + kGuardBytes &&
            l.region_bytes <= kMaxAllocation, "registered ranges overlap/overflow");
    std::vector<unsigned> covered(rows);
    ForEachPipelineChunk(c, [&](auto, auto, RowRange r) {
      Require(r.begin < r.end && r.end <= c.rows && r.begin * bytes % 64 == 0, "cache ownership crossed");
      for (auto row = r.begin; row < r.end; ++row) ++covered[row];
    });
    for (auto n : covered) Require(n == 1, "missing/duplicated row");
  }
  Config c; c.iterations = 2; c.warmup = 0;
  KernelResult r{}; r.status = Result::kOk; r.completed_rounds = 2; r.counter_frequency = 1000000000;
  r.requested_threads = c.scatter_threads; r.active_threads = ActiveThreads(c); r.kernel_revision = kKernelRevision;
  std::vector<Sample> samples{{100, 20, 80, 10, 0}, {200, 30, 170, 10, 40}};
  const auto report = Summarize(c, r, samples);
  Require(report.mean_ns == 150 && report.wait_ns == 25 && report.scatter_ns == 125 &&
          report.prepare_ns == 10 && report.verify_ns == 20, "timing boundaries changed");
  samples[0].scatter_ticks++;
  bool rejected = false; try { (void)Summarize(c, r, samples); } catch (...) { rejected = true; }
  Require(rejected, "inconsistent timing accepted");
}
struct ChunkIO {
  std::uint64_t token{}, ticks{}, cleans{};
  std::vector<RowRange> copied;
  Result Status() { return Result::kPending; }
  std::uint64_t Now() { return ++ticks; }
  std::uint64_t ReadRangeDone(std::uint32_t) { return token; }
  void RangeReady(std::uint32_t, std::uint64_t) {}
  void CopySpan(std::uint32_t, RowRange r, std::uint64_t) { copied.push_back(r); }
  void CleanRange(std::uint32_t, std::uint64_t) { ++cleans; }
  void Relax() {}
};
void ReadyAndPosting() {
  Config c; c.pipeline_chunk_rows = 256;
  std::array<std::uint64_t, 4> tokens{};
  auto read = [&](auto q) { return tokens[q]; };
  tokens.fill(PipelineProgressToken(3, 4)); tokens[3] = 0;
  Require(ChainReadyProgress(c, 0, 3, read) == 0, "one queue released incomplete data");
  tokens[3] = PipelineProgressToken(3, 2);
  Require(ChainReadyProgress(c, 0, 3, read) == tokens[3], "ready prefix minimum lost");
  tokens[3] = PipelineProgressToken(4, 1);
  Require(ChainReadyProgress(c, 0, 3, read) >> 32 == 4, "future marker accepted");
  ChunkIO io; io.token = PipelineProgressToken(3, 4);
  Require(RunPipelineChunks(c, 0, 3, 0, 100, io) == Result::kOk && io.copied.size() == 1 &&
          io.copied[0].begin == 0 && io.copied[0].end == 1024 && io.cleans == 1, "prefix merge lost data");
  io = {}; io.token = PipelineProgressToken(2, 4);
  Require(RunPipelineChunks(c, 0, 3, 0, 3, io) == Result::kTimeout && io.copied.empty(), "stale marker released data");
  std::array<std::size_t, 4> next{}; unsigned posts = 0;
  Require(DispatchWriteChain(4, [](auto) { return 21U; }, [](auto) { return 84U; },
      [&](auto q, auto begin, auto end) {
        if (posts < 4) Require(q == posts && begin == 0 && end == 21, "suffix preceded first prefixes");
        ForEachChainPost(begin, end, 32, [&](auto a, auto b) {
          Require(a == next[q] && b > a && b - a <= 32, "post skipped/duplicated WR"); next[q] = b;
        });
        ++posts; return true;
      }), "chain dispatch failed");
  for (auto n : next) Require(n == 84, "unposted tail");
  unsigned attempted = 0;
  Require(!DispatchWriteChain(4, [](auto) { return 1U; }, [](auto) { return 2U; },
      [&](auto q, auto, auto) { ++attempted; return q != 1; }) && attempted == 2, "post continued after failure");
}
void Completions() {
  Tracker t(2, 4); Tracker::Handle a{}, b{};
  Tracker::Fence af[]{{0, 3, 0}, {1, 4, 0}}, bf{0, 2, 0};
  bool posted = false; std::uint64_t id = 0;
  Require(t.TryBegin(1, a, af, 2, &posted).ok() && posted, "credit reserve failed");
  Require(t.TryBegin(2, b, &bf, 1, &posted).ok() && !posted, "overcommitted SQ");
  Require(!t.TryBegin(1, a, af, 2, &posted).ok(), "duplicate live submit accepted");
  const auto stale = af[0].context;
  Require(!t.Complete(1, stale, &id).ok(), "wrong queue accepted");
  Require(t.Complete(0, stale, &id).ok() && !id && a.token, "partial batch exposed");
  Require(t.TryBegin(2, b, &bf, 1, &posted).ok() && posted, "released credits unavailable");
  Require(!t.Complete(0, stale, &id).ok(), "duplicate CQE accepted");
  Require(t.Complete(1, af[1].context, &id).ok() && id == 1 && !a.token, "last tail not fenced");
  Require(t.Complete(0, bf.context, &id).ok() && id == 2 && t.Empty(), "queue did not drain");
  Require(t.TryBegin(1, a, af, 2, &posted).ok() && posted && !t.Complete(0, stale, &id).ok(), "stale generation accepted");
  // Two plans have opposite queue-tail order; a capacity-one caller must lose no CQEs.
  Tracker burst(2, 4); std::array<Tracker::Handle, 2> handles{};
  std::array<std::deque<std::uint64_t>, 2> queues; std::array<unsigned, 3> hits{};
  for (unsigned i = 0; i < 2; ++i) {
    Tracker::Fence f[]{{0, 1, 0}, {1, 1, 0}};
    Require(burst.TryBegin(i + 1, handles[i], f, 2, &posted).ok() && posted, "burst reserve failed");
    queues[0].push_back(f[0].context); queues[1].push_front(f[1].context);
  }
  for (unsigned pass = 0; !burst.Empty() && pass < 10; ++pass) {
    std::size_t count = 0; std::uint64_t context = 0;
    auto status = SweepAsyncCqs(2, pass % 2, 8, 1, &count,
        [&](auto q) { return burst.Pending(q); },
        [&](auto q, auto limit, auto* n) {
          Require(limit == 1 && !queues[q].empty(), "CQ read exceeds caller capacity");
          *n = 1; context = queues[q].front(); queues[q].pop_front(); return kvaf::Status::OK();
        }, [&](auto q, auto, bool* finished) {
          auto s = burst.Complete(q, context, &id); *finished = id != 0; if (id) ++hits[id]; return s;
        });
    Require(status.ok() && count <= 1, "CQ sweep overflow/failure");
  }
  Require(burst.Empty() && hits[1] == 1 && hits[2] == 1, "lost/duplicated completion");
}
void SourcesAndOutput() {
  std::vector<std::uint8_t> memory(4096); std::vector<kvaf::CpuUrmaTransport::HostSge> merged;
  kvaf::CpuUrmaTransport::HostSge input[]{{memory.data(), 656}, {memory.data() + 656, 656}};
  std::uint64_t offsets[]{0, 656}; std::vector<std::uint64_t> merged_offsets;
  auto merge = [&](std::uint64_t limit) {
    return CoalesceWriteSpans(input, offsets, 2, memory.size(), limit,
        [](auto, auto) { return 1; }, &merged, &merged_offsets);
  };
  Require(merge(65536).ok() && merged.size() == 1 && merged[0].bytes == 1312, "contiguous spans not merged");
  Require(merge(656).ok() && merged.size() == 2, "message boundary crossed");
  input[1].data = memory.data() + 672;
  Require(merge(65536).ok() && merged.size() == 2, "source gap discarded");
  Config c; c.rows = 4; c.verify_payload = c.verify_each_round = 1;
  Layout l{}; Require(MakeLayout(c, &l), "output layout rejected");
  OutputReadbackVerifier verifier(c, l); OutputMismatch mismatch{};
  std::vector<std::uint8_t> source(l.payload_bytes), output(l.output_bytes, 0xA5);
  for (std::uint64_t seq : {1, 2, 257}) {
    for (std::uint64_t slot = 0; slot < c.rows; ++slot) {
      const auto row = LogicalRowForOutputSlot(c, slot), tag = RowTag(seq, row);
      std::memcpy(source.data() + slot * c.row_bytes, &tag, 8);
      for (std::uint64_t b = 8; b < c.row_bytes; ++b) source[slot * c.row_bytes + b] = PayloadPattern(c, seq, row, b);
    }
    Require(CopyScatterRows(output.data(), source.data(), c.rows, c.row_bytes, l.stride) &&
            verifier.Check(seq, output.data(), &mismatch), "payload/guard copy mismatch");
    output[c.row_bytes] ^= 1;
    Require(!verifier.Check(seq, output.data(), &mismatch) && mismatch.offset == c.row_bytes, "guard overwrite accepted");
    output[c.row_bytes] ^= 1;
  }
}
struct AckIO {
  std::uint64_t clock{}, token{};
  std::uint64_t Now() { return ++clock; }
  void PublishOutputReady(std::uint64_t) {}
  Control ReadOutputVerified() { Control c{}; c.sequence = token; return c; }
  void Relax() {}
};
void Acknowledgements() {
  AckIO io; io.token = OutputVerificationToken(3, true);
  Require(VerifyOutputViaUrma(3, 10, io) == Result::kOk, "valid ACK rejected");
  io.token = OutputVerificationToken(3, false);
  Require(VerifyOutputViaUrma(3, 10, io) == Result::kDataMismatch, "failed ACK ignored");
  io.token = OutputVerificationToken(4, true);
  Require(VerifyOutputViaUrma(3, 10, io) == Result::kSequenceError, "future ACK accepted");
  io.token = OutputVerificationToken(2, true);
  Require(VerifyOutputViaUrma(3, 2, io) == Result::kTimeout, "stale ACK accepted");
}
}  // namespace
int main() {
  try {
    Bounds(); DeviceSelection(); RangesAndMetrics(); ReadyAndPosting();
    Completions(); SourcesAndOutput(); Acknowledgements();
    std::puts("aicpu ordered pipeline protocol: PASS"); return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
