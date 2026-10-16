#include "device_loop.h"
#include "metrics.h"
#include "scatter_pool.h"
#include "pipeline.h"
#include "scatter_copy.h"
#include "output_readback.h"
#include "cpu_urma/indexed_async_write_tracker.h"
#include "cpu_urma/contiguous_write_spans.h"
#include "cpu_urma/write_chain_schedule.h"
#include <array>
#include <deque>

#include <cstdio>
#include <cstring>
#include <functional>

namespace {
using namespace kvaf::aicpu_test;
using namespace kvaf::cpu_urma_detail;
using Tracker = IndexedAsyncWriteTracker;
struct Simulator {
  Config config;
  Layout layout{};
  std::vector<std::uint8_t> staging, output;
  std::uint64_t clock{}, sequence{}, done{}, publications{};
  int reads{};
  bool timeout{}, future{}, corrupt{};
  explicit Simulator(Config c) : config(c) {
    Require(MakeLayout(c, &layout), "simulator config");
    staging.resize(layout.payload_bytes); output.assign(layout.output_bytes, 0xA5);
  }
  std::uint64_t Frequency() { return 1000; }
  std::uint64_t Now() { return ++clock; }
  void PublishRequest(std::uint64_t seq) {
    Require(seq == sequence + 1, "sequence must increase after scatter");
    sequence = seq; reads = 0; ++publications;
  }
  std::uint64_t ReadDone() {
    if (future) return sequence + 1;
    if (timeout || ++reads < 3) return done;
    // Model the CPU WRITE and delayed completion, with a new tag every round.
    for (std::uint64_t row = 0; row < config.rows; ++row) {
      auto* dst = staging.data() + row * config.row_bytes;
      auto tag = RowTag(sequence, row);
      if (corrupt && row == 1) --tag;
      std::memcpy(dst, &tag, 8);
      for (std::uint64_t b = 8; b < config.row_bytes; ++b) dst[b] = Pattern(row, b);
    }
    return done = sequence;
  }
  void Scatter(const Config& c, const Layout& l) {
    Require(done == sequence, "scatter ran before completion");
    for (std::uint64_t row = 0; row < c.rows; ++row)
      std::memcpy(output.data() + OutputOffset(c, l, row), staging.data() + row * c.row_bytes, c.row_bytes);
    clock += 5;
  }
  bool VerifyTags(std::uint64_t seq, const Config& c, const Layout& l) {
    for (std::uint64_t row = 0; row < c.rows; ++row) {
      std::uint64_t tag;
      std::memcpy(&tag, output.data() + OutputOffset(c, l, row), 8);
      if (tag != RowTag(seq, row)) return false;
    }
    return true;
  }
};
void SuccessAndMetrics() {
  Config c; c.rows = 3; c.row_bytes = 128; c.warmup = 2; c.iterations = 4;
  Simulator io(c);
  std::vector<Sample> samples(c.iterations); KernelResult result{};
  Require(RunDevice(c, io, samples.data(), &result) == Result::kOk, "device loop failed");
  Require(io.publications == 6 && result.completed_rounds == 6, "warmup/round count mismatch");
  for (std::uint64_t slot = 0; slot < c.rows; ++slot) {
    const auto row = c.rows - 1 - slot;
    const auto* bytes = io.output.data() + slot * io.layout.stride;
    for (std::uint64_t b = 8; b < c.row_bytes; ++b)
      Require(bytes[b] == Pattern(row, b), "incorrect reversed scatter payload");
    for (std::uint64_t b = c.row_bytes; b < io.layout.stride; ++b)
      Require(bytes[b] == 0xA5, "scatter overwrote guard");
  }
  const auto report = Summarize(c, result, samples);
  Require(report.status == 0 && report.mean_ns == report.wait_ns + report.scatter_ns &&
          report.p50_ns > 0 && report.p99_ns >= report.p50_ns && report.gbps > 0, "metrics mismatch");
  samples[0].scatter_ticks++;
  bool rejected = false;
  try { (void)Summarize(c, result, samples); } catch (...) { rejected = true; }
  Require(rejected, "invalid timing accepted");
}
void Failures() {
  Config c; c.rows = 3; c.row_bytes = 64; c.iterations = 1; c.warmup = 0; c.timeout_ms = 5;
  Sample sample{}; KernelResult result{};
  Simulator missing(c); missing.timeout = true;
  Require(RunDevice(c, missing, &sample, &result) == Result::kTimeout &&
          result.completed_rounds == 0 && result.failed_sequence == 1, "missing completion must timeout");
  Simulator future(c); future.future = true;
  Require(RunDevice(c, future, &sample, &result) == Result::kSequenceError, "future sequence accepted");
  Simulator corrupt(c); corrupt.corrupt = true;
  Require(RunDevice(c, corrupt, &sample, &result) == Result::kDataMismatch, "stale row accepted");
}
void Bounds() {
  Config c; Layout l{};
  Require(MakeLayout(c, &l), "default layout invalid");
  c.rows = UINT64_MAX; Require(!MakeLayout(c, &l), "overflow accepted");
  c = {}; c.row_bytes = 7; Require(!MakeLayout(c, &l), "undersized tag accepted");
  c = {}; c.warmup = UINT64_MAX; Require(!MakeLayout(c, &l), "round overflow accepted");
  c = {}; c.iterations = 0; Require(!MakeLayout(c, &l), "zero samples accepted");
  c = {}; c.timeout_ms = 0; Require(!MakeLayout(c, &l), "unbounded wait accepted");
  c = {}; c.magic = 0; Require(!MakeLayout(c, &l), "invalid ABI accepted");
}
void ParallelScatter() {
  for (auto bytes : {8, 65, 656, 4096}) for (auto rows : {3, 67, 4096})
    for (auto threads : {1, 2, 4, 8}) {
      Config c; c.rows = rows; c.row_bytes = bytes; c.scatter_threads = threads;
      Simulator io(c);
      struct Context { Simulator* io; } context{&io};
      ScatterPool pool;
      Require(pool.Initialize(ActiveThreads(c), [](void* opaque, std::uint32_t worker) {
        auto& io = *static_cast<Context*>(opaque)->io;
        auto range = ScatterRange(io.config, worker, ActiveThreads(io.config));
        for (auto row = range.begin; row < range.end; ++row)
          std::memcpy(io.output.data() + OutputOffset(io.config, io.layout, row),
                      io.staging.data() + row * io.config.row_bytes, io.config.row_bytes);
      }, &context) == 0, "pthread initialization failed");
      std::uint64_t previous = 0;
      for (std::uint32_t worker = 0; worker < ActiveThreads(c); ++worker) {
        auto range = ScatterRange(c, worker, ActiveThreads(c));
        Require(range.begin == previous && range.end > range.begin &&
                (range.begin * c.row_bytes) % 64 == 0, "overlapping/unaligned partition");
        previous = range.end;
      }
      Require(previous == c.rows, "missing rows");
      for (unsigned round = 0; round < 4; ++round) {
        for (std::size_t b = 0; b < io.staging.size(); ++b)
          io.staging[b] = static_cast<std::uint8_t>(b * 17 + round);
        pool.Run();
        for (std::uint64_t row = 0; row < c.rows; ++row) {
          const auto* dst = io.output.data() + OutputOffset(c, io.layout, row);
          Require(std::memcmp(dst, io.staging.data() + row * c.row_bytes, c.row_bytes) == 0,
                  "parallel scatter data mismatch");
          for (auto b = c.row_bytes; b < io.layout.stride; ++b)
            Require(dst[b] == 0xA5, "parallel scatter guard overwritten");
        }
      }
    }
  Config c; Layout l{};
  c.rows = 4096; c.row_bytes = 656;
  Require(MakeLayout(c, &l) && l.payload_bytes == 2686976 && l.stride == 768,
          "4096x656 layout mismatch");
  c.scatter_threads = 0; Require(!MakeLayout(c, &l), "zero threads accepted");
  c.scatter_threads = 33; Require(!MakeLayout(c, &l), "too many threads accepted");
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
  try { ParallelScatter(); Bounds(); DeviceSelection(); SuccessAndMetrics(); Failures();
        ReadyAndPosting(); Completions(); SourcesAndOutput(); Acknowledgements();
        std::puts("aicpu get protocol: PASS"); return 0; }
  catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
