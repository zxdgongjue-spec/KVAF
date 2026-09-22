#include "device_loop.h"
#include "metrics.h"

#include <cstdio>
#include <cstring>
#include <functional>

namespace {
using namespace kvaf::aicpu_test;
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
  c = {}; c.row_bytes = 65; Require(!MakeLayout(c, &l), "partial cache line accepted");
  c = {}; c.warmup = UINT64_MAX; Require(!MakeLayout(c, &l), "round overflow accepted");
  c = {}; c.iterations = 0; Require(!MakeLayout(c, &l), "zero samples accepted");
  c = {}; c.timeout_ms = 0; Require(!MakeLayout(c, &l), "unbounded wait accepted");
  c = {}; c.magic = 0; Require(!MakeLayout(c, &l), "invalid ABI accepted");
}
}  // namespace
int main() {
  try { Bounds(); SuccessAndMetrics(); Failures(); std::puts("aicpu get protocol: PASS"); return 0; }
  catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
