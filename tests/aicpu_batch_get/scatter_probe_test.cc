#include "scatter_probe_report.h"

#include <array>
#include <memory>

namespace {
using namespace kvaf::aicpu_test;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class F> void Reject(F work) {
  bool rejected = false;
  try { work(); } catch (const std::runtime_error&) { rejected = true; }
  Check(rejected, "invalid scatter probe result accepted");
}

void Layouts() {
  ScatterProbeConfig c; ScatterProbeLayout l{};
  Check(MakeScatterProbeLayout(c, &l) && l.rows == 4096 && l.stride == 768 &&
      l.source_bytes == 2686976 && l.output_bytes == 3145728, "fixed work layout mismatch");
  c.mode = ScatterProbeMode::kPacked;
  Check(MakeScatterProbeLayout(c, &l) && l.stride == 656 && l.output_bytes == l.source_bytes,
      "packed output changed source payload");
  c.rows_per_worker = 1023;
  Check(!MakeScatterProbeLayout(c, &l), "shared cache-line worker boundary accepted");
  c = {}; c.row_bytes = 655;
  Check(!MakeScatterProbeLayout(c, &l), "inexact read/write vector size accepted");
  c = {}; c.workers = 0;
  Check(!MakeScatterProbeLayout(c, &l), "empty worker team accepted");
  c = {}; c.warmup = kMaxScatterProbeRounds;
  Check(!MakeScatterProbeLayout(c, &l), "unbounded sample rounds accepted");
  c = {}; c.mode = ScatterProbeMode::kCount;
  Check(!MakeScatterProbeLayout(c, &l), "unknown operation accepted");
  c = {}; c.pmu = kScatterProbePmuMask + 1;
  Check(!MakeScatterProbeLayout(c, &l), "invalid PMU switch accepted");
  c = {}; c.workers = 32; c.rows_per_worker = 65536; c.row_bytes = 65536;
  Check(!MakeScatterProbeLayout(c, &l), "oversized device allocation accepted");
  c = {}; c.row_bytes = 640; c.rows_per_worker = 1;
  Check(MakeScatterProbeLayout(c, &l), "aligned one-row worker rejected");
}

void ExactOperations() {
  constexpr std::uint64_t rows = 4, bytes = 656, stride = 768;
  std::vector<std::uint8_t> source(rows * bytes), output(64 + rows * stride + 64, 0xA5);
  for (std::uint64_t i = 0; i < source.size(); ++i) source[i] = Pattern(i / bytes, i % bytes);
  const auto original = source;
  std::array<std::uint8_t, 16> expected{}, checksum{};
  for (std::uint64_t i = 0; i < source.size(); ++i) expected[i % 16] ^= source[i];
  ProbeReadVectors(source.data(), source.size(), checksum.data());
  Check(expected == checksum && source == original, "read checksum/source corruption");
  ProbeWriteRows(output.data() + 64, rows, bytes, stride);
  for (std::uint64_t i = 0; i < output.size(); ++i) {
    const bool payload = i >= 64 && i < 64 + rows * stride && (i - 64) % stride < bytes;
    Check(output[i] == (payload ? 0x3C : 0xA5), "write crossed payload/guard boundary");
  }
  CopyScatterRows(output.data() + 64, source.data(), rows, bytes, stride);
  for (std::uint64_t row = 0; row < rows; ++row)
    Check(std::memcmp(output.data() + 64 + row * stride, source.data() + row * bytes, bytes) == 0,
        "probe copy lost source mapping");
  std::fill(output.begin(), output.end(), 0xA5);
  CopyScatterRows(output.data() + 64, source.data(), rows, bytes, bytes);
  Check(std::memcmp(output.data() + 64, source.data(), source.size()) == 0 &&
      output[63] == 0xA5 && output[64 + source.size()] == 0xA5, "packed copy boundary mismatch");
  Check(ProbeCompute(100, 1) == ProbeCompute(100, 1) && ProbeCompute(100, 1) != ProbeCompute(100, 2),
      "compute control lost seed dependence");
}

ScatterProbeSample Sample(std::uint64_t base, std::uint64_t op_begin, std::uint64_t op_end) {
  ScatterProbeSample s{};
  s.invalidate_begin = base; s.invalidate_issued = base + 2; s.invalidate_end = base + 5;
  s.op_begin = base + op_begin; s.op_end = s.drain_begin = s.drain_end = base + op_end;
  s.clean_begin = base + 60; s.clean_issued = base + 62; s.clean_end = base + 65;
  s.first_cpu = s.last_cpu = 5;
  s.op_cpu_ns = (op_end - op_begin) * 1000;
  return s;
}
void TimelineAndReport() {
  ScatterProbeConfig c; c.workers = 2; c.rows_per_worker = 4; c.iterations = 2;
  auto result = std::make_unique<ScatterProbeResult>();
  result->counter_frequency = 1000000; result->completed_workers = 2;
  std::vector<ScatterProbeSample> s{Sample(100, 10, 20), Sample(100, 15, 25),
                                  Sample(1000, 10, 20), Sample(1000, 15, 25)};
  ValidateScatterProbe(c, *result, s);
  Check(ProbeMaxOverlap(s.data(), 2) == 2, "parallel intervals not recognized");
  std::FILE* file = std::tmpfile(); Check(file != nullptr, "tmpfile failed");
  PrintScatterProbe(file, c, *result, s, 1);
  std::rewind(file);
  char text[8192]{}; std::fread(text, 1, sizeof(text) - 1, file); std::fclose(file);
  Check(std::strstr(text, "avg_op_envelope_us=15.000 avg_sum_worker_op_us=20.000") &&
      std::strstr(text, "avg_max_overlap=2.000 max_overlap=2") &&
      std::strstr(text, "avg_invalidate_us=5.000 avg_invalidate_issue_us=2.000 avg_invalidate_dsb_us=3.000") &&
      std::strstr(text, "avg_op_us=10.000") &&
      std::strstr(text, "op_begin_us=10.000 op_end_us=20.000"), "phase gaps/units included in operation time");
  c.mode = ScatterProbeMode::kSerialCopy;
  Reject([&] { ValidateScatterProbe(c, *result, s); });
  s[1] = Sample(100, 30, 40); s[3] = Sample(1000, 30, 40);
  ValidateScatterProbe(c, *result, s);
  Check(ProbeMaxOverlap(s.data(), 2) == 1, "serialized intervals reported parallel");
  s[0].op_begin = s[0].invalidate_end - 1;
  Reject([&] { ValidateScatterProbe(c, *result, s); });
}
void PmuUnavailable() {
  ScatterProbeConfig c; c.workers = 1; c.rows_per_worker = 4; c.iterations = 1; c.pmu = kScatterProbePmuMask;
  auto result = std::make_unique<ScatterProbeResult>();
  result->counter_frequency = 1000000; result->completed_workers = 1;
  std::vector<ScatterProbeSample> samples{Sample(100, 10, 20)};
  for (auto& error : samples[0].pmu.event_errno) error = EACCES;
  std::FILE* file = std::tmpfile(); Check(file != nullptr, "tmpfile failed");
  PrintScatterProbe(file, c, *result, samples, 0);
  std::rewind(file);
  char text[8192]{}; std::fread(text, 1, sizeof(text) - 1, file); std::fclose(file);
  Check(std::strstr(text, "event=cycles valid_rounds=0 avg_raw=unavailable avg_scaled=unavailable") &&
      std::strstr(text, "event=backend_stalls"), "missing PMU events reported as zero counters");
  ScatterProbePmu disabled;
  ScatterProbePmuSample sample{}; disabled.Open(false); disabled.Begin(sample); disabled.End(sample);
  Check(sample.available_mask == 0 && sample.read_errno == 0, "disabled PMU touched hardware counters");
}
}  // namespace
int main() {
  try {
    Layouts(); ExactOperations(); TimelineAndReport(); PmuUnavailable();
    std::puts("aicpu scatter probe: PASS"); return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what()); return 1;
  }
}
