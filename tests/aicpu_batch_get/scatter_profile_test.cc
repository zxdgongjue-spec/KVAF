#include "scatter_profile_report.h"

#include <cerrno>
#include <cstring>
#include <memory>

namespace {
using namespace kvaf::aicpu_test;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class F> void Reject(F work) {
  bool rejected = false;
  try { work(); } catch (const std::runtime_error&) { rejected = true; }
  Check(rejected, "invalid profile accepted");
}

void TimingAndReporting() {
  Config c; c.rows = 8; c.row_bytes = 64; c.scatter_threads = 2; c.iterations = 2;
  auto p = std::make_unique<ScatterProfile>();
  p->revision = kKernelRevision; p->workers = 2;
  for (unsigned i = 0; i < 2; ++i) {
    const auto range = ScatterRange(c, i, 2);
    p->worker[i].row_begin = range.begin; p->worker[i].row_end = range.end;
    p->worker[i].affinity_capacity = kProfileCpuLimit;
    p->worker[i].allowed_cpus[0] = (1ULL << 3) | (1ULL << 7);
  }
  // Two overlapping workers, repeated with a different absolute epoch. Wall
  // work envelope is 35 ticks per round; summed worker time is 60, not 35.
  for (auto base : {100ULL, 10000ULL}) {
    RecordWorkerTiming(p->worker[0], base, base + 10, base + 20, base + 35, base + 40);
    RecordWorkerTiming(p->worker[1], base, base + 15, base + 25, base + 40, base + 45);
    for (unsigned i = 0; i < 2; ++i) {
      RecordCpuSample(p->worker[i], i ? 7 : 3, 0);
      RecordCpuSample(p->worker[i], i ? 7 : 3, 0);
      ++p->worker[i].cpu_time_rounds;
      p->worker[i].thread_cpu_ns += 30000;
    }
    RecordPoolTiming(*p, base, base + 50);
  }
  ValidateScatterProfile(c, *p);
  Check(p->pool_ticks == 100 && p->dispatch_ticks == 20 && p->envelope_ticks == 70 &&
        p->join_ticks == 10, "parallel wall timeline mismatch");
  Check(p->worker[0].invalidate_ticks == 20 && p->worker[0].memcpy_ticks == 30 &&
        p->worker[0].clean_ticks == 10, "phase totals mismatch");
  Check(ProfileCpuList(p->worker[0], true) == "3,7" &&
        ProfileCpuList(p->worker[1], false) == "7:4", "CPU distribution mismatch");
  std::FILE* file = std::tmpfile();
  Check(file != nullptr, "tmpfile failed");
  PrintScatterProfile(file, c, 1000000, *p);
  std::rewind(file);
  char output[8192]{};
  const auto bytes = std::fread(output, 1, sizeof(output) - 1, file);
  std::fclose(file);
  Check(bytes > 0 && std::strstr(output, "avg_pool_us=50.000") &&
        std::strstr(output, "avg_invalidate_us=10.000 avg_memcpy_us=15.000 avg_clean_us=5.000") &&
        std::strstr(output, "cpu_parallelism=1.200000"), "reported units/parallelism mismatch");
  ++p->worker[0].rounds;
  Reject([&] { ValidateScatterProfile(c, *p); }); // catches warmup leakage
  --p->worker[0].rounds;
  ++p->revision;
  Reject([&] { ValidateScatterProfile(c, *p); });
}

void CpuErrorsAndSingleWorker() {
  Config c; c.rows = 4; c.row_bytes = 656; c.iterations = 1;
  auto p = std::make_unique<ScatterProfile>();
  p->revision = kKernelRevision; p->workers = 1;
  auto& w = p->worker[0];
  w.row_end = 4; w.affinity_capacity = kProfileCpuLimit; w.affinity_errno = EINVAL;
  w.cpu_time_errors = 1; w.cpu_time_errno = ENOSYS;
  RecordCpuSample(w, -1, ENOSYS);
  RecordCpuSample(w, kProfileCpuLimit + 3, 0);
  RecordWorkerTiming(w, 100, 105, 115, 140, 150);
  RecordPoolTiming(*p, 100, 160);
  ValidateScatterProfile(c, *p);
  Check(w.cpu_sample_errors == 1 && w.cpu_sample_errno == ENOSYS && w.cpu_out_of_range == 1 &&
        w.first_cpu == kProfileCpuLimit + 3 && w.last_cpu == w.first_cpu,
        "CPU failures/out-of-range samples silently lost");
  Check(ProfileCpuList(w, true) == "unavailable" && ProfileCpuList(w, false) == "none",
        "unavailable affinity was reported as a valid mask");
  Check(p->pool_ticks == 60 && p->dispatch_ticks == 5 && p->envelope_ticks == 45 && p->join_ticks == 10,
        "single worker wall timeline mismatch");
  std::FILE* file = std::tmpfile();
  Check(file != nullptr, "tmpfile failed");
  PrintScatterProfile(file, c, 1000000, *p);
  std::rewind(file);
  char output[4096]{};
  std::fread(output, 1, sizeof(output) - 1, file);
  std::fclose(file);
  Check(std::strstr(output, "cpu_parallelism=unavailable") &&
        std::strstr(output, "avg_thread_cpu_us=unavailable"), "failed CPU clock treated as zero CPU use");
  ++p->join_ticks;
  Reject([&] { ValidateScatterProfile(c, *p); });
}
}  // namespace
int main() {
  try {
    TimingAndReporting(); CpuErrorsAndSingleWorker();
    std::puts("aicpu scatter profile: PASS"); return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what()); return 1;
  }
}
