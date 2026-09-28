#include "core_probe_report.h"

#include <cstdio>
#include <cstring>
#include <memory>

namespace {
using namespace kvaf::aicpu_test;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void ReportingAndValidation() {
  auto result = std::make_unique<CoreProbeResult>();
  result->requested_blocks = 2;
  result->hold_ticks = 20000;
  result->counter_frequency = 1000000;
  result->status = Result::kOk;
  result->claimed = result->completed = 2;
  result->slot[0].slot = 0; result->slot[0].start_tick = 100; result->slot[0].finish_tick = 200;
  result->slot[0].first_cpu = result->slot[0].last_cpu = 3;
  result->slot[0].allowed_cpus[0] = (1ULL << 3) | (1ULL << 7);
  result->slot[1].slot = 1; result->slot[1].start_tick = 110; result->slot[1].finish_tick = 210;
  result->slot[1].first_cpu = result->slot[1].last_cpu = 7;
  result->slot[1].allowed_cpus[0] = (1ULL << 3) | (1ULL << 7);
  Check(CoreProbeDistinctCpus(*result) == 2 && CoreProbeMaxOverlap(*result) == 2,
        "core probe lost concurrent cores");
  std::FILE* file = std::tmpfile();
  Check(file != nullptr, "tmpfile failed");
  PrintCoreProbe(file, *result, 2);
  std::rewind(file);
  char text[2048]{};
  std::fread(text, 1, sizeof(text) - 1, file);
  std::fclose(file);
  Check(std::strstr(text, "multicore_ready=1") && std::strstr(text, "allowed_cpus=3,7"),
        "core probe report is incomplete");
  result->revision = 99;
  bool rejected = false;
  try { PrintCoreProbe(stderr, *result, 2); } catch (const std::runtime_error&) { rejected = true; }
  Check(rejected, "invalid core probe accepted");
}
}  // namespace

int main() {
  try {
    ReportingAndValidation();
    std::puts("aicpu core probe: PASS");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
