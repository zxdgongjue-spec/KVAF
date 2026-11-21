#include "cpu_pull_profile.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

void Require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
bool Near(double a, double b) { return std::abs(a-b) < 1e-9; }
std::string Contents(FILE* file) {
  std::fflush(file);
  std::rewind(file);
  std::string result;
  char data[1024];
  while (const auto bytes = std::fread(data, 1, sizeof(data), file)) result.append(data, bytes);
  return result;
}
int main() { try {
  using namespace kvaf::offload;
  const auto stats = SummarizeTimings({100, 0, 20, 10});
  Require(Near(stats.mean, 32.5) && Near(stats.p50, 15) && Near(stats.p95, 88) &&
          Near(stats.p99, 97.6) && stats.min == 0 && stats.max == 100, "interpolated percentiles");
  const auto one = SummarizeTimings({7});
  Require(one.mean == 7 && one.p50 == 7 && one.p99 == 7, "single sample quantiles");
  FILE* file = std::tmpfile(); Require(file, "temporary summary output");
  CpuPullProfileSummary profile;
  profile.Reset(2);
  profile.Record(true, 1, 2, 3, 10);
  profile.Record(false, 1000, 2000, 3000, 10000); // Failed warmup is not a timing sample.
  profile.Record(true, 10, 20, 30, 100);
  profile.Record(false, 1000, 2000, 3000, 10000);
  profile.Record(true, 20, 40, 60, 200);
  profile.Report(file, 4, 0x1000, 0x2000, "source_release");
  const auto output = Contents(file);
  Require(output.find("attempts=5 samples=3 failed=2") != std::string::npos, "all scope counts");
  Require(output.find("attempts=3 samples=2 failed=1") != std::string::npos, "warmup excluded by batch ordinal");
  Require(output.find("scope=measured stage=total samples=2 mean_us=150.000 p50_us=150.000 p95_us=195.000 p99_us=199.000") != std::string::npos,
          "failed batches excluded from distributions");
  Require(output.find("scope=measured stage=other samples=2 mean_us=60.000") != std::string::npos,
          "unattributed time is total minus measured phases");
  profile.Report(file, 4, 0x1000, 0x2000, "source_release");
  Require(Contents(file) == output, "close retries do not duplicate summaries");
  std::fclose(file);
  file = std::tmpfile(); Require(file, "temporary empty summary output");
  profile.Reset(10); profile.Record(true, 1, 2, 3, 10);
  profile.Report(file, 5, 0, 0, "source_release");
  const auto empty = Contents(file);
  Require(empty.find("attempts=0 samples=0 failed=0") != std::string::npos &&
          empty.find("scope=measured stage=") == std::string::npos, "no fabricated timings without measured samples");
  std::fclose(file);
  std::cout << "CPU Pull timing summaries passed\n";
  return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; } }
