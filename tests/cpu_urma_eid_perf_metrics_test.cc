#include "cpu_urma_eid_perf_metrics.h"

#include <cmath>
#include <iostream>

namespace {
void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
bool Near(double a, double b) { return std::abs(a - b) < 1e-9; }
}

int main() {
  try {
    const auto stats = kvaf::eid_perf::Summarize({500000, 1500000}, 1000000);
    Check(Near(stats.mean_ns, 1000000) && Near(stats.p50_ns, 1000000), "mean/median timing");
    Check(Near(stats.p99_ns, 1490000), "interpolated p99");
    Check(Near(stats.mean_gbps, 1) && Near(stats.p50_gbps, 1),
          "bandwidth must be total bytes over total time, in decimal GB/s");
    const auto single = kvaf::eid_perf::Summarize({2000000}, 8388608);
    Check(Near(single.mean_gbps, 4.194304) && Near(single.p99_ns, 2000000), "single sample and MiB payload");
    for (const auto& samples : std::vector<std::vector<std::uint64_t>>{{}, {0}, {100, 0}}) {
      bool rejected = false;
      try { (void)kvaf::eid_perf::Summarize(samples, 1024); }
      catch (const std::invalid_argument&) { rejected = true; }
      Check(rejected, "invalid timing accepted as bandwidth");
    }
    Check(kvaf::eid_perf::CsvCell("urma0") == "urma0", "plain CSV cell");
    Check(kvaf::eid_perf::CsvCell("failed, status=\"x\"\nretry") ==
              "\"failed, status=\"\"x\"\"\nretry\"", "CSV error escaping");
    std::cout << "EID performance metrics tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
