#include "aicpu_urma_client_metrics.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool Near(double actual, double expected) { return std::abs(actual - expected) < 1e-9; }

bool Rejects(const std::vector<double>& samples, std::size_t payload_bytes) {
  try {
    (void)kvaf::aicpu_urma_client::SummarizeTiming(samples, payload_bytes);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  const auto stats = kvaf::aicpu_urma_client::SummarizeTiming({4.0, 1.0, 3.0, 2.0}, 1000000);
  assert(Near(stats.mean_ms, 2.5));
  assert(Near(stats.min_ms, 1.0) && Near(stats.max_ms, 4.0));
  assert(Near(stats.p50_ms, 2.5) && Near(stats.p95_ms, 3.85) && Near(stats.p99_ms, 3.97));
  assert(Near(stats.mean_gbps, 0.4));

  const auto single = kvaf::aicpu_urma_client::SummarizeTiming({4.0}, 8000000);
  assert(Near(single.p50_ms, 4.0) && Near(single.p99_ms, 4.0));
  assert(Near(single.mean_gbps, 2.0));

  assert(Rejects({}, 1));
  assert(Rejects({1.0}, 0));
  assert(Rejects({0.0}, 1));
  assert(Rejects({std::numeric_limits<double>::infinity()}, 1));
  std::cout << "AICPU URMA native client metrics: PASS\n";
}
