#include "abi.h"
#include "scatter_cache.h"

#include <cstdio>
#include <set>
#include <stdexcept>

namespace {
using namespace kvaf::aicpu_test;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void PayloadCoverage() {
  for (auto bytes : {8, 63, 64, 65, 656, 4096})
    for (auto rows : {3, 67}) for (auto threads : {1, 2, 4, 8})
      for (auto alignment : {0, 1, 16, 63}) {
        Config c; c.rows = rows; c.row_bytes = bytes; c.scatter_threads = threads;
        Layout l{}; Check(MakeLayout(c, &l), "invalid test layout");
        const std::uintptr_t output = 0x10000 + alignment;
        std::set<std::uintptr_t> all_cleaned;
        for (std::uint32_t worker = 0; worker < ActiveThreads(c); ++worker) {
          const auto range = ScatterRange(c, worker, ActiveThreads(c));
          std::set<std::uintptr_t> dirty;
          // Derive the required lines independently from every written byte.
          for (auto row = range.begin; row < range.end; ++row)
            for (std::uint64_t b = 0; b < c.row_bytes; ++b)
              dirty.insert((output + row * l.stride + b) & ~std::uintptr_t{63});
          unsigned barriers = 0;
          CleanScatterPayload(output + range.begin * l.stride, range.end - range.begin,
                              c.row_bytes, l.stride, [&](std::uintptr_t line) {
            Check(barriers == 0, "completion occurred before all cleans");
            Check(dirty.erase(line) == 1, "cleaned an unwritten or duplicate line");
            Check(all_cleaned.insert(line).second, "workers share an output cache line");
          }, [&] {
            Check(dirty.empty(), "payload line missing at completion");
            ++barriers;
          });
          Check(barriers == 1, "expected exactly one completion per worker");
        }
      }
}

void BenchmarkCountsAndEmptyRange() {
  Config c; c.rows = 4096; c.row_bytes = 656; c.scatter_threads = 4;
  Layout l{}; Check(MakeLayout(c, &l), "invalid benchmark layout");
  std::uint64_t total = 0;
  for (unsigned worker = 0; worker < 4; ++worker) {
    const auto range = ScatterRange(c, worker, 4);
    std::uint64_t lines = 0;
    unsigned barriers = 0;
    CleanScatterPayload(0x10000 + range.begin * l.stride, range.end - range.begin,
                        c.row_bytes, l.stride, [&](std::uintptr_t) { ++lines; },
                        [&] { ++barriers; });
    Check(lines == 11264 && barriers == 1, "4096x656 worker clean count mismatch");
    total += lines;
  }
  Check(total == 45056 && c.rows * l.stride / 64 - total == 4096,
        "expected one skipped guard line per row");
  unsigned barriers = 0;
  CleanScatterPayload(0x10000, 0, 656, 768,
                      [](std::uintptr_t) { throw std::runtime_error("cleaned empty range"); },
                      [&] { ++barriers; });
  Check(barriers == 1, "empty range completion mismatch");
}
}  // namespace
int main() {
  try {
    PayloadCoverage(); BenchmarkCountsAndEmptyRange();
    std::puts("aicpu scatter cache: PASS"); return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what()); return 1;
  }
}
