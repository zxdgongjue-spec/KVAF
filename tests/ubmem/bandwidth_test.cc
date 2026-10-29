#include "bandwidth_utils.h"
#include "ubmem/test_pattern.h"
#include "kernel_operator.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define __attribute__(...)
#include "../../access_operator/src/ubmem/batch_kernel.asc"
#undef __attribute__
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

using namespace kvaf::ubmem;
std::uint64_t Address(const void* p) { return reinterpret_cast<std::uintptr_t>(p); }

void CheckRawKernel(void (*kernel)(RawCopyArgs), unsigned rows, unsigned cores, bool random) {
  constexpr unsigned bytes = 656, tokens = 8192;
  std::vector<std::uint8_t> remote(tokens * bytes), local(rows * bytes + 32, 0xD3);
  for (std::size_t i = 0; i < remote.size(); ++i) remote[i] = TestPattern(i, 19);
  std::mt19937_64 generator(20260903);
  std::vector<RawCopyRequest> requests(rows);
  auto offsets = perf::MakeRequests(requests, Address(remote.data()), Address(local.data() + 16),
      tokens, bytes, random, 1, generator);
  std::vector<RawCopyReport> reports(cores);
  RawCopyArgs args{Address(requests.data()), Address(reports.data()), rows, bytes, cores};
  for (model_core = 0; model_core < cores; ++model_core) {
    reports[model_core].cycles = reports[model_core].completed = UINT64_MAX;
    kernel(args);
    CHECK(!model_simt && !model_simt_ready && !model_mte2 && !model_mte3);
    CHECK(reports[model_core].completed == rows / cores + (model_core < rows % cores));
    CHECK(reports[model_core].cycles == 1);  // Model tick, never a performance estimate.
  }
  for (std::size_t row = 0; row < rows; ++row) {
    CHECK(offsets[row] % bytes == 0 && offsets[row] <= remote.size() - bytes);
    CHECK(requests[row].source == Address(remote.data()) + offsets[row]);
    CHECK(requests[row].destination == Address(local.data() + 16) + row * bytes);
    for (std::size_t b = 0; b < bytes; ++b)
      CHECK(local[16 + row * bytes + b] == TestPattern(offsets[row] + b, 19));
  }
  for (unsigned i = 0; i < 16; ++i) {
    CHECK(local[i] == 0xD3);
    CHECK(local[local.size() - 1 - i] == 0xD3);
  }
  auto next = perf::MakeRequests(requests, Address(remote.data()), Address(local.data() + 16),
      tokens, bytes, random, 2, generator);
  CHECK(next != offsets);
}

int main() {
  for (auto kernel : {KvafUbmemRawMteKernel, KvafUbmemRawSimtKernel}) {
    CheckRawKernel(kernel, 4096, 3, true);  // Uneven core assignment, random source addresses.
    CheckRawKernel(kernel, 2, 4, false);   // Idle cores still produce reports.
  }
  std::mt19937_64 generator(0);
  std::vector<RawCopyRequest> requests(3);
  const auto offsets = perf::MakeRequests(requests, 0x124000000000ULL, 0x120000000000ULL,
      4, 656, false, 1, generator);
  CHECK((offsets == std::vector<std::uint64_t>{1968, 0, 656}));
  CHECK(std::abs(perf::GBps(4096ULL * 656, 100.0) - 26.86976) < 1e-9);
  CHECK(perf::Percentile({10, 20, 30, 40}, 0.5) == 25);
  CHECK(perf::Percentile({10, 20, 30, 40}, 0.9) == 37);
  CHECK(perf::Percentile({10}, 0.99) == 10);
  std::puts("UBmem raw MTE/SIMT GET: 4096x656 bytes, direct address plans, guards, completion and bandwidth units passed");
}
