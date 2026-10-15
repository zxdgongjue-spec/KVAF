#include "scatter_copy.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
using namespace kvaf::aicpu_test;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void ExactRowsAndFailures() {
  for (std::size_t bytes : {8U, 65U, 656U, 657U, 4096U}) {
    constexpr std::size_t rows = 4;
    const auto stride = ((bytes + 63) & ~std::size_t{63}) + 64;
    std::vector<std::uint8_t> source(19 + rows * bytes), output(37 + rows * stride, 0xa5);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<std::uint8_t>(i * 17 + 31);
    const auto source_before = source;
    for (std::size_t fail : {0U, 1U, 3U}) {
      std::fill(output.begin(), output.end(), 0xa5);
      auto expected = output;
      const auto successful = fail ? fail - 1 : rows;
      for (std::size_t row = 0; row < successful; ++row)
        std::memcpy(expected.data() + 37 + row * stride, source.data() + 19 + row * bytes, bytes);
      kvaf_securec_stub::calls.clear(); kvaf_securec_stub::fail_call = fail;
      Check(CopyScatterRows(output.data() + 37, source.data() + 19, rows, bytes, stride) == (fail == 0),
            "securec failure was not propagated");
      Check(output == expected && source == source_before, "securec changed payload, stride guard or source");
      Check(kvaf_securec_stub::calls.size() == (fail ? fail : rows), "securec was skipped or continued after failure");
      for (std::size_t row = 0; row < kvaf_securec_stub::calls.size(); ++row) {
        const auto& call = kvaf_securec_stub::calls[row];
        Check(call.destination == output.data() + 37 + row * stride &&
              call.source == source.data() + 19 + row * bytes &&
              call.capacity == bytes && call.bytes == bytes, "securec row byte count or address mismatch");
      }
    }

  }
  kvaf_securec_stub::calls.clear();
  Check(CopyScatterRows(nullptr, nullptr, 0, 656, 768) &&
        kvaf_securec_stub::calls.empty(), "empty securec scatter called the library");
}
}
int main() {
  try { ExactRowsAndFailures(); std::puts("securec scatter wrapper: PASS (host stub)"); return 0; }
  catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
