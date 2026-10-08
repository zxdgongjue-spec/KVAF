#include "abi.h"
#include "scatter_copy.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
using namespace kvaf::aicpu_test;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void AlignmentAndExactTail() {
  for (unsigned src_offset = 0; src_offset < 64; ++src_offset)
    for (unsigned dst_offset = 0; dst_offset < 64; ++dst_offset) {
      std::vector<std::uint8_t> src(656 + 128), dst(656 + 128, 0xA5);
      for (std::size_t b = 0; b < src.size(); ++b)
        src[b] = static_cast<std::uint8_t>(b * 17 + src_offset * 131 + dst_offset);
      const auto source_before = src;
      auto expected = dst;
      std::memcpy(expected.data() + dst_offset, src.data() + src_offset, 656);
      CopyRow656(dst.data() + dst_offset, src.data() + src_offset);
      Check(dst == expected, "656-byte copy changed payload or surrounding guard");
      Check(src == source_before, "656-byte copy modified source");
    }
}

void RowsAndFallback() {
  for (auto bytes : {8, 65, 656, 657, 4096}) for (auto rows : {1, 3, 67}) {
    Config c; c.rows = rows; c.row_bytes = bytes;
    Layout l{}; Check(MakeLayout(c, &l), "invalid copy test layout");
    std::vector<std::uint8_t> src(19 + l.payload_bytes), dst(37 + l.output_bytes, 0xA5);
    for (unsigned round = 0; round < 4; ++round) {
      for (std::size_t b = 0; b < src.size(); ++b)
        src[b] = static_cast<std::uint8_t>(b * 17 + (b >> 8) + round * 61);
      const auto source_before = src;
      auto expected = dst;
      for (std::uint64_t row = 0; row < c.rows; ++row)
        std::memcpy(expected.data() + 37 + row * l.stride,
                    src.data() + 19 + row * c.row_bytes, c.row_bytes);
      CopyScatterRows(dst.data() + 37, src.data() + 19, c.rows, c.row_bytes, l.stride);
      Check(dst == expected, "row copy/fallback stride or guard mismatch");
      Check(src == source_before, "row copy/fallback modified source");
    }
    Check(std::strcmp(ScatterCopyBackendName(c.row_bytes),
                      c.row_bytes == 656 ? "neon_656" : "libc") == 0,
          "copy backend label mismatch");
  }
  CopyScatterRows(nullptr, nullptr, 0, 656, 768);
  CopyScatterRows(nullptr, nullptr, 0, 65, 192);
}

#if defined(__unix__) || defined(__APPLE__)
struct GuardedTail {
  std::uint8_t* mapping;
  std::size_t page;
  GuardedTail() : mapping(nullptr), page(0) {
    const auto size = sysconf(_SC_PAGESIZE);
    Check(size >= 656, "invalid guard page size");
    page = static_cast<std::size_t>(size);
    void* memory = mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Check(memory != MAP_FAILED, "guard page allocation failed");
    mapping = static_cast<std::uint8_t*>(memory);
    if (mprotect(mapping + page, page, PROT_NONE) != 0) {
      munmap(mapping, 2 * page);
      throw std::runtime_error("guard page protection failed");
    }
  }
  ~GuardedTail() { munmap(mapping, 2 * page); }
  std::uint8_t* data() const { return mapping + page - 656; }
};
void ProtectedTail() {
  GuardedTail src, dst;
  for (unsigned b = 0; b < 656; ++b) src.data()[b] = static_cast<std::uint8_t>(b * 17);
  // Any source overread or destination overstore hits the protected next page.
  CopyScatterRows(dst.data(), src.data(), 1, 656, 768);
  Check(std::memcmp(dst.data(), src.data(), 656) == 0, "protected tail copy mismatch");
}
#endif
}  // namespace

int main() {
  try {
    AlignmentAndExactTail(); RowsAndFallback();
#if defined(__unix__) || defined(__APPLE__)
    ProtectedTail();
#endif
#if defined(__aarch64__)
    std::puts("aicpu scatter copy: PASS (native NEON)");
#else
    std::puts("aicpu scatter copy: PASS (16-byte host emulation; NEON needs AArch64)");
#endif
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what()); return 1;
  }
}
