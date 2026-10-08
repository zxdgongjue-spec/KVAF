#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#if defined(__aarch64__)
#if !defined(__ARM_NEON)
#error "The 656-byte AICPU copy requires AArch64 Advanced SIMD"
#endif
#include <arm_neon.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define KVAF_AICPU_COPY_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define KVAF_AICPU_COPY_INLINE __forceinline
#else
#define KVAF_AICPU_COPY_INLINE inline
#endif

namespace kvaf::aicpu_test {
constexpr std::uint64_t kVectorCopyRowBytes = 656;

// Describes the device path. Non-AArch64 unit tests emulate its 16-byte loads
// and stores, while the AICPU kernel is always built for AArch64 with NEON.
inline const char* ScatterCopyBackendName(std::uint64_t row_bytes) {
  return row_bytes == kVectorCopyRowBytes ? "neon_656" : "libc";
}

namespace scatter_copy_detail {
#if defined(__aarch64__)
using Vector16 = uint8x16_t;
KVAF_AICPU_COPY_INLINE Vector16 Load16(const std::uint8_t* src) { return vld1q_u8(src); }
KVAF_AICPU_COPY_INLINE void Store16(std::uint8_t* dst, Vector16 value) { vst1q_u8(dst, value); }
#else
struct Vector16 { std::uint8_t bytes[16]; };
KVAF_AICPU_COPY_INLINE Vector16 Load16(const std::uint8_t* src) {
  Vector16 value; std::memcpy(value.bytes, src, 16); return value;
}
KVAF_AICPU_COPY_INLINE void Store16(std::uint8_t* dst, Vector16 value) {
  std::memcpy(dst, value.bytes, 16);
}
#endif

template <std::size_t Offset>
KVAF_AICPU_COPY_INLINE void Copy64(std::uint8_t* dst, const std::uint8_t* src) {
  const auto a = Load16(src + Offset);
  const auto b = Load16(src + Offset + 16);
  const auto c = Load16(src + Offset + 32);
  const auto d = Load16(src + Offset + 48);
  Store16(dst + Offset, a);
  Store16(dst + Offset + 16, b);
  Store16(dst + Offset + 32, c);
  Store16(dst + Offset + 48, d);
}

template <std::size_t... Group>
KVAF_AICPU_COPY_INLINE void Copy656(std::uint8_t* dst, const std::uint8_t* src,
                                   std::index_sequence<Group...>) {
  // Ten fully expanded 64-byte groups plus an exact 16-byte tail. No load or
  // store crosses the row boundary, even for the last row in the allocation.
  (Copy64<Group * 64>(dst, src), ...);
  Store16(dst + 640, Load16(src + 640));
}
}  // namespace scatter_copy_detail

KVAF_AICPU_COPY_INLINE void CopyRow656(std::uint8_t* dst, const std::uint8_t* src) {
  scatter_copy_detail::Copy656(dst, src, std::make_index_sequence<10>{});
}

// Source rows are packed; output rows retain their stride and guard bytes.
// Dispatch once per worker, with the 656-byte body inlined into its row loop.
inline void CopyScatterRows(std::uint8_t* dst, const std::uint8_t* src,
                            std::uint64_t rows, std::uint64_t row_bytes,
                            std::uint64_t dst_stride) {
  if (row_bytes == kVectorCopyRowBytes) {
    for (std::uint64_t row = 0; row < rows; ++row) {
      CopyRow656(dst, src);
      src += kVectorCopyRowBytes; dst += dst_stride;
    }
  } else {
    for (std::uint64_t row = 0; row < rows; ++row) {
      std::memcpy(dst, src, row_bytes);
      src += row_bytes; dst += dst_stride;
    }
  }
}
}  // namespace kvaf::aicpu_test

#undef KVAF_AICPU_COPY_INLINE
