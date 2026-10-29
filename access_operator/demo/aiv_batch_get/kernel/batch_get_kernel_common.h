#pragma once
#include "kernel_operator.h"

namespace kvaf_aiv {
using namespace AscendC;
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline void Flush(uint64_t address, uint64_t bytes) {
  if (!bytes) return;
  const auto first = address & ~uint64_t(63);
  const auto last = (address + bytes - 1) & ~uint64_t(63);
  GlobalTensor<uint8_t> buffer;
  buffer.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(first));
  for (uint64_t p = first; p <= last; p += 64)
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(buffer[p - first]);
}
__aicore__ inline uint64_t LoadWord(uint64_t address) {
  Flush(address, 8);
  return *reinterpret_cast<__gm__ volatile uint64_t*>(address);
}
__aicore__ inline void PublishWord(uint64_t address, uint64_t value) {
  *reinterpret_cast<__gm__ uint64_t*>(address) = value;
  Flush(address, 8);
}
__aicore__ inline bool Range(uint64_t base, uint64_t bytes, uint64_t address, uint64_t n) {
  return n && n <= bytes && address >= base && address - base <= bytes - n;
}

template<class Args>
__aicore__ uint64_t Source(const Args& a, uint32_t i) {
  const auto& r = a.request;
  const auto index = reinterpret_cast<__gm__ const int32_t*>(r.indices)[i];
  const auto offset = reinterpret_cast<__gm__ const int32_t*>(r.offsets)[i];
  return reinterpret_cast<__gm__ const uint64_t*>(r.address_table)[index] + offset;
}
template<class Args>
__aicore__ bool ValidateRows(const Args& a, uint32_t first, uint32_t last) {
  const auto& r = a.request;
  const uint64_t block_bytes = uint64_t(r.token_bytes) * r.tokens_per_block;
  for (uint32_t i = first; i < last; ++i) {
    const auto index = reinterpret_cast<__gm__ const int32_t*>(r.indices)[i];
    const auto offset = reinterpret_cast<__gm__ const int32_t*>(r.offsets)[i];
    const auto row = reinterpret_cast<__gm__ const int32_t*>(r.output_indices)[i];
    if (index < 0 || uint32_t(index) >= r.table_capacity || offset < 0 ||
        uint64_t(offset) > block_bytes - r.token_bytes || row < 0 || uint32_t(row) >= r.output_rows ||
        !reinterpret_cast<__gm__ const uint8_t*>(r.valid_table)[index]) return false;
    const auto base = reinterpret_cast<__gm__ const uint64_t*>(r.address_table)[index];
    if (!Range(a.source_base, a.source_bytes, base, block_bytes)) return false;
  }
  return true;
}

}  // namespace kvaf_aiv
