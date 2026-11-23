// Shared keyfile / flag / payload-pattern contract for the CPU-SGL push demo.
#ifndef KVAF_DEMO_CPU_SGL_PUSH_PROTOCOL_H_
#define KVAF_DEMO_CPU_SGL_PUSH_PROTOCOL_H_

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace kvaf::demo {

// Layout of the registered NPU HBM buffer:
//   [0, 4096)                 flag area; first 8 bytes = uint64 seq written by CPU
//   [4096, 4096+data_bytes)   run area; iteration writes runs sequentially from base
constexpr uint64_t kFlagBytes = 4096U;
// seq value that tells the NPU poller to stop and start verification.
constexpr uint64_t kStopSeq = ~0ULL;

// Self-describing token row pattern. Both sides know it, so the NPU can verify
// any landed row without knowing the miss list.
inline uint64_t RowWord(uint64_t token_id, uint64_t word_index) {
  return token_id * 0x9e3779b97f4a7c15ULL + word_index * 0xbf58476d1ce4e5b9ULL;
}

inline void FillRow(uint8_t *row, uint32_t token_bytes, uint64_t token_id) {
  for (uint64_t off = 0; off < token_bytes; off += 8) {
    uint64_t word = RowWord(token_id, off / 8);
    std::memcpy(row + off, &word, 8);
  }
}

// Returns token_id on success, ~0 on mismatch. Checks head/mid/tail words.
// token_id is recovered from word 0 via the modular inverse of the RowWord
// multiplier (odd constant, invertible mod 2^64).
inline uint64_t VerifyRow(const uint8_t *row, uint32_t token_bytes) {
  constexpr uint64_t kRowWordMulInv = 0xf1de83e19937733dULL;
  uint64_t token_id = 0;
  std::memcpy(&token_id, row, 8);
  token_id *= kRowWordMulInv;
  const uint64_t offs[3] = {0, uint64_t(token_bytes / 2) & ~7ULL, uint64_t(token_bytes - 8)};
  for (uint64_t off : offs) {
    uint64_t word = 0;
    std::memcpy(&word, row + off, 8);
    if (word != RowWord(token_id, off / 8)) return ~0ULL;
  }
  return token_id;
}

inline void WriteHex(FILE *f, const char *key, const uint8_t *bytes, uint32_t count) {
  std::fprintf(f, "%s=", key);
  for (uint32_t i = 0; i < count; ++i) std::fprintf(f, "%02x", bytes[i]);
  std::fprintf(f, "\n");
}

inline bool ReadHex(const char *text, uint8_t *out, uint32_t count) {
  if (std::strlen(text) != count * 2) return false;
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t value = 0;
    if (std::sscanf(text + 2 * i, "%2x", &value) != 1) return false;
    out[i] = static_cast<uint8_t>(value);
  }
  return true;
}

inline uint64_t NowNs() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}

}  // namespace kvaf::demo

#endif
