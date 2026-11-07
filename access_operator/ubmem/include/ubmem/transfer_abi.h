#pragma once

#include <stdint.h>

#ifndef KVAF_UBMEM_INLINE
#define KVAF_UBMEM_INLINE inline
#endif

namespace kvaf_ubmem {

constexpr uint32_t kAbiVersion = 1;
constexpr uint32_t kMaxWorkers = 64;
constexpr uint32_t kCopyChunkBytes = 8192;
constexpr uint32_t kLocalMemoryBytes = 80 * 1024;
constexpr uint32_t kGetLocalMemoryBytes = 32 * 1024;
constexpr uint64_t kStatusMagic = 0x55424d454d303031ULL;

enum Result : uint64_t {
  kSuccess = 0,
  kInvalidCount = 1,
  kInvalidLookup = 2,
  kInvalidSource = 3,
  kInvalidDestination = 4,
  kInvalidAbi = 5,
};

// Exactly one 32-byte write sector per core; do not share a partial sector.
struct CoreStatus {
  uint64_t magic;
  uint64_t result;
  uint64_t failed_request;
  uint64_t copied_requests;
};

struct GetArgs {
  uint64_t address_table, valid_table;
  uint64_t indices, offsets, miss_count, output_indices;
  uint64_t output, status;
  uint64_t region_base, region_bytes, slot_bytes;
  uint32_t table_capacity, request_capacity, output_rows;
  uint32_t token_bytes, tokens_per_block, workers;
  uint32_t abi_version, reserved;
};

struct PutEntry {
  uint64_t input_offset, length, destination;
};

struct PutArgs {
  uint64_t input, input_bytes, entries, status;
  uint64_t region_base, region_bytes, slot_bytes;
  uint32_t count, abi_version;
};

static_assert(sizeof(CoreStatus) == 32, "UBmem status ABI");
static_assert(sizeof(GetArgs) == 120, "UBmem Get ABI");
static_assert(sizeof(PutEntry) == 24, "UBmem Put metadata ABI");
static_assert(sizeof(PutArgs) == 64, "UBmem Put ABI");

// Subtraction checks avoid overflow even at the end of an address space.
KVAF_UBMEM_INLINE bool Contains(uint64_t base, uint64_t bytes,
                                 uint64_t address, uint64_t width) {
  return base != 0 && bytes != 0 && width != 0 &&
         bytes <= UINT64_MAX - base && address >= base &&
         width <= bytes && address - base <= bytes - width;
}

KVAF_UBMEM_INLINE bool ValidOffset(uint64_t bytes, uint64_t offset,
                                    uint64_t width) {
  return width != 0 && width <= bytes && offset <= bytes - width;
}

// With a 32-byte-aligned output, 656-byte rows share a sector only with their
// even/odd neighbor. 2624-byte rows are sector aligned. A duplicate output
// index also has one owner and is processed in request order on that owner.
KVAF_UBMEM_INLINE uint32_t OutputOwner(uint32_t row, uint32_t token_bytes,
                                        uint32_t workers) {
  return (token_bytes == 656 ? row / 2 : row) % workers;
}

}  // namespace kvaf_ubmem
