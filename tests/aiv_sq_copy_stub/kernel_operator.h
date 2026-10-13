#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#define __gm__
#define __ubuf__
#define __aicore__
#define __simt_vf__
#define __global__
#define LAUNCH_BOUND(...)

enum { PIPE_S, PIPE_V, PIPE_MTE2, PIPE_MTE3, EVENT_ID0 = 0, EVENT_ID1 = 1 };

namespace kvaf::aiv_sq_copy_stub {

struct PendingCopy {
  void* destination;
  const void* source;
  std::size_t bytes;
};

inline thread_local std::uint32_t worker = 0U;
inline thread_local std::uint32_t thread = 0U;
alignas(64) inline thread_local std::array<std::uint8_t, 16384> local_memory{};
inline thread_local std::vector<PendingCopy> pending;
inline thread_local std::vector<PendingCopy> pending_reads;
inline thread_local std::uint64_t mte2_reads = 0U;
inline thread_local std::uint64_t clock_reads = 0U;
inline thread_local std::uint64_t reads_at_copy_begin = 0U;
inline thread_local std::uint64_t mte3_waits = 0U;
inline thread_local bool require_completed_copy = true;
inline std::atomic<std::uint64_t> clock{0U};

inline void CompleteCopies() {
  for (const auto& copy : pending) std::memcpy(copy.destination, copy.source, copy.bytes);
  pending.clear();
}

}

struct SimulatedThreadIndex {
  operator std::uint32_t() const { return kvaf::aiv_sq_copy_stub::thread; }
};

inline const struct { SimulatedThreadIndex x; } threadIdx;

inline void set_loop_size_ubtoout(std::uint64_t) {}
inline void set_loop1_stride_ubtoout(std::uint64_t) {}
inline void set_loop2_stride_ubtoout(std::uint64_t) {}
inline void set_atomic_none() {}
inline void set_flag(int, int, int) {}

inline void wait_flag(int source, int, int) {
  if (source == PIPE_MTE3) {
    kvaf::aiv_sq_copy_stub::CompleteCopies();
    ++kvaf::aiv_sq_copy_stub::mte3_waits;
  }
  if (source == PIPE_MTE2) {
    for (const auto& copy : kvaf::aiv_sq_copy_stub::pending_reads)
      std::memcpy(copy.destination, copy.source, copy.bytes);
    kvaf::aiv_sq_copy_stub::pending_reads.clear();
  }
}

inline void* get_imm(std::uint32_t offset) {
  assert(offset < kvaf::aiv_sq_copy_stub::local_memory.size());
  return kvaf::aiv_sq_copy_stub::local_memory.data() + offset;
}

inline void copy_ubuf_to_gm_align_v2(float* destination, float* source,
                                    std::uint64_t configuration, std::uint64_t) {
  assert(kvaf::aiv_sq_copy_stub::pending_reads.empty());
  kvaf::aiv_sq_copy_stub::pending.push_back({destination, source,
      static_cast<std::size_t>(configuration >> 25U)});
}

template <typename... Arguments>
inline void copy_gm_to_ubuf_align_v2(std::uint8_t* destination, std::uint8_t* source,
    unsigned, unsigned, std::uint32_t bytes, Arguments...) {
  assert(kvaf::aiv_sq_copy_stub::pending.empty());
  assert(bytes <= kvaf::aiv_sq_copy_stub::local_memory.size());
  kvaf::aiv_sq_copy_stub::pending_reads.push_back({destination, source, bytes});
  ++kvaf::aiv_sq_copy_stub::mte2_reads;
}

namespace AscendC {

inline std::uint32_t GetBlockIdx() { return kvaf::aiv_sq_copy_stub::worker; }
inline std::uint64_t GetSystemCycle() {
  assert(kvaf::aiv_sq_copy_stub::pending_reads.empty());
  if (kvaf::aiv_sq_copy_stub::clock_reads++ == 1U)
    kvaf::aiv_sq_copy_stub::reads_at_copy_begin = kvaf::aiv_sq_copy_stub::mte2_reads;
  if (kvaf::aiv_sq_copy_stub::require_completed_copy)
    assert(kvaf::aiv_sq_copy_stub::pending.empty());
  return kvaf::aiv_sq_copy_stub::clock.fetch_add(1U, std::memory_order_relaxed);
}

namespace Simt {

struct Dim3 { std::uint32_t x; std::uint32_t y; std::uint32_t z; };

template <auto Function, typename... Arguments>
inline void VF_CALL(Dim3 dimensions, Arguments... arguments) {
  for (std::uint32_t index = 0U; index < dimensions.x; ++index) {
    kvaf::aiv_sq_copy_stub::thread = index;
    Function(arguments...);
  }
}

}

}
