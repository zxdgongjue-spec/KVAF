#ifndef KVAF_BATCH_GET_HOST_TEST_KERNEL_OPERATOR_H_
#define KVAF_BATCH_GET_HOST_TEST_KERNEL_OPERATOR_H_

// CPU-only execution model for testing the real shared helper's control flow.
// This does not model NPU scheduling, cache coherence or transport hardware.
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>

#define __aicore__
#define __gm__
#define __ubuf__
#define __simt_vf__
#define LAUNCH_BOUND(value)

namespace AscendC {
enum Pipe { PIPE_S, PIPE_V, PIPE_MTE3, PIPE_MTE2, PIPE_ALL };
using event_t = uint32_t;
struct ThreadIndex { uint32_t x; };
inline uint32_t block_count = 2U;
inline uint32_t GetBlockNum() { return block_count; }
template <Pipe> inline void PipeBarrier() {}
inline ThreadIndex threadIdx{0U};
inline std::array<std::array<std::array<int, 8>, 5>, 5> events{};
alignas(64) inline std::array<uint8_t, 160U * 1024U> ub{};
inline void (*doorbell_hook)(uint32_t, uint32_t *) = nullptr;
inline void set_flag(Pipe from, Pipe to, event_t event) {
  assert(events[from][to][event] == 0);
  ++events[from][to][event];
}
inline void wait_flag(Pipe from, Pipe to, event_t event) {
  assert(events[from][to][event] == 1);
  --events[from][to][event];
}
inline uintptr_t get_imm(uint32_t offset) {
  assert(offset < ub.size());
  return reinterpret_cast<uintptr_t>(ub.data() + offset);
}
inline void set_loop_size_ubtoout(uint64_t) {}
inline void set_loop1_stride_ubtoout(uint64_t) {}
inline void set_loop2_stride_ubtoout(uint64_t) {}
inline void set_atomic_none() {}
inline void copy_ubuf_to_gm_align_v2(float *dst, float *src,
                                    uint64_t config, uint64_t) {
  std::memcpy(dst, src, config >> 25U);
}
inline void st_dev(uint32_t value, uint32_t *address, uint32_t) {
  if (doorbell_hook != nullptr) {
    doorbell_hook(value, address);
  }
  *address = value;
}
inline uint64_t GetSystemCycle() {
  static uint64_t cycle = 0U;
  return ++cycle;
}
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template <typename T> struct GlobalTensor {
  T *ptr = nullptr;
  void SetGlobalBuffer(T *value) { ptr = value; }
  T GetValue(uint64_t offset) const { return ptr[offset]; }
  void SetValue(uint64_t offset, T value) { ptr[offset] = value; }
  GlobalTensor operator[](uint64_t offset) const { return {ptr + offset}; }
};
template <typename T, CacheLine, DcciDst>
inline void DataCacheCleanAndInvalid(GlobalTensor<T>) {}
namespace Simt {
struct Dim3 { uint32_t x, y, z; };
template <auto Function, typename... Args>
inline void VF_CALL(Dim3 dim, Args... args) {
  for (uint32_t thread = 0U; thread < dim.x; ++thread) {
    threadIdx.x = thread;
    Function(args...);
  }
}
}  // namespace Simt
}  // namespace AscendC
#endif
