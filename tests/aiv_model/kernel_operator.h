#pragma once

// CPU-only shim for executing the actual .asc control flow. It models neither
// AIV pipeline timing nor DMA/cache visibility; those require CANN/A5 tests.
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define __aicore__
#define __global__
#define __gm__
#define __ubuf__
#define __simt_vf__
#define LAUNCH_BOUND(n)

namespace aiv_model {
alignas(64) inline uint8_t ub[160 * 1024];
inline uint32_t core = 0;
inline uint64_t cycle = 0;
void Poll(void* address);
void Doorbell(uint32_t value, uint32_t* address);
}  // namespace aiv_model

inline struct { uint32_t x; } threadIdx;
enum { PIPE_S, PIPE_V, PIPE_MTE3 };
using event_t = uint32_t;
inline bool model_events[3][3][3]{};
inline void set_flag(int source, int target, event_t event) {
  if (model_events[source][target][event]) std::abort();
  model_events[source][target][event] = true;
}
inline void wait_flag(int source, int target, event_t event) {
  if (!model_events[source][target][event]) std::abort();
  model_events[source][target][event] = false;
}
inline void set_loop_size_ubtoout(uint64_t) {}
inline void set_loop1_stride_ubtoout(uint64_t) {}
inline void set_loop2_stride_ubtoout(uint64_t) {}
inline void set_atomic_none() {}
inline uintptr_t get_imm(uint32_t offset) {
  return reinterpret_cast<uintptr_t>(aiv_model::ub + offset);
}
inline void copy_ubuf_to_gm_align_v2(float* dst, float* src, uint64_t config, uint64_t) {
  std::memcpy(dst, src, config >> 25U);
}
inline void st_dev(uint32_t value, uint32_t* address, int) {
  aiv_model::Doorbell(value, address);
}

namespace AscendC {
enum class TPosition { VECIN };
enum class HardEvent { MTE2_MTE3, MTE3_MTE2, MTE3_S };
inline bool hard_events[3][8]{};
template <HardEvent Event> void SetFlag(event_t event) {
  auto& flag = hard_events[static_cast<unsigned>(Event)][event];
  if (flag) std::abort();
  flag = true;
}
template <HardEvent Event> void WaitFlag(event_t event) {
  auto& flag = hard_events[static_cast<unsigned>(Event)][event];
  if (!flag) std::abort();
  flag = false;
}
template <typename T> struct LocalTensor {
  struct { uint8_t logicPos; uint32_t bufferAddr, dataLen; } address_{};
};
struct DataCopyExtParams { uint32_t blocks, bytes, src_stride, dst_stride, reserved; };
template <typename T> struct DataCopyPadExtParams { bool pad; T left, right, value; };
inline uint32_t GetBlockIdx() { return aiv_model::core; }
inline uint64_t GetSystemCycle() { return ++aiv_model::cycle; }
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template <typename T> struct GlobalTensor {
  T* data;
  void SetGlobalBuffer(T* value) { data = value; }
  GlobalTensor operator[](uint64_t offset) { return GlobalTensor{data + offset}; }
};
template <typename T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams params,
                 DataCopyPadExtParams<T>) {
  std::memcpy(aiv_model::ub + dst.address_.bufferAddr, src.data, params.bytes);
}
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams params) {
  std::memcpy(dst.data, aiv_model::ub + src.address_.bufferAddr, params.bytes);
}
template <typename T, CacheLine, DcciDst>
void DataCacheCleanAndInvalid(GlobalTensor<T> tensor) {
  aiv_model::Poll(tensor.data);
}
namespace Simt {
struct Dim3 { uint32_t x, y, z; };
template <auto Function, typename... Args>
void VF_CALL(Dim3 dim, Args... args) {
  for (threadIdx.x = 0; threadIdx.x < dim.x; ++threadIdx.x) Function(args...);
}
}  // namespace Simt
}  // namespace AscendC

struct FunLevelKType { struct { unsigned int type, bytes, kind; } value; };
enum { F_TYPE_KTYPE = 0, K_TYPE_AIV = 1 };
