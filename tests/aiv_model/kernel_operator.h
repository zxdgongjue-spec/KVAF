#pragma once

// CPU-only shim for executing the actual .asc control flow. It models neither
// AIV pipeline timing nor DMA/cache visibility; those require CANN/A5 tests.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

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
enum class PostStage { Fill, Copy, Doorbell };
inline std::vector<PostStage> post_trace;
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
  aiv_model::post_trace.push_back(aiv_model::PostStage::Copy);
  std::memcpy(dst, src, config >> 25U);
}
inline void st_dev(uint32_t value, uint32_t* address, int) {
  aiv_model::Doorbell(value, address);
}

namespace AscendC {
inline uint32_t GetBlockIdx() { return aiv_model::core; }
inline uint64_t GetSystemCycle() { return ++aiv_model::cycle; }
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template <typename T> struct GlobalTensor {
  T* data;
  void SetGlobalBuffer(T* value) { data = value; }
  GlobalTensor operator[](uint64_t offset) { return GlobalTensor{data + offset}; }
};
template <typename T, CacheLine, DcciDst>
void DataCacheCleanAndInvalid(GlobalTensor<T> tensor) {
  aiv_model::Poll(tensor.data);
}
namespace Simt {
struct Dim3 { uint32_t x, y, z; };
template <auto Function, typename... Args>
void VF_CALL(Dim3 dim, Args... args) {
  aiv_model::post_trace.push_back(aiv_model::PostStage::Fill);
  for (threadIdx.x = 0; threadIdx.x < dim.x; ++threadIdx.x) Function(args...);
}
}  // namespace Simt
}  // namespace AscendC

struct FunLevelKType { struct { unsigned int type, bytes, kind; } value; };
enum { F_TYPE_KTYPE = 0, K_TYPE_AIV = 1 };
