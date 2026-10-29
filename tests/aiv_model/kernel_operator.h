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
enum { PIPE_S, PIPE_V, PIPE_MTE3, PIPE_MTE2, PIPE_ALL };
using event_t = uint32_t;
inline bool model_events[4][4][3]{};
namespace aiv_model {
// Defer public DMA copies until their producer event is waited on. This catches
// consuming an unfilled bank, overwriting a bank still being read, and returning
// before the final MTE3 write. It does not emulate CANN instruction generation.
struct PendingCopy {
  uint8_t* destination{};
  const uint8_t* source{};
  uint32_t bytes{}, padded_bytes{};
  void Finish() {
    if (!destination) return;
    std::memcpy(destination, source, bytes);
    if (padded_bytes > bytes) std::memset(destination + bytes, 0xdc, padded_bytes - bytes);
    *this = {};
  }
};
inline PendingCopy mte2, mte3;
inline uint64_t mte2_copies{}, mte3_copies{};
// Coarse contract for state inherited from a preceding operator, not a model
// of any hardware register encoding or errcode 78. DMA needs initialized state.
inline bool dma_state_dirty{};
}
inline void set_flag(int source, int target, event_t event) {
  if (model_events[source][target][event]) std::abort();
  model_events[source][target][event] = true;
}
inline void wait_flag(int source, int target, event_t event) {
  if (!model_events[source][target][event]) std::abort();
  model_events[source][target][event] = false;
  if (source == PIPE_MTE2) aiv_model::mte2.Finish();
  if (source == PIPE_MTE3) aiv_model::mte3.Finish();
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
inline void InitSocState() { aiv_model::dma_state_dirty = false; }
template <int pipe> inline void PipeBarrier() {
  static_assert(pipe == PIPE_ALL);
  for (const auto& pipeline : model_events)
    for (const auto& direction : pipeline)
      for (bool pending : direction) if (pending) std::abort();
  aiv_model::mte2.Finish();
  aiv_model::mte3.Finish();
}
inline uint32_t GetBlockIdx() { return aiv_model::core; }
inline uint64_t GetSystemCycle() { return ++aiv_model::cycle; }
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
enum class CacheMode { CACHE_MODE_DISABLE };
enum class TPosition : uint8_t { VECIN };
template <typename T> struct GlobalTensor {
  T* data;
  void SetGlobalBuffer(T* value) { data = value; }
  void SetL2CacheHint(CacheMode) {}
  GlobalTensor operator[](uint64_t offset) { return GlobalTensor{data + offset}; }
};
template <typename T> struct LocalTensor {
  struct { uint8_t logicPos; uintptr_t bufferAddr; uint32_t dataLen; } address_{};
};
struct DataCopyExtParams {
  uint16_t blockCount;
  uint32_t blockLen;
  int64_t srcStride, dstStride;
  uint32_t rsv;
};
template <typename T> struct DataCopyPadExtParams {
  bool isPad;
  uint8_t leftPadding, rightPadding;
  T paddingValue;
};
template <typename T>
void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                 const DataCopyExtParams& p, const DataCopyPadExtParams<T>& pad) {
  const uint32_t padded = (p.blockLen + 31) & ~31U;
  if (aiv_model::dma_state_dirty || p.blockCount != 1 || p.srcStride || p.dstStride || p.rsv || pad.isPad ||
      pad.leftPadding || pad.rightPadding || dst.address_.bufferAddr % 32 ||
      padded > dst.address_.dataLen || aiv_model::mte2.destination ||
      aiv_model::mte3.destination) std::abort();
  aiv_model::mte2 = {reinterpret_cast<uint8_t*>(dst.address_.bufferAddr),
                    reinterpret_cast<const uint8_t*>(src.data), p.blockLen, padded};
  ++aiv_model::mte2_copies;
}
template <typename T>
void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                 const DataCopyExtParams& p) {
  if (aiv_model::dma_state_dirty || p.blockCount != 1 || p.srcStride || p.dstStride || p.rsv ||
      src.address_.bufferAddr % 32 || p.blockLen > src.address_.dataLen ||
      aiv_model::mte2.destination || aiv_model::mte3.destination) std::abort();
  aiv_model::mte3 = {reinterpret_cast<uint8_t*>(dst.data),
                    reinterpret_cast<const uint8_t*>(src.address_.bufferAddr), p.blockLen, 0};
  ++aiv_model::mte3_copies;
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
