#pragma once

// CPU-only shim for executing the actual .asc control flow. It does not model
// AIV register initialization, pipeline timing, or DMA/cache visibility;
// those require CANN/A5 tests.
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
inline uint64_t mte2_aligned_copies{}, mte2_tail_copies{};
inline uint64_t ub_scalar_reads{};
inline uint64_t out_to_ub_resets{}, event_waits[4][4]{};
inline bool corrupt_next_mte2{};
// Optional exact source interval for tests: a padded UB tail must never cause
// a read outside the requested GM token, even if the allocation has slack.
inline uintptr_t mte2_source_begin{}, mte2_source_end{};
inline void CheckMte2Source(const void* source, uint64_t bytes) {
  if (!mte2_source_end) return;
  const auto address = reinterpret_cast<uintptr_t>(source);
  if (address < mte2_source_begin || address > mte2_source_end ||
      bytes > mte2_source_end - address) std::abort();
}
}
inline void set_flag(int source, int target, event_t event) {
  if (model_events[source][target][event]) std::abort();
  model_events[source][target][event] = true;
}
inline void wait_flag(int source, int target, event_t event) {
  if (!model_events[source][target][event]) std::abort();
  model_events[source][target][event] = false;
  ++aiv_model::event_waits[source][target];
  if (source == PIPE_MTE2) {
    auto* dst = aiv_model::mte2.destination;
    const auto bytes = aiv_model::mte2.bytes;
    aiv_model::mte2.Finish();
    if (dst && bytes && aiv_model::corrupt_next_mte2) {
      dst[bytes - 1] ^= 1;
      aiv_model::corrupt_next_mte2 = false;
    }
  }
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
enum class DataCopyMVType : uint8_t { UB_TO_OUT, OUT_TO_UB };
inline void ResetLoopModePara(DataCopyMVType type) {
  // Hardware register effects need device validation; model only the selected
  // direction. The MTE2 probe must not configure the MTE3 pipeline.
  if (type != DataCopyMVType::OUT_TO_UB) std::abort();
  ++aiv_model::out_to_ub_resets;
}
template <typename T> struct GlobalTensor {
  T* data;
  void SetGlobalBuffer(T* value) { data = value; }
  void SetL2CacheHint(CacheMode) {}
  GlobalTensor operator[](uint64_t offset) const { return GlobalTensor{data + offset}; }
};
template <typename T> struct LocalTensor {
  struct { uint8_t logicPos; uintptr_t bufferAddr; uint32_t dataLen; } address_{};
  LocalTensor() = default;
  LocalTensor(TPosition pos, uint32_t offset, uint32_t elements) {
    if (offset % 32 || uint64_t(offset) + uint64_t(elements) * sizeof(T) > sizeof(aiv_model::ub))
      std::abort();
    address_ = {static_cast<uint8_t>(pos), get_imm(offset), elements * uint32_t(sizeof(T))};
  }
  LocalTensor operator[](uint32_t offset) const {
    const uint64_t byte_offset = uint64_t(offset) * sizeof(T);
    if (byte_offset > address_.dataLen) std::abort();
    auto result = *this;
    result.address_.bufferAddr += byte_offset;
    result.address_.dataLen -= byte_offset;
    return result;
  }
  T GetValue(uint32_t index) const {
    // A scalar consumer must wait for MTE2 before reading its destination.
    if (aiv_model::mte2.destination || uint64_t(index + 1) * sizeof(T) > address_.dataLen)
      std::abort();
    ++aiv_model::ub_scalar_reads;
    return reinterpret_cast<const T*>(address_.bufferAddr)[index];
  }
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
void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count) {
  const uint64_t bytes = uint64_t(count) * sizeof(T);
  aiv_model::CheckMte2Source(src.data, bytes);
  if (!bytes || bytes % 32 || dst.address_.bufferAddr % 32 ||
      reinterpret_cast<uintptr_t>(src.data) % sizeof(T) || bytes > dst.address_.dataLen ||
      aiv_model::mte2.destination || aiv_model::mte3.destination) std::abort();
  aiv_model::mte2 = {reinterpret_cast<uint8_t*>(dst.address_.bufferAddr),
                    reinterpret_cast<const uint8_t*>(src.data), uint32_t(bytes), uint32_t(bytes)};
  ++aiv_model::mte2_copies;
  ++aiv_model::mte2_aligned_copies;
}
template <typename T>
void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                 const DataCopyExtParams& p, const DataCopyPadExtParams<T>& pad) {
  const uint32_t padded = (p.blockLen + 31) & ~31U;
  aiv_model::CheckMte2Source(src.data, p.blockLen);
  if (p.blockCount != 1 || p.srcStride || p.dstStride || p.rsv || pad.isPad ||
      pad.leftPadding || pad.rightPadding || dst.address_.bufferAddr % 32 ||
      padded > dst.address_.dataLen || aiv_model::mte2.destination ||
      aiv_model::mte3.destination) std::abort();
  aiv_model::mte2 = {reinterpret_cast<uint8_t*>(dst.address_.bufferAddr),
                    reinterpret_cast<const uint8_t*>(src.data), p.blockLen, padded};
  ++aiv_model::mte2_copies;
  ++aiv_model::mte2_tail_copies;
}
template <typename T>
void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                 const DataCopyExtParams& p) {
  if (p.blockCount != 1 || p.srcStride || p.dstStride || p.rsv ||
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
