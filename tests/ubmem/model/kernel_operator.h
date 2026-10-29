#pragma once
// CPU model for address/tail correctness and MTE/SIMT handoff dependencies.
// Work runs at the corresponding wait; hardware timing/cache/fabric are unmodeled.
#include "ubmem/batch_abi.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#define __global__
#define __aicore__
#define __gm__
#define __simt_vf__
#define LAUNCH_BOUND(n)
inline unsigned model_core;
inline std::array<std::uint8_t, kvaf::ubmem::kMteBufferBytes> model_ub;
inline std::function<void()> model_mte2, model_mte3;
inline std::function<void()> model_simt;
inline bool model_simt_ready = false;
inline void ModelCheck(bool condition) { if (!condition) std::abort(); }
enum pipe_t { PIPE_S, PIPE_V };
enum event_t { EVENT_ID0 };
inline void set_flag(pipe_t, pipe_t, event_t) {}
inline void wait_flag(pipe_t from, pipe_t to, event_t) {
  if (from == PIPE_S && to == PIPE_V) {
    ModelCheck(!model_simt && !model_simt_ready);
    model_simt_ready = true;
  } else {
    ModelCheck(from == PIPE_V && to == PIPE_S && static_cast<bool>(model_simt));
    auto action = std::move(model_simt); model_simt = {}; action();
  }
}
struct ModelThreadIndex { unsigned x{}; };
inline ModelThreadIndex threadIdx;
namespace AscendC {
namespace Simt {
struct Dim3 { unsigned x, y, z; };
template <auto Function, typename... Args>
void VF_CALL(Dim3 dimensions, Args... args) {
  ModelCheck(model_simt_ready && !model_simt && !model_mte2 && !model_mte3);
  ModelCheck(dimensions.x != 0 && dimensions.y == 1 && dimensions.z == 1);
  model_simt_ready = false;
  model_simt = [=] {
    // Reverse lane order also catches accidental dependence on lane 0
    // publishing data for other lanes without a thread-group barrier.
    for (unsigned lane = dimensions.x; lane > 0; --lane) {
      threadIdx.x = lane - 1;
      Function(args...);
    }
  };
}
}
inline unsigned GetBlockIdx() { return model_core; }
enum class TPosition { VECIN };
enum class HardEvent { S_MTE2, MTE2_MTE3, MTE3_MTE2, MTE3_S };
template <HardEvent> void SetFlag(unsigned) {}
template <HardEvent event> void WaitFlag(unsigned) {
  if constexpr (event == HardEvent::MTE2_MTE3) {
    ModelCheck(static_cast<bool>(model_mte2));
    auto action = std::move(model_mte2); model_mte2 = {}; action();
  } else if constexpr (event == HardEvent::MTE3_MTE2 || event == HardEvent::MTE3_S) {
    ModelCheck(static_cast<bool>(model_mte3));
    auto action = std::move(model_mte3); model_mte3 = {}; action();
  }
}
template <typename T> struct GlobalTensor {
  T* data{};
  void SetGlobalBuffer(T* value) { data = value; }
};
template <typename T> struct LocalTensor {
  struct { std::uint8_t logicPos{}; std::uint64_t bufferAddr{}; std::uint32_t dataLen{}; } address_;
};
struct DataCopyExtParams {
  std::uint16_t blockCount;
  std::uint32_t blockLen, srcStride, dstStride, rsv;
  DataCopyExtParams(std::uint16_t count, std::uint32_t bytes, std::uint32_t src,
                    std::uint32_t dst, std::uint32_t reserved)
      : blockCount(count), blockLen(bytes), srcStride(src), dstStride(dst), rsv(reserved) {}
};
template <typename T> struct DataCopyPadExtParams { bool isPad; std::uint8_t left, right; T value; };
template <typename T>
std::uint8_t* UbAddress(const LocalTensor<T>& tensor, const DataCopyExtParams& copy) {
  const auto offset = tensor.address_.bufferAddr;
  const auto padded_bytes = (copy.blockLen + 31U) / 32U * 32U;
  ModelCheck(copy.blockCount == 1 && copy.srcStride == 0 && copy.dstStride == 0);
  ModelCheck(offset % 32 == 0 && offset <= model_ub.size());
  ModelCheck(tensor.address_.dataLen <= model_ub.size() - offset && padded_bytes <= tensor.address_.dataLen);
  return model_ub.data() + offset;
}
template <typename T>
void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                 const DataCopyExtParams& copy, const DataCopyPadExtParams<T>&) {
  ModelCheck(!model_mte2 && !model_mte3);  // UB is not still being read by MTE3.
  auto* ub = UbAddress(dst, copy);
  model_mte2 = [ub, src, copy] {
    std::memcpy(ub, src.data, copy.blockLen);
    std::memset(ub + copy.blockLen, 0xA5, (copy.blockLen + 31U) / 32U * 32U - copy.blockLen);
  };
}
template <typename T>
void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& copy) {
  ModelCheck(!model_mte2 && !model_mte3);  // MTE2 must have finished producing UB.
  auto* ub = UbAddress(src, copy);
  model_mte3 = [ub, dst, copy] { std::memcpy(dst.data, ub, copy.blockLen); };
}
}
struct FunLevelKType { struct { unsigned type, bytes, kind; } value; };
enum { F_TYPE_KTYPE, K_TYPE_AIV };
