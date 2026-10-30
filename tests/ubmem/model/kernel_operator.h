#pragma once
// CPU model for address/tail correctness and MTE handoff dependencies.
// Work runs at the corresponding wait; hardware timing/cache/fabric are unmodeled.
#include "ubmem_batch_abi.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <utility>
#define __global__
#define __aicore__
#define __gm__
inline unsigned model_core;
inline std::array<std::uint8_t, kvaf::ubmem::kMteBufferBytes> model_ub;
inline void ModelCheck(bool condition) { if (!condition) std::abort(); }
struct ModelTransfer {
  std::uint64_t begin, end;
  std::function<void()> execute;
};
struct ModelMtePipe {
  std::deque<ModelTransfer> pending;
  std::uint64_t issued = 0, completed = 0, scalar_visible = 0;
  std::size_t peak_pending = 0;
  explicit operator bool() const { return !pending.empty(); }
  bool Overlaps(std::uint64_t begin, std::uint64_t end) const {
    for (const auto& op : pending)
      if (begin < op.end && op.begin < end) return true;
    return false;
  }
  void Push(std::uint64_t begin, std::uint64_t end, std::function<void()> action) {
    pending.push_back({begin, end, std::move(action)});
    ++issued;
    if (pending.size() > peak_pending) peak_pending = pending.size();
  }
  void CompleteThrough(std::uint64_t ticket) {
    ModelCheck(ticket <= issued);
    while (completed < ticket) {
      auto action = std::move(pending.front().execute);
      pending.pop_front();
      action();
      ++completed;
    }
  }
};
inline ModelMtePipe model_mte2, model_mte3;
inline bool model_mte_overlap = false;
struct ModelEvent { bool armed = false; std::uint64_t ticket = 0; };
inline std::array<std::array<ModelEvent, 2>, 4> model_events;
inline bool ModelEventsIdle() {
  for (const auto& direction : model_events)
    for (const auto& event : direction)
      if (event.armed) return false;
  return true;
}
enum event_t { EVENT_ID0, EVENT_ID1 };
namespace AscendC {
inline unsigned GetBlockIdx() { return model_core; }
inline std::uint64_t GetSystemCycle() {
  // The ending timestamp must follow a scalar-visible completion fence, not
  // merely cross-pipe waits protecting the two UB banks.
  ModelCheck(!model_mte2 && !model_mte3);
  ModelCheck(model_mte3.scalar_visible == model_mte3.issued);
  static std::uint64_t tick = 0;
  return ++tick;
}
enum class TPosition { VECIN };
enum class HardEvent { S_MTE2, MTE2_MTE3, MTE3_MTE2, MTE3_S };
template <HardEvent event> void SetFlag(unsigned id) {
  ModelCheck(id < 2);
  auto& flag = model_events[static_cast<unsigned>(event)][id];
  ModelCheck(!flag.armed);
  flag.armed = true;
  if constexpr (event == HardEvent::MTE2_MTE3) flag.ticket = model_mte2.issued;
  else if constexpr (event != HardEvent::S_MTE2) flag.ticket = model_mte3.issued;
}
template <HardEvent event> void WaitFlag(unsigned id) {
  ModelCheck(id < 2);
  auto& flag = model_events[static_cast<unsigned>(event)][id];
  ModelCheck(flag.armed);
  flag.armed = false;
  if constexpr (event == HardEvent::MTE2_MTE3) {
    model_mte2.CompleteThrough(flag.ticket);
  } else if constexpr (event == HardEvent::MTE3_MTE2 || event == HardEvent::MTE3_S) {
    model_mte3.CompleteThrough(flag.ticket);
    if constexpr (event == HardEvent::MTE3_S) model_mte3.scalar_visible = flag.ticket;
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
  auto* ub = UbAddress(dst, copy);
  const auto begin = dst.address_.bufferAddr;
  const auto end = begin + (copy.blockLen + 31U) / 32U * 32U;
  ModelCheck(!model_mte2.Overlaps(begin, end) && !model_mte3.Overlaps(begin, end));
  model_mte2.Push(begin, end, [ub, src, copy] {
    std::memcpy(ub, src.data, copy.blockLen);
    std::memset(ub + copy.blockLen, 0xA5, (copy.blockLen + 31U) / 32U * 32U - copy.blockLen);
  });
  model_mte_overlap |= static_cast<bool>(model_mte3);
}
template <typename T>
void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& copy) {
  auto* ub = UbAddress(src, copy);
  const auto begin = src.address_.bufferAddr;
  const auto end = begin + (copy.blockLen + 31U) / 32U * 32U;
  ModelCheck(!model_mte2.Overlaps(begin, end));  // This bank must be ready.
  model_mte3.Push(begin, end, [ub, dst, copy] { std::memcpy(dst.data, ub, copy.blockLen); });
  model_mte_overlap |= static_cast<bool>(model_mte2);
}
}
struct FunLevelKType { struct { unsigned type, bytes, kind; } value; };
enum { F_TYPE_KTYPE, K_TYPE_AIV };
