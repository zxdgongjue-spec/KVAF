#pragma once

// CPU model of the actual UBmem .asc control flow. This models exact byte
// transfers and event-credit pairing, NOT CANN compilation, device caches,
// MTE timing, Host-DDR visibility, or hardware completion.
#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

#define __aicore__
#define __global__
#define __gm__
#define KVAF_UBMEM_CPU_MODEL 1

namespace ubmem_model {
inline uint32_t core = 0, workers = 1;
inline bool events[6][8]{};
inline uintptr_t tracked_begin = 0, tracked_end = 0;
inline std::map<uintptr_t, uint32_t> owners;
inline void TrackWrites(void* address, size_t bytes) {
  tracked_begin = reinterpret_cast<uintptr_t>(address);
  tracked_end = tracked_begin + bytes;
  owners.clear();
}
inline void Write(void* address, size_t bytes) {
  const auto start = reinterpret_cast<uintptr_t>(address);
  if (start >= tracked_begin && start < tracked_end) {
    assert(bytes <= tracked_end - start);
    for (auto sector = start / 32; sector <= (start + bytes - 1) / 32; ++sector) {
      auto found = owners.emplace(sector, core);
      assert(found.second || found.first->second == core);
    }
  }
}
inline void CheckDrained() { for (auto& row : events) for (bool event : row) assert(!event); }
}  // namespace ubmem_model

namespace AscendC {
using event_t = uint32_t;
constexpr event_t EVENT_ID0 = 0;
enum class HardEvent { S_MTE2, MTE2_S, MTE2_MTE3, MTE3_MTE2, MTE3_S, S_MTE3 };
template <HardEvent Event> void SetFlag(event_t id) {
  auto& credit = ubmem_model::events[static_cast<unsigned>(Event)][id];
  assert(!credit); credit = true;
}
template <HardEvent Event> void WaitFlag(event_t id) {
  auto& credit = ubmem_model::events[static_cast<unsigned>(Event)][id];
  assert(credit); credit = false;
}
inline uint32_t GetBlockNum() { return ubmem_model::workers; }
inline uint32_t GetBlockIdx() { return ubmem_model::core; }
enum class TPosition { VECCALC };
enum class PaddingMode { Normal };
template <typename T> struct LocalTensor {
  T* data;
  LocalTensor<T> operator[](size_t i) const { return {data + i}; }
  T GetValue(size_t index) const { T value; std::memcpy(&value, data + index, sizeof(T)); return value; }
  void SetValue(size_t index, T value) { std::memcpy(data + index, &value, sizeof(T)); }
};
template <typename T> struct GlobalTensor {
  T* data{nullptr};
  T GetValue(size_t i) const { T value; std::memcpy(&value, data + i, sizeof(T)); return value; }
  void SetGlobalBuffer(T* value) { data = value; }
};
enum class CacheLine { ENTIRE_DATA_CACHE };
template <typename T, CacheLine> void DataCacheCleanAndInvalid(GlobalTensor<T>) {}
template <TPosition> struct TBuf {
  std::vector<uint8_t> bytes;
  template <typename T> LocalTensor<T> Get() { return {reinterpret_cast<T*>(bytes.data())}; }
};
struct TPipe {
  template <TPosition Position> void InitBuffer(TBuf<Position>& buffer, size_t bytes) { buffer.bytes.resize(bytes); }
};
struct DataCopyExtParams {
  uint16_t blockCount;
  uint32_t blockLen, srcStride, dstStride, reserved;
};
template <typename T> struct DataCopyPadExtParams { bool enabled; uint8_t left, right; T value; };
template <typename T, PaddingMode>
void DataCopyPad(LocalTensor<T> destination, GlobalTensor<T> source,
                 DataCopyExtParams copy, DataCopyPadExtParams<T>) {
  assert(copy.blockCount == 1 && copy.srcStride == 0 && copy.dstStride == 0);
  std::memcpy(destination.data, source.data, copy.blockLen);
}
template <typename T, PaddingMode>
void DataCopyPad(GlobalTensor<T> destination, LocalTensor<T> source, DataCopyExtParams copy) {
  assert(copy.blockCount == 1 && copy.srcStride == 0 && copy.dstStride == 0);
  ubmem_model::Write(destination.data, copy.blockLen);
  std::memcpy(destination.data, source.data, copy.blockLen);
}
}  // namespace AscendC

struct FunLevelKType { struct { unsigned int type, bytes, kind; } value; };
enum { F_TYPE_KTYPE = 0, K_TYPE_AIV = 1 };
