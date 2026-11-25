#pragma once
// CPU model for the actual AIV SGL control flow, SIMT lanes and DMA commands.
// Does not simulate hardware cache coherence or asynchronous MTE timing.
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#define __aicore__
#define __global__
#if defined(KVAF_CPU_SGL_AIV_ADDRESS_SPACE_CHECK)
// Clang can enforce GM/local type separation without the CANN toolchain.
#define __gm__ __attribute__((address_space(1)))
#else
#define __gm__
#endif
#define __ubuf__
#define __simt_vf__
#define LAUNCH_BOUND(n)
namespace sgl_model {
inline thread_local uint32_t core{};
alignas(64) inline thread_local uint8_t ub[4096 * 8 + 256 * 4];
inline uint64_t Load(uint64_t address) {
  std::this_thread::yield();
  return __atomic_load_n(reinterpret_cast<uint64_t*>(address), __ATOMIC_ACQUIRE);
}
inline void Store(uint64_t address, uint64_t value) {
  __atomic_store_n(reinterpret_cast<uint64_t*>(address), value, __ATOMIC_RELEASE);
}
inline std::atomic<uint32_t> descriptor_copies{};
inline uint32_t pause_row = UINT32_MAX;
inline std::atomic<bool> paused{}, resume{true};
inline uint64_t descriptor{}, descriptor_end{};
}
inline thread_local struct { uint32_t x; } threadIdx;
enum { PIPE_S, PIPE_V, PIPE_MTE3, PIPE_ALL };
using event_t = uint32_t;
inline thread_local bool events[3][3][2]{};
inline void set_flag(int from, int to, event_t e) { assert(!events[from][to][e]); events[from][to][e] = true; }
inline void wait_flag(int from, int to, event_t e) { assert(events[from][to][e]); events[from][to][e] = false; }
inline uintptr_t get_imm(uint32_t n) { return reinterpret_cast<uintptr_t>(sgl_model::ub + n); }
template<class T> T asc_ldcg(__gm__ T* p) { return *p; }
namespace AscendC {
inline uint32_t GetBlockIdx() { return sgl_model::core; }
inline uint64_t GetSystemCycle() { return std::chrono::steady_clock::now().time_since_epoch().count(); }
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
enum class CacheMode { CACHE_MODE_DISABLE };
enum class TPosition { VECIN };
template<class T> struct GlobalTensor {
  __gm__ T* data{};
  void SetGlobalBuffer(__gm__ T* p) { data = p; }
  void SetL2CacheHint(CacheMode) {}
  GlobalTensor operator[](uint64_t n) { return {data + n}; }
};
template<class T, CacheLine, DcciDst> void DataCacheCleanAndInvalid(GlobalTensor<T>) {}
template<int> void PipeBarrier() { for (auto& from : events) for (auto& to : from) for (bool e : to) assert(!e); }
template<class T> struct LocalTensor {
  T* data;
  LocalTensor(TPosition, uint32_t offset, uint32_t) : data(reinterpret_cast<T*>(sgl_model::ub + offset)) {}
};
struct DataCopyExtParams { uint16_t blocks; uint32_t bytes, src_stride, dst_stride, reserved; };
inline void DataCopyPad(GlobalTensor<uint8_t> dst, LocalTensor<uint8_t> src, DataCopyExtParams p) {
  const auto address = reinterpret_cast<uint64_t>(dst.data);
  assert(address >= sgl_model::descriptor && address + p.bytes <= sgl_model::descriptor_end);
  assert(p.blocks == 1);
  if (address == sgl_model::descriptor + uint64_t(sgl_model::pause_row) * 8) {
    sgl_model::paused = true;
    while (!sgl_model::resume.load()) std::this_thread::yield();
  }
  ++sgl_model::descriptor_copies;
#if defined(KVAF_CPU_SGL_AIV_ADDRESS_SPACE_CHECK)
  for (uint32_t i = 0; i < p.bytes; ++i) dst.data[i] = src.data[i];
#else
  std::memcpy(dst.data, src.data, p.bytes);
#endif
}
namespace Simt {
struct Dim3 { uint32_t x, y, z; };
template<auto Function, class... Args> void VF_CALL(Dim3 d, Args... args) {
  for (threadIdx.x = 0; threadIdx.x < d.x; ++threadIdx.x) Function(args...);
}
}
}
struct FunLevelKType { struct { unsigned int type, bytes, kind; } value; };
enum { F_TYPE_KTYPE = 0, K_TYPE_AIV = 1 };
