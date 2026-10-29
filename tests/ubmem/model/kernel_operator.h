#pragma once
// Executes the actual kernel's control flow on CPU. Cache visibility and UB
// hardware access are deliberately unmodeled; use kvaf_ubmem_test on Ascend950.
#include <cstdint>
#define __global__
#define __aicore__
#define __simt_vf__
#define __gm__
#define LAUNCH_BOUND(n)
inline struct { unsigned x; } threadIdx;
inline unsigned model_core;
enum { PIPE_ALL };
namespace AscendC {
template <int> void PipeBarrier() {}
inline unsigned GetBlockIdx() { return model_core; }
namespace Simt {
struct Dim3 { unsigned x, y, z; };
template <auto Function, typename... Args>
void VF_CALL(Dim3 dimensions, Args... args) {
  for (threadIdx.x = 0; threadIdx.x < dimensions.x; ++threadIdx.x) Function(args...);
}
}
}
struct FunLevelKType { struct { unsigned type, bytes, kind; } value; };
enum { F_TYPE_KTYPE, K_TYPE_AIV };
