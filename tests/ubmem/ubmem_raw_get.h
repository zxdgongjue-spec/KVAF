#pragma once
#include "raw_copy_abi.h"

// Raw-address entry for bandwidth tests; shares the production MTE pipeline.
extern "C" __global__ __aicore__ void KvafUbmemRawGetKernel(kvaf::ubmem::RawCopyArgs a) {
  using namespace kvaf::ubmem;
  const uint32_t core = AscendC::GetBlockIdx();
  auto* requests = reinterpret_cast<__gm__ const RawCopyRequest*>(a.requests);
  uint64_t completed = 0;
  const uint64_t begin = static_cast<uint64_t>(AscendC::GetSystemCycle());
  MteCopyPipeline copy(a.row_bytes);
  for (uint64_t row = core; row < a.rows; row += a.blocks) {
    copy.Copy(requests[row].destination, requests[row].source, a.row_bytes);
    ++completed;
  }
  copy.Finish();
  const uint64_t end = static_cast<uint64_t>(AscendC::GetSystemCycle());
  auto* report = reinterpret_cast<__gm__ RawCopyReport*>(a.reports) + core;
  report->cycles = end - begin;
  report->completed = completed;
}
static const struct FunLevelKType KvafUbmemRawGetKernel_type __attribute__(
    (used, section(".ascend.meta.KvafUbmemRawGetKernel"))) =
    {{F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};
