// Production loaders must select independent binaries, symbols and argument ABIs.
#include "kvaf_aiv_batch_get.h"
#include "scatter_batch_get.h"
#include "gather_batch_get.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
std::vector<std::unique_ptr<std::string>> binaries;
void Check(bool ok) { if (!ok) std::abort(); }
constexpr char kDirect[] = "KvafExecuteBatchGetAivKernelV2";
constexpr char kScatter[] = "KvafScatterBatchGetAivKernelV1";
constexpr char kGather[] = "KvafGatherBatchGetAivKernelV1";
std::vector<std::string> launched;
}
aclrtBinary aclrtCreateBinary(const void* data, size_t bytes) {
  return new std::string(static_cast<const char*>(data), bytes);
}
int aclrtBinaryLoad(aclrtBinary binary, aclrtBinHandle* handle) {
  binaries.push_back(std::make_unique<std::string>(*static_cast<std::string*>(binary)));
  *handle = binaries.back().get(); return 0;
}
int aclrtDestroyBinary(aclrtBinary binary) { delete static_cast<std::string*>(binary); return 0; }
int aclrtBinaryGetFunction(aclrtBinHandle binary, const char* name, aclrtFuncHandle* function) {
  if (*static_cast<std::string*>(binary) != name) return 1;
  *function = binary; return 0;
}
int aclrtLaunchKernelWithHostArgs(aclrtFuncHandle function, uint32_t, aclrtStream,
    const aclrtLaunchKernelCfg* cfg, void*, size_t bytes, void*, size_t) {
  const auto symbol = *static_cast<std::string*>(function);
  Check(symbol == kScatter || symbol == kGather);
  Check(bytes == (symbol == kScatter ? sizeof(KvafScatterBatchGetArgs) : sizeof(KvafGatherBatchGetArgs)));
  Check(cfg->numAttrs == 4 && cfg->attrs[3].value.engineType == ACL_RT_ENGINE_TYPE_AIV);
  launched.push_back(symbol); return 0;
}
int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  Check(argc == 2);
  const std::string scenario = argv[1];
  std::string pattern = (fs::temp_directory_path() / "kvaf-loader-XXXXXX").string();
  char* created = mkdtemp(pattern.data()); Check(created != nullptr);
  const fs::path directory(created), original = fs::current_path();
  const auto direct = directory / "kvaf_execute_batch_get_kernels.o";
  const auto legacy = directory / "kvaf_sequential_batch_get_kernels.o";
  const auto scatter = directory / "kvaf_scatter_batch_get_kernels.o";
  const auto gather = directory / "kvaf_gather_batch_get_kernels.o";
  std::ofstream(direct) << kDirect;
  Check(KvafScatterBatchGetLoadKernel(nullptr) == KVAF_BATCH_GET_INVALID_ARGUMENT);
  Check(KvafGatherBatchGetLoadKernel("") == KVAF_BATCH_GET_INVALID_ARGUMENT);
  Check(KvafExecuteBatchGetLoadKernel(direct.c_str()) == 0);
  if (scenario == "missing") {
    Check(KvafScatterBatchGetLoadKernel(direct.c_str()) == KVAF_BATCH_GET_RUNTIME_ERROR);
    Check(KvafGatherBatchGetLoadKernel(direct.c_str()) == KVAF_BATCH_GET_RUNTIME_ERROR);
    Check(binaries.size() == 1);
  }
  // Gather can load before scatter even exists; it does not load the old combined module.
  std::ofstream(gather) << kGather;
  auto gather_path = scenario == "legacy" ? legacy : direct;
  auto scatter_path = gather_path;
  if (scenario == "explicit") { gather_path = gather; scatter_path = scatter; }
  if (scenario == "custom") {
    gather_path = directory / "custom-gather.o";
    scatter_path = directory / "custom-scatter.o";
    std::ofstream(gather_path) << kGather; std::ofstream(scatter_path) << kScatter;
  }
  if (scenario == "relative") {
    fs::current_path(directory); gather_path = direct.filename(); scatter_path = direct.filename();
  }
  if (scenario == "wrong_symbol") {
    Check(KvafScatterBatchGetLoadKernel(gather.c_str()) == KVAF_BATCH_GET_RUNTIME_ERROR);
  }
  Check(KvafGatherBatchGetLoadKernel(gather_path.c_str()) == 0);
  std::ofstream(scatter) << kScatter;
  Check(KvafScatterBatchGetLoadKernel(scatter_path.c_str()) == 0);
  const auto loaded = binaries.size();
  KvafScatterBatchGetArgs s{}; s.request.block_count = 1;
  KvafGatherBatchGetArgs g{}; g.request.block_count = 1;
  Check(KvafGatherBatchGetLaunch(&g, reinterpret_cast<void*>(1)) == 0);
  Check(KvafScatterBatchGetLaunch(&s, reinterpret_cast<void*>(1)) == 0);
  Check(KvafGatherBatchGetLaunch(&g, reinterpret_cast<void*>(1)) == 0);
  Check((launched == std::vector<std::string>{kGather, kScatter, kGather}));
  Check(KvafScatterBatchGetLoadKernel(scatter_path.c_str()) == 0);
  Check(KvafGatherBatchGetLoadKernel(gather_path.c_str()) == 0 && binaries.size() == loaded);
  fs::current_path(original); fs::remove_all(directory);
  std::cout << "independent loaders " << scenario << ": PASS\n";
}
