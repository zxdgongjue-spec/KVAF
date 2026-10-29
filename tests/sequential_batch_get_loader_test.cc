// Use both production loaders; fake ACL validates the selected binary/symbol.
#include "kvaf_aiv_batch_get.h"
#include "sequential_batch_get.h"
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
constexpr char kSequential[] = "KvafSequentialBatchGetAivKernelV1";
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
    const aclrtLaunchKernelCfg*, void*, size_t bytes, void*, size_t) {
  Check(*static_cast<std::string*>(function) == kSequential);
  Check(bytes == sizeof(KvafSequentialBatchGetArgs)); return 0;
}

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  Check(argc == 2);
  const std::string scenario = argv[1];
  std::string pattern = (fs::temp_directory_path() / "kvaf-loader-XXXXXX").string();
  char* created = mkdtemp(pattern.data()); Check(created != nullptr);
  const fs::path directory(created), original = fs::current_path();
  const auto direct = directory / "kvaf_execute_batch_get_kernels.o";
  const auto sequential = directory / "kvaf_sequential_batch_get_kernels.o";
  const auto custom = directory / "custom.o";
  std::ofstream(direct) << kDirect;
  Check(KvafSequentialBatchGetLoadKernel(nullptr) == KVAF_BATCH_GET_INVALID_ARGUMENT);
  Check(KvafSequentialBatchGetLoadKernel("") == KVAF_BATCH_GET_INVALID_ARGUMENT);
  Check(KvafExecuteBatchGetLoadKernel(direct.c_str()) == 0);
  if (scenario == "missing") {
    // Do not try loading the direct-read module when the companion is missing.
    Check(KvafSequentialBatchGetLoadKernel(direct.c_str()) == KVAF_BATCH_GET_RUNTIME_ERROR);
    Check(binaries.size() == 1);
  }
  std::ofstream(sequential) << kSequential;
  std::ofstream(custom) << kSequential;
  auto path = direct;
  if (scenario == "explicit") path = sequential;
  if (scenario == "custom") path = custom;
  if (scenario == "relative") { fs::current_path(directory); path = direct.filename(); }
  Check(KvafSequentialBatchGetLoadKernel(path.c_str()) == 0);
  Check(binaries.size() == 2);
  Check(*binaries[0] == kDirect && *binaries[1] == kSequential);
  KvafSequentialBatchGetArgs args{}; args.request.block_count = 1;
  Check(KvafSequentialBatchGetLaunch(&args, reinterpret_cast<void*>(1)) == 0);
  Check(KvafSequentialBatchGetLoadKernel(path.c_str()) == 0 && binaries.size() == 2);
  fs::current_path(original);
  fs::remove_all(directory);
  std::cout << "sequential loader " << scenario << ": PASS\n";
}
