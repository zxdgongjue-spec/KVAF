#include "aicpu_urma/kernel_loader.h"

#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

std::set<std::string> entries{"KvafAicpuCpuPushScatterExecutionV3"};
unsigned loads = 0, lookups = 0, launches = 0;
int handle;
aclError aclrtBinaryLoadFromFile(const char*, const aclrtBinaryLoadOptions*, aclrtBinHandle* out) {
  ++loads;
  *out = &handle;
  return ACL_SUCCESS;
}
aclError aclrtBinaryUnLoad(aclrtBinHandle) { return ACL_SUCCESS; }
aclError aclrtBinaryGetFunction(aclrtBinHandle binary, const char* name, aclrtFuncHandle* out) {
  CHECK(binary == &handle);
  ++lookups;
  if (!entries.count(name)) return 1;
  *out = &handle;
  return ACL_SUCCESS;
}
aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, std::uint32_t, aclrtStream,
    const aclrtLaunchKernelCfg*, const void*, std::size_t, void*, std::size_t) {
  ++launches;
  return ACL_SUCCESS;
}

int main() {
  using kvaf::aicpu_urma::PrepareUmdkKernel;
  using Kind = kvaf::aicpu_urma::UmdkKernelKind;
  CHECK(!PrepareUmdkKernel("/nonexistent/kvaf/kernel.json", Kind::Put).ok());
  CHECK(loads == 0);
  char path[] = "/tmp/kvaf-loader-XXXXXX";
  int fd = mkstemp(path);
  CHECK(fd >= 0);
  close(fd);
  // Scatter requires neither the control session nor Put/Gather entries.
  CHECK(PrepareUmdkKernel(path, Kind::Scatter).ok());
  CHECK(loads == 1 && lookups == 1 && launches == 0);
  entries.insert("KvafAicpuUmdkControlSessionsV3");
  auto old = PrepareUmdkKernel(path, Kind::Put);
  CHECK(!old.ok() && old.message().find("SessionsV5") != std::string::npos);
  CHECK(launches == 0);
  entries.insert("KvafAicpuUmdkControlSessionsV5");
  entries.insert("KvafAicpuBatchPut");
  CHECK(PrepareUmdkKernel(path, Kind::Put).ok());
  CHECK(loads == 1 && launches == 0);  // Missing Gather does not block Put.
  auto status = PrepareUmdkKernel(path, Kind::Gather);
  CHECK(!status.ok());
  CHECK(status.message().find("KvafAicpuGatherReadExecutionV3") != std::string::npos);
  CHECK(status.message().find("gather_read") != std::string::npos);
  CHECK(launches == 0);  // Missing entry is reported before any device work.
  entries.insert("KvafAicpuGatherReadExecutionV3");
  const auto before_gather = lookups;
  std::vector<std::thread> callers;
  for (int i = 0; i < 8; ++i)
    callers.emplace_back([&] { CHECK(PrepareUmdkKernel(path, Kind::Gather).ok()); });
  for (auto& caller : callers) caller.join();
  CHECK(lookups == before_gather + 1);  // Concurrent first users resolve once.
  const auto prepared_lookups = lookups;
  CHECK(PrepareUmdkKernel(path, Kind::Put).ok());
  CHECK(loads == 1 && lookups == prepared_lookups);
  int argument = 0;
  CHECK(kvaf::aicpu_urma::LaunchUmdkKernel("KvafAicpuGatherReadExecutionV3", &argument,
                                        sizeof(argument), &handle, 4) == 0);
  CHECK(launches == 1 && lookups == prepared_lookups);
  char other[] = "/tmp/kvaf-loader-other-XXXXXX";
  fd = mkstemp(other);
  CHECK(fd >= 0);
  close(fd);
  CHECK(!PrepareUmdkKernel(other, Kind::Put).ok());  // Never silently change the loaded package.
  CHECK(loads == 1);
  unlink(path);
  unlink(other);
  std::cout << "AICPU loader preflight and entry checks passed\n";
}
