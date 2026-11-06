#include "aicpu_urma/kernel_loader.h"

#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <unistd.h>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

std::set<std::string> entries{"KvafAicpuUmdkControlSessionsV2", "KvafAicpuBatchPut", "KvafAicpuGatherRead"};
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
  CHECK(!PrepareUmdkKernel("/nonexistent/kvaf/kernel.json", false).ok());
  CHECK(loads == 0);
  char path[] = "/tmp/kvaf-loader-XXXXXX";
  int fd = mkstemp(path);
  CHECK(fd >= 0);
  close(fd);
  CHECK(PrepareUmdkKernel(path, false).ok());
  CHECK(loads == 1 && lookups == 3 && launches == 0);
  auto status = PrepareUmdkKernel(path, true);
  CHECK(!status.ok());
  CHECK(status.message().find("KvafAicpuCpuPushScatterA12V24") != std::string::npos);
  CHECK(status.message().find("cpu_push_scatter") != std::string::npos);
  CHECK(launches == 0);  // Missing Scatter is reported before any device work.
  entries.insert("KvafAicpuCpuPushScatterA12V24");
  CHECK(PrepareUmdkKernel(path, true).ok());
  const auto prepared_lookups = lookups;
  CHECK(PrepareUmdkKernel(path, false).ok());
  CHECK(loads == 1 && lookups == prepared_lookups);
  int argument = 0;
  CHECK(kvaf::aicpu_urma::LaunchUmdkKernel("KvafAicpuGatherRead", &argument,
                                        sizeof(argument), &handle, 4) == 0);
  CHECK(launches == 1 && lookups == prepared_lookups);
  char other[] = "/tmp/kvaf-loader-other-XXXXXX";
  fd = mkstemp(other);
  CHECK(fd >= 0);
  close(fd);
  CHECK(!PrepareUmdkKernel(other, false).ok());  // Never silently change the loaded package.
  CHECK(loads == 1);
  unlink(path);
  unlink(other);
  std::cout << "AICPU loader preflight and entry checks passed\n";
}
