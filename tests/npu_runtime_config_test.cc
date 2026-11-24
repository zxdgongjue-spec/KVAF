#include "npu_runtime.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

namespace fs = std::filesystem;
using namespace kvaf;
using namespace kvaf::detail;

const std::string registration = "name:kvaf-aicpu-urma.tar.gz\ninstall_path:2\noptional:true\n"
    "package_path:opp/vendors/cust/op_impl/aicpu/kernel\nload_as_per_soc:false\n";
const char* config_path = "conf/ascend_package_load.ini";
const char* json_path = "opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json";
const char* tar_path = "opp/vendors/cust/op_impl/aicpu/kernel/kvaf-aicpu-urma.tar.gz";

void Write(const fs::path& path, const std::string& contents) {
  fs::create_directories(path.parent_path());
  std::ofstream file(path);
  file << contents;
  CHECK(file.good());
}

int main() {
  NpuRuntimeOptions defaults{656, 2};
  CHECK(NormalizeExecutionPolicy(&defaults).ok());
  CHECK((defaults.default_get == GetExecutionConfig{Transport::Urma, Executor::Aiv, GetAlgorithm::Direct}));
  for (const auto algorithm : {GetAlgorithm::Gather, GetAlgorithm::Scatter}) {
    for (const auto executor : {Executor::Aiv, Executor::Aicpu}) {
      CHECK(ValidateExecution(TransportMode::Urma, GetExecutionConfig{Transport::Urma, executor, algorithm}).ok());
      CHECK(!ValidateExecution(TransportMode::Ubmem, GetExecutionConfig{Transport::Ubmem, executor, algorithm}).ok());
    }
  }

  char pattern[] = "/tmp/kvaf-config-XXXXXX";
  const auto* directory = mkdtemp(pattern);
  CHECK(directory);
  const fs::path temp(directory), root = temp / "cann", alternate = temp / "alternate";
  for (const auto& path : {root, alternate}) {
    Write(path / config_path, registration);
    Write(path / json_path, "{}");
    Write(path / tar_path, "archive checked by installer");
  }
  AicpuKernelPackage package;
  AicpuPackageSearch search;
  search.default_roots = {(temp / "absent").string(), root.string(), alternate.string()};
  CHECK(ResolveAicpuKernelPackage(search, &package).ok());
  CHECK(package.cann_root == fs::canonical(root).string());
  CHECK(package.json_path == fs::canonical(root / json_path).string());
  search.ascend_cann_package_path = root.string();
  CHECK(ResolveAicpuKernelPackage(search, &package).ok());
  fs::create_directory_symlink(root, temp / "latest");
  search.ascend_home_path = (temp / "latest").string();
  CHECK(ResolveAicpuKernelPackage(search, &package).ok());
  search.ascend_cann_package_path = alternate.string();
  CHECK(ResolveAicpuKernelPackage(search, &package).message().find("roots differ") != std::string::npos);
  search.ascend_cann_package_path.clear();
  CHECK(ResolveAicpuKernelPackage(search, &package).ok());
  search.ascend_home_path = (temp / "missing").string();
  CHECK(!ResolveAicpuKernelPackage(search, &package).ok());  // No fallback for explicit paths.
  search.ascend_home_path = root.string();
  Write(root / json_path, "");
  CHECK(!ResolveAicpuKernelPackage(search, &package).ok());
  search.ascend_home_path.clear();
  CHECK(!ResolveAicpuKernelPackage(search, &package).ok());  // Do not switch installs for a missing package.
  Write(root / json_path, "{}");
  Write(root / tar_path, "");
  CHECK(!ResolveAicpuKernelPackage(search, &package).ok());
  Write(root / tar_path, "package");
  Write(root / config_path, "unrelated package\n");
  CHECK(!ResolveAicpuKernelPackage(search, &package).ok());
  Write(root / config_path, registration + registration);
  CHECK(!ResolveAicpuKernelPackage(search, &package).ok());
  Write(root / config_path, "name:kvaf-aicpu-urma.tar.gz\ninstall_path:3\n");
  CHECK(!ResolveAicpuKernelPackage(search, &package).ok());
  Write(root / config_path, registration);
  CHECK(ResolveAicpuKernelPackage(search, &package).ok());
  search.default_roots = {(temp / "missing").string()};
  CHECK(!ResolveAicpuKernelPackage(search, &package).ok());
  fs::remove_all(temp);
  std::cout << "Backend selection and installed AICPU package resolution passed\n";
}
