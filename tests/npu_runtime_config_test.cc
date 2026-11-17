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
  CHECK(defaults.batch_get_token_bytes == 656);
  CHECK(defaults.batch_get_tokens_per_block == 2);
  CHECK(defaults.transport_mode == TransportMode::Urma);
  CHECK(!defaults.default_put && !defaults.default_get);
  CHECK(defaults.put_resources.empty() && defaults.get_resources.empty());

  std::vector<NpuEid> candidates(3), selected;
  for (std::size_t i = 0; i < candidates.size(); ++i) candidates[i][0] = i + 1;
  ExecutionResources budget;
  budget.cores = 2;
  budget.jettys = 2;
  // EID routing depends on the selected provider and its resolved worker budget.
  CHECK(SelectProviderEids(Transport::Urma, Executor::Aiv, true, true,
                           budget, candidates, &selected).ok());
  CHECK(selected == candidates);
  budget.cores = 1;
  CHECK(!SelectProviderEids(Transport::Urma, Executor::Aiv, true, true,
                            budget, candidates, &selected).ok());
  CHECK(selected == candidates);  // A failing budget check does not truncate the list.
  for (const auto executor : {Executor::Aiv, Executor::Aicpu}) {
    for (const bool get : {false, true}) {
      if (executor == Executor::Aiv && get) continue;
      CHECK(SelectProviderEids(Transport::Urma, executor, get, true,
                               budget, candidates, &selected).ok());
      CHECK(selected.size() == 1 && selected.front() == candidates.front());
    }
  }
  CHECK(SelectProviderEids(Transport::Urma, Executor::Aiv, true, false,
                           budget, candidates, &selected).ok());
  CHECK(selected.size() == 1 && selected.front() == candidates.front());
  CHECK(SelectProviderEids(Transport::Ubmem, Executor::Aiv, true, true,
                           budget, {}, &selected).ok() && selected.empty());
  CHECK(!SelectProviderEids(Transport::Urma, Executor::Aicpu, true, true,
                            budget, {}, &selected).ok());
  CHECK(!SelectProviderEids(Transport::Urma, Executor::Aiv, true, true,
                            budget, candidates, nullptr).ok());

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
  std::cout << "Provider configuration and installed AICPU package resolution passed\n";
}
