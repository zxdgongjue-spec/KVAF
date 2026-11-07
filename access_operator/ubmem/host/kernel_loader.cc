#include "kernel_loader.h"

#include <fstream>
#include <cstring>

#include "ubmem/transfer_abi.h"

namespace kvaf::ubmem {

Status Kernel::Load(const std::string& path, const char* name) {
  if (binary_ != nullptr) return Status::InvalidState("UBmem kernel already loaded");
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() <= 0) return Status::InvalidArgument("cannot read UBmem kernel: " + path);
  local_memory_bytes_ = std::strcmp(name, "KvafUbmemBatchGet") == 0
      ? kvaf_ubmem::kGetLocalMemoryBytes : kvaf_ubmem::kLocalMemoryBytes;
  const auto size = file.tellg();
  bytes_.resize(static_cast<size_t>(size));
  file.seekg(0);
  if (!file.read(bytes_.data(), size)) return Status::Internal("cannot read full UBmem kernel: " + path);
  aclrtBinary binary = aclrtCreateBinary(bytes_.data(), bytes_.size());
  if (binary == nullptr) return Status::Internal("aclrtCreateBinary failed for UBmem kernel");
  aclError error = aclrtBinaryLoad(binary, &binary_);
  aclrtDestroyBinary(binary);
  if (error == ACL_SUCCESS) error = aclrtBinaryGetFunction(binary_, name, &function_);
  if (error != ACL_SUCCESS) {
    (void)Unload();
    return Status::Internal("loading UBmem kernel failed: ACL=" + std::to_string(error));
  }
  return Status::OK();
}

Status Kernel::Unload() {
  if (binary_ != nullptr) {
    const aclError error = aclrtBinaryUnLoad(binary_);
    if (error != ACL_SUCCESS) return Status::Internal("unloading UBmem kernel failed: ACL=" + std::to_string(error));
  }
  binary_ = nullptr;
  function_ = nullptr;
  bytes_.clear();
  return Status::OK();
}

aclError Kernel::Launch(const void* arguments, size_t bytes, uint32_t workers,
                        aclrtStream stream) const {
  aclrtLaunchKernelAttr attrs[4]{};
  attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  attrs[0].value.schemMode = 1;
  attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  attrs[1].value.dynUBufSize = local_memory_bytes_;
  attrs[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
  attrs[2].value.timeoutUs.timeoutLow = 180U * 1000000U;
  attrs[2].value.timeoutUs.timeoutHigh = 0;
  attrs[3].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  attrs[3].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
  aclrtLaunchKernelCfg config{};
  config.numAttrs = 4;
  config.attrs = attrs;
  return aclrtLaunchKernelWithHostArgs(function_, workers, stream, &config,
      const_cast<void*>(arguments), bytes, nullptr, 0);
}

}  // namespace kvaf::ubmem
