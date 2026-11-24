#pragma once
#include <cstdio>
#include "direct_hccp/direct_hccp_abi.h"
#include "urma_memory_binding.h"

namespace kvaf::direct_hccp {
class HccpRemoteMemoryBinding {
 public:
  HccpRemoteMemoryBinding() = default;
  HccpRemoteMemoryBinding(const HccpRemoteMemoryBinding&) = delete;
  HccpRemoteMemoryBinding& operator=(const HccpRemoteMemoryBinding&) = delete;
  Status Import(void* context, RmemImportFn import, RmemUnimportFn unimport,
                const UrmaRemoteMemoryDesc& descriptor) {
    if (context_ || handle_ || !context || !import || !unimport || !descriptor.valid() ||
        descriptor.key_bytes > kMemKeyBytes || descriptor.key_bytes > UINT8_MAX)
      return Status::InvalidArgument("invalid HCCP remote memory import");
    context_ = context; unimport_ = unimport; descriptor_ = descriptor;
    MrImportInfo info{};
    std::memcpy(info.in.key.value, descriptor.key.data(), descriptor.key_bytes);
    info.in.key.size = static_cast<std::uint8_t>(descriptor.key_bytes);
    info.in.ub.flags = MemImportFlags();
    info.in.ub.mapping_addr = 0;
    info.in.ub.token_value = descriptor.token_value;
    const auto result = import(context_, &info, &handle_);
    if (result != 0 || !handle_)
      std::fprintf(stderr, "[KVAF_AIV] DIRECT_REMOTE_DDR_IMPORT_FAILED ret=%d handle=%p\n", result, handle_);
    return result == 0 && handle_ ? Status::OK()
        : Status::Internal("HCCP remote DDR import failed: " + std::to_string(result));
  }
  Status Close() {
    if (handle_) {
      const auto result = unimport_(context_, handle_);
      if (result) {
        std::fprintf(stderr, "[KVAF_AIV] DIRECT_REMOTE_DDR_UNIMPORT_FAILED ret=%d handle=%p\n", result, handle_);
        return Status::Internal("HCCP remote DDR unimport failed: " + std::to_string(result));
      }
      handle_ = nullptr;
    }
    context_ = nullptr;
    return Status::OK();
  }
  const UrmaRemoteMemoryDesc& descriptor() const { return descriptor_; }
 private:
  void* context_{};
  void* handle_{};
  RmemUnimportFn unimport_{};
  UrmaRemoteMemoryDesc descriptor_;
};
}  // namespace kvaf::direct_hccp
