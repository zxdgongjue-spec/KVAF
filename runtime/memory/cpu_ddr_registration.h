#pragma once

#include <urma_api.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include "status.h"
#include "urma_memory_binding.h"

namespace kvaf {
// Owns the MR and token only. The region owns DDR and the resource pool owns
// context/API lifetime. Both must outlive a successful Close, including retries.
class CpuDdrRegistration {
 public:
  CpuDdrRegistration() = default;
  CpuDdrRegistration(const CpuDdrRegistration&) = delete;
  CpuDdrRegistration& operator=(const CpuDdrRegistration&) = delete;
  static urma_reg_seg_flag_t Flags() {
    urma_reg_seg_flag_t flag{};
    flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    flag.bs.cacheable = URMA_NON_CACHEABLE;
    // Preserve the deployed DDR contract. LOCAL_WRITE is rejected by this
    // driver; remote atomic and non-pinned registration are required.
    flag.bs.access = (1U << 1U) | (1U << 2U) | (1U << 3U);
    flag.bs.non_pin = 1U;
    flag.bs.token_id_valid = 1U;
    return flag;
  }
  Status Register(urma_context_t* context, void* memory, std::uint64_t bytes,
                  std::uint32_t token_value) {
    if (token_ || segment_ || !context || !memory || !bytes || !token_value)
      return Status::InvalidArgument("invalid CPU DDR registration");
    token_value_ = token_value;
    token_ = urma_alloc_token_id(context);
    if (!token_) return Error("urma_alloc_token_id");
    urma_seg_cfg_t cfg{};
    cfg.va = reinterpret_cast<std::uint64_t>(memory);
    cfg.len = bytes;
    cfg.token_id = token_;
    cfg.token_value.token = token_value;
    cfg.flag = Flags();
    segment_ = urma_register_seg(context, &cfg);
    return segment_ ? Status::OK() : Error("urma_register_seg");
  }
  Status Close() {
    if (segment_) {
      if (urma_unregister_seg(segment_) != URMA_SUCCESS) return Error("urma_unregister_seg");
      segment_ = nullptr;
    }
    if (token_) {
      if (urma_free_token_id(token_) != URMA_SUCCESS) return Error("urma_free_token_id");
      token_ = nullptr;
    }
    return Status::OK();
  }
  urma_target_seg_t* segment() const { return segment_; }
  std::uint32_t token_id() const { return token_ ? token_->token_id : 0; }
  std::uint32_t token_value() const { return token_value_; }
  UrmaRemoteMemoryDesc Describe(std::uint64_t session) const {
    UrmaRemoteMemoryDesc result;
    if (!segment_ || !token_) return result;
    result.session_id = session;
    result.registered_address = segment_->seg.ubva.va;
    result.registered_bytes = segment_->seg.len;
    result.token_id = token_id(); result.token_value = token_value_;
    result.key_bytes = sizeof(segment_->seg);
    static_assert(sizeof(segment_->seg) <= result.key.size());
    std::memcpy(result.key.data(), &segment_->seg, result.key_bytes);
    std::memcpy(result.eid.data(), segment_->seg.ubva.eid.raw, result.eid.size());
    return result;
  }
 private:
  static Status Error(const char* operation) {
    const int error = errno;
    return Status::Internal(std::string(operation) + " failed; errno=" + std::to_string(error));
  }
  urma_token_id_t* token_{};
  urma_target_seg_t* segment_{};
  std::uint32_t token_value_{};
};
}  // namespace kvaf
