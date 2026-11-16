#pragma once
#include <urma_api.h>
#include <cerrno>
#include <cstring>
#include <utility>
#include <cassert>
#include "urma_memory_binding.h"

namespace kvaf::aicpu_urma::detail {
// This owner lives entirely inside the AICPU session registry. Host control
// messages contain descriptors/session IDs, never this handle or context.
class UmdkRemoteMemoryBinding {
 public:
  UmdkRemoteMemoryBinding() = default;
  UmdkRemoteMemoryBinding(const UmdkRemoteMemoryBinding&) = delete;
  UmdkRemoteMemoryBinding& operator=(const UmdkRemoteMemoryBinding&) = delete;
  UmdkRemoteMemoryBinding(UmdkRemoteMemoryBinding&& other) { *this = std::move(other); }
  UmdkRemoteMemoryBinding& operator=(UmdkRemoteMemoryBinding&& other) {
    assert(!handle_);
    handle_ = std::exchange(other.handle_, nullptr);
    context_ = other.context_; canonical_ = other.canonical_;
    descriptor_ = other.descriptor_; unimport_ = other.unimport_;
    return *this;
  }
  int Import(urma_context_t* context, decltype(&urma_import_seg) import,
             decltype(&urma_unimport_seg) unimport, const urma_seg_t& segment,
             std::uint32_t token, std::uint64_t canonical, std::uint64_t session) {
    if (!context || !import || !unimport || !canonical || !segment.len ||
        segment.len > UINT64_MAX - canonical || segment.len > UINT64_MAX - segment.ubva.va) {
      errno = EINVAL; return -1;
    }
    UrmaRemoteMemoryDesc remote;
    remote.session_id = session; remote.registered_address = segment.ubva.va;
    remote.registered_bytes = segment.len; remote.token_id = segment.token_id;
    remote.token_value = token; remote.key_bytes = sizeof(segment);
    static_assert(sizeof(segment) <= remote.key.size());
    std::memcpy(remote.key.data(), &segment, sizeof(segment));
    std::memcpy(remote.eid.data(), segment.ubva.eid.raw, remote.eid.size());
    if (handle_) {
      if (context != context_ || canonical != canonical_ ||
          remote.token_value != descriptor_.token_value || remote.session_id != descriptor_.session_id ||
          remote.key != descriptor_.key) { errno = EINVAL; return -1; }
      return 0;
    }
    urma_token_t value{}; value.token = token;
    urma_import_seg_flag_t flags{};
    flags.bs.cacheable = URMA_NON_CACHEABLE;
    flags.bs.access = (1U << 1U) | (1U << 2U);
    flags.bs.mapping = URMA_SEG_NOMAP;
    auto copy = segment;
    handle_ = import(context, &copy, &value, 0, flags);
    if (!handle_) return -1;
    context_ = context; canonical_ = canonical; descriptor_ = remote; unimport_ = unimport;
    return 0;
  }
  int Close() {
    if (!handle_) return 0;
    const int result = unimport_(handle_);
    if (result == 0) handle_ = nullptr;
    return result;
  }
  urma_target_seg_t* handle() const { return handle_; }
  std::uint64_t canonical() const { return canonical_; }
  const UrmaRemoteMemoryDesc& descriptor() const { return descriptor_; }
 private:
  urma_context_t* context_{};
  urma_target_seg_t* handle_{};
  decltype(&urma_unimport_seg) unimport_{};
  std::uint64_t canonical_{};
  UrmaRemoteMemoryDesc descriptor_;
};
}  // namespace kvaf::aicpu_urma::detail
