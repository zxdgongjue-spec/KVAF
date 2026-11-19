#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include "status.h"

namespace kvaf {
// A description, not a driver handle. Raw UMDK tokens stay in wire format;
// conversion to the AIV WQE encoding happens only at the AIV boundary.
struct UrmaRemoteMemoryDesc {
  std::uint64_t epoch{}, region_id{}, session_id{};
  std::uint64_t registered_address{}, registered_bytes{};
  std::uint32_t token_id{}, token_value{}, key_bytes{};
  std::array<std::uint8_t, 16> eid{};
  std::array<std::uint8_t, 256> key{};
  std::uint32_t wqe_token_id() const { return token_id >> 8U; }
  bool valid() const {
    return registered_address && registered_bytes && registered_bytes <= UINT64_MAX - registered_address &&
        token_value && key_bytes && key_bytes <= key.size();
  }
};
inline std::uint64_t NewBindingGeneration() {
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}
enum class MemoryDirection { Put, Get };
struct UrmaBindingKey {
  MemoryDirection direction{};
  std::uintptr_t context{};
  std::uint64_t context_generation{}, epoch{}, region{}, session{};
  std::uint64_t base{}, bytes{};
  std::uint32_t token_id{}, token{}, permissions{}, key_bytes{};
  std::array<std::uint8_t, 16> eid{};
  std::array<std::uint8_t, 256> key{};
  auto fields() const {
    return std::tie(direction, context, context_generation, epoch, region, session,
                    base, bytes, token_id, token, permissions, key_bytes, eid, key);
  }
  bool operator<(const UrmaBindingKey& b) const { return fields() < b.fields(); }
  static UrmaBindingKey Make(MemoryDirection d, void* context, std::uint64_t generation,
                            const UrmaRemoteMemoryDesc& remote, std::uint32_t permissions) {
    return {d, reinterpret_cast<std::uintptr_t>(context), generation,
            remote.epoch, remote.region_id, remote.session_id,
            remote.registered_address, remote.registered_bytes, remote.token_id,
            remote.token_value, permissions, remote.key_bytes, remote.eid, remote.key};
  }
};

// Provider type is part of the cache identity. Owners must Release all leases
// before destroying the context, its API library or the remote session. A failed
// Release retains the lease and blocks reuse, allowing the owner to retry Close.
template<class Binding>
class UrmaBindingCache {
 public:
  explicit UrmaBindingCache(MemoryDirection direction) : direction_(direction) {}
  template<class Create>
  Status Acquire(const UrmaBindingKey& key, Create create, std::shared_ptr<Binding>* out) {
    if (!out || *out || key.direction != direction_ || !key.context || !key.context_generation)
      return Status::InvalidArgument("invalid URMA binding identity or occupied lease");
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(key);
    const bool inserted = it == entries_.end();
    if (inserted) {
      // Allocate before publishing the entry; allocation failure leaves no
      // unusable null binding in the index.
      Entry entry;
      entry.binding = std::make_shared<Binding>();
      it = entries_.emplace(key, std::move(entry)).first;
    }
    auto& entry = it->second;
    if (inserted) {
      entry.failure = Status::Internal("incomplete URMA remote memory import");
      try { entry.failure = create(*entry.binding); }
      catch (...) { /* Preserve partial resources and the failed lease below. */ }
    }
    ++entry.users;
    *out = entry.binding;  // Keep even partially imported handles on failure.
    return entry.failure;
  }
  Status Release(std::shared_ptr<Binding>* lease) {
    if (!lease || !*lease) return Status::OK();
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto& entry = it->second;
      if (entry.binding != *lease) continue;
      if (entry.users == 1) {
        auto s = entry.binding->Close();
        if (!s.ok()) { entry.failure = s; return s; }
        entries_.erase(it);
      } else --entry.users;
      lease->reset();
      return Status::OK();
    }
    return Status::InvalidState("URMA binding belongs to another resource domain");
  }
 private:
  struct Entry { std::shared_ptr<Binding> binding; std::size_t users{}; Status failure; };
  MemoryDirection direction_;
  std::mutex mutex_;
  std::map<UrmaBindingKey, Entry> entries_;
};
}  // namespace kvaf
