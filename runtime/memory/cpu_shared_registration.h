#pragma once

#include "memory/cpu_ddr_registration.h"
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

namespace kvaf {
// One instance belongs to one allocation generation. Leases are released only
// after their execution queues and producer have stopped. The allocation owner
// must Close this pool before destroying its contexts or backing DDR.
class CpuDdrRegistrationPool {
 public:
  struct Key {
    std::uint64_t epoch{}, allocation{}, allocation_generation{}, context_generation{};
    std::uintptr_t context{}, base{};
    std::uint64_t bytes{};
    std::uint32_t candidate{}, flags{}, token{};
    std::array<std::uint8_t, 16> eid{};
    auto Fields() const {
      return std::tie(epoch, allocation, allocation_generation, context_generation, context, base, bytes,
                      candidate, flags, token, eid);
    }
    bool operator<(const Key& other) const { return Fields() < other.Fields(); }
  };
  Status Acquire(const Key& key, std::shared_ptr<CpuDdrRegistration>* lease) {
    if (!lease || *lease || !key.context || !key.context_generation || !key.allocation_generation || !key.base || !key.bytes ||
        key.bytes > UINT64_MAX - key.base || !key.token || key.flags != CpuDdrRegistration::Flags().value)
      return Status::InvalidArgument("invalid shared CPU DDR registration identity");
    std::shared_ptr<Entry> entry;
    {
      std::lock_guard<std::mutex> index(index_);
      if (closing_) return Status::InvalidState("CPU DDR registration pool is closing");
      auto it = entries_.find(key);
      if (it == entries_.end()) it = entries_.emplace(key, std::make_shared<Entry>()).first;
      entry = it->second;
      // Publish the lease before dropping the index lock. Close cannot retire
      // this entry while its first creator is waiting on the setup mutex.
      *lease = entry->registration;
    }
    std::lock_guard<std::mutex> create(entry->mutex);
    if (!entry->attempted) {
      entry->attempted = true;
      entry->status = entry->registration->Register(reinterpret_cast<urma_context_t*>(key.context),
          reinterpret_cast<void*>(key.base), key.bytes, key.token);
    }
    // Keep partial token/MR ownership on errors. A different execution session
    // must never recreate a failed binding under the same allocation identity.
    *lease = entry->registration;
    return entry->status;
  }
  Status Close() {
    std::vector<std::pair<Key, std::shared_ptr<Entry>>> entries;
    {
      std::lock_guard<std::mutex> index(index_);
      closing_ = true;
      entries.assign(entries_.begin(), entries_.end());
    }
    for (const auto& item : entries) {
      std::lock_guard<std::mutex> entry(item.second->mutex);
      if (item.second->registration.use_count() != 1)
        return Status::Busy("CPU DDR registration still has execution leases");
    }
    for (const auto& item : entries) {
      std::lock_guard<std::mutex> entry(item.second->mutex);
      auto status = item.second->registration->Close();
      if (!status.ok()) return status;
      std::lock_guard<std::mutex> index(index_);
      entries_.erase(item.first);
    }
    return Status::OK();
  }
 private:
  struct Entry {
    std::mutex mutex;
    bool attempted{false};
    Status status{Status::Internal("shared CPU DDR registration creation is incomplete")};
    std::shared_ptr<CpuDdrRegistration> registration{std::make_shared<CpuDdrRegistration>()};
  };
  bool closing_{false};
  std::mutex index_;
  std::map<Key, std::shared_ptr<Entry>> entries_;
};
}  // namespace kvaf
