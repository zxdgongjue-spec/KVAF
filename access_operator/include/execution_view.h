#pragma once

#include <array>
#include <atomic>
#include "execution_resources.h"
#include "cache_region.h"

namespace kvaf {
// A view borrows storage protected by the caller's execution lease. Stamps
// detect rebinding/reinitialization; they do not extend the storage lifetime.
enum class ExecutionDirection { Put, Get };
// Failure belongs to the physical resource, and survives invocation rebinding.
struct ExecutionHealth { bool failed{}; };
class ExecutionViewState;
class ExecutionViewCommon {
 public:
  ExecutionViewCommon() = default;
  Status Validate(const ExecutionState*, ExecutionDirection) const;
  bool ready() const;
  const CacheRegionDesc& region() const { return region_; }
 private:
  friend class ExecutionViewState;
  const ExecutionViewState* state_{};
  std::uint64_t resource_{}, generation_{}, binding_{}, preparation_{};
  ExecutionDirection direction_{ExecutionDirection::Get};
  CacheRegionDesc region_{};
};

// Mutated only under the provider's direction lease/lock. No driver calls or
// allocation occur when checking a view on the launch path.
class ExecutionViewState {
 public:
  ExecutionViewState() : resource_(next_.fetch_add(1, std::memory_order_relaxed)) {}
  ExecutionViewState(const ExecutionViewState&) = delete;
  ExecutionViewState& operator=(const ExecutionViewState&) = delete;
  void Initialize(const RemoteRegionDesc& region, std::uint64_t region_id = 0, std::uint64_t epoch = 0) {
    ++generation_;
    // Legacy providers receive only the canonical range. Server identity stays
    // unknown here; local lifetime stamps must never masquerade as wire epochs.
    region_ = {region_id, epoch, region.base, region.length, region.slot_size};
    ready_ = true;
  }
  void Close() { ready_ = false; ++generation_; }
  void Bind(const ExecutionState* state) {
    if (active_ != state) { active_ = state; ++binding_; }
  }
  void Prepare(ExecutionDirection direction) { ++preparation_[static_cast<unsigned>(direction)]; }
  void Prepare() { for (auto& sequence : preparation_) ++sequence; }
  ExecutionViewCommon View(ExecutionDirection direction) const {
    ExecutionViewCommon view;
    view.state_ = this;
    view.resource_ = resource_; view.generation_ = generation_;
    view.binding_ = binding_; view.preparation_ = preparation_[static_cast<unsigned>(direction)];
    view.direction_ = direction; view.region_ = region_;
    return view;
  }
 private:
  friend class ExecutionViewCommon;
  inline static std::atomic<std::uint64_t> next_{1};
  const std::uint64_t resource_;
  std::uint64_t generation_{0}, binding_{0};
  std::array<std::uint64_t, 2> preparation_{};
  const ExecutionState* active_{};
  CacheRegionDesc region_{};
  bool ready_{false};
};
inline bool ExecutionViewCommon::ready() const {
  return state_ && state_->ready_ && resource_ == state_->resource_ &&
      generation_ == state_->generation_ && binding_ == state_->binding_ &&
      preparation_ == state_->preparation_[static_cast<unsigned>(direction_)];
}
inline Status ExecutionViewCommon::Validate(const ExecutionState* call, ExecutionDirection direction) const {
  return ready() && call && state_->active_ == call && direction_ == direction
      ? Status::OK() : Status::InvalidState("expired or mismatched execution resource view");
}
}  // namespace kvaf
