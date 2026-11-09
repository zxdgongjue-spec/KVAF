#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include "gather_shard_layout.h"

namespace kvaf::peer {
// Single coordinator owns both registered snapshots and all transport calls.
// Helpers only publish atomic DDR progress. A live snapshot is immutable until
// its matching CQ arrives, including partial-post failures.
class GatherProgressPump {
 public:
  struct Ops {
    void* context;
    bool (*post)(void*, uint32_t slot, uint64_t id, uint32_t bytes);
    int (*poll)(void*, uint64_t* id); // 0 pending, 1 completion, -1 error
  };
  GatherProgressPump(uint64_t* frames, Ops ops) : frames_(frames), ops_(ops) {}
  GatherProgressPump(const GatherProgressPump&) = delete;
  GatherProgressPump& operator=(const GatherProgressPump&) = delete;
  bool Advance(uint64_t sequence, const uint64_t* progress, uint32_t shards) {
    if (failed_ || !sequence || sequence > UINT32_MAX || !progress || !shards || shards > kGatherMaxShards)
      return Fail();
    if (!Reap()) return false;
    if (sequence != sequence_) {
      if (Pending()) return Fail();
      sequence_ = sequence;
      std::memset(last_, 0, sizeof(last_));
    }
    uint32_t slot = 0;
    while (slot < 2 && live_[slot]) ++slot;
    if (slot == 2) { latest_ = false; ++busy_checks_; return true; }
    auto* frame = frames_ + slot * kGatherMaxShards;
    bool changed = false;
    for (uint32_t s = 0; s < shards; ++s) {
      const auto word = __atomic_load_n(progress + s, __ATOMIC_ACQUIRE);
      const auto generation = word >> 32;
      if (generation > sequence || (generation == sequence && last_[s] > word)) return Fail();
      frame[s] = generation == sequence && uint32_t(word) ? word : 0;
      changed |= frame[s] != last_[s];
    }
    latest_ = true;
    if (!changed) return true;
    live_[slot] = true;
    ids_[slot] = ++next_id_;
    // The backend must preserve placement order between these same-address
    // snapshots. One strong ordered WRITE per snapshot needs no second WR.
    if (!ops_.post(ops_.context, slot, ids_[slot], shards * 8)) return Fail();
    std::memcpy(last_, frame, shards * 8);
    ++posts_;
    return true;
  }
  bool Reap() {
    if (failed_) return false;
    for (uint32_t n = 0; n < 2 && Pending(); ++n) {
      uint64_t id = 0;
      const auto result = ops_.poll(ops_.context, &id);
      if (result < 0) return Fail();
      if (!result) break;
      uint32_t slot = 0;
      while (slot < 2 && (!live_[slot] || ids_[slot] != id)) ++slot;
      if (slot == 2) return Fail();
      live_[slot] = false;
    }
    return true;
  }
  bool Pending() const { return live_[0] || live_[1]; }
  bool Latest() const { return latest_; }
  bool ok() const { return !failed_; }
  uint64_t posts() const { return posts_; }
  uint64_t busy_checks() const { return busy_checks_; }
 private:
  bool Fail() { failed_ = true; return false; }
  uint64_t* frames_;
  Ops ops_;
  uint64_t last_[kGatherMaxShards]{}, ids_[2]{}, sequence_{}, next_id_{}, posts_{}, busy_checks_{};
  bool live_[2]{}, latest_{}, failed_{};
};
}  // namespace kvaf::peer
