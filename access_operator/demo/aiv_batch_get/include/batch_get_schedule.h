#ifndef KVAF_AIV_BATCH_GET_SCHEDULE_H_
#define KVAF_AIV_BATCH_GET_SCHEDULE_H_

#include <stdint.h>

// The kernel supplies the AIV qualifier; host tests exercise this same policy.
#ifndef KVAF_BATCH_GET_SCHEDULE_INLINE
#define KVAF_BATCH_GET_SCHEDULE_INLINE inline
#endif

namespace kvaf_batch_get {

constexpr uint32_t kWqesPerPost = 128U;

KVAF_BATCH_GET_SCHEDULE_INLINE uint32_t PostCount(uint32_t count) {
  return count / kWqesPerPost + (count % kWqesPerPost != 0U);
}

KVAF_BATCH_GET_SCHEDULE_INLINE uint32_t CoreBudget(
    uint32_t total, uint32_t cores, uint32_t core) {
  return total / cores + (core < total % cores);
}

struct Group {
  uint32_t begin;
  uint32_t count;
  uint32_t first_jetty;
};

KVAF_BATCH_GET_SCHEDULE_INLINE uint32_t AssignedRequests(
    const Group &group, uint32_t jetty, uint32_t jettys) {
  const uint32_t relative = (jetty + jettys - group.first_jetty) % jettys;
  const uint32_t full_posts = group.count / kWqesPerPost;
  return (full_posts / jettys + (relative < full_posts % jettys)) *
             kWqesPerPost +
         (relative == full_posts % jettys ? group.count % kWqesPerPost : 0U);
}

// Two FIFO slots. A slot is reused only after ALL of its Jetty checkpoints
// succeed. Odd budgets alternate floor/ceil halves, so any two adjacent
// groups fit. budget == 0 retains a single, whole-slice submission.
struct Schedule {
  Group groups[2];
  uint32_t cursor;
  uint32_t end;
  uint32_t budget;
  uint32_t jettys;
  uint32_t next_jetty;
  uint32_t oldest;
  uint32_t pending;

  KVAF_BATCH_GET_SCHEDULE_INLINE bool CanPost() const {
    return cursor < end && pending < 2U;
  }

  KVAF_BATCH_GET_SCHEDULE_INLINE Group Next() const {
    const uint32_t slot = (oldest + pending) % 2U;
    const uint32_t capacity =
        budget == 0U ? end - cursor : budget / 2U + slot * (budget % 2U);
    const uint32_t remaining = end - cursor;
    return Group{cursor, remaining < capacity ? remaining : capacity, next_jetty};
  }

  KVAF_BATCH_GET_SCHEDULE_INLINE void Posted(const Group &group) {
    groups[(oldest + pending) % 2U] = group;
    ++pending;
    cursor += group.count;
    next_jetty = (next_jetty + PostCount(group.count)) % jettys;
  }

  KVAF_BATCH_GET_SCHEDULE_INLINE void Completed() {
    oldest = (oldest + 1U) % 2U;
    --pending;
  }
};

}  // namespace kvaf_batch_get

#endif  // KVAF_AIV_BATCH_GET_SCHEDULE_H_
