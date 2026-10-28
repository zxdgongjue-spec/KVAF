#ifndef KVAF_AIV_BATCH_GET_SCHEDULE_H_
#define KVAF_AIV_BATCH_GET_SCHEDULE_H_

#include <stdint.h>

// The kernel supplies the AIV qualifier; host tests exercise this same policy.
#ifndef KVAF_BATCH_GET_SCHEDULE_INLINE
#define KVAF_BATCH_GET_SCHEDULE_INLINE inline
#endif

namespace kvaf_batch_get {

constexpr uint32_t kWqesPerPost = 128U;

#ifndef KVAF_BATCH_GET_SMALL_POST_MAX_REQUESTS
#define KVAF_BATCH_GET_SMALL_POST_MAX_REQUESTS 0
#endif

// This compile-time experiment changes scheduling only. UB bank capacity,
// SIMT width and the public kernel ABI remain unchanged.
constexpr uint32_t kSmallPostMaxRequests =
    KVAF_BATCH_GET_SMALL_POST_MAX_REQUESTS;

KVAF_BATCH_GET_SCHEDULE_INLINE uint32_t PostCount(
    uint32_t count, uint32_t post_width = kWqesPerPost) {
  return count / post_width + (count % post_width != 0U);
}

KVAF_BATCH_GET_SCHEDULE_INLINE uint32_t SelectPostWidth(
    uint32_t total_requests, uint32_t core_requests, uint32_t jettys,
    uint32_t small_post_max_requests = kSmallPostMaxRequests) {
  if (small_post_max_requests == 0U ||
      total_requests > small_post_max_requests || core_requests == 0U ||
      jettys <= 1U) {
    return kWqesPerPost;
  }
  // Use roughly one post per Jetty on short slices, without padding READs.
  // A tiny/uneven slice can use fewer Jettys than configured.
  const uint32_t width = core_requests / jettys +
      (core_requests % jettys != 0U);
  return width < kWqesPerPost ? width : kWqesPerPost;
}

KVAF_BATCH_GET_SCHEDULE_INLINE uint32_t CoreBudget(
    uint32_t total, uint32_t cores, uint32_t core) {
  return total / cores + (core < total % cores);
}

struct Group {
  uint32_t begin;
  uint32_t count;
  uint32_t first_jetty;
  uint32_t post_width = kWqesPerPost;
};

KVAF_BATCH_GET_SCHEDULE_INLINE uint32_t AssignedRequests(
    const Group &group, uint32_t jetty, uint32_t jettys) {
  const uint32_t relative = (jetty + jettys - group.first_jetty) % jettys;
  const uint32_t full_posts = group.count / group.post_width;
  return (full_posts / jettys + (relative < full_posts % jettys)) *
             group.post_width +
         (relative == full_posts % jettys ? group.count % group.post_width : 0U);
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
  uint32_t post_width = kWqesPerPost;

  KVAF_BATCH_GET_SCHEDULE_INLINE bool CanPost() const {
    return cursor < end && pending < 2U;
  }

  KVAF_BATCH_GET_SCHEDULE_INLINE Group Next() const {
    const uint32_t slot = (oldest + pending) % 2U;
    const uint32_t capacity =
        budget == 0U ? end - cursor : budget / 2U + slot * (budget % 2U);
    const uint32_t remaining = end - cursor;
    return Group{cursor, remaining < capacity ? remaining : capacity,
                 next_jetty, post_width};
  }

  KVAF_BATCH_GET_SCHEDULE_INLINE void Posted(const Group &group) {
    groups[(oldest + pending) % 2U] = group;
    ++pending;
    cursor += group.count;
    next_jetty = (next_jetty + PostCount(group.count, group.post_width)) % jettys;
  }

  KVAF_BATCH_GET_SCHEDULE_INLINE void Completed() {
    oldest = (oldest + 1U) % 2U;
    --pending;
  }
};

}  // namespace kvaf_batch_get

#endif  // KVAF_AIV_BATCH_GET_SCHEDULE_H_
