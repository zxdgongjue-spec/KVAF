#ifndef KVAF_AIV_BATCH_GET_SCHEDULE_H_
#define KVAF_AIV_BATCH_GET_SCHEDULE_H_

#include <stdint.h>

// The kernel supplies the AIV qualifier; host tests exercise this same policy.
#ifndef KVAF_BATCH_GET_SCHEDULE_INLINE
#define KVAF_BATCH_GET_SCHEDULE_INLINE inline
#endif

namespace kvaf_batch_get {

constexpr uint32_t kWqesPerPost = 128U;

#ifndef KVAF_BATCH_GET_LEAN_SCHEDULE
#define KVAF_BATCH_GET_LEAN_SCHEDULE 1
#endif
static_assert(KVAF_BATCH_GET_LEAN_SCHEDULE == 0 || KVAF_BATCH_GET_LEAN_SCHEDULE == 1,
              "KVAF_BATCH_GET_LEAN_SCHEDULE must be 0 or 1");
constexpr bool kLeanSchedule = KVAF_BATCH_GET_LEAN_SCHEDULE != 0;

// Defaults retain the historical combined switch. Explicit definitions allow
// either fast path to be measured independently of resource-prefix trimming.
#ifndef KVAF_BATCH_GET_SINGLE_GROUP_FAST_PATH
#define KVAF_BATCH_GET_SINGLE_GROUP_FAST_PATH KVAF_BATCH_GET_LEAN_SCHEDULE
#endif
#ifndef KVAF_BATCH_GET_SINGLE_POST_FAST_PATH
#define KVAF_BATCH_GET_SINGLE_POST_FAST_PATH KVAF_BATCH_GET_LEAN_SCHEDULE
#endif
static_assert(KVAF_BATCH_GET_SINGLE_GROUP_FAST_PATH == 0 ||
              KVAF_BATCH_GET_SINGLE_GROUP_FAST_PATH == 1, "single-group fast path must be 0 or 1");
static_assert(KVAF_BATCH_GET_SINGLE_POST_FAST_PATH == 0 ||
              KVAF_BATCH_GET_SINGLE_POST_FAST_PATH == 1, "single-post fast path must be 0 or 1");
constexpr bool kSingleGroupFastPath = KVAF_BATCH_GET_SINGLE_GROUP_FAST_PATH != 0;
constexpr bool kSinglePostFastPath = KVAF_BATCH_GET_SINGLE_POST_FAST_PATH != 0;

#ifndef KVAF_BATCH_GET_MIN_TAIL_READS
#define KVAF_BATCH_GET_MIN_TAIL_READS 0
#endif
static_assert(KVAF_BATCH_GET_MIN_TAIL_READS >= 0 && KVAF_BATCH_GET_MIN_TAIL_READS <= 128,
              "KVAF_BATCH_GET_MIN_TAIL_READS must be in [0, 128]");
constexpr uint32_t kMinTailReads = KVAF_BATCH_GET_MIN_TAIL_READS;

KVAF_BATCH_GET_SCHEDULE_INLINE uint32_t PostCount(uint32_t count) {
  return count / kWqesPerPost + (count % kWqesPerPost != 0U);
}

KVAF_BATCH_GET_SCHEDULE_INLINE uint32_t CoreBudget(
    uint32_t total, uint32_t cores, uint32_t core) {
  return total / cores + (core < total % cores);
}

// A whole-slice group starts at Jetty 0. Reducing its local scheduling width
// to this prefix preserves baseline routing when min_tail_reads == 0. With a
// threshold, a short final post wraps onto Jetty 0 instead of opening a new
// lane. Full posts still contain 128 READs; the reused lane signals only its
// final post. If posts outnumber configured lanes, the tail already reuses one.
// Windowed groups retain the configured width because later groups rotate.
// This is NOT the stride of the core-major entity/queue resource tables.
KVAF_BATCH_GET_SCHEDULE_INLINE uint32_t ManagedJettys(
    uint32_t core_requests, uint32_t core_budget, uint32_t configured_jettys,
    bool lean = kLeanSchedule, uint32_t min_tail_reads = kMinTailReads) {
  if (!lean) return configured_jettys;
  if (core_requests == 0U) return 0U;
  if (core_budget != 0U) return configured_jettys;
  const uint32_t posts = PostCount(core_requests);
  const uint32_t tail = core_requests % kWqesPerPost;
  if (posts > 1U && posts <= configured_jettys &&
      tail != 0U && tail < min_tail_reads) {
    return posts - 1U;
  }
  return posts < configured_jettys ? posts : configured_jettys;
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
