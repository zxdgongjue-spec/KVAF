#include "batch_get_abi.h"
#include "batch_get_schedule.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

using kvaf_batch_get::Group;
using kvaf_batch_get::Schedule;

// Independent per-post reference, including non-power-of-two Jetty counts.
void CheckAssignment(const Group& group, uint32_t jettys) {
  std::array<uint32_t, 64> expected{};
  uint32_t jetty = group.first_jetty;
  for (uint32_t remaining = group.count; remaining != 0;) {
    const uint32_t count = std::min(remaining, group.post_width);
    expected[jetty] += count;
    remaining -= count;
    jetty = (jetty + 1U) % jettys;
  }
  for (uint32_t j = 0; j < jettys; ++j) {
    CHECK(kvaf_batch_get::AssignedRequests(group, j, jettys) == expected[j]);
  }
}

void CheckLaunch(uint32_t misses, uint32_t cores, uint32_t jettys,
                 uint32_t budget, uint32_t small_post_limit = 0U) {
  std::vector<Schedule> schedules;
  std::vector<uint64_t> outstanding(cores), retired(cores);
  uint64_t allocated = 0;
  for (uint32_t c = 0; c < cores; ++c) {
    const uint32_t local_budget = kvaf_batch_get::CoreBudget(budget, cores, c);
    const uint32_t begin = static_cast<uint32_t>(uint64_t(misses) * c / cores);
    const uint32_t end = static_cast<uint32_t>(uint64_t(misses) * (c + 1) / cores);
    allocated += local_budget;
    schedules.push_back(Schedule{
        {}, begin, end, budget && misses > budget ? local_budget : 0U,
        jettys, 0, 0, 0,
        kvaf_batch_get::SelectPostWidth(misses, end - begin, jettys,
                                        small_post_limit)});
  }
  CHECK(allocated == budget);
  uint64_t total_outstanding = 0, total_retired = 0;
  uint32_t random = 1234567;
  // Advance cores independently, deliberately allowing some to run far ahead.
  while (total_retired < misses) {
    random = random * 1664525U + 1013904223U;
    const uint32_t core = (random >> 16U) % cores;
    auto& schedule = schedules[core];
    if (schedule.CanPost()) {
      const Group group = schedule.Next();
      CHECK(group.count != 0);
      CHECK(group.begin == schedule.cursor);
      CHECK(group.post_width >= 1 && group.post_width <= 128);
      CheckAssignment(group, jettys);
      schedule.Posted(group);
      outstanding[core] += group.count;
      total_outstanding += group.count;
      CHECK(schedule.pending <= 2);
      if (schedule.budget) CHECK(outstanding[core] <= schedule.budget);
    } else if (schedule.pending) {
      const Group group = schedule.groups[schedule.oldest];
      outstanding[core] -= group.count;
      total_outstanding -= group.count;
      retired[core] += group.count;
      total_retired += group.count;
      schedule.Completed();
    }
    if (budget) CHECK(total_outstanding <= budget);
  }
  CHECK(total_outstanding == 0);
  for (uint32_t c = 0; c < cores; ++c) {
    CHECK(schedules[c].cursor == schedules[c].end);
    CHECK(schedules[c].pending == 0);
    CHECK(retired[c] == uint64_t(misses) * (c + 1) / cores -
                           uint64_t(misses) * c / cores);
  }
}

int main() {
  static_assert(offsetof(KvafExecuteBatchGetKernelArgs,
                        batch_get_max_inflight_total) == 108);
  static_assert(sizeof(KvafBatchGetCoreStatus) == 1216);
  Schedule tail{{}, 0, 2049, 2048, 16, 0, 0, 0};
  std::string trace;
  std::vector<Group> groups;
  while (tail.cursor < tail.end || tail.pending) {
    if (tail.CanPost()) {
      groups.push_back(tail.Next());
      tail.Posted(groups.back());
      trace += 'P';
    } else {
      tail.Completed();
      trace += 'D';
    }
  }
  CHECK(trace == "PPDPDD");
  CHECK(groups.size() == 3);
  CHECK(groups[0].count == 1024 && groups[0].first_jetty == 0);
  CHECK(groups[1].count == 1024 && groups[1].first_jetty == 8);
  CHECK(groups[2].count == 1 && groups[2].first_jetty == 0);

  Schedule odd{{}, 0, 17, 5, 3, 0, 0, 0};
  for (uint32_t expected : {2U, 3U, 2U, 3U, 2U, 3U, 2U}) {
    if (!odd.CanPost()) odd.Completed();
    const Group group = odd.Next();
    CHECK(group.count == expected);
    odd.Posted(group);
  }
  CHECK(odd.cursor == 17);

  CHECK(kvaf_batch_get::SelectPostWidth(32, 16, 2, 0) == 128);
  CHECK(kvaf_batch_get::SelectPostWidth(32, 16, 2, 256) == 8);
  CHECK(kvaf_batch_get::SelectPostWidth(64, 32, 2, 256) == 16);
  CHECK(kvaf_batch_get::SelectPostWidth(128, 64, 2, 256) == 32);
  CHECK(kvaf_batch_get::SelectPostWidth(256, 128, 2, 256) == 64);
  CHECK(kvaf_batch_get::SelectPostWidth(257, 129, 2, 256) == 128);
  CHECK(kvaf_batch_get::SelectPostWidth(32, 32, 1, 256) == 128);
  CHECK(kvaf_batch_get::SelectPostWidth(1, 0, 64, 256) == 128);
  CHECK(kvaf_batch_get::SelectPostWidth(1, 1, 64, 256) == 1);
  CHECK(kvaf_batch_get::SelectPostWidth(33, 17, 2, 256) == 9);
  CHECK(kvaf_batch_get::SelectPostWidth(UINT32_MAX, UINT32_MAX, 64,
                                       UINT32_MAX) == 128);
  CHECK(kvaf_batch_get::PostCount(UINT32_MAX, 1) == UINT32_MAX);
  CHECK(kvaf_batch_get::PostCount(UINT32_MAX, 3) == 1431655765U);
  const Group small_tail{0, 19, 1, 3};
  CHECK(kvaf_batch_get::AssignedRequests(small_tail, 0, 3) == 6);
  CHECK(kvaf_batch_get::AssignedRequests(small_tail, 1, 3) == 7);
  CHECK(kvaf_batch_get::AssignedRequests(small_tail, 2, 3) == 6);
  Schedule small_rotation{{}, 0, 33, 16, 3, 0, 0, 0, 3};
  const Group first_small = small_rotation.Next();
  CHECK(first_small.count == 8);
  small_rotation.Posted(first_small);
  CHECK(small_rotation.Next().first_jetty == 0);  // three posts, not one
  small_rotation.Completed();

  for (uint32_t cores : {1U, 2U, 3U, 4U, 32U, 48U}) {
    for (uint32_t jettys : {1U, 2U, 3U, 8U, 16U, 64U}) {
      for (uint32_t budget : {0U, 96U, 2048U, 2049U}) {
        for (uint32_t misses : {0U, 1U, 127U, 128U, 129U, 1023U, 1024U,
                                2047U, 2048U, 2049U, 3072U, 4096U, 8193U}) {
          CheckLaunch(misses, cores, jettys, budget);
        }
      }
    }
  }
  CheckLaunch(65537, 1, 64, 2048);
  CheckLaunch(65537, 3, 3, 2049);
  for (uint32_t cores : {1U, 2U, 3U, 4U, 32U, 48U}) {
    for (uint32_t jettys : {1U, 2U, 3U, 4U, 8U, 64U}) {
      for (uint32_t budget : {0U, 96U, 2048U}) {
        for (uint32_t misses : {0U, 1U, 2U, 3U, 31U, 32U, 33U, 63U, 64U,
                                65U, 127U, 128U, 129U, 255U, 256U, 257U,
                                511U, 512U, 513U, 2048U, 2049U}) {
          CheckLaunch(misses, cores, jettys, budget, 256);
        }
      }
    }
  }
  // Narrow groups must retain routing and credit accounting when an explicit
  // small inflight budget is combined with the small-post experiment.
  CheckLaunch(257, 3, 3, 9, 512);
  CheckLaunch(8193, 1, 64, 2048, 8193);
  CHECK(kvaf_batch_get::PostCount(UINT32_MAX) == 33554432U);
  for (uint32_t jettys : {1U, 3U, 64U}) {
    uint64_t sum = 0;
    for (uint32_t j = 0; j < jettys; ++j) {
      sum += kvaf_batch_get::AssignedRequests(Group{0, UINT32_MAX, jettys - 1}, j, jettys);
    }
    CHECK(sum == UINT32_MAX);
  }
  std::cout << "BatchGet schedule: baseline/small posts, credits and boundaries passed\n";
}
