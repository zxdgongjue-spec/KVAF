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
    const uint32_t count = std::min(remaining, 128U);
    expected[jetty] += count;
    remaining -= count;
    jetty = (jetty + 1U) % jettys;
  }
  for (uint32_t j = 0; j < jettys; ++j) {
    CHECK(kvaf_batch_get::AssignedRequests(group, j, jettys) == expected[j]);
  }
}

void CheckLaunch(uint32_t misses, uint32_t cores, uint32_t jettys,
                 uint32_t budget) {
  std::vector<Schedule> schedules;
  std::vector<uint64_t> outstanding(cores), retired(cores);
  uint64_t allocated = 0;
  for (uint32_t c = 0; c < cores; ++c) {
    const uint32_t local_budget = kvaf_batch_get::CoreBudget(budget, cores, c);
    allocated += local_budget;
    schedules.push_back(Schedule{
        {}, static_cast<uint32_t>(uint64_t(misses) * c / cores),
        static_cast<uint32_t>(uint64_t(misses) * (c + 1) / cores),
        budget && misses > budget ? local_budget : 0U, jettys, 0, 0, 0});
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
  CHECK(kvaf_batch_get::PostCount(UINT32_MAX) == 33554432U);
  for (uint32_t jettys : {1U, 3U, 64U}) {
    uint64_t sum = 0;
    for (uint32_t j = 0; j < jettys; ++j) {
      sum += kvaf_batch_get::AssignedRequests(Group{0, UINT32_MAX, jettys - 1}, j, jettys);
    }
    CHECK(sum == UINT32_MAX);
  }
  std::cout << "BatchGet schedule: 1874 launch cases and boundary checks passed\n";
}
