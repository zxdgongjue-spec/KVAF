#pragma once

#include <algorithm>
#include <cstdio>
#include <set>
#include <vector>

#include "cpu_gather_team.h"

namespace kvaf::peer {

// Aggregate after the final ACK, outside the measured Copy() interval. Print
// once when the client region closes; no per-round logging or wire ABI change.
class CpuGatherCopyStats {
 public:
  explicit CpuGatherCopyStats(std::size_t max_threads) : workers_(max_threads) {}

  void Accumulate(const CpuGatherTeam& team) {
    const auto& copy = team.copy_profile();
    ++rounds_;
    min_threads_ = std::min(min_threads_, copy.workers);
    max_threads_ = std::max(max_threads_, copy.workers);
    expected_bytes_ += team.total_bytes();
    team_wall_ns_ += copy.end_ns - copy.begin_ns;
    join_wait_ns_ += copy.join_wait_ns;
    join_wait_max_ns_ = std::max(join_wait_max_ns_, copy.join_wait_ns);
    std::uint64_t copied = 0U;
    for (std::size_t index = 0U; index < copy.workers; ++index) {
      const auto& sample = team.copy_worker_profile(index);
      auto& sum = workers_[index];
      ++sum.rounds;
      sum.bytes += sample.bytes;
      copied += sample.bytes;
      const auto delay = sample.begin_ns - copy.begin_ns;
      sum.start_delay_ns += delay;
      sum.start_delay_max_ns = std::max(sum.start_delay_max_ns, delay);
      sum.work_wall_ns += sample.end_ns - sample.begin_ns;
      sum.thread_cpu_ns += sample.thread_cpu_ns;
      if (sample.thread_cpu_available) ++sum.thread_cpu_rounds;
      if (sample.start_cpu >= 0) sum.cpus.insert(sample.start_cpu);
      if (sample.end_cpu >= 0) sum.cpus.insert(sample.end_cpu);
    }
    copied_bytes_ += copied;
    if (copied != team.total_bytes()) ++byte_mismatch_rounds_;
  }

  void Print(std::FILE* output, std::uint64_t session, std::uint64_t base,
             std::uint32_t lane = 0U) const {
    if (rounds_ == 0U) return;
    const auto divisor = static_cast<double>(rounds_);
    std::fprintf(output,
                 "[KVAF_CPU] GATHER_COPY_PROFILE session=%llu base=0x%llx rounds=%llu "
                 "active_threads_min=%zu active_threads_max=%zu expected_bytes_avg=%.1f "
                 "copied_bytes_avg=%.1f byte_mismatch_rounds=%llu team_wall_avg_us=%.1f "
                 "main_join_wait_avg_us=%.1f main_join_wait_max_us=%.1f lane=%u\n",
                 static_cast<unsigned long long>(session), static_cast<unsigned long long>(base),
                 static_cast<unsigned long long>(rounds_), min_threads_, max_threads_,
                 expected_bytes_ / divisor, copied_bytes_ / divisor,
                 static_cast<unsigned long long>(byte_mismatch_rounds_),
                 team_wall_ns_ / divisor / 1000.0, join_wait_ns_ / divisor / 1000.0,
                 join_wait_max_ns_ / 1000.0, lane);
    for (std::size_t index = 0U; index < workers_.size(); ++index) {
      const auto& sum = workers_[index];
      if (sum.rounds == 0U) continue;
      const auto count = static_cast<double>(sum.rounds);
      std::fprintf(output,
                   "[KVAF_CPU] GATHER_COPY_WORKER session=%llu worker=%zu rounds=%llu "
                   "bytes_avg=%.1f start_delay_avg_us=%.1f start_delay_max_us=%.1f "
                   "work_wall_avg_us=%.1f ",
                   static_cast<unsigned long long>(session), index,
                   static_cast<unsigned long long>(sum.rounds), sum.bytes / count,
                   sum.start_delay_ns / count / 1000.0, sum.start_delay_max_ns / 1000.0,
                   sum.work_wall_ns / count / 1000.0);
      if (sum.thread_cpu_rounds != 0U)
        std::fprintf(output, "work_thread_cpu_avg_us=%.1f ",
                     sum.thread_cpu_ns / static_cast<double>(sum.thread_cpu_rounds) / 1000.0);
      else
        std::fprintf(output, "work_thread_cpu_avg_us=unavailable ");
      std::fprintf(output, "thread_cpu_rounds=%llu/%llu cpu_ids=",
                   static_cast<unsigned long long>(sum.thread_cpu_rounds),
                   static_cast<unsigned long long>(sum.rounds));
      bool first = true;
      for (const auto cpu : sum.cpus) {
        std::fprintf(output, "%s%d", first ? "" : ",", cpu);
        first = false;
      }
      if (first) std::fprintf(output, "unavailable");
      std::fprintf(output, " lane=%u\n", lane);
    }
  }

 private:
  struct WorkerSum {
    std::uint64_t rounds{0U}, bytes{0U}, start_delay_ns{0U}, start_delay_max_ns{0U};
    std::uint64_t work_wall_ns{0U}, thread_cpu_ns{0U}, thread_cpu_rounds{0U};
    std::set<int> cpus;
  };
  std::vector<WorkerSum> workers_;
  std::uint64_t rounds_{0U}, expected_bytes_{0U}, copied_bytes_{0U}, byte_mismatch_rounds_{0U};
  std::uint64_t team_wall_ns_{0U}, join_wait_ns_{0U}, join_wait_max_ns_{0U};
  std::size_t min_threads_{SIZE_MAX}, max_threads_{0U};
};

}  // namespace kvaf::peer
