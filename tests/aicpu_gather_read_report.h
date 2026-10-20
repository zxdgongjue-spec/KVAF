#pragma once
#include <cstdio>
#include <vector>

#include "gather_read_profile.h"
namespace kvaf::aicpu_gather_read_client {
inline void PrintGatherReadTimeline(std::FILE* output, double mean_ms,
                                    const std::vector<GatherReadTiming>& rounds) {
  const auto us = [&](auto field) {
    long double sum = 0;
    for (const auto& round : rounds) sum += round.*field;
    return static_cast<double>(sum / rounds.size() / 1000);
  };
  std::fprintf(output,
               "GATHER_READ_TIMELINE average per measured round (us)\n  GatherRead call: %.1f us\n",
               mean_ms * 1000);
  for (const auto& entry :
       {std::pair{"Compute payload size / output extent", &GatherReadTiming::layout_compute_ns},
        {"Assign sequence / enable CPU", &GatherReadTiming::configure_ns},
        {"Pack current lists", &GatherReadTiming::metadata_pack_ns},
        {"Look up HBM registration", &GatherReadTiming::register_ns},
        {"Enqueue metadata H2D (submission only)", &GatherReadTiming::metadata_enqueue_ns},
        {"Launch AICPU", &GatherReadTiming::launch_ns},
        {"Enqueue status D2H (submission only)", &GatherReadTiming::status_enqueue_ns},
        {"Host waits for completion (includes transfers and CPU/AICPU)",
         &GatherReadTiming::sync_ns},
        {"AICPU target layout preparation", &GatherReadTiming::output_layout_ns},
        {"AICPU WRITE source list", &GatherReadTiming::source_write_ns},
        {"AICPU notify CPU", &GatherReadTiming::notify_ns},
        {"CPU prepare source list", &GatherReadTiming::cpu_source_cache_ns},
        {"CPU gather DDR -> staging", &GatherReadTiming::cpu_copy_ns},
        {"AICPU READ staging -> HBM (includes ready waits)", &GatherReadTiming::scatter_total_ns},
        {"Wait for ready prefixes (within READ)", &GatherReadTiming::ready_wait_ns},
        {"Build URMA WR/SGL (within READ)", &GatherReadTiming::scatter_plan_ns},
        {"Provider submit, includes SQ writes (within READ)", &GatherReadTiming::scatter_post_ns},
        {"Wait for final READ CQE (within READ)", &GatherReadTiming::scatter_cq_ns},
        {"CPU ACK wait (final confirmation with pipeline)", &GatherReadTiming::ack_poll_ns},
        {"Decode status", &GatherReadTiming::status_decode_ns}})
    std::fprintf(output, "  %s: %.1f us\n", entry.first, us(entry.second));
  std::fprintf(output,
               "  CPU gather overlaps READ with pipeline. Nested phases must not be added.\n"
               "GATHER_READ_BACKEND umdk sq_write_timing=included_in_provider_submit "
               "pipeline_chunk_bytes=%u\n",
               rounds.front().pipeline_chunk_bytes);
}
inline void PrintGatherReadWorkers(std::FILE* output, const std::vector<GatherReadTiming>& rounds) {
  for (unsigned worker = 0; worker < rounds.front().aicpu_cores; ++worker) {
    const auto us = [&](auto field) {
      long double sum = 0;
      for (const auto& round : rounds) sum += round.workers[worker].*field;
      return static_cast<double>(sum / rounds.size() / 1000);
    };
    long double polls = 0, bytes = 0;
    for (const auto& round : rounds) {
      polls += round.workers[worker].cq_poll_calls;
      bytes += round.workers[worker].read_bytes;
    }
    std::fprintf(
        output,
        "GATHER_READ_WORKER worker=%u start=%.1f end=%.1f build=%.1f post=%.1f cq_wait=%.1f "
        "read_bytes=%.0Lf cq_poll_calls=%.1Lf\n",
        worker, us(&GatherReadWorkerTiming::start_offset_ns),
        us(&GatherReadWorkerTiming::end_offset_ns), us(&GatherReadWorkerTiming::build_ns),
        us(&GatherReadWorkerTiming::post_ns), us(&GatherReadWorkerTiming::cq_ns),
        bytes / rounds.size(), polls / rounds.size());
  }
}
}  // namespace kvaf::aicpu_gather_read_client
