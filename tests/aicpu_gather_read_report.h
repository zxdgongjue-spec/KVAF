#pragma once

#include "gather_read_profile.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace kvaf::aicpu_gather_read_client {

inline void PrintGatherReadTimeline(std::FILE* output, double mean_ms,
                                    const std::vector<GatherReadTiming>& profiles) {
  if (output == nullptr || profiles.empty())
    throw std::invalid_argument("GatherRead timeline needs an output and measured profiles");
  const auto average = [&](std::uint64_t GatherReadTiming::*field) {
    std::uint64_t sum = 0U;
    for (const auto& profile : profiles) sum += profile.*field;
    return static_cast<double>(sum) / profiles.size();
  };
  const auto average_us = [&](std::uint64_t GatherReadTiming::*field) {
    return average(field) / 1000.0;
  };
  const double copy_us = average_us(&GatherReadTiming::cpu_copy_ns);
  const double plan_us = average_us(&GatherReadTiming::scatter_plan_ns);
  const double post_us = average_us(&GatherReadTiming::scatter_post_ns);
  const double cq_us = average_us(&GatherReadTiming::scatter_cq_ns);
  double read_us = 0.0;
  for (const auto& profile : profiles)
    read_us += static_cast<double>(profile.scatter_total_ns != 0U
        ? profile.scatter_total_ns
        : profile.scatter_plan_ns + profile.scatter_post_ns + profile.scatter_cq_ns);
  read_us /= profiles.size() * 1000.0;
  const bool parallel = average(&GatherReadTiming::aicpu_cores) > 1.0;
  const bool pipeline = profiles.front().pipeline_chunk_bytes != 0U;
  const bool direct = average(&GatherReadTiming::source_write_ns) > 0.0;
  const double metadata_us = average_us(&GatherReadTiming::metadata_upload_ns);

  std::fprintf(output,
               "GATHER_READ_TIMELINE average per measured round (us; 3 decimals = 1 ns)\n"
               "  GatherRead call: %.3f us\n"
               "  1 Host prepares before AICPU launch:\n"
               "    Validate layout: %.3f us\n",
               mean_ms * 1000.0, average_us(&GatherReadTiming::validate_ns));
  if (direct) {
    if (average(&GatherReadTiming::configure_ns) > 0.0)
      std::fprintf(output, "    Assign sequence / enable CPU worker on first call: %.3f us\n",
                   average_us(&GatherReadTiming::configure_ns));
    if (metadata_us > 0.0)
      std::fprintf(output, "    Pack source list + HBM targets and upload once: %.3f us\n",
                   metadata_us);
    if (metadata_us == 0.0 || average(&GatherReadTiming::source_upload_ns) > 0.0)
      std::fprintf(output, "    Upload DDR source list to HBM: %.3f us\n",
                   average_us(&GatherReadTiming::source_upload_ns));
  } else
    std::fprintf(output, "    Send DDR source list to CPU: %.3f us\n",
                 average_us(&GatherReadTiming::configure_ns));
  if (metadata_us == 0.0 || average(&GatherReadTiming::descriptor_upload_ns) > 0.0)
    std::fprintf(output, "    Upload HBM targets: %.3f us\n",
                 average_us(&GatherReadTiming::descriptor_upload_ns));
  std::fprintf(output,
               "    Look up HBM registration: %.3f us | Launch AICPU: %.3f us\n",
               average_us(&GatherReadTiming::register_ns),
               average_us(&GatherReadTiming::launch_ns));
  if (!direct)
    std::fprintf(output,
                 "    CPU worker before notification: Dispatch: %.3f us | Wait for notify: %.3f us"
                 " (overlaps Host setup)\n",
                 average_us(&GatherReadTiming::cpu_dispatch_ns),
                 average_us(&GatherReadTiming::cpu_notify_spin_ns));
  std::fprintf(output,
               "  2 Host waits for completion: %.3f us (includes CPU/AICPU work below)\n",
               average_us(&GatherReadTiming::sync_ns));
  std::fprintf(output, "    AICPU validate HBM targets and prepare worker starts: %.3f us\n",
               average_us(&GatherReadTiming::output_layout_ns));
  if (direct)
    std::fprintf(output, "    AICPU WRITE source list to CPU DDR: %.3f us\n",
                 average_us(&GatherReadTiming::source_write_ns));
  std::fprintf(output, "    AICPU notify CPU: %.3f us\n",
               average_us(&GatherReadTiming::notify_ns));
  if (!pipeline)
    std::fprintf(output, "    AICPU wait for CPU gather ack: %.3f us (overlaps CPU gather)\n",
                 average_us(&GatherReadTiming::ack_poll_ns));
  else
    std::fprintf(output,
                 "    CPU gather and AICPU READ overlap; CPU publishes ready prefixes in %u-byte chunks.\n",
                 profiles.front().pipeline_chunk_bytes);
  if (direct) {
    double started_ns = 0.0;
    std::size_t started_rounds = 0U;
    for (const auto& profile : profiles) {
      if (profile.ack_started_ns == 0U) continue;
      started_ns += profile.ack_started_ns;
      ++started_rounds;
    }
    if (!pipeline && started_rounds != 0U)
      std::fprintf(output,
                   "      CPU start observed after ACK polling began: %.3f us (%zu/%zu rounds)\n",
                   started_ns / started_rounds / 1000.0, started_rounds,
                   profiles.size());
    else if (!pipeline)
      std::fprintf(output, "      CPU start marker not observed before final ACK\n");
    std::fprintf(output, "      CPU read and prepare source list: %.3f us\n",
                 average_us(&GatherReadTiming::cpu_source_cache_ns));
  }
  std::fprintf(output,
               "      CPU gather DDR -> staging: %.3f us\n"
               "    AICPU READ staging -> HBM%s: %.3f us (%.2f READ WQEs on average)\n",
               copy_us, pipeline ? " (includes waiting for ready prefixes)" : "",
               read_us, average(&GatherReadTiming::read_wqes));
  if (pipeline)
    std::fprintf(output,
                 "      Wait for ready prefixes: %.3f us (max accumulated per worker; included in READ wall time)\n",
                 average_us(&GatherReadTiming::ready_wait_ns));
  if (parallel)
    std::fprintf(output,
                 "      %.0f AICPU workers, independent SQ/CQ; READ time is wall-clock, not a sum.\n"
                 "      Details below are per-round maxima across cores; they overlap and must not be added.\n",
                 average(&GatherReadTiming::aicpu_cores));
  std::fprintf(output,
               "      Worker/task initialization (includes WQE templates): %.3f us | Build local SGL: %.3f us\n"
               "      Submit WQEs (encode, write SQ, doorbell): %.3f us | Wait for READ completions: %.3f us\n",
               average_us(&GatherReadTiming::scatter_init_ns), plan_us, post_us, cq_us);
  if (profiles.front().read_task_spans != 0U)
    std::fprintf(output,
                 "      Dynamic READ: up to %u spans/task, %.2f tasks/round; "
                 "claim tasks: %.3f us (max per worker, included in READ wall time)\n",
                 profiles.front().read_task_spans, average(&GatherReadTiming::read_tasks),
                 average_us(&GatherReadTiming::scatter_schedule_ns));
  if (profiles.front().profile_detail)
    std::fprintf(output,
                 "        Within submit: Encode WQEs: %.3f us | Write SQ: %.3f us | Doorbell and queue update: %.3f us\n",
                 average_us(&GatherReadTiming::scatter_encode_ns),
                 average_us(&GatherReadTiming::scatter_sq_write_ns),
                 average_us(&GatherReadTiming::scatter_doorbell_ns));
  else
    std::fprintf(output, "        Per-SQ-group timing: disabled (READ wall and batch timing remain enabled)\n");
  std::fprintf(output, "      SQ writes: %.2f contiguous ranges, %.2f WQEBBs (64 bytes each), all workers combined\n",
               average(&GatherReadTiming::sq_write_calls), average(&GatherReadTiming::sq_wqebbs));
  if (parallel)
    std::fprintf(output,
                 "      Worker start spread: %.3f us | Longest worker duration: %.3f us (both within READ wall time)\n",
                 average_us(&GatherReadTiming::scatter_start_skew_ns),
                 average_us(&GatherReadTiming::scatter_worker_ns));
  if (pipeline)
    std::fprintf(output, "    Final CPU ACK confirmation after worker 0 READs: %.3f us (overlaps other workers)\n",
                 average_us(&GatherReadTiming::ack_poll_ns));
  std::fprintf(output,
               "  3 Host finishes: Release HBM reference: %.3f us | Read status: %.3f us\n"
               "  Overlap: Host wait includes CPU/AICPU; %s\n",
               average_us(&GatherReadTiming::release_ns),
               average_us(&GatherReadTiming::status_copy_ns),
               pipeline ? "CPU gather overlaps READ; ready waits are included in READ wall time."
                        : "AICPU ack wait overlaps CPU gather.");
}

inline void PrintGatherReadTechnicalProfile(std::FILE* output,
                                            const std::vector<GatherReadTiming>& profiles) {
  if (output == nullptr || profiles.empty())
    throw std::invalid_argument("GatherRead technical profile needs measured profiles");
  const auto average = [&](std::uint64_t GatherReadTiming::*field) {
    std::uint64_t sum = 0U;
    for (const auto& profile : profiles) sum += profile.*field;
    return static_cast<double>(sum) / profiles.size();
  };
  const bool direct = average(&GatherReadTiming::source_write_ns) > 0.0;
  std::fprintf(output,
               "GATHER_READ_TECHNICAL avg_ns (only with --profile-detail 1; nested times overlap)\n"
               "  Host: validate=%.0f %s=%.0f source_upload=%.0f descriptor_upload=%.0f metadata_upload=%.0f "
               "hbm_register=%.0f launch=%.0f stream_sync=%.0f hbm_release=%.0f status_copy=%.0f\n",
               average(&GatherReadTiming::validate_ns),
               direct ? "direct_setup" : "configure_rpc",
               average(&GatherReadTiming::configure_ns),
               average(&GatherReadTiming::source_upload_ns),
               average(&GatherReadTiming::descriptor_upload_ns),
               average(&GatherReadTiming::metadata_upload_ns),
               average(&GatherReadTiming::register_ns),
               average(&GatherReadTiming::launch_ns),
               average(&GatherReadTiming::sync_ns),
               average(&GatherReadTiming::release_ns),
               average(&GatherReadTiming::status_copy_ns));
  std::fprintf(output,
               "  AICPU control: source_write=%.0f source_write_post=%.0f source_write_cq=%.0f "
               "notify=%.0f notify_post=%.0f notify_cq=%.0f "
               "ack_poll=%.0f ack_started=%.0f ack_post=%.0f ack_cq=%.0f ack_cache=%.0f ack_polls=%.2f\n",
               average(&GatherReadTiming::source_write_ns),
               average(&GatherReadTiming::source_write_post_ns),
               average(&GatherReadTiming::source_write_cq_ns),
               average(&GatherReadTiming::notify_ns),
               average(&GatherReadTiming::notify_post_ns),
               average(&GatherReadTiming::notify_cq_ns),
               average(&GatherReadTiming::ack_poll_ns),
               average(&GatherReadTiming::ack_started_ns),
               average(&GatherReadTiming::ack_post_ns),
               average(&GatherReadTiming::ack_cq_ns),
               average(&GatherReadTiming::ack_cache_ns),
               average(&GatherReadTiming::ack_polls));
  std::fprintf(output,
               "  CPU: dispatch=%.0f %s=%.0f descriptor_prepare=%.0f copy=%.0f\n",
               average(&GatherReadTiming::cpu_dispatch_ns),
               direct ? "idle_since_previous_ack" : "notify_spin",
               average(&GatherReadTiming::cpu_notify_spin_ns),
               average(&GatherReadTiming::cpu_source_cache_ns),
               average(&GatherReadTiming::cpu_copy_ns));
  std::fprintf(output,
               "  AICPU data: scatter_cache=%.0f scatter_plan=%.0f "
               "scatter_post=%.0f scatter_cq=%.0f read_wqes=%.2f scatter_wall=%.0f cores=%.0f "
               "(worker phases: max per core, overlapping)\n",
               average(&GatherReadTiming::scatter_cache_ns),
               average(&GatherReadTiming::scatter_plan_ns),
               average(&GatherReadTiming::scatter_post_ns),
               average(&GatherReadTiming::scatter_cq_ns),
               average(&GatherReadTiming::read_wqes),
               average(&GatherReadTiming::scatter_total_ns),
               average(&GatherReadTiming::aicpu_cores));
  std::fprintf(output,
               "  READ preparation: output_layout=%.0f worker_init=%.0f start_spread=%.0f longest_worker=%.0f\n"
               "  Within submit (max per core): encode=%.0f sq_write=%.0f doorbell_queue_update=%.0f\n",
               average(&GatherReadTiming::output_layout_ns),
               average(&GatherReadTiming::scatter_init_ns),
               average(&GatherReadTiming::scatter_start_skew_ns),
               average(&GatherReadTiming::scatter_worker_ns),
               average(&GatherReadTiming::scatter_encode_ns),
               average(&GatherReadTiming::scatter_sq_write_ns),
               average(&GatherReadTiming::scatter_doorbell_ns));
  std::fprintf(output,
               "  READ schedule: task_spans=%u tasks=%.2f claim_ns=%.0f "
               "(claim_ns: max per worker, included in scatter_wall)\n",
               profiles.front().read_task_spans, average(&GatherReadTiming::read_tasks),
               average(&GatherReadTiming::scatter_schedule_ns));
  if (profiles.front().pipeline_chunk_bytes != 0U)
    std::fprintf(output, "  Pipeline: chunk_bytes=%u ready_wait_ns=%.0f (max per worker; included in scatter_wall)\n",
                 profiles.front().pipeline_chunk_bytes, average(&GatherReadTiming::ready_wait_ns));
}

inline void PrintGatherReadWorkers(std::FILE* output,
                                   const std::vector<GatherReadTiming>& profiles) {
  if (output == nullptr || profiles.empty() || profiles.front().aicpu_cores == 0U ||
      profiles.front().aicpu_cores > 8U)
    throw std::invalid_argument("GatherRead workers need measured profiles and 1..8 workers");
  const auto workers = profiles.front().aicpu_cores;
  for (const auto& profile : profiles)
    if (profile.aicpu_cores != workers)
      throw std::invalid_argument("GatherRead worker count changed within measured profiles");
  std::fprintf(output,
               "GATHER_READ_WORKERS relative to first READ worker entry in each round (us)\n"
               "  Worker IDs identify SQ/CQ slots, not physical CPU IDs. Worker intervals overlap.\n");
  for (std::size_t worker = 0U; worker < workers; ++worker) {
    double start_ns = 0.0, end_ns = 0.0, duration_ns = 0.0;
    double wqes = 0.0, copies = 0.0, blocks = 0.0, tasks = 0.0;
    std::vector<std::uint64_t> starts;
    starts.reserve(profiles.size());
    for (const auto& profile : profiles) {
      const auto& timing = profile.workers[worker];
      starts.push_back(timing.start_offset_ns);
      start_ns += timing.start_offset_ns;
      end_ns += timing.end_offset_ns;
      duration_ns += timing.duration_ns;
      wqes += timing.read_wqes;
      copies += timing.sq_write_calls;
      blocks += timing.sq_wqebbs;
      tasks += timing.read_tasks;
    }
    std::sort(starts.begin(), starts.end());
    const auto percentile_us = [&](double fraction) {
      const double position = fraction * (starts.size() - 1U);
      const auto lower = static_cast<std::size_t>(position);
      const auto upper = std::min(lower + 1U, starts.size() - 1U);
      return (starts[lower] + (static_cast<double>(starts[upper]) - starts[lower]) *
                                 (position - lower)) / 1000.0;
    };
    const double rounds = profiles.size();
    std::fprintf(output,
                 "  worker=%zu start_avg=%.3f start_p50=%.3f start_p95=%.3f end_avg=%.3f "
                 "duration_avg=%.3f read_wqes=%.2f sq_write_ranges=%.2f sq_wqebbs=%.2f read_tasks=%.2f\n",
                 worker, start_ns / rounds / 1000.0, percentile_us(0.5), percentile_us(0.95),
                 end_ns / rounds / 1000.0, duration_ns / rounds / 1000.0,
                 wqes / rounds, copies / rounds, blocks / rounds, tasks / rounds);
  }
}

}
