#pragma once

#include "gather_read_profile.h"

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
  const bool direct = average(&GatherReadTiming::source_write_ns) > 0.0;

  std::fprintf(output,
               "GATHER_READ_TIMELINE average per measured round (us; 3 decimals = 1 ns)\n"
               "  GatherRead call: %.3f us\n"
               "  1 Host prepares before AICPU launch:\n"
               "    Validate layout: %.3f us\n",
               mean_ms * 1000.0, average_us(&GatherReadTiming::validate_ns));
  if (direct) {
    if (average(&GatherReadTiming::configure_ns) > 0.0)
      std::fprintf(output, "    Check DDR mailbox (enable worker on first call): %.3f us\n",
                   average_us(&GatherReadTiming::configure_ns));
    std::fprintf(output, "    Upload DDR source list to HBM: %.3f us\n",
                 average_us(&GatherReadTiming::source_upload_ns));
  } else
    std::fprintf(output, "    Send DDR source list to CPU: %.3f us\n",
                 average_us(&GatherReadTiming::configure_ns));
  std::fprintf(output,
               "    Upload HBM targets: %.3f us | Look up HBM registration: %.3f us | Launch AICPU: %.3f us\n",
               average_us(&GatherReadTiming::descriptor_upload_ns),
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
  if (direct)
    std::fprintf(output, "    AICPU WRITE source list to CPU DDR: %.3f us\n",
                 average_us(&GatherReadTiming::source_write_ns));
  std::fprintf(output,
               "    AICPU notify CPU: %.3f us\n"
               "    AICPU wait for CPU gather ack: %.3f us (overlaps CPU gather)\n",
               average_us(&GatherReadTiming::notify_ns),
               average_us(&GatherReadTiming::ack_poll_ns));
  if (direct) {
    double started_ns = 0.0;
    std::size_t started_rounds = 0U;
    for (const auto& profile : profiles) {
      if (profile.ack_started_ns == 0U) continue;
      started_ns += profile.ack_started_ns;
      ++started_rounds;
    }
    if (started_rounds != 0U)
      std::fprintf(output,
                   "      CPU start observed after ACK polling began: %.3f us (%zu/%zu rounds)\n",
                   started_ns / started_rounds / 1000.0, started_rounds,
                   profiles.size());
    else
      std::fprintf(output, "      CPU start marker not observed before final ACK\n");
    std::fprintf(output, "      CPU read and prepare source list: %.3f us\n",
                 average_us(&GatherReadTiming::cpu_source_cache_ns));
  }
  std::fprintf(output,
               "      CPU gather DDR -> staging: %.3f us\n"
               "    AICPU READ staging -> HBM: %.3f us (%.2f READ WQEs on average)\n"
               "      Build READ WQEs: %.3f us\n"
               "      Submit WQEs (encode, write SQ, doorbell): %.3f us | Wait for READ completions: %.3f us\n",
               copy_us,
               plan_us + post_us + cq_us,
               average(&GatherReadTiming::read_wqes),
               plan_us, post_us, cq_us);
  std::fprintf(output,
               "  3 Host finishes: Release HBM reference: %.3f us | Read status: %.3f us\n"
               "  Overlap: Host wait includes CPU/AICPU; AICPU ack wait overlaps CPU gather.\n",
               average_us(&GatherReadTiming::release_ns),
               average_us(&GatherReadTiming::status_copy_ns));
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
               "  Host: validate=%.0f %s=%.0f source_upload=%.0f descriptor_upload=%.0f "
               "hbm_register=%.0f launch=%.0f stream_sync=%.0f hbm_release=%.0f status_copy=%.0f\n",
               average(&GatherReadTiming::validate_ns),
               direct ? "direct_setup" : "configure_rpc",
               average(&GatherReadTiming::configure_ns),
               average(&GatherReadTiming::source_upload_ns),
               average(&GatherReadTiming::descriptor_upload_ns),
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
               "scatter_post=%.0f scatter_cq=%.0f read_wqes=%.2f\n",
               average(&GatherReadTiming::scatter_cache_ns),
               average(&GatherReadTiming::scatter_plan_ns),
               average(&GatherReadTiming::scatter_post_ns),
               average(&GatherReadTiming::scatter_cq_ns),
               average(&GatherReadTiming::read_wqes));
}

}
