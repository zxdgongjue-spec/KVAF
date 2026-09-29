#pragma once

#include "gather_read_profile.h"

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
  const double target_cache_us = average_us(&GatherReadTiming::scatter_cache_ns);
  const double plan_us = average_us(&GatherReadTiming::scatter_plan_ns);
  const double post_us = average_us(&GatherReadTiming::scatter_post_ns);
  const double cq_us = average_us(&GatherReadTiming::scatter_cq_ns);

  std::fprintf(output,
               "GATHER_READ_TIMELINE average per measured round (us; 3 decimals = 1 ns)\n"
               "  GatherRead call: %.3f us\n"
               "  1 Host prepares before AICPU launch:\n"
               "    Validate layout: %.3f us | Send DDR source list to CPU: %.3f us\n"
               "    Upload HBM targets: %.3f us | Look up HBM registration: %.3f us | Launch AICPU: %.3f us\n"
               "    CPU worker before notification: Dispatch: %.3f us | Wait for notify: %.3f us"
               " (overlaps Host setup)\n"
               "  2 Host waits for completion: %.3f us (includes CPU/AICPU work below)\n"
               "    AICPU notify CPU: %.3f us\n"
               "    AICPU wait for CPU gather ack: %.3f us (overlaps CPU gather)\n",
               mean_ms * 1000.0,
               average_us(&GatherReadTiming::validate_ns),
               average_us(&GatherReadTiming::configure_ns),
               average_us(&GatherReadTiming::descriptor_upload_ns),
               average_us(&GatherReadTiming::register_ns),
               average_us(&GatherReadTiming::launch_ns),
               average_us(&GatherReadTiming::cpu_dispatch_ns),
               average_us(&GatherReadTiming::cpu_notify_spin_ns),
               average_us(&GatherReadTiming::sync_ns),
               average_us(&GatherReadTiming::notify_ns),
               average_us(&GatherReadTiming::ack_poll_ns));
  std::fprintf(output,
               "      CPU gather DDR -> staging: %.3f us\n"
               "        Parallel copy and worker synchronization; no data cache maintenance\n"
               "    AICPU READ staging -> HBM: %.3f us (%.2f READ WQEs on average)\n"
               "      HBM cache maintenance: %.3f us | Build READ WQEs: %.3f us\n"
               "      Submit WQEs (encode, flush SQ, doorbell): %.3f us | Wait for READ completions: %.3f us\n",
               copy_us,
               target_cache_us + plan_us + post_us + cq_us,
               average(&GatherReadTiming::read_wqes),
               target_cache_us, plan_us, post_us, cq_us);
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
  std::fprintf(output,
               "GATHER_READ_TECHNICAL avg_ns (only with --profile-detail 1; nested times overlap)\n"
               "  Host: validate=%.0f configure_rpc=%.0f descriptor_upload=%.0f "
               "hbm_register=%.0f launch=%.0f stream_sync=%.0f hbm_release=%.0f status_copy=%.0f\n",
               average(&GatherReadTiming::validate_ns),
               average(&GatherReadTiming::configure_ns),
               average(&GatherReadTiming::descriptor_upload_ns),
               average(&GatherReadTiming::register_ns),
               average(&GatherReadTiming::launch_ns),
               average(&GatherReadTiming::sync_ns),
               average(&GatherReadTiming::release_ns),
               average(&GatherReadTiming::status_copy_ns));
  std::fprintf(output,
               "  AICPU control: notify=%.0f notify_post=%.0f notify_cq=%.0f "
               "ack_poll=%.0f ack_post=%.0f ack_cq=%.0f ack_cache=%.0f ack_polls=%.2f\n",
               average(&GatherReadTiming::notify_ns),
               average(&GatherReadTiming::notify_post_ns),
               average(&GatherReadTiming::notify_cq_ns),
               average(&GatherReadTiming::ack_poll_ns),
               average(&GatherReadTiming::ack_post_ns),
               average(&GatherReadTiming::ack_cq_ns),
               average(&GatherReadTiming::ack_cache_ns),
               average(&GatherReadTiming::ack_polls));
  std::fprintf(output,
               "  CPU: dispatch=%.0f notify_spin=%.0f copy=%.0f\n",
               average(&GatherReadTiming::cpu_dispatch_ns),
               average(&GatherReadTiming::cpu_notify_spin_ns),
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
