#pragma once

#include "control.h"
#include "cpu_urma/cpu_urma_transport.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace kvaf::aicpu_test {
struct TransferProfileHeader {
  std::uint64_t magic{kMagic}, revision{kKernelRevision}, rounds{}, frequency{};
  std::uint64_t sample_bytes{sizeof(Sample)};
};
static_assert(sizeof(TransferProfileHeader) == 40);

struct CpuTransferSample {
  std::uint64_t sequence{}, request_poll_ns{}, request_reads{};
  std::uint64_t source_update_ns{}, write_call_ns{}, notify_call_ns{}, response_ns{}, gap_ns{};
  CpuUrmaTransport::BatchTiming request{}, data{}, notify{};
  CpuUrmaTransport::WriteQueueProfile queues{};
};

inline void FinishCpuTransferSample(CpuTransferSample& s, std::uint64_t observed,
    std::uint64_t source_end, std::uint64_t write_begin, std::uint64_t write_end,
    std::uint64_t notify_begin, std::uint64_t notify_end) {
  Require(observed <= source_end && source_end <= write_begin && write_begin <= write_end &&
          write_end <= notify_begin && notify_begin <= notify_end, "non-monotonic CPU transfer timestamps");
  s.source_update_ns = source_end - observed;
  s.write_call_ns = write_end - write_begin;
  s.notify_call_ns = notify_end - notify_begin;
  s.response_ns = notify_end - observed;
  s.gap_ns = (write_begin - source_end) + (notify_begin - write_end);
}
inline std::uint64_t TransferOtherNs(const CpuUrmaTransport::BatchTiming& t) {
  const auto covered = t.plan_ns + t.setup_ns + t.prepare_ns + t.post_ns + t.poll_ns;
  Require(covered <= t.total_ns, "URMA phase sum exceeds total");
  return t.total_ns - covered;
}
inline void ValidateTransferSamples(const Config& c, std::uint64_t frequency,
    const std::vector<CpuTransferSample>& cpu, const std::vector<Sample>& device) {
  Layout layout{};
  Require(MakeLayout(c, &layout) && c.profile_transfer == 1 &&
          c.transfer_mode == TransferMode::kScatter && frequency > 0 &&
          cpu.size() == c.iterations && device.size() == cpu.size(), "invalid transfer profile");
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    const auto& s = cpu[i]; const auto& d = device[i];
    Require(s.sequence == c.warmup + i + 1 && s.request_reads > 0 &&
            s.request.total_ns <= s.request_poll_ns &&
            s.response_ns == s.source_update_ns + s.write_call_ns + s.notify_call_ns + s.gap_ns &&
            s.write_call_ns >= s.data.total_ns && s.notify_call_ns >= s.notify.total_ns &&
            d.total_ticks > 0 && d.wait_ticks <= d.total_ticks &&
            d.scatter_ticks == d.total_ticks - d.wait_ticks &&
            d.request_publish_ticks <= d.wait_ticks && d.done_poll_count > 0,
            "incomplete/inconsistent transfer sample");
    (void)TransferOtherNs(s.request); (void)TransferOtherNs(s.data); (void)TransferOtherNs(s.notify);
    Require(s.queues.jettys > 0 && s.queues.jettys <= s.queues.queue.size() && s.queues.waves > 0,
            "missing WRITE queue profile");
    std::uint64_t rows = 0, wrs = 0;
    for (std::size_t j = 0; j < s.queues.jettys; ++j) {
      const auto& q = s.queues.queue[j];
      rows += q.rows; wrs += q.wr_count;
      Require(q.cq_tail_ns <= q.post_to_cq_ns && q.tail_waves <= q.waves,
              "invalid WRITE queue timing");
    }
    Require(rows == c.rows && wrs == s.data.wr_count &&
            s.queues.jettys == cpu.front().queues.jettys, "inconsistent WRITE queue counts");
  }
}

inline constexpr const char* kTransferColumns[] = {
  "cpu_request_poll_ns", "cpu_request_reads", "cpu_matching_read_ns",
  "cpu_matching_read_setup_ns", "cpu_matching_read_prepare_ns", "cpu_matching_read_post_ns",
  "cpu_matching_read_poll_ns", "cpu_matching_read_other_ns",
  "cpu_source_update_ns", "cpu_response_ns", "cpu_gap_ns", "cpu_data_write_call_ns",
  "cpu_data_transport_ns", "cpu_data_plan_ns", "cpu_data_setup_ns", "cpu_data_prepare_ns",
  "cpu_data_post_ns", "cpu_data_poll_tail_ns", "cpu_data_other_ns", "cpu_data_wrapper_ns",
  "cpu_done_notify_call_ns", "cpu_done_transport_ns", "cpu_done_setup_ns",
  "cpu_done_prepare_ns", "cpu_done_post_ns", "cpu_done_poll_ns", "cpu_done_other_ns",
  "cpu_done_wrapper_ns", "cpu_data_wrs", "cpu_data_waves", "cpu_cq_spread_ns",
  "npu_wait_ns", "npu_request_publish_ns", "npu_done_wait_ns", "npu_done_polls",
  "npu_scatter_ns", "npu_total_ns", "npu_cache_prepare_ns", "npu_verify_ns"
};
constexpr std::size_t kTransferColumnCount = sizeof(kTransferColumns) / sizeof(kTransferColumns[0]);
inline std::array<double, kTransferColumnCount> TransferValues(
    const CpuTransferSample& s, const Sample& d, double ns_per_tick) {
  const auto n = [](std::uint64_t v) { return static_cast<double>(v); };
  return {{n(s.request_poll_ns), n(s.request_reads), n(s.request.total_ns),
    n(s.request.setup_ns), n(s.request.prepare_ns), n(s.request.post_ns), n(s.request.poll_ns),
    n(TransferOtherNs(s.request)), n(s.source_update_ns), n(s.response_ns), n(s.gap_ns),
    n(s.write_call_ns), n(s.data.total_ns), n(s.data.plan_ns), n(s.data.setup_ns),
    n(s.data.prepare_ns), n(s.data.post_ns), n(s.data.poll_ns), n(TransferOtherNs(s.data)),
    n(s.write_call_ns - s.data.total_ns), n(s.notify_call_ns), n(s.notify.total_ns),
    n(s.notify.setup_ns), n(s.notify.prepare_ns), n(s.notify.post_ns), n(s.notify.poll_ns),
    n(TransferOtherNs(s.notify)), n(s.notify_call_ns - s.notify.total_ns), n(s.data.wr_count),
    n(s.queues.waves), n(s.queues.cq_spread_ns), d.wait_ticks * ns_per_tick,
    d.request_publish_ticks * ns_per_tick, (d.wait_ticks - d.request_publish_ticks) * ns_per_tick,
    n(d.done_poll_count), d.scatter_ticks * ns_per_tick, d.total_ticks * ns_per_tick,
    d.cache_prepare_ticks * ns_per_tick, d.verify_ticks * ns_per_tick}};
}

inline void PrintTransferProfile(FILE* file, const Config& c, std::uint64_t frequency,
    const std::vector<CpuTransferSample>& cpu, const std::vector<Sample>& device) {
  ValidateTransferSamples(c, frequency, cpu, device);
  const double factor = 1e9 / static_cast<double>(frequency);
  long double data_ns = 0;
  for (const auto& s : cpu) data_ns += s.data.total_ns;
  std::fprintf(file, "[AICPU_TRANSFER] SUMMARY mode=scatter rounds=%llu "
      "avg_data_transport_us=%.3f staging_data_GBps=%.6f "
      "clocks=local_only request_poll_scope=includes_previous_round cq_time=software_observed\n",
      (unsigned long long)c.iterations, (double)(data_ns / cpu.size() / 1000),
      data_ns > 0 ? (double)((long double)c.rows * c.row_bytes * cpu.size() / data_ns) : 0.0);
  for (std::size_t column = 0; column < kTransferColumnCount; ++column) {
    std::vector<double> values; values.reserve(cpu.size());
    long double sum = 0;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
      const auto value = TransferValues(cpu[i], device[i], factor)[column];
      sum += value; values.push_back(value);
    }
    std::sort(values.begin(), values.end());
    std::fprintf(file, "[AICPU_TRANSFER] STAGE name=%s mean=%.1f p50=%.1f p99=%.1f\n",
        kTransferColumns[column], static_cast<double>(sum / cpu.size()),
        values[static_cast<std::size_t>(std::ceil(values.size() * .50)) - 1],
        values[static_cast<std::size_t>(std::ceil(values.size() * .99)) - 1]);
  }
  for (std::size_t j = 0; j < cpu.front().queues.jettys; ++j) {
    long double rows = 0, wrs = 0, post = 0, cq = 0, tail = 0;
    std::uint64_t tail_waves = 0;
    for (const auto& s : cpu) {
      const auto& q = s.queues.queue[j];
      rows += q.rows; wrs += q.wr_count; post += q.post_ns;
      cq += q.post_to_cq_ns; tail += q.cq_tail_ns; tail_waves += q.tail_waves;
    }
    const double count = static_cast<double>(cpu.size());
    std::fprintf(file, "[AICPU_TRANSFER] JETTY id=%zu avg_rows=%.1f avg_wrs=%.1f "
        "avg_post_us=%.3f avg_post_to_cq_us=%.3f avg_cq_tail_us=%.3f tail_waves=%llu\n",
        j, (double)rows / count, (double)wrs / count, (double)post / count / 1000,
        (double)cq / count / 1000, (double)tail / count / 1000, (unsigned long long)tail_waves);
  }
}

inline void PrintTransferCsv(FILE* file, const Config& c, std::uint64_t frequency,
    const std::vector<CpuTransferSample>& cpu, const std::vector<Sample>& device) {
  ValidateTransferSamples(c, frequency, cpu, device);
  std::fprintf(file, "sequence");
  for (const auto* name : kTransferColumns) std::fprintf(file, ",%s", name);
  const auto jettys = cpu.front().queues.jettys;
  for (std::size_t j = 0; j < jettys; ++j)
    std::fprintf(file, ",jetty_%zu_rows,jetty_%zu_wrs,jetty_%zu_post_ns,jetty_%zu_post_to_cq_ns,"
        "jetty_%zu_cq_tail_ns,jetty_%zu_tail_waves", j, j, j, j, j, j);
  std::fprintf(file, "\n");
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    std::fprintf(file, "%llu", (unsigned long long)cpu[i].sequence);
    for (const auto value : TransferValues(cpu[i], device[i], 1e9 / static_cast<double>(frequency)))
      std::fprintf(file, ",%.1f", value);
    for (std::size_t j = 0; j < jettys; ++j) {
      const auto& q = cpu[i].queues.queue[j];
      std::fprintf(file, ",%llu,%llu,%llu,%llu,%llu,%llu", (unsigned long long)q.rows,
          (unsigned long long)q.wr_count, (unsigned long long)q.post_ns,
          (unsigned long long)q.post_to_cq_ns, (unsigned long long)q.cq_tail_ns,
          (unsigned long long)q.tail_waves);
    }
    std::fprintf(file, "\n");
  }
  Require(std::fflush(file) == 0 && !std::ferror(file), "cannot write transfer profile CSV");
}
}  // namespace kvaf::aicpu_test
