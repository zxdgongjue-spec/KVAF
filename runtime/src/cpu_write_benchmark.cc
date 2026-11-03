#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "cpu_push_get_transport.h"
#include "batch_get_profile.h"
#include "cpu_write_diagnostics.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

namespace kvaf::cpu_push {
Status Transport::BenchmarkWrite(std::uint8_t* memory, std::uint64_t base,
    std::uint64_t capacity, const WriteBenchmarkOptions& o, WriteBenchmarkResult* out,
    const WriteBenchmarkDiagnosticOptions* d, WriteBenchmarkDiagnostics* records) {
  const auto message = std::min(o.message_bytes, max_bytes_);
  const auto source_window = o.source_mode ? o.source_bytes :
      (d && d->source_window_bytes ? d->source_window_bytes : o.payload_bytes);
  const auto target_window = d && d->target_window_bytes ? d->target_window_bytes : o.payload_bytes;
  const auto flight = d ? d->flight_bytes : 0;
  if (!out || !context_ || !max_sge_ || !memory || !o.payload_bytes || o.payload_bytes % 64 ||
      o.payload_bytes > layout_.bytes - layout_.staging || !o.source_bytes || o.source_bytes % 8 ||
      o.source_bytes > capacity || o.source_mode > 1 || !o.queues || o.queues > kJettyCount ||
      o.payload_bytes < o.queues * 64U || message < 8 || message % 8 ||
      !o.post_batch_wrs || o.post_batch_wrs > 8192 || !o.iterations ||
      o.iterations > kWriteBenchmarkSamples || o.warmup > kWriteBenchmarkSamples ||
      (o.source_mode == 0 && o.source_bytes < source_window) ||
      (o.source_mode == 1 && (o.fragment_bytes < 8 || o.fragment_bytes % 8 ||
                              o.fragment_bytes > o.source_bytes)))
    return Status::InvalidArgument("invalid CPU WRITE benchmark sizes/counts; payload must fit registered HBM");
  if ((d == nullptr) != (records == nullptr)) return Status::InvalidArgument("missing diagnostic options/result");
  if (d && (d->reserved || d->cpu < -1 || d->idle_us > 1000000 ||
      o.payload_bytes % (o.queues * 64U) || target_window > o.payload_bytes ||
      target_window % (o.queues * 64U) || target_window < o.queues * 64U ||
      (o.source_mode == 0 && (source_window > o.payload_bytes || source_window % (o.queues * 64U) ||
                              source_window < o.queues * 64U)) ||
      (flight && (flight > o.payload_bytes || flight % (o.queues * 64U) || flight < o.queues * 64U)) ||
      (target_window < o.payload_bytes && (!flight || flight > target_window || target_window % flight)) ||
      (o.source_mode && (d->source_window_bytes || d->target_window_bytes || flight))))
    return Status::InvalidArgument("invalid diagnostic windows: use contiguous mode; target reuse needs a dividing drained flight");
  if (!tracker_.Empty()) return Status::Busy("WRITE benchmark has outstanding DMA");
  for (std::uint32_t index = 0; index < kJettyCount; ++index) {
    const auto& q = queues_[index];
    if (q.poisoned || q.pending || (index < o.queues &&
        (!q.channel.local_jetty || !q.channel.local_jfc)))
      return Status::InvalidState("WRITE benchmark queue is not initialized and idle");
  }
  diagnostic::Affinity affinity;
  if (d) {
    const auto status = affinity.Bind(d->cpu);
    if (!status.ok()) return status;
    *records = {};
    records->source_window_bytes = source_window;
    records->target_window_bytes = target_window;
    records->flight_bytes = flight;
    records->cpu_requested = d->cpu;
    records->idle_us = d->idle_us;
  }
  struct Restore {
    std::uint32_t& target;
    std::uint32_t value;
    ~Restore() { target = value; }
  } restore{post_batch_wrs_, post_batch_wrs_};
  post_batch_wrs_ = o.post_batch_wrs;
  prepared_.clear();
  prepared_count_ = UINT32_MAX;
  *out = {};
  out->hbm_address = hbm_base_ + layout_.staging;
  out->payload_bytes = o.payload_bytes; out->source_bytes = o.source_bytes;
  out->message_bytes = message; out->max_sge = max_sge_;
  out->queues = o.queues; out->post_batch_wrs = post_batch_wrs_;
#ifdef __OPTIMIZE__
  out->optimized = 1;
#endif
  // Preparation is deliberately outside every transfer interval.
  for (std::uint64_t i = 0; i < o.source_bytes / 8; ++i) {
    const auto word = WriteBenchmarkMix(o.seed + i);
    std::memcpy(memory + i * 8, &word, 8);
  }
  diagnostic::Memory memory_info;
  const int initial_cpu = d ? sched_getcpu() : -1;
  if (d) memory_info = diagnostic::ReadMemory(memory, source_window);
  std::array<std::vector<PreparedQueue>, kJettyCount> shards;
  const auto sge_limit = o.source_mode == 0 ? 1U : max_sge_;
  for (std::uint32_t q = 0; q < o.queues; ++q) {
    auto& waves = shards[q];
    waves.emplace_back();
    auto cursor = o.payload_bytes / 64 * q / o.queues * 64;
    const auto queue_begin = cursor;
    const auto end = o.payload_bytes / 64 * (q + 1) / o.queues * 64;
    const auto source_begin = o.source_mode ? 0 : source_window / 64 * q / o.queues * 64;
    const auto source_slice = o.source_mode ? source_window :
        source_window / 64 * (q + 1) / o.queues * 64 - source_begin;
    const auto target_begin = target_window / 64 * q / o.queues * 64;
    const auto target_slice = target_window / 64 * (q + 1) / o.queues * 64 - target_begin;
    const auto flight_slice = flight ? flight / o.queues : end - queue_begin;
    while (cursor < end) {
      const auto relative = cursor - queue_begin;
      if (waves.back().local.size() + sge_limit > 8192 || (relative && relative % flight_slice == 0))
        waves.emplace_back();
      auto& queue = waves.back();
      const auto destination = out->hbm_address + target_begin + relative % target_slice;
      const auto wr_limit = std::min<std::uint64_t>({message, end - cursor,
          target_slice - relative % target_slice, flight_slice - relative % flight_slice});
      std::uint32_t bytes = 0, sges = 0;
      while (cursor < end && bytes < wr_limit && sges < sge_limit) {
        const auto current = cursor - queue_begin;
        auto offset = source_begin + current % source_slice;
        auto length = wr_limit - bytes;
        if (!o.source_mode) length = std::min<std::uint64_t>(length, source_slice - current % source_slice);
        if (o.source_mode) {
          const auto within = cursor % o.fragment_bytes;
          offset = WriteBenchmarkMix(o.seed + cursor / o.fragment_bytes) %
              (o.source_bytes / o.fragment_bytes) * o.fragment_bytes + within;
          length = std::min<std::uint64_t>(length, o.fragment_bytes - within);
        }
        if (++out->source_sges > kWriteBenchmarkMaxSges)
          return Status::InvalidArgument("WRITE benchmark exceeds 1048576 source SGEs; reduce payload or increase fragment/message size");
        queue.local.push_back({base + offset, static_cast<std::uint32_t>(length), source_, nullptr});
        bytes += length; ++sges; cursor += length;
      }
      AppendPrepared(queue, q, destination, bytes, sges);
      ++out->data_wrs;
    }
  }
  std::size_t waves = 0;
  for (const auto& shard : shards) waves = std::max(waves, shard.size());
  for (std::size_t w = 0; w < waves; ++w) {
    auto plan = std::make_unique<PreparedWrite>();
    for (std::uint32_t q = 0; q < o.queues; ++q)
      if (w < shards[q].size()) plan->queues[q] = std::move(shards[q][w]);
    SealPrepared(*plan);
    if (plan->fence_count) prepared_.push_back(std::move(plan));
  }
  out->waves = prepared_.size();
  std::uint64_t planned_bytes = 0;
  for (const auto& plan : prepared_)
    for (const auto& queue : plan->queues)
      for (const auto& remote : queue.remote) planned_bytes += remote.len;
  if (planned_bytes != o.payload_bytes) return Status::Internal("WRITE plan byte count differs from requested traffic");
  const auto cgroup_before = d ? diagnostic::ReadCgroup() : diagnostic::Cgroup{};
  std::uint64_t previous_finish = 0;
  for (std::uint32_t iteration = 0; iteration < o.warmup + o.iterations; ++iteration) {
    WriteBenchmarkSample sample{};
    Profile polled{};
    std::uint64_t first = 0, finished = 0;
    std::uint64_t thread_begin = 0;
    rusage usage_begin{};
    int cpu_start = -1;
    for (std::size_t w = 0; w < prepared_.size(); ++w) {
      auto& plan = *prepared_[w];
      auto status = BeginPrepared(plan, w + 1);
      if (!status.ok()) return status;
      if (d && w == 0) {
        if (getrusage(RUSAGE_THREAD, &usage_begin)) return Status::Internal("getrusage diagnostic failed");
        cpu_start = sched_getcpu();
        thread_begin = diagnostic::ThreadCpuNs();
        if (thread_begin == UINT64_MAX) return Status::Internal("thread CPU clock diagnostic failed");
      }
      const auto begin = BatchGetProfileNowNs();
      if (!first) first = begin;
      status = PostPrepared(plan);
      const auto posted = BatchGetProfileNowNs();
      if (!status.ok()) return status;  // Keep in-flight WR/SGL storage owned.
      status = PollPrepared(w + 1, &polled);
      finished = BatchGetProfileNowNs();
      if (!status.ok()) return status;
      sample.post_ns += posted - begin;
      sample.cq_wait_ns += finished - posted;
    }
    sample.comm_ns = finished - first;  // Includes gaps between drained waves.
    sample.tail_cqes = polled.tail_cqes;
    if (d) {
      const auto thread_end = diagnostic::ThreadCpuNs();
      rusage usage_end{};
      const int cpu_end = sched_getcpu();
      if (thread_end == UINT64_MAX || getrusage(RUSAGE_THREAD, &usage_end))
        return Status::Internal("thread resource diagnostic failed");
      if (iteration >= o.warmup) {
        auto& record = records->samples[out->count];
        record.thread_cpu_ns = thread_end - thread_begin;
        record.voluntary_switches = usage_end.ru_nvcsw - usage_begin.ru_nvcsw;
        record.involuntary_switches = usage_end.ru_nivcsw - usage_begin.ru_nivcsw;
        record.minor_faults = usage_end.ru_minflt - usage_begin.ru_minflt;
        record.major_faults = usage_end.ru_majflt - usage_begin.ru_majflt;
        record.cpu_start = cpu_start; record.cpu_end = cpu_end;
        record.inter_sample_gap_ns = previous_finish ? first - previous_finish : 0;
      }
    }
    if (iteration >= o.warmup) out->samples[out->count++] = sample;
    previous_finish = finished;
    if (d && d->idle_us) std::this_thread::sleep_for(std::chrono::microseconds(d->idle_us));
  }
  if (d) {
    const auto environment = diagnostic::Environment(memory_info, initial_cpu, cgroup_before, diagnostic::ReadCgroup());
    std::snprintf(records->environment, sizeof(records->environment), "%s", environment.c_str());
  }
  return Status::OK();
}
}  // namespace kvaf::cpu_push
