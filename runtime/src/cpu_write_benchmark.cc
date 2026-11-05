#include "cpu_push_get_transport.h"
#include "batch_get_profile.h"
#include <algorithm>
#include <cstring>

namespace kvaf::cpu_push {
Status Transport::BenchmarkWrite(std::uint8_t* memory, std::uint64_t base,
    std::uint64_t capacity, const WriteBenchmarkOptions& o, WriteBenchmarkResult* out) {
  const auto message = std::min(o.message_bytes, max_bytes_);
  if (!out || !context_ || !max_sge_ || !memory || !o.payload_bytes || o.payload_bytes % 64 ||
      o.payload_bytes > layout_.bytes - layout_.staging || !o.source_bytes || o.source_bytes % 8 ||
      o.source_bytes > capacity || o.source_mode > 1 || !o.queues || o.queues > kJettyCount ||
      o.payload_bytes < o.queues * 64U || message < 8 || message % 8 ||
      !o.post_batch_wrs || o.post_batch_wrs > 8192 || !o.iterations ||
      o.iterations > kWriteBenchmarkSamples || o.warmup > kWriteBenchmarkSamples ||
      (o.source_mode == 0 && o.source_bytes < o.payload_bytes) ||
      (o.source_mode == 1 && (o.fragment_bytes < 8 || o.fragment_bytes % 8 ||
                              o.fragment_bytes > o.source_bytes)))
    return Status::InvalidArgument("invalid CPU WRITE benchmark sizes/counts; payload must fit registered HBM");
  if (!tracker_.Empty()) return Status::Busy("WRITE benchmark has outstanding DMA");
  for (std::uint32_t index = 0; index < kJettyCount; ++index) {
    const auto& q = queues_[index];
    if (q.poisoned || (index < o.queues &&
        (!q.channel.local_jetty || !q.channel.local_jfc)))
      return Status::InvalidState("WRITE benchmark queue is not initialized and idle");
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
  std::array<std::vector<PreparedQueue>, kJettyCount> shards;
  const auto sge_limit = o.source_mode == 0 ? 1U : max_sge_;
  for (std::uint32_t q = 0; q < o.queues; ++q) {
    auto& waves = shards[q];
    waves.emplace_back();
    auto cursor = o.payload_bytes / 64 * q / o.queues * 64;
    const auto end = o.payload_bytes / 64 * (q + 1) / o.queues * 64;
    while (cursor < end) {
      if (waves.back().local.size() + sge_limit > 8192) waves.emplace_back();
      auto& queue = waves.back();
      const auto destination = out->hbm_address + cursor;
      std::uint32_t bytes = 0, sges = 0;
      while (cursor < end && bytes < message && sges < sge_limit) {
        auto offset = cursor;
        auto length = std::min<std::uint64_t>(end - cursor, message - bytes);
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
  for (std::uint32_t iteration = 0; iteration < o.warmup + o.iterations; ++iteration) {
    WriteBenchmarkSample sample{};
    Profile polled{};
    std::uint64_t first = 0, finished = 0;
    for (std::size_t w = 0; w < prepared_.size(); ++w) {
      auto& plan = *prepared_[w];
      auto status = BeginPrepared(plan, w + 1);
      if (!status.ok()) return status;
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
    if (iteration >= o.warmup) out->samples[out->count++] = sample;
  }
  return Status::OK();
}
}  // namespace kvaf::cpu_push
