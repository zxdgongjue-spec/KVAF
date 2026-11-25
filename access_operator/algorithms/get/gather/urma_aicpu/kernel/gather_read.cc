#include "gather_read.h"

#include <unistd.h>

#include <atomic>

#include "cpu_gather.h"
#include "gather_pull_plan.h"
#include "transport_internal.h"
#include "aicpu_urma/cache.h"

namespace kvaf::aicpu_urma {
namespace {
using namespace detail;


template <typename T>
void EnsureSlots(std::vector<T>& slots, std::size_t count) {
  if (count > slots.size()) slots.resize(count);
}

void BeginGatherWave(RequestBuffer& queue, std::uint32_t sgl_size, std::uint32_t wave_wrs) {
  EnsureSlots(queue.wrs, wave_wrs);
  EnsureSlots(queue.local_sges, queue.wrs.size() * sgl_size);
  EnsureSlots(queue.remote_sges, queue.wrs.size());
  queue.wr_count = 0U;
}

// Resolve this request's fixed addresses and registration once, before batches.
// Only allocation capacity survives a request; no WR/layout contents are cached.
struct GatherBuildLayout {
  std::uint64_t output, staging;
  std::uint32_t sgl_size, token_bytes;
  const std::int32_t* rows;
  urma_target_seg_t* local_segment;
  urma_target_seg_t* remote_segment;
  urma_jfs_wr_t prototype{};

  GatherBuildLayout(Queue& worker, const GatherReadArgs& args,
                    const AicpuBatchGetInput& request)
      : output(args.output),
        staging(worker.route->remote.handle()->seg.ubva.va +
                (args.remote_staging - worker.route->remote.canonical())),
        sgl_size(std::min(args.sgl_size, kvaf::kGatherMaxSglSize)),
        token_bytes(request.token_bytes),
        rows(reinterpret_cast<const std::int32_t*>(request.output_indices)),
        local_segment(LocalSegment(args.output, args.output_extent)),
        remote_segment(worker.route->remote.handle()) {
    prototype.opcode = URMA_OPC_READ;
    prototype.flag.bs.comp_order = 1;
    prototype.flag.bs.place_order = URMA_RELAX_ORDER;
    prototype.tjetty = worker.target;
    prototype.rw.src.num_sge = 1U;
  }
};

template <std::uint32_t FixedSgl = 0>
void BuildReadBatch(Queue& worker, const GatherBuildLayout& layout, std::uint32_t& span_index,
                    std::uint32_t last, std::uint64_t& source_offset,
                    std::uint32_t batch_wrs, std::uint32_t wave_wrs, std::uint32_t& local_index,
                    urma_target_jetty_t* target) {
  auto& queue = worker.requests;
  auto index = span_index;
  auto source_position = source_offset;
  auto* local = queue.local_sges.data();
  auto* remote = queue.remote_sges.data();
  auto* requests = queue.wrs.data();
  auto sequence = worker.sequence;
  auto wr_count = queue.wr_count;
  const auto wr_limit = wr_count + std::min(batch_wrs, wave_wrs - wr_count);
  auto prototype = layout.prototype;
  prototype.tjetty = target;
  while (index < last && wr_count < wr_limit) {
    const auto wr_index = wr_count++;
    const auto local_begin = local_index;
    const auto source = layout.staging + source_position;
    // The full-batch path has a compile-time SGL length, so the compiler can
    // unroll the six row lookups. Each SGE holds one complete output token.
    const auto local_count = FixedSgl ? FixedSgl : std::min(last - index, layout.sgl_size);
    // The validated per-lane payload fits uint32, including every READ within it.
    const auto bytes = local_count * layout.token_bytes;
    for (std::uint32_t row = 0; row < local_count; ++row) {
      const auto target = layout.output + std::uint64_t(layout.rows[index + row]) * layout.token_bytes;
      local[local_begin + row] = {target, layout.token_bytes, layout.local_segment, nullptr};
    }
    index += local_count;
    local_index += local_count;
    source_position += bytes;
    remote[wr_index] = {source, bytes, layout.remote_segment, nullptr};
    auto& wr = requests[wr_index];
    wr = prototype;
    wr.user_ctx = ++sequence;
    wr.rw.src.sge = reinterpret_cast<urma_sge_t*>(&remote[wr_index]);
    wr.rw.dst.sge = reinterpret_cast<urma_sge_t*>(local + local_begin);
    wr.rw.dst.num_sge = local_count;
    wr.next = requests + wr_index + 1U;
  }
  if (wr_count != 0U) {
    auto& tail = requests[wr_count - 1U];
    tail.next = nullptr;
    const bool final = index == last || wr_count == wave_wrs;
    tail.flag.bs.place_order = final ? URMA_STRONG_ORDER : URMA_RELAX_ORDER;
    tail.flag.bs.complete_enable = final;
  }
  queue.wr_count = wr_count;
  worker.sequence = sequence;
  span_index = index;
  source_offset = source_position;
}

struct GroupedRequest {
  GatherPullPlan plan;
  UmdkTransfer head[3]{};
  std::uint32_t updates{}, address_groups_per_post{};
  bool chunked{};
  bool enabled() const { return plan.head_rows != 0; }
};

struct GatherPrefix {
  const GatherReadArgs& args;
  GatherReadStatus& gather;
  Queue* control;
  BatchGetWorkerProfile* profile;
  GroupedRequest& grouped;
  std::uint64_t result_word;
  std::uint64_t ready_bytes{};
  std::uint64_t query_start{};

  // The caller owns this lane's control mutex through the matching CQE.
  bool Begin(UmdkControlStatus& error) {
    gather.phase = kWaitReadyPrefix;
    const UmdkTransfer word{
        args.remote_descriptor + offsetof(kvaf::peer::GatherDescriptorHeader, ready_bytes),
        result_word, sizeof(ready_bytes)};
    if (profile) ++profile->ready_queries;
    query_start = NowNs();
    if (PostOne(*control, false, word, error)) return true;
    gather.result = kGatherTransportFailed;
    return false;
  }

  bool Finish(UmdkControlStatus& error) {
    if (!WaitCompletion(*control, control->single_wr.user_ctx, query_start, 1000, error)) {
      gather.result = kGatherTransportFailed;
      return false;
    }
    FlushUmdkCache(reinterpret_cast<const void*>(result_word), sizeof(ready_bytes));
    const auto progress = *reinterpret_cast<const volatile std::uint64_t*>(result_word);
    if (progress < ready_bytes || progress > args.bytes || (progress & kvaf::peer::kGatherErrorBit)) {
      gather.ack_seen = progress;
      gather.result = kCpuGatherFailed;
      return Fail(error, "CPU ready prefix error");
    }
    if (grouped.enabled() && progress > ready_bytes) {
      // Polls may skip intermediate groups. Only a complete CPU group grants READs.
      if (!grouped.plan.IsReadyBoundary(progress)) {
        gather.ack_seen = progress;
        gather.result = kCpuGatherFailed;
        return Fail(error, "unexpected grouped ready boundary");
      }
      ++grouped.updates;
    }
    if (profile && !grouped.enabled()) {
      if (profile->ready_queries == 1)
        gather.ready_profile_first = static_cast<std::uint32_t>(progress);
      else if (progress == ready_bytes)
        ++gather.ready_profile_second;
    }
    ready_bytes = progress;
    gather.phase = kReadStaging;
    return true;
  }
};

// Each signaled batch tail is ordered after earlier submissions on its Jetty.
// Keep its CQE context until consumed. WR/SGL storage lives for the whole wave.
struct PendingBatches {
  std::array<std::uint64_t, 2> contexts{};
  unsigned count{};

  bool RetireOldest(Queue& data, std::uint64_t start, UmdkControlStatus& error) {
    if (!WaitCompletion(data, contexts[0], start, 1000, error)) return false;
    contexts[0] = contexts[1];
    --count;
    return true;
  }
};

// Mode 2 submits fixed-size posts up to the bounded WR wave. Each used data
// Jetty signals only its last post in that wave.
struct DataWaveTails {
  const GatherReadArgs& args;
  const std::uint64_t start;
  const std::uint32_t jettys, posts;
  std::array<std::uint64_t, kUmdkMaxDataJettys> contexts{};
  std::array<bool, kUmdkMaxDataJettys> touched{};
  bool retired{};

  bool IsTail(std::uint32_t submitted) const {
    return submitted / args.sq_batch_wqes + jettys >= posts;
  }

  bool Drain(UmdkControlStatus& error) {
    retired = true;
    bool ok = true, scratch_available = true;
    for (std::uint32_t j = 0; j < jettys; ++j) {
      if (!touched[j]) continue;
      auto& data = DataQueue(args.worker_index + j * args.worker_count);
      UmdkControlStatus completion{};
      bool done = false;
      if (!data.poisoned && contexts[j]) {
        done = WaitCompletion(data, contexts[j], start, 1000, completion);
      } else if (!data.poisoned && scratch_available) {
        // If a failure prevented the tail from being posted, fence the
        // earlier READs with a mailbox READ into a separate scratch word.
        const UmdkTransfer fence{args.remote_ack, args.scratch + 48U, 8U};
        const auto fence_start = NowNs();
        done = PostOne(data, false, fence, completion, true) &&
               WaitCompletion(data, data.single_wr.user_ctx, fence_start, 1000, completion);
        scratch_available = done;  // A failed fence may still own the DMA destination.
      }
      if (!done) {
        data.poisoned = true;
        if (!error.result && completion.result) error = completion;
        ok = false;
      }
    }
    return ok;
  }

  ~DataWaveTails() {
    if (posts && !retired) {
      UmdkControlStatus cleanup{};
      Drain(cleanup);
    }
  }
};

void WaitAck(const GatherReadArgs& args, GatherReadStatus& status);

void BuildSourceSpans(const AicpuBatchGetInput&, std::uint32_t, std::uint32_t, peer::GatherSpan*);

// One pending address chain per worker. Build the next window in a disjoint
// descriptor range while the previous chain is in flight.
struct AddressBlocks {
  const GatherPullPlan& plan;
  std::uint32_t published, groups_per_post;

  std::uint32_t BuildNext(const GatherReadArgs& args, const AicpuBatchGetInput& request,
                          std::uint32_t lane_first, BatchGetWorkerProfile* profile, bool startup = false) const {
    const auto groups = startup ? 1U : groups_per_post;
    const auto rows = std::min<std::uint64_t>(
        std::uint64_t{groups} * plan.round_rows, plan.total_rows - published);
    const auto end = published + static_cast<std::uint32_t>(rows);
    if (end == published) return end;
    const auto offset = peer::GatherDescriptorBytes(published, true);
    auto* spans = reinterpret_cast<peer::GatherSpan*>(args.source_descriptor + offset);
    BatchGetProfileScope lookup(profile ? &profile->request_ns : nullptr);
    BuildSourceSpans(request, lane_first + published, lane_first + end, spans);
    BatchGetProfileScope flush(profile ? &profile->descriptor_cache_ns : nullptr);
    FlushUmdkCache(spans, (end - published) * sizeof(*spans));
    return end;
  }

  bool Post(Queue& control, const GatherReadArgs& args, std::uint32_t end,
            UmdkControlStatus& error, BatchGetWorkerProfile* profile) {
    BatchGetProfileScope publish(profile ? &profile->notify_ns : nullptr);
    // Publish this address window before its cumulative row count on the same Jetty.
    auto* marker = reinterpret_cast<std::uint64_t*>(args.scratch + 32U);
    *marker = end;
    const auto offset = peer::GatherDescriptorBytes(published, true);
    const UmdkTransfer writes[] = {
        {args.remote_descriptor + offset, args.source_descriptor + offset,
         std::uint64_t{end - published} * sizeof(peer::GatherSpan)},
        {args.remote_notify + offsetof(peer::GatherControl, suffix_sequence),
         reinterpret_cast<std::uint64_t>(marker), sizeof(*marker)}};
    FlushUmdkCache(marker, sizeof(*marker));
    if (!PostOrderedWrites(control, writes, 2U, error)) return false;
    published = end;
    return true;
  }
};

// The head resets the previous sequence's progress. Retire it before the
// first ready read; use its DMA time to resolve the disjoint tail addresses.
bool PublishGrouped(Queue& control, const GroupedRequest& grouped, const GatherReadArgs& args,
                    const AicpuBatchGetInput& request, std::uint32_t first, std::uint32_t last,
                    UmdkControlStatus& error, BatchGetWorkerProfile* profile) {
  const auto start = NowNs();
  {
    BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
    if (!PostOrderedWrites(control, grouped.head, 3, error)) return false;
  }
  const auto tail = first + grouped.plan.head_rows;
  const auto offset = grouped.head[0].bytes;
  const auto tail_bytes = std::uint64_t{last - tail} * sizeof(peer::GatherSpan);
  if (tail < last) {
    BatchGetProfileScope timer(profile ? &profile->request_ns : nullptr);
    auto* spans = reinterpret_cast<peer::GatherSpan*>(args.source_descriptor + offset);
    BuildSourceSpans(request, tail, last, spans);
    BatchGetProfileScope flush(profile ? &profile->descriptor_cache_ns : nullptr);
    FlushUmdkCache(spans, tail_bytes);
  }
  BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
  if (!WaitCompletion(control, control.requests.wrs.back().user_ctx, start, 1000, error)) return false;
  if (tail == last) return true;
  const UmdkTransfer writes[] = {
      {args.remote_descriptor + offset, args.source_descriptor + offset, tail_bytes},
      {args.remote_notify + offsetof(peer::GatherControl, suffix_sequence), args.scratch, 8U}};
  return PostOrderedWrites(control, writes, 2, error);
}

bool ReadBatches(Queue& worker, GroupedRequest& grouped, const AicpuBatchGetInput& request,
                 const GatherReadArgs& args, GatherReadStatus& status,
                 std::uint32_t first, std::uint32_t last, UmdkControlStatus& error,
                 BatchGetWorkerProfile* profile) {
  if (first == last) return true;
  const auto setup_start = profile ? BatchGetProfileNowNs() : 0U;
  const GatherBuildLayout layout(worker, args, request);
  if (!layout.local_segment) return Fail(error, "READ output is not registered");
  Queue* control = nullptr;
  std::unique_lock<std::mutex> control_lock;
  if (args.pipeline_chunk_bytes) {
    control_lock = std::unique_lock<std::mutex>(active_session->control_mutexes[args.worker_index]);
    control = FindQueue(kUmdkControlQueueIndex + args.worker_index);
    if (!control || control->poisoned || !control->target || !control->route || !control->route->remote.handle())
      return Fail(error, "control queue is unavailable");
  }
  const auto result_word = grouped.enabled() ?
      args.source_descriptor + offsetof(peer::GatherDescriptorHeader, ready_bytes) : args.scratch;
  GatherPrefix prefix{args, status, control, profile, grouped, result_word};
  const auto lane_first = first;
  AddressBlocks addresses{grouped.plan, grouped.plan.head_rows, grouped.address_groups_per_post};
  bool tail_pending = false;
  std::uint64_t tail_start = 0;
  const auto retire_tail = [&] (UmdkControlStatus& result) {
    if (!tail_pending) return true;
    tail_pending = false;
    BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
    return WaitCompletion(*control, control->requests.wrs.back().user_ctx, tail_start, 1000, result);
  };
  if (profile) profile->read_setup_ns += BatchGetProfileNowNs() - setup_start;
  const auto max_wrs = (args.source_span_count + layout.sgl_size - 1U) / layout.sgl_size;
  const auto wave_wrs = static_cast<std::uint32_t>(std::min<std::uint64_t>(args.read_wave_wrs, max_wrs));
  std::uint64_t remote_offset = 0;
  while (first < last) {
    auto& queue = worker.requests;
    {
      BatchGetProfileScope timer(profile ? &profile->read_setup_ns : nullptr);
      BeginGatherWave(queue, layout.sgl_size, wave_wrs);
    }
    if (grouped.enabled() && !remote_offset) {
      if (grouped.chunked) {
        {
          BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
          tail_start = NowNs();
          if (!PostOrderedWrites(*control, grouped.head, 3, error)) return false;
          tail_pending = true;
        }
        const auto end = addresses.BuildNext(args, request, lane_first, profile, /*startup=*/true);
        // The head resets ready_bytes and the row marker for this sequence.
        if (!retire_tail(error)) return false;
        if (end != addresses.published) {
          tail_start = NowNs();
          if (!addresses.Post(*control, args, end, error, profile)) return false;
          tail_pending = true;
        }
      } else {
        if (!PublishGrouped(*control, grouped, args, request, first, last, error, profile)) return false;
        tail_pending = grouped.plan.head_rows < grouped.plan.total_rows;
        tail_start = NowNs();
      }
    }
    std::uint32_t local_index = 0;
    const auto start = NowNs();
    const auto jettys = CurrentState().data_jettys;
    const auto remaining_wrs = (last - first + layout.sgl_size - 1U) / layout.sgl_size;
    const auto wrs_this_wave = std::min<std::size_t>(remaining_wrs, wave_wrs);
    const auto posts_this_wave = grouped.chunked ?
        1U + (static_cast<std::uint32_t>(wrs_this_wave) - 1U) / args.sq_batch_wqes : 0U;
    DataWaveTails data_tails{args, start, jettys, posts_this_wave};
    std::array<PendingBatches, kUmdkMaxDataJettys> pending{};
    std::uint32_t next_jetty = 0;
    std::uint32_t submitted = 0, polls = 0;
    while (submitted < queue.wr_count || (first < last && queue.wr_count < wave_wrs)) {
      // Submit complete WRs as soon as their entire source range is ready.
      // PostBatch temporarily terminates the chain; the rest stays encoded.
      const auto prepared = queue.wr_count - submitted;
      auto ready_end = submitted + std::min(prepared, args.sq_batch_wqes);
      if (control && prefix.ready_bytes < remote_offset) {
        const auto limit = ready_end;
        ready_end = submitted;
        while (ready_end < limit) {
          const auto& source = queue.remote_sges[ready_end];
          if (source.addr - layout.staging + source.len > prefix.ready_bytes) break;
          ++ready_end;
        }
      }
      if (grouped.enabled() && ready_end - submitted < args.sq_batch_wqes) {
        const bool wave_end = ready_end == queue.wr_count && (first == last || queue.wr_count == wave_wrs);
        if (!wave_end) ready_end = submitted;
      }
      if (submitted < ready_end) {
        auto& data = DataQueue(args.worker_index + next_jetty * args.worker_count);
        auto& inflight = pending[next_jetty];
        if (!grouped.chunked && inflight.count == inflight.contexts.size()) {
          BatchGetProfileScope wait(profile ? &profile->cq_wait_ns : nullptr);
          if (!inflight.RetireOldest(data, start, error)) return false;
        }
        BatchGetProfileScope timer(profile ? &profile->post_ns : nullptr);
        auto& tail = queue.wrs[ready_end - 1U];
        if (jettys > 1 || grouped.enabled()) {
          // Chunked batches bind their Jetty while building. Other modes may
          // submit partial batches, so they retain late binding.
          if (!grouped.chunked)
            for (auto i = submitted; i < ready_end; ++i) queue.wrs[i].tjetty = data.target;
          const bool signal = !grouped.chunked || data_tails.IsTail(submitted);
          tail.flag.bs.complete_enable = signal;
          tail.flag.bs.place_order = signal ? URMA_STRONG_ORDER : URMA_RELAX_ORDER;
        }
        if (grouped.chunked) data_tails.touched[next_jetty] = true;
        if (!PostBatch(data, &queue.wrs[submitted], &tail, error)) return false;
        if (tail.flag.bs.complete_enable) {
          if (grouped.chunked) data_tails.contexts[next_jetty] = tail.user_ctx;
          else inflight.contexts[inflight.count++] = tail.user_ctx;
        }
        next_jetty = (next_jetty + 1) % jettys;
        if (profile) {
          if (!profile->post_calls) profile->first_post_wrs = ready_end - submitted;
          profile->read_wqes += ready_end - submitted;
          ++profile->post_calls;
        }
        submitted = ready_end;
        polls = 0;
        continue;
      }

      // At most two batches prepared ahead. Bound the next batch without
      // walking output_indices twice. The control and data WRs have separate storage.
      const bool can_build = first < last && queue.wr_count < wave_wrs &&
                             prepared <= args.sq_batch_wqes;
      const auto slots = std::min<std::size_t>(args.sq_batch_wqes, wave_wrs - queue.wr_count);
      const auto max_wr_bytes = std::uint64_t(layout.token_bytes) * layout.sgl_size;
      const auto batch_bound = std::min<std::uint64_t>(args.bytes - remote_offset, slots * max_wr_bytes);
      const bool query = control && (!grouped.enabled() || prefix.ready_bytes < args.bytes) &&
                         (prepared || prefix.ready_bytes < remote_offset + batch_bound);
      if (query) {
        BatchGetProfileScope timer(profile ? &profile->ready_wait_ns : nullptr);
        if (++polls > 10000U) {
          status.result = kGatherTimeout;
          status.ack_seen = prefix.ready_bytes;
          errno = ETIMEDOUT;
          return Fail(error, "waiting for CPU ready prefix timed out", ETIMEDOUT);
        }
        if (!prefix.Begin(error)) {
          // Local validation can reject the query before posting. Retire the preceding WRITE.
          UmdkControlStatus cleanup{};
          if (!control->poisoned) retire_tail(cleanup);
          return false;
        }
      }
      if (can_build) {
        BatchGetProfileScope timer(profile ? &profile->wr_build_ns : nullptr);
        // Join only an unsent tail. A later prefix submission may span both batches.
        if (prepared) queue.wrs[queue.wr_count - 1U].next = &queue.wrs[queue.wr_count];
        auto* target = worker.target;
        if (grouped.chunked) {
          const auto ahead = prepared / args.sq_batch_wqes;
          target = DataQueue(args.worker_index +
              ((next_jetty + ahead) % jettys) * args.worker_count).target;
        }
        const auto build_wrs = std::min<std::size_t>(args.sq_batch_wqes, wave_wrs - queue.wr_count);
        if (layout.sgl_size == kGatherMaxSglSize &&
            last - first >= build_wrs * kGatherMaxSglSize)
          BuildReadBatch<kGatherMaxSglSize>(worker, layout, first, last, remote_offset,
                                           args.sq_batch_wqes, wave_wrs, local_index, target);
        else
          BuildReadBatch(worker, layout, first, last, remote_offset,
                         args.sq_batch_wqes, wave_wrs, local_index, target);
      }
      if (query) {
        const auto end = grouped.chunked ? addresses.BuildNext(args, request, lane_first, profile)
                                         : addresses.published;
        if (!retire_tail(error)) return false;  // This write precedes the query on the control CQ.
        if (end != addresses.published) {
          // Queue the next address chain while the current ready READ is in
          // flight, then consume the READ's CQE before the new chain's CQE.
          tail_start = NowNs();
          if (!addresses.Post(*control, args, end, error, profile)) {
            UmdkControlStatus cleanup{};
            if (!control->poisoned)
              WaitCompletion(*control, control->single_wr.user_ctx, prefix.query_start, 1000, cleanup);
            return false;
          }
          tail_pending = true;
        }
        bool ready;
        {
          BatchGetProfileScope timer(profile ? &profile->ready_wait_ns : nullptr);
          ready = prefix.Finish(error);
        }
        if (!ready) {
          UmdkControlStatus cleanup{};
          if (!control->poisoned) retire_tail(cleanup);
          if (grouped.enabled() && status.result == kCpuGatherFailed) break;  // Drain posted READs.
          return false;
        }
      }
    }
    if (grouped.enabled() && first == last && !status.result) {
      // Dispatch of the final WR requires total bytes. The grouped CPU releases
      // this boundary only after its final ACK. Still drain every data Jetty.
      if (!retire_tail(error)) return false;
      status.ack_seen = args.sequence;
    } else if (control && first == last && !status.result) {
      // All data WRs are posted. WaitAck takes the control lock itself and
      // can now run while the last data READs are still in flight.
      control_lock.unlock();
      status.phase = kWaitCpuAck;
      BatchGetProfileScope timer(profile ? &profile->ack_ns : nullptr);
      WaitAck(args, status);
      if (!status.result) status.phase = kReadStaging;
    }
    // Drain all data queues even if the final ACK reported an error.
    {
      BatchGetProfileScope timer(profile ? &profile->cq_wait_ns : nullptr);
      if (grouped.chunked && !data_tails.Drain(error)) return false;
      for (std::uint32_t j = 0; !grouped.chunked && j < jettys; ++j) {
        auto& data = DataQueue(args.worker_index + j * args.worker_count);
        while (pending[j].count)
          if (!pending[j].RetireOldest(data, start, error)) return false;
      }
    }
    if (status.result) {
      if (grouped.enabled()) return false;  // Finish recorded the failed group before draining data.
      return Fail(error, "final CPU ACK failed", status.result);
    }
  }
  return true;
}

std::uint32_t ReadStaging(ExecutionSession& session, GroupedRequest& grouped, const AicpuBatchGetInput& request,
                         const GatherReadArgs& args, GatherReadStatus& status,
                         std::uint32_t first, std::uint32_t last,
                         BatchGetWorkerProfile* profile) {
  const auto worker_index = args.worker_index;
  std::lock_guard<std::mutex> worker_lock(session.worker_mutexes[worker_index]);
  auto& worker = DataQueue(worker_index);
  if (worker.poisoned || !worker.target || !worker.route || !worker.route->remote.handle()) return kGatherTransportFailed;
  UmdkControlStatus error{};
  if (!ReadBatches(worker, grouped, request, args, status, first, last, error, profile)) {
    worker.poisoned = true;  // A previously submitted prefix may still be in flight.
    std::fprintf(stderr,
                 "GATHER_UMDK_READ_ERROR worker=%u stage=%s result=%d errno=%d cq_status=%d "
                 "cq_context=%llu\n",
                 worker_index, error.stage, error.result, error.error_number,
                 error.cq_status, static_cast<unsigned long long>(error.cq_context));
    status.cqe_word = error.cq_status != 0 ? error.cq_status : error.error_number;
    return status.result != 0 ? static_cast<std::uint32_t>(status.result) : kGatherTransportFailed;
  }
  return 0;
}

void Flush(const void* address, std::uint64_t bytes) { FlushUmdkCache(address, bytes); }

void WaitAck(const GatherReadArgs& args, GatherReadStatus& status) {
  const UmdkTransfer transfer{args.remote_ack, args.scratch + 8U, 8U};
  for (unsigned poll = 0; poll < 10000U; ++poll) {
    status.result = TransferControlSync(false, &transfer, 1U, args.worker_index);
    if (status.result) return;
    // READ DMA replaced the ACK word in this running kernel, without a CANN handoff.
    Flush(reinterpret_cast<void*>(transfer.local), transfer.bytes);
    status.ack_seen = *reinterpret_cast<const volatile std::uint64_t*>(transfer.local);
    if (status.ack_seen == args.sequence) return;
    if (status.ack_seen == (args.sequence | kvaf::peer::kGatherErrorBit)) {
      status.result = kCpuGatherFailed;
      return;
    }
  }
  status.result = kGatherTimeout;
}

// Build only the source descriptor consumed by the remote CPU. READ destinations
// come directly from the caller's output_indices; no target list is materialized.
void BuildSourceSpans(const AicpuBatchGetInput& request, std::uint32_t first,
                      std::uint32_t last, peer::GatherSpan* out) {
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(request.address_table);
  const auto* indices = reinterpret_cast<const std::int32_t*>(request.indices);
  const auto* offsets = reinterpret_cast<const std::int32_t*>(request.offsets);
  // Input tensors are ready before this CANN kernel starts, as in the control
  // and CPU-push kernels. AICPU only reads them; no per-request cache sweep.
  if (!request.indices) {
    for (auto i = first; i < last; ++i)
      *out++ = {addresses[i], request.token_bytes, 0U};
  } else {
    // Overlap independent random table loads. Only the active slice has valid indices.
    constexpr std::uint32_t lookahead = 16;
    for (auto i = first; i < last; ++i) {
      if (last - i > lookahead)
        __builtin_prefetch(addresses + indices[i + lookahead], 0, 1);
      *out++ = {addresses[indices[i]] + offsets[i], request.token_bytes, 0U};
    }
  }
}

// Hold just this control queue while its WR storage and header are in flight.
bool PublishHeadAndBuildTail(const UmdkTransfer (&head)[3], std::uint32_t lane,
                             const AicpuBatchGetInput& request, std::uint32_t first,
                             std::uint32_t last, peer::GatherSpan* tail,
                             UmdkControlStatus& error, BatchGetWorkerProfile* profile) {
  std::unique_lock<std::mutex> lock(active_session->control_mutexes[lane], std::defer_lock);
  Queue* control;
  std::uint64_t start;
  {
    BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
    lock.lock();
    control = FindQueue(kUmdkControlQueueIndex + lane);
    if (!control || control->poisoned || !control->target || !control->route || !control->route->remote.handle())
      return Fail(error, "control queue is unavailable");
    start = NowNs();
    if (!PostOrderedWrites(*control, head, 3U, error)) return false;
  }
  {
    // The 256-row head ends on a cache line. Tail stores/flushes are disjoint
    // from the head descriptor and the scratch words used by the NIC.
    BatchGetProfileScope timer(profile ? &profile->request_ns : nullptr);
    BuildSourceSpans(request, first, last, tail);
    BatchGetProfileScope flush(profile ? &profile->descriptor_cache_ns : nullptr);
    Flush(tail, std::uint64_t{last - first} * sizeof(*tail));
  }
  BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
  return WaitCompletion(*control, control->requests.wrs.back().user_ctx, start, 1000, error);
}

// Publish the header and head, then let the CPU gather while this lane resolves the tail.
bool StartGather(const AicpuBatchGetInput& request, GatherReadArgs& args, GatherReadStatus& status,
                 std::uint32_t first, std::uint32_t last, BatchGetWorkerProfile* profile,
                 GatherChunkConfig chunks, GroupedRequest& grouped) {
  const auto count = last - first;
  const auto sgl = std::min(args.sgl_size, kGatherMaxSglSize);
  if (args.pipeline_chunk_bytes && !chunks.enabled()) {
    if (args.pull_groups == kGatherChunkedPullEnabled) {
      grouped.chunked = true;
      grouped.plan = GatherPullPlan::MakeChunked(count, request.token_bytes, sgl, args.sq_batch_wqes);
    } else if (args.pull_groups == kGatherPullEnabled) {
      grouped.plan = GatherPullPlan::Make(count, request.token_bytes, sgl, args.sq_batch_wqes,
                                         CurrentState().data_jettys);
    }
  }
  const auto head = grouped.enabled() ? grouped.plan.head_rows :
                    args.pipeline_chunk_bytes ? std::min(count, peer::kGatherHeadRows) : count;
  const bool early = head < count;
  const bool extended = grouped.enabled() || early || chunks.enabled();
  const auto head_bytes = peer::GatherDescriptorBytes(head, extended);
  auto* header = reinterpret_cast<peer::GatherDescriptorHeader*>(args.source_descriptor);
  auto* spans = reinterpret_cast<peer::GatherSpan*>(args.source_descriptor + peer::GatherDescriptorBytes(0, extended));
  {
    BatchGetProfileScope timer(profile ? &profile->request_ns : nullptr);
    const auto bytes = std::uint64_t{count} * request.token_bytes;
    if (!request.address_table || (request.indices && !request.offsets) || !request.output_indices ||
        !args.output || !request.token_bytes || count > GatherLaneCapacity(args.worker_count, args.window_capacity) || bytes > UINT32_MAX) {
      status.result = kInvalidGatherRequest;
      return false;
    }
    args.source_span_count = count;
    args.bytes = static_cast<std::uint32_t>(bytes);
    if (!count) {
      status.ack_seen = args.sequence;
      return false;
    }
    if (!chunks.valid()) {
      status.result = kInvalidGatherRequest;
      return false;
    }
    if (chunks.enabled()) {
      args.pipeline_chunk_bytes = chunks.Tail(bytes);
      if (profile) {
        profile->first_chunk_bytes = chunks.First(bytes);
        profile->tail_chunk_bytes = args.pipeline_chunk_bytes;
      }
    }
    const auto magic = grouped.chunked ? kGatherChunkedPullMagic :
                       grouped.enabled() ? kGatherPullPlanMagic : chunks.enabled() ? peer::kGatherSizedChunksMagic :
                       early ? peer::kGatherHead256Magic : peer::kGatherDescriptorMagic;
    *header = {magic, args.sequence,
               args.remote_staging, count, args.bytes, args.pipeline_chunk_bytes, request.token_bytes};
    if (extended) {
      std::memset(header + 1, 0, peer::GatherDescriptorBytes(0, true) - sizeof(*header));
      if (grouped.enabled()) std::memcpy(header + 1, &grouped.plan, sizeof(grouped.plan));
      else if (chunks.enabled()) std::memcpy(header + 1, &chunks, sizeof(chunks));
    }
    BuildSourceSpans(request, first, first + head, spans);
    BatchGetProfileScope flush(profile ? &profile->descriptor_cache_ns : nullptr);
    Flush(header, head_bytes);
  }
  status.phase = kNotifyCpu;
  auto* scratch = reinterpret_cast<std::uint64_t*>(args.scratch);
  scratch[0] = args.sequence;
  scratch[2] = args.remote_descriptor;
  // AICPU just wrote sequence/descriptor pointer; URMA WRITE DMA reads this line.
  // AICPU does not write it again until the next request, so polls need no pre-READ flush.
  if (early || grouped.enabled()) scratch[3] = grouped.chunked ? head : 0;
  Flush(scratch, early || grouped.enabled() ? 32U : 24U);
  // Strong placement on each WR orders descriptor, command/reset and the final
  // notification at the peer. Only the chain tail requests a completion.
  const UmdkTransfer head_publish[] = {
      {args.remote_descriptor, args.source_descriptor, head_bytes},
      {args.remote_command, args.scratch + 16U, early || grouped.enabled() ? 16U : 8U},
      {args.remote_notify, args.scratch, 8U}};
  if (grouped.enabled()) {
    std::copy(std::begin(head_publish), std::end(head_publish), std::begin(grouped.head));
    return true;
  }
  if (early) {
    UmdkControlStatus error{};
    if (!PublishHeadAndBuildTail(head_publish, args.worker_index, request, first + head, last,
                                 spans + head, error, profile)) {
      std::fprintf(stderr, "UMDK_CONTROL_ERROR worker=%u write=1 stage=%s result=%d errno=%d cq_status=%d\n",
                   args.worker_index, error.stage, error.result, error.error_number, error.cq_status);
      status.result = kGatherTransportFailed;
      return false;
    }
    const auto tail_bytes = std::uint64_t{count - head} * sizeof(peer::GatherSpan);
    const UmdkTransfer tail_publish[] = {
        {args.remote_descriptor + head_bytes, args.source_descriptor + head_bytes, tail_bytes},
        {args.remote_notify + offsetof(peer::GatherControl, suffix_sequence), args.scratch, 8U}};
    BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
    status.result = TransferControlSync(true, tail_publish, 2U, args.worker_index, URMA_STRONG_ORDER);
  } else {
    BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
    status.result = TransferControlSync(true, head_publish, 3U, args.worker_index, URMA_STRONG_ORDER);
  }
  if (!status.result && !args.pipeline_chunk_bytes) {
    status.phase = kWaitCpuAck;
    BatchGetProfileScope timer(profile ? &profile->ack_ns : nullptr);
    WaitAck(args, status);
  }
  return !status.result;
}
}  // namespace

std::uint32_t ExecuteGatherRead(UmdkBatchGetArgs* launch) {
  if (!launch || !launch->request.miss_count || !launch->gather.core_status || !launch->gather.scratch ||
      !launch->gather.execution_control ||
      !launch->gather.worker_count || !ValidGatherReadWaveConfig(launch->gather.read_wave_wrs) ||
      !ValidGatherWindow(launch->request.token_bytes, launch->gather.worker_count,
                         launch->gather.window_capacity))
    return 1U;
  auto args = launch->gather;
  auto& control = *reinterpret_cast<GatherExecutionControl*>(args.execution_control);
  if (__atomic_load_n(&control.failed, __ATOMIC_ACQUIRE)) return 1U;
  BatchGetWorkerProfile measured{};
  auto* profile = args.profile ? &measured : nullptr;
  if (profile) profile->enter_ns = BatchGetProfileNowNs();
  auto session = detail::FindSession(args.context.session);
  if (!session) return 1U;
  // Keep one registry reference and read lock through all control messages,
  // data READs and ACK polls. Teardown/registration waits for this invocation.
  std::shared_lock<std::shared_mutex> lock(session->mutex);
  SessionScope scope(*session);
  const auto ticket = ClaimBatchGetTicket(&control.claimed, args.worker_count);
  args.sequence = ticket.sequence;
  const auto worker = ticket.worker;

  args.worker_index = worker;
  args.core_status += worker * sizeof(GatherReadStatus);
  args.scratch += worker * 64U;
  args.source_descriptor += worker * GatherLaneDescriptorBytes(args.worker_count, args.window_capacity);
  args.remote_staging += worker * GatherLaneStagingBytes(launch->request.token_bytes,
                                                       args.worker_count, args.window_capacity);
  args.remote_descriptor += worker * GatherLaneDescriptorBytes(args.worker_count, args.window_capacity);
  args.remote_notify += worker * sizeof(kvaf::peer::GatherControl);
  args.remote_ack += worker * sizeof(kvaf::peer::GatherControl);
  args.remote_command += worker * sizeof(kvaf::peer::GatherControl);
  auto& status = *reinterpret_cast<GatherReadStatus*>(args.core_status);
  status = {};
  status.magic = kGatherStatusMagic;
  status.sequence = args.sequence;
  status.failed_request = UINT64_MAX;
  if (session->state.queue_profile == UmdkQueueProfile::Put ||
      args.worker_count != session->state.worker_count ||
      args.context.owner_pid != static_cast<std::uint64_t>(getpid()) ||
      args.context.session != session->state.session || args.sgl_size == 0 || args.sq_batch_wqes == 0) {
    status.result = kGatherTransportFailed;
    __atomic_store_n(&control.failed, std::uint64_t{1}, __ATOMIC_RELEASE);
    return 1U;
  }
  const auto active = *reinterpret_cast<const volatile std::int32_t*>(launch->request.miss_count);
  // Retain GatherRead's wide wire sequence. Leave enough ticket headroom for
  // every instance of the failed launch, so the counter cannot wrap first.
  if (args.sequence > (peer::kGatherErrorBit - 1U) / args.worker_count) {
    status.result = kGatherSequenceExhausted;
    __atomic_store_n(&control.failed, std::uint64_t{1}, __ATOMIC_RELEASE);
    return 1U;
  }
  if (active < 0 || static_cast<std::uint32_t>(active) > launch->request.request_capacity) {
    status.result = kInvalidGatherRequest;
    __atomic_store_n(&control.failed, std::uint64_t{1}, __ATOMIC_RELEASE);
    return 1U;
  }
  const auto count = static_cast<std::uint32_t>(active);
  std::uint32_t updates = 0, groups = 0;
  // Resolve from this launch's total active count, independently of tensor
  // capacity, worker slices and memory windows. The captured args retain auto.
  args.read_wave_wrs = ResolveGatherReadWaveWrs(args.read_wave_wrs, count);
  // Each lane advances independently through the same global windows. Only
  // window-local descriptors/staging are reused; all input/output indices stay global.
  for (std::uint64_t base = 0; base < count; base += args.window_capacity) {
    const auto window_count = std::min<std::uint64_t>(args.window_capacity, count - base);
    const auto first = static_cast<std::uint32_t>(base + window_count * worker / args.worker_count);
    const auto last = static_cast<std::uint32_t>(base + window_count * (worker + 1U) / args.worker_count);
    if (__atomic_load_n(&control.failed, __ATOMIC_ACQUIRE)) {
      status.result = kGatherTransportFailed;
      status.failed_request = first;
      break;
    }
    if (first == last) continue;
    auto& wire_sequence = control.window_sequence[worker];
    const auto previous = __atomic_load_n(&wire_sequence, __ATOMIC_ACQUIRE);
    if (previous >= peer::kGatherErrorBit - 1U) {
      status.result = kGatherSequenceExhausted;
      status.failed_request = first;
      break;
    }
    args.sequence = previous + 1U;
    __atomic_store_n(&wire_sequence, args.sequence, __ATOMIC_RELEASE);
    args.bytes = args.source_span_count = 0U;
    status.ack_seen = status.cqe_word = 0U;
    GroupedRequest grouped{};
    grouped.address_groups_per_post = control.address_groups_per_post;
    const auto previous_bytes = profile ? profile->bytes : 0U;
    const auto previous_spans = profile ? profile->spans : 0U;
    if (StartGather(launch->request, args, status, first, last, profile, control.chunks, grouped)) {
      const auto head_request = profile ? profile->request_ns : 0U;
      const auto head_publish = profile ? profile->notify_ns : 0U;
      const auto previous_ack = profile ? profile->ack_ns : 0U;
      {
        BatchGetProfileScope timer(profile ? &profile->read_ns : nullptr);
        status.phase = kReadStaging;
        status.result = ReadStaging(*session, grouped, launch->request, args, status, first, last, profile);
      }
      if (profile) profile->read_ns -= (profile->ack_ns - previous_ack) +
          (profile->notify_ns - head_publish) + (profile->request_ns - head_request);
    }
    if (profile) {
      profile->bytes = previous_bytes + args.bytes;
      profile->spans = previous_spans + args.source_span_count;
      if (grouped.enabled() && !status.result) {
        updates += grouped.updates;
        groups += grouped.plan.GroupCount();
      }
    }
    if (status.result) { status.failed_request = first; break; }
    status.completed_requests += last - first;
  }
  if (profile) {
    if (groups) StampGatherPullPlan(status, *profile, updates, groups);
    else status.ready_profile_stamp = profile->enter_ns ^ kGatherReadyProgressCookie;
    profile->exit_ns = BatchGetProfileNowNs();
    auto* destination = reinterpret_cast<BatchGetWorkerProfile*>(args.profile);
    destination[worker] = *profile;
  }
  if (status.result) __atomic_store_n(&control.failed, std::uint64_t{1}, __ATOMIC_RELEASE);
  return status.result ? 1U : 0U;
}

}  // namespace kvaf::aicpu_urma
