#include "gather_read.h"

#include <unistd.h>

#include <atomic>

#include "cpu_gather.h"
#include "transport_internal.h"
#include "aicpu_urma/cache.h"

namespace kvaf::aicpu_urma {
namespace {
using namespace detail;

unsigned ClaimWorker(unsigned count) {
  if (count == 1U) return 0U;
  // Runtime drains the previous launch before reuse. Exactly N claims assign
  // every slot once, even if a new session changes N or starts at another slot.
  static std::atomic<unsigned> next{0U};
  auto observed = next.load(std::memory_order_relaxed);
  for (;;) {
    const auto worker = observed % count;
    const auto successor = (worker + 1U) % count;
    if (next.compare_exchange_weak(observed, successor, std::memory_order_relaxed))
      return worker;
  }
}

template <typename T>
void EnsureSlots(std::vector<T>& slots, std::size_t count) {
  if (count > slots.size()) slots.resize(count);
}

constexpr std::uint32_t kGatherWaveWrs = 4096U;

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
  const std::uint8_t* mask;
  const std::int32_t* rows;
  urma_target_seg_t* local_segment;
  urma_target_seg_t* remote_segment;
  urma_jfs_wr_t prototype{};

  GatherBuildLayout(Queue& worker, const GatherReadArgs& args,
                    const AicpuBatchGetInput& request)
      : output(args.output),
        staging(args.remote_staging),
        sgl_size(std::min(args.sgl_size, kvaf::kGatherMaxSglSize)),
        token_bytes(request.token_bytes),
        mask(reinterpret_cast<const std::uint8_t*>(request.mask)),
        rows(reinterpret_cast<const std::int32_t*>(request.output_indices)),
        local_segment(LocalSegment(args.output, args.output_extent)),
        remote_segment(state.remote) {
    prototype.opcode = URMA_OPC_READ;
    prototype.flag.bs.comp_order = 1;
    prototype.flag.bs.place_order = URMA_RELAX_ORDER;
    prototype.tjetty = worker.target;
    prototype.rw.src.num_sge = 1U;
  }
};

void BuildReadBatch(Queue& worker, const GatherBuildLayout& layout, std::uint32_t& span_index,
                    std::uint32_t last, std::uint32_t& span_offset, std::uint64_t& source_offset,
                    std::uint32_t batch_wrs, std::uint32_t& local_index) {
  auto& queue = worker.requests;
  auto index = span_index;
  auto offset = span_offset;
  auto source_position = source_offset;
  while (index < last && layout.mask && !layout.mask[index]) ++index;
  auto* local = queue.local_sges.data();
  auto* remote = queue.remote_sges.data();
  auto* requests = queue.wrs.data();
  auto sequence = worker.sequence;
  auto wr_count = queue.wr_count;
  const auto wave_wrs = static_cast<std::uint32_t>(queue.wrs.size());
  const auto wr_limit = wr_count + std::min(batch_wrs, wave_wrs - wr_count);
  while (index < last && wr_count < wr_limit) {
    const auto wr_index = wr_count++;
    const auto local_begin = local_index;
    std::uint32_t bytes = 0U, local_count = 0U;
    const auto source = layout.staging + source_position;
    while (index < last && local_count < layout.sgl_size && bytes < 65536U) {
      const auto length = std::min(layout.token_bytes - offset, 65536U - bytes);
      const auto target = layout.output +
          static_cast<std::uint64_t>(layout.rows[index]) * layout.token_bytes + offset;
      local[local_index++] = {target, length, layout.local_segment, nullptr};
      ++local_count;
      bytes += length;
      source_position += length;
      offset += length;
      if (offset == layout.token_bytes) {
        ++index;
        offset = 0U;
        while (index < last && layout.mask && !layout.mask[index]) ++index;
      }
    }
    remote[wr_index] = {source, bytes, layout.remote_segment, nullptr};
    auto wr = layout.prototype;
    wr.user_ctx = ++sequence;
    wr.rw.src.sge = reinterpret_cast<urma_sge_t*>(&remote[wr_index]);
    wr.rw.dst.sge = reinterpret_cast<urma_sge_t*>(local + local_begin);
    wr.rw.dst.num_sge = local_count;
    wr.next = requests + wr_index + 1U;
    requests[wr_index] = wr;
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
  span_offset = offset;
  source_offset = source_position;
}

struct GatherPrefix {
  const GatherReadArgs& args;
  GatherReadStatus& gather;
  std::uint32_t worker_index;
  std::uint64_t ready_bytes = 0;
  BatchGetWorkerProfile* profile = nullptr;
};

bool WaitBatchPrefix(GatherPrefix& prefix, const urma_jfs_wr_t& tail,
                     UmdkControlStatus& submitted) {
  const auto& args = prefix.args;
  const auto& source = *reinterpret_cast<const Sge*>(tail.rw.src.sge);
  const auto required = source.addr + source.len - args.remote_staging;
  if (required <= prefix.ready_bytes) return true;
  auto& gather = prefix.gather;
  gather.phase = kWaitReadyPrefix;
  const auto scratch = args.scratch;  // Already offset to this worker's mailbox.
  const UmdkTransfer transfer{
      args.remote_descriptor + offsetof(kvaf::peer::GatherDescriptorHeader, ready_bytes), scratch,
      8};
  bool success = false;
  for (std::uint32_t poll = 0; poll < 10000; ++poll) {
    UmdkControlStatus query{};
    if (prefix.profile) ++prefix.profile->ready_queries;
    // The dedicated control Jetty can complete this query while data READs
    // remain outstanding; controls use their own WR storage.
    if (!TransferControl(false, &transfer, 1, query, prefix.worker_index)) {
      errno = query.error_number;
      Fail(submitted, query.stage, query.result);
      gather.result = kGatherTransportFailed;
      break;
    }
    FlushUmdkCache(reinterpret_cast<const void*>(scratch), 8);
    prefix.ready_bytes = *reinterpret_cast<const volatile std::uint64_t*>(scratch);
    if ((prefix.ready_bytes & kvaf::peer::kGatherErrorBit) != 0 ||
        prefix.ready_bytes > args.bytes) {
      gather.result = kCpuGatherFailed;
      Fail(submitted, "CPU ready prefix error");
      break;
    }
    if (prefix.ready_bytes >= required) {
      success = true;
      break;
    }
  }
  if (!success) {
    gather.ack_seen = prefix.ready_bytes;
    if (gather.result == 0) {
      gather.result = kGatherTimeout;
      errno = ETIMEDOUT;
      Fail(submitted, "waiting for CPU ready prefix timed out", ETIMEDOUT);
    }
  } else
    gather.phase = kReadStaging;
  return success;
}

bool ReadBatches(Queue& worker, const AicpuBatchGetInput& request,
                 const GatherReadArgs& args, GatherReadStatus& status,
                 std::uint32_t first, std::uint32_t last, UmdkControlStatus& error,
                 BatchGetWorkerProfile* profile) {
  if (first == last) return true;
  const auto setup_start = profile ? BatchGetProfileNowNs() : 0U;
  const GatherBuildLayout layout(worker, args, request);
  if (!layout.local_segment) return Fail(error, "READ output is not registered");
  GatherPrefix prefix{args, status, args.worker_index, 0U, profile};
  if (profile) profile->read_setup_ns += BatchGetProfileNowNs() - setup_start;
  // A WR ends at the SGL limit, the 64 KiB limit, or the end of this request.
  const auto max_wrs = (static_cast<std::uint64_t>(args.bytes) + 65535U) / 65536U +
                       (args.source_span_count + layout.sgl_size - 1U) / layout.sgl_size;
  const auto wave_wrs = static_cast<std::uint32_t>(std::min<std::uint64_t>(kGatherWaveWrs, max_wrs));
  std::uint32_t span_offset = 0;
  std::uint64_t remote_offset = 0;
  while (first < last) {
    auto& queue = worker.requests;
    {
      BatchGetProfileScope timer(profile ? &profile->read_setup_ns : nullptr);
      BeginGatherWave(queue, layout.sgl_size, wave_wrs);
    }
    std::uint32_t local_index = 0;
    const auto start = NowNs();
    do {
      const auto batch_first = queue.wr_count;
      {
        BatchGetProfileScope timer(profile ? &profile->wr_build_ns : nullptr);
        BuildReadBatch(worker, layout, first, last, span_offset, remote_offset,
                          args.sq_batch_wqes, local_index);
      }
      auto* tail = &queue.wrs[queue.wr_count - 1U];
      bool ready = true;
      if (args.pipeline_chunk_bytes) {
        BatchGetProfileScope timer(profile ? &profile->ready_wait_ns : nullptr);
        ready = WaitBatchPrefix(prefix, *tail, error);
      }
      bool posted = false;
      if (ready) {
        BatchGetProfileScope timer(profile ? &profile->post_ns : nullptr);
        posted = PostBatch(worker, &queue.wrs[batch_first], tail, error);
      }
      if (!ready || !posted) {
        worker.poisoned = true;
        return false;
      }
      if (profile) {
        profile->read_wqes += queue.wr_count - batch_first;
        ++profile->post_calls;
      }
    } while (first < last && queue.wr_count < queue.wrs.size());
    // Retain every WR/SGL slot until the final ordered CQE for this wave.
    {
      BatchGetProfileScope timer(profile ? &profile->cq_wait_ns : nullptr);
      if (!WaitCompletion(worker, queue.wrs[queue.wr_count - 1U].user_ctx, start, 1000, error))
        return false;
    }
  }
  return true;
}

std::uint32_t ReadStaging(const AicpuBatchGetInput& request,
                         const GatherReadArgs& args, GatherReadStatus& status,
                         std::uint32_t first, std::uint32_t last,
                         BatchGetWorkerProfile* profile) {
  std::shared_lock<std::shared_mutex> lock(state_mutex);
  const auto worker_index = args.worker_index;
  if (worker_index >= state.worker_count || args.worker_count != state.worker_count ||
      args.context.owner_pid != static_cast<std::uint64_t>(getpid()) ||
      args.context.session != state.session || args.sgl_size == 0 || args.sq_batch_wqes == 0)
    return kGatherTransportFailed;
  std::lock_guard<std::mutex> worker_lock(worker_mutexes[worker_index]);
  auto& worker = DataQueue(worker_index);
  if (worker.poisoned || worker.target == nullptr) return kGatherTransportFailed;
  UmdkControlStatus error{};
  if (!ReadBatches(worker, request, args, status, first, last, error, profile)) {
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
    status.result = UmdkTransferSync(&args.context, false, &transfer, 1U, args.worker_index);
    if (status.result) return;
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
// come directly from the caller's mask/output_indices; no target list is materialized.
bool BuildCurrentRequest(const UmdkBatchGetArgs& launch, GatherReadArgs& args,
                         std::uint32_t first, std::uint32_t last, BatchGetWorkerProfile* profile) {
  const auto& request = launch.request;
  if (!request.address_table || !request.indices || !request.offsets || (!request.mask && !request.miss_count) ||
      !request.output_indices || !args.output || !request.token_bytes ||
      request.request_capacity > kGatherMaxSpans)
    return false;
  {
    BatchGetProfileScope timer(profile ? &profile->input_cache_ns : nullptr);
    Flush(reinterpret_cast<void*>(request.address_table), request.table_capacity * 8ULL);
    Flush(reinterpret_cast<void*>(request.indices + first * 4ULL), (last - first) * 4ULL);
    Flush(reinterpret_cast<void*>(request.offsets + first * 4ULL), (last - first) * 4ULL);
    if (request.mask) Flush(reinterpret_cast<void*>(request.mask + first), last - first);
    Flush(reinterpret_cast<void*>(request.output_indices + first * 4ULL), (last - first) * 4ULL);
  }
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(request.address_table);
  const auto* indices = reinterpret_cast<const std::int32_t*>(request.indices);
  const auto* offsets = reinterpret_cast<const std::int32_t*>(request.offsets);
  const auto* mask = reinterpret_cast<const std::uint8_t*>(request.mask);
  auto* descriptor = reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(args.source_descriptor);
  auto* sources = reinterpret_cast<kvaf::peer::GatherSpan*>(descriptor + 1);
  for (auto i = first; i < last; ++i) {
    if (mask && !mask[i]) continue;
    sources[args.source_span_count++] = {addresses[indices[i]] + offsets[i], request.token_bytes, 0U};
  }
  const auto bytes = static_cast<std::uint64_t>(args.source_span_count) * request.token_bytes;
  if (bytes > UINT32_MAX) return false;
  args.bytes = static_cast<std::uint32_t>(bytes);
  *descriptor = {kvaf::peer::kGatherDescriptorMagic, args.sequence, args.remote_staging,
                 args.source_span_count, args.bytes, args.pipeline_chunk_bytes, request.token_bytes};
  {
    BatchGetProfileScope timer(profile ? &profile->descriptor_cache_ns : nullptr);
    Flush(descriptor, kvaf::peer::GatherDescriptorBytes(args.source_span_count));
  }
  return true;
}

bool StartGather(GatherReadArgs& args, GatherReadStatus& status, BatchGetWorkerProfile* profile) {
  const UmdkTransfer descriptor{args.remote_descriptor, args.source_descriptor,
                                kvaf::peer::GatherDescriptorBytes(args.source_span_count)};
  status.phase = kWriteSourceList;
  {
    BatchGetProfileScope timer(profile ? &profile->source_write_ns : nullptr);
    status.result = UmdkTransferSync(&args.context, true, &descriptor, 1U, args.worker_index);
  }
  if (status.result) return false;
  status.phase = kNotifyCpu;
  auto* scratch = reinterpret_cast<std::uint64_t*>(args.scratch);
  scratch[0] = args.sequence;
  scratch[2] = args.remote_descriptor;
  // Clean the entire mailbox line once before DMA. Until the next request,
  // AICPU only reads it: polling needs cache maintenance after CQ, not before READ.
  Flush(scratch, 24U);
  const UmdkTransfer notify[] = {{args.remote_command, args.scratch + 16U, 8U},
                                 {args.remote_notify, args.scratch, 8U}};
  {
    BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
    status.result = UmdkTransferSync(&args.context, true, notify, 2U, args.worker_index);
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
  if (!launch || !launch->gather.core_status || !launch->gather.scratch ||
      !launch->gather.worker_count ||
      launch->gather.worker_count > kGatherMaxWorkers)
    return 1U;
  auto args = launch->gather;
  BatchGetWorkerProfile measured{};
  auto* profile = args.profile ? &measured : nullptr;
  if (profile) profile->enter_ns = BatchGetProfileNowNs();
  const auto worker = ClaimWorker(args.worker_count);
  args.worker_index = worker;
  args.core_status += worker * sizeof(GatherReadStatus);
  args.scratch += worker * 64U;
  args.source_descriptor += worker * GatherLaneDescriptorBytes(args.worker_count);
  args.remote_staging += worker * GatherLaneStagingBytes(launch->request.token_bytes,
                                                       args.worker_count);
  args.remote_descriptor += worker * GatherLaneDescriptorBytes(args.worker_count);
  args.remote_notify += worker * sizeof(kvaf::peer::GatherControl);
  args.remote_ack += worker * sizeof(kvaf::peer::GatherControl);
  args.remote_command += worker * sizeof(kvaf::peer::GatherControl);
  auto& status = *reinterpret_cast<GatherReadStatus*>(args.core_status);
  status = {};
  status.magic = kGatherStatusMagic;
  status.sequence = args.sequence;
  auto count = launch->request.request_capacity;
  if (launch->request.miss_count) {
    Flush(reinterpret_cast<const void*>(launch->request.miss_count), sizeof(std::int32_t));
    const auto active = *reinterpret_cast<const volatile std::int32_t*>(launch->request.miss_count);
    if (active < 0 || static_cast<std::uint32_t>(active) > count) {
      status.result = kInvalidGatherRequest;
      Flush(&status, sizeof(status));
      return 1U;
    }
    count = static_cast<std::uint32_t>(active);
  }
  const auto first = static_cast<std::uint64_t>(count) * worker / args.worker_count;
  const auto last = static_cast<std::uint64_t>(count) * (worker + 1U) / args.worker_count;
  bool built;
  {
    BatchGetProfileScope timer(profile ? &profile->request_ns : nullptr);
    built = BuildCurrentRequest(*launch, args, first, last, profile);
  }
  if (!built)
    status.result = kInvalidGatherRequest;
  else if (!args.bytes)
    status.ack_seen = args.sequence;
  else if (StartGather(args, status, profile)) {
    {
      BatchGetProfileScope timer(profile ? &profile->read_ns : nullptr);
      status.phase = kReadStaging;
      status.result = ReadStaging(launch->request, args, status, first, last, profile);
    }
    if (!status.result && args.pipeline_chunk_bytes) {
      BatchGetProfileScope timer(profile ? &profile->ack_ns : nullptr);
      WaitAck(args, status);
    }
  }
  Flush(&status, sizeof(status));
  if (profile) {
    profile->bytes = args.bytes;
    profile->spans = args.source_span_count;
    profile->exit_ns = BatchGetProfileNowNs();
    auto* destination = reinterpret_cast<BatchGetWorkerProfile*>(args.profile) + worker;
    *destination = *profile;
    Flush(destination, sizeof(*destination));
  }
  return status.result ? 1U : 0U;
}

}  // namespace kvaf::aicpu_urma
