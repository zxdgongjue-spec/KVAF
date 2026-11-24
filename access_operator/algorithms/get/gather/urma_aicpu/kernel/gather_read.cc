#include "gather_read.h"

#include <unistd.h>

#include <atomic>

#include "cpu_gather.h"
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
        staging(CurrentState().remote.handle()->seg.ubva.va +
                (args.remote_staging - CurrentState().remote.canonical())),
        sgl_size(std::min(args.sgl_size, kvaf::kGatherMaxSglSize)),
        token_bytes(request.token_bytes),
        rows(reinterpret_cast<const std::int32_t*>(request.output_indices)),
        local_segment(LocalSegment(args.output, args.output_extent)),
        remote_segment(CurrentState().remote.handle()) {
    prototype.opcode = URMA_OPC_READ;
    prototype.flag.bs.comp_order = 1;
    prototype.flag.bs.place_order = URMA_RELAX_ORDER;
    prototype.tjetty = worker.target;
    prototype.rw.src.num_sge = 1U;
  }
};

void BuildReadBatch(Queue& worker, const GatherBuildLayout& layout, std::uint32_t& span_index,
                    std::uint32_t last, std::uint32_t& span_offset, std::uint64_t& source_offset,
                    std::uint32_t batch_wrs, std::uint32_t wave_wrs,
                    std::uint32_t& local_index) {
  auto& queue = worker.requests;
  auto index = span_index;
  auto offset = span_offset;
  auto source_position = source_offset;
  auto* local = queue.local_sges.data();
  auto* remote = queue.remote_sges.data();
  auto* requests = queue.wrs.data();
  auto sequence = worker.sequence;
  auto wr_count = queue.wr_count;
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
  std::uint64_t registered_staging;
  std::uint64_t ready_bytes = 0;
  BatchGetWorkerProfile* profile = nullptr;
};

bool WaitBatchPrefix(GatherPrefix& prefix, const urma_jfs_wr_t& tail,
                     UmdkControlStatus& submitted) {
  const auto& args = prefix.args;
  const auto& source = *reinterpret_cast<const Sge*>(tail.rw.src.sge);
  const auto required = source.addr - prefix.registered_staging + source.len;
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
    // READ DMA replaced the cached progress word; CQ completion precedes this load.
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
  GatherPrefix prefix{args, status, args.worker_index, layout.staging, 0U, profile};
  if (profile) profile->read_setup_ns += BatchGetProfileNowNs() - setup_start;
  // A WR ends at the SGL limit, the 64 KiB limit, or the end of this request.
  const auto max_wrs = (static_cast<std::uint64_t>(args.bytes) + 65535U) / 65536U +
                       (args.source_span_count + layout.sgl_size - 1U) / layout.sgl_size;
  const auto wave_wrs = static_cast<std::uint32_t>(std::min<std::uint64_t>(args.read_wave_wrs, max_wrs));
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
                          args.sq_batch_wqes, wave_wrs, local_index);
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
    } while (first < last && queue.wr_count < wave_wrs);
    // Retain every WR/SGL slot until the final ordered CQE for this wave.
    {
      BatchGetProfileScope timer(profile ? &profile->cq_wait_ns : nullptr);
      if (!WaitCompletion(worker, queue.wrs[queue.wr_count - 1U].user_ctx, start, 1000, error))
        return false;
    }
  }
  return true;
}

std::uint32_t ReadStaging(ExecutionSession& session, const AicpuBatchGetInput& request,
                         const GatherReadArgs& args, GatherReadStatus& status,
                         std::uint32_t first, std::uint32_t last,
                         BatchGetWorkerProfile* profile) {
  const auto worker_index = args.worker_index;
  std::lock_guard<std::mutex> worker_lock(session.worker_mutexes[worker_index]);
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
bool BuildCurrentRequest(const UmdkBatchGetArgs& launch, GatherReadArgs& args,
                         std::uint32_t first, std::uint32_t last, BatchGetWorkerProfile* profile) {
  const auto& request = launch.request;
  if (!request.address_table || (request.indices && !request.offsets) || !request.miss_count ||
      !request.output_indices || !args.output || !request.token_bytes ||
      last < first || last - first > GatherLaneCapacity(args.worker_count, args.window_capacity))
    return false;
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(request.address_table);
  const auto* indices = reinterpret_cast<const std::int32_t*>(request.indices);
  const auto* offsets = reinterpret_cast<const std::int32_t*>(request.offsets);
  // Input tensors are ready before this CANN kernel starts, as in the control
  // and CPU-push kernels. AICPU only reads them; no per-request cache sweep.
  auto* descriptor = reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(args.source_descriptor);
  auto* sources = reinterpret_cast<kvaf::peer::GatherSpan*>(descriptor + 1);
  if (!request.indices) {
    for (auto i = first; i < last; ++i)
      sources[args.source_span_count++] = {addresses[i], request.token_bytes, 0U};
  } else {
    for (auto i = first; i < last; ++i)
      sources[args.source_span_count++] = {addresses[indices[i]] + offsets[i], request.token_bytes, 0U};
  }
  const auto bytes = static_cast<std::uint64_t>(args.source_span_count) * request.token_bytes;
  if (bytes > UINT32_MAX) return false;
  args.bytes = static_cast<std::uint32_t>(bytes);
  *descriptor = {kvaf::peer::kGatherDescriptorMagic, args.sequence, args.remote_staging,
                 args.source_span_count, args.bytes, args.pipeline_chunk_bytes, request.token_bytes};
  if (args.bytes) {
    BatchGetProfileScope timer(profile ? &profile->descriptor_cache_ns : nullptr);
    // AICPU wrote this descriptor; the following in-kernel URMA WRITE DMA reads it.
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
    status.result = TransferControlSync(true, &descriptor, 1U, args.worker_index);
  }
  if (status.result) return false;
  status.phase = kNotifyCpu;
  auto* scratch = reinterpret_cast<std::uint64_t*>(args.scratch);
  scratch[0] = args.sequence;
  scratch[2] = args.remote_descriptor;
  // AICPU just wrote sequence/descriptor pointer; URMA WRITE DMA reads this line.
  // AICPU does not write it again until the next request, so polls need no pre-READ flush.
  Flush(scratch, 24U);
  const UmdkTransfer notify[] = {{args.remote_command, args.scratch + 16U, 8U},
                                 {args.remote_notify, args.scratch, 8U}};
  {
    BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
    status.result = TransferControlSync(true, notify, 2U, args.worker_index);
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
      !launch->gather.worker_count || !ValidGatherReadWaveWrs(launch->gather.read_wave_wrs) ||
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
  if (session->state.queue_profile != UmdkQueueProfile::Gather ||
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
    bool built;
    {
      BatchGetProfileScope timer(profile ? &profile->request_ns : nullptr);
      built = BuildCurrentRequest(*launch, args, first, last, profile);
    }
    if (profile) { profile->bytes += args.bytes; profile->spans += args.source_span_count; }
    if (!built)
      status.result = kInvalidGatherRequest;
    else if (StartGather(args, status, profile)) {
      {
        BatchGetProfileScope timer(profile ? &profile->read_ns : nullptr);
        status.phase = kReadStaging;
        status.result = ReadStaging(*session, launch->request, args, status, first, last, profile);
      }
      if (!status.result && args.pipeline_chunk_bytes) {
        BatchGetProfileScope timer(profile ? &profile->ack_ns : nullptr);
        WaitAck(args, status);
      }
    }
    if (status.result) { status.failed_request = first; break; }
    // Both READ completions and the CPU's terminal ACK have been observed.
    status.completed_requests += last - first;
  }
  // Status/profile are consumed by CANN D2H after kernel completion on this stream.
  if (profile) {
    profile->exit_ns = BatchGetProfileNowNs();
    auto* destination = reinterpret_cast<BatchGetWorkerProfile*>(args.profile) + worker;
    *destination = *profile;
  }
  if (status.result) __atomic_store_n(&control.failed, std::uint64_t{1}, __ATOMIC_RELEASE);
  return status.result ? 1U : 0U;
}

}  // namespace kvaf::aicpu_urma
