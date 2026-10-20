#include <time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>

#include "cpu_gather.h"
#include "gather_cache.h"
#include "gather_read.h"
#include "umdk_gather.h"
#if defined(__linux__)
#include <sched.h>
#endif
namespace {
using namespace kvaf::aicpu_urma;
struct Memory {
  std::uint64_t local_hbm_address, local_hbm_size, remote_host_ddr_address, remote_host_ddr_size;
  std::uint32_t local_token_id{0U};
};
struct Transfer {
  std::uint64_t remote_address, local_address;
  std::uint32_t length, token;
};
bool InRange(std::uint64_t base, std::uint64_t size, std::uint64_t address, std::uint64_t bytes) {
  return bytes && address >= base && bytes <= size && address - base <= size - bytes;
}
std::uint64_t NowNs() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

void Barrier() { std::atomic_signal_fence(std::memory_order_seq_cst); }

std::uint64_t BeginReadCpuProfile(kvaf::aicpu_urma::GatherReadStatus& status, bool detailed) {
  status.thread_cpu_ns = status.thread_cpu_available = 0U;
  status.start_cpu = status.end_cpu = -1;
  if (!detailed) return 0U;
#if defined(__linux__)
  status.start_cpu = sched_getcpu();
#endif
  timespec value{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) return 0U;
  status.thread_cpu_available = 1U;
  return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}

void EndReadCpuProfile(kvaf::aicpu_urma::GatherReadStatus& status, std::uint64_t begin_ns) {
  if (status.thread_cpu_available == 0U) return;
  timespec value{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) {
    status.thread_cpu_available = 0U;
    return;
  }
  const auto end_ns = static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL + value.tv_nsec;
  status.thread_cpu_ns = end_ns >= begin_ns ? end_ns - begin_ns : 0U;
#if defined(__linux__)
  status.end_cpu = sched_getcpu();
#endif
}

void FlushLines(const void* address, std::uint64_t bytes) {
  kvaf::aicpu_urma::FlushGatherCacheLines(address, bytes);
}

void Flush(const void* address, std::uint64_t bytes) {
  FlushLines(address, bytes);
  Barrier();
}

bool GatherEntity(const GatherReadArgs& args, unsigned, Memory& memory, const Memory*& entity) {
  UmdkGatherMemory region{};
  if (KvafAicpuUmdkGatherMemory(&args, &region)) return false;
  memory = {region.local_base, region.local_bytes, region.remote_base, region.remote_bytes, 0U};
  entity = &memory;
  return true;
}
bool MakeGetTransfer(const KvafExecuteBatchGetKernelArgs& args, const Memory& entity,
                     std::uint32_t index, Transfer* transfer) {
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(args.address_table);
  const auto* valid = reinterpret_cast<const std::uint8_t*>(args.valid_table);
  const auto* indices = reinterpret_cast<const std::int32_t*>(args.indices);
  const auto* offsets = reinterpret_cast<const std::int32_t*>(args.offsets);
  const auto* output_indices = reinterpret_cast<const std::int32_t*>(args.output_indices);
  const auto table_index = indices[index];
  const auto offset = offsets[index];
  const auto output_index = output_indices[index];
  const std::uint64_t block_bytes =
      static_cast<std::uint64_t>(args.token_bytes) * args.tokens_per_block;
  if (table_index < 0 || static_cast<std::uint32_t>(table_index) >= args.table_capacity ||
      valid[table_index] == 0U || offset < 0 || block_bytes < args.token_bytes ||
      static_cast<std::uint64_t>(offset) > block_bytes - args.token_bytes || output_index < 0 ||
      static_cast<std::uint32_t>(output_index) >= args.output_rows ||
      !InRange(entity.remote_host_ddr_address, entity.remote_host_ddr_size, addresses[table_index],
               block_bytes))
    return false;
  const std::uint64_t output_offset = static_cast<std::uint64_t>(output_index) * args.token_bytes;
  if (args.output > UINT64_MAX - output_offset ||
      !InRange(entity.local_hbm_address, entity.local_hbm_size, args.output + output_offset,
               args.token_bytes))
    return false;
  *transfer = {addresses[table_index] + static_cast<std::uint32_t>(offset),
               args.output + output_offset, args.token_bytes, 0U};
  return true;
}
void WaitAck(const GatherReadArgs& args, GatherReadStatus& status) {
  const auto scratch = args.output + ((args.output_extent + 63U) & ~std::uint64_t{63U});
  const UmdkTransfer transfer{args.remote_ack, scratch + 8U,
                              sizeof(kvaf::peer::GatherControl) - 8U};
  const auto start = NowNs();
  for (unsigned poll = 0; poll < 10000U; ++poll) {
    std::uint64_t post = 0U, cq = 0U;
    const auto before = args.profile_detail ? NowNs() : 0U;
    Flush(reinterpret_cast<void*>(scratch + 8U), transfer.bytes);
    if (args.profile_detail) status.ack_cache_ns += NowNs() - before;
    status.result = KvafAicpuUmdkGatherTransfer(&args, false, &transfer, 1U, 0U, &post, &cq);
    const auto after = args.profile_detail ? NowNs() : 0U;
    Flush(reinterpret_cast<void*>(scratch + 8U), transfer.bytes);
    if (args.profile_detail) status.ack_cache_ns += NowNs() - after;
    status.ack_post_ns += post;
    status.ack_cq_ns += cq;
    ++status.ack_polls;
    if (status.result) break;
    status.ack_seen = *reinterpret_cast<const volatile std::uint64_t*>(scratch + 8U);
    if (!status.ack_started_ns && status.ack_seen == kvaf::peer::GatherStartedAck(args.sequence))
      status.ack_started_ns = NowNs() - start;
    if (status.ack_seen == args.sequence ||
        status.ack_seen == (args.sequence | kvaf::peer::kGatherErrorBit))
      break;
  }
  status.ack_poll_ns = NowNs() - start;
  if (!status.result && status.ack_seen == args.sequence)
    std::memcpy(&status.cpu, reinterpret_cast<const void*>(scratch + 24U), sizeof(status.cpu));
  else if (!status.result)
    status.result = (status.ack_seen & kvaf::peer::kGatherErrorBit) ? 10U : 9U;
}
void ReadRange(const GatherReadArgs& args, unsigned worker, GatherReadStatus& status) {
  const auto first =
      static_cast<std::uint64_t>(args.output_span_count) * worker / args.worker_count;
  const auto last =
      static_cast<std::uint64_t>(args.output_span_count) * (worker + 1U) / args.worker_count;
  const auto* queues = reinterpret_cast<const GatherQueueState*>(args.queue_state);
  status.scatter_begin_ns = NowNs();
  const auto cpu = BeginReadCpuProfile(status, args.profile_detail != 0);
  status.read_tasks = first != last ? 1U : 0U;
  status.phase = 4U;
  status.result = KvafAicpuUmdkGatherScatter(&args, &status, first, last,
                                             queues[worker].staging_offset, worker);
  EndReadCpuProfile(status, cpu);
  status.scatter_end_ns = NowNs();
}
}  // namespace

extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuUmdkBatchGet(
    kvaf::aicpu_urma::UmdkBatchGetArgs* launch) {
  if (launch == nullptr || launch->request.core_status == 0U) return 1U;
  const auto& args = launch->request;
  auto& status = *reinterpret_cast<KvafBatchGetCoreStatus*>(args.core_status);
  status = {};
  status.magic = KVAF_AIV_BATCH_GET_STATUS_MAGIC;
  status.result = 2U;
  status.failed_request_index = UINT64_MAX;
  if (args.address_table == 0 || args.valid_table == 0 || args.indices == 0 || args.offsets == 0 ||
      args.mask == 0 || args.output_indices == 0 || args.output == 0 || args.token_bytes == 0 ||
      args.tokens_per_block == 0 || args.output_rows == 0) {
    Flush(&status, sizeof(status));
    return 1U;
  }
  kvaf::aicpu_urma::GatherReadArgs session{};
  session.umdk_session = launch->session;
  session.umdk_owner_pid = launch->owner_pid;
  session.output = args.output;
  session.output_extent = static_cast<std::uint64_t>(args.output_rows) * args.token_bytes;
  Memory memory{};
  const Memory* entity = nullptr;
  if (!GatherEntity(session, 0, memory, entity)) {
    Flush(&status, sizeof(status));
    return 1U;
  }
  Flush(reinterpret_cast<const void*>(args.address_table),
        static_cast<std::uint64_t>(args.table_capacity) * 8U);
  Flush(reinterpret_cast<const void*>(args.valid_table), args.table_capacity);
  Flush(reinterpret_cast<const void*>(args.indices),
        static_cast<std::uint64_t>(args.request_capacity) * 4U);
  Flush(reinterpret_cast<const void*>(args.offsets),
        static_cast<std::uint64_t>(args.request_capacity) * 4U);
  Flush(reinterpret_cast<const void*>(args.mask), args.request_capacity);
  Flush(reinterpret_cast<const void*>(args.output_indices),
        static_cast<std::uint64_t>(args.request_capacity) * 4U);
  const auto* mask = reinterpret_cast<const std::uint8_t*>(args.mask);
  Transfer transfer{};
  for (std::uint32_t index = 0; index < args.request_capacity; ++index)
    if (mask[index] != 0 && !MakeGetTransfer(args, *entity, index, &transfer)) {
      status.failed_request_index = index;
      Flush(&status, sizeof(status));
      return 1U;
    }
  status.result = 0;
  kvaf::aicpu_urma::UmdkTransfer transfers[32];
  std::uint32_t count = 0;
  for (std::uint32_t index = 0; index < args.request_capacity; ++index) {
    if (mask[index] != 0) {
      (void)MakeGetTransfer(args, *entity, index, &transfer);
      Flush(reinterpret_cast<void*>(transfer.local_address), transfer.length);
      transfers[count++] = {transfer.remote_address, transfer.local_address, transfer.length};
    }
    if (count != 0 && (count == 32 || index + 1 == args.request_capacity)) {
      status.result =
          KvafAicpuUmdkGatherTransfer(&session, false, transfers, count, 0, nullptr, nullptr);
      if (status.result != 0) {
        status.failed_request_index = index;
        break;
      }
      ++status.completed_batch_count;
      count = 0;
    }
  }
  Flush(reinterpret_cast<void*>(args.output), session.output_extent);
  Flush(&status, sizeof(status));
  return status.result == 0 ? 0U : 1U;
}

extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherRead(
    GatherReadArgs* args) {
  if (!args || !args->umdk_session || !args->queue_state || !args->core_status ||
      !args->worker_count || args->worker_count > 8U)
    return 1U;
  // Operation failures are returned in status so the stream can copy diagnostics.
  auto* statuses = reinterpret_cast<GatherReadStatus*>(args->core_status);
  auto* queues = reinterpret_cast<GatherQueueState*>(args->queue_state);
  for (unsigned worker = 0; worker < args->worker_count; ++worker) {
    statuses[worker] = {};
    statuses[worker].magic = kGatherStatusMagic;
    statuses[worker].sequence = args->sequence;
    statuses[worker].profile_detail = args->profile_detail;
  }
  queues[0].next_worker = queues[0].pipeline_sequence = 0U;
  auto& status = statuses[0];
  status.result = 2U;
  UmdkGatherMemory memory{};
  const auto scratch_offset = (args->output_extent + 63U) & ~std::uint64_t{63U};
  const auto source_bytes = kvaf::peer::GatherDescriptorBytes(args->source_span_count);
  if (KvafAicpuUmdkGatherMemory(args, &memory) || !args->sequence ||
      args->sequence >= kvaf::peer::kGatherErrorBit || !args->bytes || !args->output_spans ||
      !args->output_span_count || args->output_span_count > kGatherMaxSpans ||
      !args->source_span_count || args->source_span_count > kGatherMaxSpans || !args->sgl_size ||
      args->output_extent > UINT64_MAX - 63U || args->output != memory.local_base ||
      (args->pipeline_chunk_bytes &&
       (args->pipeline_chunk_bytes < 64U || args->pipeline_chunk_bytes % 64U)) ||
      (args->sq_batch_wqes != 8U && args->sq_batch_wqes != 16U && args->sq_batch_wqes != 32U &&
       args->sq_batch_wqes != 64U) ||
      !InRange(memory.local_base, memory.local_bytes, args->output,
               scratch_offset + GatherScratchBytes(args->pipeline_chunk_bytes)) ||
      !InRange(memory.local_base, memory.local_bytes, args->source_descriptor, source_bytes) ||
      !InRange(memory.local_base, memory.local_bytes, args->output_spans,
               args->output_span_count * sizeof(kvaf::GatherOutputSpan)) ||
      !InRange(memory.remote_base, memory.remote_bytes, args->remote_descriptor, source_bytes) ||
      !InRange(memory.remote_base, memory.remote_bytes, args->remote_staging, args->bytes) ||
      !InRange(memory.remote_base, memory.remote_bytes, args->remote_notify,
               sizeof(kvaf::peer::GatherControl)))
    return 0U;
  const auto layout_start = NowNs();
  const auto* spans = reinterpret_cast<const kvaf::GatherOutputSpan*>(args->output_spans);
  std::uint64_t bytes = 0U;
  unsigned next_worker = 0U;
  for (unsigned index = 0; index < args->output_span_count; ++index) {
    while (next_worker < args->worker_count &&
           index == static_cast<std::uint64_t>(args->output_span_count) * next_worker /
                        args->worker_count)
      queues[next_worker++].staging_offset = bytes;
    const auto& span = spans[index];
    if (span.reserved || !span.length || span.offset > args->output_extent ||
        span.length > args->output_extent - span.offset || bytes > args->bytes ||
        span.length > args->bytes - bytes)
      return 0U;
    bytes += span.length;
  }
  while (next_worker < args->worker_count) queues[next_worker++].staging_offset = bytes;
  if (bytes != args->bytes) return 0U;
  status.output_layout_ns = NowNs() - layout_start;
  status.result = 0U;
  const auto scratch = args->output + scratch_offset;
  *reinterpret_cast<std::uint64_t*>(scratch + 8U) = 0U;
  *reinterpret_cast<std::uint64_t*>(scratch + 16U) = args->remote_descriptor;
  const UmdkTransfer descriptor{args->remote_descriptor, args->source_descriptor, source_bytes};
  status.phase = 1U;
  const auto source_start = NowNs();
  status.result = KvafAicpuUmdkGatherTransfer(
      args, true, &descriptor, 1U, 0U, &status.source_write_post_ns, &status.source_write_cq_ns);
  status.source_write_ns = NowNs() - source_start;
  if (status.result) return 0U;
  status.phase = 2U;
  const auto notify_start = NowNs();
  *reinterpret_cast<std::uint64_t*>(scratch) = args->sequence;
  Flush(reinterpret_cast<void*>(scratch), 24U);
  const UmdkTransfer notify[] = {{args->remote_command, scratch + 16U, 8U},
                                 {args->remote_notify, scratch, 8U}};
  status.result = KvafAicpuUmdkGatherTransfer(args, true, notify, 2U, 0U, &status.notify_post_ns,
                                              &status.notify_cq_ns);
  status.notify_ns = NowNs() - notify_start;
  if (!status.result && !args->pipeline_chunk_bytes) {
    status.phase = 3U;
    WaitAck(*args, status);
  }
  if (!status.result && args->pipeline_chunk_bytes)
    __atomic_store_n(&queues[0].pipeline_sequence, args->sequence, __ATOMIC_RELEASE);
  if (!status.result && args->worker_count == 1U) {
    ReadRange(*args, 0U, status);
    if (!status.result && args->pipeline_chunk_bytes) WaitAck(*args, status);
  }
  Barrier();
  return 0U;
}

extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherScatter(
    GatherReadArgs* args) {
  if (!args || !args->queue_state || !args->core_status || args->worker_count < 2U ||
      args->worker_count > 8U)
    return 1U;
  auto* queues = reinterpret_cast<GatherQueueState*>(args->queue_state);
  const auto worker = __atomic_fetch_add(&queues[0].next_worker, 1U, __ATOMIC_RELAXED);
  if (worker >= args->worker_count) return 1U;
  auto* statuses = reinterpret_cast<GatherReadStatus*>(args->core_status);
  auto& status = statuses[worker];
  const bool ready =
      (args->pipeline_chunk_bytes
           ? __atomic_load_n(&queues[0].pipeline_sequence, __ATOMIC_ACQUIRE) == args->sequence
           : statuses[0].ack_seen == args->sequence);
  if (!ready) {
    if (worker != 0U) status.result = 11U;
    return 0U;
  }
  ReadRange(*args, worker, status);
  if (!worker && !status.result && args->pipeline_chunk_bytes) WaitAck(*args, status);
  Barrier();
  return 0U;
}
