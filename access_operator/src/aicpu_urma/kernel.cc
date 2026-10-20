#include <atomic>
#include <cstring>

#include "cpu_gather.h"
#include "gather_cache.h"
#include "umdk_gather.h"

namespace {
using namespace kvaf::aicpu_urma;

bool InRange(std::uint64_t base, std::uint64_t size, std::uint64_t address, std::uint64_t bytes) {
  return bytes && address >= base && bytes <= size && address - base <= size - bytes;
}
void Flush(const void* address, std::uint64_t bytes) { FlushGatherCache(address, bytes); }

void WaitAck(const GatherReadArgs& args, GatherReadStatus& status) {
  const UmdkTransfer transfer{args.remote_ack, args.scratch + 8U, 8U};
  for (unsigned poll = 0; poll < 10000U; ++poll) {
    Flush(reinterpret_cast<void*>(transfer.local), transfer.bytes);
    status.result = KvafAicpuUmdkGatherTransfer(&args, false, &transfer, 1U, 0U);
    if (status.result) return;
    Flush(reinterpret_cast<void*>(transfer.local), transfer.bytes);
    status.ack_seen = *reinterpret_cast<const volatile std::uint64_t*>(transfer.local);
    if (status.ack_seen == args.sequence) return;
    if (status.ack_seen == (args.sequence | kvaf::peer::kGatherErrorBit)) {
      status.result = 10U;
      return;
    }
  }
  status.result = 9U;
}

void ReadRange(const GatherReadArgs& args, unsigned worker, GatherReadStatus& status) {
  const auto first =
      static_cast<std::uint64_t>(args.output_span_count) * worker / args.worker_count;
  const auto last =
      static_cast<std::uint64_t>(args.output_span_count) * (worker + 1U) / args.worker_count;
  const auto* queues = reinterpret_cast<const GatherQueueState*>(args.queue_state);
  status.phase = 4U;
  status.result = KvafAicpuUmdkGatherScatter(&args, &status, first, last,
                                             queues[worker].staging_offset, worker);
  const auto* targets = reinterpret_cast<const kvaf::GatherOutputSpan*>(args.output_spans);
  for (auto index = first; index < last; ++index)
    Flush(reinterpret_cast<void*>(args.output + targets[index].offset), targets[index].length);
}

bool BuildCurrentRequest(const UmdkBatchGetArgs& launch, GatherReadArgs& args) {
  const auto& request = launch.request;
  const auto n = request.request_capacity;
  UmdkGatherMemory memory{};
  if (!request.address_table || !request.valid_table || !request.indices || !request.offsets ||
      !request.mask || !request.output_indices || !args.output || !request.token_bytes ||
      !request.tokens_per_block || n > kGatherMaxSpans || KvafAicpuUmdkGatherMemory(&args, &memory))
    return false;
  Flush(reinterpret_cast<void*>(request.address_table), request.table_capacity * 8ULL);
  Flush(reinterpret_cast<void*>(request.valid_table), request.table_capacity);
  Flush(reinterpret_cast<void*>(request.indices), n * 4ULL);
  Flush(reinterpret_cast<void*>(request.offsets), n * 4ULL);
  Flush(reinterpret_cast<void*>(request.mask), n);
  Flush(reinterpret_cast<void*>(request.output_indices), n * 4ULL);
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(request.address_table);
  const auto* valid = reinterpret_cast<const std::uint8_t*>(request.valid_table);
  const auto* indices = reinterpret_cast<const std::int32_t*>(request.indices);
  const auto* offsets = reinterpret_cast<const std::int32_t*>(request.offsets);
  const auto* mask = reinterpret_cast<const std::uint8_t*>(request.mask);
  const auto* output_indices = reinterpret_cast<const std::int32_t*>(request.output_indices);
  auto* descriptor = reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(args.source_descriptor);
  auto* sources = reinterpret_cast<kvaf::peer::GatherSpan*>(descriptor + 1);
  auto* targets = reinterpret_cast<kvaf::GatherOutputSpan*>(args.output_spans);
  const auto block_bytes =
      static_cast<std::uint64_t>(request.token_bytes) * request.tokens_per_block;
  std::uint64_t total = 0;
  for (unsigned i = 0; i < n; ++i) {
    if (!mask[i]) continue;
    const auto index = indices[i], offset = offsets[i], row = output_indices[i];
    if (index < 0 || static_cast<unsigned>(index) >= request.table_capacity || !valid[index] ||
        offset < 0 || static_cast<std::uint64_t>(offset) > block_bytes - request.token_bytes ||
        row < 0 || static_cast<unsigned>(row) >= request.output_rows ||
        !InRange(memory.remote_base, args.remote_staging - memory.remote_base, addresses[index],
                 block_bytes) ||
        total > UINT32_MAX - request.token_bytes)
      return false;
    sources[args.source_span_count++] = {addresses[index] + offset, request.token_bytes, 0U};
    const auto output_offset = static_cast<std::uint64_t>(row) * request.token_bytes;
    targets[args.output_span_count++] = {output_offset, request.token_bytes, 0U};
    total += request.token_bytes;
  }
  args.bytes = total;
  *descriptor = {kvaf::peer::kGatherDescriptorMagic,
                 args.sequence,
                 args.remote_staging,
                 args.source_span_count,
                 args.bytes,
                 args.pipeline_chunk_bytes};
  Flush(descriptor, kvaf::peer::GatherDescriptorBytes(args.source_span_count));
  Flush(targets, args.output_span_count * sizeof(*targets));
  Flush(reinterpret_cast<void*>(args.output), args.output_extent);
  return true;
}

bool StartGather(GatherReadArgs& args, GatherReadStatus& status) {
  const auto* targets = reinterpret_cast<const kvaf::GatherOutputSpan*>(args.output_spans);
  auto* queues = reinterpret_cast<GatherQueueState*>(args.queue_state);
  std::uint64_t bytes = 0U;
  unsigned next = 0U;
  for (unsigned i = 0; i < args.output_span_count; ++i) {
    while (next < args.worker_count &&
           i == static_cast<std::uint64_t>(args.output_span_count) * next / args.worker_count)
      queues[next++].staging_offset = bytes;
    bytes += targets[i].length;
  }
  while (next < args.worker_count) queues[next++].staging_offset = bytes;
  const UmdkTransfer descriptor{args.remote_descriptor, args.source_descriptor,
                                kvaf::peer::GatherDescriptorBytes(args.source_span_count)};
  status.phase = 1U;
  status.result = KvafAicpuUmdkGatherTransfer(&args, true, &descriptor, 1U, 0U);
  if (status.result) return false;
  status.phase = 2U;
  auto* scratch = reinterpret_cast<std::uint64_t*>(args.scratch);
  scratch[0] = args.sequence;
  scratch[1] = 0U;
  scratch[2] = args.remote_descriptor;
  Flush(scratch, 24U);
  const UmdkTransfer notify[] = {{args.remote_command, args.scratch + 16U, 8U},
                                 {args.remote_notify, args.scratch, 8U}};
  status.result = KvafAicpuUmdkGatherTransfer(&args, true, notify, 2U, 0U);
  if (!status.result && !args.pipeline_chunk_bytes) {
    status.phase = 3U;
    WaitAck(args, status);
  }
  if (!status.result)
    __atomic_store_n(&queues[0].pipeline_sequence, args.sequence, __ATOMIC_RELEASE);
  return !status.result;
}
}  // namespace

extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherRead(
    UmdkBatchGetArgs* launch) {
  if (!launch || !launch->gather.core_status || !launch->gather.scratch ||
      !launch->gather.queue_state || !launch->gather.worker_count ||
      launch->gather.worker_count > 8U)
    return 1U;
  auto& args = *reinterpret_cast<GatherReadArgs*>(launch->gather.scratch + kGatherArgsOffset);
  args = launch->gather;
  auto* statuses = reinterpret_cast<GatherReadStatus*>(args.core_status);
  for (unsigned i = 0; i < args.worker_count; ++i) {
    statuses[i] = {};
    statuses[i].magic = kGatherStatusMagic;
    statuses[i].sequence = args.sequence;
  }
  auto* queues = reinterpret_cast<GatherQueueState*>(args.queue_state);
  queues[0].next_worker = queues[0].pipeline_sequence = 0U;
  auto& status = statuses[0];
  if (!BuildCurrentRequest(*launch, args))
    status.result = 2U;
  else if (!args.bytes)
    status.ack_seen = args.sequence;
  else if (StartGather(args, status) && args.worker_count == 1U) {
    ReadRange(args, 0U, status);
    if (!status.result && args.pipeline_chunk_bytes) WaitAck(args, status);
  }
  Flush(&args, sizeof(args));
  Flush(queues, args.worker_count * sizeof(*queues));
  Flush(statuses, args.worker_count * sizeof(*statuses));
  return status.result ? 1U : 0U;
}

extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherScatter(
    UmdkBatchGetArgs* launch) {
  if (!launch || !launch->gather.scratch) return 1U;
  auto* current = reinterpret_cast<GatherReadArgs*>(launch->gather.scratch + kGatherArgsOffset);
  Flush(current, sizeof(*current));
  const auto args = *current;
  auto* queues = reinterpret_cast<GatherQueueState*>(args.queue_state);
  auto* statuses = reinterpret_cast<GatherReadStatus*>(args.core_status);
  const auto worker = __atomic_fetch_add(&queues[0].next_worker, 1U, __ATOMIC_RELAXED);
  if (worker >= args.worker_count) return 1U;
  auto& status = statuses[worker];
  if (!args.bytes) return 0U;
  if (__atomic_load_n(&queues[0].pipeline_sequence, __ATOMIC_ACQUIRE) != args.sequence)
    status.result = 11U;
  else {
    ReadRange(args, worker, status);
    if (!worker && !status.result && args.pipeline_chunk_bytes) WaitAck(args, status);
  }
  Flush(&status, sizeof(status));
  return status.result ? 1U : 0U;
}
