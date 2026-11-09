// Executes the production .asc against decoded SQ/CQ and a real CPU gather executor.
// DMA cache visibility and CANN code generation still require hardware tests.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "kernel_operator.h"
#include "cpu_gather_executor.h"
#include "cpu_push_get_abi.h"
#include "sequential_batch_get_abi.h"
#include "../access_operator/demo/aiv_batch_get/kernel/execute_scatter_batch_get_aiv.asc"
// The production Ascend metadata section is not a valid native Mach-O section.
// Keep the metadata object/value in this model, as in the direct kernel model.
#define __attribute__(...)
#include "../access_operator/demo/aiv_batch_get/kernel/execute_gather_batch_get_aiv.asc"
#undef __attribute__

namespace {
using namespace kvaf_scatter;
using kvaf_gather::kRead;
using kvaf_gather::kWrite;
using kvaf_gather::kErrorBit;
// Test fixture only; production launch ABIs are independent.
struct ModelArgs {
  KvafExecuteBatchGetKernelArgs request;
  uint64_t workspace;
  uint64_t workspace_bytes;
  uint64_t workspace_entity;
  uint64_t descriptor_offset;
  uint64_t staging_offset;
  uint64_t lane_descriptor_bytes;
  uint64_t lane_staging_bytes;
  uint64_t remote_staging;
  uint64_t remote_descriptor;
  uint64_t remote_control;
  uint64_t source_base;
  uint64_t source_bytes;
  uint64_t sequence;
  uint32_t mode;
  uint32_t sgl_size;
  uint32_t max_polls;
  uint32_t scatter_probe;  // Former reserved word; zero preserves normal execution.
  uint64_t profile{};
  uint32_t gather_probe{};
  uint32_t coalesce_sges{};
  uint32_t pipeline_chunk_bytes{};
  uint32_t pipeline_shards{};
  uint32_t progress_mode{};
};
void Launch(const ModelArgs& a) {
  if (a.mode == KVAF_SEQ_SCATTER) {
    KvafScatterBatchGetAivKernelV1({a.request, a.workspace, a.workspace_bytes,
      a.descriptor_offset, a.staging_offset, a.source_base, a.source_bytes,
      a.sequence, a.max_polls, a.scatter_probe});
  } else {
    KvafGatherBatchGetAivKernelV7({a.request, a.workspace, a.workspace_bytes,
      a.workspace_entity, a.descriptor_offset, a.lane_descriptor_bytes,
      a.lane_staging_bytes, a.remote_staging, a.remote_descriptor, a.remote_control,
      a.source_base, a.source_bytes, a.sequence, a.sgl_size, a.max_polls, a.profile,
      a.gather_probe, a.coalesce_sges, a.pipeline_chunk_bytes, a.pipeline_shards, a.progress_mode, 0});
  }
}
uint64_t Address(const void* p) { return reinterpret_cast<uint64_t>(p); }
uint64_t Align(uint64_t n) { return (n + 63) & ~uint64_t(63); }
template<class T> struct AlignedAllocator {
  using value_type = T;
  AlignedAllocator() = default;
  template<class U> AlignedAllocator(const AlignedAllocator<U>&) {}
  T* allocate(std::size_t n) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, n * sizeof(T))) throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) { std::free(p); }
  template<class U> bool operator==(const AlignedAllocator<U>&) const { return true; }
  template<class U> bool operator!=(const AlignedAllocator<U>&) const { return false; }
};
struct ModelQueue {
  struct PostedWqe {
    uint64_t pi;
    uint32_t blocks;
    std::array<uint32_t, 64> words{};
  };
  std::vector<uint32_t, AlignedAllocator<uint32_t>> sq =
      std::vector<uint32_t, AlignedAllocator<uint32_t>>(16 * 16);
  std::vector<PostedWqe> posted;
  std::vector<std::vector<uint64_t>> write_batches;
  alignas(64) uint32_t cq[2][16]{};
  uint32_t db{}, cq_db{};
  uint64_t pi{}, completed{}, cq_pi{}, cq_completed{}, doorbells{}, first_db_vfs{};
  uint64_t fail_cq_at{UINT64_MAX};
  uint64_t drop_cq_at{UINT64_MAX};
  uint64_t release_vf{}, sq_cache_lines{};
  uint32_t deferred_word{};
  bool deferred_cq{};
};
struct Fixture;
Fixture* fixture;
struct Fixture {
  uint32_t count, token, workers;
  int32_t active;
  ModelArgs args{};
  std::vector<uint8_t, AlignedAllocator<uint8_t>> remote, workspace;
  std::vector<uint8_t> output, expected;
  std::vector<uint8_t, AlignedAllocator<uint8_t>> valid;
  std::vector<uint64_t, AlignedAllocator<uint64_t>> addresses;
  std::vector<int32_t, AlignedAllocator<int32_t>> indices, offsets, rows;
  std::vector<KvafSequentialStatus> status;
  std::vector<KvafGatherBatchGetCoreProfile> profiles;
  std::vector<KvafBatchGetQueueState> states;
  std::vector<KvafBatchGetDirectUrmaEntity> control_entities, data_entities;
  std::vector<std::unique_ptr<ModelQueue>> queues;
  std::vector<uint64_t> notified;
  std::vector<uint64_t> read_vf_begin, read_overlap_begin;
  std::vector<uint64_t> scratch_cache_lines;
  std::vector<uint32_t> ack_reads;
  std::vector<uint32_t> ready_reads;
  std::vector<uint64_t> first_read_ready;
  std::vector<bool> cpu_done;
  std::vector<uint8_t> shard_read_rows;
  bool push_done{}, timeout{}, remote_error{}, cq_error{}, data_cq_error{}, future_done{}, data_timeout{};
  uint32_t push_polls{}, push_requests{}, data_reads{}, max_sges{}, completions{};
  uint64_t launch_vf_begin{};
  uint64_t descriptor_cache_lines{}, valid_table_cache_lines{};
  uint64_t simt_input_cache_lines{}, row_cache_lines{};
  // Deterministic CPU producer: progress queries advance one copy chunk, while
  // the CPU's final ACK is deliberately withheld until three ACK queries.
  uint32_t ready_fault{}, ready_step{1}, ready_idle{2};
  bool ack_timeout{}, ack_error{}, queried_with_data_inflight{}, all_shards_ready{};
  bool odd_ready_wave{}, tail_in_middle{}, prepare_error{};
  std::vector<bool> packed_query_had_backlog = std::vector<bool>(8);
  bool hold_data_cq_for_ack{}, delay_progress_cq{}, ack_with_data_inflight{}, prefetched_during_vf{};

  Fixture(uint32_t n, uint32_t bytes, uint32_t cores, uint32_t mode, uint32_t sgl)
      : count(n), token(bytes), workers(cores), active(n), valid(n, 1), addresses(n),
        indices(n), offsets(n), rows(n), status(cores), profiles(cores), states(cores * 2),
        control_entities(cores * 2), data_entities(cores * 2), notified(cores),
        read_vf_begin(cores), read_overlap_begin(cores),
        scratch_cache_lines(cores), ack_reads(cores), ready_reads(cores),
        first_read_ready(cores), cpu_done(cores) {
    fixture = this;
    const uint64_t source_bytes = uint64_t(n) * bytes * 2;
    const uint64_t lane_staging = Align(uint64_t((n + cores - 1) / cores) * bytes);
    const uint64_t lane_descriptor = Align(48 + uint64_t((n + cores - 1) / cores) * 16) + kvaf::kGatherProgressBytes;
    const auto staging_offset = Align(source_bytes);
    const auto descriptor_offset = staging_offset + cores * lane_staging;
    const auto control_offset = descriptor_offset + cores * lane_descriptor;
    remote.resize(control_offset + cores * 64 + 128);
    for (uint64_t i = 0; i < source_bytes; ++i) remote[i] = (i * 31 + i / 7) % 251;
    output.assign(uint64_t(n * 2 + 2) * bytes, 0xad);
    for (uint32_t i = 0; i < n; ++i) {
      addresses[i] = Address(remote.data()) + uint64_t(i) * bytes * 2;
      indices[i] = (n - i - 1) / 2; // Repeated sources, shuffled relative to output.
      offsets[i] = (i & 1) ? bytes : 0;
      rows[i] = 2 * (n - i - 1) + 1;
    }
    auto& r = args.request;
    r.address_table = Address(addresses.data()); r.valid_table = Address(valid.data());
    r.indices = Address(indices.data()); r.offsets = Address(offsets.data());
    r.output_indices = Address(rows.data()); r.miss_count = Address(&active);
    r.output = Address(output.data()); r.table_capacity = n; r.request_capacity = n;
    r.output_rows = n * 2 + 2; r.token_bytes = bytes; r.tokens_per_block = 2;
    r.block_count = cores; r.core_status = Address(status.data());
    r.queue_state = Address(states.data());
    args.mode = mode; args.sequence = 1; args.sgl_size = sgl; args.max_polls = 64;
    args.source_base = Address(remote.data()); args.source_bytes = source_bytes;
    args.lane_descriptor_bytes = lane_descriptor; args.lane_staging_bytes = lane_staging;
    args.remote_staging = Address(remote.data()) + staging_offset;
    args.remote_descriptor = Address(remote.data()) + descriptor_offset;
    args.remote_control = Address(remote.data()) + control_offset;
    args.descriptor_offset = mode == KVAF_SEQ_SCATTER ? 192 + cores * 64 : cores * kvaf::kGatherScratchBytes;
    args.staging_offset = Align(args.descriptor_offset + n * 8ULL);
    args.workspace_bytes = mode == KVAF_SEQ_SCATTER ? args.staging_offset + n * uint64_t(bytes) + 64 :
        args.descriptor_offset + cores * lane_descriptor;
    // Match the cache-line alignment of real registered DDR/HBM allocations.
    workspace.resize(args.workspace_bytes);
    args.workspace = Address(workspace.data());
    for (uint32_t q = 0; q < cores * 2; ++q) {
      queues.push_back(std::make_unique<ModelQueue>());
      auto& m = *queues.back();
      auto& e = control_entities[q];
      e.magic = 0x4b564155; e.version = 1; e.local_token_id = 1 + q;
      e.sq_va = Address(m.sq.data()); e.sq_db_va = Address(&m.db); e.sq_depth = 16; e.wqe_size = 64;
      e.cq_va = Address(m.cq); e.cq_db_va = Address(&m.cq_db); e.cq_depth = 2; e.cqe_size = 64;
      e.remote_token_id = 0xab100002 + q; e.remote_token_value = 0xabcd0003 + q;
      e.tp_id = 0x12ab0004 + q;
      for (uint32_t i = 0; i < 16; ++i) e.remote_eid[i] = q * 17 + i + 1;
      data_entities[q] = e;
    }
    args.workspace_entity = Address(control_entities.data());
    r.channel_entity = Address(data_entities.data());
  }
  void CheckGuard() { assert(std::all_of(output.begin(), output.end(), [](uint8_t b) { return b == 0xad; })); }
  void ConsecutiveReadRows(bool segmented) {
    for (uint32_t i = 0; i < count; ++i) rows[i] = i + 1;
    if (!segmented) return;
    std::vector<uint32_t> starts;
    for (uint32_t core = 0; core < workers; ++core) {
      const uint32_t first = uint64_t(active) * core / workers;
      const uint32_t last = uint64_t(active) * (core + 1) / workers;
      for (uint32_t begin = first; begin + args.sgl_size <= last; begin += args.sgl_size)
        starts.push_back(begin);
    }
    for (std::size_t i = 0; i < starts.size(); ++i)
      for (uint32_t j = 0; j < args.sgl_size; ++j)
        rows[starts[i] + j] = starts[starts.size() - i - 1] + j + 1;
  }
  void DataQueueDepth(uint32_t depth) {
    for (uint32_t core = 0; core < workers; ++core) {
      auto& m = *queues[core * 2];
      m.sq.resize(depth * 16);
      data_entities[core * 2].sq_va = Address(m.sq.data());
      data_entities[core * 2].sq_depth = depth;
    }
  }
  void Poll(void* p) {
    for (uint32_t q = 0; q < workers * 2; ++q) {
      auto& m = *queues[q];
      if (Address(p) >= Address(m.sq.data()) && Address(p) < Address(m.sq.data()) + m.sq.size() * 4)
        ++m.sq_cache_lines;
      if (m.deferred_cq && m.release_vf != UINT64_MAX &&
          aiv_model::vf_calls >= m.release_vf && Address(p) == Address(m.cq[m.cq_pi % 2])) {
        m.cq[m.cq_pi % 2][0] = m.deferred_word;
        ++m.cq_pi; ++completions;
        m.deferred_cq = false;
        prefetched_during_vf = true;
      }
    }
    if (args.mode == KVAF_SEQ_GATHER) {
      const auto address = Address(p);
      if (args.progress_mode == kvaf::kGatherProgressPush && address >= args.workspace &&
          address < args.workspace + workers * uint64_t(kvaf::kGatherScratchBytes) &&
          (address - args.workspace) % kvaf::kGatherScratchBytes == 64) {
        const auto worker = uint32_t((address - args.workspace) / kvaf::kGatherScratchBytes);
        if (notified[worker] == args.sequence) {
          AdvanceShards(worker);
          const auto* h = reinterpret_cast<const kvaf::peer::GatherDescriptorHeader*>(
              args.remote_descriptor + worker * args.lane_descriptor_bytes);
          std::memcpy(reinterpret_cast<void*>(address),
              reinterpret_cast<const void*>(Address(h) + kvaf::GatherProgressOffset(h->span_count)),
              kvaf::kGatherProgressBytes);
        }
      }
      if (address >= args.workspace && address < args.workspace + workers * uint64_t(kvaf::kGatherScratchBytes) &&
          (address - args.workspace) % kvaf::kGatherScratchBytes < 64)
        ++scratch_cache_lines[(address - args.workspace) / kvaf::kGatherScratchBytes];
      if (address >= args.workspace + args.descriptor_offset &&
          address < args.workspace + args.workspace_bytes) ++descriptor_cache_lines;
      // Ignore partial boundary lines which unrelated allocations can share.
      if (address >= Address(valid.data()) && address + 64 <= Address(valid.data()) + valid.size())
        ++valid_table_cache_lines;
      if ((address >= Address(addresses.data()) && address < Address(addresses.data()) + addresses.size() * 8) ||
          (address >= Address(indices.data()) && address < Address(indices.data()) + indices.size() * 4) ||
          (address >= Address(offsets.data()) && address < Address(offsets.data()) + offsets.size() * 4))
        ++simt_input_cache_lines;
      if (address >= Address(rows.data()) && address < Address(rows.data()) + rows.size() * 4)
        ++row_cache_lines;
    }
    if (args.mode != KVAF_SEQ_SCATTER || Address(p) < (args.workspace & ~uint64_t(63)) ||
        Address(p) >= args.workspace + args.workspace_bytes || push_done) return;
    auto* request = reinterpret_cast<kvaf::cpu_push::Request*>(workspace.data());
    uint64_t notify; std::memcpy(&notify, workspace.data() + 64, 8);
    if (notify != args.sequence) return;
    CheckGuard();
    if (timeout || ++push_polls < 4) return;
    assert(request->magic == kvaf::cpu_push::kMagic && request->sequence == args.sequence);
    assert(request->count == uint32_t(active));
    const auto* src = reinterpret_cast<const uint64_t*>(workspace.data() + args.descriptor_offset);
    for (uint32_t i = 0; i < uint32_t(active); ++i)
      std::memcpy(workspace.data() + args.staging_offset + uint64_t(i) * token,
                  reinterpret_cast<void*>(src[i]), token);
    const uint64_t done = ((args.sequence + (future_done ? 1 : 0)) << 32) | (remote_error ? 5 : 1);
    std::memcpy(workspace.data() + 128, &done, 8);
    push_done = true; ++push_requests;
  }
  void CheckDescriptor(uint32_t worker) {
    assert(aiv_model::raw_copies.empty());  // Ready before the NIC reads HBM.
    const uint32_t first = uint64_t(active) * worker / workers;
    const uint32_t last = uint64_t(active) * (worker + 1) / workers;
    const auto* local = workspace.data() + args.descriptor_offset + worker * args.lane_descriptor_bytes;
    const auto* h = reinterpret_cast<const kvaf::peer::GatherDescriptorHeader*>(local);
    assert(h->magic == (args.pipeline_shards ? kvaf::kGatherShardMagic : kvaf::peer::kGatherDescriptorMagic) && h->sequence == args.sequence);
    assert(h->staging_address == args.remote_staging + worker * args.lane_staging_bytes);
    assert(h->span_count == last - first && h->total_bytes == (last - first) * token);
    assert(h->pipeline_chunk_bytes == args.pipeline_chunk_bytes && h->span_bytes == token && h->ready_bytes == (args.pipeline_shards ? args.sgl_size : 0));
    const auto* spans = reinterpret_cast<const kvaf::peer::GatherSpan*>(local + sizeof(*h));
    for (uint32_t i = first; i < last; ++i) {
      const auto& span = spans[i - first];
      assert(span.address == addresses[indices[i]] + offsets[i]);
      assert(span.length == token && !span.reserved);
    }
    const auto bytes = sizeof(*h) + (last - first) * sizeof(*spans);
    const auto padded = Align(bytes) + (args.pipeline_shards ? kvaf::kGatherProgressBytes : 0);
    for (uint64_t i = bytes; i < padded; ++i) assert(local[i] == 0);
    for (uint64_t i = padded; i < args.lane_descriptor_bytes; ++i) assert(local[i] == 0xa7);
  }
  uint64_t ShardReadyBytes(uint32_t worker) {
    const auto* h = reinterpret_cast<const kvaf::peer::GatherDescriptorHeader*>(
        args.remote_descriptor + worker * args.lane_descriptor_bytes);
    const auto total = (h->span_count + args.sgl_size - 1) / args.sgl_size;
    const auto shards = kvaf::GatherShardCount(total);
    const auto* progress = reinterpret_cast<const uint64_t*>(Address(h) + kvaf::GatherProgressOffset(h->span_count));
    uint64_t bytes = 0;
    for (uint32_t s = 0; s < shards; ++s) {
      if ((progress[s] >> 32) != args.sequence) continue;
      const auto first = kvaf::GatherShardBoundary(total, shards, s);
      bytes += std::min<uint64_t>(h->total_bytes, uint64_t(first + uint32_t(progress[s])) * args.sgl_size * token) -
          uint64_t(first) * args.sgl_size * token;
    }
    return bytes;
  }
  void AdvanceShards(uint32_t worker) {
    if (args.pipeline_shards == 2) {
      uint64_t read_bytes = 0;
      for (uint32_t i = uint64_t(active) * worker / workers;
           i < uint64_t(active) * (worker + 1) / workers; ++i)
        read_bytes += uint64_t(shard_read_rows[i]) * token;
      packed_query_had_backlog[worker] = ShardReadyBytes(worker) > read_bytes;
    }
    ++ready_reads[worker];
    auto* h = reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(
        args.remote_descriptor + worker * args.lane_descriptor_bytes);
    const auto total = (h->span_count + args.sgl_size - 1) / args.sgl_size;
    const auto shards = kvaf::GatherShardCount(total);
    auto* progress = reinterpret_cast<uint64_t*>(Address(h) + kvaf::GatherProgressOffset(h->span_count));
    if (!queues[worker * 2]->posted.empty()) queried_with_data_inflight = true;
    if (timeout || prepare_error || ready_reads[worker] <= ready_idle) return;
    // Delay the first logical shard. Later shards really do get copied and
    // consumed before it, rather than merely reporting a fabricated frontier.
    for (uint32_t reverse = shards; reverse > 0; --reverse) {
      const auto shard = reverse - 1;
      if (!shard && shards > 1 && !first_read_ready[worker] && !all_shards_ready && !odd_ready_wave)
        continue; // Keep shard zero unavailable until a later shard is consumed.
      if (odd_ready_wave && ready_reads[worker] == ready_idle + 1 && !(shard & 1)) continue;
      const auto first = kvaf::GatherShardBoundary(total, shards, shard);
      const auto last = kvaf::GatherShardBoundary(total, shards, shard + 1);
      const auto old = uint32_t(progress[shard]);
      if (old == last - first) continue;
      const auto step = std::max<uint32_t>(1, args.pipeline_chunk_bytes / (args.sgl_size * token));
      const auto done = std::min(last - first, old + step * ready_step);
      const auto begin = uint64_t(first + old) * args.sgl_size;
      const auto end = std::min<uint64_t>(h->span_count, uint64_t(first + done) * args.sgl_size);
      const auto* spans = reinterpret_cast<const kvaf::peer::GatherSpan*>(h + 1);
      for (auto i = begin; i < end; ++i)
        std::memcpy(reinterpret_cast<void*>(h->staging_address + i * token),
                    reinterpret_cast<const void*>(spans[i].address), token);
      progress[shard] = kvaf::GatherShardProgress(args.sequence, done);
      if (ready_fault == 1) progress[shard] = kvaf::GatherShardProgress(args.sequence - 1, done);
      if (ready_fault == 2) progress[shard] = kvaf::GatherShardProgress(args.sequence + 1, done);
      if (ready_fault == 4) progress[shard] = kvaf::GatherShardProgress(args.sequence, last - first + 1);
      if (ready_fault == 3 && first_read_ready[worker])
        progress[shards - 1] = kvaf::GatherShardProgress(args.sequence, 0);
      if (!all_shards_ready && !odd_ready_wave) break;
    }
    if (ready_fault == 5) {
      auto* c = reinterpret_cast<kvaf::peer::GatherControl*>(args.remote_control) + worker;
      c->ack = args.sequence | kErrorBit;
    }
  }
  void AdvanceGather(uint32_t worker) {
    ++ready_reads[worker];
    auto* h = reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(
        args.remote_descriptor + worker * args.lane_descriptor_bytes);
    assert(h->sequence == args.sequence && h->pipeline_chunk_bytes == args.pipeline_chunk_bytes);
    if (!queues[worker * 2]->posted.empty()) queried_with_data_inflight = true;
    if (timeout || prepare_error || ready_reads[worker] <= ready_idle) return;
    const auto begin = h->ready_bytes;
    const auto end = std::min<uint64_t>(h->total_bytes,
        begin + uint64_t(args.pipeline_chunk_bytes) * ready_step);
    const auto* spans = reinterpret_cast<const kvaf::peer::GatherSpan*>(h + 1);
    for (auto offset = begin; offset < end;) {
      const auto index = offset / token, in_span = offset % token;
      const auto bytes = std::min<uint64_t>(token - in_span, end - offset);
      std::memcpy(reinterpret_cast<void*>(h->staging_address + offset),
                  reinterpret_cast<const void*>(spans[index].address + in_span), bytes);
      offset += bytes;
    }
    h->ready_bytes = end;
    // Faults 3/5 follow a real partial submission, exercising an unsignaled
    // data wave. Other malformed responses fail before any payload submission.
    if (ready_fault && ((ready_fault != 3 && ready_fault != 5) || first_read_ready[worker])) {
      if (ready_fault == 1) --h->sequence;
      if (ready_fault == 2) ++h->sequence;
      if (ready_fault == 3) h->ready_bytes = 0;
      if (ready_fault == 4) h->ready_bytes = uint64_t(h->total_bytes) + 64;
      if (ready_fault == 5) h->ready_bytes = kErrorBit;
      if (ready_fault == 6) h->ready_bytes = 1;
    }
  }
  void Doorbell(uint32_t value, uint32_t* p) {
    for (uint32_t q = 0; q < workers * 2; ++q) {
      auto& m = *queues[q];
      if (p == &m.cq_db) {
        assert(m.cq_completed < m.cq_pi);
        ++m.cq_completed;
        assert(value == (m.cq_completed & 0x003fffffU));
        m.cq_db = value; m.completed = m.pi; m.posted.clear();
        return;
      }
      if (p != &m.db) continue;
      const uint32_t worker = q / 2;
      const uint32_t depth = m.sq.size() / 16;
      // Already published headers/data must never be patched or overwritten
      // before the ordered tail CQ has retired their physical SQ wave.
      for (const auto& previous : m.posted)
        for (uint32_t i = 0; i < previous.blocks * 16; ++i)
          assert(m.sq[(previous.pi * 16 + i) % m.sq.size()] == previous.words[i]);
      if (!m.doorbells)
        m.first_db_vfs = aiv_model::vf_calls - (q % 2 ? launch_vf_begin : read_vf_begin[worker]);
      ++m.doorbells;
      const uint64_t end = m.pi + uint32_t(value - uint32_t(m.pi));
      std::vector<uint64_t> writes;
      auto word = [&](uint64_t base, uint32_t index) {
        return m.sq[((base * 16) + index) % m.sq.size()];
      };
      while (m.pi < end) {
        const auto header = word(m.pi, 0);
        assert((header >> 31) == (1U - ((m.pi / depth) & 1)));
        const auto op = (word(m.pi, 1) >> 8) & 255;
        const auto sges = word(m.pi, 2) >> 24;
        assert(sges >= 1 && sges <= 13);
        const auto& entity = q % 2 ? control_entities[q] : data_entities[q];
        assert((word(m.pi, 2) & 0xffffffU) == (entity.tp_id & 0xffffffU));
        assert(word(m.pi, 3) == (entity.remote_token_id & 0xfffffU));
        assert(word(m.pi, 8) == entity.remote_token_value && word(m.pi, 9) == 0);
        for (uint32_t i = 0; i < 4; ++i) {
          uint32_t eid_word;
          std::memcpy(&eid_word, entity.remote_eid + i * 4, 4);
          assert(word(m.pi, 4 + i) == eid_word);
        }
        const uint32_t blocks = (48 + 16 * sges + 63) / 64;
        assert(m.pi + blocks - m.completed <= depth);
        assert(m.posted.size() < depth / 4);
        ModelQueue::PostedWqe snapshot{m.pi, blocks};
        for (uint32_t i = 0; i < blocks * 16; ++i) snapshot.words[i] = word(m.pi, i);
        m.posted.push_back(snapshot);
        if (args.profile) {
          // Observe the production record while nested Gather/Queue calls are
          // still active. Early stamps must already live in this core's GM
          // record, not in a local aggregate copied back only at kernel exit.
          const auto& p = profiles[worker];
          const uint64_t reached = op == kWrite ? 0x7 : args.pipeline_chunk_bytes ?
              (p.read_done_cycle ? 0x2f : 0xf) : !(q % 2) ? 0x1f : 0xf;
          assert(p.magic == 0 && p.sequence == args.sequence);
          assert(p.valid_mask == reached);
          assert(p.kernel_start_cycle && p.input_done_cycle && p.descriptor_done_cycle);
          assert(p.kernel_start_cycle <= p.input_done_cycle && p.input_done_cycle <= p.descriptor_done_cycle);
          if (reached >= 0xf) assert(p.notify_done_cycle >= p.descriptor_done_cycle);
          if (reached == 0x1f) assert(p.ack_done_cycle >= p.notify_done_cycle);
        }
        uint64_t remote_address = uint64_t(word(m.pi, 10)) | (uint64_t(word(m.pi, 11)) << 32);
        if (q % 2 && op == kWrite) writes.push_back(remote_address);
        if (q % 2 && op == kWrite &&
            remote_address == args.remote_descriptor + worker * args.lane_descriptor_bytes)
          CheckDescriptor(worker);
        if (q % 2 && op == kRead && args.pipeline_shards) {
          const auto* h = reinterpret_cast<const kvaf::peer::GatherDescriptorHeader*>(
              args.remote_descriptor + worker * args.lane_descriptor_bytes);
          if (remote_address == Address(h) + kvaf::GatherProgressOffset(h->span_count))
            AdvanceShards(worker);
        }
        if (q % 2 && op == kRead && args.pipeline_chunk_bytes && !args.pipeline_shards &&
            remote_address == args.remote_descriptor + worker * args.lane_descriptor_bytes + 8) {
          assert(sges == 1 && word(m.pi, 12) == 40);
          AdvanceGather(worker);
        }
        if (q % 2 && op == kRead && remote_address == args.remote_control + worker * 64 + 8) {
          ++ack_reads[worker];
          if (!queues[worker * 2]->posted.empty()) ack_with_data_inflight = true;
          const bool complete = args.pipeline_shards ?
              ShardReadyBytes(worker) == uint64_t(uint64_t(active) * (worker + 1) / workers -
                  uint64_t(active) * worker / workers) * token && ack_reads[worker] >= 3 : ack_reads[worker] == 3;
          if (!timeout && !ack_timeout && complete) {
            auto* c = reinterpret_cast<kvaf::peer::GatherControl*>(args.remote_control) + worker;
            assert(c->notify == args.sequence);
            const auto* h = reinterpret_cast<const kvaf::peer::GatherDescriptorHeader*>(c->command);
            assert(h->pipeline_chunk_bytes == args.pipeline_chunk_bytes && h->sequence == args.sequence);
            if (!args.pipeline_chunk_bytes) {
              kvaf::peer::CpuGatherExecutor cpu(remote.data(), Address(remote.data()), 2);
              assert(cpu.PrepareDirect(remote.size(), c->command, args.sequence, count));
              cpu.Copy();
              read_vf_begin[worker] = aiv_model::vf_calls;
              read_overlap_begin[worker] = aiv_model::vf_with_pending_mte3;
            } else {
              assert(args.pipeline_shards || h->ready_bytes == h->total_bytes);
              assert(!queues[worker * 2]->posted.empty());
              auto& data = *queues[worker * 2];
              if (data.deferred_cq && data.release_vf == UINT64_MAX) {
                data.cq[data.cq_pi % 2][0] = data.deferred_word;
                ++data.cq_pi; ++completions;
                data.deferred_cq = false;
              }
            }
            c->ack = args.sequence | ((remote_error || ack_error) ? kErrorBit : 0);
            cpu_done[worker] = true;
          }
        }
        if (!(q % 2)) {
          assert(op == kRead);
          if (args.pipeline_chunk_bytes) {
            const auto* h = reinterpret_cast<const kvaf::peer::GatherDescriptorHeader*>(
                args.remote_descriptor + worker * args.lane_descriptor_bytes);
            uint64_t bytes = 0;
            for (uint32_t i = 0; i < sges; ++i) bytes += word(m.pi, 12 + i * 4);
            assert(remote_address >= h->staging_address);
            if (args.pipeline_shards) {
              const auto total = (h->span_count + args.sgl_size - 1) / args.sgl_size;
              const auto shards = kvaf::GatherShardCount(total);
              const auto wr = (remote_address - h->staging_address) / (args.sgl_size * token);
              if (wr + 1 == total && m.pi + blocks < end) tail_in_middle = true;
              const auto* progress = reinterpret_cast<const uint64_t*>(Address(h) + kvaf::GatherProgressOffset(h->span_count));
              uint32_t shard = 0;
              while (kvaf::GatherShardBoundary(total, shards, shard + 1) <= wr) ++shard;
              const auto first = kvaf::GatherShardBoundary(total, shards, shard);
              const auto row_begin = uint64_t(active) * worker / workers + (remote_address - h->staging_address) / token;
              assert(bytes % token == 0);
              for (uint64_t row = row_begin; row < row_begin + bytes / token; ++row) {
                assert(row < uint64_t(active) && !shard_read_rows[row]);
                shard_read_rows[row] = 1;
              }
              assert((progress[shard] >> 32) == args.sequence);
              assert(remote_address + bytes <= h->staging_address +
                  std::min<uint64_t>(h->total_bytes, uint64_t(first + uint32_t(progress[shard])) * args.sgl_size * token));
              if (!first_read_ready[worker]) {
                first_read_ready[worker] = ShardReadyBytes(worker);
                if (shards > 1 && !all_shards_ready) assert(wr > 0); // Late shard 0 cannot block the first READ.
              }
            } else {
              assert(remote_address + bytes <= h->staging_address + h->ready_bytes);
              if (!first_read_ready[worker]) first_read_ready[worker] = h->ready_bytes;
              assert(!cpu_done[worker] && !ack_reads[worker]);
            }
          } else {
            assert(cpu_done[worker] && ack_reads[worker] >= 3);
          }
          if (args.coalesce_sges) assert(sges == 1);
          ++data_reads; max_sges = std::max(max_sges, sges);
        }
        for (uint32_t i = 0; i < sges; ++i) {
          const auto bytes = word(m.pi, 12 + i * 4);
          assert(word(m.pi, 13 + i * 4) == entity.local_token_id);
          const auto local = uint64_t(word(m.pi, 14 + i * 4)) | (uint64_t(word(m.pi, 15 + i * 4)) << 32);
          if (op == kWrite) std::memcpy(reinterpret_cast<void*>(remote_address), reinterpret_cast<void*>(local), bytes);
          else std::memcpy(reinterpret_cast<void*>(local), reinterpret_cast<void*>(remote_address), bytes);
          remote_address += bytes;
        }
        if (q % 2 && op == kWrite) {
          auto* c = reinterpret_cast<kvaf::peer::GatherControl*>(args.remote_control) + worker;
          if (c->notify == args.sequence && notified[worker] != args.sequence) {
            notified[worker] = args.sequence;
            c->ack = prepare_error ? args.sequence | kErrorBit : ~args.sequence;
            // Even a tempting ready prefix must not bypass the final ACK.
            if (!args.pipeline_shards) reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(c->command)->ready_bytes = 0;
            read_vf_begin[worker] = aiv_model::vf_calls;
            read_overlap_begin[worker] = aiv_model::vf_with_pending_mte3;
          }
        }
        if (header & (1U << 21)) {
          assert(((header >> 16) & 3) == 2);
          assert(m.pi + blocks == end);
          if (!(data_timeout && !(q % 2)) && m.cq_pi != m.drop_cq_at) {
            const auto completion_word = ((1U - ((m.cq_pi / 2) & 1)) << 2) |
                ((cq_error || m.cq_pi == m.fail_cq_at || (data_cq_error && !(q % 2))) ? 1U << 24 : 0);
            const auto* descriptor = reinterpret_cast<const kvaf::peer::GatherDescriptorHeader*>(
                args.remote_descriptor + worker * args.lane_descriptor_bytes);
            const auto progress_end = args.pipeline_shards ? Address(descriptor) +
                kvaf::GatherProgressOffset(descriptor->span_count) + kvaf::kGatherProgressBytes : Address(descriptor) + 48;
            const bool defer_progress = delay_progress_cq && q % 2 && op == kRead &&
                (args.pipeline_shards == 2 ? packed_query_had_backlog[worker] :
                  args.pipeline_shards || first_read_ready[worker]) && remote_address == progress_end;
            if ((hold_data_cq_for_ack && !(q % 2) && !cpu_done[worker]) || defer_progress) {
              assert(!m.deferred_cq);
              m.deferred_cq = true;
              m.deferred_word = completion_word;
              m.release_vf = defer_progress ? aiv_model::vf_calls + 1 : UINT64_MAX;
            } else {
              m.cq[m.cq_pi % 2][0] = completion_word;
              ++m.cq_pi; ++completions;
            }
          }
        } else {
          assert(((header >> 16) & 3) == 1);
        }
        m.pi += blocks;
      }
      assert(m.pi == end); m.db = value;
      if (!writes.empty()) m.write_batches.push_back(std::move(writes));
      return;
    }
    assert(false && "unknown doorbell");
  }
  void Run(bool success = true, bool allow_partial = false) {
    const auto cycles = aiv_model::cycle;
    const auto reads = aiv_model::mte2_copies, writes = aiv_model::mte3_copies;
    const auto old_row_cache_lines = row_cache_lines;
    if (args.pipeline_shards) shard_read_rows.assign(count, 0);
    if (args.mode == KVAF_SEQ_GATHER)
      std::fill(workspace.begin() + args.descriptor_offset, workspace.end(), 0xa7);
    expected = output;
    if (success && active >= 0) {
      for (uint32_t core = 0; core < workers; ++core) {
        const uint32_t first = uint64_t(active) * core / workers;
        uint32_t last = uint64_t(active) * (core + 1) / workers;
        if (args.gather_probe == KVAF_GATHER_PROBE_FIRST_READ)
          last = std::min(last, first + 128 * args.sgl_size);
        else if (args.gather_probe) last = first;
        for (uint32_t i = first; i < last; ++i)
          std::memcpy(expected.data() + uint64_t(rows[i]) * token,
                      reinterpret_cast<void*>(addresses[indices[i]] + offsets[i]), token);
      }
    }
    for (uint32_t core = 0; core < workers; ++core) {
      aiv_model::core = core;
      launch_vf_begin = aiv_model::vf_calls;
      Launch(args);
      assert(status[core].magic == KVAF_SEQ_STATUS_MAGIC);
      assert(status[core].sequence == args.sequence);
      if (success) {
        if (status[core].result != (args.gather_probe ? KVAF_SEQ_PROBE_COMPLETE : KVAF_SEQ_OK))
          std::cerr << "kernel failure mode=" << args.mode << " shard=" << args.pipeline_shards
              << " requests=" << active << " token=" << token << " sgl=" << args.sgl_size
              << " core=" << core << " result=" << status[core].result
              << " ready_queries=" << ready_reads[core] << " ack_queries=" << ack_reads[core]
              << " delayed_progress=" << delay_progress_cq << '\n';
        assert(status[core].result == (args.gather_probe ? KVAF_SEQ_PROBE_COMPLETE : KVAF_SEQ_OK));
        if (args.gather_probe)
          assert(status[core].phase == KVAF_GATHER_AFTER_ACK + args.gather_probe - 1);
      } else assert(status[core].result != KVAF_SEQ_OK);
      assert(aiv_model::raw_copies.empty());
      for (const auto& source : model_events)
        for (const auto& direction : source)
          for (bool pending : direction) assert(!pending);
    }
    if (args.mode == KVAF_SEQ_GATHER) {
      // Gather DMAs WQE banks through the raw MTE3 primitive, never payload
      // through the public GM/UB copy helpers used by scatter.
      assert(aiv_model::mte2_copies == reads && aiv_model::mte3_copies == writes);
      assert(!push_requests);
      assert(!descriptor_cache_lines && !valid_table_cache_lines);
      assert(!simt_input_cache_lines);
      if (uint64_t(token) * args.sgl_size <= 65536)
        assert(row_cache_lines == old_row_cache_lines);
    }
    if (success) {
      assert(output == expected);
      if (args.pipeline_shards)
        for (uint32_t i = 0; i < count; ++i) assert(shard_read_rows[i] == uint8_t(i < uint32_t(active)));
    }
    else if (!allow_partial) CheckGuard();
    if (!args.profile) assert(aiv_model::cycle == cycles);
  }
  void EnableProfile() { args.profile = Address(profiles.data()); }
  void CheckProfiles() {
    for (uint32_t core = 0; core < workers; ++core) {
      const auto& p = profiles[core];
      assert(p.magic == KVAF_GATHER_PROFILE_MAGIC && p.sequence == args.sequence);
      assert(p.result == KVAF_SEQ_OK && p.valid_mask == KVAF_GATHER_PROFILE_ALL_EVENTS);
      const uint64_t boundaries[] = {p.kernel_start_cycle, p.input_done_cycle,
          p.descriptor_done_cycle, p.notify_done_cycle, p.pipeline_chunk_bytes ? p.notify_done_cycle : p.ack_done_cycle,
          p.read_done_cycle, p.kernel_finish_cycle};
      assert(boundaries[0] && std::is_sorted(std::begin(boundaries), std::end(boundaries)));
      const auto rows = uint64_t(active) * (core + 1) / workers - uint64_t(active) * core / workers;
      assert(p.requests == rows && p.data_wrs == status[core].data_wrs);
      assert(p.packed_posts == uint64_t(rows && args.pipeline_shards == 2));
      assert(p.progress_mode == (rows ? args.progress_mode : 0));
      if (rows && args.pipeline_shards == 2) assert(p.read_vf_calls == p.data_doorbells);
      if (!rows) {
        assert(p.input_done_cycle == p.read_done_cycle);
        assert(!p.data_wrs && !p.ack_queries && !p.data_wqebbs && !p.data_doorbells);
        assert(!p.data_cq_wait_cycles && !p.read_build_cycles);
      } else {
        assert((args.pipeline_shards ? p.ack_queries >= 3 : p.ack_queries == 3) && p.data_wqebbs >= p.data_wrs);
        assert(p.data_doorbells && p.data_cq_wait_cycles && p.read_build_cycles);
        if (args.pipeline_chunk_bytes) {
          assert(p.pipeline_chunk_bytes == args.pipeline_chunk_bytes);
          assert(p.ready_queries == (args.progress_mode == kvaf::kGatherProgressPush ? 0 : ready_reads[core]) && p.ready_wait_cycles);
          assert(p.notify_done_cycle < p.first_ready_cycle && p.first_ready_cycle < p.first_read_submit_cycle);
          assert(p.first_read_ready_bytes == first_read_ready[core]);
          assert(p.first_read_submit_cycle < p.ack_done_cycle && p.ack_done_cycle < p.read_done_cycle);
          assert(p.ack_done_cycle < p.kernel_finish_cycle);
        } else {
          assert(p.data_cq_wait_cycles + p.read_build_cycles <= p.read_done_cycle - p.ack_done_cycle);
        }
      }
    }
  }
  void Next(uint32_t n) {
    ++args.sequence; active = n;
    std::fill(output.begin(), output.end(), 0xad);
    std::fill(cpu_done.begin(), cpu_done.end(), false);
    std::fill(ack_reads.begin(), ack_reads.end(), 0);
    std::fill(ready_reads.begin(), ready_reads.end(), 0);
    std::fill(first_read_ready.begin(), first_read_ready.end(), 0);
    push_done = false; push_polls = 0;
    // Keep old notify/done/ACK and persistent PI/CI to exercise sequence isolation.
    for (uint64_t i = 0; i < args.source_bytes; ++i) remote[i] ^= 0x35;
    std::reverse(indices.begin(), indices.end());
  }
};
}
namespace aiv_model {
void Poll(void* p) { fixture->Poll(p); }
void Doorbell(uint32_t value, uint32_t* p) { fixture->Doorbell(value, p); }
}
int main() {
  const auto& metadata = KvafGatherBatchGetAivKernelV7_kernel_type_section.value;
  assert(metadata.type == F_TYPE_KTYPE && metadata.bytes == sizeof(unsigned int) && metadata.kind == K_TYPE_AIV);
  // GDR3 consumers must read later ready shards before delayed shard zero,
  // retain in-flight READs across shards, and finish ACK before data-CQ drain.
  for (uint32_t cores : {1U, 3U, 8U})
    for (uint32_t sgl : {1U, 6U, 13U})
      for (uint32_t chunk : {64U, 4096U, 131072U}) {
        Fixture f(8193, 656, cores, KVAF_SEQ_GATHER, sgl);
        f.args.pipeline_chunk_bytes = chunk; f.args.pipeline_shards = 2;
        f.args.max_polls = 512;
        f.DataQueueDepth(32768); f.EnableProfile();
        f.Run(); f.CheckProfiles();
        assert(f.queried_with_data_inflight && f.ack_with_data_inflight);
        for (uint32_t core = 0; core < cores; ++core) {
          const auto rows = 8193ULL * (core + 1) / cores - 8193ULL * core / cores;
          assert(f.queues[core * 2]->cq_completed == ((rows + sgl - 1) / sgl + 8191) / 8192);
        }
        for (uint32_t count : {1U, 0U, 8193U}) { f.Next(count); f.Run(); f.CheckProfiles(); }
      }
  {
    Fixture f(8193, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 131072; f.args.pipeline_shards = 2;
    f.args.max_polls = 512; f.DataQueueDepth(32768);
    f.ready_idle = 0;
    f.delay_progress_cq = true; f.hold_data_cq_for_ack = true;
    f.EnableProfile(); f.Run(); f.CheckProfiles();
    assert(f.prefetched_during_vf && f.ack_with_data_inflight);
    assert(aiv_model::vf_with_pending_mte3 > f.read_overlap_begin[0]);
  }
  {
    Fixture f(8192, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 131072; f.args.pipeline_shards = 2;
    f.all_shards_ready = true; f.ready_idle = 0;
    f.DataQueueDepth(32768); f.EnableProfile(); f.Run(); f.CheckProfiles();
    assert(f.profiles[0].data_doorbells == (((8192 + 5) / 6) + 127) / 128);
    assert(f.queues[0]->cq_completed == 1);
  }
  for (uint32_t depth : {4U, 16U, 520U}) {
    Fixture f(1024, 13, 3, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 4096; f.args.pipeline_shards = 2;
    f.args.max_polls = 512; f.DataQueueDepth(depth); f.Run(); f.Next(19); f.Run();
  }
  for (uint32_t coalesce : {0U, 1U}) {
    Fixture f(8193, 4096, 2, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 131072; f.args.pipeline_shards = 2;
    f.args.coalesce_sges = coalesce; f.args.max_polls = 512;
    if (coalesce) f.ConsecutiveReadRows(true);
    f.DataQueueDepth(32768); f.EnableProfile(); f.Run(); f.CheckProfiles();
  }
  for (uint32_t fault : {1U, 2U, 3U, 4U, 5U}) {
    Fixture f(8193, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 131072; f.args.pipeline_shards = 2;
    f.ready_fault = fault; f.args.max_polls = 128; f.DataQueueDepth(32768);
    f.Run(false, true);
    assert(f.status[0].result == (fault == 1 ? KVAF_SEQ_TIMEOUT : KVAF_SEQ_REMOTE_ERROR));
    assert(f.states[0].poisoned && f.states[1].poisoned);
    f.ready_fault = 0; f.Next(8193); f.Run(false, true);
    assert(f.status[0].result == KVAF_SEQ_QUEUE_ERROR);
  }
  for (uint32_t failure = 0; failure < 6; ++failure) {
    Fixture f(1024, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 4096; f.args.pipeline_shards = 2;
    f.args.max_polls = 128; f.DataQueueDepth(32768);
    f.timeout = failure == 0; f.ack_timeout = failure == 1;
    f.ack_error = failure == 2; f.data_timeout = failure == 3; f.data_cq_error = failure == 4;
    if (failure == 5) f.queues[1]->fail_cq_at = 4;
    f.Run(false, true);
    assert(f.status[0].result == (failure == 2 ? KVAF_SEQ_REMOTE_ERROR :
        failure >= 4 ? KVAF_SEQ_CQ_ERROR : KVAF_SEQ_TIMEOUT));
    assert(f.states[0].poisoned && f.states[1].poisoned);
  }
  for (uint32_t scheduler : {1U, 2U}) {
    Fixture f(4096, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 131072; f.args.pipeline_shards = scheduler;
    f.args.max_polls = 512; f.ready_idle = 0; f.DataQueueDepth(32768); f.EnableProfile();
    f.Run(); f.CheckProfiles();
    if (scheduler == 2) {
      assert(f.profiles[0].read_vf_calls < 64);
      assert(f.profiles[0].data_doorbells < 64);
      assert(f.queues[0]->first_db_vfs == 1);
    } else assert(f.profiles[0].read_vf_calls == 127 && f.profiles[0].data_doorbells == 64);
  }
  {
    Fixture f(4096, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 131072; f.args.pipeline_shards = 2;
    f.odd_ready_wave = true; f.ready_idle = 0; f.DataQueueDepth(32768); f.EnableProfile();
    f.Run(); f.CheckProfiles();
    assert(f.profiles[0].multi_run_posts && f.profiles[0].ready_query_skips);
    assert(f.profiles[0].data_doorbells == 6 && f.profiles[0].read_vf_calls == 6);
    assert(f.profiles[0].ready_queries == 2 && f.tail_in_middle);
  }
  for (uint32_t mode : {kvaf::kGatherProgressPull, kvaf::kGatherProgressPush}) {
    for (uint32_t token : {656U, 4096U}) {
      Fixture f(4097, token, 3, KVAF_SEQ_GATHER, 6);
      f.args.pipeline_chunk_bytes = 65536; f.args.pipeline_shards = 2; f.args.progress_mode = mode;
      f.args.max_polls = 512; f.DataQueueDepth(32768); f.EnableProfile();
      f.Run(); f.CheckProfiles();
      for (const auto& p : f.profiles) {
        if (mode == kvaf::kGatherProgressPull) {
          assert(p.ready_control_wrs == p.ready_queries + p.ready_ack_checks);
          assert(p.ack_queries < p.ready_queries && !p.ready_local_polls);
        } else assert(!p.ready_queries && p.ready_local_polls && p.ready_local_updates);
      }
      for (uint32_t n : {19U, 0U, 4097U}) { f.Next(n); f.Run(); f.CheckProfiles(); }
    }
    {
      Fixture f(4097, 656, 1, KVAF_SEQ_GATHER, 6);
      f.args.pipeline_chunk_bytes = 65536; f.args.pipeline_shards = 2; f.args.progress_mode = mode;
      f.args.max_polls = 512; f.prepare_error = true; f.EnableProfile();
      f.Run(false);
      assert(f.status[0].result == KVAF_SEQ_REMOTE_ERROR && !f.data_reads);
      assert(f.profiles[0].ready_ack_checks && f.profiles[0].ack_queries);
    }
    for (uint32_t failure = 0; failure < 4; ++failure) {
      Fixture f(4097, 656, 1, KVAF_SEQ_GATHER, 6);
      f.args.pipeline_chunk_bytes = 65536; f.args.pipeline_shards = 2; f.args.progress_mode = mode;
      f.args.max_polls = 512; f.DataQueueDepth(32768);
      f.ack_timeout = failure == 0; f.ack_error = failure == 1;
      f.data_timeout = failure == 2; f.data_cq_error = failure == 3;
      f.Run(false, true);
      assert(f.status[0].result == (failure == 1 ? KVAF_SEQ_REMOTE_ERROR :
          failure == 3 ? KVAF_SEQ_CQ_ERROR : KVAF_SEQ_TIMEOUT));
      assert(f.states[0].poisoned && f.states[1].poisoned);
    }
    for (uint32_t fault : {1U, 2U, 3U, 4U, 5U}) {
      Fixture f(4097, 656, 1, KVAF_SEQ_GATHER, 6);
      f.args.pipeline_chunk_bytes = 65536; f.args.pipeline_shards = 2; f.args.progress_mode = mode;
      f.args.max_polls = 512; f.ready_fault = fault; f.DataQueueDepth(32768);
      f.Run(false, true);
      assert(f.status[0].result == (fault == 1 ? KVAF_SEQ_TIMEOUT : KVAF_SEQ_REMOTE_ERROR));
      assert(f.states[0].poisoned && f.states[1].poisoned);
    }
  }
  for (uint32_t chunk : {0U, 4096U}) {
    Fixture f(19, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = chunk;
    f.Run();
    // Three publication WRs plus one SQ clean per control query. Ready fields
    // share one scratch-line invalidate, rather than one per field.
    assert(f.queues[1]->sq_cache_lines == 3 + f.ready_reads[0] + f.ack_reads[0]);
    assert(f.scratch_cache_lines[0] == 1 + f.ready_reads[0] + f.ack_reads[0]);
  }
  {
    Fixture f(8193, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 131072; f.ready_idle = 0; f.ready_step = 4;
    f.delay_progress_cq = true; f.hold_data_cq_for_ack = true;
    f.DataQueueDepth(32768); f.EnableProfile(); f.Run(); f.CheckProfiles();
    assert(f.queues[0]->first_db_vfs == 1); // No three-VF startup bubble.
    assert(f.prefetched_during_vf && f.ack_with_data_inflight);
    assert(!f.queues[0]->deferred_cq && !f.queues[1]->deferred_cq);
    assert(f.queues[0]->cq_completed == 1); // No per-frontier data CQ drain.
    assert(aiv_model::vf_with_pending_mte3 > f.read_overlap_begin[0]);
  }
  for (uint32_t fault : {3U, 5U}) {
    Fixture f(8193, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 131072; f.ready_idle = 0; f.ready_step = 4;
    f.ready_fault = fault; f.DataQueueDepth(32768); f.EnableProfile();
    f.Run(false, true);
    assert(f.status[0].result == KVAF_SEQ_REMOTE_ERROR);
    assert(f.states[0].poisoned && f.states[1].poisoned);
    assert(f.status[0].data_wrs == 128); // Built next VF is discarded, never doorbelled.
  }
  for (bool timeout : {false, true}) {
    Fixture f(8193, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 131072; f.ready_idle = 0; f.ready_step = 16;
    f.DataQueueDepth(32768); f.EnableProfile();
    if (timeout) f.queues[1]->drop_cq_at = 3; // Two publication CQs + initial ready; lookahead is lost.
    else f.ready_fault = 5; // Error while a VF and an MTE3 post are still live.
    f.Run(false, true);
    assert(f.status[0].result == (timeout ? KVAF_SEQ_TIMEOUT : KVAF_SEQ_REMOTE_ERROR));
    assert(f.states[0].poisoned && f.states[1].poisoned && f.data_reads);
    assert(!f.queues[0]->posted.empty() && !f.queues[0]->cq_completed);
    assert(f.status[0].data_wrs < (8193 + 5) / 6);
  }
  // CPU chunks and NIC READs overlap without per-chunk data CQ waits. The
  // producer only copies the next chunk when queried: an early data doorbell
  // therefore proves that later staging bytes have not been gathered yet.
  for (uint32_t cores : {1U, 3U, 8U})
    for (uint32_t sgl : {1U, 6U, 13U})
      for (uint32_t chunk : {64U, 4096U, 131072U}) {
        Fixture f(8193, 656, cores, KVAF_SEQ_GATHER, sgl);
        f.args.pipeline_chunk_bytes = chunk;
        f.args.max_polls = 256; // 64-byte chunks can bisect a 13-SGE READ.
        f.DataQueueDepth(32768); f.EnableProfile();
        for (uint32_t count : {8193U, 1U, 0U, 8193U}) {
          f.Next(count); f.Run(); f.CheckProfiles();
          if (count == 8193) {
            assert(f.queried_with_data_inflight);
            for (uint32_t core = 0; core < cores; ++core) {
              const auto bytes = (uint64_t(count) * (core + 1) / cores - uint64_t(count) * core / cores) * f.token;
              assert(f.first_read_ready[core] && f.first_read_ready[core] < bytes);
            }
          }
        }
        for (uint32_t core = 0; core < cores; ++core) {
          const auto rows = 8193ULL * (core + 1) / cores - 8193ULL * core / cores;
          const auto wrs = (rows + sgl - 1) / sgl;
          assert(f.queues[core * 2]->cq_completed == 2 * ((wrs + 8191) / 8192) + (core + 1 == cores));
        }
      }
  // 128 KiB makes 32 x 4 KiB READs ready: publish that short post immediately.
  // Both encodings retain lane-wide SGL grouping and partial final WRs.
  for (uint32_t coalesce : {0U, 1U}) {
    Fixture f(8192, 4096, 2, KVAF_SEQ_GATHER, coalesce ? 6 : 1);
    f.args.pipeline_chunk_bytes = 131072; f.args.coalesce_sges = coalesce;
    if (coalesce) f.ConsecutiveReadRows(true);
    f.DataQueueDepth(32768); f.EnableProfile(); f.Run(); f.CheckProfiles();
    for (uint32_t core = 0; core < 2; ++core) {
      assert(f.profiles[core].first_read_ready_bytes == 131072);
      assert(f.queues[core * 2]->cq_completed == 1);
      assert(f.profiles[core].data_doorbells > 1);
    }
  }
  {
    Fixture f(1024, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 4096; f.DataQueueDepth(32768);
    f.Run(); // Profiling disabled: the production path must read no clocks.
    f.Next(1); f.Run();
  }
  // Small SQ/CQ rings and readiness jumps still respect physical wave limits.
  for (uint32_t depth : {4U, 16U, 520U}) {
    Fixture f(8193, 13, 3, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 4096; f.ready_step = 3;
    f.DataQueueDepth(depth); f.EnableProfile();
    f.Run(); f.CheckProfiles(); f.Next(8193); f.Run(); f.CheckProfiles();
  }
  for (uint32_t fault : {1U, 2U, 3U, 4U, 5U, 6U}) {
    Fixture f(1024, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 4096; f.ready_fault = fault;
    f.DataQueueDepth(32768); f.EnableProfile(); f.Run(false, true);
    assert(f.status[0].result == KVAF_SEQ_REMOTE_ERROR && f.states[0].poisoned && f.states[1].poisoned);
    if (fault == 3 || fault == 5) {
      assert(f.data_reads && !f.queues[0]->cq_completed);
      assert(!f.queues[0]->posted.empty()); // Unsignaled partial wave is retained.
    } else assert(!f.data_reads);
    f.ready_fault = 0; f.Next(1024); f.Run(false, true);
    assert(f.status[0].result == KVAF_SEQ_QUEUE_ERROR);
  }
  for (uint32_t failure : {0U, 1U, 2U, 3U, 4U, 5U}) {
    Fixture f(1024, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = 4096; f.DataQueueDepth(32768); f.EnableProfile();
    f.timeout = failure == 0; f.ack_timeout = failure == 1;
    f.ack_error = failure == 2; f.data_timeout = failure == 3; f.data_cq_error = failure == 4;
    if (failure == 5) f.queues[1]->fail_cq_at = 4; // Progress CQ error after a partial READ post.
    f.Run(false, true);
    const auto expected = failure == 2 ? KVAF_SEQ_REMOTE_ERROR : failure >= 4 ? KVAF_SEQ_CQ_ERROR : KVAF_SEQ_TIMEOUT;
    assert(f.status[0].result == expected && f.states[0].poisoned && f.states[1].poisoned);
    if (failure == 0) assert(!f.data_reads);
    else assert(f.data_reads);
  }
  for (uint32_t chunk : {1U, 63U, 65U}) {
    Fixture f(19, 656, 1, KVAF_SEQ_GATHER, 6);
    f.args.pipeline_chunk_bytes = chunk; f.Run(false);
    assert(f.status[0].result == KVAF_SEQ_INVALID && !f.data_reads);
  }
  // Descriptor/command/notify share one ordered tail CQ when capacity permits.
  // Repeated calls wrap SQ/CQ and leave stale ACK/mailbox contents.
  for (uint32_t cores : {1U, 3U, 8U}) {
    Fixture f(8193, 13, cores, KVAF_SEQ_GATHER, 6);
    for (uint32_t core = 0; core < cores; ++core) {
      // The descriptor leaves command at the last SQ slot and notify at slot 0.
      const auto rows = 8193ULL * (core + 1) / cores - 8193ULL * core / cores;
      const auto descriptor_wrs = (48 + rows * 16 + 65535) / 65536;
      auto& q = *f.queues[core * 2 + 1];
      q.sq.resize(64 * 16);
      f.control_entities[core * 2 + 1].sq_va = Address(q.sq.data());
      f.control_entities[core * 2 + 1].sq_depth = 64;
      auto& s = f.states[core * 2 + 1];
      s.magic = 0x4b564151; s.initialized = 1;
      s.sq_producer_index = q.pi = q.completed = 64 - descriptor_wrs - 1;
      q.db = uint32_t(q.pi);
    }
    f.EnableProfile();
    for (uint32_t count : {8193U, 1U, 0U, 8193U}) {
      f.Next(count);
      const auto old_flushes = f.scratch_cache_lines;
      std::vector<uint64_t> old_batches(cores), old_cqs(cores);
      for (uint32_t core = 0; core < cores; ++core) {
        old_batches[core] = f.queues[core * 2 + 1]->write_batches.size();
        old_cqs[core] = f.queues[core * 2 + 1]->cq_completed;
      }
      f.Run(); f.CheckProfiles();
      for (uint32_t core = 0; core < cores; ++core) {
        const auto rows = uint64_t(count) * (core + 1) / cores - uint64_t(count) * core / cores;
        const auto& q = *f.queues[core * 2 + 1];
        if (!rows) {
          assert(q.write_batches.size() == old_batches[core]);
          assert(q.cq_completed == old_cqs[core]);
          assert(f.scratch_cache_lines[core] == old_flushes[core]);
          continue;
        }
        assert(q.write_batches.size() == old_batches[core] + 1);
        const auto descriptor = f.args.remote_descriptor + core * f.args.lane_descriptor_bytes;
        std::vector<uint64_t> descriptor_writes;
        for (uint64_t offset = 0; offset < 48 + rows * 16; offset += 65536)
          descriptor_writes.push_back(descriptor + offset);
        const auto control = f.args.remote_control + core * 64;
        descriptor_writes.push_back(control + 16);
        descriptor_writes.push_back(control);
        assert(q.write_batches.back() == descriptor_writes);
        assert(q.cq_completed - old_cqs[core] == 1 + f.ack_reads[core]);
        // One clean for both source words; ACK READs still invalidate after CQ.
        assert(f.scratch_cache_lines[core] - old_flushes[core] == 1 + f.ack_reads[core]);
      }
    }
  }
  // Combined publication still drains at the transport's physical WR limit.
  // No notify may reach the CPU before all descriptor bytes and command.
  for (uint32_t depth : {4U, 8U, 16U, 64U}) {
    Fixture f(8193, 13, 1, KVAF_SEQ_GATHER, 1);
    auto& q = *f.queues[1];
    q.sq.resize(depth * 16);
    f.control_entities[1].sq_va = Address(q.sq.data());
    f.control_entities[1].sq_depth = depth;
    f.EnableProfile();
    for (uint32_t rows : {8193U, 1U, 8193U}) {
      f.Next(rows);
      const auto batches = q.write_batches.size();
      const auto cq = q.cq_completed;
      f.Run(); f.CheckProfiles();
      const uint32_t writes = (48 + rows * 16 + 65535) / 65536 + 2;
      const uint32_t waves = (writes + depth / 4 - 1) / (depth / 4);
      assert(q.write_batches.size() - batches == waves);
      assert(q.cq_completed - cq == waves + f.ack_reads[0]);
    }
  }
  {
    Fixture f(19, 656, 1, KVAF_SEQ_GATHER, 6);
    f.EnableProfile();
    f.queues[1]->fail_cq_at = 0;  // Combined descriptor/notify tail CQ fails.
    f.Run(false);
    assert(f.status[0].result == KVAF_SEQ_CQ_ERROR && f.status[0].phase == KVAF_SEQ_NOTIFY);
    assert(f.states[1].poisoned && !f.ack_reads[0] && !f.data_reads);
    assert(f.profiles[0].valid_mask == 0x47 && f.queues[1]->write_batches.size() == 1);
    f.queues[1]->fail_cq_at = UINT64_MAX;
    f.Next(19); f.Run(false);  // A failed control queue cannot be reused.
    assert(f.status[0].result == KVAF_SEQ_QUEUE_ERROR);
  }
  // Header/span/tail contents are checked before the descriptor's first NIC
  // WRITE. Stress 128-thread tails, both 24-KiB banks, lane boundaries, and a
  // smaller launch after a larger one without relying on zeroed HBM/UB.
  for (uint32_t cores : {1U, 3U, 8U}) {
    Fixture f(65536, 13, cores, KVAF_SEQ_GATHER, 6);
    f.args.gather_probe = KVAF_GATHER_PROBE_ACK; f.EnableProfile();
    for (uint32_t count : {0U, 1U, 127U, 128U, 129U, 1532U * cores,
                           1533U * cores, 1533U * cores + 1U, 3069U * cores,
                           3070U * cores, 65536U, 1U, 65536U}) {
      f.Next(count);
      f.args.gather_probe = count ? KVAF_GATHER_PROBE_ACK : KVAF_GATHER_PROBE_NONE;
      const auto overlap = aiv_model::vf_with_pending_mte3;
      f.Run();
      for (uint32_t core = 0; core < cores; ++core) {
        const auto rows = uint64_t(count) * (core + 1) / cores - uint64_t(count) * core / cores;
        assert(f.profiles[core].valid_mask == (rows ? 0x5f : 0x7f));
        if (!rows) {
          const auto* lane = f.workspace.data() + f.args.descriptor_offset + core * f.args.lane_descriptor_bytes;
          for (uint64_t i = 0; i < f.args.lane_descriptor_bytes; ++i) assert(lane[i] == 0xa7);
        }
      }
      if (cores == 1 && count > 1533) assert(aiv_model::vf_with_pending_mte3 > overlap);
      assert(!f.data_reads);
    }
  }
  for (uint32_t probe = KVAF_GATHER_PROBE_ACK; probe <= KVAF_GATHER_PROBE_FIRST_READ; ++probe)
    for (uint32_t cores : {1U, 3U, 8U})
      for (uint32_t sgl : {1U, 6U, 13U}) {
        Fixture f(8193, 656, cores, KVAF_SEQ_GATHER, sgl);
        f.DataQueueDepth(4096); f.args.gather_probe = probe;
        for (uint32_t count : {8193U, 1U, 8193U}) {
          f.Next(count);
          const auto config_writes = aiv_model::ub_to_out_config_writes;
          std::vector<uint64_t> before(cores);
          for (uint32_t c = 0; c < cores; ++c) before[c] = f.queues[c * 2]->doorbells;
          f.Run();
          // All nonempty lanes now configure descriptor MTE3. ACK/WQE probes
          // must still avoid configuring the subsequent data-SQ MTE3 path.
          const auto descriptor_config = 4 * std::min(count, cores);
          if (probe <= KVAF_GATHER_PROBE_WQE)
            assert(aiv_model::ub_to_out_config_writes == config_writes + descriptor_config);
          else assert(aiv_model::ub_to_out_config_writes > config_writes + descriptor_config);
          for (uint32_t c = 0; c < cores; ++c) {
            const auto rows = uint64_t(count) * (c + 1) / cores - uint64_t(count) * c / cores;
            const auto added = f.queues[c * 2]->doorbells - before[c];
            assert(added == uint64_t(probe == KVAF_GATHER_PROBE_FIRST_READ && rows));
            assert(f.status[c].data_wrs == (added ? std::min<uint64_t>(128, (rows + sgl - 1) / sgl) : 0));
          }
        }
      }
  {
    Fixture f(8193, 656, 1, KVAF_SEQ_GATHER, 6);
    f.DataQueueDepth(4096); f.args.gather_probe = KVAF_GATHER_PROBE_FIRST_READ;
    f.data_cq_error = true; f.Run(false, true);
    assert(f.status[0].result == KVAF_SEQ_CQ_ERROR && f.states[0].poisoned);
  }
  // Real queue depth: early doorbells do not introduce per-post CQ waits.
  // VF(next bank) executes while the deferred MTE3(previous bank) is pending.
  for (uint32_t sgl : {6U, 13U}) {
    Fixture f(8193, 656, 1, KVAF_SEQ_GATHER, sgl);
    f.DataQueueDepth(32768); f.EnableProfile();
    f.Run(); f.CheckProfiles();
    const auto& q = *f.queues[0];
    const uint64_t doorbells = sgl == 6 ? 11 : 5;
    assert(q.doorbells == doorbells && q.cq_completed == 1 && q.first_db_vfs == 3);
    assert(f.profiles[0].data_wrs == (sgl == 6 ? 1366 : 631));
    assert(f.profiles[0].data_wqebbs == (sgl == 6 ? 4097 : 2522));
    assert(f.profiles[0].data_doorbells == doorbells);
    assert(aiv_model::vf_with_pending_mte3 > f.read_overlap_begin[0]);
    f.Next(8193); f.Run(); f.CheckProfiles();
  }
  // Bank/tail boundaries for every SGL, including a partially populated last
  // WR and an SQ ring wrap inside one multi-SGE WQE/MTE3 post.
  for (uint32_t sgl = 1; sgl <= 13; ++sgl)
    for (uint32_t coalesce : {0U, 1U}) {
      Fixture f(257 * sgl, 13, 1, KVAF_SEQ_GATHER, sgl);
      f.args.coalesce_sges = coalesce;
      f.DataQueueDepth(4096); f.EnableProfile();
      auto& q = *f.queues[0];
      auto& s = f.states[0];
      s.magic = 0x4b564151; s.initialized = 1;
      s.sq_producer_index = q.pi = q.completed = (1ULL << 32) - 1;
      q.db = uint32_t(q.pi);
      s.cq_consumer_index = q.cq_pi = q.cq_completed = (1ULL << 22) - 1;
      for (auto& entry : q.cq) entry[0] = 0; // expected owner is 1 at this CI
      for (uint32_t wrs : {1U, 127U, 128U, 129U, 255U, 256U, 257U}) {
        f.Next(wrs * sgl - (sgl - 1));
        if (coalesce) f.ConsecutiveReadRows(false);
        const auto cq = q.cq_completed, db = q.doorbells;
        f.Run(); f.CheckProfiles();
        assert(q.cq_completed - cq == 1);
        assert(q.doorbells - db == (wrs + 127) / 128);
        if (coalesce) assert(f.profiles[0].data_wqebbs == wrs);
      }
      for (uint32_t tail = 1; tail <= sgl; ++tail) {
        f.Next(128 * sgl + tail);
        if (coalesce) f.ConsecutiveReadRows(false);
        const auto cq = q.cq_completed, db = q.doorbells;
        f.Run(); f.CheckProfiles();
        assert(q.cq_completed - cq == 1 && q.doorbells - db == 2);
        if (coalesce) assert(f.profiles[0].data_wqebbs == 129);
      }
    }
  // Same logical 24-KiB READs in both layouts and encodings, including 16-KiB
  // tails. Coalescing changes SQ occupancy, not WR/doorbell/completion counts.
  for (bool segmented : {false, true})
    for (uint32_t coalesce : {0U, 1U}) {
      Fixture f(8192, 4096, 2, KVAF_SEQ_GATHER, 6);
      f.args.coalesce_sges = coalesce;
      f.ConsecutiveReadRows(segmented);
      f.DataQueueDepth(32768); f.EnableProfile(); f.Run(); f.CheckProfiles();
      assert(f.max_sges == (coalesce ? 1U : 6U));
      for (uint32_t core = 0; core < 2; ++core) {
        const auto& p = f.profiles[core];
        assert(p.data_wrs == 683 && p.data_doorbells == 6);
        assert(p.data_wqebbs == (coalesce ? 683 : 2048));
        assert(f.queues[core * 2]->cq_completed == 1);
      }
    }
  {
    Fixture f(19, 10922, 3, KVAF_SEQ_GATHER, 6);
    f.args.coalesce_sges = 1; f.ConsecutiveReadRows(true); f.Run();
    Fixture exact(19, 16384, 2, KVAF_SEQ_GATHER, 4);
    exact.args.coalesce_sges = 1; exact.ConsecutiveReadRows(false); exact.Run();
    Fixture too_large(19, 10923, 3, KVAF_SEQ_GATHER, 6);
    too_large.args.coalesce_sges = 1; too_large.Run(false);
    assert(!too_large.data_reads && !too_large.ack_reads[0]);
    Fixture invalid(1, 656, 1, KVAF_SEQ_GATHER, 6);
    invalid.args.coalesce_sges = 2; invalid.Run(false);
    assert(!invalid.data_reads);
  }
  // Physical WR capacity drains and CQ owner wrap, rather than a 128-WR
  // outstanding budget. Multi-core lanes keep independent banks and queues.
  for (uint32_t sgl : {1U, 6U, 13U}) {
    Fixture f(65536, 13, 1, KVAF_SEQ_GATHER, sgl);
    f.DataQueueDepth(32768); f.EnableProfile();
    f.Run(); f.CheckProfiles();
    const uint32_t wrs = (65536 + sgl - 1) / sgl;
    assert(f.queues[0]->cq_completed == (wrs + 8191) / 8192);
    assert(f.queues[0]->doorbells == (wrs + 127) / 128);
  }
  for (uint32_t cores : {3U, 8U})
    for (uint32_t sgl : {6U, 13U}) {
      Fixture f(8193, 656, cores, KVAF_SEQ_GATHER, sgl);
      f.DataQueueDepth(4096); f.Run(); f.Next(1024); f.Run();
    }
  const std::pair<uint32_t, uint32_t> read_shapes[] = {
      {10922, 6}, {10923, 6}, {65536, 1}, {65537, 1},
      {5041, 13}, {5042, 13}, {65535, 13}};
  for (const auto& [token, sgl] : read_shapes) {
    Fixture f(19, token, 1, KVAF_SEQ_GATHER, sgl);
    f.DataQueueDepth(520); f.EnableProfile();
    f.Run(); f.CheckProfiles();
    assert((aiv_model::vf_calls > f.read_vf_begin[0]) == (uint64_t(token) * sgl <= 65536));
    f.Next(7); f.Run(); f.CheckProfiles();
  }
  {
    Fixture f(257, 12000, 3, KVAF_SEQ_GATHER, 6);
    f.EnableProfile(); f.Next(7);
    const auto before = f.row_cache_lines;
    f.Run(); f.CheckProfiles();
    uint64_t expected = 0;
    for (uint32_t core = 0; core < f.workers; ++core) {
      const auto first = 7 * core / f.workers, last = 7 * (core + 1) / f.workers;
      expected += (last * 4 - 1) / 64 - first * 4 / 64 + 1;
    }
    assert(f.row_cache_lines - before == expected); // Active lane, not capacity.
  }
  {
    Fixture f(1800, 13, 1, KVAF_SEQ_GATHER, 6);
    f.DataQueueDepth(520); f.EnableProfile(); f.Run(); f.CheckProfiles();
    assert(f.queues[0]->cq_completed == 3 && f.queues[0]->doorbells == 5);
  }
  {
    Fixture f(257 * 13, 13, 1, KVAF_SEQ_GATHER, 13);
    f.DataQueueDepth(520); f.EnableProfile();
    for (uint32_t count : {257U * 13, 256U * 13 + 1, 257U * 13}) {
      f.Next(count);
      const auto cq = f.queues[0]->cq_completed, db = f.queues[0]->doorbells;
      f.Run(); f.CheckProfiles();
      assert(f.queues[0]->cq_completed - cq == 2 && f.queues[0]->doorbells - db == 3);
    }
  }
  for (uint32_t sgl : {0U, 14U}) {
    Fixture f(19, 656, 1, KVAF_SEQ_GATHER, sgl);
    f.EnableProfile(); f.Run(false);
    assert(f.status[0].result == KVAF_SEQ_INVALID && f.profiles[0].valid_mask == 0x41);
    assert(!f.data_reads && !f.states[0].initialized);
  }
  {
    Fixture f(8193, 656, 1, KVAF_SEQ_GATHER, 6);
    f.DataQueueDepth(32768); f.EnableProfile(); f.data_timeout = true;
    f.Run(false, true);
    assert(f.status[0].result == KVAF_SEQ_TIMEOUT && f.states[0].poisoned);
    assert(f.profiles[0].data_doorbells == 11 && f.profiles[0].valid_mask == 0x5f);
    f.Next(8193); f.data_timeout = false; f.Run(false);
  }
  // Profile every lane, including empty lanes and repeated queue wraps. The
  // disabled executions throughout this fixture assert zero timer reads.
  for (uint32_t workers : {1U, 3U, 8U})
    for (uint32_t sgl : {1U, 6U, 13U}) {
      Fixture f(127, 656, workers, KVAF_SEQ_GATHER, sgl);
      f.EnableProfile();
      for (uint32_t count : {0U, 1U, 127U, 7U, 0U, 127U}) {
        f.Next(count);
        std::vector<uint64_t> old_pi(workers), old_ci(workers);
        for (uint32_t core = 0; core < workers; ++core) {
          old_pi[core] = f.states[core * 2].sq_producer_index;
          old_ci[core] = f.states[core * 2].cq_consumer_index;
        }
        f.Run(); f.CheckProfiles();
        for (uint32_t core = 0; core < workers; ++core) {
          assert(f.profiles[core].data_wqebbs == f.states[core * 2].sq_producer_index - old_pi[core]);
          assert(f.profiles[core].data_doorbells >= f.states[core * 2].cq_consumer_index - old_ci[core]);
        }
      }
      const auto before = f.profiles;
      f.args.profile = 0; f.Next(7); f.Run();
      for (uint32_t core = 0; core < workers; ++core)
        assert(f.profiles[core].sequence == before[core].sequence);
    }
  {
    Fixture large_token(19, 65535, 3, KVAF_SEQ_GATHER, 6);
    large_token.EnableProfile(); large_token.Run(); large_token.CheckProfiles();
    // Exercise the unchecked gather path with safe addresses and rows.
    Fixture unchecked(7, 656, 1, KVAF_SEQ_GATHER, 6);
    std::fill(unchecked.valid.begin(), unchecked.valid.end(), 0);
    unchecked.EnableProfile(); unchecked.Run(); unchecked.CheckProfiles();
    Fixture invalid(7, 656, 1, KVAF_SEQ_GATHER, 6);
    invalid.EnableProfile(); invalid.active = -1; invalid.Run(false);
    assert(invalid.profiles[0].result == KVAF_SEQ_INVALID);
    assert(invalid.profiles[0].valid_mask == 0x41);
    Fixture timeout(7, 656, 1, KVAF_SEQ_GATHER, 6);
    timeout.EnableProfile(); timeout.timeout = true; timeout.Run(false);
    assert(timeout.profiles[0].result == KVAF_SEQ_TIMEOUT);
    assert(timeout.profiles[0].valid_mask == 0x4f && timeout.profiles[0].ack_queries == timeout.args.max_polls);
    Fixture cq(19, 656, 1, KVAF_SEQ_GATHER, 6);
    cq.EnableProfile(); cq.data_cq_error = true; cq.Run(false, true);
    assert(cq.profiles[0].result == KVAF_SEQ_CQ_ERROR);
    assert(cq.profiles[0].valid_mask == 0x5f && cq.profiles[0].data_doorbells);
  }
  // Match the hardware smoke test: the first launch is empty, followed by
  // nonempty launches on the same queues. Empty work must finish all worker
  // statuses without notifying the CPU or advancing any SQ/CQ state.
  for (uint32_t workers : {1U, 3U, 8U})
    for (uint32_t sgl : {1U, 6U, 13U}) {
      Fixture gather(19, 656, workers, KVAF_SEQ_GATHER, sgl);
      gather.active = 0;
      gather.Run();
      assert(!gather.completions && !gather.data_reads && !gather.push_requests);
      for (const auto& s : gather.status) {
        assert(s.phase == KVAF_SEQ_COMPLETE && s.sequence == 1);
        assert(!s.requests && !s.data_wrs && !s.sq_pi && !s.cq_ci);
      }
      for (const auto& s : gather.states)
        assert(!s.initialized && !s.sq_producer_index && !s.cq_consumer_index);
      for (const auto& q : gather.queues)
        assert(!q->pi && !q->cq_pi && !q->db && !q->cq_db);
      for (const auto notify : gather.notified) assert(!notify);
      gather.Next(1); gather.Run();
      gather.Next(19); gather.Run();
      assert(gather.completions && gather.data_reads);
    }
  const std::pair<uint32_t, uint64_t> probes[] = {
      {KVAF_SEQ_PROBE_DONE, KVAF_SEQ_AFTER_DONE}, {KVAF_SEQ_PROBE_FLUSH, KVAF_SEQ_AFTER_FLUSH},
      {KVAF_SEQ_PROBE_SETUP, KVAF_SCATTER_AFTER_SETUP}, {KVAF_SEQ_PROBE_DMA, KVAF_SCATTER_AFTER_DMA},
      {KVAF_SEQ_PROBE_LOAD, KVAF_SCATTER_AFTER_LOAD_VERIFIED}, {KVAF_SEQ_PROBE_COPY, KVAF_SEQ_AFTER_COPY}};
  for (const auto& [probe, phase] : probes)
    for (uint32_t workers : {1U, 3U, 8U})
      for (uint32_t bytes : {656U, 16385U}) {
        Fixture f(19, bytes, workers, KVAF_SEQ_SCATTER, 6);
        f.args.scatter_probe = probe;
        for (uint32_t count : {19U, 1U, 7U}) {
          f.Next(count);
          auto expected = f.output;
          if (probe == KVAF_SEQ_PROBE_COPY) for (uint32_t i = 0; i < count; ++i)
            std::memcpy(expected.data() + uint64_t(f.rows[i]) * bytes,
                        reinterpret_cast<void*>(f.addresses[f.indices[i]] + f.offsets[i]), bytes);
          const auto reads = aiv_model::mte2_copies, writes = aiv_model::mte3_copies;
          const auto aligned_reads = aiv_model::mte2_aligned_copies;
          const auto tail_reads = aiv_model::mte2_tail_copies;
          const auto scalar_reads = aiv_model::ub_scalar_reads;
          const auto resets = aiv_model::out_to_ub_resets;
          const auto setup_waits = aiv_model::event_waits[PIPE_S][PIPE_MTE2];
          const auto dma_waits = aiv_model::event_waits[PIPE_MTE2][PIPE_S];
          for (uint32_t core = 0; core < workers; ++core) {
            aiv_model::core = core;
            Launch(f.args);
            const auto& s = f.status[core];
            assert(s.magic == KVAF_SEQ_STATUS_MAGIC && s.sequence == f.args.sequence);
            assert(s.result == KVAF_SEQ_PROBE_COMPLETE);
            assert(s.phase == phase);
          }
          assert(f.push_done);  // Every probe still needs the real notify/done protocol.
          assert(f.output == expected);
          const auto chunks = count * ((bytes + kCopyBytes - 1) / kCopyBytes);
          const bool configured = probe != KVAF_SEQ_PROBE_DONE && probe != KVAF_SEQ_PROBE_FLUSH;
          const bool copied = configured && probe != KVAF_SEQ_PROBE_SETUP;
          const auto expected_aligned = copied ? count * (bytes / kCopyBytes +
              (bytes % kCopyBytes >= 32 ? 1 : 0)) : 0;
          const auto expected_tail = copied && bytes % 32 ? count : 0;
          const auto expected_reads = expected_aligned + expected_tail;
          assert(aiv_model::out_to_ub_resets - resets == (configured ? chunks : 0));
          assert(aiv_model::event_waits[PIPE_S][PIPE_MTE2] - setup_waits == (configured ? chunks : 0));
          assert(aiv_model::event_waits[PIPE_MTE2][PIPE_S] - dma_waits == expected_reads);
          assert(aiv_model::mte2_copies - reads == expected_reads);
          assert(aiv_model::mte2_aligned_copies - aligned_reads == expected_aligned);
          assert(aiv_model::mte2_tail_copies - tail_reads == expected_tail);
          assert(aiv_model::mte3_copies - writes == (probe == KVAF_SEQ_PROBE_COPY ? chunks : 0));
          assert(aiv_model::ub_scalar_reads - scalar_reads ==
              (probe == KVAF_SEQ_PROBE_LOAD ? uint64_t(count) * bytes : 0));
          assert(!aiv_model::mte2.destination && !aiv_model::mte3.destination);
          for (const auto& pipeline : model_events)
            for (const auto& direction : pipeline)
              for (bool pending : direction) assert(!pending);
        }
      }
  {
    // DMA completion alone cannot establish correct contents: even corrupted
    // UB is deliberately not read by dma. The load case below must reject it.
    Fixture dma(3, 656, 1, KVAF_SEQ_SCATTER, 6);
    dma.args.scatter_probe = KVAF_SEQ_PROBE_DMA;
    const auto scalar_reads = aiv_model::ub_scalar_reads, writes = aiv_model::mte3_copies;
    const auto output = dma.output;
    aiv_model::corrupt_next_mte2 = true;
    dma.Run(false);
    assert(!aiv_model::corrupt_next_mte2);
    assert(dma.status[0].result == KVAF_SEQ_PROBE_COMPLETE);
    assert(dma.status[0].phase == KVAF_SCATTER_AFTER_DMA);
    assert(aiv_model::ub_scalar_reads == scalar_reads && aiv_model::mte3_copies == writes);
    assert(dma.output == output);
  }
  {
    // A successful DMA completion with one wrong UB byte must fail the load
    // probe, without changing output or issuing any MTE3 payload operation.
    Fixture corrupt(3, 656, 1, KVAF_SEQ_SCATTER, 6);
    corrupt.args.scatter_probe = KVAF_SEQ_PROBE_LOAD;
    const auto writes = aiv_model::mte3_copies;
    aiv_model::corrupt_next_mte2 = true;
    corrupt.Run(false);
    assert(!aiv_model::corrupt_next_mte2);
    assert(corrupt.status[0].result == KVAF_SCATTER_MTE2_MISMATCH);
    assert(corrupt.status[0].phase == KVAF_SEQ_PAYLOAD);
    assert(aiv_model::mte3_copies == writes);
  }
  {
    for (uint32_t probe : {KVAF_SEQ_PROBE_DONE, KVAF_SEQ_PROBE_SETUP, KVAF_SEQ_PROBE_DMA, KVAF_SEQ_PROBE_LOAD}) {
      Fixture empty(1, 656, 1, KVAF_SEQ_SCATTER, 6);
      empty.args.scatter_probe = probe; empty.active = 0; empty.Run(false);
      assert(empty.status[0].result == KVAF_SEQ_INVALID && !empty.push_requests);
      Fixture timeout(1, 656, 1, KVAF_SEQ_SCATTER, 6);
      timeout.args.scatter_probe = probe; timeout.timeout = true; timeout.Run(false);
      assert(timeout.status[0].result == KVAF_SEQ_TIMEOUT);
      Fixture remote(1, 656, 1, KVAF_SEQ_SCATTER, 6);
      remote.args.scatter_probe = probe; remote.remote_error = true; remote.Run(false);
      assert(remote.status[0].result == KVAF_SEQ_REMOTE_ERROR);
    }
    Fixture invalid(1, 656, 1, KVAF_SEQ_SCATTER, 6);
    invalid.args.scatter_probe = 99; invalid.Run(false); assert(!invalid.push_requests);
    Fixture gather(1, 656, 1, KVAF_SEQ_GATHER, 6);
    // Scatter-only configuration cannot reach the gather ABI.
    gather.args.scatter_probe = KVAF_SEQ_PROBE_DONE; gather.Run(); assert(gather.completions);
  }
  {
    // Isolate the real payload helper from URMA: all GM alignment residues,
    // nonaligned byte counts, UB chunk boundaries, and repeated staging writes.
    // Bound every MTE2 read to this token; vector allocation slack must not
    // conceal a rounded-up read of the final tail.
    Fixture f(1, 656, 1, KVAF_SEQ_SCATTER, 1);
    for (uint32_t bytes : {1U, 13U, 31U, 32U, 33U, 63U, 64U, 65U, 656U,
                           16383U, 16384U, 16385U, 32769U, 65535U}) {
      std::vector<uint8_t> source(bytes + 128), output(bytes + 128), expected;
      for (uint32_t offset = 0; offset < 64; ++offset) {
        const auto destination_offset = (offset * 17 + 3) % 64;
        for (uint32_t epoch = 0; epoch < 2; ++epoch) {
          for (uint32_t i = 0; i < source.size(); ++i)
            source[i] = (i * 13 + offset * 7 + epoch * 31) % 251;
          std::fill(output.begin(), output.end(), 0xad);
          std::memset(aiv_model::ub, 0xe9, sizeof(aiv_model::ub));
          expected = output;
          std::copy_n(source.data() + offset, bytes, expected.data() + destination_offset);
          // First validate MTE2 independently against the GM source. No output
          // change, MTE3 copy, or padding byte may participate in verification.
          const auto writes = aiv_model::mte3_copies;
          const auto scalar_reads = aiv_model::ub_scalar_reads;
          aiv_model::mte2_source_begin = Address(source.data() + offset);
          aiv_model::mte2_source_end = aiv_model::mte2_source_begin + bytes;
          assert(CopyPayload(Address(source.data() + offset), Address(output.data() + destination_offset),
                             bytes, KVAF_SEQ_PROBE_LOAD) == KVAF_SEQ_OK);
          assert(aiv_model::mte3_copies == writes);
          assert(aiv_model::ub_scalar_reads - scalar_reads == bytes);
          assert(std::all_of(output.begin(), output.end(), [](uint8_t b) { return b == 0xad; }));
          assert(CopyPayload(Address(source.data() + offset), Address(output.data() + destination_offset),
                             bytes) == KVAF_SEQ_OK);
          aiv_model::mte2_source_begin = aiv_model::mte2_source_end = 0;
          assert(output == expected);
          assert(!aiv_model::mte2.destination && !aiv_model::mte3.destination);
          assert(std::all_of(aiv_model::ub + kCopyBytes, aiv_model::ub + sizeof(aiv_model::ub),
                             [](uint8_t byte) { return byte == 0xe9; }));
          for (const auto& pipeline : model_events)
            for (const auto& direction : pipeline)
              for (bool pending : direction) assert(!pending);
        }
      }
    }
  }
  for (auto mode : {KVAF_SEQ_SCATTER, KVAF_SEQ_GATHER})
    for (uint32_t workers : {1U, 3U, 8U})
      for (uint32_t bytes : {1U, 13U, 656U, 4096U, 65535U}) {
        Fixture f(19, bytes, workers, mode, 6);
        f.Run(); f.Next(7); f.Run(); f.Next(0); f.Run(); f.Next(19); f.Run();
        if (mode == KVAF_SEQ_GATHER && bytes == 656 && workers == 1) assert(f.max_sges == 6);
      }
  for (auto mode : {KVAF_SEQ_SCATTER, KVAF_SEQ_GATHER}) {
    Fixture f(1, 656, 1, mode, 1); f.Run();
    Fixture large(65536, 1, 1, mode, 6); large.Run();
    Fixture negative(7, 656, 1, mode, 6); negative.active = -1; negative.Run(false);
    Fixture excessive(7, 656, 1, mode, 6); excessive.active = 8; excessive.Run(false);
    if (mode == KVAF_SEQ_SCATTER) {
      Fixture invalid(7, 656, 1, mode, 6); invalid.rows[0] = -1; invalid.Run(false);
      Fixture bad_source(7, 656, 1, mode, 6); bad_source.addresses[bad_source.indices[0]] = 1; bad_source.Run(false);
    }
    Fixture timeout(7, 656, 1, mode, 6); timeout.timeout = true; timeout.Run(false);
    Fixture error(7, 656, 1, mode, 6); error.remote_error = true; error.Run(false);
  }
  Fixture future(7, 656, 1, KVAF_SEQ_SCATTER, 6); future.future_done = true; future.Run(false);
  Fixture cq(7, 656, 1, KVAF_SEQ_GATHER, 6); cq.cq_error = true; cq.Run(false);
  assert(cq.states[1].poisoned);
  for (uint32_t sgl = 1; sgl <= 13; ++sgl) {
    Fixture sges(19, 656, 1, KVAF_SEQ_GATHER, sgl); sges.Run();
    assert(sges.max_sges == sgl);
  }
  Fixture data_error(19, 656, 1, KVAF_SEQ_GATHER, 6);
  data_error.data_cq_error = true; data_error.Run(false, true);
  assert(data_error.states[0].poisoned && data_error.data_reads);
  data_error.Next(19); data_error.data_cq_error = false; data_error.Run(false);
  std::cout << "sequential kernel model: PASS\n";
}
