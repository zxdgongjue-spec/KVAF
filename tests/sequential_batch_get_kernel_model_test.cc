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
// The CPU model exercises the kernel body, without Ascend ELF section placement.
#define __attribute__(...)
#include "../access_operator/demo/aiv_batch_get/kernel/execute_scatter_batch_get_aiv.asc"
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
};
void Launch(const ModelArgs& a) {
  if (a.mode == KVAF_SEQ_SCATTER) {
    KvafScatterBatchGetAivKernelV1({a.request, a.workspace, a.workspace_bytes,
      a.descriptor_offset, a.staging_offset, a.source_base, a.source_bytes,
      a.sequence, a.max_polls, a.scatter_probe});
  } else {
    KvafGatherBatchGetAivKernelV1({a.request, a.workspace, a.workspace_bytes,
      a.workspace_entity, a.descriptor_offset, a.lane_descriptor_bytes,
      a.lane_staging_bytes, a.remote_staging, a.remote_descriptor, a.remote_control,
      a.source_base, a.source_bytes, a.sequence, a.sgl_size, a.max_polls});
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
  std::vector<uint32_t> sq = std::vector<uint32_t>(16 * 16);
  alignas(64) uint32_t cq[2][16]{};
  uint32_t db{}, cq_db{};
  uint64_t pi{}, cq_pi{};
};
struct Fixture;
Fixture* fixture;
struct Fixture {
  uint32_t count, token, workers;
  int32_t active;
  ModelArgs args{};
  std::vector<uint8_t, AlignedAllocator<uint8_t>> remote, workspace;
  std::vector<uint8_t> output, expected, valid;
  std::vector<uint64_t> addresses;
  std::vector<int32_t> indices, offsets, rows;
  std::vector<KvafSequentialStatus> status;
  std::vector<KvafBatchGetQueueState> states;
  std::vector<KvafBatchGetDirectUrmaEntity> control_entities, data_entities;
  std::vector<std::unique_ptr<ModelQueue>> queues;
  std::vector<uint64_t> notified;
  std::vector<uint32_t> ack_reads;
  std::vector<bool> cpu_done;
  bool push_done{}, timeout{}, remote_error{}, cq_error{}, data_cq_error{}, future_done{};
  uint32_t push_polls{}, push_requests{}, data_reads{}, max_sges{}, completions{};

  Fixture(uint32_t n, uint32_t bytes, uint32_t cores, uint32_t mode, uint32_t sgl)
      : count(n), token(bytes), workers(cores), active(n), valid(n, 1), addresses(n),
        indices(n), offsets(n), rows(n), status(cores), states(cores * 2),
        control_entities(cores * 2), data_entities(cores * 2), notified(cores),
        ack_reads(cores), cpu_done(cores) {
    fixture = this;
    const uint64_t source_bytes = uint64_t(n) * bytes * 2;
    const uint64_t lane_staging = Align(uint64_t((n + cores - 1) / cores) * bytes);
    const uint64_t lane_descriptor = Align(48 + uint64_t((n + cores - 1) / cores) * 16);
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
    args.descriptor_offset = mode == KVAF_SEQ_SCATTER ? 192 + cores * 64 : cores * 64;
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
      e.magic = 0x4b564155; e.version = 1; e.local_token_id = 1;
      e.sq_va = Address(m.sq.data()); e.sq_db_va = Address(&m.db); e.sq_depth = 16; e.wqe_size = 64;
      e.cq_va = Address(m.cq); e.cq_db_va = Address(&m.cq_db); e.cq_depth = 2; e.cqe_size = 64;
      e.remote_token_id = 2; e.remote_token_value = 3; e.tp_id = 4;
      data_entities[q] = e;
    }
    args.workspace_entity = Address(control_entities.data());
    r.channel_entity = Address(data_entities.data());
  }
  void CheckGuard() { assert(std::all_of(output.begin(), output.end(), [](uint8_t b) { return b == 0xad; })); }
  void Poll(void* p) {
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
  void Doorbell(uint32_t value, uint32_t* p) {
    for (uint32_t q = 0; q < workers * 2; ++q) {
      auto& m = *queues[q];
      if (p == &m.cq_db) { m.cq_db = value; return; }
      if (p != &m.db) continue;
      const uint32_t worker = q / 2;
      auto word = [&](uint64_t base, uint32_t index) {
        return m.sq[((base * 16) + index) % m.sq.size()];
      };
      while (m.pi < value) {
        const auto header = word(m.pi, 0);
        assert((header >> 31) == (1U - ((m.pi / 16) & 1)));
        const auto op = (word(m.pi, 1) >> 8) & 255;
        const auto sges = word(m.pi, 2) >> 24;
        assert(sges >= 1 && sges <= 6);
        uint64_t remote_address = uint64_t(word(m.pi, 10)) | (uint64_t(word(m.pi, 11)) << 32);
        if (q % 2 && op == kRead && remote_address == args.remote_control + worker * 64 + 8) {
          if (!timeout && ++ack_reads[worker] == 3) {
            auto* c = reinterpret_cast<kvaf::peer::GatherControl*>(args.remote_control) + worker;
            assert(c->notify == args.sequence);
            const auto* h = reinterpret_cast<const kvaf::peer::GatherDescriptorHeader*>(c->command);
            assert(h->pipeline_chunk_bytes == 0 && h->sequence == args.sequence);
            kvaf::peer::CpuGatherExecutor cpu(remote.data(), Address(remote.data()), 2);
            assert(cpu.PrepareDirect(remote.size(), c->command, args.sequence, count));
            cpu.Copy();
            c->ack = args.sequence | (remote_error ? kErrorBit : 0);
            cpu_done[worker] = true;
          }
        }
        if (!(q % 2)) {
          assert(op == kRead && cpu_done[worker]);
          assert(ack_reads[worker] >= 3); // Must not treat started ACK as completed.
          ++data_reads; max_sges = std::max(max_sges, sges);
        }
        for (uint32_t i = 0; i < sges; ++i) {
          const auto bytes = word(m.pi, 12 + i * 4);
          const auto local = uint64_t(word(m.pi, 14 + i * 4)) | (uint64_t(word(m.pi, 15 + i * 4)) << 32);
          if (op == kWrite) std::memcpy(reinterpret_cast<void*>(remote_address), reinterpret_cast<void*>(local), bytes);
          else std::memcpy(reinterpret_cast<void*>(local), reinterpret_cast<void*>(remote_address), bytes);
          remote_address += bytes;
        }
        if (q % 2 && op == kWrite) {
          auto* c = reinterpret_cast<kvaf::peer::GatherControl*>(args.remote_control) + worker;
          if (c->notify == args.sequence && notified[worker] != args.sequence) {
            notified[worker] = args.sequence;
            c->ack = ~args.sequence;
            // Even a tempting ready prefix must not bypass the final ACK.
            reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(c->command)->ready_bytes = 0;
          }
        }
        if (header & (1U << 21)) {
          assert(((header >> 16) & 3) == 2);
          m.cq[m.cq_pi % 2][0] = ((1U - ((m.cq_pi / 2) & 1)) << 2) |
              ((cq_error || (data_cq_error && !(q % 2))) ? 1U << 24 : 0);
          ++m.cq_pi; ++completions;
        }
        m.pi += (48 + 16 * sges + 63) / 64;
      }
      assert(m.pi == value); m.db = value;
      return;
    }
    assert(false && "unknown doorbell");
  }
  void Run(bool success = true, bool allow_partial = false) {
    const auto reads = aiv_model::mte2_copies, writes = aiv_model::mte3_copies;
    expected = output;
    if (success && active >= 0) for (int32_t i = 0; i < active; ++i)
      std::memcpy(expected.data() + uint64_t(rows[i]) * token,
                  reinterpret_cast<void*>(addresses[indices[i]] + offsets[i]), token);
    for (uint32_t core = 0; core < workers; ++core) {
      aiv_model::core = core;
      Launch(args);
      assert(status[core].magic == KVAF_SEQ_STATUS_MAGIC);
      assert(status[core].sequence == args.sequence);
      assert((status[core].result == KVAF_SEQ_OK) == success);
    }
    if (args.mode == KVAF_SEQ_GATHER) {
      assert(aiv_model::mte2_copies == reads && aiv_model::mte3_copies == writes);
      assert(!push_requests);
    }
    if (success) assert(output == expected);
    else if (!allow_partial) CheckGuard();
  }
  void Next(uint32_t n) {
    ++args.sequence; active = n;
    std::fill(output.begin(), output.end(), 0xad);
    std::fill(cpu_done.begin(), cpu_done.end(), false);
    std::fill(ack_reads.begin(), ack_reads.end(), 0);
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
  // Match the hardware smoke test: the first launch is empty, followed by
  // nonempty launches on the same queues. Empty work must finish all worker
  // statuses without notifying the CPU or advancing any SQ/CQ state.
  for (uint32_t workers : {1U, 3U, 8U})
    for (uint32_t sgl : {1U, 6U}) {
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
      {KVAF_SEQ_PROBE_GM, KVAF_SCATTER_AFTER_GM}, {KVAF_SEQ_PROBE_BODY, KVAF_SCATTER_AFTER_BODY},
      {KVAF_SEQ_PROBE_UB, KVAF_SCATTER_AFTER_UB},
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
            assert(s.data_wrs == KVAF_SCATTER_DIAGNOSTIC_REVISION);
            if (probe == KVAF_SEQ_PROBE_GM || probe == KVAF_SEQ_PROBE_UB) {
              const uint32_t begin = uint64_t(count) * core / workers;
              const uint32_t end = uint64_t(count) * (core + 1) / workers;
              uint64_t checksum = 0;
              for (uint32_t row = begin; row < end; ++row) {
                const auto* source = reinterpret_cast<const uint8_t*>(f.addresses[f.indices[row]] + f.offsets[row]);
                for (uint32_t offset = 0; offset < bytes; offset += kCopyBytes) {
                  uint64_t part = 0;
                  for (uint32_t j = offset; j < std::min(bytes, offset + kCopyBytes); ++j)
                    part += source[j];
                  checksum ^= part;
                }
              }
              assert(s.sq_pi == checksum && s.cq_ci == uint64_t(end - begin) * bytes);
            }
          }
          assert(f.push_done);  // Every probe still needs the real notify/done protocol.
          assert(f.output == expected);
          const auto chunks = count * ((bytes + kCopyBytes - 1) / kCopyBytes);
          const bool configured = probe != KVAF_SEQ_PROBE_DONE && probe != KVAF_SEQ_PROBE_FLUSH;
          const bool copied = configured && probe != KVAF_SEQ_PROBE_SETUP && probe != KVAF_SEQ_PROBE_GM;
          const auto expected_aligned = copied ? count * (bytes / kCopyBytes +
              (bytes % kCopyBytes >= 32 ? 1 : 0)) : 0;
          const auto expected_tail = copied && probe != KVAF_SEQ_PROBE_BODY && bytes % 32 ? count : 0;
          const auto expected_reads = expected_aligned + expected_tail;
          assert(aiv_model::out_to_ub_resets - resets == (configured ? chunks : 0));
          assert(aiv_model::event_waits[PIPE_S][PIPE_MTE2] - setup_waits == (configured ? chunks : 0));
          assert(aiv_model::event_waits[PIPE_MTE2][PIPE_S] - dma_waits == expected_reads);
          assert(aiv_model::mte2_copies - reads == expected_reads);
          assert(aiv_model::mte2_aligned_copies - aligned_reads == expected_aligned);
          assert(aiv_model::mte2_tail_copies - tail_reads == expected_tail);
          assert(aiv_model::mte3_copies - writes == (probe == KVAF_SEQ_PROBE_COPY ? chunks : 0));
          assert(aiv_model::ub_scalar_reads - scalar_reads ==
              (probe == KVAF_SEQ_PROBE_LOAD || probe == KVAF_SEQ_PROBE_UB ? uint64_t(count) * bytes : 0));
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
    assert(corrupt.status[0].sq_pi == 639);  // Last byte of the 640-byte body was corrupted.
    const auto packed = corrupt.status[0].cq_ci;
    assert(((packed >> 8) & 255) == ((packed & 255) ^ 1));
    assert(aiv_model::mte3_copies == writes);
  }
  {
    for (uint32_t probe : {KVAF_SEQ_PROBE_DONE, KVAF_SEQ_PROBE_SETUP, KVAF_SEQ_PROBE_DMA, KVAF_SEQ_PROBE_LOAD,
                           KVAF_SEQ_PROBE_GM, KVAF_SEQ_PROBE_BODY, KVAF_SEQ_PROBE_UB}) {
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
    Fixture invalid(7, 656, 1, mode, 6); invalid.rows[0] = -1; invalid.Run(false);
    Fixture bad_source(7, 656, 1, mode, 6); bad_source.addresses[bad_source.indices[0]] = 1; bad_source.Run(false);
    Fixture timeout(7, 656, 1, mode, 6); timeout.timeout = true; timeout.Run(false);
    Fixture error(7, 656, 1, mode, 6); error.remote_error = true; error.Run(false);
  }
  Fixture future(7, 656, 1, KVAF_SEQ_SCATTER, 6); future.future_done = true; future.Run(false);
  Fixture cq(7, 656, 1, KVAF_SEQ_GATHER, 6); cq.cq_error = true; cq.Run(false);
  assert(cq.states[1].poisoned);
  for (uint32_t sgl = 1; sgl <= 6; ++sgl) {
    Fixture sges(19, 656, 1, KVAF_SEQ_GATHER, sgl); sges.Run();
    assert(sges.max_sges == sgl);
  }
  Fixture data_error(19, 656, 1, KVAF_SEQ_GATHER, 6);
  data_error.data_cq_error = true; data_error.Run(false, true);
  assert(data_error.states[0].poisoned && data_error.data_reads);
  data_error.Next(19); data_error.data_cq_error = false; data_error.Run(false);
  std::cout << "sequential kernel model: PASS\n";
}
