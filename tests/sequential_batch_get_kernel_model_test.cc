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
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <malloc.h>
#endif
#include "kernel_operator.h"
#include "cpu_gather_executor.h"
#include "cpu_push_get_abi.h"
#include "cpu_push_get_layout.h"
#include "sequential_batch_get_abi.h"
// The CPU model exercises the kernel body, without Ascend ELF section placement.
#define __attribute__(...)
#include "../access_operator/demo/aiv_batch_get/kernel/execute_scatter_batch_get_aiv.asc"
#include "../access_operator/demo/aiv_batch_get/kernel/execute_gather_batch_get_aiv.asc"
#undef __attribute__
#include "scatter_mailbox_templates.h"

namespace {
using namespace kvaf_scatter;
using kvaf_gather::kRead;
using kvaf_gather::kWrite;
using kvaf_gather::kErrorBit;
constexpr std::array<uint32_t, 1> kScatterOptions{kvaf::kScatterOptionsV1};
constexpr auto kCheckedScatterOptions = kScatterOptions;
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
  uint32_t reserved;
};
void Launch(const ModelArgs& a) {
  if (a.mode == KVAF_SEQ_SCATTER) {
    KvafScatterBatchGetAivKernelV1({a.request, a.workspace, a.workspace_bytes,
      a.descriptor_offset, a.staging_offset, a.source_base, a.source_bytes,
      a.sequence, a.max_polls, a.reserved});
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
#ifdef _WIN32
    void* p = _aligned_malloc(n * sizeof(T), 64);
    if (!p) throw std::bad_alloc();
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, n * sizeof(T))) throw std::bad_alloc();
#endif
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
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
  std::vector<uint8_t, AlignedAllocator<uint8_t>> mailbox_memory;
  KvafScatterMailboxContext mailbox{};
  uint32_t mailbox_sq_copies{}, mailbox_posts{};
  bool mailbox_cq_timeout{};
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
  bool hold_done_until_scatter{}, copied_before_done{}, stall_ready{}, future_ready{}, error_after_scatter{};
  bool hold_next_jetty{}, late_first_jetty{}, copied_before_all_ready{}, waited_with_output_pending{};
  struct PushPart { uint32_t worker, chunk; kvaf::cpu_push::Range range; };
  std::vector<PushPart> push_parts;
  struct HeadPart { uint32_t worker, chunk, queue; kvaf::cpu_push::Range range; bool suffix, ready; };
  std::vector<HeadPart> head_parts;
  bool head_started_early{};
  std::vector<bool> payload_read;
  uint32_t push_step{}, read_rows{};
  uint32_t ready_per_poll{1};
  uint32_t push_polls{}, push_requests{}, data_reads{}, max_sges{}, completions{};
  std::mutex poll_mutex;

  Fixture(uint32_t n, uint32_t bytes, uint32_t cores, uint32_t mode, uint32_t sgl)
      : count(n), token(bytes), workers(cores), active(n), valid(n, 1), addresses(n),
        indices(n), offsets(n), rows(n), status(cores), states(cores * 2),
        control_entities(cores * 2), data_entities(cores * 2), notified(cores),
        ack_reads(cores), cpu_done(cores) {
    fixture = this;
    aiv_model::payload_observer = [](const void* p, uint64_t bytes) { fixture->PayloadRead(p, bytes); };
    payload_read.resize(n);
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
    args.mode = mode; args.sequence = 1; args.sgl_size = sgl;
    args.max_polls = mode == KVAF_SEQ_SCATTER && cores > 1 ? 100000 : 64;
    if (mode == KVAF_SEQ_SCATTER) args.reserved = kScatterOptions.back();
    args.source_base = Address(remote.data()); args.source_bytes = source_bytes;
    args.lane_descriptor_bytes = lane_descriptor; args.lane_staging_bytes = lane_staging;
    args.remote_staging = Address(remote.data()) + staging_offset;
    args.remote_descriptor = Address(remote.data()) + descriptor_offset;
    args.remote_control = Address(remote.data()) + control_offset;
    args.descriptor_offset = mode == KVAF_SEQ_SCATTER ? 192 + cores * 64 : cores * 64;
    args.staging_offset = Align(args.descriptor_offset + n * 8ULL);
    args.workspace_bytes = mode == KVAF_SEQ_SCATTER ?
        Align(args.staging_offset + n * uint64_t(bytes)) + sizeof(mailbox) + uint64_t(cores + 1) * 64 :
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
    if (mode == KVAF_SEQ_SCATTER) {
      mailbox_memory.resize(args.staging_offset);
      mailbox.entity = control_entities[0];
      mailbox.entity.remote_host_ddr_address = Address(mailbox_memory.data());
      mailbox.entity.remote_host_ddr_size = mailbox_memory.size();
      mailbox.max_write_bytes = 65536;
      kvaf::PrepareScatterMailboxTemplates(&mailbox);
      r.channel_entity = Address(&mailbox);
      r.queue_state = Address(&mailbox.queue);
    }
  }
  void CheckGuard() { assert(std::all_of(output.begin(), output.end(), [](uint8_t b) { return b == 0xad; })); }
  void Poll(void* p) {
    std::lock_guard<std::mutex> lock(poll_mutex);
    const auto address = Address(p);
    const bool control_cq = address >= Address(queues[0]->cq) &&
                            address < Address(queues[0]->cq) + sizeof(queues[0]->cq);
    if (args.mode != KVAF_SEQ_SCATTER || push_done ||
        (!control_cq && (address < (args.workspace & ~uint64_t(63)) ||
                        address >= args.workspace + args.workspace_bytes))) return;
    auto* request = reinterpret_cast<kvaf::cpu_push::Request*>(mailbox_memory.data());
    const auto notify = aiv_model::AtomicLoadWord(Address(mailbox_memory.data()) + 64);
    if ((notify & ~kvaf::cpu_push::kProfileNotify) != args.sequence) return;
    const bool pipeline = kvaf::ScatterOptionEnabled(args.reserved, kvaf::kScatterPipeline);
    const bool head = (request->reserved & kvaf::cpu_push::kHeadStart) != 0;
    if (pipeline && !aiv_model::mte3.empty()) waited_with_output_pending = true;
    if (!pipeline) CheckGuard();
    if (timeout || (++push_polls < 4 && !head)) return;
    assert(request->magic == kvaf::cpu_push::kMagic && request->sequence == args.sequence);
    assert(request->count == uint32_t(active));
    if (!head) {
      const auto base = args.workspace + args.workspace_bytes - uint64_t(workers + 1) * 64;
      for (uint32_t core = 0; core < workers; ++core)
        assert(aiv_model::AtomicLoadWord(base + core * 64) == args.sequence);
    }
    const auto* src = reinterpret_cast<const uint64_t*>(mailbox_memory.data() + args.descriptor_offset);
    if (head) {
      const auto cut = uint32_t(request->reserved_words[0]);
      if (head_parts.empty()) {
        for (uint32_t phase = 0; phase < 2; ++phase)
          kvaf::cpu_push::ForEachPipelineChunk(active, token, workers,
            [&](uint32_t worker, uint32_t chunk, kvaf::cpu_push::Range range) {
              for (uint32_t q = 0; q < 4; ++q) {
                const auto shard = kvaf::cpu_push::JettyRange(range, q);
                const auto first = std::max(shard.begin, phase ? cut : 0U);
                const auto last = std::min(shard.end, phase ? uint32_t(active) : cut);
                if (first < last) head_parts.push_back({worker, chunk, q, {first, last},
                                                       phase != 0, last == shard.end});
              }
            });
      }
      const bool suffix = aiv_model::AtomicLoadWord(Address(mailbox_memory.data()) +
                            kvaf::cpu_push::kSuffixNotifyOffset) == args.sequence;
      if (push_step < head_parts.size()) {
        const auto& part = head_parts[push_step];
        if (part.suffix && !suffix) return;
        if (!part.suffix) head_started_early |= aiv_model::AtomicLoadWord(
            args.workspace + args.workspace_bytes - uint64_t(workers + 1) * 64) != args.sequence;
        for (auto row = part.range.begin; row < part.range.end; ++row)
          std::memcpy(workspace.data() + args.staging_offset + uint64_t(row) * token,
                      reinterpret_cast<void*>(src[row]), token);
        if (pipeline && part.ready) aiv_model::AtomicStoreWord(
            args.workspace + 192 + part.worker * 64 + part.queue * 8,
            kvaf::cpu_push::Progress(args.sequence, part.chunk + 1));
        ++push_step;
        return;
      }
      assert(suffix); // A complete prefix alone must never publish done.
      if (hold_done_until_scatter && read_rows < uint32_t(active)) return;
    } else if (pipeline && !remote_error && !future_done) {
      if (push_parts.empty()) kvaf::cpu_push::ForEachPipelineChunk(active, token, workers,
          [&](uint32_t worker, uint32_t chunk, kvaf::cpu_push::Range range) {
            push_parts.push_back({worker, chunk, range});
          });
      if (push_step < push_parts.size() * kvaf::cpu_push::kJettyCount) {
        // Deliver test bursts only on Ready refreshes so unrelated polls do
        // not change which progress words the consumer sees together.
        if (ready_per_poll > 1 && address != args.workspace + 192) return;
        for (uint32_t i = 0; i < ready_per_poll &&
             push_step < push_parts.size() * kvaf::cpu_push::kJettyCount; ++i, ++push_step) {
          const auto& part = push_parts[push_step / kvaf::cpu_push::kJettyCount];
          const auto q = (push_step + (late_first_jetty ? 1U : 0U)) % kvaf::cpu_push::kJettyCount;
          if (hold_next_jetty && push_step % kvaf::cpu_push::kJettyCount) {
            const auto previous = kvaf::cpu_push::JettyRange(part.range, (q + 3) % 4);
            for (auto row = previous.begin; row < previous.end; ++row) if (!payload_read[row]) return;
          }
          const auto shard = kvaf::cpu_push::JettyRange(part.range, q);
          for (auto row = shard.begin; row < shard.end; ++row)
            std::memcpy(workspace.data() + args.staging_offset + uint64_t(row) * token,
                        reinterpret_cast<void*>(src[row]), token);
          if (!stall_ready || q != 3) aiv_model::AtomicStoreWord(
              args.workspace + 192 + part.worker * 64 + q * 8,
              kvaf::cpu_push::Progress(args.sequence + future_ready, part.chunk + 1));
        }
        return; // Delay other queues/chunks and final done until later polls.
      }
      if (stall_ready || (hold_done_until_scatter && read_rows < uint32_t(active))) return;
    } else {
      for (uint32_t i = 0; i < uint32_t(active); ++i)
        std::memcpy(workspace.data() + args.staging_offset + uint64_t(i) * token,
                    reinterpret_cast<void*>(src[i]), token);
    }
    const uint64_t done = ((args.sequence + (future_done ? 1 : 0)) << 32) |
        ((remote_error || error_after_scatter) ? 5 : 1);
    if (notify & kvaf::cpu_push::kProfileNotify) {
      assert(request->reserved == (kvaf::cpu_push::kInlineProfile |
          (pipeline ? 0U : kvaf::cpu_push::kBatchDoneOnly) |
          (head ? kvaf::cpu_push::kHeadStart : 0U)));
      const uint64_t report[] = {kvaf::cpu_push::ProfileToken(args.sequence), 11, 22, 1000, 800, 200, 500};
      std::memcpy(workspace.data() + 136, report, sizeof(report));
    }
    push_done = true; ++push_requests;
    aiv_model::AtomicStoreWord(args.workspace + 128, done);
  }
  void PayloadRead(const void* p, uint64_t bytes) {
    if (!kvaf::ScatterOptionEnabled(args.reserved, kvaf::kScatterPipeline)) return;
    std::lock_guard<std::mutex> lock(poll_mutex);
    const auto offset = Address(p) - args.workspace - args.staging_offset;
    const auto begin = offset / token, end = (offset + bytes + token - 1) / token;
    assert(end <= uint32_t(active));
    for (uint32_t i = begin; i < end; ++i) {
      for (uint32_t worker = 0; worker < workers; ++worker) {
        const auto range = kvaf::cpu_push::WorkerRange(active, token, worker, workers);
        if (i < range.begin || i >= range.end) continue;
        const auto chunk = (i - range.begin) / kvaf::cpu_push::kChunkRows + 1;
        const auto part = kvaf::cpu_push::ChunkRange(range, chunk - 1);
        bool all_ready = true;
        for (uint32_t q = 0; q < kvaf::cpu_push::kJettyCount; ++q) {
          const auto progress = aiv_model::AtomicLoadWord(args.workspace + 192 + worker * 64 + q * 8);
          const bool ready = progress >> 32 == args.sequence && uint32_t(progress) >= chunk;
          all_ready &= ready;
          const auto shard = kvaf::cpu_push::JettyRange(part, q);
          if (i >= shard.begin && i < shard.end) assert(ready);
        }
        copied_before_all_ready |= !all_ready;
      }
      if (!payload_read[i]) { payload_read[i] = true; ++read_rows; }
    }
    copied_before_done |= !push_done;
  }
  void Doorbell(uint32_t value, uint32_t* p) {
    std::lock_guard<std::mutex> lock(poll_mutex);
    for (uint32_t q = 0; q < workers * 2; ++q) {
      auto& m = *queues[q];
      if (p == &m.cq_db) { m.cq_db = value; return; }
      if (p != &m.db) continue;
      const bool scatter = args.mode == KVAF_SEQ_SCATTER;
      if (scatter) {
        assert(q == 0 && aiv_model::mte3.empty());
        ++mailbox_posts;
        mailbox_sq_copies += 1 + ((m.pi % 16) + value - m.pi > 16);
      }
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
        if (!scatter && !(q % 2)) {
          assert(op == kRead && cpu_done[worker]);
          assert(ack_reads[worker] >= 3); // Must not treat started ACK as completed.
          ++data_reads; max_sges = std::max(max_sges, sges);
        }
        for (uint32_t i = 0; i < sges; ++i) {
          const auto bytes = word(m.pi, 12 + i * 4);
          const auto local = uint64_t(word(m.pi, 14 + i * 4)) | (uint64_t(word(m.pi, 15 + i * 4)) << 32);
          if (scatter) {
            assert(op == kWrite && sges == 1 && bytes <= mailbox.max_write_bytes);
            assert(word(m.pi, 3) == mailbox.entity.remote_token_id);
            assert(word(m.pi, 13) == mailbox.entity.local_token_id);
            const auto offset = remote_address - Address(mailbox_memory.data());
            assert(offset + bytes <= mailbox_memory.size());
            assert(local == args.workspace + offset);
            assert((offset < 64 && offset + bytes <= 64) ||
                   (offset >= args.descriptor_offset && offset + bytes <= args.descriptor_offset + active * 8) ||
                   ((offset == 64 || offset == kvaf::cpu_push::kSuffixNotifyOffset) && bytes == 8));
            if (offset == 64 || offset == kvaf::cpu_push::kSuffixNotifyOffset) {
              // Commit cannot precede any header/list placement. Verify exact
              // fresh contents, including on same-sized consecutive calls.
              assert(((header >> 16) & 3) == 2);
              assert(!std::memcmp(mailbox_memory.data(), workspace.data(), 64));
              const auto* request = reinterpret_cast<const kvaf::cpu_push::Request*>(mailbox_memory.data());
              const auto rows = offset == 64 && (request->reserved & kvaf::cpu_push::kHeadStart) ?
                                  request->reserved_words[0] : uint64_t(active);
              assert(!std::memcmp(mailbox_memory.data() + args.descriptor_offset,
                                  workspace.data() + args.descriptor_offset, rows * 8));
              aiv_model::AtomicStoreWord(remote_address, aiv_model::AtomicLoadWord(local));
            } else {
              const auto* request = reinterpret_cast<const kvaf::cpu_push::Request*>(mailbox_memory.data());
              const bool suffix = (request->reserved & kvaf::cpu_push::kHeadStart) &&
                  offset >= args.descriptor_offset + request->reserved_words[0] * 8;
              if (suffix) assert(aiv_model::AtomicLoadWord(Address(mailbox_memory.data()) + 72) != args.sequence);
              else assert((aiv_model::AtomicLoadWord(Address(mailbox_memory.data()) + 64) &
                            ~kvaf::cpu_push::kProfileNotify) != args.sequence);
              std::memcpy(reinterpret_cast<void*>(remote_address), reinterpret_cast<void*>(local), bytes);
            }
          }
          else if (op == kWrite) std::memcpy(reinterpret_cast<void*>(remote_address), reinterpret_cast<void*>(local), bytes);
          else std::memcpy(reinterpret_cast<void*>(local), reinterpret_cast<void*>(remote_address), bytes);
          remote_address += bytes;
        }
        if (!scatter && q % 2 && op == kWrite) {
          auto* c = reinterpret_cast<kvaf::peer::GatherControl*>(args.remote_control) + worker;
          if (c->notify == args.sequence && notified[worker] != args.sequence) {
            notified[worker] = args.sequence;
            c->ack = ~args.sequence;
            // Even a tempting ready prefix must not bypass the final ACK.
            reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(c->command)->ready_bytes = 0;
          }
        }
        if ((header & (1U << 21)) && !(scatter && mailbox_cq_timeout)) {
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
  void SetModelInputs() {
    aiv_model::mte2_source_begin = args.workspace + args.staging_offset;
    aiv_model::mte2_source_end = aiv_model::mte2_source_begin +
        uint64_t(std::max(active, 0)) * token;
    // Exact active prefixes: invalid capacity suffixes cannot be DMA-read.
    const auto prefix = uint64_t(std::max(active, 0)) * 4;
    aiv_model::metadata_sources = {
        {args.request.indices, args.request.indices + prefix},
        {args.request.offsets, args.request.offsets + prefix},
        {args.request.output_indices, args.request.output_indices + prefix},
        {args.request.address_table, args.request.address_table + uint64_t(args.request.table_capacity) * 8},
        {args.request.valid_table, args.request.valid_table + args.request.table_capacity}};
    aiv_model::metadata_sources.push_back({Address(mailbox.write_templates),
        Address(mailbox.write_templates) + sizeof(mailbox.write_templates)});
    aiv_model::address_table_begin = args.request.address_table;
    aiv_model::address_table_end = args.request.address_table + uint64_t(args.request.table_capacity) * 8;
    aiv_model::descriptor_begin = args.workspace + args.descriptor_offset;
    aiv_model::descriptor_end = aiv_model::descriptor_begin + prefix * 2;
  }
  void ClearModelInputs() {
    aiv_model::metadata_sources.clear();
    aiv_model::address_table_begin = aiv_model::address_table_end = 0;
    aiv_model::descriptor_begin = aiv_model::descriptor_end = 0;
    aiv_model::mte2_source_begin = aiv_model::mte2_source_end = 0;
  }
  void Run(bool success = true, bool allow_partial = false) {
    const auto reads = aiv_model::mte2_copies, writes = aiv_model::mte3_copies;
    const auto payload_cache_lines = aiv_model::payload_cache_lines;
    expected = output;
    if (success && active >= 0) for (int32_t i = 0; i < active; ++i)
      std::memcpy(expected.data() + uint64_t(rows[i]) * token,
                  reinterpret_cast<void*>(addresses[indices[i]] + offsets[i]), token);
    auto run_core = [&](uint32_t core) {
      aiv_model::core = core;
      if (args.mode == KVAF_SEQ_SCATTER) {
        std::memset(aiv_model::ub, 0xe9, sizeof(aiv_model::ub));
        SetModelInputs();
      }
      Launch(args);
      assert(status[core].magic == KVAF_SEQ_STATUS_MAGIC);
      assert(status[core].sequence == args.sequence);
      if ((status[core].result == KVAF_SEQ_OK) != success)
        std::cerr << "kernel failure: mode=" << args.mode << " count=" << active << " token=" << token
                  << " workers=" << workers << " core=" << core << " options=" << args.reserved
                  << " phase=" << status[core].phase << " result=" << status[core].result << '\n';
      assert((status[core].result == KVAF_SEQ_OK) == success);
      assert(aiv_model::mte2.empty() && aiv_model::mte3.empty());
      if (args.mode == KVAF_SEQ_SCATTER) {
        const auto used_ub = kTileBufferBytes;
        assert(std::all_of(aiv_model::ub + used_ub, aiv_model::ub + sizeof(aiv_model::ub),
                           [](uint8_t byte) { return byte == 0xe9; }));
      }
      ClearModelInputs();
    };
    if (args.mode == KVAF_SEQ_SCATTER && workers > 1) {
      // Actual concurrent builders: core 0 may only notify after every worker
      // has completed its disjoint descriptor DMA and published this sequence.
      std::vector<std::thread> threads;
      for (uint32_t core = 0; core < workers; ++core) threads.emplace_back(run_core, core);
      for (auto& thread : threads) thread.join();
    } else {
      for (uint32_t core = 0; core < workers; ++core) run_core(core);
    }
    if (args.mode == KVAF_SEQ_GATHER) {
      assert(aiv_model::mte2_copies == reads && aiv_model::mte3_copies == writes);
      assert(!push_requests);
    } else assert(aiv_model::payload_cache_lines == payload_cache_lines);
    if (success) assert(output == expected);
    else if (!allow_partial) CheckGuard();
  }
  void Next(uint32_t n) {
    ++args.sequence; active = n;
    std::fill(output.begin(), output.end(), 0xad);
    std::fill(cpu_done.begin(), cpu_done.end(), false);
    std::fill(ack_reads.begin(), ack_reads.end(), 0);
    push_done = false; push_polls = 0;
    push_parts.clear(); head_parts.clear(); push_step = read_rows = 0; copied_before_done = false;
    head_started_early = false;
    copied_before_all_ready = waited_with_output_pending = false;
    std::fill(payload_read.begin(), payload_read.end(), false);
    // Keep old notify/done/ACK and persistent PI/CI to exercise sequence isolation.
    for (uint64_t i = 0; i < args.source_bytes; ++i) remote[i] ^= 0x35;
    std::reverse(indices.begin(), indices.end());
  }
};
}
namespace aiv_model {
void Poll(void* p) {
  // Check completion at status publication, before the entry's final PIPE_ALL
  // barrier could hide an early-success bug in the scatter drain.
  if (fixture->args.mode == KVAF_SEQ_SCATTER) {
    const auto& status = fixture->status[core];
    const auto line = Address(&status) & ~uint64_t(63);
    if (Address(p) == line && status.phase == KVAF_SEQ_COMPLETE)
      assert(mte2.empty() && mte3.empty());
  }
  fixture->Poll(p);
}
void Doorbell(uint32_t value, uint32_t* p) { fixture->Doorbell(value, p); }
}
int main() {
  {
    // Cross-DMA waits must queue dependencies, not block Scalar. Refill the
    // same bank before either engine is drained; it must still output the old
    // contents first. This exercises the model used by the real stage ring.
    using namespace AscendC;
    alignas(32) uint8_t first[64], second[64], output[64]{};
    std::memset(first, 0x21, sizeof(first));
    std::memset(second, 0x43, sizeof(second));
    LocalTensor<uint8_t> bank(TPosition::VECIN, 0, 64);
    GlobalTensor<uint8_t> source, destination;
    source.SetGlobalBuffer(first); destination.SetGlobalBuffer(output);
    DataCopy(bank, source, 64);
    set_flag(PIPE_MTE2, PIPE_MTE3, 0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, 0);
    assert(!aiv_model::mte2.empty());
    DataCopyPad(destination, bank, DataCopyExtParams{1, 64, 0, 0, 0});
    set_flag(PIPE_MTE3, PIPE_MTE2, 0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, 0);
    assert(!aiv_model::mte3.empty() && output[0] == 0);
    source.SetGlobalBuffer(second);
    DataCopy(bank, source, 64);
    set_flag(PIPE_MTE2, PIPE_S, 0);
    wait_flag(PIPE_MTE2, PIPE_S, 0);
    assert(std::memcmp(output, first, 64) == 0);
    assert(bank.GetValue(0) == second[0]);
    assert(aiv_model::mte2.empty() && aiv_model::mte3.empty());
  }
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
  for (uint32_t workers : {1U, 3U, 8U}) {
    Fixture scatter(19, 656, workers, KVAF_SEQ_SCATTER, 6);
    const auto cycles = aiv_model::cycle;
    scatter.active = 0; scatter.Run();
    assert(!scatter.push_requests);
    for (const auto& status : scatter.status)
      assert(status.phase == KVAF_SEQ_COMPLETE && !status.requests);
    scatter.Next(1); scatter.Run();
    scatter.Next(19); scatter.Run();
    assert(aiv_model::cycle == cycles); // No timing reads in the baseline path.
  }
  // Exercise the batched copy through the existing complete scatter launch,
  // including CPU notification, changing payloads, output holes and statuses.
  // Deferred model writes must finish before a tile reuses the same UB bank.
  for (const auto options : kScatterOptions)
    for (uint32_t workers : {1U, 3U, 8U})
      for (uint32_t bytes : {31U, 656U, 672U, 673U, 1000U, 4096U, 32768U, 32769U}) {
        Fixture f(129, bytes, workers, KVAF_SEQ_SCATTER, 6);
        // Permute unique sparse destinations: output has no fixed stride.
        for (uint32_t i = 0; i < f.count; ++i) f.rows[i] = 2 * ((i * 37U + 11U) % f.count) + 1;
        f.args.reserved = options;
        for (uint32_t count : {0U, 1U, 7U, 31U, 32U, 33U, 63U, 64U, 65U,
                               95U, 96U, 97U, 129U}) {
          f.Next(count);
          std::memset(aiv_model::ub, 0xe9, sizeof(aiv_model::ub));
          const auto reads = aiv_model::mte2_copies;
          const auto writes = aiv_model::mte3_copies;
          const auto waits = aiv_model::event_waits[PIPE_MTE3][PIPE_S];
          const auto input_waits = aiv_model::event_waits[PIPE_MTE2][PIPE_S];
          const auto recycle_waits = aiv_model::event_waits[PIPE_MTE3][PIPE_MTE2];
          const auto metadata_reads = aiv_model::metadata_reads;
          const auto descriptor_writes = aiv_model::descriptor_writes;
          const auto control_copies = f.mailbox_sq_copies, control_posts = f.mailbox_posts;
          f.Run();
          if (bytes <= kTileBytes) {
            const auto layout = MakeRowCopyLayout(bytes);
            const auto per_tile = layout.rows;
            assert(layout.banks >= 2 && layout.banks <= kMaxBanks);
            assert(layout.banks * layout.bank_bytes +
                (layout.cached_rows ? layout.banks * kRowIndexBytes : 0U) <= kTileBufferBytes);
            if (bytes == 656) assert(layout.banks == 3 && layout.cached_rows && per_tile == 32);
            uint64_t tiles = 0, active_workers = 0;
            for (uint32_t core = 0; core < workers; ++core) {
              const auto rows = uint64_t(count) * (core + 1) / workers -
                  uint64_t(count) * core / workers;
              const auto core_tiles = (rows + per_tile - 1) / per_tile;
              tiles += core_tiles;
              if (core_tiles) ++active_workers;
            }
            uint64_t build_tiles = 0;
            {
              const auto groups = (count + 7U) / 8U;
              for (uint32_t core = 0; core < workers; ++core) {
                const auto begin = uint64_t(groups) * core / workers * 8;
                const auto end = std::min(uint64_t(count), uint64_t(groups) * (core + 1) / workers * 8);
                if (end > begin) build_tiles += (end - begin + kSimtBuildRows - 1) / kSimtBuildRows;
              }
            }
            const auto metadata_tiles = layout.cached_rows ? tiles : 0;
            assert(aiv_model::metadata_reads - metadata_reads == metadata_tiles + f.mailbox_posts - control_posts);
            assert(aiv_model::descriptor_writes - descriptor_writes == build_tiles);
            assert(aiv_model::mte2_copies - reads == tiles + metadata_tiles + f.mailbox_posts - control_posts);
            assert(aiv_model::mte3_copies - writes == count + build_tiles +
                   f.mailbox_sq_copies - control_copies);
            // Scatter no longer waits for MTE3 on Scalar at every tile.
            assert(aiv_model::event_waits[PIPE_MTE3][PIPE_S] - waits == build_tiles +
                   f.mailbox_posts - control_posts);
            assert(aiv_model::event_waits[PIPE_MTE3][PIPE_MTE2] - recycle_waits == tiles);
            assert(aiv_model::event_waits[PIPE_MTE2][PIPE_S] - input_waits ==
                metadata_tiles + active_workers + f.mailbox_posts - control_posts);
          }
          assert(aiv_model::mte2.empty() && aiv_model::mte3.empty());
        }
      }
  // A late invalid row can leave private partial descriptors in fused mode,
  // but must never publish a new notify or let CPU dereference that list.
  for (const auto options : kCheckedScatterOptions)
    for (uint32_t workers : {1U, 3U, 8U})
      for (uint32_t field = 0; field < 8; ++field) {
        Fixture f(65, 656, workers, KVAF_SEQ_SCATTER, 6);
        f.args.reserved = options;
        f.Run();
        const auto requests = f.push_requests;
        f.Next(65);
        if (field == 0) f.indices.back() = f.count;
        if (field == 1) f.offsets.back() = f.token + 1;
        if (field == 2) f.rows.back() = f.args.request.output_rows;
        if (field == 3) f.valid[f.indices.back()] = 0;
        if (field == 4) f.addresses[f.indices.back()] = f.args.source_base + f.args.source_bytes;
        if (field == 5) f.indices.back() = -1;
        if (field == 6) f.offsets.back() = -1;
        if (field == 7) f.rows.back() = -1;
        const auto reads = aiv_model::mte2_copies, writes = aiv_model::mte3_copies;
        const auto metadata_reads = aiv_model::metadata_reads;
        const auto descriptor_writes = aiv_model::descriptor_writes;
        f.Run(false);
        assert(f.push_requests == requests);
        // Other builders may finish private tiles; no payload may be touched.
        assert(aiv_model::mte2_copies - reads == aiv_model::metadata_reads - metadata_reads);
        assert(aiv_model::mte3_copies - writes == aiv_model::descriptor_writes - descriptor_writes);
        assert(*reinterpret_cast<const uint64_t*>(f.workspace.data() + 64) != f.args.sequence);
        for (const auto& status : f.status) assert(status.result == KVAF_SEQ_INVALID);
      }
  // UB tile boundaries and different table sizes, through full
  // BatchGet including CPU service and changing metadata at the same addresses.
  for (uint32_t capacity : {1024U, 4096U, 4097U})
    for (uint32_t workers : {1U, 4U, 8U}) {
      Fixture f(capacity, 31, workers, KVAF_SEQ_SCATTER, 6);
      for (uint32_t count : {1U, 7U, 255U, 256U, 257U, 1023U, 1024U}) {
        f.Next(count); f.Run();
        // Change offsets and output maps as well as payload/indices.
        for (auto& offset : f.offsets) offset = f.token - offset;
        std::reverse(f.rows.begin(), f.rows.end());
      }
      f.Next(capacity); f.Run();
    }
  {
    // A missing builder must time out without ever notifying CPU. Workers that
    // arrive afterwards observe the failure; their ready flags cannot revive it.
    Fixture stalled(65, 656, 4, KVAF_SEQ_SCATTER, 6);
    stalled.args.max_polls = 2;
    stalled.SetModelInputs();
    aiv_model::core = 0; Launch(stalled.args);
    assert(stalled.status[0].result == KVAF_SEQ_TIMEOUT);
    for (uint32_t core = 1; core < stalled.workers; ++core) {
      aiv_model::core = core; Launch(stalled.args);
      assert(stalled.status[core].result == KVAF_SEQ_INVALID);
    }
    assert(!stalled.push_requests);
    stalled.CheckGuard();
    stalled.ClearModelInputs();
  }
  {
    Fixture small_workspace(65, 31, 4, KVAF_SEQ_SCATTER, 6);
    small_workspace.args.workspace_bytes = small_workspace.args.staging_offset + 64;
    small_workspace.Run(false);
    assert(!small_workspace.push_requests);
  }
  for (const auto options : {2U, kvaf::kScatterOptionsV1 | 64U, kvaf::kScatterOptionsV1 | 128U,
                             kvaf::kScatterOptionsV1 ^ 0x100U, kvaf::kScatterProfileFlag | 2U}) {
    Fixture invalid(1, 656, 1, KVAF_SEQ_SCATTER, 6);
    invalid.args.reserved = options; invalid.Run(false);
    assert(!invalid.push_requests && invalid.status[0].result == KVAF_SEQ_INVALID);
  }
  {
    Fixture legacy(65, 656, 1, KVAF_SEQ_SCATTER, 6);
    legacy.args.reserved = 0; legacy.Run();
  }
  // The SIMT builder handles reference-sized sparse tables and changing table
  // contents at the same address, without scalar table DCCI.
  for (const bool trusted : {false, true})
    for (uint32_t workers : {1U, 4U, 8U}) {
      Fixture f(1024, 31, workers, KVAF_SEQ_SCATTER, 6);
      constexpr uint32_t table_capacity = 1277952;
      f.addresses.resize(table_capacity);
      f.valid.resize(table_capacity, 1);
      f.args.request.table_capacity = table_capacity;
      f.args.request.address_table = Address(f.addresses.data());
      f.args.request.valid_table = Address(f.valid.data());
      if (trusted) f.args.reserved |= kvaf::kScatterTrustedInputs;
      for (uint32_t i = 0; i < f.count; ++i)
        f.indices[i] = (i * 1237U + 37U) % table_capacity;
      for (uint32_t count : {0U, 1U, 7U, 255U, 256U, 257U, 1024U, 19U}) {
        f.Next(count);
        for (uint32_t i = 0; i < f.count; ++i)
          f.addresses[f.indices[i]] = f.args.source_base +
              uint64_t((i + f.args.sequence) % f.count) * f.token * 2;
        const auto flushed = aiv_model::address_table_cache_lines;
        const auto simt_reads = aiv_model::simt_metadata_reads;
        f.Run();
        assert(aiv_model::address_table_cache_lines == flushed);
        assert(aiv_model::simt_metadata_reads - simt_reads == count * (trusted ? 3 : 5));
      }
      if (!trusted) {
        f.Next(1);
        f.indices[0] = table_capacity;
        const auto flushed = aiv_model::address_table_cache_lines;
        const auto notified = f.push_requests;
        f.Run(false);
        assert(aiv_model::address_table_cache_lines == flushed);
        assert(f.push_requests == notified);
      }
    }
  // SIMT tile boundaries through the complete CPU-push/scatter flow,
  // including the extra prefix tile for eligible requests.
  for (const bool trusted : {false, true})
      for (uint32_t workers : {1U, 4U, 8U}) {
        Fixture f(8193, 31, workers, KVAF_SEQ_SCATTER, 6);
        f.args.reserved = kvaf::kScatterOptionsV1 |
            (trusted ? kvaf::kScatterTrustedInputs : 0U);
        for (uint32_t count : {0U, 1U, 255U, 256U, 4095U, 4096U, 4097U, 8192U, 8193U, 7U}) {
          f.Next(count);
          const auto calls = aiv_model::simt_calls;
          const auto writes = aiv_model::descriptor_writes;
          const auto prefix = trusted && count > kvaf::cpu_push::kChunkRows ? kvaf::cpu_push::kChunkRows : 0U;
          uint64_t tiles = (prefix + kSimtBuildRows - 1) / kSimtBuildRows;
          const auto groups = (count + 7U) / 8U;
          for (uint32_t core = 0; core < workers; ++core) {
            const auto begin = std::max(uint64_t(prefix), uint64_t(groups) * core / workers * 8);
            const auto end = std::min(uint64_t(count), uint64_t(groups) * (core + 1) / workers * 8);
            if (begin < end) tiles += (end - begin + kSimtBuildRows - 1) / kSimtBuildRows;
          }
          f.Run();
          assert(aiv_model::simt_calls - calls == tiles);
          assert(aiv_model::descriptor_writes - writes == tiles);
        }
      }
  // Trusted callers skip row/valid checks in the builder. Exercise the real
  // launch with changing addresses/prefixes and different table sizes;
  // output guards, done, all-core status and DMA completion remain required.
  for (const auto options : kScatterOptions)
    for (uint32_t workers : {1U, 4U, 8U})
      for (uint32_t capacity : {1024U, 4097U}) {
        Fixture f(capacity, 656, workers, KVAF_SEQ_SCATTER, 6);
        f.args.reserved = options | kvaf::kScatterTrustedInputs;
        for (uint32_t count : {0U, 1U, 7U, 256U, 257U, 1024U, 19U}) {
          f.Next(count);
          f.Run();
        }
      }
  for (const auto options : {0U, kvaf::kScatterOptionsV1,
                             kvaf::kScatterOptionsV1 | kvaf::kScatterPipeline})
    for (uint32_t workers : {1U, 3U, 8U}) {
      Fixture f(19, 656, workers, KVAF_SEQ_SCATTER, 6);
      f.status.resize(workers * 4); // 64-byte statuses + 192-byte V2 records.
      f.args.request.core_status = Address(f.status.data());
      f.args.reserved = kvaf::ScatterWithProfiling(options, true);
      assert(kvaf::ScatterOptionsValid(f.args.reserved));
      assert(kvaf::ScatterWithProfiling(f.args.reserved, false) == options);
      for (uint32_t count : {1U, 19U, 0U, 7U}) {
        f.Next(count); f.Run();
        const auto* profiles = reinterpret_cast<const kvaf::ScatterBatchGetCoreProfile*>(
            f.args.request.core_status + workers * sizeof(KvafSequentialStatus));
        for (uint32_t core = 0; core < workers; ++core) {
          const auto& p = profiles[core];
          assert(p.magic == kvaf::kScatterProfileMagic && p.sequence == f.args.sequence);
          assert(p.start_cycle && p.start_cycle <= p.inputs_ready_cycle);
          assert(p.inputs_ready_cycle <= p.done_cycle && p.done_cycle <= p.finish_cycle);
          const auto requests = uint64_t(count) * (core + 1) / workers - uint64_t(count) * core / workers;
          assert(p.requests == requests && p.bytes == requests * 656);
          assert(p.pipeline == kvaf::ScatterOptionEnabled(f.args.reserved, kvaf::kScatterPipeline));
          assert(bool(p.scatter_cycles) == bool(requests));
          if (!core && count) {
            assert(p.notify_cycle >= p.inputs_ready_cycle && p.done_cycle >= p.notify_cycle);
            assert(p.cpu_magic == kvaf::cpu_push::ProfileToken(f.args.sequence) && p.cpu_comm_ns == 800);
            assert(p.cpu_header_ns == 11 && p.cpu_descriptor_ns == 22 && p.cpu_write_ns == 1000);
          } else assert(!p.cpu_magic && !p.notify_cycle);
        }
      }
    }
  // The producer deliberately withholds final done until payload DMA has
  // started. A whole-batch-only consumer times out; a pipelined consumer must
  // copy only fully published chunks and still wait for final done afterwards.
  for (uint32_t bytes : {31U, 656U, 4096U})
    for (uint32_t workers : {1U, 3U, 8U})
      {
        Fixture f(8193, bytes, workers, KVAF_SEQ_SCATTER, 6);
        f.args.max_polls = 1000000;
        f.args.reserved = kvaf::kScatterOptionsV1 |
            kvaf::kScatterPipeline | kvaf::kScatterTrustedInputs;
        f.hold_done_until_scatter = true;
        for (uint32_t count : {8193U, 1U, 19U, 4097U, 0U, 8192U}) {
          f.Next(count); f.Run();
          if (count) assert(f.copied_before_done && f.push_done && f.read_rows == count);
        }
      }
  for (const auto options : kCheckedScatterOptions)
    for (bool trusted : {false, true}) {
      Fixture f(257, 656, 3, KVAF_SEQ_SCATTER, 6);
      f.args.reserved = options | kvaf::kScatterPipeline |
          (trusted ? kvaf::kScatterTrustedInputs : 0U);
      f.hold_done_until_scatter = true;
      f.Run();
      assert(f.copied_before_done && f.read_rows == 257 && f.push_done);
    }
  // Re-prime partially used rings across uneven Ready shards, including
  // three->two stage layout boundaries, uncached row indices and large rows.
  for (uint32_t bytes : {673U, 1000U, 32768U, 32769U})
    for (uint32_t workers : {1U, 3U}) {
      Fixture f(129, bytes, workers, KVAF_SEQ_SCATTER, 6);
      f.args.reserved |= kvaf::kScatterPipeline;
      f.args.max_polls = 100000;
      f.hold_done_until_scatter = f.hold_next_jetty = f.late_first_jetty = true;
      for (uint32_t i = 0; i < f.count; ++i) f.rows[i] = 2 * ((i * 37U + 11U) % f.count) + 1;
      for (uint32_t count : {1U, 33U, 95U, 97U, 129U, 7U}) {
        std::rotate(f.rows.begin(), f.rows.begin() + 1, f.rows.end());
        f.Next(count); f.Run();
        assert(f.read_rows == count && f.push_done && f.copied_before_done);
      }
    }
  for (bool future : {false, true}) {
    Fixture f(19, 656, 1, KVAF_SEQ_SCATTER, 6);
    f.args.reserved |= kvaf::kScatterPipeline;
    f.stall_ready = !future; f.future_ready = future;
    f.Run(false, !future);
    assert(f.read_rows == (future ? 0U : 15U) && !f.push_done);
    assert(f.status[0].result == (future ? KVAF_SEQ_REMOTE_ERROR : KVAF_SEQ_TIMEOUT));
  }
  // A late queue cannot hold back a ready queue. The fake CPU withholds the
  // next Jetty until this one's shard has been read; whole-chunk or fixed-q0
  // consumers time out. MTE3 must stay live across Ready boundaries as well.
  for (bool late_first : {false, true}) {
    Fixture f(1024, 656, 1, KVAF_SEQ_SCATTER, 6);
    f.args.max_polls = 10000;
    f.args.reserved = kvaf::kScatterOptionsV1 | kvaf::kScatterPipeline;
    f.hold_next_jetty = f.hold_done_until_scatter = true;
    f.late_first_jetty = late_first;
    f.Run();
    assert(f.read_rows == 1024 && f.copied_before_done && f.copied_before_all_ready);
    assert(f.waited_with_output_pending);
  }
  for (uint32_t ready_per_poll : {2U, 4U}) {
    Fixture f(1024, 656, 1, KVAF_SEQ_SCATTER, 6);
    f.status.resize(4);
    f.args.request.core_status = Address(f.status.data());
    f.args.reserved = kvaf::ScatterWithProfiling(kvaf::kScatterOptionsV1 | kvaf::kScatterPipeline, true);
    f.ready_per_poll = ready_per_poll;
    f.late_first_jetty = f.hold_done_until_scatter = true;
    f.Run();
    const auto* profile = reinterpret_cast<const kvaf::ScatterBatchGetCoreProfile*>(f.status.data() + 1);
    // q1 and q2 share a range. q0 and q3 remain separate unless all four
    // queues are visible in the same refresh.
    assert(profile->scatter_chunks == (ready_per_poll == 2 ? 3U : 1U));
    assert(f.read_rows == 1024 && f.copied_before_done && f.push_done);
  }
  {
    Fixture f(19, 656, 1, KVAF_SEQ_SCATTER, 6);
    f.args.reserved |= kvaf::kScatterPipeline;
    f.hold_done_until_scatter = f.error_after_scatter = true;
    f.Run(false, true);
    assert(f.read_rows == 19 && f.push_done && f.copied_before_done);
    assert(f.status[0].result == KVAF_SEQ_REMOTE_ERROR); // Ready is never success.
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
          aiv_model::mte2_source_begin = Address(source.data() + offset);
          aiv_model::mte2_source_end = aiv_model::mte2_source_begin + bytes;
          CopyPayload(Address(source.data() + offset), Address(output.data() + destination_offset), bytes);
          aiv_model::mte2_source_begin = aiv_model::mte2_source_end = 0;
          assert(output == expected);
          assert(aiv_model::mte2.empty() && aiv_model::mte3.empty());
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
      for (uint32_t bytes : {1U, 13U, 656U, 4096U, 16385U, 65535U}) {
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
  for (auto pipeline : {0U, kvaf::kScatterPipeline})
    for (auto trusted : {0U, kvaf::kScatterTrustedInputs})
      for (auto workers : {1U, 3U, 4U}) {
        Fixture head(8193, 13, workers, KVAF_SEQ_SCATTER, 6);
        head.args.reserved = kvaf::kScatterOptionsV1 | pipeline | trusted;
        head.args.max_polls = 100000;
        for (auto count : {8193U, 8193U, 1024U, 1025U, 1031U, 0U, 8193U}) {
          head.Next(count); head.Run();
          const bool expect_head = count > kvaf::cpu_push::kChunkRows && trusted;
          assert(head.head_started_early == expect_head);
          assert(head.head_parts.empty() == !expect_head);
        }
        if (!trusted) {
          const auto requests = head.push_requests;
          head.Next(8193); head.indices.back() = -1; head.Run(false);
          assert(head.push_requests == requests); // Checked failures never publish a prefix.
        }
      }
  {
    // Small negotiated messages cross control post/SQ/CQ boundaries. A fresh
    // list must reach CPU before every commit, including same-count reuse.
    Fixture mailbox(257, 13, 1, KVAF_SEQ_SCATTER, 6);
    mailbox.mailbox.max_write_bytes = 64;
    for (auto count : {257U, 257U, 19U, 1U, 0U, 257U}) {
      mailbox.Next(count); mailbox.Run();
    }
    assert(mailbox.mailbox_posts > 16 && mailbox.mailbox.queue.completed_batch_count == 5);
    Fixture control_error(19, 656, 1, KVAF_SEQ_SCATTER, 6);
    control_error.cq_error = true; control_error.Run(false);
    assert(control_error.mailbox.queue.poisoned && control_error.status[0].result == KVAF_SEQ_CQ_ERROR);
    control_error.Next(19); control_error.cq_error = false; control_error.Run(false);
    Fixture control_timeout(19, 656, 1, KVAF_SEQ_SCATTER, 6);
    control_timeout.mailbox_cq_timeout = true; control_timeout.Run(false);
    assert(control_timeout.mailbox.queue.poisoned && control_timeout.status[0].result == KVAF_SEQ_TIMEOUT);
  }
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
