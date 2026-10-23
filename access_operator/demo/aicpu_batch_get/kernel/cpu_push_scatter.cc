#include "cpu_push_scatter.h"
#include "cpu_push_pipeline.h"
#include "aicpu_urma/cache.h"

#include <cstring>
#if defined(__linux__) && defined(__aarch64__) && !defined(KVAF_CPU_PUSH_HOST_TEST)
#include <securec.h>
#endif

namespace kvaf::cpu_push {
namespace {
void Barrier() {
#if defined(__aarch64__)
  asm volatile("dsb sy" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}
void Visible(const void* p, std::uint64_t bytes) {
#if !defined(KVAF_CPU_PUSH_HOST_TEST)
  aicpu_urma::FlushUmdkCache(p, bytes);
#else
  (void)p;
  (void)bytes;
#endif
  Barrier();
}
std::uint64_t Load(const Word& w) { return __atomic_load_n(&w.value, __ATOMIC_ACQUIRE); }
void Store(Word& w, std::uint64_t value) { __atomic_store_n(&w.value, value, __ATOMIC_RELEASE); }
void Fail(Coordination& c, Result result) {
  std::uint64_t pending = 0;
  __atomic_compare_exchange_n(&c.result.value, &pending, static_cast<std::uint64_t>(result),
                              false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
struct Execution {
  const KernelArgs& args;
  Layout layout;
  Coordination& coordination;
  std::uint32_t worker;
  std::uint64_t deadline;
  BatchGetWorkerProfile* profile;
  std::uint8_t* Memory() const { return reinterpret_cast<std::uint8_t*>(args.workspace); }
  void Relax() {
#if defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
  }
  Result Check() {
    auto result = static_cast<Result>(Load(coordination.result));
    if (result != Result::kPending) return result;
    if (BatchGetProfileNowNs() >= deadline) return Result::kTimeout;
    auto* done = reinterpret_cast<volatile Completion*>(Memory() + layout.done);
    Visible(const_cast<const Completion*>(done), sizeof(Completion));
    const auto token = done->token, sequence = token >> 32;
    if (sequence > args.sequence) return Result::kSequenceError;
    if (sequence == args.sequence && static_cast<Result>(std::uint32_t(token)) != Result::kOk)
      return static_cast<Result>(std::uint32_t(token));
    return Result::kPending;
  }
  bool Wait(const Word& word, std::uint64_t value) {
    while (Load(word) < value) {
      const auto result = Check();
      if (result != Result::kPending) { Fail(coordination, result); return false; }
      Relax();
    }
    return true;
  }
  std::uint64_t ReadReady(std::uint32_t q) {
    auto* ready = reinterpret_cast<Ready*>(Memory() + layout.ready) + worker;
    // The line is DMA-owned and is never dirtied by AICPU during execution.
    if (!q) {
      Visible(ready, sizeof(*ready));
      if (profile) ++profile->ready_queries;
    }
    return reinterpret_cast<volatile Ready*>(ready)->jetty[q];
  }
  bool Copy(Range range) {
    BatchGetProfileScope timer(profile ? &profile->scatter_ns : nullptr);
    const auto bytes = args.request.token_bytes;
    const auto* rows = reinterpret_cast<const std::int32_t*>(args.request.output_indices);
    auto* source = Memory() + layout.staging;
    Visible(source + std::uint64_t{range.begin} * bytes, std::uint64_t{range.end - range.begin} * bytes);
    for (auto i = range.begin; i < range.end; ++i) {
      auto* target = reinterpret_cast<std::uint8_t*>(args.output) + std::uint64_t(rows[i]) * bytes;
#if defined(__linux__) && defined(__aarch64__) && !defined(KVAF_CPU_PUSH_HOST_TEST)
      if (memcpy_sp(target, bytes, source + std::uint64_t{i} * bytes, bytes) != EOK) return false;
#else
      std::memcpy(target, source + std::uint64_t{i} * bytes, bytes);
#endif
    }
    Barrier();
    return true;
  }
  bool Build(std::uint32_t* count) {
    const auto& r = args.request;
    BatchGetProfileScope timer(profile ? &profile->request_ns : nullptr);
    if (!r.miss_count || !r.address_table || !r.indices || !r.offsets || !r.output_indices ||
        !args.output || r.request_capacity > kMaxRequests) return false;
    Visible(reinterpret_cast<const void*>(r.miss_count), 4);
    const auto active = *reinterpret_cast<const volatile std::int32_t*>(r.miss_count);
    if (active < 0 || static_cast<std::uint32_t>(active) > r.request_capacity) return false;
    *count = active;
    {
      BatchGetProfileScope cache(profile ? &profile->input_cache_ns : nullptr);
      Visible(reinterpret_cast<const void*>(r.address_table), r.table_capacity * 8ULL);
      for (auto p : {r.indices, r.offsets, r.output_indices})
        Visible(reinterpret_cast<const void*>(p), *count * 4ULL);
    }
    const auto* addresses = reinterpret_cast<const std::uint64_t*>(r.address_table);
    const auto* indices = reinterpret_cast<const std::int32_t*>(r.indices);
    const auto* offsets = reinterpret_cast<const std::int32_t*>(r.offsets);
    const auto* rows = reinterpret_cast<const std::int32_t*>(r.output_indices);
    auto* sources = reinterpret_cast<peer::GatherSpan*>(Memory() + layout.descriptor);
    for (std::uint32_t i = 0; i < *count; ++i) {
      if (indices[i] < 0 || std::uint32_t(indices[i]) >= r.table_capacity || offsets[i] < 0 ||
          rows[i] < 0 || std::uint64_t(rows[i]) >= args.output_bytes / r.token_bytes ||
          addresses[indices[i]] > UINT64_MAX - std::uint32_t(offsets[i])) return false;
      sources[i] = {addresses[indices[i]] + std::uint32_t(offsets[i]), r.token_bytes, 0};
    }
    auto* header = reinterpret_cast<Request*>(Memory() + layout.request);
    *header = {kMagic, args.sequence, *count, r.token_bytes, args.workers, 0, {}};
    {
      BatchGetProfileScope cache(profile ? &profile->descriptor_cache_ns : nullptr);
      Visible(sources, *count * sizeof(*sources));
      Visible(header, sizeof(*header));
    }
    // Purge prior device writes before any worker performs partial-line stores.
    // Output cache maintenance is centralized across all scatter workers.
    if (*count) {
      BatchGetProfileScope cache(profile ? &profile->output_cache_ns : nullptr);
      Visible(reinterpret_cast<const void*>(args.output), args.output_bytes);
    }
    return true;
  }
};
}  // namespace

std::uint32_t ExecuteCpuPushScatter(KernelArgs* args) {
  Layout layout;
  if (!args || !args->workspace || !args->coordination || !args->status || !args->sequence ||
      args->sequence > UINT32_MAX || !args->timeout_ms ||
      !MakeLayout(args->request.token_bytes, args->workers, &layout)) return 1;
  auto& c = *reinterpret_cast<Coordination*>(args->coordination);
  // Host reset is DMA. The previous launch cleaned its coordination writes;
  // invalidate any clean cached copy before this launch starts using atomics.
  Visible(&c, sizeof(c));
  const auto worker = static_cast<std::uint32_t>(__atomic_fetch_add(&c.claimed.value, 1, __ATOMIC_ACQ_REL));
  if (worker >= args->workers) { Fail(c, Result::kInvalidRequest); return 1; }
  BatchGetWorkerProfile measured{};
  auto* profile = args->profile ? &measured : nullptr;
  if (profile) profile->enter_ns = BatchGetProfileNowNs();
  Execution io{*args, layout, c, worker,
      BatchGetProfileNowNs() + std::uint64_t{args->timeout_ms} * 1000000, profile};
  std::uint32_t count = 0;
  if (!worker) {
    if (!io.Build(&count)) Fail(c, Result::kInvalidRequest);
    if (Load(c.result) == 0 && count) {
      BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
      auto* notify = reinterpret_cast<Word*>(io.Memory() + layout.notify);
      notify->value = args->sequence;
      Visible(notify, sizeof(*notify));
    }
    Store(c.built, 1);
  } else if (io.Wait(c.built, 1)) {
    // AICPU workers share coherent normal memory; Build publishes this header.
    count = reinterpret_cast<const Request*>(io.Memory())->count;
  }
  const auto range = WorkerRange(count, args->request.token_bytes, worker, args->workers);
  std::uint32_t consumed = 0;
  if (Load(c.result) == 0 && count) {
    BatchGetProfileScope timer(profile ? &profile->read_ns : nullptr);
    const auto result = Consume(range, args->sequence, io, &consumed);
    if (result != Result::kOk) Fail(c, result);
  }
  __atomic_fetch_add(&c.arrivals.value, 1, __ATOMIC_ACQ_REL);
  if (!worker) {
    if (io.Wait(c.arrivals, args->workers) && Load(c.result) == 0 && count) {
      {
        BatchGetProfileScope timer(profile ? &profile->ack_ns : nullptr);
        for (;;) {
          const auto result = io.Check();
          if (result != Result::kPending) { Fail(c, result); break; }
          const auto token = reinterpret_cast<volatile Completion*>(io.Memory() + layout.done)->token;
          if ((token >> 32) == args->sequence) break;
          io.Relax();
        }
      }
      if (Load(c.result) == 0) {
        BatchGetProfileScope timer(profile ? &profile->output_cache_ns : nullptr);
        Visible(reinterpret_cast<const void*>(args->output), args->output_bytes);
      }
    }
    Store(c.finished, 1);
  } else io.Wait(c.finished, 1);
  auto* status = reinterpret_cast<WorkerStatus*>(args->status) + worker;
  *status = {kMagic, args->sequence, Load(c.result) ? Load(c.result) : std::uint64_t(Result::kOk),
             worker, consumed, {}};
  Visible(status, sizeof(*status));
  if (profile) {
    profile->spans = range.end - range.begin;
    profile->bytes = profile->spans * args->request.token_bytes;
    profile->ready_wait_ns = profile->read_ns - profile->scatter_ns;
    profile->exit_ns = BatchGetProfileNowNs();
    auto* destination = reinterpret_cast<BatchGetWorkerProfile*>(args->profile) + worker;
    *destination = measured;
    Visible(destination, sizeof(*destination));
  }
  // No coordination writes remain after successful finalization. Publish dirty
  // CPU lines before Host is allowed to reset this allocation for the next launch.
  Visible(&c, sizeof(c));
  return status->result == std::uint64_t(Result::kOk) ? 0 : 1;
}
}  // namespace kvaf::cpu_push
