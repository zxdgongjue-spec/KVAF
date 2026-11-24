#include "cpu_push_scatter.h"
#include "cpu_push_pipeline.h"
#include "aicpu_urma/cache.h"

namespace kvaf::cpu_push {
namespace {
void Barrier() {
#if defined(__aarch64__)
  asm volatile("dsb sy" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}
std::uint64_t CounterNow() {
#if defined(__linux__) && defined(__aarch64__)
  std::uint64_t value;
  asm volatile("isb; mrs %0, cntvct_el0" : "=r"(value) :: "memory");
  return value;
#else
  return BatchGetProfileNowNs();
#endif
}
std::uint64_t CounterFrequency() {
#if defined(__linux__) && defined(__aarch64__)
  std::uint64_t value;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(value));
  return value;
#else
  return 1000000000;
#endif
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
  std::uint64_t start, timeout;
  BatchGetWorkerProfile* profile;
  std::uint32_t idle_polls{};
  std::uint8_t* Memory() const { return reinterpret_cast<std::uint8_t*>(args.workspace); }
  void RefreshWindowMemory(const void* address, std::uint64_t bytes) {
    if (!args.window_base || !bytes) return;
    BatchGetProfileScope timer(profile ? &profile->input_cache_ns : nullptr);
    aicpu_urma::FlushUmdkCache(address, bytes);
    Barrier();
  }
  void Relax(bool check_remote = true) {
    // Sample remote errors every 32 stalled polls.
    if (check_remote && (++idle_polls & 31) == 0) {
      const auto result = CompletionStatus();
      if (result != Result::kPending) Fail(coordination, result);
    }
#if defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
  }
  std::uint64_t Now() { return CounterNow(); }
  Result Status() {
    return static_cast<Result>(Load(coordination.result));
  }
  Result CompletionStatus() {
    Barrier();
    auto* done = reinterpret_cast<volatile Completion*>(Memory() + layout.done);
    RefreshWindowMemory(const_cast<const Completion*>(done), sizeof(Completion));
    const auto token = __atomic_load_n(&done->token, __ATOMIC_ACQUIRE), sequence = token >> 32;
    if (sequence > args.sequence) return Result::kSequenceError;
    if (sequence == args.sequence && static_cast<Result>(std::uint32_t(token)) != Result::kOk)
      return static_cast<Result>(std::uint32_t(token));
    return Result::kPending;
  }
  Result Check(bool check_remote = true) {
    auto result = Status();
    if (result != Result::kPending) return result;
    if (check_remote) {
      result = CompletionStatus();
      if (result != Result::kPending) return result;
    }
    return CounterNow() - start >= timeout ? Result::kTimeout : Result::kPending;
  }
  bool Wait(const Word& word, std::uint64_t value, bool check_remote = true) {
    while (Load(word) < value) {
      const auto result = Check(check_remote);
      if (result != Result::kPending) { Fail(coordination, result); return false; }
      Relax(check_remote);
    }
    return true;
  }
  ReadySnapshot ReadReady(std::uint32_t w) {
    Barrier();
    auto* ready = reinterpret_cast<volatile Ready*>(Memory() + layout.ready) + w;
    RefreshWindowMemory(const_cast<const Ready*>(ready), sizeof(Ready));
    if (profile) ++profile->ready_queries;
    ReadySnapshot snapshot;
    for (std::uint32_t q = 0; q < kJettyCount; ++q)
      snapshot[q] = __atomic_load_n(&ready->jetty[q], __ATOMIC_ACQUIRE);
    return snapshot;
  }
  void FinishRange() { Barrier(); }
  bool Copy(Range range) {
    BatchGetProfileScope timer(profile ? &profile->scatter_ns : nullptr);
    const auto bytes = args.request.token_bytes;
    const auto* rows = reinterpret_cast<const std::int32_t*>(args.request.output_indices) + args.window_base;
    auto* source = Memory() + layout.staging;
    Barrier();
    if (args.window_base != 0U && range.begin != range.end) {
      // Ready proves the CPU's WRITE for this shard is complete. Unlike a new
      // launch, window reuse has no CANN cache handoff: invalidate previously
      // consumed staging lines before loading the next payload at those addresses.
      RefreshWindowMemory(source + std::uint64_t{range.begin} * bytes,
                           std::uint64_t{range.end - range.begin} * bytes);
    }
    const bool copied = CopyMappedRows(reinterpret_cast<std::uint8_t*>(args.output), source, rows, range, bytes);
    if (profile) profile->scatter_copies += range.end - range.begin;
    return copied;
  }
  bool Build(std::uint32_t worker, std::uint32_t count, Range* range) {
    const auto& r = args.request;
    BatchGetProfileScope timer(profile ? &profile->request_ns : nullptr);
    if (!r.miss_count || !r.address_table || !r.indices || !r.offsets || !r.output_indices ||
        !args.output || count > args.window_capacity) return false;
    *range = WorkerRange(count, r.token_bytes, worker, args.workers);
    if (range->begin == range->end) return true;
    const auto* addresses = reinterpret_cast<const std::uint64_t*>(r.address_table);
    const auto* indices = reinterpret_cast<const std::int32_t*>(r.indices);
    const auto* offsets = reinterpret_cast<const std::int32_t*>(r.offsets);
    auto* sources = reinterpret_cast<std::uint64_t*>(Memory() + layout.descriptor);
    for (auto i = range->begin; i < range->end; ++i) {
      const auto global = args.window_base + i;
      sources[i] = addresses[indices[global]] + static_cast<std::uint32_t>(offsets[global]);
    }
    if (!worker) {
      auto* header = reinterpret_cast<Request*>(Memory() + layout.request);
      *header = {kMagic, args.sequence, count, r.token_bytes, args.workers, 0, {}};
    }
    Barrier();
    return true;
  }
  void Notify() {
    BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
    // All descriptor writers have crossed built. Publish their stores before
    // the CPU can observe notify; avoid sweeping caller input/output tensors.
    RefreshWindowMemory(Memory() + layout.request, sizeof(Request));
    const auto count = reinterpret_cast<const Request*>(Memory() + layout.request)->count;
    RefreshWindowMemory(Memory() + layout.descriptor, std::uint64_t{count} * sizeof(std::uint64_t));
    auto* notify = reinterpret_cast<Word*>(Memory() + layout.notify);
    Store(*notify, args.sequence | (profile ? kProfileNotify : 0));
    RefreshWindowMemory(notify, sizeof(Word));
    Barrier();
  }
};
}  // namespace

std::uint32_t ExecuteCpuPushScatter(KernelArgs* args) {
  Layout layout;
  if (!args || !args->workspace || !args->coordination || !args->status || !args->timeout_ms ||
      !args->request.miss_count ||
      !MakeLayout(args->request.token_bytes, args->workers, &layout, args->window_capacity)) return 1;
  auto& c = *reinterpret_cast<Coordination*>(args->coordination);
  // Preserve the failed execution's reports across later graph replays.
  if (Load(c.result)) return 1;
  const auto ticket = ClaimBatchGetTicket(&c.claimed.value, args->workers);
  const auto worker = ticket.worker;
  auto current = *args;
  current.sequence = 0;
  args = &current;
  BatchGetWorkerProfile measured{};
  auto* profile = args->profile ? &measured : nullptr;
  if (profile) profile->enter_ns = BatchGetProfileNowNs();
  Execution io{*args, layout, c, CounterNow(),
      CounterFrequency() * args->timeout_ms / 1000, profile};
  // Reserve wire sequences once for the launch; graph/status identities remain
  // the ticket's launch sequence even when each replay needs different windows.
  if (!worker) {
    const auto active = *reinterpret_cast<const volatile std::int32_t*>(args->request.miss_count);
    if (active < 0 || static_cast<std::uint32_t>(active) > args->request.request_capacity)
      Fail(c, Result::kInvalidRequest);
    else {
      const auto count = static_cast<std::uint32_t>(active);
      const auto windows = count ? count / args->window_capacity + (count % args->window_capacity != 0U) : 1U;
      const auto previous = Load(c.window_next);
      if (previous > UINT32_MAX - windows)
        Fail(c, Result::kSequenceError);
      else {
        Store(c.launch_first_window, previous + 1U);
        Store(c.launch_count, count);
        Store(c.window_next, previous + windows);
      }
    }
    Store(c.started, ticket.sequence);
  } else io.Wait(c.started, ticket.sequence, false);
  const auto count = Load(c.launch_count);
  const auto windows = count ? count / args->window_capacity + (count % args->window_capacity != 0U) : 1U;
  const auto first_window = Load(c.launch_first_window);
  std::uint64_t consumed = 0, completed = 0, failed_request = UINT64_MAX;
  for (std::uint64_t window = 0; window < windows && io.Status() == Result::kPending; ++window) {
    current.sequence = first_window + window;
    current.window_base = window * args->window_capacity;
    const auto window_count = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(args->window_capacity, count - current.window_base));
    const auto target = current.sequence * args->workers;
    io.start = CounterNow(); io.idle_polls = 0;
    Range range{};
    if (!io.Build(worker, window_count, &range)) Fail(c, Result::kInvalidRequest);
    // Every worker participates in every window, including empty tail slices.
    __atomic_fetch_add(&c.built.value, 1, __ATOMIC_ACQ_REL);
    if (!worker) {
      if (io.Wait(c.built, target, false) && io.Status() == Result::kPending) {
        if (window_count) io.Notify();
        Store(c.published, current.sequence);
      }
    } else io.Wait(c.published, current.sequence, false);
    std::uint32_t window_consumed = 0;
    if (io.Status() == Result::kPending && range.begin < range.end) {
      BatchGetProfileScope timer(profile ? &profile->read_ns : nullptr);
      const auto result = Consume(window_count, args->request.token_bytes, args->workers, worker,
                                  current.sequence, io.start, io.timeout, io, &window_consumed);
      if (result != Result::kOk) Fail(c, result);
    }
    consumed += window_consumed;
    __atomic_fetch_add(&c.arrivals.value, 1, __ATOMIC_ACQ_REL);
    if (!worker) {
      if (io.Wait(c.arrivals, target) && Load(c.result) == 0 && window_count) {
        BatchGetProfileScope timer(profile ? &profile->ack_ns : nullptr);
        for (;;) {
          const auto result = io.Check();
          if (result != Result::kPending) { Fail(c, result); break; }
          const auto token = __atomic_load_n(
              &reinterpret_cast<volatile Completion*>(io.Memory() + layout.done)->token, __ATOMIC_ACQUIRE);
          if ((token >> 32) == current.sequence) break;
          io.Relax();
        }
      }
      Store(c.finished, current.sequence);
    } else io.Wait(c.finished, current.sequence);
    if (io.Status() != Result::kPending) {
      failed_request = current.window_base + range.begin;
      break;
    }
    completed += range.end - range.begin;
    if (profile) profile->spans += range.end - range.begin;
  }
  auto* status = reinterpret_cast<WorkerStatus*>(args->status) + worker;
  *status = {kMagic, ticket.sequence, Load(c.result) ? Load(c.result) : std::uint64_t(Result::kOk),
             worker, consumed, completed, failed_request, 0};
  Barrier();
  if (profile) {
    profile->bytes = profile->spans * args->request.token_bytes;
    profile->ready_wait_ns = profile->read_ns - profile->scatter_ns;
    profile->exit_ns = BatchGetProfileNowNs();
    auto* destination = reinterpret_cast<BatchGetWorkerProfile*>(args->profile) + worker;
    *destination = measured;
  }
  return status->result == std::uint64_t(Result::kOk) ? 0 : 1;
}
}  // namespace kvaf::cpu_push
