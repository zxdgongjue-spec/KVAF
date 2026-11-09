#include "cpu_push_scatter.h"
#include "cpu_push_pipeline.h"

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
  std::uint32_t chunks{};
  std::uint8_t* Memory() const { return reinterpret_cast<std::uint8_t*>(args.workspace); }
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
    const auto token = done->token, sequence = token >> 32;
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
  std::uint64_t ReadRangeDone(std::uint32_t w) {
    Barrier();
    auto* ready = reinterpret_cast<volatile Ready*>(Memory() + layout.ready) + w;
    if (profile) ++profile->ready_queries;
    return ChainReadyProgress(chunks, args.sequence, [&](std::uint32_t q) { return ready->jetty[q]; });
  }
  void FinishRange() { Barrier(); }
  bool Copy(Range range) {
    BatchGetProfileScope timer(profile ? &profile->scatter_ns : nullptr);
    const auto bytes = args.request.token_bytes;
    const auto* rows = reinterpret_cast<const std::int32_t*>(args.request.output_indices);
    auto* source = Memory() + layout.staging;
    Barrier();
    const bool copied = CopyMappedRows(reinterpret_cast<std::uint8_t*>(args.output), source, rows, range, bytes);
    if (profile) profile->scatter_copies += range.end - range.begin;
    return copied;
  }
  bool Build(std::uint32_t worker, std::uint32_t* count, Range* range) {
    const auto& r = args.request;
    BatchGetProfileScope timer(profile ? &profile->request_ns : nullptr);
    if (!r.miss_count || !r.address_table || !r.indices || !r.offsets || !r.output_indices ||
        !args.output || r.request_capacity > kMaxRequests) return false;
    const auto active = *reinterpret_cast<const volatile std::int32_t*>(r.miss_count);
    if (active < 0 || static_cast<std::uint32_t>(active) > r.request_capacity) return false;
    *count = active;
    *range = WorkerRange(*count, r.token_bytes, worker, args.workers);
    if (range->begin == range->end) return true;
    const auto* addresses = reinterpret_cast<const std::uint64_t*>(r.address_table);
    const auto* indices = reinterpret_cast<const std::int32_t*>(r.indices);
    const auto* offsets = reinterpret_cast<const std::int32_t*>(r.offsets);
    auto* sources = reinterpret_cast<std::uint64_t*>(Memory() + layout.descriptor);
    for (auto i = range->begin; i < range->end; ++i)
      sources[i] = addresses[indices[i]] + static_cast<std::uint32_t>(offsets[i]);
    if (!worker) {
      auto* header = reinterpret_cast<Request*>(Memory() + layout.request);
      *header = {kMagic, args.sequence, *count, r.token_bytes, args.workers, 0, {}};
    }
    Barrier();
    return true;
  }
  void Notify() {
    BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
    auto* notify = reinterpret_cast<Word*>(Memory() + layout.notify);
    Store(*notify, args.sequence | (profile ? kProfileNotify : 0));
    Barrier();
  }
};
}  // namespace

std::uint32_t ExecuteCpuPushScatter(KernelArgs* args) {
  Layout layout;
  if (!args || !args->workspace || !args->coordination || !args->status || !args->sequence ||
      args->sequence > UINT32_MAX || !args->timeout_ms ||
      !MakeLayout(args->request.token_bytes, args->workers, &layout)) return 1;
  auto& c = *reinterpret_cast<Coordination*>(args->coordination);
  const auto worker = static_cast<std::uint32_t>(__atomic_fetch_add(&c.claimed.value, 1, __ATOMIC_ACQ_REL));
  if (worker >= args->workers) { Fail(c, Result::kInvalidRequest); return 1; }
  BatchGetWorkerProfile measured{};
  auto* profile = args->profile ? &measured : nullptr;
  if (profile) profile->enter_ns = BatchGetProfileNowNs();
  Execution io{*args, layout, c, CounterNow(),
      CounterFrequency() * args->timeout_ms / 1000, profile};
  std::uint32_t count = 0;
  Range range{};
  if (io.Status() == Result::kPending && !io.Build(worker, &count, &range))
    Fail(c, Result::kInvalidRequest);
  if (count && io.Status() == Result::kPending) {
    const auto builders = ActiveWorkers(count, args->request.token_bytes, args->workers);
    // Each slice is visible before arrival; the extra tick releases scatter after notify.
    if (range.begin < range.end) __atomic_fetch_add(&c.built.value, 1, __ATOMIC_ACQ_REL);
    if (!worker) {
      if (io.Wait(c.built, builders, false) && io.Status() == Result::kPending) {
        io.Notify();
        Store(c.built, builders + 1);
      }
    } else {
      io.Wait(c.built, builders + 1, false);
    }
  }
  io.chunks = ChunkCount(range);
  std::uint32_t consumed = 0;
  if (io.Status() == Result::kPending && io.chunks) {
    BatchGetProfileScope timer(profile ? &profile->read_ns : nullptr);
    const auto result = Consume(count, args->request.token_bytes, args->workers, worker,
                                args->sequence, io.start, io.timeout, io, &consumed);
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
    }
    Store(c.finished, 1);
  } else io.Wait(c.finished, 1);
  auto* status = reinterpret_cast<WorkerStatus*>(args->status) + worker;
  *status = {kMagic, args->sequence, Load(c.result) ? Load(c.result) : std::uint64_t(Result::kOk),
             worker, consumed, {}};
  Barrier();
  if (profile) {
    profile->spans = range.end - range.begin;
    profile->bytes = profile->spans * args->request.token_bytes;
    profile->ready_wait_ns = profile->read_ns - profile->scatter_ns;
    profile->exit_ns = BatchGetProfileNowNs();
    auto* destination = reinterpret_cast<BatchGetWorkerProfile*>(args->profile) + worker;
    *destination = measured;
  }
  return status->result == std::uint64_t(Result::kOk) ? 0 : 1;
}
}  // namespace kvaf::cpu_push
