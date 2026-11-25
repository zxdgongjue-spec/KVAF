#include "cpu_push_scatter.h"
#include "cpu_push_pipeline.h"
#include "get/cpu_sgl_push/kernel/aicpu_mailbox.h"

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
  const OutputTarget* output_target;
  Layout layout;
  Coordination& coordination;
  std::uint64_t start, timeout;
  BatchGetWorkerProfile* profile;
  bool pipeline{};
  bool chunked{};
  const MailboxLaunch* mailbox{};
  std::uint32_t idle_polls{};
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
  ReadySnapshot ReadReady(std::uint32_t w) {
    Barrier();
    auto* ready = reinterpret_cast<volatile Ready*>(Memory() + layout.ready) + w;
    if (profile) ++profile->ready_queries;
    ReadySnapshot snapshot;
    for (std::uint32_t q = 0; q < kJettyCount; ++q) snapshot[q] = ready->jetty[q];
    return snapshot;
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
  bool FillSources(Range range) {
    const auto& r = args.request;
    const auto* addresses = reinterpret_cast<const std::uint64_t*>(r.address_table);
    const auto* indices = reinterpret_cast<const std::int32_t*>(r.indices);
    const auto* offsets = reinterpret_cast<const std::int32_t*>(r.offsets);
    const auto* rows = reinterpret_cast<const std::int32_t*>(r.output_indices);
    auto* sources = reinterpret_cast<std::uint64_t*>(Memory() + layout.descriptor);
    for (auto i = range.begin; i < range.end; ++i) {
      if (output_target && rows[i] != static_cast<std::int32_t>(i)) return false;
      sources[i] = addresses[indices[i]] + static_cast<std::uint32_t>(offsets[i]);
    }
    return true;
  }
  bool Build(std::uint32_t worker, std::uint32_t* count, Range* range) {
    const auto& r = args.request;
    BatchGetProfileScope timer(profile ? &profile->request_ns : nullptr);
    if (!r.miss_count || !r.address_table || !r.indices || !r.offsets || !r.output_indices ||
        !args.output || r.request_capacity > kMaxRequests) return false;
    const auto active = *reinterpret_cast<const volatile std::int32_t*>(r.miss_count);
    if (active < 0 || static_cast<std::uint32_t>(active) > r.request_capacity) return false;
    const bool direct = output_target != nullptr;
    if (direct && std::uint64_t{static_cast<std::uint32_t>(active)} * r.token_bytes > args.output_bytes)
      return false;
    *count = active;
    *range = WorkerRange(*count, r.token_bytes, worker, args.workers);
    // Baseline validates the whole mapping before notify. Pipeline validates
    // the head before notify, then publishes suffix success/failure separately.
    const auto head = std::min(*count, kPipelineHeadRows);
    const auto build_range = !pipeline ? *range : !worker ? Range{0, head} :
        Range{std::max(range->begin, head), std::max(range->end, head)};
    if (!chunked && !FillSources(build_range)) return false;
    if (!worker) {
      auto* header = reinterpret_cast<Request*>(Memory() + layout.request);
      *header = {kMagic, args.sequence, *count, r.token_bytes, args.workers, direct ? kDirectOutput : kScatterOutput, {}};
      if (pipeline) header->reserved_words[0] = head;
      if (chunked) { header->mode = kChunkedOutput; header->reserved_words[0] = kChunkRows; }
      if (direct)
        *reinterpret_cast<OutputTarget*>(Memory() + layout.output_target) = *output_target;
    }
    Barrier();
    return true;
  }
  void Notify(std::uint32_t count = 0) {
    BatchGetProfileScope timer(profile ? &profile->notify_ns : nullptr);
    auto* notify = reinterpret_cast<Word*>(Memory() + layout.notify);
    Store(*notify, args.sequence | (profile ? kProfileNotify : 0));
    Barrier();
    if (mailbox && PostDescriptorMailbox(*mailbox, args, layout, count, false)) Fail(coordination, Result::kTransportError);
  }
};
}  // namespace

static std::uint32_t ExecuteCpuPush(KernelArgs* args, const OutputTarget* output_target,
                                    bool pipeline = false, bool chunked = false, const MailboxLaunch* mailbox = nullptr) {
  Layout layout;
  if (!args || !args->workspace || !args->coordination || !args->status || !args->timeout_ms ||
      !MakeLayout(args->request.token_bytes, args->workers, &layout, output_target != nullptr)) return 1;
  auto& c = *reinterpret_cast<Coordination*>(args->coordination);
  // Preserve the failed execution's reports across later graph replays.
  if (Load(c.result)) return 1;
  const auto ticket = ClaimBatchGetTicket(&c.claimed.value, args->workers);
  const auto worker = ticket.worker;
  auto current = *args;
  current.sequence = ticket.sequence;
  args = &current;
  if (args->sequence > UINT32_MAX) Fail(c, Result::kSequenceError);
  const auto target = args->sequence * args->workers;
  BatchGetWorkerProfile measured{};
  auto* profile = args->profile ? &measured : nullptr;
  if (profile) profile->enter_ns = BatchGetProfileNowNs();
  Execution io{*args, output_target, layout, c, CounterNow(),
      CounterFrequency() * args->timeout_ms / 1000, profile, pipeline, chunked, mailbox};
  std::uint32_t count = 0;
  Range range{};
  if (io.Status() == Result::kPending && !io.Build(worker, &count, &range))
    Fail(c, Result::kInvalidRequest);
  bool head_notified = false;
  if (chunked) {
    range = {};
    const auto chunks = (count + kChunkRows - 1) / kChunkRows;
    auto* progress = reinterpret_cast<Word*>(io.Memory() + layout.ready) + worker;
    std::uint32_t completed = 0;
    for (auto chunk = worker; chunk < chunks; chunk += args->workers) {
      const Range part{chunk * kChunkRows, std::min(count, (chunk + 1) * kChunkRows)};
      range.end += part.end - part.begin;
      if (io.Status() != Result::kPending) continue;
      {
        BatchGetProfileScope timer(profile ? &profile->request_ns : nullptr);
        if (!io.FillSources(part)) Fail(c, Result::kInvalidRequest);
        Barrier();
      }
      if (io.Status() != Result::kPending) continue;
      Store(*progress, Progress(args->sequence, ++completed));
      Barrier();
      if (!worker && !head_notified) {
        io.Notify(count);
        head_notified = true;
        Store(c.published, args->sequence);
      }
    }
    if (io.Status() != Result::kPending)
      Store(*progress, Progress(args->sequence, kChunkFailed | std::uint32_t(io.Status())));
    else if (!completed) Store(*progress, Progress(args->sequence, 0));
    Barrier();
  }
  if (pipeline && !worker && count && io.Status() == Result::kPending) {
    io.Notify(count);
    head_notified = true;
    Store(c.published, args->sequence);
    BatchGetProfileScope timer(profile ? &profile->request_ns : nullptr);
    if (!io.FillSources({std::max(range.begin, std::min(count, kPipelineHeadRows)), range.end}))
      Fail(c, Result::kInvalidRequest);
    Barrier();
  }
  // Empty slices participate as well, keeping the target independent of shape.
  __atomic_fetch_add(&c.built.value, 1, __ATOMIC_ACQ_REL);
  if (!worker) {
    const bool built = io.Wait(c.built, target, false);
    if (head_notified && !chunked) {
      const auto result = io.Status() == Result::kPending && built ? Result::kOk : io.Status();
      auto* suffix = reinterpret_cast<Word*>(io.Memory() + layout.notify);
      if (mailbox && RetireDescriptorMailbox(*mailbox)) Fail(c, Result::kTransportError);
      const auto committed = io.Status() == Result::kPending ? result : io.Status();
      __atomic_store_n(&suffix->reserved[0], Progress(args->sequence, std::uint32_t(committed)), __ATOMIC_RELEASE);
      Barrier();
      if (mailbox && count > std::min(count, kPipelineHeadRows) &&
          (PostDescriptorMailbox(*mailbox, *args, layout, count, true, committed != Result::kOk) ||
           RetireDescriptorMailbox(*mailbox))) Fail(c, Result::kTransportError);
    }
    if (built && io.Status() == Result::kPending) {
      if (count && !pipeline && !chunked) io.Notify(count);
      Store(c.published, args->sequence);
    }
  } else {
    io.Wait(c.published, args->sequence, false);
  }
  std::uint32_t consumed = 0;
  if (io.Status() == Result::kPending && range.begin < range.end && !output_target) {
    BatchGetProfileScope timer(profile ? &profile->read_ns : nullptr);
    const auto result = Consume(count, args->request.token_bytes, args->workers, worker,
                                args->sequence, io.start, io.timeout, io, &consumed);
    if (result != Result::kOk) Fail(c, result);
  }
  __atomic_fetch_add(&c.arrivals.value, 1, __ATOMIC_ACQ_REL);
  if (!worker) {
    const bool arrived = io.Wait(c.arrivals, target);
    if ((head_notified || (arrived && Load(c.result) == 0)) && count) {
      {
        BatchGetProfileScope timer(profile ? &profile->ack_ns : nullptr);
        for (;;) {
          // A pipelined head may already be in flight after suffix validation
          // fails. Ignore the local failure until CPU publishes drained done.
          const auto result = head_notified ? (io.Now() - io.start >= io.timeout ? Result::kTimeout : Result::kPending) : io.Check();
          if (result != Result::kPending) { Fail(c, result); break; }
          Barrier();
          const auto token = reinterpret_cast<volatile Completion*>(io.Memory() + layout.done)->token;
          if ((token >> 32) > args->sequence) { Fail(c, Result::kSequenceError); break; }
          if ((token >> 32) == args->sequence) {
            if (std::uint32_t(token) != std::uint32_t(Result::kOk))
              Fail(c, static_cast<Result>(std::uint32_t(token)));
            break;
          }
          io.Relax(!head_notified);
        }
      }
    }
    Store(c.finished, args->sequence);
  } else io.Wait(c.finished, args->sequence);
  // Order observations of done against subsequent output reads on this core.
  Barrier();
  auto* status = reinterpret_cast<WorkerStatus*>(args->status) + worker;
  *status = {kMagic, args->sequence, Load(c.result) ? Load(c.result) : std::uint64_t(Result::kOk),
             worker, consumed, {}};
  Barrier();
  if (profile) {
    if (output_target && !worker && count && status->result == std::uint64_t(Result::kOk)) {
      const auto& sample = reinterpret_cast<const volatile Completion*>(io.Memory() + layout.done)->cpu_sgl_write;
      profile->cpu_sgl_write = {sample.elapsed_ns, sample.data_wrs, sample.waves, sample.tail_cqes};
    }
    profile->spans = range.end - range.begin;
    profile->bytes = profile->spans * args->request.token_bytes;
    profile->ready_wait_ns = profile->read_ns - profile->scatter_ns;
    profile->exit_ns = BatchGetProfileNowNs();
    auto* destination = reinterpret_cast<BatchGetWorkerProfile*>(args->profile) + worker;
    *destination = measured;
  }
  return status->result == std::uint64_t(Result::kOk) ? 0 : 1;
}
std::uint32_t ExecuteCpuPushScatter(KernelArgs* args) { return ExecuteCpuPush(args, nullptr); }
std::uint32_t ExecuteCpuSglWrite(CpuSglWriteArgs* args) {
  return args ? ExecuteCpuPush(&args->execution, &args->output_target,
      (args->execution.sequence & kPipelineLaunch) != 0, (args->execution.sequence & kChunkPipelineLaunch) != 0) : 1;
}
}  // namespace kvaf::cpu_push

namespace kvaf::cpu_push {
std::uint32_t ExecuteCpuSglPush(CpuSglPushArgs* args) {
  return args ? ExecuteCpuPush(&args->write.execution, &args->write.output_target, true, false, &args->mailbox) : 1;
}
}  // namespace kvaf::cpu_push
