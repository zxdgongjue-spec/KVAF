#define KVAF_AICPU_TEST_DEVICE_KERNEL 1
#include "abi.h"
#include "pipeline.h"
#include <cstdint>

#if !defined(__aarch64__)
#error "Cross-compile this kernel for AArch64"
#endif
namespace {
using namespace kvaf::aicpu_test;
void Barrier() { asm volatile("dsb sy" ::: "memory"); }
std::uint64_t Load(const CoordinationWord& word) { return __atomic_load_n(&word.value, __ATOMIC_ACQUIRE); }
void Store(CoordinationWord& word, std::uint64_t value) { __atomic_store_n(&word.value, value, __ATOMIC_RELEASE); }
std::uint64_t Add(CoordinationWord& word) { return __atomic_fetch_add(&word.value, 1, __ATOMIC_ACQ_REL); }
Result Status(const MultiBlockCoordination& c) { return static_cast<Result>(Load(c.status)); }
void Fail(MultiBlockCoordination& c, Result status, std::uint64_t sequence) {
  std::uint64_t expected = static_cast<std::uint64_t>(Result::kPending);
  if (__atomic_compare_exchange_n(&c.status.value, &expected, static_cast<std::uint64_t>(status),
                                  false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) Store(c.failure_sequence, sequence);
}
void Minimum(CoordinationWord& word, std::uint64_t value) {
  auto previous = Load(word);
  while (value < previous && !__atomic_compare_exchange_n(&word.value, &previous, value,
      false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {}
}
struct DeviceIO {
  std::uint8_t* region;
  Config config;
  Layout layout;
  std::uint64_t sequence{};
  std::uint8_t* Output() { return region + RegisteredOutputOffset(layout); }
  std::uint64_t Now() { std::uint64_t v; asm volatile("isb; mrs %0, cntvct_el0" : "=r"(v) :: "memory"); return v; }
  std::uint64_t Frequency() { std::uint64_t v; asm volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v; }
  void PublishRequest(std::uint64_t seq) {
    reinterpret_cast<volatile Control*>(region + kRequestOffset)->sequence = seq;
    Barrier();
  }
  std::uint64_t ReadDone() {
    Barrier();
    return reinterpret_cast<volatile Control*>(region + kDoneOffset)->sequence;
  }
  std::uint64_t ReadRangeDone(std::uint32_t worker) {
    const auto offset = ChainReadyOffset(layout, worker);
    Barrier();
    auto* words = reinterpret_cast<volatile std::uint64_t*>(region + offset);
    return ChainReadyProgress(config, worker, sequence, [&](std::uint32_t q) { return words[q]; });
  }
  void PublishOutputReady(std::uint64_t seq) {
    reinterpret_cast<volatile Control*>(region + kOutputReadyOffset)->sequence = seq;
    Barrier();
  }
  std::uint64_t ReadOutputVerified() {
    Barrier();
    return reinterpret_cast<volatile Control*>(region + kOutputVerifiedOffset)->sequence;
  }
  void Relax() { asm volatile("yield" ::: "memory"); }
};
bool WaitFor(DeviceIO& io, MultiBlockCoordination& c, const CoordinationWord& word,
             std::uint64_t target, std::uint64_t start, std::uint64_t timeout, std::uint64_t sequence) {
  for (;;) {
    if (Load(word) >= target) return true;
    if (Status(c) != Result::kPending) return false;
    if (io.Now() - start >= timeout) { Fail(c, Result::kTimeout, sequence); return false; }
    io.Relax();
  }
}
struct RangeIO {
  DeviceIO& device;
  MultiBlockCoordination& coordination;
  Result Status() { return ::Status(coordination); }
  std::uint64_t Now() { return device.Now(); }
  std::uint64_t ReadDone() { return device.ReadDone(); }
  std::uint64_t ReadRangeDone(std::uint32_t worker) { return device.ReadRangeDone(worker); }
  void RangeReady(std::uint64_t now) { Minimum(coordination.ready_tick, now); }
  void Relax() { device.Relax(); }
  void CopySpan(RowRange r) {
    Barrier();
    if (!CopyScatterRows(device.Output() + r.begin * device.layout.stride,
        device.region + kStagingOffset + r.begin * device.config.row_bytes,
        r.end - r.begin, device.config.row_bytes, device.layout.stride))
      Fail(coordination, Result::kCopyError, device.sequence);
  }
  void FinishRange() { Barrier(); }
};
Result Run(DeviceIO& io, Sample* samples, KernelResult* result, MultiBlockCoordination& c) {
  const auto workers = ActiveThreads(io.config);
  if (c.magic != kMagic || c.revision != kKernelRevision || c.workers != workers) {
    *result = {}; result->status = Result::kInvalidArgs; result->kernel_revision = kKernelRevision;
    Barrier(); return Result::kInvalidArgs;
  }
  const auto worker = static_cast<std::uint32_t>(Add(c.claimed));
  if (worker >= workers) { Fail(c, Result::kInvalidArgs, 0); return Result::kInvalidArgs; }
  const auto frequency = io.Frequency();
  const auto timeout = frequency * io.config.timeout_ms / 1000;
  if (!worker) {
    *result = {};
    result->active_threads = workers; result->kernel_revision = kKernelRevision;
    result->counter_frequency = frequency;
    if (!timeout) Fail(c, Result::kInvalidArgs, 0);
    Store(c.initialized, 1);
  }
  const auto startup = io.Now();
  bool initialized = WaitFor(io, c, c.initialized, 1, startup, frequency, 0);
  if (initialized) {
    Add(c.ready);
    initialized = WaitFor(io, c, c.ready, workers, startup, frequency, 0);
  }
  for (std::uint64_t sequence = 1; initialized && sequence <= io.layout.rounds; ++sequence) {
    if (!worker) {
      result->failed_sequence = sequence;
      const auto start = io.Now(); Store(c.request_start, start); Store(c.ready_tick, UINT64_MAX);
      io.PublishRequest(sequence); Store(c.request_sequence, sequence);
    } else if (!WaitFor(io, c, c.request_sequence, sequence, io.Now(), timeout, sequence)) break;
    if (Status(c) != Result::kPending) break;
    io.sequence = sequence;
    const auto start = Load(c.request_start);
    RangeIO range{io, c};
    const auto status = RunPipelineChunks(io.config, worker, sequence, start, timeout, range);
    if (status != Result::kOk) { Fail(c, status, sequence); break; }
    if (Add(c.arrivals) + 1 == sequence * workers) Store(c.scatter_done, sequence);
    if (!worker) {
      if (!WaitFor(io, c, c.scatter_done, sequence, start, timeout, sequence)) break;
      const auto done = WaitForProducerDone(sequence, start, timeout, range);
      if (done != Result::kOk) { Fail(c, done, sequence); break; }
      const auto end = io.Now(), ready = Load(c.ready_tick);
      std::uint64_t verification = 0;
      if (sequence == io.layout.rounds) {
        const auto verify_start = io.Now();
        const auto verified = VerifyOutputViaUrma(sequence, timeout, io);
        if (verified != Result::kOk) { Fail(c, verified, sequence); break; }
        verification = io.Now() - verify_start;
      }
      if (sequence > io.config.warmup)
        samples[sequence - io.config.warmup - 1] = {end - start, ready - start, end - ready, 0, verification};
      result->completed_rounds = sequence; Store(c.round_complete, sequence);
    } else if (!WaitFor(io, c, c.round_complete, sequence, io.Now(), timeout, sequence)) break;
  }
  Add(c.exited);
  if (worker) return Status(c) == Result::kPending ? Result::kOk : Status(c);
  WaitFor(io, c, c.exited, workers, io.Now(), frequency, result->failed_sequence);
  auto status = Status(c); if (status == Result::kPending) status = Result::kOk;
  result->status = status;
  result->failed_sequence = status == Result::kOk ? 0 : Load(c.failure_sequence);
  Barrier(); return status;
}
}
extern "C" __attribute__((visibility("default")))
std::uint32_t KvafAicpuBatchGetV25(kvaf::aicpu_test::KernelArgs* args) {
  if (!args || !args->region || !args->samples || !args->result || !args->coordination) return 1;
  Layout layout{};
  auto* result = reinterpret_cast<KernelResult*>(args->result);
  if (!MakeLayout(args->config, &layout) || args->region > UINT64_MAX - layout.region_bytes) {
    *result = {}; result->status = Result::kInvalidArgs; result->kernel_revision = kKernelRevision;
    Barrier(); return 1;
  }
  DeviceIO io{reinterpret_cast<std::uint8_t*>(args->region), args->config, layout};
  return Run(io, reinterpret_cast<Sample*>(args->samples), result,
      *reinterpret_cast<MultiBlockCoordination*>(args->coordination)) == Result::kOk ? 0 : 1;
}
