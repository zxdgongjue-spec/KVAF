#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "device_loop.h"
#include "scatter_pool.h"
#include "scatter_profile.h"

#include <cerrno>
#include <cstring>
#include <sched.h>
#include <time.h>

#if !defined(__aarch64__)
#error "This AICPU kernel must be cross-compiled for aarch64"
#endif

namespace {
using namespace kvaf::aicpu_test;

void Barrier() { asm volatile("dsb sy" ::: "memory"); }
void Clean(const void* data, std::uint64_t bytes) {
  auto begin = reinterpret_cast<std::uintptr_t>(data) & ~std::uintptr_t{63};
  const auto end = reinterpret_cast<std::uintptr_t>(data) + bytes;
  for (; begin < end; begin += 64) asm volatile("dc cvac, %0" :: "r"(begin) : "memory");
  Barrier();
}
void Invalidate(const void* data, std::uint64_t bytes) {
  auto begin = reinterpret_cast<std::uintptr_t>(data) & ~std::uintptr_t{63};
  const auto end = reinterpret_cast<std::uintptr_t>(data) + bytes;
  // DC IVAC is privileged and faults at EL0. AICPU user kernels use CIVAC
  // (subject to platform cache-maintenance permission), as in reference kernels.
  // DMA owns these input lines. Do not dirty them before invalidating, or the
  // clean portion could overwrite the CPU's newer RDMA data.
  for (; begin < end; begin += 64) asm volatile("dc civac, %0" :: "r"(begin) : "memory");
  Barrier();
}

struct DeviceIO {
  std::uint8_t* region;
  std::uint8_t* output;
  Config config;
  Layout layout;
  ScatterProfile* profile;
  ScatterPool pool;
  enum class Job { kScatter, kProfileInit, kProfileFinish };
  Job job{Job::kScatter};
  std::uint64_t scatter_round{}, dispatch_tick{};

  static void ScatterWorker(void* context, std::uint32_t worker) {
    auto& io = *static_cast<DeviceIO*>(context);
    if (io.job == Job::kProfileInit) { io.InitializeProfileWorker(worker); return; }
    if (io.job == Job::kProfileFinish) { io.FinishProfileWorker(worker); return; }
    io.ScatterRows(worker, io.dispatch_tick,
                   io.profile && io.scatter_round > io.config.warmup);
  }

  void InitializeProfileWorker(std::uint32_t worker) {
    if (!profile) return;
    const auto range = ScatterRange(config, worker, ActiveThreads(config));
    auto& p = profile->worker[worker];
    p = {};
    p.row_begin = range.begin; p.row_end = range.end;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    p.affinity_capacity = std::min<unsigned>(CPU_SETSIZE, kProfileCpuLimit);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) p.affinity_errno = errno;
    else for (unsigned cpu = 0; cpu < p.affinity_capacity; ++cpu)
      if (CPU_ISSET(cpu, &allowed)) p.allowed_cpus[cpu / 64] |= std::uint64_t{1} << (cpu % 64);
  }
  void FinishProfileWorker(std::uint32_t worker) {
    if (profile) Clean(&profile->worker[worker], sizeof(WorkerProfile));
  }
  void ScatterRows(std::uint32_t worker, std::uint64_t dispatch, bool measure) {
    const auto range = ScatterRange(config, worker, ActiveThreads(config));
    int cpu_start = -1, cpu_start_error = 0, cpu_time_error = 0;
    timespec cpu_time_start{};
    if (measure) {
      cpu_start = sched_getcpu();
      if (cpu_start < 0) cpu_start_error = errno;
      if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_time_start) != 0) cpu_time_error = errno;
    }
    const auto start = measure ? Now() : 0;
    Invalidate(region + kStagingOffset + range.begin * config.row_bytes,
               (range.end - range.begin) * config.row_bytes);
    const auto invalidated = measure ? Now() : 0;
    for (auto slot = range.begin; slot < range.end; ++slot)
      std::memcpy(output + OutputSlotOffset(layout, slot),
                  region + kStagingOffset + slot * config.row_bytes, config.row_bytes);
    const auto copied = measure ? Now() : 0;
    Clean(output + OutputSlotOffset(layout, range.begin),
          (range.end - range.begin) * layout.stride);
    if (!measure) return;
    const auto cleaned = Now();
    timespec cpu_time_end{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_time_end) != 0 && !cpu_time_error)
      cpu_time_error = errno;
    const int cpu_end = sched_getcpu();
    const int cpu_end_error = cpu_end < 0 ? errno : 0;
    auto& p = profile->worker[worker];
    RecordWorkerTiming(p, dispatch, start, invalidated, copied, cleaned);
    RecordCpuSample(p, cpu_start, cpu_start_error);
    RecordCpuSample(p, cpu_end, cpu_end_error);
    if (cpu_start >= 0 && cpu_end >= 0 && cpu_start != cpu_end) ++p.boundary_cpu_changes;
    if (cpu_time_error) {
      ++p.cpu_time_errors;
      if (!p.cpu_time_errno) p.cpu_time_errno = cpu_time_error;
    } else {
      p.thread_cpu_ns += (cpu_time_end.tv_sec - cpu_time_start.tv_sec) * 1000000000LL +
                         cpu_time_end.tv_nsec - cpu_time_start.tv_nsec;
      ++p.cpu_time_rounds;
    }
  }
  std::uint64_t Frequency() {
    std::uint64_t value;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(value));
    return value;
  }
  std::uint64_t Now() {
    std::uint64_t value;
    asm volatile("isb; mrs %0, cntvct_el0" : "=r"(value) :: "memory");
    return value;
  }
  void PublishRequest(std::uint64_t sequence) {
    auto* control = reinterpret_cast<volatile Control*>(region + kRequestOffset);
    control->sequence = sequence;
    Clean(region + kRequestOffset, sizeof(Control));
  }
  std::uint64_t ReadDone() {
    Invalidate(region + kDoneOffset, sizeof(Control));
    return reinterpret_cast<volatile Control*>(region + kDoneOffset)->sequence;
  }
  void InitializeProfile() {
    if (!profile) return;
    profile->revision = kKernelRevision;
    profile->workers = ActiveThreads(config);
    profile->rounds = profile->pool_ticks = profile->dispatch_ticks =
        profile->envelope_ticks = profile->join_ticks = 0;
    job = Job::kProfileInit; pool.Run(); job = Job::kScatter;
  }
  void FinishProfile() {
    if (!profile) return;
    job = Job::kProfileFinish; pool.Run();
    Clean(profile, offsetof(ScatterProfile, worker));
  }
  void Scatter(const Config&, const Layout&) {
    ++scatter_round;
    if (!profile || scatter_round <= config.warmup) { pool.Run(); return; }
    dispatch_tick = Now(); pool.Run();
    RecordPoolTiming(*profile, dispatch_tick, Now());
  }
  bool VerifyTags(std::uint64_t seq, const Config& c, const Layout& l) {
    Invalidate(output, l.output_bytes);
    for (std::uint64_t row = 0; row < c.rows; ++row) {
      std::uint64_t value;
      std::memcpy(&value, output + OutputOffset(c, l, row), sizeof(value));
      if (value != RowTag(seq, row)) return false;
    }
    return true;
  }
};

std::uint64_t Load(const CoordinationWord& word) {
  return __atomic_load_n(&word.value, __ATOMIC_ACQUIRE);
}
void Store(CoordinationWord& word, std::uint64_t value) {
  __atomic_store_n(&word.value, value, __ATOMIC_RELEASE);
}
std::uint64_t Add(CoordinationWord& word, std::uint64_t value = 1) {
  return __atomic_fetch_add(&word.value, value, __ATOMIC_ACQ_REL);
}
Result Status(const MultiBlockCoordination& c) {
  return static_cast<Result>(Load(c.status));
}
void Fail(MultiBlockCoordination& c, Result status, std::uint64_t sequence) {
  std::uint64_t expected = static_cast<std::uint64_t>(Result::kPending);
  if (__atomic_compare_exchange_n(&c.status.value, &expected, static_cast<std::uint64_t>(status),
                                  false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    Store(c.failure_sequence, sequence);
}
bool WaitFor(DeviceIO& io, MultiBlockCoordination& c, const CoordinationWord& word,
             std::uint64_t target, std::uint64_t start, std::uint64_t timeout_ticks,
             std::uint64_t sequence) {
  for (;;) {
    if (Load(word) >= target) return true;
    if (Status(c) != Result::kPending) return false;
    if (io.Now() - start >= timeout_ticks) {
      Fail(c, Result::kTimeout, sequence);
      return false;
    }
  }
}
std::uint64_t StartupTimeoutTicks(DeviceIO& io) {
  // A serial/undersized AICPU scheduler must fail quickly instead of holding
  // the CPU transport for the per-round timeout.
  return io.Frequency();
}
void InitializeMultiBlock(DeviceIO& io, KernelResult* result, MultiBlockCoordination& c) {
  *result = {};
  result->requested_threads = io.config.scatter_threads;
  result->active_threads = ActiveThreads(io.config);
  result->kernel_revision = kKernelRevision;
  result->counter_frequency = io.Frequency();
  if (!result->counter_frequency) Fail(c, Result::kInvalidArgs, 0);
  if (io.profile) {
    io.profile->revision = kKernelRevision;
    io.profile->workers = ActiveThreads(io.config);
    io.profile->rounds = io.profile->pool_ticks = io.profile->dispatch_ticks =
        io.profile->envelope_ticks = io.profile->join_ticks = 0;
  }
  Store(c.initialized, 1);
}
Result FinishMultiBlock(DeviceIO& io, KernelResult* result, Sample* samples,
                        MultiBlockCoordination& c, std::uint32_t worker) {
  const auto workers = ActiveThreads(io.config);
  if (io.profile) {
    io.FinishProfileWorker(worker);
    Add(c.profile_flushed);
  }
  Add(c.exited);
  if (worker != 0) {
    const auto status = Status(c);
    return status == Result::kPending ? Result::kOk : status;
  }
  const auto wait_start = io.Now();
  const auto wait_ticks = StartupTimeoutTicks(io);
  WaitFor(io, c, c.exited, workers, wait_start, wait_ticks, result->failed_sequence);
  if (io.profile)
    WaitFor(io, c, c.profile_flushed, workers, wait_start, wait_ticks, result->failed_sequence);
  auto status = Status(c);
  if (status == Result::kPending) status = Result::kOk;
  result->status = status;
  if (status == Result::kOk) {
    result->failed_sequence = 0;
    Clean(samples, io.config.iterations * sizeof(Sample));
  } else {
    result->failed_sequence = Load(c.failure_sequence);
  }
  if (io.profile) Clean(io.profile, offsetof(ScatterProfile, worker));
  Clean(result, sizeof(*result));
  return status;
}

Result RunMultiBlock(DeviceIO& io, Sample* samples, KernelResult* result,
                     MultiBlockCoordination& c) {
  const auto workers = ActiveThreads(io.config);
  if (!IsValidMultiBlockCoordination(c, workers)) {
    *result = {}; result->status = Result::kInvalidArgs; result->kernel_revision = kKernelRevision;
    Clean(result, sizeof(*result)); return Result::kInvalidArgs;
  }
  const auto worker = static_cast<std::uint32_t>(Add(c.claimed));
  if (worker >= workers) {
    Fail(c, Result::kInvalidArgs, 0);
    return Result::kInvalidArgs;
  }
  if (worker == 0) InitializeMultiBlock(io, result, c);
  const auto startup_start = io.Now();
  if (!WaitFor(io, c, c.initialized, 1, startup_start, StartupTimeoutTicks(io), 0))
    return FinishMultiBlock(io, result, samples, c, worker);
  if (io.profile) {
    io.InitializeProfileWorker(worker);
    Add(c.profile_ready);
    if (!WaitFor(io, c, c.profile_ready, workers, startup_start, StartupTimeoutTicks(io), 0))
      return FinishMultiBlock(io, result, samples, c, worker);
  }
  Add(c.ready);
  if (!WaitFor(io, c, c.ready, workers, startup_start, StartupTimeoutTicks(io), 0))
    return FinishMultiBlock(io, result, samples, c, worker);

  const auto frequency = result->counter_frequency;
  const auto timeout_ticks = frequency * io.config.timeout_ms / 1000;
  for (std::uint64_t sequence = 1; sequence <= io.layout.rounds; ++sequence) {
    if (worker == 0) {
      result->failed_sequence = sequence;
      const auto start = io.Now();
      io.PublishRequest(sequence);
      for (;;) {
        const auto done = io.ReadDone();
        if (done == sequence) break;
        if (done > sequence) { Fail(c, Result::kSequenceError, sequence); break; }
        if (io.Now() - start >= timeout_ticks) { Fail(c, Result::kTimeout, sequence); break; }
      }
      if (Status(c) == Result::kPending) {
        Store(c.request_start, start);
        Store(c.ready_tick, io.Now());
        Store(c.request_sequence, sequence);
      }
    } else if (!WaitFor(io, c, c.request_sequence, sequence, io.Now(), timeout_ticks, sequence)) {
      break;
    }
    if (Status(c) != Result::kPending) break;
    const auto ready = Load(c.ready_tick);
    const bool profile_round = io.profile && sequence > io.config.warmup;
    io.ScatterRows(worker, ready, profile_round);
    const auto arrivals = Add(c.arrivals) + 1;
    if (arrivals == sequence * workers) Store(c.scatter_done, sequence);
    if (worker == 0) {
      if (!WaitFor(io, c, c.scatter_done, sequence, io.Now(), timeout_ticks, sequence)) break;
      const auto end = io.Now();
      if (!io.VerifyTags(sequence, io.config, io.layout)) { Fail(c, Result::kDataMismatch, sequence); break; }
      if (sequence > io.config.warmup) {
        const auto start = Load(c.request_start);
        samples[sequence - io.config.warmup - 1] = {end - start, ready - start, end - ready};
        if (io.profile) RecordPoolTiming(*io.profile, ready, end);
      }
      result->completed_rounds = sequence;
      Store(c.round_complete, sequence);
    } else if (!WaitFor(io, c, c.round_complete, sequence, io.Now(), timeout_ticks, sequence)) {
      break;
    }
  }
  return FinishMultiBlock(io, result, samples, c, worker);
}

void ProbeCpu(CoreProbeSlot& slot) {
  const int cpu = sched_getcpu();
  if (cpu < 0) {
    ++slot.cpu_sample_errors;
    if (!slot.cpu_sample_errno) slot.cpu_sample_errno = errno;
    return;
  }
  if (slot.first_cpu < 0) slot.first_cpu = cpu;
  slot.last_cpu = cpu;
  if (cpu >= 1024) ++slot.cpu_out_of_range;
}
void ProbeAffinity(CoreProbeSlot& slot) {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) { slot.affinity_errno = errno; return; }
  for (unsigned cpu = 0; cpu < 1024 && cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &allowed)) slot.allowed_cpus[cpu / 64] |= std::uint64_t{1} << (cpu % 64);
}
std::uint64_t ProbeNow() {
  std::uint64_t value;
  asm volatile("isb; mrs %0, cntvct_el0" : "=r"(value) :: "memory");
  return value;
}
std::uint64_t ProbeFrequency() {
  std::uint64_t value;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(value));
  return value;
}
}  // namespace

extern "C" __attribute__((visibility("default")))
std::uint32_t KvafAicpuCoreProbeV1(CoreProbeArgs* args) {
  if (!args || !args->result || !args->requested_blocks ||
      args->requested_blocks > kMaxScatterThreads || !args->hold_ms) return 1;
  auto* result = reinterpret_cast<CoreProbeResult*>(args->result);
  if (!IsValidCoreProbeResult(*result, static_cast<std::uint32_t>(args->requested_blocks))) return 1;
  const auto slot_id = __atomic_fetch_add(&result->claimed, std::uint64_t{1}, __ATOMIC_ACQ_REL);
  if (slot_id >= args->requested_blocks) return 1;
  auto& slot = result->slot[slot_id];
  slot = {}; slot.slot = slot_id;
  ProbeAffinity(slot);
  ProbeCpu(slot);
  const auto hold_ticks = ProbeFrequency() * args->hold_ms / 1000;
  if (!hold_ticks) return 1;
  slot.start_tick = ProbeNow();
  while (ProbeNow() - slot.start_tick < hold_ticks) asm volatile("yield" ::: "memory");
  slot.finish_tick = ProbeNow();
  ProbeCpu(slot);
  Clean(&slot, sizeof(slot));
  if (slot_id == 0) {
    result->counter_frequency = ProbeFrequency();
    result->hold_ticks = hold_ticks;
  }
  const auto completed = __atomic_fetch_add(&result->completed, std::uint64_t{1}, __ATOMIC_ACQ_REL) + 1;
  if (completed == args->requested_blocks) {
    result->status = Result::kOk;
    Clean(result, offsetof(CoreProbeResult, slot));
  }
  return 0;
}

extern "C" __attribute__((visibility("default")))
std::uint32_t KvafAicpuBatchGetV5(KernelArgs* args) {
  if (!args || !args->region || !args->output || !args->samples || !args->result) return 1;
  auto* result = reinterpret_cast<KernelResult*>(args->result);
  auto* samples = reinterpret_cast<Sample*>(args->samples);
  Layout layout{};
  if (!MakeLayout(args->config, &layout)) {
    *result = {}; result->status = Result::kInvalidArgs;
    Clean(result, sizeof(*result)); return 1;
  }
  DeviceIO io{reinterpret_cast<std::uint8_t*>(args->region),
              reinterpret_cast<std::uint8_t*>(args->output), args->config, layout,
              reinterpret_cast<ScatterProfile*>(args->profile), {}};
  if (args->coordination) {
    const auto status = RunMultiBlock(io, samples, result,
        *reinterpret_cast<MultiBlockCoordination*>(args->coordination));
    return status == Result::kOk ? 0 : 1;
  }
  const auto error = io.pool.Initialize(ActiveThreads(args->config), &DeviceIO::ScatterWorker, &io);
  if (error) {
    *result = {}; result->status = Result::kThreadError;
    result->requested_threads = args->config.scatter_threads;
    result->pthread_error = error; result->kernel_revision = kKernelRevision;
    Clean(result, sizeof(*result)); return 1;
  }
  io.InitializeProfile();
  const auto status = RunDevice(args->config, io, samples, result);
  io.FinishProfile();
  result->status = status;
  if (status == Result::kOk) Clean(samples, args->config.iterations * sizeof(Sample));
  Clean(result, sizeof(*result));
  return status == Result::kOk ? 0 : 1;
}
