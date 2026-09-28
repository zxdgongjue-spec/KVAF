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
  // (subject to platform cache-maintenance permission), as in the reference kernels.
  // These DMA-owned lines are only read by AICPU; do not dirty them before
  // invalidation, or the clean part could overwrite newer DMA data.
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
    const auto range = ScatterRange(io.config, worker, ActiveThreads(io.config));
    if (io.job == Job::kProfileInit) {
      auto& p = io.profile->worker[worker];
      p = {};
      p.row_begin = range.begin; p.row_end = range.end;
      cpu_set_t allowed;
      CPU_ZERO(&allowed);
      p.affinity_capacity = std::min<unsigned>(CPU_SETSIZE, kProfileCpuLimit);
      if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) p.affinity_errno = errno;
      else for (unsigned cpu = 0; cpu < p.affinity_capacity; ++cpu)
        if (CPU_ISSET(cpu, &allowed)) p.allowed_cpus[cpu / 64] |= std::uint64_t{1} << (cpu % 64);
      return;
    }
    if (io.job == Job::kProfileFinish) {
      // Flush on the writer, outside the measured rounds, for host readback.
      Clean(&io.profile->worker[worker], sizeof(WorkerProfile));
      return;
    }
    const bool measure = io.profile && io.scatter_round > io.config.warmup;
    int cpu_start = -1, cpu_start_error = 0, cpu_time_error = 0;
    timespec cpu_time_start{};
    if (measure) {
      cpu_start = sched_getcpu();
      if (cpu_start < 0) cpu_start_error = errno;
      if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_time_start) != 0) cpu_time_error = errno;
    }
    const auto start = measure ? io.Now() : 0;
    Invalidate(io.region + kStagingOffset + range.begin * io.config.row_bytes,
               (range.end - range.begin) * io.config.row_bytes);
    const auto invalidated = measure ? io.Now() : 0;
    for (auto row = range.begin; row < range.end; ++row)
      std::memcpy(io.output + OutputOffset(io.config, io.layout, row),
                  io.region + kStagingOffset + row * io.config.row_bytes, io.config.row_bytes);
    const auto copied = measure ? io.Now() : 0;
    Clean(io.output + (io.config.rows - range.end) * io.layout.stride,
          (range.end - range.begin) * io.layout.stride);
    if (measure) {
      const auto cleaned = io.Now();
      timespec cpu_time_end{};
      if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_time_end) != 0 && !cpu_time_error)
        cpu_time_error = errno;
      const int cpu_end = sched_getcpu();
      const int cpu_end_error = cpu_end < 0 ? errno : 0;
      auto& p = io.profile->worker[worker];
      RecordWorkerTiming(p, io.dispatch_tick, start, invalidated, copied, cleaned);
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
    job = Job::kProfileInit;
    pool.Run();
    job = Job::kScatter;
  }
  void FinishProfile() {
    if (!profile) return;
    job = Job::kProfileFinish;
    pool.Run();
    Clean(profile, offsetof(ScatterProfile, worker));
  }
  void Scatter(const Config&, const Layout&) {
    ++scatter_round;
    if (!profile || scatter_round <= config.warmup) { pool.Run(); return; }
    dispatch_tick = Now();
    pool.Run();
    const auto returned = Now();
    RecordPoolTiming(*profile, dispatch_tick, returned);
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
}  // namespace

extern "C" __attribute__((visibility("default")))
std::uint32_t KvafAicpuBatchGetV3(KernelArgs* args) {
  if (!args || !args->region || !args->output || !args->samples || !args->result)
    return 1;
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
