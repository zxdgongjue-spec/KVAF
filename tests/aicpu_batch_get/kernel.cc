#include "device_loop.h"
#include "scatter_pool.h"

#include <cstring>

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
  ScatterPool pool;
  static void ScatterWorker(void* context, std::uint32_t worker) {
    auto& io = *static_cast<DeviceIO*>(context);
    const auto range = ScatterRange(io.config, worker, ActiveThreads(io.config));
    Invalidate(io.region + kStagingOffset + range.begin * io.config.row_bytes,
               (range.end - range.begin) * io.config.row_bytes);
    for (auto row = range.begin; row < range.end; ++row)
      std::memcpy(io.output + OutputOffset(io.config, io.layout, row),
                  io.region + kStagingOffset + row * io.config.row_bytes, io.config.row_bytes);
    Clean(io.output + (io.config.rows - range.end) * io.layout.stride,
          (range.end - range.begin) * io.layout.stride);
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
  void Scatter(const Config&, const Layout&) { pool.Run(); }
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
std::uint32_t KvafAicpuBatchGetV2(KernelArgs* args) {
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
              reinterpret_cast<std::uint8_t*>(args->output), args->config, layout, {}};
  const auto error = io.pool.Initialize(ActiveThreads(args->config), &DeviceIO::ScatterWorker, &io);
  if (error) {
    *result = {}; result->status = Result::kThreadError;
    result->requested_threads = args->config.scatter_threads;
    result->pthread_error = error; result->kernel_revision = kKernelRevision;
    Clean(result, sizeof(*result)); return 1;
  }
  const auto status = RunDevice(args->config, io, samples, result);
  result->status = status;
  if (status == Result::kOk) Clean(samples, args->config.iterations * sizeof(Sample));
  Clean(result, sizeof(*result));
  return status == Result::kOk ? 0 : 1;
}
