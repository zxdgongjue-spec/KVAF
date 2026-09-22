#include "device_loop.h"

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
  for (; begin < end; begin += 64) asm volatile("dc ivac, %0" :: "r"(begin) : "memory");
  Barrier();
}
struct DeviceIO {
  std::uint8_t* region;
  std::uint8_t* output;
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
  void Scatter(const Config& c, const Layout& l) {
    // This baseline intentionally measures AICPU memcpy, NOT SDMA bandwidth.
    Invalidate(region + kStagingOffset, l.payload_bytes);
    for (std::uint64_t row = 0; row < c.rows; ++row) {
      auto* dst = output + OutputOffset(c, l, row);
      std::memcpy(dst, region + kStagingOffset + row * c.row_bytes, c.row_bytes);
      Clean(dst, c.row_bytes);
    }
  }
  bool VerifyTags(std::uint64_t seq, const Config& c, const Layout& l) {
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
std::uint32_t KvafAicpuBatchGet(KernelArgs* args) {
  if (!args || !args->region || !args->output || !args->samples || !args->result)
    return 1;
  auto* result = reinterpret_cast<KernelResult*>(args->result);
  auto* samples = reinterpret_cast<Sample*>(args->samples);
  DeviceIO io{reinterpret_cast<std::uint8_t*>(args->region),
              reinterpret_cast<std::uint8_t*>(args->output)};
  const auto status = RunDevice(args->config, io, samples, result);
  result->status = status;
  if (status == Result::kOk) Clean(samples, args->config.iterations * sizeof(Sample));
  Clean(result, sizeof(*result));
  return status == Result::kOk ? 0 : 1;
}
