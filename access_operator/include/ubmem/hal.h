#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "status.h"

namespace kvaf::ubmem {

// Public ascend_hal_define.h ABI. Allocation handles remain opaque: their
// private layout differs between SVM v2/v3. No HAL or UMDK headers are needed.
struct MemoryProperty {
  std::uint32_t side, device, module, page_type, memory_type;
  std::uint64_t reserved;
};
static_assert(sizeof(MemoryProperty) == 32);
static_assert(offsetof(MemoryProperty, reserved) == 24);
using FabricHandle = std::array<std::uint8_t, 128>;
constexpr std::size_t kPageBytes = 2ULL * 1024 * 1024;

class Hal {
 public:
  ~Hal();
  Status Load(const std::string& library = "libascend_hal.so");
  Hal() = default;
  Hal(const Hal&) = delete;
  Hal& operator=(const Hal&) = delete;

  int (*create)(void**, std::size_t, const MemoryProperty*, std::uint64_t){};
  int (*release)(void*){};
  int (*reserve)(void**, std::size_t, std::size_t, void*, std::uint64_t){};
  int (*free_address)(void*){};
  int (*map)(void*, std::size_t, std::size_t, void*, std::uint64_t){};
  int (*unmap)(void*){};
  int (*export_memory)(void*, int, std::uint64_t, FabricHandle*){};
  int (*translate_handle)(int, FabricHandle*, std::uint32_t*, std::uint64_t*){};
  struct ShareAttribute { std::uint32_t enable; std::uint32_t reserved[8]; };
  int (*share_attribute)(std::uint64_t, int, ShareAttribute){};

 private:
  void* library_{};
  std::string path_;
};

Status HalError(const char* operation, int result);

class Mapping {
 public:
  explicit Mapping(std::shared_ptr<Hal> hal) : hal_(std::move(hal)) {}
  ~Mapping();
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;

  Status AllocateHost(std::uint64_t bytes, int numa_node, FabricHandle* descriptor);
  Status Close();
  std::uint64_t address() const { return reinterpret_cast<std::uintptr_t>(address_); }
  std::uint64_t bytes() const { return bytes_; }

 private:
  Status ReserveAndMap();
  std::shared_ptr<Hal> hal_;
  void* handle_{};
  void* address_{};
  std::uint64_t bytes_{};
  bool mapped_{};
};

}  // namespace kvaf::ubmem
