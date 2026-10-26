#include "cpu_memory_page_policy.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); std::exit(1); \
} } while (false)

int main() {
  using kvaf::CpuMemoryPageMode;
  using kvaf::detail::CpuMemoryAdvice;
  using kvaf::detail::MemoryPagePolicy;
  constexpr auto thp = MemoryPagePolicy(CpuMemoryPageMode::kTransparentHugePage);
  constexpr auto no_thp = MemoryPagePolicy(CpuMemoryPageMode::kNoHugePage);
  constexpr auto origin = MemoryPagePolicy(CpuMemoryPageMode::kOrigin);
  // The causal comparison must change advice only, including future changes.
  CHECK(thp.alignment == no_thp.alignment);
  CHECK(thp.pre_touch == no_thp.pre_touch);
  CHECK(thp.alignment == 2U * 1024U * 1024U);
  CHECK(thp.pre_touch);
  CHECK(thp.advice == CpuMemoryAdvice::kHugePage);
  CHECK(no_thp.advice == CpuMemoryAdvice::kNoHugePage);
  // Retain the existing practical origin path rather than silently pre-touch it.
  CHECK(origin.alignment == 4096U);
  CHECK(!origin.pre_touch && origin.advice == CpuMemoryAdvice::kNone);
  const kvaf::CpuRuntimeOptions defaults;
  CHECK(defaults.memory_page_mode == CpuMemoryPageMode::kTransparentHugePage);
  CHECK(!defaults.profile_initialize);
  void* allocation = nullptr;
  CHECK(posix_memalign(&allocation, no_thp.alignment, 4096U) == 0);
  CHECK(allocation != nullptr);
  CHECK(reinterpret_cast<std::uintptr_t>(allocation) % no_thp.alignment == 0U);
  std::free(allocation);
  std::puts("CPU memory page policy tests passed");
  return 0;
}
