#include "gather_host_plan.h"
#include "gather_output_layout.h"
#include "gather_read_storage.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

namespace {

bool Reference(std::uint64_t base, std::uint64_t region_bytes, std::uint64_t staging,
               std::uint64_t descriptor, const std::vector<kvaf::peer::GatherSpan>& sources,
               std::uint64_t output_bytes, const std::vector<kvaf::GatherOutputSpan>& targets,
               std::uint32_t* total, std::uint64_t* extent) {
  if (!kvaf::peer::ValidateGather(base, region_bytes, staging, sources, total)) return false;
  if (descriptor != 0U) {
    const auto bytes = kvaf::peer::GatherDescriptorBytes(sources.size());
    if (descriptor % 64U != 0U ||
        !kvaf::peer::GatherRange(base, kvaf::peer::GatherControlOffset(region_bytes),
                                 descriptor, bytes) ||
        (descriptor < staging + *total && staging < descriptor + bytes)) return false;
    for (const auto& source : sources)
      if (descriptor < source.address + source.length && source.address < descriptor + bytes)
        return false;
  }
  return kvaf::ValidateGatherOutput(output_bytes,
      targets.empty() ? std::vector<kvaf::GatherOutputSpan>{{0U, *total, 0U}} : targets,
      *total, extent);
}

void CompareLayouts() {
  constexpr std::uint64_t base = 4096U;
  constexpr std::uint64_t region_bytes = 65536U;
  std::mt19937 random(42U);
  kvaf::GatherHostPlan plan;
  std::vector<kvaf::peer::GatherSpan> sources;
  std::vector<kvaf::GatherOutputSpan> targets;
  std::size_t valid = 0U, invalid = 0U;
  for (std::uint32_t round = 0U; round < 10000U; ++round) {
    sources.clear();
    targets.clear();
    const auto count = 1U + random() % 32U;
    for (std::uint32_t index = 0U; index < count; ++index) {
      const auto length = static_cast<std::uint32_t>(1U + random() % 64U);
      sources.push_back({base + random() % 16384U, length, 0U});
      targets.push_back({index * 128U, length, 0U});
    }
    std::shuffle(targets.begin(), targets.end(), random);
    auto staging = base + 16384U;
    auto descriptor = round % 2U == 0U ? base + 32768U : 0U;
    std::uint64_t output_bytes = 8192U;
    switch (round % 12U) {
      case 0U: targets.clear(); break;
      case 1U: staging = sources.front().address; break;
      case 2U: staging = sources.front().address + 1U; break;
      case 3U: descriptor = sources.front().address & ~std::uint64_t{63U}; break;
      case 4U: descriptor = staging; break;
      case 5U: sources.front().reserved = 1U; break;
      case 6U: sources.front().address = UINT64_MAX - 3U; break;
      case 7U: output_bytes = 64U; break;
      case 8U: targets.front().offset = UINT64_MAX - 1U; break;
      case 9U: staging = base + region_bytes - 128U; break;
      default: break;
    }
    std::uint32_t total = 0U;
    std::uint64_t extent = 0U;
    const bool expected = Reference(base, region_bytes, staging, descriptor, sources,
                                     output_bytes, targets, &total, &extent);
    const bool actual = kvaf::PrepareGatherHostLayout(base, region_bytes, staging, descriptor,
        sources, output_bytes, targets, &plan);
    assert(actual == expected);
    if (actual) {
      ++valid;
      assert(plan.total_bytes == total && plan.output_extent == extent);
    } else ++invalid;
  }
  assert(valid > 1000U && invalid > 1000U);
}

void Boundaries() {
  kvaf::GatherHostPlan plan;
  const std::vector<kvaf::peer::GatherSpan> sources{{4096U, 32U, 0U}, {4352U, 32U, 0U}};
  assert(kvaf::PrepareGatherHostLayout(4096U, 1024U, 4288U, 4480U,
                                      sources, 128U, {}, &plan));
  assert(!kvaf::PrepareGatherHostLayout(4096U, 1024U, 4289U, 4480U,
                                       sources, 128U, {}, &plan));
  assert(!kvaf::PrepareGatherHostLayout(4096U, 1024U, 4288U, 4481U,
                                       sources, 128U, {}, &plan));
  assert(!kvaf::PrepareGatherHostLayout(4096U, 1024U, 4288U, 0U,
                                       sources, 127U, {}, &plan));
  assert(!kvaf::PrepareGatherHostLayout(UINT64_MAX - 32U, 1024U, 4288U, 0U,
                                       sources, 128U, {}, &plan));
  assert(!kvaf::PrepareGatherHostLayout(4096U, 63U, 4288U, 0U,
                                       sources, 128U, {}, &plan));
  const std::vector<kvaf::peer::GatherSpan> huge{{4096U, UINT32_MAX, 0U},
                                              {4096U, 1U, 0U}};
  assert(!kvaf::PrepareGatherHostLayout(4096U, 1ULL << 34U, 1ULL << 33U, 0U,
                                       huge, 1ULL << 34U, {}, &plan));
  const std::vector<kvaf::peer::GatherSpan> many(65537U, {4096U, 1U, 0U});
  assert(!kvaf::PrepareGatherHostLayout(4096U, 1ULL << 20U, 8192U, 0U,
                                       many, 1ULL << 20U, {}, &plan));
  std::vector<kvaf::GatherOutputSpan> targets{{64U, 16U, 0U}, {0U, 16U, 0U},
                                            {32U, 32U, 0U}};
  assert(kvaf::PrepareGatherHostLayout(4096U, 1024U, 4288U, 4480U,
                                      sources, 192U, targets, &plan));
  targets[2].offset = 56U;
  assert(!kvaf::PrepareGatherHostLayout(4096U, 1024U, 4288U, 4480U,
                                       sources, 192U, targets, &plan));
}

void PackedStorage() {
  using namespace kvaf::aicpu_urma;
  for (std::uint32_t count : {1U, 3U, 4096U}) {
    const auto source_bytes = kvaf::peer::GatherDescriptorBytes(count);
    const auto target_offset = GatherPackedTargetOffset(source_bytes);
    assert(target_offset % 64U == 0U && target_offset >= source_bytes);
    assert(target_offset - source_bytes < 64U);
    for (std::uint32_t task_spans : {0U, 1U, 192U}) {
      const auto required = GatherReadPackedOutputBytes(656U, source_bytes, count, task_spans);
      assert(required == 704U + 64U + target_offset + GatherOutputStorageBytes(count, task_spans));
      const auto upload_end = 704U + 64U + target_offset + count * sizeof(kvaf::GatherOutputSpan);
      assert(upload_end <= required);
      assert(GatherReadPackedOutputBytes(656U, source_bytes, count, task_spans, 131072U) ==
             required + 7U * 64U);
      if (task_spans != 0U)
        assert(required >= upload_end + sizeof(GatherTaskState));
    }
  }
  assert(GatherReadPackedOutputBytes(UINT64_MAX, 48U, 1U, 0U) == 0U);
  assert(GatherReadPackedOutputBytes(128U, UINT64_MAX, 1U, 0U) == 0U);
  assert(GatherReadPackedOutputBytes(UINT64_MAX - 63U, 48U, 1U, 0U) == 0U);
}

}

int main() {
  CompareLayouts();
  Boundaries();
  PackedStorage();
  std::cout << "GatherRead Host plan: PASS\n";
}
