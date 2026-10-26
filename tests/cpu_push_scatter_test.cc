#include "cpu_push_scatter_plan.h"
#include "aicpu_batch_get/kernel/cpu_push_scatter.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

using namespace kvaf::cpu_push;

struct alignas(64) CacheLine { std::uint8_t bytes[64]{}; };
template <class T> std::uint64_t Pointer(T* value) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(value));
}

void CheckDeviceKernel(const std::vector<std::uint64_t>& addresses,
                       const std::vector<std::int32_t>& indices,
                       const std::vector<std::int32_t>& offsets,
                       const std::vector<std::int32_t>& rows, std::uint32_t active,
                       std::uint32_t bytes, std::uint32_t workers,
                       const std::vector<std::uint8_t>& staging,
                       const std::vector<std::uint8_t>& expected) {
  Layout layout;
  CHECK(MakeLayout(bytes, workers, &layout));
  std::vector<CacheLine> workspace((layout.bytes + 63) / 64);
  auto* memory = reinterpret_cast<std::uint8_t*>(workspace.data());
  if (!staging.empty()) std::memcpy(memory + layout.staging, staging.data(), staging.size());
  // Model a completed producer. The actual production kernel still builds
  // its plan, claims workers and consumes ready/done through the real pipeline.
  reinterpret_cast<Completion*>(memory + layout.done)->token = Progress(1, 1);
  auto* ready = reinterpret_cast<Ready*>(memory + layout.ready);
  for (std::uint32_t w = 0; w < workers; ++w)
    for (auto& token : ready[w].jetty)
      token = Progress(1, ChunkCount(WorkerRange(active, bytes, w, workers)));
  Coordination coordination{};
  std::vector<WorkerStatus> statuses(workers);
  std::vector<kvaf::BatchGetWorkerProfile> profiles(workers);
  std::vector<std::uint8_t> output(expected.size(), 0xA5);
  const auto count = static_cast<std::int32_t>(active);
  KernelArgs args{};
  args.request = {Pointer(addresses.data()), Pointer(indices.data()), Pointer(offsets.data()),
                  Pointer(&count), Pointer(rows.data()), static_cast<std::uint32_t>(addresses.size()),
                  static_cast<std::uint32_t>(indices.size()), bytes};
  args.workspace = Pointer(memory);
  args.output = Pointer(output.data());
  args.output_bytes = output.size() - 64;
  args.coordination = Pointer(&coordination);
  args.status = Pointer(statuses.data());
  args.profile = Pointer(profiles.data());
  args.sequence = 1;
  args.workers = workers;
  args.timeout_ms = 5000;
  std::vector<std::thread> threads;
  for (std::uint32_t w = 0; w < workers; ++w)
    threads.emplace_back([&] { CHECK(ExecuteCpuPushScatter(&args) == 0); });
  for (auto& thread : threads) thread.join();
  CHECK(output == expected);
  const auto notify = reinterpret_cast<Word*>(memory + layout.notify)->value;
  if (active) {
    CHECK(NotifySequence(notify) == 1 && NotifyCount(notify) == active);
    CHECK((notify & kProfileNotify) != 0);
  } else CHECK(notify == 0);
  for (std::uint32_t w = 0; w < workers; ++w) {
    CHECK(statuses[w].magic == kMagic && statuses[w].sequence == 1);
    CHECK(statuses[w].result == static_cast<std::uint64_t>(Result::kOk));
    CHECK(statuses[w].worker == w);
    if (active == 4096 && workers == 4) CHECK(profiles[w].scatter_copies == 1);
    CHECK(profiles[w].bytes ==
          std::uint64_t{WorkerRange(active, bytes, w, workers).end -
                        WorkerRange(active, bytes, w, workers).begin} * bytes);
  }
}

void CheckRadixDigitsAndStability() {
  std::vector<std::int32_t> rows{INT32_MAX, 256, 65536, 0, 256, 16777216, 255, 65535, 0};
  std::vector<std::uint32_t> expected(rows.size()), order(rows.size()), scratch(rows.size());
  std::iota(expected.begin(), expected.end(), 0);
  std::stable_sort(expected.begin(), expected.end(), [&](auto a, auto b) { return rows[a] < rows[b]; });
  const auto* sorted = OrderByOutput(rows.data(), rows.size(), order.data(), scratch.data());
  CHECK(std::equal(expected.begin(), expected.end(), sorted));
  std::sort(rows.begin(), rows.end());
  sorted = OrderByOutput(rows.data(), rows.size(), order.data(), scratch.data());
  for (std::uint32_t i = 0; i < rows.size(); ++i) CHECK(sorted[i] == i);
}

void CheckNotifyEncoding() {
  for (auto sequence : {0U, 1U, UINT32_MAX})
    for (auto count : {0U, 1U, 4096U, kMaxRequests})
      for (auto profile : {false, true}) {
        const auto notify = RequestNotify(sequence, count, profile);
        CHECK(NotifySequence(notify) == sequence);
        CHECK(NotifyCount(notify) == count);
        CHECK(((notify & kProfileNotify) != 0) == profile);
      }
}

void CheckRound(std::uint32_t capacity, std::uint32_t active, std::uint32_t bytes,
                std::uint32_t step, std::uint32_t workers, std::uint32_t seed) {
  std::mt19937 random(seed);
  const auto block_bytes = 2 * bytes;
  std::vector<std::uint8_t> remote(std::size_t{capacity} * block_bytes);
  for (auto& value : remote) value = static_cast<std::uint8_t>(random());
  std::vector<std::uint64_t> addresses(capacity);
  std::vector<std::int32_t> indices(capacity), offsets(capacity), rows(capacity);
  std::iota(indices.begin(), indices.end(), 0);
  std::iota(rows.begin(), rows.end(), 0);
  std::shuffle(indices.begin(), indices.end(), random);
  std::shuffle(rows.begin(), rows.end(), random);
  for (std::uint32_t i = 0; i < capacity; ++i) {
    addresses[i] = reinterpret_cast<std::uintptr_t>(remote.data() + std::size_t{i} * block_bytes);
    offsets[i] = (random() % 2) * bytes;
    rows[i] *= step;
  }
  const auto output_bytes = std::size_t{capacity * 2} * bytes;
  std::vector<std::uint8_t> expected(output_bytes + 64, 0xA5), output = expected;
  // Independent oracle follows the caller's original request order.
  for (std::uint32_t i = 0; i < active; ++i)
    std::memcpy(expected.data() + static_cast<std::size_t>(rows[i]) * bytes,
        reinterpret_cast<const void*>(addresses[indices[i]] + offsets[i]), bytes);
  // Inactive metadata may be invalid and must never be read by the plan.
  for (auto i = active; i < capacity; ++i) indices[i] = offsets[i] = rows[i] = -1;
  const auto original_indices = indices, original_offsets = offsets, original_rows = rows;
  std::vector<std::uint64_t> sources(capacity, UINT64_MAX);
  std::vector<std::uint32_t> destinations(capacity, UINT32_MAX), scratch(capacity);
  BuildScatterPlan(addresses.data(), indices.data(), offsets.data(), rows.data(), active,
                   sources.data(), destinations.data(), scratch.data());
  CHECK(indices == original_indices && offsets == original_offsets && rows == original_rows);
  CHECK(std::is_sorted(destinations.begin(), destinations.begin() + active));
  for (auto i = active; i < capacity; ++i) {
    CHECK(sources[i] == UINT64_MAX);
    CHECK(destinations[i] == UINT32_MAX);
  }
  std::vector<std::uint8_t> staging(std::size_t{active} * bytes);
  for (std::uint32_t i = 0; i < active; ++i)
    std::memcpy(staging.data() + std::size_t{i} * bytes,
                reinterpret_cast<const void*>(sources[i]), bytes);
  std::uint32_t copies = 0;
  std::uint64_t copied = 0;
  for (std::uint32_t worker = 0; worker < workers; ++worker) {
    const auto range = WorkerRange(active, bytes, worker, workers);
    // Release one chunk at a time: coalescing must not touch unready staging.
    for (std::uint32_t chunk = 0; chunk < ChunkCount(range); ++chunk) {
      const auto ready = ChunkRange(range, chunk);
      CHECK(CopyScatterRuns(destinations.data(), ready, bytes,
          [&](std::uint64_t destination, std::uint64_t source, std::uint64_t length) {
        CHECK(source >= std::uint64_t{ready.begin} * bytes);
        CHECK(source + length <= std::uint64_t{ready.end} * bytes);
        CHECK(destination + length <= output_bytes);
        std::memcpy(output.data() + destination, staging.data() + source, length);
        ++copies;
        copied += length;
        return true;
      }));
    }
  }
  CHECK(output == expected);
  CHECK(copied == std::uint64_t{active} * bytes);
  if ((capacity == 4096 && step == 1 && workers == 4) ||
      (capacity == 4103 && step == 2 && workers == 8))
    CheckDeviceKernel(addresses, indices, offsets, rows, active, bytes, workers, staging, expected);
  if (capacity == 4096 && active == capacity && bytes == 656 && step == 1 && workers == 4) {
    CHECK(copies == 4);
    std::cout << "dense 4096x656: 4096 row copies reduced to " << copies << " ready-range copies\n";
  }
}

void CheckBoundedCopiesAndFailure() {
  const std::uint32_t rows[]{2, 3, 4};
  std::uint64_t total = 0;
  std::uint32_t calls = 0;
  CHECK(CopyScatterRuns(rows, {0, 3}, 2 * 1024 * 1024,
      [&](auto destination, auto source, auto bytes) {
    CHECK(bytes <= 1024 * 1024);
    CHECK(source == total && destination == 4 * 1024 * 1024 + total);
    total += bytes;
    ++calls;
    return true;
  }));
  CHECK(total == 6 * 1024 * 1024 && calls == 6);
  calls = 0;
  CHECK(!CopyScatterRuns(rows, {0, 3}, 656, [&](auto, auto, auto) { ++calls; return false; }));
  CHECK(calls == 1);
}

int main() {
  CheckNotifyEncoding();
  CheckRadixDigitsAndStability();
  CheckBoundedCopiesAndFailure();
  for (const auto capacity : {1U, 17U, 4096U, 4103U, kMaxRequests}) {
    for (const auto active : {0U, capacity * 3 / 4, capacity}) {
      for (const auto step : {1U, 2U}) {
        for (const auto workers : {1U, 4U, 8U}) {
          CheckRound(capacity, active, 656, step, workers, capacity + active + step + workers);
        }
      }
    }
  }
  for (const auto bytes : {1U, 17U, 64U, 4096U})
    CheckRound(4103, 3077, bytes, 2, 8, bytes);
  std::cout << "CPU push scatter plan and guarded-output tests passed\n";
}
