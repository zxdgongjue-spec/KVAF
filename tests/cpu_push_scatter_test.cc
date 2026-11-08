#include "aicpu_batch_get/kernel/cpu_push_scatter.h"
#include "aicpu_batch_get/kernel/cpu_push_pipeline.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
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
  CHECK((notify & ~kProfileNotify) == (active ? 1U : 0U));
  CHECK(((notify & kProfileNotify) != 0) == (active != 0));
  const auto* sources = reinterpret_cast<const std::uint64_t*>(memory + layout.descriptor);
  for (std::uint32_t i = 0; i < active; ++i)
    CHECK(sources[i] == addresses[indices[i]] + offsets[i]);
  std::uint64_t copies = 0;
  for (std::uint32_t w = 0; w < workers; ++w) {
    const auto range = WorkerRange(active, bytes, w, workers);
    CHECK(statuses[w].magic == kMagic && statuses[w].sequence == 1);
    CHECK(statuses[w].result == static_cast<std::uint64_t>(Result::kOk));
    CHECK(statuses[w].worker == w && statuses[w].consumed == ChunkCount(range));
    CHECK(profiles[w].bytes == std::uint64_t{range.end - range.begin} * bytes);
    CHECK(profiles[w].scatter_copies == range.end - range.begin);
    copies += profiles[w].scatter_copies;
  }
  CHECK(copies == active);
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
    addresses[i] = Pointer(remote.data() + std::size_t{i} * block_bytes);
    offsets[i] = (random() % 2) * bytes;
    rows[i] *= step;
  }
  const auto output_bytes = std::size_t{capacity * 2} * bytes;
  std::vector<std::uint8_t> expected(output_bytes + 64, 0xA5), output = expected;
  std::vector<std::uint8_t> staging(std::size_t{active} * bytes);
  for (std::uint32_t i = 0; i < active; ++i) {
    const auto* source = reinterpret_cast<const void*>(addresses[indices[i]] + offsets[i]);
    std::memcpy(expected.data() + static_cast<std::size_t>(rows[i]) * bytes, source, bytes);
    std::memcpy(staging.data() + std::size_t{i} * bytes, source, bytes);
  }
  // Invalid inactive metadata must never be traversed or sorted.
  for (auto i = active; i < capacity; ++i) indices[i] = offsets[i] = rows[i] = -1;
  const auto original_indices = indices, original_offsets = offsets, original_rows = rows;
  std::vector<unsigned> coverage(active);
  ForEachPipelineChunk(active, bytes, workers, [&](auto worker, auto chunk, Range ready) {
    const auto range = WorkerRange(active, bytes, worker, workers);
    CHECK(range.begin * bytes % 64 == 0 && ready.begin >= range.begin && ready.end <= range.end);
    CHECK(chunk < ChunkCount(range));
    for (auto i = ready.begin; i < ready.end; ++i) ++coverage[i];
    CHECK(CopyMappedRows(output.data(), staging.data(), rows.data(), ready, bytes));
  });
  CHECK(output == expected);
  for (const auto covered : coverage) CHECK(covered == 1);
  if ((capacity <= 17 && step == 1 && workers == 8) ||
      (capacity == 4096 && step == 1 && workers == 4) ||
      (capacity == 4103 && step == 2 && workers == 8))
    CheckDeviceKernel(addresses, indices, offsets, rows, active, bytes, workers, staging, expected);
  CHECK(indices == original_indices && offsets == original_offsets && rows == original_rows);
}

struct Model {
  std::vector<std::uint64_t> tokens;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> copied;
  std::uint32_t polls{}, finishes{};
  std::uint64_t ticks{};
  bool fail_copy{};
  Result status{Result::kPending};
  Result Status() { return status; }
  std::uint64_t Now() { return ++ticks; }
  std::uint64_t ReadRangeDone(std::uint32_t) {
    const auto index = std::min<std::size_t>(polls++, tokens.size() - 1);
    return tokens[index];
  }
  void Relax() {}
  bool Copy(Range range) {
    copied.emplace_back(range.begin, range.end);
    if (fail_copy) status = Result::kCopyError;
    return !fail_copy;
  }
  void FinishRange() { ++finishes; }
};

void CheckPipeline() {
  for (std::uint32_t worker = 0; worker < 4; ++worker) {
    const auto range = WorkerRange(10000, 656, worker, 4);
    auto check = [&](std::vector<std::uint64_t> tokens, Result expected,
                     std::vector<std::pair<std::uint32_t, std::uint32_t>> copied,
                     std::uint32_t chunks, std::uint32_t finishes, bool fail = false) {
      Model io;
      io.tokens = std::move(tokens); io.fail_copy = fail;
      std::uint32_t consumed = 0;
      CHECK(Consume(10000, 656, 4, worker, 7, 0, 30, io, &consumed) == expected);
      CHECK(io.copied == copied && consumed == chunks && io.finishes == finishes);
    };
    check({Progress(6, 1), Progress(7, 1), Progress(7, 3)}, Result::kOk,
          {{range.begin, range.begin + 1024}, {range.begin + 1024, range.end}}, 3, 1);
    check({Progress(7, 3)}, Result::kOk, {{range.begin, range.end}}, 3, 1);
    check({Progress(8, 1)}, Result::kSequenceError, {}, 0, 0);
    check({Progress(7, 0)}, Result::kSequenceError, {}, 0, 0);
    check({Progress(7, 4)}, Result::kSequenceError, {}, 0, 0);
    check({Progress(7, 2), Progress(7, 1)}, Result::kSequenceError,
          {{range.begin, range.begin + 2048}}, 2, 0);
    check({Progress(6, 1)}, Result::kTimeout, {}, 0, 0);
    check({Progress(7, 1)}, Result::kCopyError, {{range.begin, range.begin + 1024}}, 0, 0, true);
    check({Progress(7, 3)}, Result::kCopyError, {{range.begin, range.end}}, 0, 1, true);
  }
  // A single slow/stale Jetty must prevent release of the other three.
  for (const auto& tokens : std::vector<std::array<std::uint64_t, 4>>{
      {Progress(7, 3), Progress(7, 2), Progress(7, 1), Progress(7, 3)},
      {Progress(7, 3), Progress(6, 3), Progress(7, 3), Progress(7, 3)}}) {
    const auto progress = ChainReadyProgress(3, 7, [&](auto q) { return tokens[q]; });
    CHECK(progress == (tokens[1] == Progress(6, 3) ? 0 : Progress(7, 1)));
  }
  for (const auto sequence : {std::uint64_t{7}, std::uint64_t{UINT32_MAX}})
    for (const auto progress : {0U, 4U}) {
      const auto invalid = ChainReadyProgress(3, sequence, [&](auto) { return Progress(sequence, progress); });
      Model io; io.tokens = {invalid};
      std::uint32_t consumed = 0;
      CHECK(Consume(10000, 656, 4, 0, sequence, 0, 30, io, &consumed) == Result::kSequenceError);
      CHECK(io.copied.empty());
    }
}

void CheckRemoteFailure(std::uint64_t done_token, Result expected) {
  Layout layout;
  CHECK(MakeLayout(656, 1, &layout));
  std::vector<CacheLine> workspace((layout.bytes + 63) / 64);
  auto* memory = reinterpret_cast<std::uint8_t*>(workspace.data());
  reinterpret_cast<Completion*>(memory + layout.done)->token = done_token;
  Coordination coordination{};
  WorkerStatus status{};
  kvaf::BatchGetWorkerProfile profile{};
  std::vector<std::uint8_t> output(720, 0xA5);
  const std::uint64_t address = 4096;
  const std::int32_t index = 0, count = 1;
  KernelArgs args{};
  args.request = {Pointer(&address), Pointer(&index), Pointer(&index), Pointer(&count),
                  Pointer(&index), 1, 1, 656};
  args.workspace = Pointer(memory);
  args.output = Pointer(output.data());
  args.output_bytes = 656;
  args.coordination = Pointer(&coordination);
  args.status = Pointer(&status);
  args.profile = Pointer(&profile);
  args.sequence = args.workers = 1;
  args.timeout_ms = expected == Result::kTimeout ? 1 : 1000;
  CHECK(ExecuteCpuPushScatter(&args) != 0);
  CHECK(status.result == static_cast<std::uint64_t>(expected));
  CHECK(profile.scatter_copies == 0);
  CHECK(std::all_of(output.begin(), output.end(), [](auto byte) { return byte == 0xA5; }));
}

void CheckInputBounds() {
  Layout layout;
  CHECK(MakeLayout(16, 1, &layout));
  std::vector<CacheLine> workspace((layout.bytes + 63) / 64);
  auto* memory = reinterpret_cast<std::uint8_t*>(workspace.data());
  for (int scenario = 0; scenario < 8; ++scenario) {
    std::uint64_t address = 4096;
    std::int32_t index = 0, offset = 0, row = 0, count = 1;
    std::uint8_t valid = 1;
    std::uint8_t output[16];
    std::fill_n(output, 16, 0xA5);
    if (scenario == 0) index = -1;
    if (scenario == 1) index = 1;
    if (scenario == 2) offset = -1;
    if (scenario == 3) offset = 1;
    if (scenario == 4) valid = 0;
    if (scenario == 5) address = 4095;
    if (scenario == 6) row = 1;
    if (scenario == 7) address = 4096 + 24;
    Coordination coordination{};
    WorkerStatus status{};
    KernelArgs args{};
    args.request = {Pointer(&address),
                    Pointer(&index),
                    Pointer(&offset),
                    Pointer(&count),
                    Pointer(&row),
                    1,
                    1,
                    16};
    args.request.valid_table = Pointer(&valid);
    args.request.canonical_base = 4096;
    args.request.logical_bytes = 32;
    args.request.block_bytes = 16;
    args.workspace = Pointer(memory);
    args.output = Pointer(output);
    args.output_bytes = 16;
    args.coordination = Pointer(&coordination);
    args.status = Pointer(&status);
    args.sequence = args.workers = 1;
    CHECK(ExecuteCpuPushScatter(&args) != 0);
    CHECK(status.result == static_cast<std::uint64_t>(Result::kInvalidRequest));
    CHECK(std::all_of(output, output + 16, [](auto value) { return value == 0xA5; }));
  }
}

int main(int argc, char**) {
  CheckInputBounds();
  if (argc > 1) return 0;

  CheckPipeline();
  CheckRemoteFailure(Progress(1, static_cast<std::uint32_t>(Result::kTransportError)), Result::kTransportError);
  CheckRemoteFailure(Progress(2, 1), Result::kSequenceError);
  CheckRemoteFailure(0, Result::kTimeout);
  for (const auto capacity : {1U, 17U, 4096U, 4103U, kMaxRequests})
    for (const auto active : {0U, capacity * 3 / 4, capacity})
      for (const auto step : {1U, 2U})
        for (const auto workers : {1U, 4U, 8U})
          CheckRound(capacity, active, 656, step, workers, capacity + active + step + workers);
  for (const auto bytes : {1U, 17U, 64U, 4096U})
    CheckRound(4103, 3077, bytes, 2, 8, bytes);
  std::cout << "CPU push pipeline and dynamic Tensor scatter tests passed\n";
}
