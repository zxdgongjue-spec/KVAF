// Run the production AICPU entry with concurrent workers and a CPU producer.
// DMA is modeled as memcpy; real device cache behavior still requires A5 tests.
#include "get/scatter/urma_aicpu/kernel/cpu_push_scatter.h"
#include "get/scatter/urma_aicpu/kernel/cpu_push_pipeline.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

using namespace kvaf;
using namespace kvaf::cpu_push;
#define CHECK(v) do { if (!(v)) { std::cerr << "line " << __LINE__ << ": " #v "\n"; std::abort(); } } while (false)

struct Fixture {
  uint32_t capacity, workers, window, token;
  Layout layout;
  std::vector<Word> memory;
  Coordination coordination{};
  std::vector<WorkerStatus> reports;
  std::vector<BatchGetWorkerProfile> profiles;
  std::vector<uint8_t> source, output;
  std::vector<uint64_t> addresses;
  std::vector<int32_t> indices, offsets, rows;
  int32_t count{};
  uint64_t launches{}, seen{}, fail_at{};
  bool timeout{};
  std::atomic<bool> stop{false};
  std::thread cpu;
  KernelArgs args{};
  Fixture(uint32_t n, uint32_t w, uint32_t c, uint32_t bytes = 7)
      : capacity(n), workers(w), window(c), token(bytes), reports(w), profiles(w),
        source(std::size_t(n) * bytes), output(source.size()), addresses(n), indices(n), offsets(n), rows(n) {
    CHECK(MakeLayout(token, workers, &layout, window));
    memory.resize(layout.bytes / sizeof(Word));
    for (uint32_t i = 0; i < n; ++i) {
      indices[i] = i; rows[i] = n - 1 - i;
      addresses[i] = reinterpret_cast<uint64_t>(source.data() + std::size_t(i) * token);
      for (uint32_t b = 0; b < token; ++b) source[std::size_t(i) * token + b] = (i * 17U + b) % 251;
    }
    args.request.address_table = reinterpret_cast<uint64_t>(addresses.data());
    args.request.indices = reinterpret_cast<uint64_t>(indices.data());
    args.request.offsets = reinterpret_cast<uint64_t>(offsets.data());
    args.request.output_indices = reinterpret_cast<uint64_t>(rows.data());
    args.request.miss_count = reinterpret_cast<uint64_t>(&count);
    args.request.request_capacity = capacity; args.request.token_bytes = token;
    args.workspace = reinterpret_cast<uint64_t>(memory.data());
    args.coordination = reinterpret_cast<uint64_t>(&coordination);
    args.status = reinterpret_cast<uint64_t>(reports.data()); args.profile = reinterpret_cast<uint64_t>(profiles.data());
    args.output = reinterpret_cast<uint64_t>(output.data()); args.output_bytes = output.size();
    args.workers = workers; args.window_capacity = window; args.timeout_ms = 5000;
    cpu = std::thread([&] { Produce(); });
  }
  ~Fixture() { stop = true; cpu.join(); }
  uint8_t* Bytes() { return reinterpret_cast<uint8_t*>(memory.data()); }
  void Produce() {
    uint64_t last = 0;
    auto* notify = reinterpret_cast<Word*>(Bytes() + layout.notify);
    auto* done = reinterpret_cast<Completion*>(Bytes() + layout.done);
    while (!stop.load()) {
      const auto sequence = __atomic_load_n(&notify->value, __ATOMIC_ACQUIRE) & ~kProfileNotify;
      if (!sequence || sequence == last) { std::this_thread::yield(); continue; }
      CHECK(sequence > last); last = sequence; ++seen;
      const auto header = *reinterpret_cast<const Request*>(Bytes() + layout.request);
      CHECK(header.magic == kMagic && header.sequence == sequence && header.count <= window);
      if (fail_at && seen == fail_at) {
        if (!timeout) __atomic_store_n(&done->token, Progress(sequence, uint32_t(Result::kTransportError)), __ATOMIC_RELEASE);
        continue;
      }
      const auto* sources = reinterpret_cast<const uint64_t*>(Bytes() + layout.descriptor);
      ForEachPipelineChunk(header.count, token, workers, [&](uint32_t worker, uint32_t chunk, Range range) {
        for (uint32_t q = 0; q < kJettyCount; ++q) {
          const auto shard = JettyRange(range, q);
          for (auto i = shard.begin; i < shard.end; ++i) {
            CHECK(SourceRange(reinterpret_cast<uint64_t>(source.data()), source.size(), sources[i], token));
            std::memcpy(Bytes() + layout.staging + uint64_t(i) * token, reinterpret_cast<void*>(sources[i]), token);
          }
          auto* ready = reinterpret_cast<Ready*>(Bytes() + layout.ready) + worker;
          __atomic_store_n(&ready->jetty[q], Progress(sequence, chunk + 1), __ATOMIC_RELEASE);
        }
        std::this_thread::yield();
      });
      __atomic_store_n(&done->token, Progress(sequence, uint32_t(Result::kOk)), __ATOMIC_RELEASE);
    }
  }
  void Run(int32_t n, bool success = true) {
    count = n; ++launches;
    std::fill(output.begin(), output.end(), 0xa5);
    std::vector<std::thread> tasks;
    for (uint32_t w = 0; w < workers; ++w) tasks.emplace_back([&] { (void)ExecuteCpuPushScatter(&args); });
    for (auto& t : tasks) t.join();
    if (!success) { CHECK(coordination.result.value != 0); return; }
    CHECK(coordination.result.value == 0);
    CHECK(coordination.claimed.value == launches * workers);
    uint64_t completed = 0, spans = 0;
    for (uint32_t w = 0; w < workers; ++w) {
      CHECK(reports[w].magic == kMagic && reports[w].sequence == launches);
      CHECK(reports[w].result == uint64_t(Result::kOk));
      CHECK(reports[w].failed_request == UINT64_MAX);
      completed += reports[w].completed_requests; spans += profiles[w].spans;
    }
    CHECK(completed == uint32_t(n) && spans == uint32_t(n));
    for (uint32_t i = 0; i < capacity; ++i) for (uint32_t b = 0; b < token; ++b)
      CHECK(output[std::size_t(rows[i]) * token + b] ==
          (i < uint32_t(n) ? source[std::size_t(i) * token + b] : 0xa5));
  }
};

int main() {
  for (uint32_t workers : {1U, 5U, 8U}) for (uint32_t window : {4096U, 65536U, 131072U}) {
    Fixture f(270001, workers, window);
    for (int32_t n : {0, 1, 63, 64, 1023, 1024, 1025, 4095, 4096, 4097, 65537, 270001, 0, 1}) f.Run(n);
  }
  { Fixture f(1000000, 5, 4096); f.Run(1000000); }
  { Fixture f(8193, 5, 4096, 656); f.Run(8193); }
  { Fixture f(8193, 5, 4096, 4096); f.Run(8193); }
  { Fixture f(20000, 5, 4096); f.fail_at = 3; f.Run(20000, false);
    auto before = f.coordination.claimed.value; f.Run(1, false); CHECK(f.coordination.claimed.value == before); }
  { Fixture f(20000, 5, 4096); f.fail_at = 2; f.timeout = true; f.args.timeout_ms = 50; f.Run(20000, false); }
  { Fixture f(1, 5, 4096); f.Run(-1, false); }
  { Fixture f(1, 5, 4096); f.Run(2, false); }
  { Fixture f(1, 5, 4096); f.coordination.window_next.value = UINT32_MAX; f.Run(1, false); }
  std::cout << "AICPU Scatter: single-launch windows, dynamic/zero replays, global output mapping and faults passed\n";
}
