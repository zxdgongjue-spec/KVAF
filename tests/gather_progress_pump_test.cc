#include "gather_progress_pump.h"
#include "cpu_gather_executor.h"
#include <array>
#include <cassert>
#include <cstring>
#include <deque>
#include <iostream>
#include <thread>
#include <vector>

using namespace kvaf::peer;
struct Backend {
  struct Write { uint32_t slot, bytes; uint64_t id; std::array<uint64_t, 64> data; };
  alignas(64) uint64_t frames[128]{}, remote[64]{};
  std::deque<Write> live;
  std::deque<uint64_t> cq;
  bool fail_post{}, fail_poll{}, eager{};
  static bool Post(void* p, uint32_t slot, uint64_t id, uint32_t bytes) {
    auto& b = *static_cast<Backend*>(p);
    Write w{slot, bytes, id, {}};
    std::memcpy(w.data.data(), b.frames + slot * 64, bytes);
    b.live.push_back(w); // failed posts may still retain a DMA source
    if (b.fail_post) return false;
    if (b.eager) b.Complete();
    return true;
  }
  void Complete() {
    const auto& w = live.front();
    assert(std::memcmp(frames + w.slot * 64, w.data.data(), w.bytes) == 0);
    std::memcpy(remote, w.data.data(), w.bytes);
    cq.push_back(w.id);
    live.pop_front();
  }
  static int Poll(void* p, uint64_t* id) {
    auto& b = *static_cast<Backend*>(p);
    if (b.fail_poll) return -1;
    if (b.cq.empty()) return 0;
    *id = b.cq.front(); b.cq.pop_front(); return 1;
  }
  GatherProgressPump::Ops ops() { return {this, Post, Poll}; }
};
int main() {
  {
    Backend b; GatherProgressPump pump(b.frames, b.ops());
    uint64_t progress[64]{};
    assert(pump.Advance(7, progress, 64) && !pump.posts());
    progress[63] = kvaf::GatherShardProgress(7, 2);
    assert(pump.Advance(7, progress, 64) && pump.posts() == 1);
    progress[0] = kvaf::GatherShardProgress(7, 3);
    assert(pump.Advance(7, progress, 64) && pump.posts() == 2);
    progress[0] = kvaf::GatherShardProgress(7, 9);
    assert(pump.Advance(7, progress, 64) && pump.posts() == 2 && !pump.Latest());
    b.Complete();
    assert(pump.Advance(7, progress, 64) && pump.posts() == 3 && pump.Latest());
    b.Complete(); b.Complete();
    assert(pump.Reap() && !pump.Pending() && b.remote[0] == progress[0]);
    assert(b.remote[63] == progress[63]);
    std::memset(progress, 0, sizeof(progress));
    progress[7] = kvaf::GatherShardProgress(8, 1);
    assert(pump.Advance(8, progress, 8)); b.Complete(); assert(pump.Reap());
    assert(b.remote[0] == 0 && b.remote[7] == progress[7]);
  }
  for (int failure = 0; failure < 4; ++failure) {
    Backend b; GatherProgressPump pump(b.frames, b.ops());
    uint64_t progress[1]{kvaf::GatherShardProgress(2, 1)};
    b.fail_post = failure == 0;
    if (!b.fail_post) assert(pump.Advance(2, progress, 1));
    if (failure == 0) assert(!pump.Advance(2, progress, 1));
    if (failure == 1) { b.fail_poll = true; assert(!pump.Reap()); }
    if (failure == 2) { b.cq.push_back(999); assert(!pump.Reap()); }
    if (failure == 3) { progress[0] = kvaf::GatherShardProgress(3, 2); assert(!pump.Advance(2, progress, 1)); }
    assert(!pump.ok());
    // A failed sender cannot overwrite its potentially live source buffer.
    const auto before = b.frames[0]; progress[0] = 0;
    assert(!pump.Advance(2, progress, 1) && b.frames[0] == before);
  }
  // Real copy executor: only its coordinator publishes, and every advertised
  // source range must already have the correct copied bytes.
  {
    constexpr uint32_t rows = 4097, token = 656, sgl = 6;
    constexpr uint64_t payload = uint64_t(rows) * token;
    const auto source = kvaf::GatherProgressOffset(rows) + kvaf::kGatherProgressBytes;
    void* allocation = nullptr;
    const auto real_staging = (source + payload + 63) & ~uint64_t(63);
    const auto real_bytes = ((real_staging + payload + 63) & ~uint64_t(63)) + 64;
    assert(!posix_memalign(&allocation, 64, real_bytes));
    auto* memory = static_cast<uint8_t*>(allocation); std::memset(memory, 0, real_bytes);
    const auto real_base = reinterpret_cast<uint64_t>(memory);
    auto* h = reinterpret_cast<GatherDescriptorHeader*>(memory);
    auto* spans = reinterpret_cast<GatherSpan*>(h + 1);
    *h = {kvaf::kGatherShardMagic, 1, real_base + real_staging, rows, uint32_t(payload), 65536, token, sgl};
    for (uint64_t i = 0; i < payload; ++i) memory[source + i] = uint8_t((i * 7 + 3) % 251);
    for (uint32_t i = 0; i < rows; ++i) spans[i] = {real_base + source + uint64_t(i) * token, token, 0};
    struct Context {
      Backend backend;
      GatherProgressPump pump{backend.frames, backend.ops()};
      const uint8_t* memory; uint64_t source, staging;
      std::thread::id coordinator{std::this_thread::get_id()};
      uint64_t calls{};
      Context(const uint8_t* m, uint64_t src, uint64_t dst) : memory(m), source(src), staging(dst) {}
    } c(memory, source, real_staging);
    c.backend.eager = true;
    {
      CpuGatherExecutor executor(memory, real_base, 8, 4096, GatherWaitMode::kActive);
      executor.SetProgressHook(&c, [](void* p, uint64_t sequence, const uint64_t* progress, uint32_t shards) {
        auto& c = *static_cast<Context*>(p);
        assert(c.coordinator == std::this_thread::get_id()); ++c.calls;
        assert(c.pump.Advance(sequence, progress, shards));
        const auto total = (rows + sgl - 1) / sgl;
        for (uint32_t s = 0; s < shards; ++s) {
          const auto done = uint32_t(c.backend.remote[s]);
          const uint64_t first = kvaf::GatherShardBoundary(total, shards, s) * uint64_t(sgl) * token;
          const auto end = std::min<uint64_t>(uint64_t(rows) * token, first + uint64_t(done) * sgl * token);
          if (done) assert(!std::memcmp(c.memory + c.source + first, c.memory + c.staging + first, end - first));
        }
      });
      assert(executor.PrepareDirect(real_bytes, real_base, 1, rows) == 8);
      executor.Copy(); assert(c.pump.Reap());
      assert(c.calls > 1 && c.pump.posts() > 1 && !c.pump.Pending());
      assert(!std::memcmp(memory + source, memory + real_staging, payload));
    }
    std::free(allocation);
  }
  std::cout << "gather progress snapshots: PASS\n";
}
