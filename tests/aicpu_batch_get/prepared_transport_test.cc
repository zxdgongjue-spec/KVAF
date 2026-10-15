#include "cpu_urma/indexed_async_write_tracker.h"
#include "cpu_urma/contiguous_write_spans.h"
#include "cpu_urma/sgl_batch_builder.h"
#include <cstdio>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <vector>

namespace {
using namespace kvaf::cpu_urma_detail;
using kvaf::CpuUrmaTransport;
using kvaf::Status;
using Tracker = IndexedAsyncWriteTracker;
void Check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void IndexedFences() {
  Tracker t(4, 4); Tracker::Handle a{}, b{}, c{};
  Tracker::Fence af[]{{0, 3, 0}, {1, 4, 0}}, bf[]{{0, 2, 0}, {2, 4, 0}}, cf{2, 4, 0};
  bool posted = false; std::uint64_t done = 0;
  Check(t.TryBegin(1, a, af, 2, &posted).ok() && posted, "first indexed reserve failed");
  const auto stale = af[0].context;
  Check(t.QueueMask() == 3 && t.Pending(0) == 1, "queue mask incorrect");
  Check(t.TryBegin(2, b, bf, 2, &posted).ok() && !posted && !b.token, "overcommitted credits");
  Check(t.TryBegin(3, c, &cf, 1, &posted).ok() && posted, "failed reserve leaked credits");
  Check(!t.Complete(2, af[0].context, &done).ok(), "wrong queue accepted");
  Check(!t.TryBegin(1, a, af, 2, &posted).ok(), "active plan mutated");
  Check(t.Complete(0, af[0].context, &done).ok() && !done && a.token, "partial batch exposed");
  Check(t.QueueMask() == 6 && !t.Pending(0), "drained queue still active");
  Check(!t.Complete(0, stale, &done).ok(), "duplicate tail accepted");
  Check(t.Complete(2, cf.context, &done).ok() && done == 3 && !c.token, "independent plan blocked");
  Check(t.TryBegin(2, b, bf, 2, &posted).ok() && posted, "released credits unavailable");
  Check(!t.Complete(0, stale, &done).ok(), "reused slot accepted stale generation");
  Check(t.Complete(2, bf[1].context, &done).ok() && !done, "one queue completed batch");
  Check(t.Complete(1, af[1].context, &done).ok() && done == 1 && !a.token, "last tail not fenced");
  Check(t.Complete(0, bf[0].context, &done).ok() && done == 2 && t.Empty(), "batch failed to drain");
  Check(!t.QueueMask(), "queue mask retained drained plans");
  Tracker::Fence bad[]{{0, 1, 0}, {0, 1, 0}};
  Check(!t.TryBegin(4, a, bad, 2, &posted).ok() && t.Empty(), "bad fence mutated state");
  Tracker::Fence too_large{3, 5, 0};
  Check(!t.TryBegin(4, a, &too_large, 1, &posted).ok() && !a.token, "oversized fence reserved");
  Check(!t.Complete(0, UINT64_C(0x5A00000000000000), &done).ok(), "non-async context accepted");
  Tracker full(1, 100);
  std::array<Tracker::Handle, 65> handles{};
  std::array<Tracker::Fence, 65> fences{};
  for (unsigned i = 0; i < 64; ++i) {
    fences[i] = {0, 1, 0};
    Check(full.TryBegin(i + 1, handles[i], &fences[i], 1, &posted).ok() && posted, "lost tracker slot");
  }
  fences[64] = {0, 1, 0};
  Check(full.TryBegin(65, handles[64], &fences[64], 1, &posted).ok() && !posted, "slot overflow");
  unsigned active = 0;
  full.ForEachActive([&](std::uint64_t) { ++active; }); Check(active == 64, "bitmap lost entries");
  for (unsigned i = 64; i; --i)
    Check(full.Complete(0, fences[i - 1].context, &done).ok() && done == i, "reverse drain failed");
  Check(full.TryBegin(65, handles[64], &fences[64], 1, &posted).ok() && posted, "no free slot after drain");
  Check(!full.Complete(0, fences[0].context, &done).ok(), "old CQ accepted after reuse");
  Check(full.Complete(0, fences[64].context, &done).ok() && full.Empty(), "reused plan did not drain");
}

void BatchedCqSweeps() {
  for (auto capacity : {1U, 2U, 7U}) for (auto batch_size : {1U, 4U, 64U}) {
    Tracker tracker(4, 100);
    std::array<Tracker::Handle, 17> handles{};
    std::array<std::deque<std::uint64_t>, 4> pending;
    bool posted = false;
    for (unsigned i = 0; i < 17; ++i) {
      Tracker::Fence fences[]{{0, 1, 0}, {2, 2, 0}};
      Check(tracker.TryBegin(i + 1, handles[i], fences, 2, &posted).ok() && posted, "sweep reserve failed");
      pending[0].push_back(fences[0].context);
      pending[2].push_front(fences[1].context); // Reordered multi-Jetty tails.
    }
    std::array<std::uint64_t, 64> read{};
    std::vector<std::uint64_t> finished;
    unsigned sweep = 0, reads = 0;
    while (!tracker.Empty()) {
      Check(sweep < 100, "CQ sweep deadlocked");
      std::size_t count = 0;
      const auto before = finished.size();
      auto status = SweepAsyncCqs(4, sweep++ % 4, batch_size, capacity, &count,
          [&](std::size_t q) { return tracker.Pending(q); },
          [&](std::size_t q, std::size_t limit, std::size_t* n) {
            Check(q == 0 || q == 2, "polled idle queue"); ++reads;
            // Exercise empty active CQs as well as bursts.
            *n = sweep % 5 == 0 ? 0 : std::min(limit, pending[q].size());
            for (std::size_t i = 0; i < *n; ++i) { read[i] = pending[q].front(); pending[q].pop_front(); }
            return Status::OK();
          },
          [&](std::size_t q, std::size_t i, bool* completed) {
            std::uint64_t id = 0; auto status = tracker.Complete(q, read[i], &id);
            if (id) { *completed = true; finished.push_back(id); }
            return status;
          });
      Check(status.ok() && count <= capacity && finished.size() - before == count, "lost/overflowed completion");
    }
    std::sort(finished.begin(), finished.end());
    Check(finished.size() == 17 && reads, "wrong completion total");
    for (unsigned i = 0; i < 17; ++i) Check(finished[i] == i + 1, "duplicate/missing completion");
    Check(pending[0].empty() && pending[2].empty(), "CQEs left after drain");
  }
  std::size_t count = 0;
  Check(!SweepAsyncCqs(1, 0, 8, 1, &count, [](auto) { return std::size_t{1}; },
      [](auto, auto limit, auto* n) { *n = limit + 1; return Status::OK(); },
      [](auto, auto, auto*) { return Status::OK(); }).ok(), "invalid provider count accepted");
  bool consumed = false;
  Check(!SweepAsyncCqs(1, 0, 8, 1, &count, [](auto) { return std::size_t{1}; },
      [](auto, auto, auto*) { return Status::Internal("simulated CQ error"); },
      [&](auto, auto, auto*) { consumed = true; return Status::OK(); }).ok() && !consumed,
      "CQ error consumed a completion");
}

void ContiguousGather() {
  std::vector<unsigned char> memory(8192), output(8192);
  for (std::size_t i = 0; i < memory.size(); ++i) memory[i] = static_cast<unsigned char>(i * 17);
  std::vector<CpuUrmaTransport::HostSge> input, merged;
  std::vector<std::uint64_t> offsets, merged_offsets;
  for (unsigned i = 0; i < 8; ++i) { input.push_back({memory.data() + i * 656, 656}); offsets.push_back(i * 656); }
  const auto segment = [&](const void* p, std::size_t n) -> unsigned {
    const auto b = reinterpret_cast<std::uintptr_t>(p), base = reinterpret_cast<std::uintptr_t>(memory.data());
    if (b >= base && b - base <= memory.size() && n <= memory.size() - (b - base)) return 1;
    return 0;
  };
  auto merge = [&](std::uint64_t max_bytes, auto registration) {
    return CoalesceWriteSpans(input.data(), offsets.data(), input.size(), output.size(), max_bytes,
                              registration, &merged, &merged_offsets);
  };
  Check(merge(2048, segment).ok() && merged.size() == 3 && merged[0].bytes == 1968, "message boundary merge failed");
  for (std::size_t i = 0; i < merged.size(); ++i)
    std::memcpy(output.data() + merged_offsets[i], merged[i].data, merged[i].bytes);
  Check(!std::memcmp(output.data(), memory.data(), 8 * 656), "merged gather payload changed");
  Check(merge(65536, segment).ok() && merged.size() == 1, "natural contiguous run not merged");
  std::size_t wrs = 0;
  Check(BuildWriteGatherShard(merged.data(), merged_offsets.data(), merged.size(), 0, merged.size(), 13,
      output.size(), 65536, [](auto, auto, const auto&) {}, [](auto, const auto&) {}, &wrs).ok() && wrs == 1,
      "merged spans failed descriptor builder");
  // Segment boundary is respected even for adjacent virtual addresses.
  const auto boundary = [&](const void* p, std::size_t n) -> unsigned {
    const auto b = reinterpret_cast<std::uintptr_t>(p) - reinterpret_cast<std::uintptr_t>(memory.data());
    if (b < 4 * 656 && n <= 4 * 656 - b) return 1;
    if (b >= 4 * 656 && b <= memory.size() && n <= memory.size() - b) return 2;
    return 0;
  };
  Check(merge(65536, boundary).ok() && merged.size() == 2, "crossed segment boundary");
  input[1].data = memory.data() + 700;
  Check(merge(65536, segment).ok() && merged.size() == 3, "source gap discarded");
  input[1].data = memory.data() + 656; offsets[1] = 700;
  Check(merge(65536, segment).ok() && merged.size() == 3, "remote gap discarded");
  offsets[1] = 656;
  std::reverse(input.begin(), input.end());
  Check(merge(65536, segment).ok() && merged.size() == input.size(), "reordered reverse SGL");
  input[0].bytes = 0; Check(!merge(65536, segment).ok(), "zero span accepted");
  input[0].bytes = 656; Check(!merge(655, segment).ok(), "oversized span accepted");
  Check(!merge(65536, [](auto, auto) { return 0; }).ok(), "unregistered span accepted");
}
}  // namespace
int main() {
  try { IndexedFences(); BatchedCqSweeps(); ContiguousGather(); std::puts("prepared transport optimization: PASS"); return 0; }
  catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
