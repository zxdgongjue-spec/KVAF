#define KVAF_CPU_SGL_REUSE_FIXTURE
#include "cpu_sgl_write.cc"
#undef KVAF_CPU_SGL_REUSE_FIXTURE
#include <iostream>
#include "cpu_sgl_push_layout.h"
namespace mailbox_model { extern std::uint64_t first_wait_ns; }

namespace {
void RunPush(Fixture& f, Result expected = Result::kOk) {
  mailbox_model::first_wait_ns = 0;
  const bool chunks = (f.args.sequence & kChunkPipelineLaunch) != 0;
  const bool cooperative = chunks && PushLaunchCooperative(f.args.sequence);
  const auto head_rows = chunks ? PushLaunchHeadRows(f.args.sequence) : kChunkRows;
  std::vector<Word> mailbox(f.layout.staging / 64);
  const auto remote = reinterpret_cast<uint64_t>(mailbox.data());
  CpuSglPushArgs pushed{f.direct_args, {remote, f.layout.staging, 0, 1, 1}};
  const auto sequence = f.coordination.claimed.value / f.args.workers + 1;
  std::vector<std::future<uint32_t>> kernels;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  auto launch = [&] {
    kernels.push_back(std::async(std::launch::async, [pushed]() mutable { return ExecuteCpuSglPush(&pushed); }));
  };
  launch();
  if (!cooperative && expected != Result::kOk && f.rows[0] == 0) {
    // Force the head to be accepted before another worker rejects the suffix.
    while ((__atomic_load_n(reinterpret_cast<uint64_t*>(remote + 64), __ATOMIC_ACQUIRE) & ~kProfileNotify) != sequence)
      assert(std::chrono::steady_clock::now() < deadline);
  }
  for (uint32_t w = 1; w < f.args.workers; ++w) launch();
  bool accepted = f.count && (expected == Result::kOk || f.rows[0] == 0);
  if (cooperative && f.count && expected != Result::kOk) {
    // A builder can reject a later partition before the collective Head is
    // accepted. Both early failure and failure after publication must finish.
    while ((__atomic_load_n(reinterpret_cast<uint64_t*>(remote + 64), __ATOMIC_ACQUIRE) & ~kProfileNotify) != sequence &&
           __atomic_load_n(&f.coordination.finished.value, __ATOMIC_ACQUIRE) < sequence)
      assert(std::chrono::steady_clock::now() < deadline);
    accepted = (__atomic_load_n(reinterpret_cast<uint64_t*>(remote + 64), __ATOMIC_ACQUIRE) & ~kProfileNotify) == sequence;
  }
  if (accepted) {
    auto* notify = reinterpret_cast<uint64_t*>(remote + 64);
    while ((__atomic_load_n(notify, __ATOMIC_ACQUIRE) & ~kProfileNotify) != sequence)
      assert(std::chrono::steady_clock::now() < deadline);
    auto* request = reinterpret_cast<Request*>(remote);
    assert(request->count == uint32_t(f.count) && request->sequence == sequence && request->mode == (chunks ? kChunkedOutput : kDirectOutput));
    const auto head = std::min(uint32_t(f.count), kPipelineHeadRows);
    assert(request->reserved_words[0] == (chunks ? kChunkRows : head));
    assert(request->reserved_words[1] == (chunks && head_rows < kChunkRows ? head_rows : 0));
    assert(request->reserved_words[2] == (cooperative ? 1U : 0U));
    const auto* target = reinterpret_cast<OutputTarget*>(remote + f.layout.output_target);
    assert(target->base == f.args.output);
    const auto* sources = reinterpret_cast<uint64_t*>(remote + f.layout.descriptor);
    auto copy = [&](uint32_t first, uint32_t last) {
      for (uint32_t i = first; i < last; ++i) {
        assert(sources[i] == f.addresses[f.indices[i]] + uint32_t(f.offsets[i]));
        std::memcpy(f.output.data() + i * kBytes, reinterpret_cast<void*>(sources[i]), kBytes);
      }
    };
    if (chunks) {
      const auto total = PushChunkCount(uint32_t(f.count), head_rows);
      std::vector<bool> copied(total);
      uint32_t completed = 0; bool failed = false;
      while (completed < total && !failed) {
        assert(std::chrono::steady_clock::now() < deadline);
        for (uint32_t w = 0; w < f.args.workers; ++w) {
          const auto token = __atomic_load_n(reinterpret_cast<uint64_t*>(remote + f.layout.ready + uint64_t(w) * 64), __ATOMIC_ACQUIRE);
          if ((token >> 32) != sequence) continue;
          if (uint32_t(token) & kChunkFailed) { failed = true; break; }
          for (uint32_t c = w; c < total && c / f.args.workers < uint32_t(token); c += f.args.workers)
            if (!copied[c]) { copy(PushChunkBegin(c, head_rows), PushChunkEnd(uint32_t(f.count), c, head_rows)); copied[c] = true; ++completed; }
        }
      }
      assert(failed == (expected != Result::kOk));
    } else copy(0, head);
    if (!chunks && uint32_t(f.count) > head) {
      uint64_t token;
      do { token = __atomic_load_n(reinterpret_cast<uint64_t*>(remote + kSuffixNotifyOffset), __ATOMIC_ACQUIRE);
           assert(std::chrono::steady_clock::now() < deadline); } while ((token >> 32) != sequence);
      assert(uint32_t(token) == uint32_t(expected));
      if (expected == Result::kOk) copy(head, f.count);
    }
    if (expected == Result::kOk)
      for (auto& kernel : kernels) assert(kernel.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout);
    else assert(kernels.front().wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout);
    auto* done = reinterpret_cast<Completion*>(f.memory() + f.layout.done);
    done->cpu_sgl_write = {1000, 10, 1, 4};
    __atomic_store_n(&done->token, Progress(sequence, uint32_t(expected)), __ATOMIC_RELEASE);
  }
  for (auto& kernel : kernels) { assert(kernel.wait_for(std::chrono::seconds(5)) == std::future_status::ready); assert((kernel.get() == 0) == (expected == Result::kOk)); }
  bool reported = false;
  for (uint32_t w = 0; w < f.args.workers; ++w) {
    const auto& status = f.statuses[w];
    reported |= status.result == uint64_t(expected) && status.sequence == sequence;
    if (expected == Result::kOk) assert(status.result == uint64_t(expected) && status.sequence == sequence);
    assert(f.profiles[w].scatter_copies == 0 && f.profiles[w].ready_queries == 0);
    if (expected == Result::kOk && f.args.profile) {
      const auto& p = f.profiles[w];
      if (cooperative) {
        uint64_t rows = 0;
        for (uint32_t c = 0; c < PushChunkCount(uint32_t(f.count), head_rows); ++c) {
          const auto begin = PushChunkBegin(c, head_rows), end = PushChunkEnd(uint32_t(f.count), c, head_rows);
          rows += PushBuildBoundary(begin, end, w + 1, f.args.workers) - PushBuildBoundary(begin, end, w, f.args.workers);
        }
        assert(p.spans == rows);
      }
      assert(HasCpuSglPushHeadProfile(p));
      if (w || !f.count) assert(p.head_build_ns == 0 && p.head_doorbell_ns == 0);
      else assert(p.head_build_ns > 0 && p.head_doorbell_ns >= p.head_build_ns &&
                  p.head_doorbell_ns <= p.exit_ns - p.enter_ns &&
                  p.enter_ns + p.head_doorbell_ns <= mailbox_model::first_wait_ns);
    }
  }
  assert(reported);
  if (expected == Result::kOk) for (int i = 0; i < f.count; ++i)
    assert(!std::memcmp(f.output.data() + i * kBytes, reinterpret_cast<void*>(f.addresses[f.indices[i]] + f.offsets[i]), kBytes));
}
}
void ConfigureHead(Fixture& f, uint32_t head, bool cooperative = false) {
  std::free(f.workspace); f.workspace = std::aligned_alloc(64, f.layout.bytes + PushWorkspaceBytes(head));
  std::memset(f.workspace, 0, f.layout.bytes + PushWorkspaceBytes(head));
  f.args.workspace = reinterpret_cast<uint64_t>(f.workspace); f.args.sequence = PushLaunch(head, cooperative);
}
int main() {
  for (uint32_t head : {128U, 256U, 512U, 1024U}) for (uint32_t workers : {1U, 4U, 5U, 8U}) {
    Fixture f(workers); ConfigureHead(f, head, true);
    for (uint32_t count : {7U, head, head + 1, head + 1024, head + 1025, 0U, 4097U}) { f.count = count; RunPush(f); }
  }
  for (uint32_t count : {1U, 7U, 4097U}) { Fixture f(8); ConfigureHead(f, 8, true); f.count = count; RunPush(f); }
  { Fixture f(4, 65536); ConfigureHead(f, 256, true); f.count = 65536; RunPush(f); }
  { Fixture f(4); ConfigureHead(f, 512, true); f.rows[511] = -1; RunPush(f, Result::kInvalidRequest); }
  // With one builder, the Head is guaranteed to have been posted before the
  // later chunk fails, exercising error notification and draining accepted DMA.
  { Fixture f(1); ConfigureHead(f, 512, true); f.rows[3000] = -1; RunPush(f, Result::kInvalidRequest); }

  for (uint32_t head : {128U, 256U, 512U}) for (uint32_t workers : {1U, 4U, 5U, 8U}) {
    Fixture f(workers); ConfigureHead(f, head);
    for (uint32_t count : {7U, head, head + 1, head + 1024, head + 1025, 0U, 4097U}) { f.count = count; RunPush(f); }
  }
  for (uint32_t head : {128U, 256U, 512U}) {
    Fixture f(4); ConfigureHead(f, head); f.rows[3000] = -1; RunPush(f, Result::kInvalidRequest);
  }
  for (uint32_t head : {1U, 7U, 1023U}) { Fixture f(4); ConfigureHead(f, head); RunPush(f); }
  { Fixture f(4, 65536); ConfigureHead(f, 256); f.count = 65536; RunPush(f); }

  for (uint32_t workers : {1U, 4U, 5U, 8U}) {
    Fixture f(workers);
    std::free(f.workspace); f.workspace = std::aligned_alloc(64, f.layout.bytes + sizeof(ChunkPushState));
    std::memset(f.workspace, 0, f.layout.bytes + sizeof(ChunkPushState));
    f.args.workspace = reinterpret_cast<uint64_t>(f.workspace); f.args.sequence = kChunkPipelineLaunch;
    for (uint32_t count : {4097U, 1024U, 1025U, 7U, 0U, 4097U}) { f.count = count; RunPush(f); }
  }
  {
    Fixture f(4); std::free(f.workspace); f.workspace = std::aligned_alloc(64, f.layout.bytes + sizeof(ChunkPushState));
    std::memset(f.workspace, 0, f.layout.bytes + sizeof(ChunkPushState));
    f.args.workspace = reinterpret_cast<uint64_t>(f.workspace); f.args.sequence = kChunkPipelineLaunch;
    f.rows[3000] = 0; RunPush(f, Result::kInvalidRequest);
  }
  for (uint32_t workers : {1U, 4U, 8U}) {
    Fixture f(workers);
    for (uint32_t count : {4097U, 1024U, 1025U, 7U, 0U}) { f.count = count; RunPush(f); }
  }
  { Fixture f(4); f.rows[0] = -1; RunPush(f, Result::kInvalidRequest); }
  { Fixture f(4); f.rows[3000] = 0; RunPush(f, Result::kInvalidRequest); }
  { Fixture f(4); f.count = 7; f.args.profile = 0; RunPush(f);
    for (const auto& p : f.profiles) assert(!p.head_build_ns && !p.head_doorbell_ns && !p.head_profile_tag); }
  std::cout << "AICPU device-push SGL mailbox model passed\n";
}
