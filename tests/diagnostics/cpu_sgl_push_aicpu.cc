#define KVAF_CPU_SGL_REUSE_FIXTURE
#include "cpu_sgl_write.cc"
#undef KVAF_CPU_SGL_REUSE_FIXTURE
#include <iostream>

namespace {
void RunPush(Fixture& f, Result expected = Result::kOk) {
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
  if (expected != Result::kOk && f.rows[0] == 0) {
    // Force the head to be accepted before another worker rejects the suffix.
    while ((__atomic_load_n(reinterpret_cast<uint64_t*>(remote + 64), __ATOMIC_ACQUIRE) & ~kProfileNotify) != sequence)
      assert(std::chrono::steady_clock::now() < deadline);
  }
  for (uint32_t w = 1; w < f.args.workers; ++w) launch();
  if (f.count && (expected == Result::kOk || f.rows[0] == 0)) {
    auto* notify = reinterpret_cast<uint64_t*>(remote + 64);
    while ((__atomic_load_n(notify, __ATOMIC_ACQUIRE) & ~kProfileNotify) != sequence)
      assert(std::chrono::steady_clock::now() < deadline);
    auto* request = reinterpret_cast<Request*>(remote);
    assert(request->count == uint32_t(f.count) && request->sequence == sequence && request->mode == kDirectOutput);
    const auto head = std::min(uint32_t(f.count), kPipelineHeadRows);
    assert(request->reserved_words[0] == head);
    const auto* target = reinterpret_cast<OutputTarget*>(remote + f.layout.output_target);
    assert(target->base == f.args.output);
    const auto* sources = reinterpret_cast<uint64_t*>(remote + f.layout.descriptor);
    auto copy = [&](uint32_t first, uint32_t last) {
      for (uint32_t i = first; i < last; ++i) {
        assert(sources[i] == f.addresses[f.indices[i]] + uint32_t(f.offsets[i]));
        std::memcpy(f.output.data() + i * kBytes, reinterpret_cast<void*>(sources[i]), kBytes);
      }
    };
    copy(0, head);
    if (uint32_t(f.count) > head) {
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
  }
  assert(reported);
  if (expected == Result::kOk) for (int i = 0; i < f.count; ++i)
    assert(!std::memcmp(f.output.data() + i * kBytes, reinterpret_cast<void*>(f.addresses[f.indices[i]] + f.offsets[i]), kBytes));
}
}
int main() {
  for (uint32_t workers : {1U, 4U, 8U}) {
    Fixture f(workers);
    for (uint32_t count : {4097U, 1024U, 1025U, 7U, 0U}) { f.count = count; RunPush(f); }
  }
  { Fixture f(4); f.rows[0] = -1; RunPush(f, Result::kInvalidRequest); }
  { Fixture f(4); f.rows[3000] = 0; RunPush(f, Result::kInvalidRequest); }
  std::cout << "AICPU device-push SGL mailbox model passed\n";
}
