#define KVAF_CPU_SGL_REUSE_FIXTURE
#include "cpu_sgl_write_aiv.cc"
#undef KVAF_CPU_SGL_REUSE_FIXTURE
#include <deque>
#include <functional>
#include <array>

namespace {
struct PushFixture : Fixture {
  std::vector<Word> mailbox;
  AivMailboxContext context{};
  std::vector<uint32_t> sq, cq;
  uint32_t sq_db{}, cq_db{}, consumed{}, produced{};
  uint64_t head_commit_cycle{};
  uint64_t later_chunk_cycle{};
  bool chunks{}, cooperative{};
  std::array<std::atomic<uint32_t>, kMaxWorkers> built_rows{}, built_head_rows{};
  uint32_t head_rows{kChunkRows};
  bool fail_first_control{};
  std::vector<std::function<void()>> source_checks;
  std::deque<std::vector<std::function<void()>>> pending_controls;
  uint32_t max_pending{};
  static inline PushFixture* active{};
  explicit PushFixture(uint32_t workers, bool chunked = false, uint32_t head = kChunkRows, uint32_t capacity = kCapacity, bool coop = false) : Fixture(workers, capacity), mailbox(layout.staging / 64), sq(64 * 16), cq(64 * 8), chunks(chunked), cooperative(coop), head_rows(head) {
    args.execution.sequence = chunks ? PushLaunch(head_rows, cooperative) : kPipelineLaunch;
    if (chunks) {
      workspace.resize((layout.bytes + PushWorkspaceBytes(head_rows)) / 64);
      args.execution.workspace = reinterpret_cast<uint64_t>(workspace.data());
      sgl_model::descriptor = args.execution.workspace + layout.descriptor;
      sgl_model::descriptor_end = args.execution.workspace + layout.staging;
    }
    context.max_write_bytes = 4096; // Exercise split descriptors and multiple WQE batches.
    auto& e = context.entity;
    e.magic = 0x4b564155; e.version = 1; e.local_token_id = 1;
    e.remote_host_ddr_address = reinterpret_cast<uint64_t>(mailbox.data());
    e.remote_host_ddr_size = layout.staging;
    e.sq_va = reinterpret_cast<uint64_t>(sq.data()); e.sq_db_va = reinterpret_cast<uint64_t>(&sq_db);
    e.sq_depth = 64; e.wqe_size = 64;
    e.cq_va = reinterpret_cast<uint64_t>(cq.data()); e.cq_db_va = reinterpret_cast<uint64_t>(&cq_db);
    e.cq_depth = 64; e.cqe_size = 32;
    sgl_model::sq_begin = e.sq_va; sgl_model::sq_end = e.sq_va + sq.size() * 4;
    active = this; sgl_model::doorbell = Doorbell; sgl_model::descriptor_copy = DescriptorCopy;
  }
  ~PushFixture() { sgl_model::doorbell = nullptr; sgl_model::descriptor_copy = nullptr; active = nullptr; }
  static void DescriptorCopy(uint32_t core, uint64_t address, uint32_t bytes) {
    auto& f = *active;
    if (!f.chunks) return;
    const auto row = uint32_t((address - f.args.execution.workspace - f.layout.descriptor) / 8);
    const auto chunk = PushChunkIndex(row, f.head_rows);
    const auto first = PushChunkBegin(chunk, f.head_rows), last = PushChunkEnd(f.count, chunk, f.head_rows);
    const auto begin = f.cooperative ? PushBuildBoundary(first, last, core, f.args.execution.workers) : first;
    const auto end = f.cooperative ? PushBuildBoundary(first, last, core + 1, f.args.execution.workers) : last;
    assert(row == begin && bytes == (end - begin) * 8);
    if (!f.cooperative) assert(core == chunk % f.args.execution.workers);
    f.built_rows[core] += bytes / 8;
    if (!chunk) f.built_head_rows[core] += bytes / 8;
  }
  void Check(std::vector<std::future<void>>& work, uint64_t sequence, Result result) {
    Fixture::Check(work, sequence, result);
    if (result != Result::kOk || !chunks) return;
    for (uint32_t w = 0; w < args.execution.workers; ++w) {
      uint32_t expected = 0;
      for (uint32_t c = 0; c < PushChunkCount(count, head_rows); ++c) {
        const auto b = PushChunkBegin(c, head_rows), e = PushChunkEnd(count, c, head_rows);
        if (cooperative) expected += PushBuildBoundary(b, e, w + 1, args.execution.workers) - PushBuildBoundary(b, e, w, args.execution.workers);
        else if (c % args.execution.workers == w) expected += e - b;
      }
      assert(built_rows[w] == expected && profiles[w].spans == expected);
    }
  }
  static void Doorbell(uint32_t value, uint64_t address) {
    auto& f = *active;
    if (address == reinterpret_cast<uint64_t>(&f.cq_db)) {
      assert(!f.pending_controls.empty());
      for (const auto& check : f.pending_controls.front()) check();
      f.pending_controls.pop_front();
      return;
    }
    assert(address == reinterpret_cast<uint64_t>(&f.sq_db));
    while (f.consumed < value) {
      const auto* w = f.sq.data() + (f.consumed++ % 64) * 16;
      const uint64_t remote = uint64_t(w[10]) | (uint64_t(w[11]) << 32);
      const uint64_t local = uint64_t(w[14]) | (uint64_t(w[15]) << 32);
      const auto bytes = w[12];
      const auto base = reinterpret_cast<uint64_t>(f.mailbox.data());
      if (f.chunks && !f.later_chunk_cycle && remote >= base + f.layout.descriptor + uint64_t(f.head_rows) * 8)
        f.later_chunk_cycle = AscendC::GetSystemCycle();
      assert(bytes && bytes <= f.context.max_write_bytes && remote >= base && remote + bytes <= base + f.layout.staging);
      assert(local >= f.args.execution.workspace && local + bytes <= f.args.execution.workspace + f.workspace.size() * 64);
      // Do not overwrite CPU-owned done/Ready sources.
      assert(remote + bytes <= base + 128 || remote >= base + f.layout.output_target ||
             (f.chunks && bytes == 8 && remote >= base + 192));
      const std::vector<uint8_t> snapshot(reinterpret_cast<uint8_t*>(local), reinterpret_cast<uint8_t*>(local) + bytes);
      f.source_checks.push_back([snapshot, local, bytes] { assert(!std::memcmp(snapshot.data(), reinterpret_cast<void*>(local), bytes)); });
      if (bytes == 8) {
        if (remote == base + 64) {
          if (f.chunks) {
            // Head must leave before clearing replay bookkeeping or bouncing
            // the publisher's own Ready line through GM. Launch poisons this
            // private state every round, including after a non-empty replay.
            const auto* state = reinterpret_cast<const ChunkPushState*>(f.args.execution.workspace + f.layout.staging);
            const auto* bytes = reinterpret_cast<const uint8_t*>(state->sent);
            for (size_t i = 0; i < sizeof(state->sent) + sizeof(state->state); ++i) assert(bytes[i] == 0xa5);
            const auto sequence = reinterpret_cast<const Request*>(base)->sequence;
            if (f.cooperative) {
              for (uint32_t core = 0; core < f.args.execution.workers; ++core) {
                const auto end = std::min(uint32_t(f.count), f.head_rows);
                const auto expected = PushBuildBoundary(0, end, core + 1, f.args.execution.workers) - PushBuildBoundary(0, end, core, f.args.execution.workers);
                assert(f.built_head_rows[core] == expected);
              }
            } else assert((sgl_model::Load(f.args.execution.workspace + f.layout.ready) >> 32) != sequence);
          }
          f.head_commit_cycle = AscendC::GetSystemCycle();
        }
        sgl_model::Store(remote, *reinterpret_cast<const uint64_t*>(local));
      }
      else std::memcpy(reinterpret_cast<void*>(remote), reinterpret_cast<void*>(local), bytes);
      if (w[0] & (1U << 21)) {
        f.cq[(f.produced % 64) * 8] = (1U - ((f.produced / 64) & 1U)) << 2;
        if (f.fail_first_control && !f.produced) f.cq[0] |= 0x10000U;
        ++f.produced;
        f.pending_controls.push_back(std::move(f.source_checks)); f.source_checks.clear();
        f.max_pending = std::max(f.max_pending, uint32_t(f.pending_controls.size()));
        assert(f.pending_controls.size() <= kPushControlWindow);
      }
    }
  }
  std::vector<std::future<void>> LaunchPush() {
    later_chunk_cycle = head_commit_cycle = 0;
    for (auto& rows : built_rows) rows = 0;
    for (auto& rows : built_head_rows) rows = 0;
    if (chunks) {
      auto* state = reinterpret_cast<ChunkPushState*>(args.execution.workspace + layout.staging);
      std::memset(state->sent, 0xa5, sizeof(state->sent) + sizeof(state->state));
    }
    const CpuSglPushArgs pushed{args, {reinterpret_cast<uint64_t>(mailbox.data()), layout.staging,
        reinterpret_cast<uint64_t>(&context), 0, 0}};
    std::vector<std::future<void>> work;
    for (uint32_t w = 0; w < args.execution.workers; ++w)
      work.push_back(std::async(std::launch::async, [pushed, w] {
        sgl_model::core = w; KvafCpuSglPushAivKernelV1(pushed);
      }));
    return work;
  }
  void WaitHead(uint64_t sequence) {
    const auto end = std::chrono::steady_clock::now() + 5s;
    const auto base = reinterpret_cast<uint64_t>(mailbox.data());
    while ((sgl_model::Load(base + 64) & ~kProfileNotify) != sequence) assert(std::chrono::steady_clock::now() < end);
    const auto* header = reinterpret_cast<Request*>(mailbox.data());
    assert(header->sequence == sequence && header->count == uint32_t(count) && header->mode == (chunks ? kChunkedOutput : kDirectOutput));
    assert(header->reserved_words[0] == (chunks ? kChunkRows : std::min(uint32_t(count), kPipelineHeadRows)));
    assert(header->reserved_words[1] == (chunks && head_rows < kChunkRows ? head_rows : 0));
    assert(header->reserved_words[2] == (cooperative ? 1U : 0U));
    const auto* target = reinterpret_cast<OutputTarget*>(base + layout.output_target);
    assert(target->base == args.execution.output && !std::memcmp(target->segment, args.output_target.segment, kOutputSegmentBytes));
  }
  void FinishCpu(uint64_t sequence, Result result = Result::kOk) {
    const auto base = reinterpret_cast<uint64_t>(mailbox.data());
    auto copy = [&](uint32_t begin, uint32_t end) {
      const auto* sources = reinterpret_cast<const uint64_t*>(base + layout.descriptor);
      for (uint32_t i = begin; i < end; ++i) {
        assert(sources[i] == addresses[indices[i]] + uint32_t(offsets[i]));
        std::memcpy(output.data() + i * kBytes, reinterpret_cast<const void*>(sources[i]), kBytes);
      }
    };
    const auto head = std::min(uint32_t(count), kPipelineHeadRows);
    if (chunks) {
      const auto total = PushChunkCount(uint32_t(count), head_rows);
      std::vector<bool> copied(total);
      uint32_t completed = 0;
      bool failed = false;
      const auto deadline = std::chrono::steady_clock::now() + 5s;
      while (completed < total && !failed) {
        assert(std::chrono::steady_clock::now() < deadline);
        for (uint32_t w = 0; w < args.execution.workers; ++w) {
          const auto token = sgl_model::Load(base + layout.ready + uint64_t(w) * 64);
          if ((token >> 32) != sequence) continue;
          if (uint32_t(token) & kChunkFailed) { failed = true; break; }
          for (uint32_t c = w; c < total && c / args.execution.workers < uint32_t(token); c += args.execution.workers)
            if (!copied[c]) { copy(PushChunkBegin(c, head_rows), PushChunkEnd(uint32_t(count), c, head_rows)); copied[c] = true; ++completed; }
        }
      }
      assert(failed == (result != Result::kOk));
      auto* done = reinterpret_cast<Completion*>(args.execution.workspace + layout.done);
      done->cpu_sgl_write = {1234, 100, 1, 4};
      sgl_model::Store(reinterpret_cast<uint64_t>(&done->token), Progress(sequence, uint32_t(result)));
      return;
    }
    copy(0, head); // CPU may consume the head before suffix publication.
    if (uint32_t(count) > head) {
      const auto end = std::chrono::steady_clock::now() + 5s;
      uint64_t token;
      do { token = sgl_model::Load(base + kSuffixNotifyOffset); assert(std::chrono::steady_clock::now() < end); }
      while ((token >> 32) != sequence);
      if (result == Result::kOk) assert(uint32_t(token) == uint32_t(Result::kOk));
      if (result == Result::kOk) copy(head, count);
    }
    auto* done = reinterpret_cast<Completion*>(args.execution.workspace + layout.done);
    done->cpu_sgl_write = {1234, 100, 1, 4};
    sgl_model::Store(reinterpret_cast<uint64_t>(&done->token), Progress(sequence, uint32_t(result)));
  }
};
}
int main() {
  for (uint32_t head : {128U, 256U, 512U, 1024U}) for (uint32_t workers : {1U, 4U, 5U, 8U}) {
    PushFixture f(workers, true, head, kCapacity, true); uint64_t sequence = 0;
    for (uint32_t count : {7U, head, head + 1, head + 1024, head + 1025, 0U, 9001U}) {
      f.count = count; ++sequence;
      auto work = f.LaunchPush();
      if (count) { f.WaitHead(sequence); f.FinishCpu(sequence); }
      f.Check(work, sequence, Result::kOk);
      assert(f.pending_controls.empty());
    }
  }
  for (uint32_t count : {1U, 7U, 9001U}) {
    PushFixture f(8, true, 8, kCapacity, true); f.count = count;
    auto work = f.LaunchPush(); f.WaitHead(1); f.FinishCpu(1); f.Check(work, 1, Result::kOk);
  }
  {
    PushFixture f(4, true, 256, 65536, true); f.count = 65536;
    auto work = f.LaunchPush(); f.WaitHead(1); f.FinishCpu(1); f.Check(work, 1, Result::kOk);
  }
  {
    // A slow Head partition gates publication, while other cores can build
    // four chunks ahead without a per-chunk global barrier.
    PushFixture f(4, true, 512, kCapacity, true); f.count = 9001;
    sgl_model::pause_row = 128; sgl_model::paused = false; sgl_model::resume = false;
    auto work = f.LaunchPush();
    const auto end = std::chrono::steady_clock::now() + 5s;
    while (!sgl_model::paused.load()) assert(std::chrono::steady_clock::now() < end);
    while (sgl_model::Load(f.args.execution.workspace + f.layout.ready + 2 * 64) != Progress(1, 4))
      assert(std::chrono::steady_clock::now() < end);
    assert(!sgl_model::Load(reinterpret_cast<uint64_t>(f.mailbox.data()) + 64));
    assert(!f.produced);
    sgl_model::resume = true; f.WaitHead(1); f.FinishCpu(1); f.Check(work, 1, Result::kOk);
    sgl_model::pause_row = UINT32_MAX;
  }
  {
    PushFixture f(4, true, 512, kCapacity, true); f.count = 9001;
    sgl_model::pause_row = 768; sgl_model::paused = false; sgl_model::resume = false;
    auto work = f.LaunchPush(); f.WaitHead(1);
    const auto end = std::chrono::steady_clock::now() + 5s;
    while (!sgl_model::paused.load()) assert(std::chrono::steady_clock::now() < end);
    while (sgl_model::Load(f.args.execution.workspace + f.layout.ready + 2 * 64) != Progress(1, 5))
      assert(std::chrono::steady_clock::now() < end);
    // This row belongs to the stalled worker in a later chunk. Fail only
    // after the Head has been accepted, so the CPU must receive drained done.
    f.rows[2816] = -1;
    assert(sgl_model::Load(reinterpret_cast<uint64_t>(f.mailbox.data()) + f.layout.ready + 64) == 0);
    sgl_model::resume = true; f.FinishCpu(1, Result::kInvalidRequest); f.Check(work, 1, Result::kInvalidRequest);
    sgl_model::pause_row = UINT32_MAX;
  }
  {
    PushFixture f(4, true, 512, kCapacity, true); f.count = 9001; f.rows[511] = -1;
    auto work = f.LaunchPush(); f.Check(work, 1, Result::kInvalidRequest); assert(!f.produced);
  }

  for (uint32_t head : {128U, 256U, 512U}) for (uint32_t workers : {1U, 4U, 5U, 8U}) {
    PushFixture f(workers, true, head); uint64_t sequence = 0;
    for (uint32_t count : {7U, head, head + 1, head + 1024, head + 1025, 0U, 9001U}) {
      f.count = count; ++sequence;
      auto work = f.LaunchPush();
      if (count) { f.WaitHead(sequence); f.FinishCpu(sequence); }
      f.Check(work, sequence, Result::kOk);
      assert(f.pending_controls.empty());
    }
  }
  for (uint32_t head : {128U, 256U, 512U}) {
    PushFixture f(4, true, head); f.count = 9001; f.rows[3000] = -1;
    auto work = f.LaunchPush(); f.WaitHead(1); f.FinishCpu(1, Result::kInvalidRequest); f.Check(work, 1, Result::kInvalidRequest);
  }
  for (uint32_t head : {1U, 7U, 1023U}) {
    PushFixture f(4, true, head); f.count = 9001;
    auto work = f.LaunchPush(); f.WaitHead(1); f.FinishCpu(1); f.Check(work, 1, Result::kOk);
  }
  {
    PushFixture f(4, true, 256, 65536); f.count = 65536;
    auto work = f.LaunchPush(); f.WaitHead(1); f.FinishCpu(1); f.Check(work, 1, Result::kOk);
    assert(PushChunkCount(f.count, f.head_rows) == 65 && f.pending_controls.empty());
  }

  {
    // A split Head may fail before notify. Do not admit a phantom deferred
    // CQE or consume stale replay state; conservatively retain the done wait.
    PushFixture f(1, true); f.count = 1025; f.context.max_write_bytes = 64;
    f.fail_first_control = true; f.args.execution.timeout_ms = 1;
    auto work = f.LaunchPush(); f.Check(work, 1, Result::kTimeout);
    assert(!f.head_commit_cycle && f.context.queue.poisoned);
    assert(f.produced == 1 && f.context.queue.cq_consumer_index == 1 && f.pending_controls.empty());
  }
  {
    PushFixture f(4, true); f.count = 9001; f.rows[0] = -1;
    auto work = f.LaunchPush(); f.Check(work, 1, Result::kInvalidRequest);
    assert(!f.head_commit_cycle && !f.produced && f.pending_controls.empty());
  }
  for (uint32_t head : {256U, 1024U}) {
    // Owner 1 is stalled in its first build. Owner 2 must reach the CPU
    // without waiting for that builder or an all-workers barrier.
    PushFixture f(4, true, head); f.count = 9001;
    sgl_model::pause_row = head;
    sgl_model::paused = false; sgl_model::resume = false;
    auto work = f.LaunchPush(); f.WaitHead(1);
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!sgl_model::paused.load()) assert(std::chrono::steady_clock::now() < deadline);
    const auto ready = reinterpret_cast<uint64_t>(f.mailbox.data()) + f.layout.ready;
    while (sgl_model::Load(ready + 2 * sizeof(Word)) != Progress(1, 2))
      assert(std::chrono::steady_clock::now() < deadline);
    assert(sgl_model::Load(ready + sizeof(Word)) == 0);
    sgl_model::resume = true;
    f.FinishCpu(1); f.Check(work, 1, Result::kOk);
    sgl_model::pause_row = UINT32_MAX;
  }
  for (uint32_t workers : {1U, 4U, 5U, 8U}) {
    PushFixture f(workers, true);
    uint64_t sequence = 0;
    for (uint32_t count : {9001U, 1024U, 1025U, 7U, 0U, 9001U}) {
      f.count = count; ++sequence;
      auto work = f.LaunchPush();
      if (count) { f.WaitHead(sequence); f.FinishCpu(sequence); }
      f.Check(work, sequence, Result::kOk);
      if (f.later_chunk_cycle) assert(f.profiles[0].enter_ns + f.profiles[0].head_doorbell_ns < f.later_chunk_cycle);
      assert(f.pending_controls.empty() && f.context.queue.cq_consumer_index == f.produced);
    }
    assert(f.max_pending > 1);
  }
  {
    PushFixture f(4, true); f.count = 9001; f.rows[7000] = 0;
    auto work = f.LaunchPush(); f.WaitHead(1); f.FinishCpu(1, Result::kInvalidRequest);
    f.Check(work, 1, Result::kInvalidRequest);
  }
  {
    PushFixture f(4, true); f.count = 1025; f.context.max_write_bytes = 64;
    auto work = f.LaunchPush(); f.WaitHead(1); f.FinishCpu(1); f.Check(work, 1, Result::kOk);
    assert(f.pending_controls.empty() && f.context.queue.cq_consumer_index == f.produced);
    assert(f.profiles[0].enter_ns + f.profiles[0].head_doorbell_ns >= f.head_commit_cycle);
  }
  {
    PushFixture f(4, true); f.count = 7; f.args.execution.profile = 0;
    auto work = f.LaunchPush(); f.WaitHead(1); f.FinishCpu(1);
    for (auto& w : work) { assert(w.wait_for(5s) == std::future_status::ready); w.get(); }
    assert(f.pending_controls.empty());
    for (const auto& p : f.profiles) assert(!p.head_build_ns && !p.head_doorbell_ns && !p.head_profile_tag);
  }
  for (uint32_t workers : {1U, 4U, 8U}) {
    PushFixture f(workers);
    uint64_t sequence = 0;
    for (uint32_t count : {9001U, 1024U, 1025U, 7U, 0U, 9001U, 9001U}) {
      f.count = count; ++sequence;
      auto work = f.LaunchPush();
      if (count) { f.WaitHead(sequence); for (auto& w : work) assert(w.wait_for(1ms) == std::future_status::timeout); f.FinishCpu(sequence); }
      f.Check(work, sequence, Result::kOk);
      if (count) assert(f.profiles[0].enter_ns + f.profiles[0].head_doorbell_ns >= f.head_commit_cycle);
      assert(f.context.queue.cq_consumer_index == f.produced);
    }
  }
  {
    // The Head spans several control batches and SQ wraps. Record its final
    // commit doorbell, not the first batch or a later suffix/CQ observation.
    PushFixture f(4); f.count = 1025; f.context.max_write_bytes = 64;
    auto work = f.LaunchPush(); f.WaitHead(1); f.FinishCpu(1); f.Check(work, 1, Result::kOk);
    assert(f.profiles[0].enter_ns + f.profiles[0].head_doorbell_ns >= f.head_commit_cycle);
    assert(f.context.queue.cq_consumer_index == f.produced);
  }
  {
    PushFixture f(4); f.count = 7; f.args.execution.profile = 0;
    auto work = f.LaunchPush(); f.WaitHead(1); f.FinishCpu(1);
    for (auto& w : work) { assert(w.wait_for(5s) == std::future_status::ready); w.get(); }
    for (const auto& p : f.profiles) assert(!p.head_build_ns && !p.head_doorbell_ns && !p.head_profile_tag);
  }
  {
    PushFixture f(4); f.count = 9001; f.rows[7000] = 0;
    auto work = f.LaunchPush(); f.WaitHead(1); f.FinishCpu(1, Result::kInvalidRequest);
    f.Check(work, 1, Result::kInvalidRequest);
  }
  {
    PushFixture f(4); f.count = 9001; f.rows[0] = -1;
    auto work = f.LaunchPush(); f.Check(work, 1, Result::kInvalidRequest);
    assert(sgl_model::Load(reinterpret_cast<uint64_t>(f.mailbox.data()) + 64) == 0);
  }
  std::cout << "AIV device-push SGL mailbox/WQE model passed\n";
}
