#define KVAF_CPU_SGL_REUSE_FIXTURE
#include "cpu_sgl_write_aiv.cc"
#undef KVAF_CPU_SGL_REUSE_FIXTURE

namespace {
struct PushFixture : Fixture {
  std::vector<Word> mailbox;
  AivMailboxContext context{};
  std::vector<uint32_t> sq, cq;
  uint32_t sq_db{}, cq_db{}, consumed{}, produced{};
  static inline PushFixture* active{};
  explicit PushFixture(uint32_t workers) : Fixture(workers), mailbox(layout.staging / 64), sq(64 * 16), cq(64 * 8) {
    args.execution.sequence = kPipelineLaunch;
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
    active = this; sgl_model::doorbell = Doorbell;
  }
  ~PushFixture() { sgl_model::doorbell = nullptr; active = nullptr; }
  static void Doorbell(uint32_t value, uint64_t address) {
    auto& f = *active;
    if (address == reinterpret_cast<uint64_t>(&f.cq_db)) return;
    assert(address == reinterpret_cast<uint64_t>(&f.sq_db));
    while (f.consumed < value) {
      const auto* w = f.sq.data() + (f.consumed++ % 64) * 16;
      const uint64_t remote = uint64_t(w[10]) | (uint64_t(w[11]) << 32);
      const uint64_t local = uint64_t(w[14]) | (uint64_t(w[15]) << 32);
      const auto bytes = w[12];
      const auto base = reinterpret_cast<uint64_t>(f.mailbox.data());
      assert(bytes && bytes <= f.context.max_write_bytes && remote >= base && remote + bytes <= base + f.layout.staging);
      assert(local >= f.args.execution.workspace && local + bytes <= f.args.execution.workspace + f.layout.staging);
      // Do not overwrite CPU-owned done/Ready sources.
      assert(remote + bytes <= base + 128 || remote >= base + f.layout.output_target);
      if (bytes == 8) sgl_model::Store(remote, *reinterpret_cast<const uint64_t*>(local));
      else std::memcpy(reinterpret_cast<void*>(remote), reinterpret_cast<void*>(local), bytes);
      if (w[0] & (1U << 21)) {
        f.cq[(f.produced % 64) * 8] = (1U - ((f.produced / 64) & 1U)) << 2;
        ++f.produced;
      }
    }
  }
  std::vector<std::future<void>> LaunchPush() {
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
    assert(header->sequence == sequence && header->count == uint32_t(count) && header->mode == kDirectOutput);
    assert(header->reserved_words[0] == std::min(uint32_t(count), kPipelineHeadRows));
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
  for (uint32_t workers : {1U, 4U, 8U}) {
    PushFixture f(workers);
    uint64_t sequence = 0;
    for (uint32_t count : {9001U, 1024U, 1025U, 7U, 0U, 9001U, 9001U}) {
      f.count = count; ++sequence;
      auto work = f.LaunchPush();
      if (count) { f.WaitHead(sequence); for (auto& w : work) assert(w.wait_for(1ms) == std::future_status::timeout); f.FinishCpu(sequence); }
      f.Check(work, sequence, Result::kOk);
      assert(f.context.queue.cq_consumer_index == f.produced);
    }
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
