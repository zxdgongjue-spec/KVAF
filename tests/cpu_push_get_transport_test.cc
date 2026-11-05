// Execute the production CPU transport with deferred URMA submissions. This
// checks protocol/data ordering, not physical DMA visibility or bandwidth.
#include "cpu_push_get_transport.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>

namespace {
using namespace kvaf::cpu_push;
std::array<urma_jetty_t, kJettyCount> jettys{};
std::array<urma_jfc_t, kJettyCount> cqs{};
std::array<std::deque<urma_jfs_wr_t*>, kJettyCount> pending;
struct Posted {
  std::uint32_t queue, opcode, bytes, order, signaled, comp_order;
  std::uint64_t destination;
};
std::vector<Posted> posted;
std::uint64_t done_address{};
bool fail_cq{};
std::uint32_t post_calls{};
std::vector<std::uint32_t> post_queues;
const kvaf::urma::Sge* later_queue_source{};
std::uint64_t Address(const void* p) { return reinterpret_cast<std::uint64_t>(p); }
struct HbmBuffer {
  void* memory{};
  std::size_t bytes;
  explicit HbmBuffer(std::size_t n) : bytes(n) {
    assert(!posix_memalign(&memory, 64, n));
    std::memset(memory, 0xad, n);
  }
  ~HbmBuffer() { std::free(memory); }
  std::uint8_t* data() { return static_cast<std::uint8_t*>(memory); }
  std::size_t size() const { return bytes; }
  std::uint8_t& operator[](std::size_t i) { return data()[i]; }
};
}

extern "C" urma_status_t __wrap_urma_post_jetty_send_wr(
    urma_jetty_t* jetty, urma_jfs_wr_t* wr, urma_jfs_wr_t**) {
  const auto q = static_cast<std::uint32_t>(jetty - jettys.data());
  assert(q < kJettyCount);
  if (later_queue_source) {
    // Queue 0 starts transferring before queue 1's source addresses are
    // refreshed. Deferred DMA below still verifies all submitted SGLs.
    assert(q == 0 && wr->opcode == URMA_OPC_WRITE);
    assert(wr->rw.src.sge[0].addr && !later_queue_source->addr);
    later_queue_source = nullptr;
  }
  ++post_calls;
  post_queues.push_back(q);
  pending[q].push_back(wr);
  for (auto* current = wr; current; current = current->next) {
    const auto& dst = current->rw.dst.sge[0];
    posted.push_back({q, static_cast<std::uint32_t>(current->opcode), dst.len,
        current->flag.bs.place_order, current->flag.bs.complete_enable,
        current->flag.bs.comp_order, dst.addr});
  }
  return static_cast<urma_status_t>(0);
}

extern "C" int __wrap_urma_poll_jfc(urma_jfc_t* cq, int, urma_cr_t* cr) {
  const auto q = static_cast<std::uint32_t>(cq - cqs.data());
  assert(q < kJettyCount);
  int completed = 0;
  while (!pending[q].empty()) {
    auto* first = pending[q].front(); pending[q].pop_front();
    std::vector<urma_jfs_wr_t*> execution;
    for (auto* wr = first; wr; wr = wr->next) execution.push_back(wr);
    for (auto* wr : execution) {
      const auto& dst = wr->rw.dst.sge[0];
      if (wr->opcode == URMA_OPC_WRITE && dst.addr == done_address) {
        assert(dst.len == 8 && wr->flag.bs.place_order == URMA_STRONG_ORDER);
        // Whole-batch done must never be posted before other Jettys drain.
        for (const auto& queue : pending) assert(queue.empty());
      }
      const auto workspace = done_address - 128;
      const auto& request = *reinterpret_cast<const Request*>(workspace);
      if (wr->opcode == URMA_OPC_WRITE && dst.addr >= workspace + 192 &&
          dst.addr < workspace + 192 + request.workers * sizeof(Ready)) {
        Layout layout; assert(MakeLayout(request.token_bytes, request.workers, &layout));
        const auto worker = (dst.addr - workspace - 192) / sizeof(Ready);
        assert((dst.addr - workspace - 192) % sizeof(Ready) == q * 8);
        assert(dst.len == 8 && wr->flag.bs.place_order == URMA_STRONG_ORDER);
        const auto progress = *reinterpret_cast<const std::uint64_t*>(wr->rw.src.sge[0].addr);
        assert(progress >> 32 == request.sequence && std::uint32_t(progress) > 0);
        const auto range = WorkerRange(request.count, request.token_bytes, worker, request.workers);
        const auto shard = JettyRange(ChunkRange(range, std::uint32_t(progress) - 1), q);
        const auto* addresses = reinterpret_cast<const std::uint64_t*>(workspace + layout.descriptor);
        // Check the producer's exact guarantee at the Ready publication point,
        // before any later queue/chunk is required to complete.
        for (auto row = shard.begin; row < shard.end; ++row)
          assert(!std::memcmp(reinterpret_cast<const void*>(workspace + layout.staging +
              std::uint64_t(row) * request.token_bytes), reinterpret_cast<const void*>(addresses[row]),
              request.token_bytes));
      }
      auto* out = reinterpret_cast<std::uint8_t*>(dst.addr);
      std::size_t bytes = 0;
      for (std::uint32_t i = 0; i < wr->rw.src.num_sge; ++i) {
        const auto& src = wr->rw.src.sge[i];
        std::memcpy(out + bytes, reinterpret_cast<void*>(src.addr), src.len);
        bytes += src.len;
      }
      assert(bytes == dst.len);
      if (wr->flag.bs.complete_enable) {
        assert(!completed);
        cr[0] = {};
        cr[0].user_ctx = wr->user_ctx;
        cr[0].status = fail_cq ? static_cast<urma_cr_status_t>(1) : URMA_CR_SUCCESS;
        completed = 1;
      }
    }
  }
  return completed;
}

namespace kvaf::cpu_push {
struct TransportTest {
  static void Run(std::uint32_t token, std::uint32_t maximum = 1024,
                  std::uint32_t max_sge = 16, std::uint32_t post_batch = 128,
                  std::uint32_t message_bytes = 65536) {
    Transport t;
    assert(MakeLayout(token, 4, &t.layout_));
    t.workers_ = 4; t.token_bytes_ = token; t.max_sge_ = max_sge; t.max_bytes_ = message_bytes;
    t.post_batch_wrs_ = post_batch;
    // Retain marker space to exercise legacy mode as well as done-only mode.
    void* memory = nullptr;
    assert(!posix_memalign(&memory, 64, t.layout_.staging + 32768));
    t.memory_ = static_cast<std::uint8_t*>(memory);
    std::memset(memory, 0, t.layout_.staging + 32768);
    // Request/Ready have alignas(64); a byte vector does not guarantee their
    // alignment, even though real registered HBM is cache-line aligned.
    HbmBuffer hbm(t.layout_.staging + std::uint64_t(maximum) * token);
    std::vector<std::uint8_t> source(std::uint64_t(maximum) * token);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = (i * 13 + i / token) % 251;
    t.hbm_base_ = Address(hbm.data()); done_address = t.hbm_base_ + t.layout_.done;
    for (std::uint32_t q = 0; q < kJettyCount; ++q) {
      t.queues_[q].channel.local_jetty = &jettys[q];
      t.queues_[q].channel.local_jfc = &cqs[q];
    }
    t.PrepareCompletions();
    std::uint64_t sequence = 0;
    const auto request = [&](std::uint32_t count, bool whole, bool profile) {
      ++sequence;
      auto* r = reinterpret_cast<Request*>(hbm.data());
      *r = {};
      r->magic = kMagic; r->sequence = sequence; r->count = count; r->token_bytes = token; r->workers = 4;
      r->reserved = (whole ? kBatchDoneOnly : 0) | (profile ? kInlineProfile : 0);
      reinterpret_cast<Word*>(hbm.data() + 64)->value = sequence | (profile ? kProfileNotify : 0);
      auto* addresses = reinterpret_cast<std::uint64_t*>(hbm.data() + t.layout_.descriptor);
      for (std::uint32_t i = 0; i < count; ++i)
        addresses[i] = Address(source.data() + std::uint64_t((count - i + sequence) % count) * token);
      // Model AIV's ordered mailbox publication, disjoint from CPU-owned
      // Ready/done/marker source words. Every call refreshes the entire prefix.
      posted.clear(); post_calls = 0;
      std::memcpy(t.memory_, r, sizeof(*r));
      std::memcpy(t.memory_ + t.layout_.descriptor, addresses, count * 8);
      __atomic_store_n(reinterpret_cast<std::uint64_t*>(t.memory_ + 64),
                      sequence | (profile ? kProfileNotify : 0), __ATOMIC_RELEASE);
      std::uint64_t observed{};
      assert(t.ReadRequest(&observed).ok() && observed == sequence);
      assert(t.ReadDescriptor(sequence).ok());
      assert(posted.empty() && !post_calls); // Local mailbox: no control READs.
      assert(!std::memcmp(t.memory_ + t.layout_.descriptor, addresses, count * 8));
      posted.clear();
      post_calls = 0;
      post_queues.clear();
      if (count >= 4 && t.prepared_count_ == count && t.prepared_batch_done_only_ == whole &&
          !t.prepared_.empty() && (t.prepared_[0]->queues[0].first_row != UINT32_MAX ||
                                 !t.prepared_[0]->queues[0].source_runs.empty()) &&
          !t.prepared_[0]->queues[1].local.empty()) {
        t.prepared_[0]->queues[0].local[0].addr = 0;
        t.prepared_[0]->queues[1].local[0].addr = 0;
        later_queue_source = &t.prepared_[0]->queues[1].local[0];
      }
      std::memset(hbm.data() + t.layout_.staging, 0xad, std::uint64_t(maximum) * token);
      assert(t.WriteBatch().ok());
      assert(!later_queue_source);
      if (!whole && std::uint64_t{max_sge} * token <= t.max_bytes_) {
        std::size_t payload_sges = 0;
        const auto* current = reinterpret_cast<const std::uint64_t*>(t.memory_ + t.layout_.descriptor);
        for (const auto& plan : t.prepared_) for (const auto& q : plan->queues) {
          assert(q.sources.empty() && q.first_row == UINT32_MAX);
          std::vector<bool> seen(q.local.size());
          for (const auto& run : q.source_runs) for (std::uint32_t i = 0; i < run.count; ++i) {
            assert(run.row + i < count && run.local + i < q.local.size());
            assert(!seen[run.local + i]); seen[run.local + i] = true;
            assert(q.local[run.local + i].addr == current[run.row + i]);
            ++payload_sges;
          }
          assert(std::count(seen.begin(), seen.end(), false) == q.markers.size());
          for (const auto& marker : q.markers) assert(*marker.value == Progress(sequence, marker.progress));
        }
        assert(payload_sges == count); // Every data SGE, excluding Ready, refreshed exactly once.
      }
      if (whole && count >= kJettyCount) {
        for (std::uint32_t q = 0; q < kJettyCount; ++q) assert(post_queues[q] == q);
        std::uint32_t expected_posts = 0;
        for (const auto& plan : t.prepared_)
          for (const auto& q : plan->queues) {
            expected_posts += q.posts.size();
            if (!q.local.empty() && std::uint64_t{max_sge} * token <= t.max_bytes_) {
              assert(q.first_row != UINT32_MAX && q.sources.empty());
              const auto* addresses = reinterpret_cast<const std::uint64_t*>(t.memory_ + t.layout_.descriptor);
              for (std::size_t i = 0; i < q.local.size(); ++i)
                assert(q.local[i].addr == addresses[q.first_row + i]);
            }
          }
        assert(post_calls == expected_posts);
      }
      for (std::uint32_t i = 0; i < count; ++i)
        assert(!std::memcmp(hbm.data() + t.layout_.staging + std::uint64_t(i) * token,
                            reinterpret_cast<void*>(addresses[i]), token));
      std::uint32_t markers = 0, data = 0, tails = 0;
      for (const auto& wr : posted) {
        if (wr.destination < t.hbm_base_ + t.layout_.descriptor) ++markers;
        else ++data;
        tails += wr.signaled;
        assert(wr.comp_order == 1);
        if (wr.signaled) assert(wr.order == URMA_STRONG_ORDER && wr.comp_order);
      }
      if (whole) assert(markers == 0);
      else if (count) assert(markers > 0);
      if (count == 1024 && max_sge == 16 && message_bytes == 65536 &&
          (token == 656 || token == 4096)) {
        assert(data == 64 && tails == 4 && post_calls == 4);
        assert(markers == (whole ? 0 : 16));
      }
      if (!whole && count == 8192 && max_sge == 16 && token == 656 && post_batch == 128) {
        assert(data == 512 && markers == 32 && tails == 4 && post_calls == 4);
        assert((post_queues == std::vector<std::uint32_t>{0, 1, 2, 3}));
      }
      for (const auto& plan : t.prepared_) for (const auto& q : plan->queues)
        for (const auto& post : q.posts) {
          std::uint32_t payloads = 0, sges = post.first_sge;
          for (auto* wr = &q.wrs[post.wr]; wr; wr = wr->next) {
            payloads += wr->rw.dst.sge[0].addr >= t.hbm_base_ + t.layout_.staging;
            sges += wr->rw.src.num_sge;
          }
          assert(payloads <= post_batch && sges == post.end_sge);
        }
      // The suffix outside the active prefix is never written.
      for (auto i = t.layout_.staging + std::uint64_t(count) * token; i < hbm.size(); ++i)
        assert(hbm[i] == 0xad);
      posted.clear();
      post_calls = 0;
      assert(t.Complete(sequence, Result::kOk).ok());
      assert(post_calls == 1);
      assert(posted.size() == (profile ? 2 : 1));
      if (profile) {
        assert(posted[0].destination == done_address + 8 && posted[0].bytes == 56);
        assert(!posted[0].signaled);
        const auto* done = reinterpret_cast<Completion*>(hbm.data() + t.layout_.done);
        assert(done->reserved[0] == ProfileToken(sequence));
      }
      assert(posted.back().destination == done_address && posted.back().signaled);
      for (const auto& wr : posted) assert(wr.comp_order == 1);
      assert(reinterpret_cast<Completion*>(hbm.data() + t.layout_.done)->token == Progress(sequence, 1));
      assert(t.tracker_.Empty());
    };
    for (auto count : {1U, maximum, maximum, 19U, 257U, 0U, maximum})
      request(count, true, sequence % 2 == 0);
    // Same count with a different mode must rebuild, not reuse a marker plan.
    for (auto count : {maximum, maximum, 19U, 257U, 0U, maximum})
      request(count, false, sequence % 2 == 0);
    request(maximum, true, true);
    if (token == 4096 && maximum == 1024) {
      t.context_ = reinterpret_cast<urma_context_t*>(1);  // Queue mocks own the data plane.
      for (auto mode : {0U, 1U}) for (auto queues : {1U, 2U, 4U}) {
        WriteBenchmarkOptions options;
        options.payload_bytes = 65536; options.source_bytes = 131072; options.seed = 23 + mode + queues;
        options.message_bytes = 4096; options.fragment_bytes = 656; options.queues = queues;
        options.post_batch_wrs = 2; options.warmup = 1; options.iterations = 3; options.source_mode = mode;
        auto result = std::make_unique<WriteBenchmarkResult>();
        posted.clear();
        assert(t.BenchmarkWrite(source.data(), Address(source.data()), source.size(), options, result.get()).ok());
        assert(result->count == options.iterations && t.tracker_.Empty());
        for (const auto& wr : posted) {
          assert(wr.destination != done_address);
          assert(wr.comp_order == 1);
        }
        for (std::uint32_t i = 0; i < result->count; ++i)
          assert(result->samples[i].tail_cqes == queues && result->samples[i].comm_ns >=
                 result->samples[i].post_ns + result->samples[i].cq_wait_ns);
        for (std::uint64_t offset = 0; offset < options.payload_bytes; offset += 8) {
          const auto source_offset = mode ? WriteBenchmarkMix(options.seed + offset / options.fragment_bytes) %
              (options.source_bytes / options.fragment_bytes) * options.fragment_bytes + offset % options.fragment_bytes : offset;
          std::uint64_t actual;
          std::memcpy(&actual, hbm.data() + t.layout_.staging + offset, 8);
          assert(actual == WriteBenchmarkMix(options.seed + source_offset / 8));
        }
      }
      t.context_ = nullptr;
    }
    // A bad CQ fails the batch and must never publish successful done.
    posted.clear(); fail_cq = true;
    assert(!t.WriteBatch().ok());
    for (const auto& wr : posted) assert(wr.destination != done_address);
    fail_cq = false;
    for (auto& q : pending) q.clear();
    // The fixtures own these dummy queues; production Shutdown frees the real
    // registered memory/WR storage only after its real queue deletion succeeds.
    for (auto& q : t.queues_) {
      q.channel.local_jetty = nullptr; q.channel.local_jfc = nullptr;
    }
    assert(t.Shutdown().ok());
  }
};
}

int main() {
  for (auto token : {13U, 656U, 4096U, 65535U}) kvaf::cpu_push::TransportTest::Run(token);
  // Cross SQ-credit/wave boundaries and small negotiated message limits.
  kvaf::cpu_push::TransportTest::Run(13, kvaf::cpu_push::kMaxRequests);
  // Payload fragmentation, multi-post chains and multiple SQ-sized waves.
  kvaf::cpu_push::TransportTest::Run(13, 1024, 16, 128, 64);
  kvaf::cpu_push::TransportTest::Run(13, kvaf::cpu_push::kMaxRequests, 16, 128, 64);
  // 128 payload WR + 8 Ready per queue must stay one post at budget 128.
  kvaf::cpu_push::TransportTest::Run(656, 8192, 16, 128);
  // Deployed SGE limit, interleaved multi-doorbell queues and reused plans with
  // changing source addresses. All runtime submission sizes preserve ordering.
  for (auto batch : {32U, 128U, 256U})
    for (auto token : {656U, 1024U})
      kvaf::cpu_push::TransportTest::Run(token, 8192, 13, batch);
  std::cout << "PASS: CPU push mailbox, payload-budget posts and completion ordering\n";
}
