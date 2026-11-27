#include "cpu_sgl_push_chunks.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <set>
#include <vector>
namespace {
using namespace kvaf::cpu_push;
struct RelayIO {
  struct Posted { unsigned chunk, owner, progress; bool first; };
  std::array<std::uint64_t, 8> progress{};
  std::array<unsigned, 8> sent{};
  std::vector<Posted> posts;
  unsigned pending{}, total{}, waits{}, errors{};
  bool notified{}, error_sent{};
  std::uint64_t ProgressWord(unsigned w) const { return progress[w]; }
  unsigned Sent(unsigned w) const { return sent[w]; }
  void SetSent(unsigned w, unsigned s) { sent[w] = s; }
  unsigned Pending() const { return pending; }
  void SetPending(unsigned p) { assert(p <= kPushControlWindow); pending = p; }
  unsigned Total() const { return total; }
  void SetTotal(unsigned t) { total = t; }
  bool Notified() const { return notified; }
  void SetNotified() { notified = true; }
  bool ErrorSent() const { return error_sent; }
  void SetErrorSent() { error_sent = true; }
  unsigned Wait() { assert(pending); ++waits; return 0; }
  unsigned Error(unsigned owner, unsigned error) { assert(owner < 8 && error); ++errors; return 0; }
  unsigned Post(unsigned c, unsigned w, unsigned p, bool first) { posts.push_back({c, w, p, first}); return 0; }
};
void CheckRelay() {
  RelayIO io;
  constexpr unsigned sequence = 3, count = 8192, workers = 4, head = 512;
  const auto chunks = PushChunkCount(count, head);
  auto relay = [&] { return RelayCooperativeChunks(io, count, workers, sequence, head); };
  io.progress.fill(Progress(sequence, 3)); io.progress[1] = Progress(sequence - 1, 8);
  assert(CooperativeProgress(io, chunks, workers, sequence) == 0 && !relay() && io.posts.empty());
  io.progress[1] = Progress(sequence, 2);
  assert(CooperativeProgress(io, chunks, workers, sequence) == 2 && !relay() && io.posts.size() == 2);
  io.progress.fill(Progress(sequence, chunks));
  assert(!relay() && io.posts.size() == chunks && io.waits == chunks - kPushControlWindow);
  for (unsigned c = 0; c < chunks; ++c) {
    const auto& post = io.posts[c];
    assert(post.chunk == c && post.owner == c % workers && post.progress == c / workers + 1 && post.first == (c == 0));
  }
  assert(!relay() && io.posts.size() == chunks); // No duplicate publication.
  io.progress[2] = Progress(sequence, chunks - 1);
  assert(relay() == unsigned(Result::kInvalidRequest) && io.errors == 1);
  assert(relay() == unsigned(Result::kInvalidRequest) && io.errors == 1);
  assert(!DrainChunkControls(io) && !io.pending);
  RelayIO failed;
  failed.progress[2] = Progress(sequence, kChunkFailed | unsigned(Result::kCopyError));
  assert(CooperativeProgress(failed, chunks, workers, sequence) == (std::uint64_t(Result::kCopyError) << 32));
  assert(RelayCooperativeChunks(failed, count, workers, sequence, head) == unsigned(Result::kCopyError));
  assert(failed.posts.empty() && !failed.errors); // Before Head: no CPU request to fail.
  failed.progress[2] = Progress(sequence + 1, 0);
  assert(CooperativeProgress(failed, chunks, workers, sequence) == (std::uint64_t(Result::kSequenceError) << 32));
  failed.progress[2] = Progress(sequence, chunks + 1);
  assert(CooperativeProgress(failed, chunks, workers, sequence) == (std::uint64_t(Result::kInvalidRequest) << 32));

  RelayIO publisher;
  publisher.progress.fill(Progress(sequence, chunks));
  publisher.progress[0] = Progress(sequence, 0);
  auto publish = [&] { return RelayCooperativeChunks(publisher, count, workers, sequence, head, true); };
  assert(!publish() && publisher.posts.empty()); // Head must include worker 0.
  publisher.progress[0] = Progress(sequence, 1);
  assert(!publish() && publisher.total == 1);
  assert(!publish() && publisher.total == chunks); // Worker 0 stays at Head forever.
  assert(publisher.posts.size() == chunks && publisher.progress[0] == Progress(sequence, 1));
  for (unsigned c = 0; c < chunks; ++c)
    assert(publisher.posts[c].owner == c % workers && publisher.posts[c].progress == c / workers + 1);
  assert(!DrainChunkControls(publisher));

  RelayIO hybrid;
  hybrid.progress.fill(Progress(sequence, kPushHeadBuilt));
  hybrid.progress[0] = Progress(sequence, kPushHeadBuilt | 1U);
  auto relay_head = [&] { return RelayReadyChunks(hybrid, count, workers, sequence, head, true); };
  assert(!relay_head() && hybrid.posts.size() == 1 && hybrid.posts[0].chunk == 0);
  // Head completion on peers must not imply suffix readiness. An independent
  // suffix can be forwarded while another owner's suffix remains stalled.
  hybrid.progress[2] = Progress(sequence, kPushHeadBuilt | 1U);
  assert(!relay_head() && hybrid.posts.size() == 2 && hybrid.posts[1].chunk == 2);
  hybrid.progress[1] = Progress(sequence, kChunkFailed | unsigned(Result::kInvalidRequest));
  assert(relay_head() == unsigned(Result::kInvalidRequest) && hybrid.errors == 1);
  assert(!DrainChunkControls(hybrid));
}
}
int main() {
  using namespace kvaf::cpu_push;
  assert(!ValidPushHeadRows(0) && !ValidPushHeadRows(1025));
  assert(PushOutputMode(1024) == kPushChunkedOutput);
  assert(PushWorkspaceBytes(1024) == sizeof(ChunkPushState));
  for (unsigned head = 1; head <= 1024; ++head) {
    assert(ValidCooperativeHeadRows(head) == (head % 8 == 0));
    assert(PushSetupHeadRows(PushOutputMode(head, true)) == (head % 8 ? 0 : head));
    assert(PushLaunchCooperative(PushLaunch(head, true)) && !PushLaunchCooperative(PushLaunch(head)));
    assert(PushLaunchHeadRows(PushLaunch(head, true)) == head);
    assert(PushLaunchHeadRows(PushLaunch(head)) == head);
    for (const bool publisher : {false, true}) {
      const auto launch = PushLaunch(head, publisher, publisher, !publisher);
      const auto mode = PushOutputMode(head, publisher, publisher, !publisher);
      assert(PushSetupHeadRows(mode) == (head % 8 ? 0 : head));
      assert(PushLaunchBuildMode(launch) == (publisher ? 2U : 3U));
      assert(PushSetupBuildMode(mode) == PushLaunchBuildMode(launch));
      assert(PushLaunchHeadRows(launch) == head);
    }
    if (head < 1024) assert(PushSetupHeadRows(PushOutputMode(head)) == head && PushOutputMode(head) > kPushChunkedOutput);
    for (unsigned count : {0U, 1U, head - 1, head, head + 1, 1024U, 1025U, 65535U, 65536U}) {
      const auto chunks = PushChunkCount(count, head);
      assert(chunks <= 65);
      std::set<unsigned long long> commits;
      unsigned end = 0;
      for (unsigned c = 0; c < chunks; ++c) {
        assert(PushChunkBegin(c, head) == end);
        end = PushChunkEnd(count, c, head);
        assert(end > PushChunkBegin(c, head) && end <= count);
        assert(PushChunkIndex(end - 1, head) == c && PushChunkIndex(PushChunkBegin(c, head), head) == c);
        const auto offset = PushCommitOffset(c);
        assert(commits.insert(offset).second && offset + sizeof(Word) <= PushWorkspaceBytes(head));
        assert(offset < kPushChunkCount * sizeof(Word) || offset >= sizeof(ChunkPushState));
        if (head % 8 == 0) for (unsigned workers = 1; workers <= 8; ++workers) {
          const auto first = PushChunkBegin(c, head);
          assert(first % 8 == 0);
          unsigned previous = first;
          for (unsigned w = 0; w < workers; ++w) {
            const auto begin = PushBuildBoundary(first, end, w, workers);
            const auto last = PushBuildBoundary(first, end, w + 1, workers);
            assert(begin == previous && begin <= last && last <= end);
            assert((begin == end || begin % 8 == 0) && (last == end || last % 8 == 0));
            previous = last;
          }
          assert(previous == end);
        }
      }
      assert(end == count);
    }
  }
  CheckRelay();
  std::puts("CPU SGL Head boundaries, cooperative partitions/relay and immutable commit layout passed");
}
