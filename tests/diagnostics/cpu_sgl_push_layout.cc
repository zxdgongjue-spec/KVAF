#include "cpu_sgl_push_layout.h"
#include <cassert>
#include <cstdio>
#include <set>
int main() {
  using namespace kvaf::cpu_push;
  assert(!ValidPushHeadRows(0) && !ValidPushHeadRows(1025));
  assert(PushOutputMode(1024) == kPushChunkedOutput);
  assert(PushWorkspaceBytes(1024) == sizeof(ChunkPushState));
  for (unsigned head = 1; head <= 1024; ++head) {
    assert(PushLaunchHeadRows(PushLaunch(head)) == head);
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
      }
      assert(end == count);
    }
  }
  std::puts("CPU SGL small-Head boundaries and immutable commit layout passed");
}
