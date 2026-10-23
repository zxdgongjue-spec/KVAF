#include "cpu_push_get_fixture.h"

#include <iostream>

using namespace kvaf::cpu_push;
void TransferCase(std::uint32_t bytes, std::uint32_t max_sge, std::uint32_t max_message) {
  push_mock::Reset(max_sge, max_message);
  {
    PushFixture f(bytes, 1031, 1);
    for (std::size_t i = 0; i < f.source.size(); ++i) f.source[i] = (i * 17 + i / bytes) % 251;
    for (std::uint64_t sequence = 1; sequence < 3; ++sequence) {
      f.source[0] = sequence;
      f.Describe(sequence);
      std::uint64_t observed = 0;
      PushRequire(f.transport->ReadRequest(&observed).ok() && observed == sequence, "request read");
      PushRequire(f.transport->ReadDescriptor(sequence).ok(), "descriptor read");
      const auto range = WorkerRange(f.count, f.bytes, 0, 1);
      for (std::uint32_t chunk = 0; chunk < ChunkCount(range); ++chunk) {
        PushRequire(f.transport->WriteChunk(0, chunk, ChunkRange(range, chunk)).ok(), "chunk WRITE");
        for (auto token : f.At<Ready>(f.layout.ready)->jetty)
          PushRequire(token == Progress(sequence, chunk + 1), "ordered marker mismatch");
      }
      PushRequire(f.transport->Complete(sequence, Result::kOk).ok(), "done WRITE");
      PushRequire(f.At<Completion>(f.layout.done)->token == Progress(sequence, 1), "done token mismatch");
      for (std::uint32_t row = 0; row < f.count; ++row)
        PushRequire(!std::memcmp(f.At<std::uint8_t>(f.layout.staging) + std::uint64_t{row} * bytes,
            f.source.data() + std::uint64_t{f.count - row - 1} * bytes * 3, bytes), "gather payload mismatch");
    }
    PushRequire(push_mock::Markers() == 16, "expected four markers per chunk");
    f.Describe(3);
    f.At<kvaf::peer::GatherSpan>(f.layout.descriptor)[0].address = 1;
    PushRequire(!f.transport->ReadDescriptor(3).ok(), "invalid source accepted");
    f.Describe(4);
    f.At<Request>(0)->token_bytes++;
    PushRequire(!f.transport->ReadDescriptor(4).ok(), "wrong token size accepted");
  }
  PushRequire(push_mock::LiveResources() == 0, "transport resource leak");
}
void FailureCase(bool completion) {
  push_mock::Reset();
  {
    PushFixture f(656, 8, 1);
    f.Describe(1);
    PushRequire(f.transport->ReadDescriptor(1).ok(), "failure setup");
    if (completion) push_mock::ErrorNextCompletion(); else push_mock::FailNextPost();
    PushRequire(!f.transport->WriteChunk(0, 0, {0, 8}).ok(), "transport error ignored");
    PushRequire(!f.transport->WriteChunk(0, 0, {0, 8}).ok(), "poisoned queue reused");
    push_mock::FailNextDelete();
    PushRequire(!f.transport->Shutdown().ok() && push_mock::LiveResources() > 0, "failed teardown discarded buffers");
    PushRequire(f.transport->Shutdown().ok(), "teardown retry failed");
  }
  PushRequire(push_mock::LiveResources() == 0, "failure leaked resources");
}
int main() {
  try {
    TransferCase(656, 3, 1024);
    TransferCase(2000, 1, 64); // More than one WR wave; tokens split across WRs.
    TransferCase(656, 13, 65536);
    FailureCase(false); FailureCase(true);
    std::cout << "CPU push transport PASS\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
