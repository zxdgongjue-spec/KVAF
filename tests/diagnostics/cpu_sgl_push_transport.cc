#define KVAF_CPU_SGL_REUSE_FIXTURE
#include "cpu_sgl_transport.cc"
#undef KVAF_CPU_SGL_REUSE_FIXTURE
#include <iostream>

namespace {
void RunMailbox(uint32_t count, uint32_t sge, bool fail_suffix = false, bool narrow_source_range = false, bool cancel = false) {
  using namespace kvaf::cpu_push;
  constexpr uint32_t token_bytes = 656;
  Layout layout; assert(MakeLayout(token_bytes, 4, &layout, true));
  auto* hbm = static_cast<uint8_t*>(std::aligned_alloc(64, layout.bytes));
  std::memset(hbm, 0xee, layout.bytes); // Remote descriptor reads would consume invalid data.
  std::vector<uint8_t> source(count * token_bytes), output(source.size());
  for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t(i * 11 + i / token_bytes);
  urma_device_t device{}; urma_context_t context{}; context.dev = &device;
  urma_target_seg_t src{}; src.seg.ubva.va = reinterpret_cast<uint64_t>(source.data()); src.seg.len = source.size();
  urma_seg_t hbm_segment{}; hbm_segment.ubva.va = reinterpret_cast<uint64_t>(hbm); hbm_segment.len = layout.bytes;
  kvaf::peer::OpenCpuSglPushRequest setup{};
  auto& base = setup.base;
  base.source_bytes = source.size(); base.hbm_base = hbm_segment.ubva.va; base.hbm_bytes = layout.bytes;
  // Keep all addresses inside the registered allocation while deliberately
  // narrowing the logical source range. These cases prove the CPU no longer
  // scans either the head-only list or the suffix for source-range violations.
  if (narrow_source_range) base.source_bytes /= 2;
  const auto source_row = [&](uint32_t i) { return narrow_source_range ? i : count - i - 1; };
  base.npu_tp = setup.control_tp = 1; base.npu_psn = setup.control_psn = 1; base.token_value = 7;
  base.segment_bytes = sizeof(urma_seg_t); base.token_bytes = token_bytes; base.workers = 4;
  base.output_mode = kPipelinedOutput; base.sge_per_wr = sge;
  std::memcpy(base.segment, &hbm_segment, sizeof(hbm_segment));
  for (auto& c : base.channels) c.bytes = 28;
  setup.control.bytes = 28;
  kvaf::peer::OpenCpuSglPushResponse reply{};
  Transport transport;
  assert(transport.Initialize(&context, &src, base, QueryTp, ImportJetty, 1, &reply.base, 0, &setup, &reply).ok());
  assert(reply.mailbox_bytes == layout.staging && reply.segment_bytes == sizeof(urma_seg_t) && reply.control.bytes == 28);
  urma_seg_t mailbox_segment{};
  std::memcpy(&mailbox_segment, reply.segment, sizeof(mailbox_segment));
  assert(mailbox_segment.attr.bs.access == 6U); // Remote READ/WRITE, no LOCAL_ONLY.
  const auto mailbox = reply.mailbox_base;
  assert(mailbox != base.hbm_base);
  const auto head = std::min(count, kPipelineHeadRows);
  retain_chunk_storage = true;
  for (uint32_t sequence : {1U, 2U}) {
    std::fill(output.begin(), output.end(), 0xa5);
    output_base = reinterpret_cast<uint64_t>(output.data()); output_bytes = output.size();
    done_address = base.hbm_base + layout.done;
    auto* request = reinterpret_cast<Request*>(mailbox);
    *request = {kMagic, sequence, count, token_bytes, 4, kDirectOutput, {head, 0, 0, 0}};
    auto* target = reinterpret_cast<OutputTarget*>(mailbox + layout.output_target);
    *target = {}; target->base = output_base; target->bytes = output.size(); target->token_value = 7;
    target->segment_bytes = sizeof(urma_seg_t);
    urma_seg_t output_segment{}; output_segment.ubva.va = output_base; output_segment.len = output.size();
    std::memcpy(target->segment, &output_segment, sizeof(output_segment));
    auto* addresses = reinterpret_cast<uint64_t*>(mailbox + layout.descriptor);
    for (uint32_t i = 0; i < head; ++i) addresses[i] = src.seg.ubva.va + uint64_t(source_row(i)) * token_bytes;
    for (uint32_t i = head; i < count; ++i) addresses[i] = UINT64_MAX;
    head_bytes = uint64_t(head) * token_bytes; head_written = 0;
    bool overlapping = false;
    data_post = [&](uint64_t destination) {
      if (destination >= output_base + head_bytes) {
        size_t pending = 0; for (const auto& cq : completions) pending += cq.second.size();
        overlapping |= pending != 0;
      }
    };
    if (count > head && !cancel) publish_suffix = [&, sequence] {
      for (uint32_t i = head; i < count; ++i) addresses[i] = src.seg.ubva.va + uint64_t(source_row(i)) * token_bytes;
      __atomic_store_n(reinterpret_cast<uint64_t*>(mailbox + kSuffixNotifyOffset),
          Progress(sequence, uint32_t(fail_suffix ? Result::kInvalidRequest : Result::kOk)), __ATOMIC_RELEASE);
    };
    read_posts = data_wrs = ready_wrs = data_tails = 0;
    __atomic_store_n(reinterpret_cast<uint64_t*>(mailbox + 64), sequence | kProfileNotify, __ATOMIC_RELEASE);
    uint64_t seen = 0; assert(transport.ReadRequest(&seen).ok() && seen == sequence);
    if (narrow_source_range && sequence == 1) {
      request->sequence = sequence + 1;
      assert(!transport.ReadDescriptor(sequence).ok());
      request->sequence = sequence;
      target->bytes = output.size() - 1;
      assert(!transport.ReadDescriptor(sequence).ok());
      target->bytes = output.size();
    }
    assert(transport.ReadDescriptor(sequence).ok());
    std::atomic<bool> stopped{cancel};
    assert(transport.WriteBatch(&stopped).ok());
    const auto result = cancel ? Result::kTimeout : fail_suffix ? Result::kInvalidRequest : Result::kOk;
    assert(transport.batch_result() == result && read_posts == 0 && ready_wrs == 0);
    if (count > head && result == Result::kOk) assert(overlapping);
    assert(transport.Complete(sequence, result).ok());
    const auto log = CaptureProfile(transport);
    assert(log.find("source_validation=disabled") != std::string::npos);
    assert(log.find("address_validate_us=0.000 ") != std::string::npos);
    assert(reinterpret_cast<Completion*>(hbm + layout.done)->token == Progress(sequence, uint32_t(result)));
    const auto written = result == Result::kOk ? count : head;
    for (uint32_t i = 0; i < written; ++i)
      assert(!std::memcmp(output.data() + uint64_t(i) * token_bytes, source.data() + uint64_t(source_row(i)) * token_bytes, token_bytes));
    for (size_t i = uint64_t(written) * token_bytes; i < output.size(); ++i) assert(output[i] == 0xa5);
    assert(retained.empty() && pending_sges.empty());
  }
  data_post = {}; publish_suffix = {}; retain_chunk_storage = false;
  assert(transport.Shutdown().ok()); std::free(hbm);
}
}
int main() {
  for (bool publisher : {false, true}) for (auto head : {8U, 256U, 512U, 1024U}) {
    for (auto count : {7U, head, head + 1, head + 1024, 8192U})
      RunChunks(count, 656, 0, true, true, 5, true, head, false, publisher, publisher, !publisher);
    for (auto fault : {1, 3, 4, 5})
      RunChunks(8192, 656, fault, true, false, 5, true, head, false, publisher, publisher, !publisher);
    RunChunks(65536, 16, 0, true, true, 5, true, head, false, publisher, publisher, !publisher);
  }
  for (auto head : {8U, 128U, 256U, 512U, 1024U}) {
    for (auto count : {7U, head, head + 1, head + 1024, head + 1025, 8192U})
      RunChunks(count, 656, 0, true, false, 4, true, head, false, true);
    for (auto fault : {1, 3, 4, 5}) RunChunks(8192, 656, fault, true, false, 4, true, head, false, true);
  }
  RunChunks(65536, 16, 0, true, true, 5, true, 256, false, true);
  RunChunks(8192, 656, 0, false, true, 8, true, 1024, false, true);
  for (auto head : {128U, 256U, 512U}) {
    for (auto count : {7U, head, head + 1, head + 1024, head + 1025, 8192U})
      RunChunks(count, 656, 0, true, false, 4, true, head);
    RunChunks(8192, 656, 0, true, true, 5, true, head);
    RunChunks(8192, 656, 1, true, false, 4, true, head);
    RunChunks(8192, 656, 3, true, false, 4, true, head);
    RunChunks(8192, 656, 4, true, false, 4, true, head);
    RunChunks(8192, 656, 5, true, false, 4, true, head);
  }
  RunChunks(257, 656, 0, true, false, 4, true, 256, true);
  RunChunks(65536, 16, 0, true, false, 4, true, 256);
  RunChunks(4097, 656, 0, true, true, 1, true, 1);
  RunChunks(4097, 656, 0, true, false, 8, true, 7);
  RunChunks(4097, 20000, 0, true, false, 4, true, 256);

  RunMailbox(4096, 4); RunMailbox(4096, 8); RunMailbox(8192, 13); RunMailbox(7, 4); RunMailbox(1024, 8); RunMailbox(1025, 4);
  RunMailbox(65536, 4); RunMailbox(4096, 4, true); RunMailbox(4096, 4, false, true); RunMailbox(4096, 4, false, false, true);
  RunMailbox(7, 4, false, true);
  for (auto count : {7U, 1024U, 1025U, 8192U, 65536U}) RunChunks(count, 656, 0, true, false, 4, true);
  RunChunks(8192, 656, 0, true, true, 5, true);
  RunChunks(8192, 656, 1, true, false, 4, true);
  RunChunks(8192, 656, 3, true, false, 4, true);
  RunChunks(8192, 656, 4, true, false, 4, true);
  RunChunks(8192, 656, 5, true, false, 4, true); // STOP drains the accepted first chunk.
  RunChunks(8192, 656, 0, false, true, 4, true);
  std::cout << "CPU device-push mailbox/SGL transport passed (zero descriptor READs, source validation disabled)\n";
}
