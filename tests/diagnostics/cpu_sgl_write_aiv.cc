#define KVAF_CPU_SGL_AIV_MODEL
#include "get/cpu_sgl_write/urma_aiv/kernel/cpu_sgl_write_aiv.asc"
#include "cpu_push_get_layout.h"
#include <future>
#include <vector>
#include <iostream>

using namespace kvaf::cpu_push;
using namespace std::chrono_literals;
namespace {
constexpr uint32_t kBytes = 656, kCapacity = 9001;
struct Fixture {
  Layout layout;
  std::vector<Word> workspace;
  AivCoordination coordination{};
  std::vector<WorkerStatus> statuses;
  std::vector<kvaf::BatchGetWorkerProfile> profiles;
  std::vector<uint8_t> source, output, valid;
  std::vector<uint64_t> addresses;
  std::vector<int32_t> indices, offsets, rows;
  int32_t count{};
  CpuSglWriteArgs args{};
  explicit Fixture(uint32_t workers) : statuses(workers), profiles(workers),
      source(kCapacity * kBytes * 2), output(kCapacity * kBytes, 0xcc), valid(kCapacity, 1),
      addresses(kCapacity), indices(kCapacity), offsets(kCapacity), rows(kCapacity) {
    assert(MakeLayout(kBytes, workers, &layout, true));
    workspace.resize(layout.bytes / sizeof(Word));
    for (uint32_t i = 0; i < source.size(); ++i) source[i] = uint8_t(i * 37 + i / kBytes);
    for (uint32_t i = 0; i < kCapacity; ++i) {
      addresses[i] = reinterpret_cast<uint64_t>(source.data() + i * kBytes * 2);
      indices[i] = (i * 7) % kCapacity; offsets[i] = (i % 2) * kBytes; rows[i] = i;
    }
    auto& a = args.execution;
    auto& r = a.request;
    r.address_table = reinterpret_cast<uint64_t>(addresses.data());
    r.valid_table = reinterpret_cast<uint64_t>(valid.data());
    r.indices = reinterpret_cast<uint64_t>(indices.data());
    r.offsets = reinterpret_cast<uint64_t>(offsets.data());
    r.output_indices = reinterpret_cast<uint64_t>(rows.data());
    r.miss_count = reinterpret_cast<uint64_t>(&count);
    r.table_capacity = r.request_capacity = kCapacity;
    r.token_bytes = kBytes; r.block_bytes = kBytes * 2;
    r.canonical_base = reinterpret_cast<uint64_t>(source.data()); r.logical_bytes = source.size();
    a.workspace = reinterpret_cast<uint64_t>(workspace.data());
    a.coordination = reinterpret_cast<uint64_t>(&coordination);
    a.status = reinterpret_cast<uint64_t>(statuses.data());
    a.profile = reinterpret_cast<uint64_t>(profiles.data());
    a.output = reinterpret_cast<uint64_t>(output.data()); a.output_bytes = output.size(); a.workers = workers;
    args.output_target.base = a.output; args.output_target.bytes = a.output_bytes;
    args.output_target.token_value = 0x12345678;
    args.output_target.segment_bytes = 48;
    for (uint32_t i = 0; i < kOutputSegmentBytes; ++i) args.output_target.segment[i] = uint8_t(i * 3 + 7);
    sgl_model::descriptor = a.workspace + layout.descriptor;
    sgl_model::descriptor_end = a.workspace + layout.staging;
  }
  std::vector<std::future<void>> Launch() {
    // Catch scalar publication that omits a field on a reused invocation.
    std::memset(statuses.data(), 0xa5, statuses.size() * sizeof(WorkerStatus));
    std::memset(profiles.data(), 0xa5, profiles.size() * sizeof(kvaf::BatchGetWorkerProfile));
    auto* request = reinterpret_cast<Request*>(workspace.data());
    for (auto& reserved : request->reserved_words) reserved = UINT64_MAX;
    std::vector<std::future<void>> work;
    for (uint32_t w = 0; w < args.execution.workers; ++w)
      work.push_back(std::async(std::launch::async, [this, w] {
        sgl_model::core = w;
        KvafCpuSglWriteAivKernelV1(args);
      }));
    return work;
  }
  void WaitNotify(uint64_t sequence) {
    const auto end = std::chrono::steady_clock::now() + 5s;
    while ((sgl_model::Load(args.execution.workspace + layout.notify) & ~kProfileNotify) != sequence) {
      assert(std::chrono::steady_clock::now() < end);
    }
    const auto* request = reinterpret_cast<Request*>(workspace.data());
    assert(request->magic == kMagic && request->sequence == sequence && request->count == uint32_t(count));
    assert(request->mode == kDirectOutput && request->workers == args.execution.workers);
    assert(request->token_bytes == kBytes);
    assert(request->reserved_words[0] == ((args.execution.sequence & kPipelineLaunch) ? std::min(uint32_t(count), kPipelineHeadRows) : 0U));
    for (uint32_t i = 1; i < 4; ++i) assert(request->reserved_words[i] == 0);
    const auto* target = reinterpret_cast<OutputTarget*>(args.execution.workspace + layout.output_target);
    assert(target->base == args.execution.output && target->bytes == output.size());
    assert(target->token_value == args.output_target.token_value);
    assert(target->segment_bytes == args.output_target.segment_bytes);
    assert(!std::memcmp(target->segment, args.output_target.segment, kOutputSegmentBytes));
  }
  void Complete(uint64_t sequence, Result result = Result::kOk) {
    if ((args.execution.sequence & kPipelineLaunch)) {
      const auto deadline = std::chrono::steady_clock::now() + 5s;
      while ((sgl_model::Load(args.execution.workspace + kSuffixNotifyOffset) >> 32) != sequence)
        assert(std::chrono::steady_clock::now() < deadline);
    }
    if (result == Result::kOk) {
      const auto* sources = reinterpret_cast<uint64_t*>(args.execution.workspace + layout.descriptor);
      for (int32_t i = 0; i < count; ++i) {
        assert(sources[i] == addresses[indices[i]] + uint32_t(offsets[i]));
        std::memcpy(output.data() + i * kBytes, reinterpret_cast<void*>(sources[i]), kBytes);
      }
    }
    auto* done = reinterpret_cast<Completion*>(args.execution.workspace + layout.done);
    done->cpu_sgl_write = {1234, 100, 1, 4};
    sgl_model::Store(reinterpret_cast<uint64_t>(&done->token), Progress(sequence, uint32_t(result)));
  }
  void Check(std::vector<std::future<void>>& work, uint64_t sequence, Result result) {
    for (auto& w : work) { assert(w.wait_for(5s) == std::future_status::ready); w.get(); }
    uint64_t spans = 0;
    for (uint32_t w = 0; w < args.execution.workers; ++w) {
      assert(statuses[w].magic == kMagic && statuses[w].sequence == sequence && statuses[w].worker == w);
      assert(statuses[w].result == uint64_t(result) && statuses[w].consumed == 0);
      for (auto reserved : statuses[w].reserved) assert(reserved == 0);
      assert(profiles[w].scatter_ns == 0 && profiles[w].scatter_copies == 0 && profiles[w].ready_queries == 0);
      assert(profiles[w].enter_ns && profiles[w].exit_ns >= profiles[w].enter_ns);
      auto remaining = profiles[w].exit_ns - profiles[w].enter_ns;
      for (const auto stage : {profiles[w].request_ns, profiles[w].notify_ns, profiles[w].ack_ns}) {
        assert(stage <= remaining);
        remaining -= stage;
      }
      if (w) assert(profiles[w].notify_ns == 0 && profiles[w].ack_ns == 0);
      assert(profiles[w].bytes == profiles[w].spans * kBytes);
      assert(profiles[w].input_cache_ns == 0 && profiles[w].descriptor_cache_ns == 0 && profiles[w].source_write_ns == 0);
      assert(profiles[w].read_ns == 0 && profiles[w].read_setup_ns == 0 && profiles[w].wr_build_ns == 0);
      assert(profiles[w].ready_wait_ns == 0 && profiles[w].post_ns == 0 && profiles[w].cq_wait_ns == 0);
      assert(profiles[w].read_wqes == 0 && profiles[w].post_calls == 0 && profiles[w].output_cache_ns == 0);
      const auto& sample = profiles[w].cpu_sgl_write;
      if (w == 0 && count > 0 && result == Result::kOk)
        assert(sample.elapsed_ns == 1234 && sample.data_wrs == 100 && sample.waves == 1 && sample.tail_cqes == 4);
      else assert(sample.elapsed_ns == 0 && sample.data_wrs == 0 && sample.waves == 0 && sample.tail_cqes == 0);
      spans += profiles[w].spans;
    }
    if (result == Result::kOk) assert(spans == uint32_t(count));
  }
};
}
int main() {
  for (bool pipeline : {false, true}) for (uint32_t workers : {1U, 4U, 8U}) {
    Fixture f(workers);
    f.args.execution.sequence = pipeline ? kPipelineLaunch : 0;
    uint64_t sequence = 0;
    for (uint32_t count : {9001U, 7U, 0U, 1U, 4097U}) {
      f.count = count; ++sequence;
      std::fill(f.output.begin(), f.output.end(), 0xcc);
      auto work = f.Launch();
      if (count) {
        f.WaitNotify(sequence);
        for (auto& w : work) assert(w.wait_for(1ms) == std::future_status::timeout);
        f.Complete(sequence);
      }
      f.Check(work, sequence, Result::kOk);
      assert(f.coordination.shared.claimed.value == sequence * workers);
      for (uint32_t i = 0; i < count; ++i)
        assert(!std::memcmp(f.output.data() + i * kBytes, reinterpret_cast<void*>(f.addresses[f.indices[i]] + f.offsets[i]), kBytes));
      for (size_t i = count * kBytes; i < f.output.size(); ++i) assert(f.output[i] == 0xcc);
      if (count) assert(f.profiles[0].cpu_sgl_write.elapsed_ns == 1234);
      // Changed tensors are consumed on the next replay, with the same args.
      f.indices[0] = (f.indices[0] + 1) % kCapacity;
    }
  }
  {
    Fixture f(1); f.args.execution.sequence = kPipelineLaunch; f.count = kCapacity; f.rows[7000] = -1;
    auto work = f.Launch(); f.WaitNotify(1);
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while ((sgl_model::Load(f.args.execution.workspace + kSuffixNotifyOffset) >> 32) != 1)
      assert(std::chrono::steady_clock::now() < deadline);
    assert(uint32_t(sgl_model::Load(f.args.execution.workspace + kSuffixNotifyOffset)) == uint32_t(Result::kInvalidRequest));
    assert(work[0].wait_for(1ms) == std::future_status::timeout);
    f.Complete(1, Result::kInvalidRequest);
    f.Check(work, 1, Result::kInvalidRequest);
  }
  for (int fault = 0; fault < 8; ++fault) {
    Fixture f(4); f.count = kCapacity;
    if (fault == 0) f.rows[7000] = 0; // Failure outside core zero's slice.
    if (fault == 1) { --f.args.execution.output_bytes; --f.args.output_target.bytes; }
    if (fault == 2) f.indices[7000] = -1;
    if (fault == 3) f.offsets[7000] = -1;
    if (fault == 4) f.valid[f.indices[7000]] = 0;
    if (fault == 5) f.count = -1;
    if (fault == 6) f.count = kCapacity + 1;
    if (fault == 7) f.addresses[f.indices[7000]] = f.args.execution.request.canonical_base - 1;
    auto work = f.Launch(); f.Check(work, 1, Result::kInvalidRequest);
    assert(sgl_model::Load(f.args.execution.workspace + f.layout.notify) == 0);
    for (auto byte : f.output) assert(byte == 0xcc);
    f.rows[7000] = 7000; // A queued replay cannot clear a prior device failure.
    work = f.Launch(); f.Check(work, 2, Result::kInvalidRequest);
    assert(sgl_model::Load(f.args.execution.workspace + f.layout.notify) == 0);
  }
  {
    Fixture f(4); f.args.execution.sequence = kPipelineLaunch; f.count = kCapacity; f.rows[17] = -1;
    auto work = f.Launch(); f.Check(work, 1, Result::kInvalidRequest);
    assert(sgl_model::Load(f.args.execution.workspace + f.layout.notify) == 0);
    for (auto byte : f.output) assert(byte == 0xcc);
  }
  {
    Fixture f(4); f.count = 7;
    auto work = f.Launch(); f.WaitNotify(1); f.Complete(1, Result::kTransportError);
    f.Check(work, 1, Result::kTransportError);
  }
  {
    Fixture f(4); f.count = 1; f.args.execution.timeout_ms = 1;
    auto work = f.Launch(); f.Check(work, 1, Result::kTimeout);
  }
  assert(sgl_model::descriptor_copies > 0);
  std::cout << "AIV CPU SGL kernel model passed\n";
}
