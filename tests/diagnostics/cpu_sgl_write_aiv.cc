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
    sgl_model::descriptor = a.workspace + layout.descriptor;
    sgl_model::descriptor_end = a.workspace + layout.staging;
  }
  std::vector<std::future<void>> Launch() {
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
    const auto* target = reinterpret_cast<OutputTarget*>(args.execution.workspace + layout.output_target);
    assert(target->base == args.execution.output && target->bytes == output.size());
  }
  void Complete(uint64_t sequence, Result result = Result::kOk) {
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
      assert(profiles[w].scatter_ns == 0 && profiles[w].scatter_copies == 0 && profiles[w].ready_queries == 0);
      spans += profiles[w].spans;
    }
    if (result == Result::kOk) assert(spans == uint32_t(count));
  }
};
}
int main() {
  for (uint32_t workers : {1U, 4U, 8U}) {
    Fixture f(workers);
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
