// Host model of the actual AICPU kernel. No CANN/URMA SDK required.
#include "get/scatter/urma_aicpu/kernel/cpu_push_scatter.h"
#include "execution_config.h"
#include "execution_protocol.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

namespace {
using namespace kvaf;
using namespace kvaf::cpu_push;
constexpr std::uint32_t kCapacity = 4097, kBytes = 656;
struct Fixture {
  Layout layout;
  void* workspace{};
  Coordination coordination;
  std::vector<WorkerStatus> statuses;
  std::vector<BatchGetWorkerProfile> profiles;
  std::vector<std::uint8_t> source, output;
  std::vector<std::uint64_t> addresses;
  std::vector<std::int32_t> indices, offsets, rows;
  std::int32_t count{kCapacity};
  CpuSglWriteArgs direct_args;
  KernelArgs& args{direct_args.execution};
  explicit Fixture(std::uint32_t workers = 4)
      : statuses(workers), profiles(workers), source(8 * 4 * kBytes),
        output(kCapacity * kBytes, 0xA5), addresses(8),
        indices(kCapacity), offsets(kCapacity), rows(kCapacity) {
    assert(MakeLayout(kBytes, workers, &layout, true));
    assert(layout.bytes == layout.staging);  // No payload staging allocation.
    workspace = std::aligned_alloc(64, layout.bytes);
    assert(workspace);
    std::memset(workspace, 0, layout.bytes);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = (i * 17 + i / kBytes) % 251;
    for (std::size_t i = 0; i < addresses.size(); ++i)
      addresses[i] = reinterpret_cast<std::uint64_t>(source.data() + i * 4 * kBytes);
    for (std::size_t i = 0; i < indices.size(); ++i) {
      indices[i] = (i * 3) % addresses.size();
      offsets[i] = (i % 4) * kBytes;
      rows[i] = i;
    }
    args.request.address_table = reinterpret_cast<std::uint64_t>(addresses.data());
    args.request.indices = reinterpret_cast<std::uint64_t>(indices.data());
    args.request.offsets = reinterpret_cast<std::uint64_t>(offsets.data());
    args.request.output_indices = reinterpret_cast<std::uint64_t>(rows.data());
    args.request.miss_count = reinterpret_cast<std::uint64_t>(&count);
    args.request.request_capacity = kCapacity;
    args.request.token_bytes = kBytes;
    args.workspace = reinterpret_cast<std::uint64_t>(workspace);
    args.output = reinterpret_cast<std::uint64_t>(output.data());
    args.output_bytes = output.size();
    args.coordination = reinterpret_cast<std::uint64_t>(&coordination);
    args.status = reinterpret_cast<std::uint64_t>(statuses.data());
    args.profile = reinterpret_cast<std::uint64_t>(profiles.data());
    args.workers = workers;
    args.timeout_ms = 1000;
    direct_args.output_target.base = args.output;
    direct_args.output_target.bytes = output.size();
  }
  ~Fixture() { std::free(workspace); }
  std::uint8_t* memory() { return static_cast<std::uint8_t*>(workspace); }
  std::vector<std::future<std::uint32_t>> Launch() {
    std::vector<std::future<std::uint32_t>> result;
    for (std::uint32_t w = 0; w < args.workers; ++w)
      result.push_back(std::async(std::launch::async, [this] { return ExecuteCpuSglWrite(&direct_args); }));
    return result;
  }
  void Run(bool success, bool produce = true) {
    const auto sequence = coordination.claimed.value / args.workers + 1;
    auto kernels = Launch();
    if (produce && count) {
      auto* notify = reinterpret_cast<Word*>(memory() + layout.notify);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while ((__atomic_load_n(&notify->value, __ATOMIC_ACQUIRE) & ~kProfileNotify) != sequence) {
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
      }
      const auto* request = reinterpret_cast<Request*>(memory());
      assert(request->mode == kDirectOutput && request->count == count);
      const auto* target = reinterpret_cast<OutputTarget*>(memory() + layout.output_target);
      assert(target->base == args.output && target->bytes == output.size());
      // Every worker must remain pending without done, even though no scatter
      // or ready consumption is required by this mode.
      for (auto& kernel : kernels)
        assert(kernel.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout);
      const auto* sources = reinterpret_cast<const std::uint64_t*>(memory() + layout.descriptor);
      const auto head = (args.sequence & kPipelineLaunch) ? std::min(std::uint32_t(count), kPipelineHeadRows) : 0U;
      if ((args.sequence & kPipelineLaunch)) {
        assert(request->reserved_words[0] == head);
        for (std::uint32_t i = 0; i < head; ++i)
          std::memcpy(output.data() + i * kBytes, reinterpret_cast<void*>(sources[i]), kBytes);
        const auto* suffix = reinterpret_cast<const std::uint64_t*>(memory() + kSuffixNotifyOffset);
        while ((__atomic_load_n(suffix, __ATOMIC_ACQUIRE) >> 32) != sequence)
          assert(std::chrono::steady_clock::now() < deadline);
        assert(std::uint32_t(*suffix) == std::uint32_t(Result::kOk));
      }
      for (int i = head; i < count; ++i)
        std::memcpy(output.data() + i * kBytes, reinterpret_cast<void*>(sources[i]), kBytes);
      auto* done = reinterpret_cast<Completion*>(memory() + layout.done);
      done->cpu_sgl_write = {sequence * 1000, 10, 1, 4};
      __atomic_store_n(&done->token, Progress(sequence, std::uint32_t(Result::kOk)), __ATOMIC_RELEASE);
    }
    for (auto& kernel : kernels) assert((kernel.get() == 0) == success);
    if (!success) {
      assert((reinterpret_cast<Word*>(memory() + layout.notify)->value & ~kProfileNotify) == 0);
      for (auto value : output) assert(value == 0xA5);
      return;
    }
    for (std::uint32_t w = 0; w < args.workers; ++w) {
      assert(statuses[w].sequence == sequence && statuses[w].result == std::uint64_t(Result::kOk));
      assert(profiles[w].scatter_ns == 0 && profiles[w].scatter_copies == 0);
      assert(profiles[w].ready_queries == 0);
      const auto& sample = profiles[w].cpu_sgl_write;
      if (w == 0 && count) {
        assert(sample.elapsed_ns == sequence * 1000 && sample.waves == 1);
        assert(sample.data_wrs == 10 && sample.tail_cqes == 4);
      } else {
        assert(sample.elapsed_ns == 0 && sample.waves == 0);
      }
    }
    for (int i = 0; i < count; ++i)
      assert(std::memcmp(output.data() + i * kBytes,
          reinterpret_cast<void*>(addresses[indices[i]] + offsets[i]), kBytes) == 0);
    for (std::size_t i = count * kBytes; i < output.size(); ++i) assert(output[i] == 0xA5);
  }
};
}  // namespace

int main() {
  for (bool pipeline : {false, true}) for (auto workers : {1U, 4U, 8U}) {
    Fixture f(workers);
    f.args.sequence = pipeline ? kPipelineLaunch : 0;
    f.Run(true);
    std::fill(f.output.begin(), f.output.end(), 0xA5);
    f.count = 7;
    f.indices[0] = 7;
    f.Run(true);  // Replay uses a new device ticket and dynamic inputs.
  }
  {
    // A bad suffix must still wait for the CPU to retire the published head.
    Fixture f(1); f.args.sequence = kPipelineLaunch; f.rows[kPipelineHeadRows + 17] = -1;
    auto kernels = f.Launch();
    auto* notify = reinterpret_cast<Word*>(f.memory() + f.layout.notify);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((__atomic_load_n(&notify->value, __ATOMIC_ACQUIRE) & ~kProfileNotify) != 1)
      assert(std::chrono::steady_clock::now() < deadline);
    auto* suffix = reinterpret_cast<std::uint64_t*>(f.memory() + kSuffixNotifyOffset);
    while ((__atomic_load_n(suffix, __ATOMIC_ACQUIRE) >> 32) != 1)
      assert(std::chrono::steady_clock::now() < deadline);
    assert(std::uint32_t(*suffix) == std::uint32_t(Result::kInvalidRequest));
    assert(kernels[0].wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout);
    auto* done = reinterpret_cast<Completion*>(f.memory() + f.layout.done);
    __atomic_store_n(&done->token, Progress(1, std::uint32_t(Result::kInvalidRequest)), __ATOMIC_RELEASE);
    assert(kernels[0].get() != 0);
    assert(f.statuses[0].result == std::uint64_t(Result::kInvalidRequest));
  }
  { Fixture f; f.count = 0; f.rows[0] = -1; f.Run(true, false); }
  { Fixture f; f.args.sequence = kPipelineLaunch; f.count = 0; f.rows[0] = -1; f.Run(true, false); }
  { Fixture f; f.args.sequence = kPipelineLaunch; f.rows[17] = -1; f.Run(false, false); }
  { Fixture f; f.count = 1; f.rows[1] = -1; f.Run(true); }
  { Fixture f; f.rows[17] = 0; f.Run(false, false); }
  { Fixture f; f.args.output_bytes = kBytes; f.Run(false, false); }
  {
    // The shared kernel must retain Scatter's arbitrary destination mapping.
    Fixture f(1);
    f.count = 3;
    std::free(f.workspace);
    assert(MakeLayout(kBytes, 1, &f.layout));
    f.workspace = std::aligned_alloc(64, f.layout.bytes);
    assert(f.workspace);
    std::memset(f.workspace, 0, f.layout.bytes);
    f.args.workspace = reinterpret_cast<std::uint64_t>(f.workspace);
    f.rows[0] = 2; f.rows[1] = 0; f.rows[2] = 1;
    auto kernel = std::async(std::launch::async, [&] { return ExecuteCpuPushScatter(&f.args); });
    auto* notify = reinterpret_cast<Word*>(f.memory() + f.layout.notify);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((__atomic_load_n(&notify->value, __ATOMIC_ACQUIRE) & ~kProfileNotify) != 1) {
      assert(std::chrono::steady_clock::now() < deadline);
      std::this_thread::yield();
    }
    for (int i = 0; i < f.count; ++i)
      std::memcpy(f.memory() + f.layout.staging + i * kBytes,
          reinterpret_cast<void*>(f.addresses[f.indices[i]] + f.offsets[i]), kBytes);
    auto* ready = reinterpret_cast<Ready*>(f.memory() + f.layout.ready);
    for (auto& jetty : ready->jetty) __atomic_store_n(&jetty, Progress(1, 1), __ATOMIC_RELEASE);
    auto* done = reinterpret_cast<Completion*>(f.memory() + f.layout.done);
    __atomic_store_n(&done->token, Progress(1, std::uint32_t(Result::kOk)), __ATOMIC_RELEASE);
    assert(kernel.get() == 0 && f.profiles[0].scatter_copies == 3);
    for (int i = 0; i < f.count; ++i)
      assert(std::memcmp(f.output.data() + f.rows[i] * kBytes,
          reinterpret_cast<void*>(f.addresses[f.indices[i]] + f.offsets[i]), kBytes) == 0);
  }
  const GetExecutionConfig config{Transport::Urma, Executor::Aicpu, GetAlgorithm::CpuSglWrite};
  assert(ValidateExecution(TransportMode::Urma, config).ok());
  assert(!ValidateExecution(TransportMode::Ubmem, config).ok());
  assert(ValidateExecution(TransportMode::Urma,
      GetExecutionConfig{Transport::Urma, Executor::Aiv, GetAlgorithm::CpuSglWrite}).ok());
  execution::Request request;
  request.op = execution::Op::PrepareCpuPullRegion;
  request.algorithm = config.algorithm;
  execution::Request decoded;
  assert(execution::Decode(execution::Encode(request), &decoded).ok());
  assert(decoded.op == request.op);
  assert(decoded.algorithm == config.algorithm);
}
