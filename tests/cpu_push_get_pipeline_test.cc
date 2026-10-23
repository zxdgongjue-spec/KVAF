#include "cpu_push_get_fixture.h"
#include "cpu_push_get_worker.h"
#include "cpu_push_scatter.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>

using namespace kvaf::cpu_push;
void RunCase(std::uint32_t workers, std::uint32_t count, std::uint32_t bytes) {
  push_mock::Reset(3, 1024);
  {
    PushFixture f(bytes, count, workers);
    Worker producer(std::move(f.transport));
    PushRequire(producer.Start().ok(), "worker start");
    std::vector<std::uint64_t> addresses(count);
    std::vector<std::int32_t> indices(count), offsets(count), rows(count);
    std::vector<std::uint8_t> output(std::uint64_t{count * 2 + 1} * bytes), expected;
    std::iota(indices.begin(), indices.end(), 0);
    std::iota(rows.begin(), rows.end(), 0);
    for (std::uint32_t row = 0; row < count; ++row) {
      addresses[row] = reinterpret_cast<std::uint64_t>(f.source.data()) + std::uint64_t{row} * bytes * 3;
      offsets[row] = (row % 3) * bytes;
    }
    std::mt19937 rng(20261023);
    std::uint64_t sequence = 0;
    // An empty prefix skips CPU notification; the following request must still work.
    for (std::int32_t active : {std::int32_t(count), 0, std::int32_t(count / 2), std::int32_t(count)}) {
      ++sequence;
      for (auto& b : f.source) b = rng() % 251;
      std::shuffle(indices.begin(), indices.end(), rng);
      std::shuffle(rows.begin(), rows.end(), rng);
      std::fill(output.begin(), output.end(), 0xA5);
      expected = output;
      for (std::int32_t i = 0; i < active; ++i)
        std::memcpy(expected.data() + std::uint64_t(rows[i]) * bytes,
            reinterpret_cast<const void*>(addresses[indices[i]] + offsets[i]), bytes);
      Coordination coordination{};
      std::array<WorkerStatus, kMaxWorkers> statuses{};
      std::array<kvaf::BatchGetWorkerProfile, kMaxWorkers> profiles{};
      KernelArgs args{};
      args.request = {reinterpret_cast<std::uint64_t>(addresses.data()),
          reinterpret_cast<std::uint64_t>(indices.data()), reinterpret_cast<std::uint64_t>(offsets.data()),
          reinterpret_cast<std::uint64_t>(&active), reinterpret_cast<std::uint64_t>(rows.data()), count, count, bytes};
      args.workspace = reinterpret_cast<std::uint64_t>(f.memory);
      args.output = reinterpret_cast<std::uint64_t>(output.data()); args.output_bytes = output.size();
      args.coordination = reinterpret_cast<std::uint64_t>(&coordination);
      args.status = reinterpret_cast<std::uint64_t>(statuses.data());
      args.profile = reinterpret_cast<std::uint64_t>(profiles.data());
      args.sequence = sequence; args.workers = workers; args.timeout_ms = 5000;
      std::vector<std::thread> kernels;
      std::array<std::uint32_t, kMaxWorkers> results{};
      for (std::uint32_t w = 0; w < workers; ++w)
        kernels.emplace_back([&, w] { results[w] = ExecuteCpuPushScatter(&args); });
      for (auto& kernel : kernels) kernel.join();
      for (std::uint32_t w = 0; w < workers; ++w) {
        PushRequire(results[w] == 0 && statuses[w].result == std::uint64_t(Result::kOk), "kernel failure");
        PushRequire(profiles[w].exit_ns >= profiles[w].enter_ns, "invalid profile timestamps");
      }
      PushRequire(output == expected, "dynamic scatter payload or untouched output mismatch");
    }
    PushRequire(producer.Stop().ok(), "worker stop");
  }
  PushRequire(push_mock::LiveResources() == 0, "pipeline resource leak");
}
void ErrorCase(bool timeout) {
  push_mock::Reset();
  {
    PushFixture f(656, 1, 1);
    std::unique_ptr<Worker> producer;
    if (!timeout) {
      producer = std::make_unique<Worker>(std::move(f.transport));
      PushRequire(producer->Start().ok(), "error producer start");
    }
    // A valid Tensor layout can still contain an invalid remote source address.
    std::uint64_t address = timeout ? reinterpret_cast<std::uint64_t>(f.source.data()) : 1;
    std::int32_t zero = 0, active = 1;
    Coordination coordination{};
    WorkerStatus status{};
    std::vector<std::uint8_t> output(656, 0xA5);
    KernelArgs args{};
    args.request = {reinterpret_cast<std::uint64_t>(&address), reinterpret_cast<std::uint64_t>(&zero),
        reinterpret_cast<std::uint64_t>(&zero), reinterpret_cast<std::uint64_t>(&active),
        reinterpret_cast<std::uint64_t>(&zero), 1, 1, 656};
    args.workspace = reinterpret_cast<std::uint64_t>(f.memory);
    args.output = reinterpret_cast<std::uint64_t>(output.data()); args.output_bytes = output.size();
    args.coordination = reinterpret_cast<std::uint64_t>(&coordination);
    args.status = reinterpret_cast<std::uint64_t>(&status);
    args.sequence = 1; args.workers = 1; args.timeout_ms = timeout ? 1 : 1000;
    PushRequire(ExecuteCpuPushScatter(&args) == 1, "kernel error ignored");
    PushRequire(status.result == std::uint64_t(timeout ? Result::kTimeout : Result::kInvalidRequest), "wrong kernel error");
    PushRequire(std::all_of(output.begin(), output.end(), [](auto b) { return b == 0xA5; }), "failed request wrote output");
    if (producer) PushRequire(producer->Stop().ok(), "error producer stop");
  }
  PushRequire(push_mock::LiveResources() == 0, "error pipeline resource leak");
}
int main() {
  try {
    RunCase(1, 2053, 656);
    RunCase(4, 4103, 656);
    RunCase(8, 3, 65); // Empty worker partitions and nonaligned row lengths.
    ErrorCase(false);
    ErrorCase(true);
    std::cout << "CPU push pipeline PASS\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
