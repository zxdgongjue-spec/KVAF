#include "aicpu_urma/host.h"
#include "aicpu_urma_client_metrics.h"
#include "aicpu_urma_client_options.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace hccp = kvaf::direct_hccp;
namespace benchmark = kvaf::aicpu_urma;

struct Options {
  std::string kernel_json{kvaf::aicpu_urma_client::DefaultKernelJson()};
  std::uint32_t device_id{0U};
  std::uint32_t workers{5U};
  std::uint32_t batch_wqes{8U};
  std::uint32_t warmup{10U};
  std::uint32_t iterations{1000U};
  std::uint64_t bytes{131072U};
  std::uint32_t eid_list_index{UINT32_MAX};
  kvaf::GatherSqCopyMode copy_mode{kvaf::GatherSqCopyMode::kMemcpy};
};

void Usage(const char* program) {
  std::fprintf(stderr,
      "Usage: %s [--device-id N] [--aicpu-cores N(1..8)] [--bytes N(multiple of 64)] "
      "[--sq-batch-wqes 8|16|32|64] [--sq-copy memcpy|paired] "
      "[--eid-list-index N] [--kernel-json PATH] [--warmup N] [--iterations N]\n"
      "Defaults: aicpu-cores=5 bytes=131072 sq-batch-wqes=8 sq-copy=memcpy warmup=10 iterations=1000.\n"
      "Compares stack -> HBM, stack -> driver SQ, and HBM -> HBM; no peer or doorbell.\n", program);
}

bool Parse(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) return false;
    const std::string flag = argv[index];
    const char* value = argv[index + 1];
    if (flag == "--kernel-json") options->kernel_json = value;
    else if (flag == "--sq-copy") {
      if (std::strcmp(value, "memcpy") == 0) options->copy_mode = kvaf::GatherSqCopyMode::kMemcpy;
      else if (std::strcmp(value, "paired") == 0) options->copy_mode = kvaf::GatherSqCopyMode::kPaired;
      else return false;
    } else {
      std::uint64_t number = 0U;
      if (!kvaf::aicpu_urma_client::Number(value, &number)) return false;
      if (flag == "--bytes") options->bytes = number;
      else {
        if (number > UINT32_MAX) return false;
        const auto parsed = static_cast<std::uint32_t>(number);
        if (flag == "--device-id") options->device_id = parsed;
        else if (flag == "--aicpu-cores") options->workers = parsed;
        else if (flag == "--sq-batch-wqes") options->batch_wqes = parsed;
        else if (flag == "--warmup") options->warmup = parsed;
        else if (flag == "--iterations") options->iterations = parsed;
        else if (flag == "--eid-list-index") options->eid_list_index = parsed;
        else return false;
      }
    }
  }
  return options->device_id <= INT32_MAX && !options->kernel_json.empty() &&
      options->workers != 0U && options->workers <= benchmark::kSqCopyBenchmarkMaxWorkers &&
      options->bytes % 64U == 0U && options->bytes >= options->workers * 64U &&
      options->bytes <= benchmark::kSqCopyBenchmarkQueueBytes * options->workers &&
      (options->batch_wqes == 8U || options->batch_wqes == 16U ||
       options->batch_wqes == 32U || options->batch_wqes == 64U) &&
      options->warmup <= 1000000U && options->iterations != 0U && options->iterations <= 1000000U;
}

void Require(bool condition, const char* operation) {
  if (!condition) throw std::runtime_error(operation);
}

void Acl(aclError error, const char* operation) {
  if (error != ACL_SUCCESS)
    throw std::runtime_error(std::string(operation) + " acl=" + std::to_string(error));
}

struct AclContext {
  explicit AclContext(std::uint32_t device) : device_id(device) {
    Acl(aclInit(nullptr), "aclInit");
    try {
      Acl(aclrtSetDevice(device_id), "aclrtSetDevice");
      active = true;
      Acl(aclrtCreateStream(&stream), "aclrtCreateStream");
    } catch (...) {
      if (active) (void)aclrtResetDevice(device_id);
      (void)aclFinalize();
      throw;
    }
  }
  ~AclContext() {
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    if (active) (void)aclrtResetDevice(device_id);
    (void)aclFinalize();
  }
  std::uint32_t device_id;
  bool active{false};
  aclrtStream stream{nullptr};
};

struct DeviceBuffer {
  explicit DeviceBuffer(std::size_t length) : bytes(length) {
    Acl(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
  }
  ~DeviceBuffer() { if (data != nullptr) (void)aclrtFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void Upload(const void* source, std::size_t length) {
    Acl(aclrtMemcpy(data, bytes, source, length, ACL_MEMCPY_HOST_TO_DEVICE), "upload");
  }
  void Download(void* destination, std::size_t length) const {
    Acl(aclrtMemcpy(destination, length, data, length, ACL_MEMCPY_DEVICE_TO_HOST), "download");
  }
  void* data{nullptr};
  std::size_t bytes;
};

struct DedicatedQueues {
  ~DedicatedQueues() {
    hccp::ReleaseDeviceAivTransport({&api, session, context, token, nullptr,
                                    workers.data(), static_cast<std::uint32_t>(workers.size())});
  }
  void Initialize(const Options& options) {
    std::uint32_t physical = 0U;
    Require(hccp::LoadDirectHccp(&api), "load HCCP");
    Require(hccp::GetCurrentNpuPhysicalDeviceId(&physical), "resolve physical device");
    Require(hccp::AcquireDirectHccpSession(physical, 18U, &session), "acquire HCCP session");
    Require(api.get_dev_eid_info_num != nullptr && api.get_dev_eid_info_list != nullptr,
            "HCCP EID enumeration unavailable");
    const hccp::RaInfo info{hccp::kOfflineHdc, physical};
    std::uint32_t count = 0U;
    Require(api.get_dev_eid_info_num(info, &count) == 0 && count != 0U && count <= 4096U,
            "enumerate NPU EID count");
    std::vector<hccp::DevEidInfo> eids(count);
    const auto capacity = count;
    Require(api.get_dev_eid_info_list(info, eids.data(), &count) == 0 &&
            count != 0U && count <= capacity, "enumerate NPU EIDs");
    Require(options.eid_list_index == UINT32_MAX || options.eid_list_index < count,
            "EID list index out of range");
    for (std::uint32_t index = 0U; index < count; ++index) {
      if (options.eid_list_index != UINT32_MAX && index != options.eid_list_index) continue;
      hccp::CtxInitCfg config{hccp::kOfflineHdc};
      hccp::CtxInitAttr attribute{};
      attribute.phy_id = physical;
      std::memcpy(attribute.ub.eid.raw, eids[index].eid.raw, hccp::kEidBytes);
      const auto result = api.ctx_init(&config, &attribute, &context);
      if (result == 0 && context != nullptr) {
        std::fprintf(stdout, "SQ_COPY_CONTEXT physical_device=%u eid_list_index=%u eid=", physical, index);
        for (const auto byte : eids[index].eid.raw) std::fprintf(stdout, "%02x", byte);
        std::fprintf(stdout, "\n");
        break;
      }
      if (context != nullptr) (void)api.ctx_deinit(context);
      context = nullptr;
    }
    Require(context != nullptr, "create NPU HCCP context");
    hccp::TokenId token_id{};
    Require(api.token_alloc(context, &token_id, &token) == 0 && token != nullptr, "allocate jetty token");
    for (std::uint32_t index = 0U; index < options.workers; ++index) {
      auto& worker = workers[index];
      worker.device_cq_info.in.depth = 1024U;
      worker.device_cq_info.in.ub.mode = hccp::kJfcUserCtlNormal;
      Require(api.cq_create(context, &worker.device_cq_info, &worker.device_user_ctl_cq) == 0 &&
              worker.device_user_ctl_cq != nullptr, "create dedicated CQ");
      Require(hccp::CreateDeviceAivJetty(api, context, worker.device_user_ctl_cq, token,
          0x13572468U, 8192U, "sq_copy_benchmark", &worker.device_user_ctl_jetty.info,
          &worker.device_user_ctl_jetty.handle), "create dedicated SQ");
    }
  }
  hccp::DirectHccpApi api{};
  hccp::DirectHccpSession* session{nullptr};
  void* context{nullptr};
  void* token{nullptr};
  std::array<hccp::DeviceAivWorkerResources, benchmark::kSqCopyBenchmarkMaxWorkers> workers{};
};

struct Samples {
  std::vector<double> wall_ms;
  double host_ns{0.0};
  std::array<double, benchmark::kSqCopyBenchmarkMaxWorkers> start_ns{};
  std::array<double, benchmark::kSqCopyBenchmarkMaxWorkers> copy_ns{};
  std::array<double, benchmark::kSqCopyBenchmarkMaxWorkers> cpu_ns{};
  std::array<double, benchmark::kSqCopyBenchmarkMaxWorkers> offcpu_ns{};
  std::array<std::uint64_t, benchmark::kSqCopyBenchmarkMaxWorkers> cpu_samples{};
};

void Print(const Options& options, const Samples& samples, const char* source, const char* target,
           const std::vector<benchmark::SqCopyBenchmarkQueue>& queues) {
  const auto stats = kvaf::aicpu_urma_client::SummarizeTiming(samples.wall_ms, options.bytes);
  std::fprintf(stdout,
      "SQ_COPY_STATS source=%s target=%s workers=%u total_bytes=%llu chunk_bytes=%u "
      "min_us=%.3f p50_us=%.3f p95_us=%.3f p99_us=%.3f max_us=%.3f mean_us=%.3f GBps=%.3f\n",
      source, target, options.workers, static_cast<unsigned long long>(options.bytes), options.batch_wqes * 192U,
      stats.min_ms * 1000.0, stats.p50_ms * 1000.0, stats.p95_ms * 1000.0,
      stats.p99_ms * 1000.0, stats.max_ms * 1000.0, stats.mean_ms * 1000.0, stats.mean_gbps);
  std::fprintf(stdout,
      "  Copy wall: first worker copy start to last worker copy end; includes final DSB and start spread.\n"
      "  Host launch + sync mean: %.3f us (includes untimed source preparation and data verification).\n"
      "  Per-worker averages (us): start offset | copy including DSB | thread CPU | estimated off-CPU.\n",
      samples.host_ns / options.iterations / 1000.0);
  for (std::uint32_t worker = 0U; worker < options.workers; ++worker) {
    std::fprintf(stdout, "  worker=%u bytes=%llu start_us=%.3f copy_us=%.3f ", worker,
        static_cast<unsigned long long>(queues[worker].bytes),
        samples.start_ns[worker] / options.iterations / 1000.0,
        samples.copy_ns[worker] / options.iterations / 1000.0);
    if (samples.cpu_samples[worker] != 0U)
      std::fprintf(stdout, "thread_cpu_us=%.3f offcpu_estimate_us=%.3f cpu_samples=%llu/%u\n",
          samples.cpu_ns[worker] / samples.cpu_samples[worker] / 1000.0,
          samples.offcpu_ns[worker] / samples.cpu_samples[worker] / 1000.0,
          static_cast<unsigned long long>(samples.cpu_samples[worker]), options.iterations);
    else std::fprintf(stdout, "thread_cpu_us=unavailable offcpu_estimate_us=unavailable\n");
  }
}

int Run(const Options& options) {
  Require(kvaf::aicpu_urma_client::ConfigureCannEnvironment(), "configure CANN environment");
  AclContext acl(options.device_id);
  Require(benchmark::Load(options.kernel_json.c_str()), "load AICPU kernel");
  DedicatedQueues dedicated;
  dedicated.Initialize(options);
  DeviceBuffer hbm(benchmark::kSqCopyBenchmarkQueueBytes * options.workers);
  DeviceBuffer source_hbm(benchmark::kSqCopyBenchmarkQueueBytes * options.workers);
  DeviceBuffer queue_buffer(sizeof(benchmark::SqCopyBenchmarkQueue) * options.workers);
  DeviceBuffer status_buffer(sizeof(benchmark::SqCopyBenchmarkStatus) * options.workers);
  DeviceBuffer counter_buffer(2U * sizeof(std::uint64_t));
  std::vector<benchmark::SqCopyBenchmarkQueue> queues(options.workers);
  const auto blocks = options.bytes / 64U;
  for (std::uint32_t worker = 0U; worker < options.workers; ++worker) {
    queues[worker] = {dedicated.workers[worker].device_user_ctl_jetty.info.ub.sq_buffer_va,
        reinterpret_cast<std::uint64_t>(hbm.data) + worker * benchmark::kSqCopyBenchmarkQueueBytes,
        benchmark::kSqCopyBenchmarkQueueBytes, 0U,
        (blocks / options.workers + (worker < blocks % options.workers ? 1U : 0U)) * 64U,
        reinterpret_cast<std::uint64_t>(source_hbm.data) + worker * benchmark::kSqCopyBenchmarkQueueBytes};
    std::fprintf(stdout, "SQ_COPY_BUFFER worker=%u sq_va=0x%llx hbm_va=0x%llx source_hbm_va=0x%llx capacity=%llu bytes=%llu\n",
        worker, static_cast<unsigned long long>(queues[worker].sq),
        static_cast<unsigned long long>(queues[worker].hbm),
        static_cast<unsigned long long>(queues[worker].source_hbm),
        static_cast<unsigned long long>(queues[worker].capacity),
        static_cast<unsigned long long>(queues[worker].bytes));
  }
  std::array<Samples, 3> samples;
  std::vector<std::uint8_t> source_data(static_cast<std::size_t>(queues.front().bytes));
  std::vector<benchmark::SqCopyBenchmarkStatus> statuses(options.workers);
  benchmark::SqCopyBenchmarkArgs args{reinterpret_cast<std::uint64_t>(queue_buffer.data),
      reinterpret_cast<std::uint64_t>(status_buffer.data), reinterpret_cast<std::uint64_t>(counter_buffer.data),
      0U, options.workers, options.batch_wqes, 0U, options.copy_mode,
      reinterpret_cast<std::uint64_t>(counter_buffer.data) + sizeof(std::uint64_t)};
  const auto rounds = static_cast<std::uint64_t>(options.warmup) + options.iterations;
  for (std::uint64_t round = 0U; round < rounds; ++round) {
    for (std::uint32_t worker = 0U; worker < options.workers; ++worker) {
      const auto& queue = queues[worker];
      for (std::uint64_t index = 0U; index < queue.bytes; ++index)
        source_data[index] = benchmark::SqCopyBenchmarkPattern(
            index, worker, round);
      Acl(aclrtMemcpy(reinterpret_cast<void*>(queue.source_hbm), queue.bytes,
          source_data.data(), queue.bytes, ACL_MEMCPY_HOST_TO_DEVICE), "upload HBM source");
    }
    for (auto& queue : queues) queue.offset = (round % (queue.capacity / queue.bytes)) * queue.bytes;
    queue_buffer.Upload(queues.data(), queues.size() * sizeof(queues.front()));
    args.round = round;
    for (std::uint32_t phase = 0U; phase < 3U; ++phase) {
      const auto target = static_cast<std::uint32_t>((round + phase) % 3U);
      args.target_sq = target;
      Acl(aclrtMemset(counter_buffer.data, counter_buffer.bytes, 0, counter_buffer.bytes), "reset counter");
      Acl(aclrtMemset(status_buffer.data, status_buffer.bytes, 0, status_buffer.bytes), "reset status");
      const auto host_begin = std::chrono::steady_clock::now();
      Require(benchmark::LaunchSqCopyBenchmark(&args, acl.stream) == 0, "launch SQ copy benchmark");
      Acl(aclrtSynchronizeStream(acl.stream), "synchronize SQ copy benchmark");
      const auto host_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - host_begin).count();
      status_buffer.Download(statuses.data(), statuses.size() * sizeof(statuses.front()));
      std::uint64_t first = UINT64_MAX;
      std::uint64_t last = 0U;
      for (std::uint32_t worker = 0U; worker < options.workers; ++worker) {
        const auto& status = statuses[worker];
        if (status.magic != benchmark::kSqCopyBenchmarkMagic || status.result != 0U ||
            status.worker != worker || status.round != round || status.bytes != queues[worker].bytes ||
            status.copy_end_ns <= status.copy_begin_ns) {
          std::fprintf(stderr, "SQ_COPY_FAIL source=%s target=%s worker=%u round=%llu result=%llu "
              "mismatch_offset=%llu expected=0x%llx actual=0x%llx\n",
              target == 2U ? "hbm" : "stack", target == 1U ? "sq" : "hbm", worker,
              static_cast<unsigned long long>(round), static_cast<unsigned long long>(status.result),
              static_cast<unsigned long long>(status.mismatch_offset),
              static_cast<unsigned long long>(status.expected), static_cast<unsigned long long>(status.actual));
          throw std::runtime_error("SQ copy result or data verification failed");
        }
        first = std::min(first, status.copy_begin_ns);
        last = std::max(last, status.copy_end_ns);
      }
      if (round < options.warmup) continue;
      auto& sample = samples[target];
      sample.wall_ms.push_back(static_cast<double>(last - first) / 1000000.0);
      sample.host_ns += host_ns;
      for (std::uint32_t worker = 0U; worker < options.workers; ++worker) {
        const auto& status = statuses[worker];
        const auto elapsed = status.copy_end_ns - status.copy_begin_ns;
        sample.start_ns[worker] += status.copy_begin_ns - first;
        sample.copy_ns[worker] += elapsed;
        if (status.thread_cpu_available != 0U) {
          ++sample.cpu_samples[worker];
          sample.cpu_ns[worker] += status.thread_cpu_ns;
          sample.offcpu_ns[worker] += elapsed > status.thread_cpu_ns ? elapsed - status.thread_cpu_ns : 0U;
        }
      }
    }
  }
  Print(options, samples[0], "stack", "hbm", queues);
  Print(options, samples[1], "stack", "sq", queues);
  Print(options, samples[2], "hbm", "hbm", queues);
  std::fprintf(stdout, "SQ_COPY_BENCHMARK_PASS workers=%u total_bytes=%llu sq_copy=%s "
      "warmup=%u iterations=%u no_doorbell=1\n", options.workers,
      static_cast<unsigned long long>(options.bytes),
      options.copy_mode == kvaf::GatherSqCopyMode::kMemcpy ? "memcpy" : "paired",
      options.warmup, options.iterations);
  return 0;
}

}

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    Usage(argv[0]);
    return 0;
  }
  Options options;
  if (!Parse(argc, argv, &options)) {
    Usage(argv[0]);
    return 2;
  }
  try {
    return Run(options);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "SQ_COPY_BENCHMARK_FAIL %s\n", error.what());
    return 1;
  }
}
