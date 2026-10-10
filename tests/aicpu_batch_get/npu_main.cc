#include "npu_target.h"
#include "core_probe_report.h"
#include "metrics.h"
#include "scatter_profile_report.h"
#include "transfer_profile.h"
#include "scatter_copy.h"
#include "scatter_probe_report.h"

#include <cstdio>
#include <cstdlib>
#include <limits.h>
#include <map>

namespace {
using namespace kvaf::aicpu_test;
struct Options {
  std::uint32_t device{};
  bool has_device{};
  bool profile_scatter{};
  std::uint32_t aicpu_blocks{1};
  std::uint32_t probe_aicpu_blocks{};
  std::uint32_t probe_hold_ms{20};
  std::uint32_t probe_scatter_blocks{};
  ScatterProbeConfig scatter_probe;
  std::string scatter_probe_case{"all"};
  std::uint64_t probe_trace_rounds{3};
  std::uint16_t port{29991};
  std::string eid, json;
};
Options Parse(int argc, char** argv) {
  Options o;
  bool scatter_probe_option = false;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--profile-scatter") { o.profile_scatter = true; continue; }
    if (flag == "--probe-pmu") { o.scatter_probe.pmu = kScatterProbePmuMask; scatter_probe_option = true; continue; }
    Require(i + 1 < argc, "missing option value");
    const char* value = argv[++i];
    if (flag == "--kernel-json") o.json = value;
    else if (flag == "--eid") o.eid = value;
    else if (flag == "--device-id") {
      const auto n = Number(value); Require(n <= INT32_MAX, "invalid device ID");
      o.device = n; o.has_device = true;
    } else if (flag == "--port") {
      const auto n = Number(value); Require(n > 0 && n <= UINT16_MAX, "invalid port"); o.port = n;
    } else if (flag == "--aicpu-blocks") {
      const auto n = Number(value); Require(n > 0 && n <= kMaxScatterThreads, "invalid AICPU block count");
      o.aicpu_blocks = static_cast<std::uint32_t>(n);
    } else if (flag == "--probe-aicpu-blocks") {
      const auto n = Number(value); Require(n > 0 && n <= kMaxScatterThreads, "invalid probe block count");
      o.probe_aicpu_blocks = static_cast<std::uint32_t>(n);
    } else if (flag == "--probe-hold-ms") {
      const auto n = Number(value); Require(n > 0 && n <= 5000, "invalid probe hold time");
      o.probe_hold_ms = static_cast<std::uint32_t>(n);
    } else if (flag == "--probe-scatter-blocks") {
      const auto n = Number(value); Require(n > 0 && n <= kMaxScatterThreads, "invalid scatter probe block count");
      o.probe_scatter_blocks = static_cast<std::uint32_t>(n); o.scatter_probe.workers = n;
    } else if (flag == "--probe-rows-per-worker") {
      o.scatter_probe.rows_per_worker = Number(value); scatter_probe_option = true;
    } else if (flag == "--probe-row-bytes") {
      o.scatter_probe.row_bytes = Number(value); scatter_probe_option = true;
    } else if (flag == "--probe-iterations") {
      o.scatter_probe.iterations = Number(value); scatter_probe_option = true;
    } else if (flag == "--probe-warmup") {
      o.scatter_probe.warmup = Number(value); scatter_probe_option = true;
    } else if (flag == "--probe-timeout-ms") {
      o.scatter_probe.timeout_ms = Number(value); scatter_probe_option = true;
    } else if (flag == "--probe-pmu-mask") {
      o.scatter_probe.pmu = Number(value); scatter_probe_option = true;
      Require(o.scatter_probe.pmu > 0 && o.scatter_probe.pmu <= kScatterProbePmuMask, "invalid PMU event mask");
    } else if (flag == "--probe-scatter-case") {
      o.scatter_probe_case = value; scatter_probe_option = true;
    } else if (flag == "--probe-trace-rounds") {
      o.probe_trace_rounds = Number(value); scatter_probe_option = true;
      Require(o.probe_trace_rounds <= 50, "scatter probe trace is limited to 50 rounds");
    } else throw std::runtime_error("unknown option " + flag);
  }
  Require(!o.json.empty(), "--kernel-json PATH is required");
  if (o.probe_aicpu_blocks) Require(o.has_device, "--probe-aicpu-blocks requires --device-id");
  Require(!scatter_probe_option || o.probe_scatter_blocks, "scatter probe options require --probe-scatter-blocks");
  if (o.probe_scatter_blocks) {
    Require(o.has_device && !o.probe_aicpu_blocks, "scatter probe requires --device-id and cannot run with core probe");
    ScatterProbeLayout l{}; Require(MakeScatterProbeLayout(o.scatter_probe, &l), "invalid scatter probe shape/rounds");
    bool known = o.scatter_probe_case == "all";
    for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(ScatterProbeMode::kCount); ++i)
      known |= o.scatter_probe_case == ScatterProbeModeName(static_cast<ScatterProbeMode>(i));
    Require(known, "unknown --probe-scatter-case");
  }
  return o;
}
struct DeviceBuffer {
  void* data{};
  ~DeviceBuffer() { if (data) aclrtFree(data); }
  void Allocate(std::uint64_t bytes, int fill = 0) {
    Acl(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate device buffer");
    Acl(aclrtMemset(data, bytes, fill, bytes), "initialize device buffer");
  }
};
class Kernel {
 public:
  ~Kernel() {
    if (stream_) aclrtDestroyStream(stream_);
    if (binary_) aclrtBinaryUnLoad(binary_);
  }
  void Load(const std::string& json, const char* name) {
    char path[PATH_MAX]; Require(realpath(json.c_str(), path), "kernel JSON not found");
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE; option.value.cpuKernelMode = 0;
    aclrtBinaryLoadOptions options{}; options.numOpt = 1; options.options = &option;
    Acl(aclrtBinaryLoadFromFile(path, &options, &binary_), "load AICPU JSON");
    Acl(aclrtBinaryGetFunction(binary_, name, &function_), "resolve AICPU function");
    Acl(aclrtCreateStream(&stream_), "create AICPU stream");
  }
  void Launch(const KernelArgs& args, std::uint32_t block_dim) {
    Layout layout{}; Require(MakeLayout(args.config, &layout), "invalid launch shape");
    Require(block_dim > 0 && block_dim <= kMaxScatterThreads, "invalid AICPU block count");
    Require(!IsDirectOutput(args.config) || (block_dim == 1 && args.coordination == 0 && args.profile == 0),
            "direct output requires one coordinator and no scatter profile/team");
    Require(!IsScatterPipeline(args.config) ||
            (block_dim == ActiveThreads(args.config) && args.coordination != 0),
            "scatter_pipeline requires a block per range and a coordination buffer");
    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(UINT32_MAX, layout.rounds * (args.config.timeout_ms / 1000 + 2) + 60));
    aclrtLaunchKernelCfg cfg{}; cfg.numAttrs = 1; cfg.attrs = &attr;
    Acl(aclrtLaunchKernelWithHostArgs(function_, block_dim, stream_, &cfg,
                                     const_cast<KernelArgs*>(&args), sizeof(args), nullptr, 0),
        "launch AICPU kernel");
  }
  void LaunchProbe(const CoreProbeArgs& args, std::uint32_t block_dim) {
    Require(block_dim > 0 && block_dim <= kMaxScatterThreads, "invalid AICPU probe block count");
    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = 60;
    aclrtLaunchKernelCfg cfg{}; cfg.numAttrs = 1; cfg.attrs = &attr;
    Acl(aclrtLaunchKernelWithHostArgs(function_, block_dim, stream_, &cfg,
        const_cast<CoreProbeArgs*>(&args), sizeof(args), nullptr, 0), "launch AICPU core probe");
  }
  void LaunchScatterProbe(const ScatterProbeArgs& args) {
    ScatterProbeLayout l{}; Require(MakeScatterProbeLayout(args.config, &l), "invalid scatter probe launch");
    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = static_cast<std::uint32_t>(60 +
        (args.config.warmup + args.config.iterations) * (args.config.workers + 4) *
        ((args.config.timeout_ms + 999) / 1000));
    aclrtLaunchKernelCfg cfg{}; cfg.numAttrs = 1; cfg.attrs = &attr;
    Acl(aclrtLaunchKernelWithHostArgs(function_, static_cast<std::uint32_t>(args.config.workers),
        stream_, &cfg, const_cast<ScatterProbeArgs*>(&args), sizeof(args), nullptr, 0),
        "launch AICPU scatter probe");
  }
  aclError Synchronize() { return aclrtSynchronizeStream(stream_); }
 private:
  aclrtBinHandle binary_{};
  aclrtFuncHandle function_{};
  aclrtStream stream_{};
};

void ValidateOutput(const Config& c, const Layout& l, const void* output) {
  // Chunked readback avoids a second whole-output host allocation.
  constexpr std::uint64_t kChunkBytes = 1U << 20;
  std::vector<std::uint8_t> copy(std::min(kChunkBytes, l.output_bytes));
  for (std::uint64_t offset = 0; offset < l.output_bytes; offset += copy.size()) {
    const auto bytes = std::min<std::uint64_t>(copy.size(), l.output_bytes - offset);
    Acl(aclrtMemcpy(copy.data(), copy.size(), static_cast<const std::uint8_t*>(output) + offset,
                     bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read output");
    for (std::uint64_t i = 0; i < bytes; ++i) {
      const auto slot = (offset + i) / l.stride;
      const auto b = (offset + i) % l.stride;
      const auto row = c.rows - 1 - slot;
      const auto tag = RowTag(l.rounds, row);
      const std::uint8_t expected = b >= c.row_bytes ? 0xA5 :
          b < 8 ? static_cast<std::uint8_t>(tag >> (8 * b)) : PayloadPattern(c, l.rounds, row, b);
      Require(copy[i] == expected, "output/guard mismatch at byte " + std::to_string(offset + i));
    }
  }
}

void ValidateDirectOutputGuards(const Layout& layout, const NpuTarget& target) {
  const auto* output = static_cast<const std::uint8_t*>(target.payload());
  std::uint8_t copy[kGuardBytes];
  for (const auto* guard : {output - kGuardBytes, output + layout.output_bytes}) {
    Acl(aclrtMemcpy(copy, sizeof(copy), guard, sizeof(copy), ACL_MEMCPY_DEVICE_TO_HOST),
        "read direct output boundary guard");
    for (auto byte : copy) Require(byte == 0xA5, "direct output boundary guard overwritten");
  }
}

Report Execute(int fd, const Config& config, NpuTarget& target, const std::string& json,
               bool profile_scatter, std::uint32_t aicpu_blocks,
               TransferProfileHeader* transfer_header, std::vector<Sample>* transfer_samples) {
  Layout layout{}; Require(MakeLayout(config, &layout), "invalid config");
  const bool direct_output = IsDirectOutput(config);
  const bool pipeline = IsScatterPipeline(config);
  const auto active_threads = ActiveThreads(config);
  const auto block_dim = direct_output ? 1U : aicpu_blocks;
  const bool collect_profile = profile_scatter && !direct_output;
  Require(block_dim == 1 || block_dim == active_threads,
          "--aicpu-blocks must equal active scatter ranges; use --aicpu-blocks 1 for pthread mode");
  Require(!pipeline || block_dim == active_threads,
          "scatter_pipeline requires --aicpu-blocks equal to active scatter ranges");
  DeviceBuffer output, samples, result, profile, coordination;
  if (!direct_output) output.Allocate(layout.output_bytes, 0xA5);
  void* output_data = direct_output ? target.payload() : output.data;
  samples.Allocate(config.iterations * sizeof(Sample));
  result.Allocate(sizeof(KernelResult));
  if (collect_profile) profile.Allocate(sizeof(ScatterProfile));
  if (block_dim > 1 || pipeline) {
    const auto host_coordination = MakeMultiBlockCoordination(active_threads);
    coordination.Allocate(sizeof(host_coordination));
    Acl(aclrtMemcpy(coordination.data, sizeof(host_coordination), &host_coordination,
                    sizeof(host_coordination), ACL_MEMCPY_HOST_TO_DEVICE), "initialize multi-block coordination");
  }
  Kernel kernel; kernel.Load(json, kKernelName);
  KernelArgs args{reinterpret_cast<std::uint64_t>(target.memory()),
                  reinterpret_cast<std::uint64_t>(output_data),
                  reinterpret_cast<std::uint64_t>(samples.data),
                  reinterpret_cast<std::uint64_t>(result.data), config,
                  reinterpret_cast<std::uint64_t>(profile.data),
                  reinterpret_cast<std::uint64_t>(coordination.data)};
  if (direct_output) {
    std::fprintf(stderr,
                 "AICPU_GET_NPU LAUNCH mode=direct_output transfer_mode=direct_output "
                 "block_dim=1 requested_block_dim=%u active_scatter_ranges=0 "
                 "sgl_layout=source_by_output_slot output_memory=registered_target_payload "
                 "source_stride=%llu output_stride=%llu scatter_clean=none scatter_copy=none "
                 "output_cache=cold_before_request cache_prepare=outside_timing\n",
                 aicpu_blocks, (unsigned long long)layout.source_stride,
                 (unsigned long long)layout.stride);
    if (profile_scatter)
      std::fprintf(stderr, "[AICPU_PROFILE] SKIPPED reason=direct_output_has_no_scatter\n");
  } else if (pipeline) {
    std::fprintf(stderr, "AICPU_GET_NPU LAUNCH mode=multi-block-pipeline block_dim=%u "
        "active_scatter_ranges=%u transfer_mode=scatter_pipeline "
        "sgl_layout=source_by_output_slot scatter_layout=forward "
        "ready=per_range staging=disjoint_full_batch scatter_clean=payload_cache_lines "
        "scatter_copy=%s output_cache=cold_before_request cache_prepare=outside_timing "
        "timing=first_ready_plus_pipeline verify_payload=%llu\n",
        block_dim, active_threads, ScatterCopyBackendName(config.row_bytes),
        (unsigned long long)config.verify_payload);
  } else {
    std::fprintf(stderr,
               "AICPU_GET_NPU LAUNCH mode=%s block_dim=%u active_scatter_ranges=%u "
               "transfer_mode=scatter sgl_layout=source_by_output_slot scatter_layout=forward "
               "scatter_clean=payload_cache_lines scatter_copy=%s "
               "output_cache=cold_before_request cache_prepare=outside_timing\n",
               block_dim == 1 ? "pthread" : "multi-block", block_dim, active_threads,
               ScatterCopyBackendName(config.row_bytes));
  }
  if (config.profile_transfer)
    std::fprintf(stderr, "[AICPU_TRANSFER] DEVICE enabled=1 mode=scatter "
        "request_publish=store_clean_dsb samples=after_run clock=cntvct_el0 revision=%llu\n",
        (unsigned long long)kKernelRevision);
  kernel.Launch(args, block_dim);
  // Once launched, do not unwind allocations until both device execution and
  // CPU transport have stopped. A lost control connection requires retaining
  // the process/resources for operator intervention, not eager buffer free.
  Ack ack{};
  try { Send(fd, &ack, sizeof(ack)); }
  catch (...) { /* Kernel times out; still perform the drain handshake below. */ }
  const auto sync = kernel.Synchronize();
  if (sync != ACL_SUCCESS) {
    // Capture immediately: later ACL calls can replace this thread's error.
    const char* recent = aclGetRecentErrMsg();
    std::fprintf(stderr, "AICPU_GET_NPU STREAM_FAILED code=%d acl_error=%s\n",
                 (int)sync, recent ? recent : "<no recent ACL error>");
  }
  Ack drained{};
  try {
    Receive(fd, &drained, sizeof(drained));
    Require(drained.magic == kMagic, "invalid CPU drain acknowledgement");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Cannot confirm CPU teardown: %s. Target retained; terminate after CPU has exited.\n", e.what());
    for (;;) pause();
  }
  Report report{};
  KernelResult device_result{};
  // An explicit kernel error (timeout, bad sequence, mismatch) can also fail
  // the stream. Try to retrieve its status before discarding this evidence.
  // A processor exception may prevent readback; report that separately.
  const auto result_read = aclrtMemcpy(&device_result, sizeof(device_result), result.data,
                                        sizeof(device_result), ACL_MEMCPY_DEVICE_TO_HOST);
  if (result_read == ACL_SUCCESS) {
    std::fprintf(stderr,
                 "AICPU_GET_NPU KERNEL_STATUS status=%llu completed=%llu failed_sequence=%llu frequency=%llu requested_threads=%llu active_threads=%llu pthread_error=%llu revision=%llu\n",
                 (unsigned long long)device_result.status,
                 (unsigned long long)device_result.completed_rounds,
                 (unsigned long long)device_result.failed_sequence,
                 (unsigned long long)device_result.counter_frequency,
                 (unsigned long long)device_result.requested_threads,
                 (unsigned long long)device_result.active_threads,
                 (unsigned long long)device_result.pthread_error,
                 (unsigned long long)device_result.kernel_revision);
  } else {
    std::fprintf(stderr, "AICPU_GET_NPU KERNEL_STATUS_UNAVAILABLE readback_code=%d\n", (int)result_read);
  }
  if (sync != ACL_SUCCESS || drained.status != 0) {
    std::fprintf(stderr, "AICPU execution failed: stream=%d cpu_status=%llu\n",
                 (int)sync, (unsigned long long)drained.status);
    // Existing transport Shutdown ignores some provider teardown errors.
    // Do not interpret its OK as proof that a failed/partial WRITE is drained.
    // Let the CPU exit, but retain device resources on this abnormal path.
    try { Send(fd, &report, sizeof(report)); } catch (...) {}
    std::fprintf(stderr, "Failed run: target retained. Stop this process after the CPU test has exited.\n");
    for (;;) pause();
  }
  Acl(result_read, "read kernel result");
  Require(device_result.status == Result::kOk, "AICPU returned an error status");
  ValidateOutput(config, layout, output_data);
  if (direct_output) ValidateDirectOutputGuards(layout, target);
  std::vector<Sample> timings(config.iterations);
  Acl(aclrtMemcpy(timings.data(), timings.size() * sizeof(Sample), samples.data,
                   timings.size() * sizeof(Sample), ACL_MEMCPY_DEVICE_TO_HOST), "read timings");
  if (collect_profile) {
    auto device_profile = std::make_unique<ScatterProfile>();
    Acl(aclrtMemcpy(device_profile.get(), sizeof(ScatterProfile), profile.data,
                     sizeof(ScatterProfile), ACL_MEMCPY_DEVICE_TO_HOST), "read scatter profile");
    PrintScatterProfile(stderr, config, device_result.counter_frequency, *device_profile);
  }
  const auto report_result = Summarize(config, device_result, timings);
  if (config.profile_transfer) {
    Require(transfer_header && transfer_samples, "missing transfer profile output");
    transfer_header->rounds = config.iterations;
    transfer_header->frequency = device_result.counter_frequency;
    *transfer_samples = std::move(timings);
  }
  if (pipeline)
    std::fprintf(stderr, "[AICPU_PIPELINE] SUMMARY workers=%u rounds=%llu "
        "avg_first_ready_us=%.3f avg_pipeline_us=%.3f avg_total_us=%.3f "
        "ready_time=observed pipeline_includes=later_range_wait_and_scatter\n",
        active_threads, (unsigned long long)config.iterations,
        report_result.wait_ns / 1000, report_result.scatter_ns / 1000, report_result.mean_ns / 1000);
  return report_result;
}

class CandidateResolver {
 public:
  explicit CandidateResolver(const Options& options) : options_(options) {}
  const std::vector<std::array<std::uint8_t, kvaf::peer::kEidBytes>>& Get(std::uint32_t requested) {
    const auto device = SelectDevice(requested, options_.has_device, options_.device);
    auto found = cache_.find(device);
    if (found != cache_.end()) return found->second;
    Acl(aclrtSetDevice(device), "select discovery device");
    std::uint32_t physical = 0;
    Require(kvaf::direct_hccp::GetCurrentNpuPhysicalDeviceId(&physical), "cannot resolve physical device-id");
    std::vector<std::array<std::uint8_t, kvaf::peer::kEidBytes>> eids;
    if (!options_.eid.empty()) {
      std::array<std::uint8_t, kvaf::peer::kEidBytes> eid{};
      Require(kvaf::npu_eid_detail::ParseEid(options_.eid, &eid), "invalid --eid");
      eids.push_back(eid);
    } else {
      std::string error;
      const bool ok = kvaf::npu_eid_detail::DiscoverDeviceEids(physical, &eids, &error);
      Require(ok, "NPU EID discovery failed: " + error);
    }
    return cache_.emplace(device, std::move(eids)).first->second;
  }
 private:
  const Options& options_;
  std::map<std::uint32_t, std::vector<std::array<std::uint8_t, kvaf::peer::kEidBytes>>> cache_;
};

void HandleDiscovery(int fd, const kvaf::peer::MessageHeader& header, CandidateResolver& resolver,
                     bool configured, std::uint32_t active_device) {
  namespace p = kvaf::peer;
  p::HbmEidCandidateResponse response{};
  response.header = p::MakeHeader(p::Command::kGetHbmEidCandidate, sizeof(response));
  response.header.command = p::ResponseCommand(p::Command::kGetHbmEidCandidate);
  try {
    Require(p::IsValidHeader(header, p::Command::kGetHbmEidCandidate, sizeof(p::HbmEidCandidateRequest)),
            "invalid candidate query");
    p::HbmEidCandidateRequest request{}; request.header = header;
    Receive(fd, reinterpret_cast<char*>(&request) + sizeof(header), sizeof(request) - sizeof(header));
    Require(!configured || request.device_id == active_device, "cannot change device during channel setup");
    const auto& eids = resolver.Get(request.device_id);
    Require(request.candidate_index < eids.size(), "NPU EID candidate out of range");
    response.candidate_count = static_cast<std::uint32_t>(eids.size());
    response.candidate_index = request.candidate_index;
    std::copy(eids[request.candidate_index].begin(), eids[request.candidate_index].end(), response.eid);
  } catch (const std::exception& e) {
    response.result = static_cast<std::int32_t>(p::Result::kInvalidRequest);
    std::fprintf(stderr, "AICPU_GET_NPU discovery: %s\n", e.what());
  }
  Send(fd, &response, sizeof(response));
}

void HandleConfigure(int fd, const kvaf::peer::MessageHeader& header, CandidateResolver& resolver,
                     std::unique_ptr<NpuTarget>& target, Config& config, std::uint32_t& device,
                     std::uint32_t aicpu_blocks) {
  kvaf::peer::AllocateHbmRegionResponse response{};
  response.header = kvaf::peer::MakeHeader(kConfigure, sizeof(response));
  response.header.command = kvaf::peer::ResponseCommand(kConfigure);
  try {
    Require(!target && kvaf::peer::IsValidHeader(header, kConfigure, sizeof(ConfigureRequest)),
            "invalid configure request or target already active");
    ConfigureRequest request{}; request.header = header;
    Receive(fd, reinterpret_cast<char*>(&request) + sizeof(header), sizeof(request) - sizeof(header));
    Layout layout{}; Require(MakeLayout(request.config, &layout), "invalid benchmark config");
    Require(!IsScatterPipeline(request.config) || aicpu_blocks == ActiveThreads(request.config),
            "scatter_pipeline requires --aicpu-blocks equal to active scatter ranges");
    const auto& eids = resolver.Get(request.device_id);
    Require(request.npu_candidate < eids.size(), "invalid NPU EID candidate");
    auto candidate = std::make_unique<NpuTarget>();
    candidate->Initialize(request.device_id, eids[request.npu_candidate], layout.region_bytes);
    if (IsDirectOutput(request.config)) candidate->PrepareDirectOutput(layout);
    response = candidate->exported();
    target = std::move(candidate); config = request.config; device = request.device_id;
    std::fprintf(stderr, "AICPU_GET_NPU device-id=%u npu_candidate=%u\n", device, request.npu_candidate);
  } catch (const std::exception& e) {
    response.result = static_cast<std::int32_t>(kvaf::peer::Result::kResourceError);
    std::fprintf(stderr, "AICPU_GET_NPU configure: %s\n", e.what());
  }
  Send(fd, &response, sizeof(response));
}

void RunCoreProbe(const Options& options) {
  Acl(aclrtSetDevice(options.device), "select AICPU probe device");
  CoreProbeResult host_result{};
  host_result.requested_blocks = options.probe_aicpu_blocks;
  DeviceBuffer device_result;
  device_result.Allocate(sizeof(host_result));
  Acl(aclrtMemcpy(device_result.data, sizeof(host_result), &host_result, sizeof(host_result),
                  ACL_MEMCPY_HOST_TO_DEVICE), "initialize AICPU core probe");
  Kernel kernel;
  kernel.Load(options.json, kCoreProbeKernelName);
  const CoreProbeArgs args{reinterpret_cast<std::uint64_t>(device_result.data),
                           options.probe_hold_ms, options.probe_aicpu_blocks};
  std::fprintf(stderr, "AICPU_GET_NPU CORE_PROBE device-id=%u block_dim=%u hold_ms=%u\n",
               options.device, options.probe_aicpu_blocks, options.probe_hold_ms);
  kernel.LaunchProbe(args, options.probe_aicpu_blocks);
  Acl(kernel.Synchronize(), "synchronize AICPU core probe");
  Acl(aclrtMemcpy(&host_result, sizeof(host_result), device_result.data, sizeof(host_result),
                  ACL_MEMCPY_DEVICE_TO_HOST), "read AICPU core probe");
  PrintCoreProbe(stderr, host_result, options.probe_aicpu_blocks);
  Require(host_result.status == Result::kOk &&
          host_result.claimed == options.probe_aicpu_blocks &&
          host_result.completed == options.probe_aicpu_blocks,
          "AICPU core probe did not run every requested block");
  if (CoreProbeDistinctCpus(host_result) < options.probe_aicpu_blocks ||
      CoreProbeMaxOverlap(host_result) < options.probe_aicpu_blocks)
    std::fprintf(stderr, "[AICPU_CORE_PROBE] MULTICORE_NOT_READY: blockDim completed but blocks did not overlap on distinct CPUs\n");
}

void InitializeProbeSource(void* allocation, std::uint64_t bytes, std::uint64_t row_bytes) {
  constexpr std::uint64_t chunk_size = 1U << 20;
  std::vector<std::uint8_t> chunk(std::min(bytes, chunk_size));
  for (std::uint64_t offset = 0; offset < bytes; offset += chunk.size()) {
    const auto count = std::min<std::uint64_t>(chunk.size(), bytes - offset);
    for (std::uint64_t i = 0; i < count; ++i)
      chunk[i] = Pattern((offset + i) / row_bytes, (offset + i) % row_bytes);
    Acl(aclrtMemcpy(static_cast<std::uint8_t*>(allocation) + kGuardBytes + offset,
        count, chunk.data(), count, ACL_MEMCPY_HOST_TO_DEVICE), "initialize scatter probe source");
  }
}

void ValidateProbeOutput(const ScatterProbeConfig& c, const ScatterProbeLayout& l,
                         const void* allocation, std::uint64_t allocation_bytes) {
  constexpr std::uint64_t chunk_size = 1U << 20;
  std::vector<std::uint8_t> chunk(std::min(allocation_bytes, chunk_size));
  for (std::uint64_t offset = 0; offset < allocation_bytes; offset += chunk.size()) {
    const auto count = std::min<std::uint64_t>(chunk.size(), allocation_bytes - offset);
    Acl(aclrtMemcpy(chunk.data(), count, static_cast<const std::uint8_t*>(allocation) + offset,
        count, ACL_MEMCPY_DEVICE_TO_HOST), "read scatter probe output/guards");
    for (std::uint64_t i = 0; i < count; ++i) {
      const auto position = offset + i;
      std::uint8_t expected = 0xA5;
      if (position >= kGuardBytes && position - kGuardBytes < l.output_bytes) {
        const auto row = (position - kGuardBytes) / l.stride;
        const auto b = (position - kGuardBytes) % l.stride;
        if (b < c.row_bytes) {
          if (ProbeWritesOnly(c.mode)) expected = 0x3C;
          else if (ProbeCopies(c.mode)) expected = Pattern(row, b);
        }
      }
      Require(chunk[i] == expected, "scatter probe payload/guard mismatch at byte " + std::to_string(position));
    }
  }
}

void ValidateProbeChecksums(const ScatterProbeConfig& c, const std::vector<ScatterProbeSample>& samples) {
  if (!ProbeReadsOnly(c.mode) && !ProbeComputesOnly(c.mode)) return;
  for (std::uint64_t worker = 0; worker < c.workers; ++worker) {
    std::uint64_t expected = 0;
    if (ProbeComputesOnly(c.mode)) expected = ProbeCompute(c.rows_per_worker * c.row_bytes / 16, worker + 1);
    else {
      std::uint8_t lanes[16]{};
      for (std::uint64_t row = worker * c.rows_per_worker; row < (worker + 1) * c.rows_per_worker; ++row)
        for (std::uint64_t b = 0; b < c.row_bytes; ++b) lanes[b % 16] ^= Pattern(row, b);
      for (auto byte : lanes) expected = expected * 131 + byte;
    }
    for (std::uint64_t round = 0; round < c.iterations; ++round)
      Require(samples[round * c.workers + worker].checksum == expected, "scatter probe checksum mismatch");
  }
}

void RunScatterProbe(const Options& options) {
  Acl(aclrtSetDevice(options.device), "select scatter probe device");
  const auto& config = options.scatter_probe;
  ScatterProbeLayout maximum{}; Require(MakeScatterProbeLayout(config, &maximum), "invalid scatter probe config");
  // Keep source/output addresses and total allocation size fixed across cases.
  DeviceBuffer source, output, samples, result, coordination;
  source.Allocate(maximum.source_bytes + 2 * kGuardBytes, 0xA5);
  output.Allocate(maximum.output_bytes + 2 * kGuardBytes, 0xA5);
  samples.Allocate(config.iterations * config.workers * sizeof(ScatterProbeSample));
  result.Allocate(sizeof(ScatterProbeResult)); coordination.Allocate(sizeof(ScatterProbeCoordination));
  InitializeProbeSource(source.data, maximum.source_bytes, config.row_bytes);
  Kernel kernel; kernel.Load(options.json, kScatterProbeKernelName);
  for (std::uint64_t mode = 0; mode < static_cast<std::uint64_t>(ScatterProbeMode::kCount); ++mode) {
    auto c = config; c.mode = static_cast<ScatterProbeMode>(mode);
    if (options.scatter_probe_case != "all" && options.scatter_probe_case != ScatterProbeModeName(c.mode)) continue;
    ScatterProbeLayout l{}; Require(MakeScatterProbeLayout(c, &l), "invalid scatter probe case");
    const ScatterProbeResult initial{};
    const ScatterProbeCoordination initial_coordination{};
    Acl(aclrtMemset(output.data, maximum.output_bytes + 2 * kGuardBytes, 0xA5,
        maximum.output_bytes + 2 * kGuardBytes), "reset scatter probe output");
    Acl(aclrtMemcpy(result.data, sizeof(initial), &initial, sizeof(initial), ACL_MEMCPY_HOST_TO_DEVICE),
        "reset scatter probe result");
    Acl(aclrtMemcpy(coordination.data, sizeof(initial_coordination), &initial_coordination,
        sizeof(initial_coordination), ACL_MEMCPY_HOST_TO_DEVICE), "reset scatter probe coordination");
    ScatterProbeArgs args{reinterpret_cast<std::uint64_t>(source.data) + kGuardBytes,
        reinterpret_cast<std::uint64_t>(output.data) + kGuardBytes,
        reinterpret_cast<std::uint64_t>(samples.data), reinterpret_cast<std::uint64_t>(result.data),
        reinterpret_cast<std::uint64_t>(coordination.data), c};
    std::fprintf(stderr, "[AICPU_SCATTER_PROBE] LAUNCH device-id=%u case=%s block_dim=%llu "
        "source_va=0x%llx output_va=0x%llx source_allocation_bytes=%llu output_allocation_bytes=%llu\n",
        options.device, ScatterProbeModeName(c.mode), (unsigned long long)c.workers,
        (unsigned long long)args.source, (unsigned long long)args.output,
        (unsigned long long)(maximum.source_bytes + 2 * kGuardBytes),
        (unsigned long long)(maximum.output_bytes + 2 * kGuardBytes));
    kernel.LaunchScatterProbe(args);
    const auto sync = kernel.Synchronize();
    auto device_result = std::make_unique<ScatterProbeResult>();
    auto device_coordination = std::make_unique<ScatterProbeCoordination>();
    Acl(aclrtMemcpy(device_result.get(), sizeof(*device_result), result.data,
        sizeof(*device_result), ACL_MEMCPY_DEVICE_TO_HOST), "read scatter probe result");
    Acl(aclrtMemcpy(device_coordination.get(), sizeof(*device_coordination), coordination.data,
        sizeof(*device_coordination), ACL_MEMCPY_DEVICE_TO_HOST), "read scatter probe status");
    std::fprintf(stderr, "[AICPU_SCATTER_PROBE] STATUS case=%s stream=%d status=%llu claimed=%llu exited=%llu completed=%llu\n",
        ScatterProbeModeName(c.mode), (int)sync, (unsigned long long)device_coordination->status.value,
        (unsigned long long)device_coordination->claimed.value, (unsigned long long)device_coordination->exited.value,
        (unsigned long long)device_result->completed_workers);
    Acl(sync, "synchronize scatter probe");
    Require(device_coordination->status.value == static_cast<std::uint64_t>(Result::kOk), "scatter probe failed/timed out");
    std::vector<ScatterProbeSample> timings(c.iterations * c.workers);
    Acl(aclrtMemcpy(timings.data(), timings.size() * sizeof(ScatterProbeSample), samples.data,
        timings.size() * sizeof(ScatterProbeSample), ACL_MEMCPY_DEVICE_TO_HOST), "read scatter probe samples");
    ValidateProbeOutput(c, l, output.data, maximum.output_bytes + 2 * kGuardBytes);
    ValidateProbeChecksums(c, timings);
    PrintScatterProbe(stderr, c, *device_result, timings, options.probe_trace_rounds);
    std::uint8_t guard[kGuardBytes];
    for (const auto offset : {std::uint64_t{0}, kGuardBytes + maximum.source_bytes}) {
      Acl(aclrtMemcpy(guard, sizeof(guard), static_cast<const std::uint8_t*>(source.data) + offset,
          sizeof(guard), ACL_MEMCPY_DEVICE_TO_HOST), "read scatter probe source guard");
      for (auto byte : guard) Require(byte == 0xA5, "scatter probe source guard mismatch");
    }
    std::fprintf(stderr, "[AICPU_SCATTER_PROBE] PASS case=%s\n", ScatterProbeModeName(c.mode));
  }
}

void Run(const Options& options) {
  Socket listener(Listen(options.port));
  std::fprintf(stderr, "AICPU_GET_NPU listening port=%u engine=aicpu_memcpy aicpu_blocks=%u\n",
               options.port, options.aicpu_blocks);
  CandidateResolver resolver(options);
  std::unique_ptr<NpuTarget> target;
  Config config{};
  std::uint32_t device = 0;
  for (;;) {
    Socket fd(Accept(listener.get())); SocketTimeout(fd.get(), config.timeout_ms + 120000);
    kvaf::peer::MessageHeader header{}; Receive(fd.get(), &header, sizeof(header));
    if (header.command == static_cast<std::uint16_t>(kvaf::peer::Command::kGetHbmEidCandidate)) {
      HandleDiscovery(fd.get(), header, resolver, target != nullptr, device); continue;
    }
    if (header.command == static_cast<std::uint16_t>(kConfigure)) {
      HandleConfigure(fd.get(), header, resolver, target, config, device, options.aicpu_blocks); continue;
    }
    if (kvaf::peer::IsValidHeader(header, kResetSetup, sizeof(header))) {
      // CPU only requests this after a failed setup, before posting any data.
      target.reset();
      Ack ack{}; Send(fd.get(), &ack, sizeof(ack)); continue;
    }
    Require(target != nullptr, "target must be configured before channel setup or START");
    if (header.command == static_cast<std::uint16_t>(kvaf::peer::Command::kCreateChannel)) {
      try { target->CreateChannel(fd.get(), header); }
      catch (const std::exception& e) {
        std::fprintf(stderr, "AICPU_GET_NPU channel setup failed: %s\n", e.what());
        kvaf::peer::CreateChannelResponse failed{};
        failed.header = kvaf::peer::MakeHeader(kvaf::peer::Command::kCreateChannel, sizeof(failed));
        failed.header.command = kvaf::peer::ResponseCommand(kvaf::peer::Command::kCreateChannel);
        failed.result = static_cast<std::int32_t>(kvaf::peer::Result::kResourceError);
        Send(fd.get(), &failed, sizeof(failed));
      }
      continue;
    }
    Require(kvaf::peer::IsValidHeader(header, kStart, sizeof(header)), "expected START");
    Report report{};
    TransferProfileHeader transfer_header{};
    std::vector<Sample> transfer_samples;
    try { report = Execute(fd.get(), config, *target, options.json, options.profile_scatter,
                           options.aicpu_blocks, &transfer_header, &transfer_samples); }
    catch (const std::exception& e) {
      report.status = 1;
      std::fprintf(stderr, "NPU benchmark failed: %s\n", e.what());
    }
    Send(fd.get(), &report, sizeof(report));
    Require(report.status == 0, "benchmark failed");
    if (config.profile_transfer) {
      Send(fd.get(), &transfer_header, sizeof(transfer_header));
      Send(fd.get(), transfer_samples.data(), transfer_samples.size() * sizeof(Sample));
    }
    std::fprintf(stderr, "AICPU_GET_NPU PASS mean_ns=%.1f e2e_GBps=%.6f\n", report.mean_ns, report.gbps);
    break;
  }
}

}  // namespace
int main(int argc, char** argv) {
  try {
    const auto options = Parse(argc, argv);
    Acl(aclInit(nullptr), "aclInit");
    if (options.probe_scatter_blocks) RunScatterProbe(options);
    else if (options.probe_aicpu_blocks) RunCoreProbe(options);
    else Run(options);
    Acl(aclFinalize(), "aclFinalize");
    return 0;
  }
  catch (const std::exception& e) {
    std::fprintf(stderr, "AICPU_GET_NPU: %s\n", e.what()); return 1;
  }
}
