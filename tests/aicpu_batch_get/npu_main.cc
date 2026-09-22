#include "npu_target.h"
#include "metrics.h"

#include <cstdio>
#include <cstdlib>
#include <limits.h>
#include <map>

namespace {
using namespace kvaf::aicpu_test;
struct Options {
  std::uint32_t device{};
  bool has_device{};
  std::uint16_t port{29991};
  std::string eid, json;
};
Options Parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    Require(i + 1 < argc, "missing option value");
    const char* value = argv[++i];
    if (flag == "--kernel-json") o.json = value;
    else if (flag == "--eid") o.eid = value;
    else if (flag == "--device-id") {
      const auto n = Number(value); Require(n <= INT32_MAX, "invalid device ID");
      o.device = n; o.has_device = true;
    } else if (flag == "--port") {
      const auto n = Number(value); Require(n > 0 && n <= UINT16_MAX, "invalid port"); o.port = n;
    } else throw std::runtime_error("unknown option " + flag);
  }
  Require(!o.json.empty(), "--kernel-json PATH is required");
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
  void Load(const std::string& json) {
    char path[PATH_MAX]; Require(realpath(json.c_str(), path), "kernel JSON not found");
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE; option.value.cpuKernelMode = 0;
    aclrtBinaryLoadOptions options{}; options.numOpt = 1; options.options = &option;
    Acl(aclrtBinaryLoadFromFile(path, &options, &binary_), "load AICPU JSON");
    Acl(aclrtBinaryGetFunction(binary_, "KvafAicpuBatchGet", &function_), "resolve AICPU function");
    Acl(aclrtCreateStream(&stream_), "create AICPU stream");
  }
  void Launch(const KernelArgs& args) {
    Layout layout{}; Require(MakeLayout(args.config, &layout), "invalid launch shape");
    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(UINT32_MAX, layout.rounds * (args.config.timeout_ms / 1000 + 2) + 60));
    aclrtLaunchKernelCfg cfg{}; cfg.numAttrs = 1; cfg.attrs = &attr;
    Acl(aclrtLaunchKernelWithHostArgs(function_, 1, stream_, &cfg,
                                     const_cast<KernelArgs*>(&args), sizeof(args), nullptr, 0),
        "launch AICPU kernel");
  }
  aclError Synchronize() { return aclrtSynchronizeStream(stream_); }
 private:
  aclrtBinHandle binary_{};
  aclrtFuncHandle function_{};
  aclrtStream stream_{};
};

void ValidateOutput(const Config& c, const Layout& l, const DeviceBuffer& output) {
  // Chunked readback avoids a second whole-output host allocation.
  constexpr std::uint64_t kChunkBytes = 1U << 20;
  std::vector<std::uint8_t> copy(std::min(kChunkBytes, l.output_bytes));
  for (std::uint64_t offset = 0; offset < l.output_bytes; offset += copy.size()) {
    const auto bytes = std::min<std::uint64_t>(copy.size(), l.output_bytes - offset);
    Acl(aclrtMemcpy(copy.data(), copy.size(), static_cast<std::uint8_t*>(output.data) + offset,
                     bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read output");
    for (std::uint64_t i = 0; i < bytes; ++i) {
      const auto slot = (offset + i) / l.stride;
      const auto b = (offset + i) % l.stride;
      const auto row = c.rows - 1 - slot;
      const auto tag = RowTag(l.rounds, row);
      const std::uint8_t expected = b >= c.row_bytes ? 0xA5 :
          b < 8 ? static_cast<std::uint8_t>(tag >> (8 * b)) : Pattern(row, b);
      Require(copy[i] == expected, "scatter output/guard mismatch at byte " + std::to_string(offset + i));
    }
  }
}

Report Execute(int fd, const Config& config, NpuTarget& target, const std::string& json) {
  Layout layout{}; Require(MakeLayout(config, &layout), "invalid config");
  DeviceBuffer output, samples, result;
  output.Allocate(layout.output_bytes, 0xA5);
  samples.Allocate(config.iterations * sizeof(Sample));
  result.Allocate(sizeof(KernelResult));
  Kernel kernel; kernel.Load(json);
  KernelArgs args{reinterpret_cast<std::uint64_t>(target.memory()),
                  reinterpret_cast<std::uint64_t>(output.data),
                  reinterpret_cast<std::uint64_t>(samples.data),
                  reinterpret_cast<std::uint64_t>(result.data), config};
  kernel.Launch(args);
  // Once launched, do not unwind allocations until both device execution and
  // CPU transport have stopped. A lost control connection requires retaining
  // the process/resources for operator intervention, not eager buffer free.
  Ack ack{};
  try { Send(fd, &ack, sizeof(ack)); }
  catch (...) { /* Kernel times out; still perform the drain handshake below. */ }
  const auto sync = kernel.Synchronize();
  Ack drained{};
  try {
    Receive(fd, &drained, sizeof(drained));
    Require(drained.magic == kMagic, "invalid CPU drain acknowledgement");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Cannot confirm CPU teardown: %s. Target retained; terminate after CPU has exited.\n", e.what());
    for (;;) pause();
  }
  Report report{};
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
  KernelResult device_result{};
  Acl(aclrtMemcpy(&device_result, sizeof(device_result), result.data, sizeof(device_result),
                   ACL_MEMCPY_DEVICE_TO_HOST), "read kernel result");
  Require(device_result.status == Result::kOk, "AICPU returned an error status");
  ValidateOutput(config, layout, output);
  std::vector<Sample> timings(config.iterations);
  Acl(aclrtMemcpy(timings.data(), timings.size() * sizeof(Sample), samples.data,
                   timings.size() * sizeof(Sample), ACL_MEMCPY_DEVICE_TO_HOST), "read timings");
  return Summarize(config, device_result, timings);
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
                     std::unique_ptr<NpuTarget>& target, Config& config, std::uint32_t& device) {
  kvaf::peer::AllocateHbmRegionResponse response{};
  response.header = kvaf::peer::MakeHeader(kConfigure, sizeof(response));
  response.header.command = kvaf::peer::ResponseCommand(kConfigure);
  try {
    Require(!target && kvaf::peer::IsValidHeader(header, kConfigure, sizeof(ConfigureRequest)),
            "invalid configure request or target already active");
    ConfigureRequest request{}; request.header = header;
    Receive(fd, reinterpret_cast<char*>(&request) + sizeof(header), sizeof(request) - sizeof(header));
    Layout layout{}; Require(MakeLayout(request.config, &layout), "invalid benchmark config");
    const auto& eids = resolver.Get(request.device_id);
    Require(request.npu_candidate < eids.size(), "invalid NPU EID candidate");
    auto candidate = std::make_unique<NpuTarget>();
    candidate->Initialize(request.device_id, eids[request.npu_candidate], layout.region_bytes);
    response = candidate->exported();
    target = std::move(candidate); config = request.config; device = request.device_id;
    std::fprintf(stderr, "AICPU_GET_NPU device-id=%u npu_candidate=%u\n", device, request.npu_candidate);
  } catch (const std::exception& e) {
    response.result = static_cast<std::int32_t>(kvaf::peer::Result::kResourceError);
    std::fprintf(stderr, "AICPU_GET_NPU configure: %s\n", e.what());
  }
  Send(fd, &response, sizeof(response));
}

void Run(const Options& options) {
  Socket listener(Listen(options.port));
  std::fprintf(stderr, "AICPU_GET_NPU listening port=%u engine=aicpu_memcpy\n", options.port);
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
      HandleConfigure(fd.get(), header, resolver, target, config, device); continue;
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
    try { report = Execute(fd.get(), config, *target, options.json); }
    catch (const std::exception& e) { std::fprintf(stderr, "NPU benchmark failed: %s\n", e.what()); }
    Send(fd.get(), &report, sizeof(report));
    Require(report.status == 0, "benchmark failed");
    std::fprintf(stderr, "AICPU_GET_NPU PASS mean_ns=%.1f e2e_GBps=%.6f\n", report.mean_ns, report.gbps);
    break;
  }
}

}  // namespace
int main(int argc, char** argv) {
  try {
    const auto options = Parse(argc, argv);
    Acl(aclInit(nullptr), "aclInit");
    Run(options);
    Acl(aclFinalize(), "aclFinalize");
    return 0;
  }
  catch (const std::exception& e) {
    std::fprintf(stderr, "AICPU_GET_NPU: %s\n", e.what()); return 1;
  }
}
