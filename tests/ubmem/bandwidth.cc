#include "bandwidth_utils.h"
#include "peer_link.h"
#include "ubmem_batch_abi.h"
#include "ubmem/device_mapping.h"
#include "test_pattern.h"

#include <acl/acl.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace kvaf::ubmem;
void Require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void Check(const kvaf::Status& status) { Require(status.ok(), status.message()); }
std::uint64_t Address(const void* ptr) { return reinterpret_cast<std::uintptr_t>(ptr); }

struct Config {
  std::string peer, kernel, pattern = "random";
  std::uint32_t device = 0, cores = 2, rows = 4096, row_bytes = 656;
  std::uint32_t remote_blocks = 4096, tokens_per_block = 128;
  std::uint32_t warmup = 3, iterations = 100, clock_mhz = 1000, seed = 20260903;
};

Config Parse(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    Require(i + 1 < argc, "missing value for " + option);
    const char* value = argv[++i];
    if (option == "--peer") c.peer = value;
    else if (option == "--kernel") c.kernel = value;
    else if (option == "--pattern") c.pattern = value;
    else {
      char* end = nullptr;
      errno = 0;
      const auto n = std::strtoull(value, &end, 10);
      Require(!errno && value[0] != '-' && end != value && *end == '\0' && n <= UINT32_MAX,
          "invalid " + option);
      if (option == "--device") c.device = n;
      else if (option == "--cores") c.cores = n;
      else if (option == "--rows") c.rows = n;
      else if (option == "--token-bytes") c.row_bytes = n;
      else if (option == "--remote-blocks") c.remote_blocks = n;
      else if (option == "--tokens-per-block") c.tokens_per_block = n;
      else if (option == "--warmup") c.warmup = n;
      else if (option == "--iterations") c.iterations = n;
      else if (option == "--clock-mhz") c.clock_mhz = n;
      else if (option == "--seed") c.seed = n;
      else throw std::runtime_error("unknown option " + option);
    }
  }
  Require(!c.peer.empty() && !c.kernel.empty(), "required: --peer CPU:PORT --kernel PATH");
  Require(c.pattern == "random" || c.pattern == "contiguous", "--pattern must be random or contiguous");
  Require(c.rows && c.row_bytes && c.remote_blocks && c.tokens_per_block && c.iterations && c.clock_mhz &&
      c.cores && c.cores <= 32, "positive shape/iterations/clock required; cores must be 1..32");
  const auto tokens = static_cast<std::uint64_t>(c.remote_blocks) * c.tokens_per_block;
  Require(tokens <= (SIZE_MAX - (kPageBytes - 1)) / c.row_bytes,
      "shape exceeds addressable memory");
  return c;
}

struct Benchmark {
  explicit Benchmark(const Config& c) : config(c), link(c.peer), mapping(new DeviceMapping) {}
  Config config;
  kvaf::peer::PeerLink link;
  kvaf::peer::PeerRegion region{};
  std::unique_ptr<DeviceMapping> mapping;
  aclrtStream stream{};
  aclrtBinHandle binary{};
  aclrtFuncHandle function{};
  std::vector<char> binary_bytes;
  std::vector<void*> allocations;
  void *output{}, *requests_device{}, *reports_device{};

  ~Benchmark() {
    if (!Close()) {
      // Match the existing UBmem runner: failed device work may still refer
      // to these addresses. Retain resources until process exit.
      (void)mapping.release();
      std::fprintf(stderr, "[KVAF_UBMEM] retaining bandwidth resources after failed drain/cleanup\n");
    }
  }
  bool Close() {
    if (stream && !Acl("bandwidth drain", aclrtSynchronizeStream(stream)).ok()) return false;
    if (binary) {
      if (!Acl("aclrtBinaryUnLoad", aclrtBinaryUnLoad(binary)).ok()) return false;
      binary = nullptr;
    }
    while (!allocations.empty()) {
      if (!Acl("aclrtFree", aclrtFree(allocations.back())).ok()) return false;
      allocations.pop_back();
    }
    if (mapping && !mapping->Close().ok()) return false;
    if (region.session_id) {
      const auto status = link.ReleaseRegion(region.base);
      if (!status.ok()) { std::fprintf(stderr, "%s\n", status.message().c_str()); return false; }
      region = {};
    }
    if (stream) {
      if (!Acl("aclrtDestroyStream", aclrtDestroyStream(stream)).ok()) return false;
      stream = nullptr;
    }
    return true;
  }
  void* Allocate(std::size_t bytes) {
    void* ptr = nullptr;
    Check(Acl("aclrtMalloc", aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY)));
    allocations.push_back(ptr);
    return ptr;
  }
  void Initialize() {
    const auto& c = config;
    const auto remote_bytes = static_cast<std::uint64_t>(c.remote_blocks) * c.tokens_per_block * c.row_bytes;
    Check(Acl("aclrtCreateStream", aclrtCreateStream(&stream)));
    Check(link.AllocateUbmemRegion(remote_bytes, c.row_bytes, &region));
    Check(mapping->Import(region.fabric_handle, region.fabric_mapping_bytes));
    // CPU establishes the source pattern once, outside all timed samples.
    Check(link.UbmemRange(region.base, 0, region.length, c.seed, true));
    output = Allocate(static_cast<std::uint64_t>(c.rows) * c.row_bytes);
    requests_device = Allocate(static_cast<std::size_t>(c.rows) * sizeof(RawCopyRequest));
    reports_device = Allocate(c.cores * sizeof(RawCopyReport));
    std::ifstream file(c.kernel, std::ios::binary | std::ios::ate);
    Require(file && file.tellg() > 0, "cannot read UBmem kernel object");
    binary_bytes.resize(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    Require(static_cast<bool>(file.read(binary_bytes.data(), binary_bytes.size())), "incomplete kernel object");
    aclrtBinary input = aclrtCreateBinary(binary_bytes.data(), binary_bytes.size());
    Require(input != nullptr, "aclrtCreateBinary failed");
    const auto load = Acl("aclrtBinaryLoad", aclrtBinaryLoad(input, &binary));
    aclrtDestroyBinary(input);
    Check(load);
    Check(Acl("aclrtBinaryGetFunction", aclrtBinaryGetFunction(binary,
        "KvafUbmemRawGetKernel", &function)));
  }
  double Copy(const std::vector<RawCopyRequest>& requests) {
    const auto& c = config;
    const auto request_bytes = requests.size() * sizeof(RawCopyRequest);
    Check(Acl("requests H2D", aclrtMemcpy(requests_device, request_bytes, requests.data(), request_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE)));
    Check(Acl("reset reports", aclrtMemsetAsync(reports_device, c.cores * sizeof(RawCopyReport),
        0xFF, c.cores * sizeof(RawCopyReport), stream)));
    RawCopyArgs args{Address(requests_device), Address(reports_device), c.rows, c.row_bytes, c.cores};
    aclrtLaunchKernelAttr attrs[4]{};
    attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE; attrs[0].value.schemMode = 1;
    attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE; attrs[1].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
    attrs[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US; attrs[2].value.timeoutUs.timeoutLow = 180000000U;
    attrs[3].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE; attrs[3].value.dynUBufSize = kMteBufferBytes;
    aclrtLaunchKernelCfg cfg{}; cfg.attrs = attrs; cfg.numAttrs = 4;
    Check(Acl("bandwidth kernel", aclrtLaunchKernelWithHostArgs(
        function, c.cores, stream, &cfg, &args, sizeof(args), nullptr, 0)));
    Check(Acl("bandwidth synchronize", aclrtSynchronizeStream(stream)));
    std::vector<RawCopyReport> reports(c.cores);
    Check(Acl("reports D2H", aclrtMemcpy(reports.data(), reports.size() * sizeof(RawCopyReport),
        reports_device, reports.size() * sizeof(RawCopyReport), ACL_MEMCPY_DEVICE_TO_HOST)));
    std::uint64_t cycles = 0;
    for (unsigned core = 0; core < c.cores; ++core) {
      Require(reports[core].completed == c.rows / c.cores + (core < c.rows % c.cores),
          "raw kernel did not complete all rows");
      Require(reports[core].cycles != UINT64_MAX, "raw kernel did not write cycle report");
      cycles = std::max(cycles, reports[core].cycles);
    }
    Require(cycles != 0, "kernel cycle counter returned zero");
    return static_cast<double>(cycles) / c.clock_mhz;
  }
  void Verify(const std::vector<std::uint64_t>& offsets) {
    const auto& c = config;
    std::vector<std::uint8_t> data(static_cast<std::size_t>(c.rows) * c.row_bytes);
    Check(Acl("payload D2H verification", aclrtMemcpy(data.data(), data.size(), output, data.size(),
        ACL_MEMCPY_DEVICE_TO_HOST)));
    for (std::size_t row = 0; row < offsets.size(); ++row) {
      for (std::uint32_t b = 0; b < c.row_bytes; ++b) {
        if (data[row * c.row_bytes + b] != TestPattern(offsets[row] + b, c.seed))
          throw std::runtime_error("raw GET payload mismatch at row=" + std::to_string(row) +
              " byte=" + std::to_string(b));
      }
    }
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::puts("Usage: kvaf_ubmem_bandwidth --peer CPU:PORT --kernel PATH\n"
        "  [--device 0] [--cores 2] [--rows 4096] [--token-bytes 656]\n"
        "  [--remote-blocks 4096] [--tokens-per-block 128] [--pattern random|contiguous]\n"
        "  [--warmup 3] [--iterations 100] [--clock-mhz 1000] [--seed 20260903]\n"
        "GET only: raw DDR source addresses -> contiguous HBM output. No lookup or Python.\n"
        "Random requests change each sample; contiguous samples advance through the DDR region.\n"
        "Timing: max per-core GetSystemCycle elapsed / clock-mhz, includes row dispatch and copy completion.\n"
        "Excludes setup, request generation/H2D, report D2H, host synchronization and payload validation.\n"
        "Bandwidth counts rows * token-bytes once; warmup/cache state can affect the measured rate.");
    return 0;
  }
  try {
    const auto c = Parse(argc, argv);
    Check(Acl("aclInit", aclInit(nullptr)));
    Check(Acl("aclrtSetDevice", aclrtSetDevice(c.device)));
    {
      Benchmark benchmark(c);
      benchmark.Initialize();
      const auto tokens = static_cast<std::uint64_t>(c.remote_blocks) * c.tokens_per_block;
      const auto payload = static_cast<std::uint64_t>(c.rows) * c.row_bytes;
      std::printf("UBMEM_BANDWIDTH_CONFIG transport=ubmem device=%u cores=%u rows=%u token_bytes=%u payload_bytes=%llu "
          "remote_bytes=%llu pattern=%s warmup=%u iterations=%u clock_mhz=%u timing=max_core_cycles\n",
          c.device, c.cores, c.rows, c.row_bytes, static_cast<unsigned long long>(payload),
          static_cast<unsigned long long>(tokens * c.row_bytes), c.pattern.c_str(), c.warmup, c.iterations, c.clock_mhz);
      std::mt19937_64 generator(c.seed);
      std::vector<RawCopyRequest> requests(c.rows);
      std::vector<std::uint64_t> offsets;
      auto prepare = [&](std::uint64_t iteration) {
        offsets = perf::MakeRequests(requests, benchmark.mapping->address(), Address(benchmark.output),
            tokens, c.row_bytes, c.pattern == "random", iteration, generator);
      };
      prepare(0);
      benchmark.Copy(requests);
      benchmark.Verify(offsets);
      std::puts("UBMEM_BANDWIDTH_PREFLIGHT_OK");
      for (std::uint64_t i = 0; i < c.warmup; ++i) { prepare(i + 1); benchmark.Copy(requests); }
      std::vector<double> samples;
      samples.reserve(c.iterations);
      for (std::uint64_t i = 0; i < c.iterations; ++i) {
        prepare(i + static_cast<std::uint64_t>(c.warmup) + 1);
        samples.push_back(benchmark.Copy(requests));
      }
      benchmark.Verify(offsets);
      Require(benchmark.Close(), "bandwidth cleanup failed");
      std::sort(samples.begin(), samples.end());
      const auto mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
      const auto p50 = perf::Percentile(samples, 0.5);
      std::printf("PERF_RESULT operation=ubmem_raw_get transport=ubmem cores=%u rows=%u token_bytes=%u "
          "payload_bytes=%llu mean_us=%.3f p50_us=%.3f p90_us=%.3f p99_us=%.3f "
          "mean_GBps=%.3f p50_GBps=%.3f clock_mhz=%u verified=1\n",
          c.cores, c.rows, c.row_bytes, static_cast<unsigned long long>(payload),
          mean, p50, perf::Percentile(samples, 0.9), perf::Percentile(samples, 0.99),
          perf::GBps(payload, mean), perf::GBps(payload, p50), c.clock_mhz);
    }
    Check(Acl("aclrtResetDevice", aclrtResetDevice(c.device)));
    Check(Acl("aclFinalize", aclFinalize()));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[KVAF_UBMEM] BANDWIDTH_FAILED: %s\n", error.what());
    return 1;
  }
}
