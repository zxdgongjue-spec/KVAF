#include "cpu_urma/cpu_urma_transport.h"
#include "cpu_urma_eid_perf_metrics.h"

#include <urma_api.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using kvaf::CpuUrmaTransport;
using kvaf::Status;
using Eid = std::array<std::uint8_t, kvaf::peer::kEidBytes>;
constexpr std::size_t kPageBytes = 4096U;

struct Options {
  std::string peer;
  std::uint32_t device_id{0};
  std::size_t count{8192};
  std::size_t token_bytes{4096};
  std::uint32_t remote_sgl_size{1};
  std::size_t warmup{5};
  std::size_t iterations{100};
  std::string output{"results/cpu_urma_eid_bandwidth.csv"};
};

void Usage(const char* program) {
  std::cerr << "Usage: " << program << " --peer HOST:PORT [--device-id N] "
      "[--count 8192] [--token-bytes 4096] [--remote-sgl-size 1] "
      "[--warmup 5] [--iterations 100] [--output results.csv]\n";
}

std::uint64_t Number(const std::string& value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("invalid unsigned integer: " + value);
  errno = 0;
  char* end = nullptr;
  const auto number = std::strtoull(value.c_str(), &end, 10);
  if (errno == ERANGE || *end != '\0') throw std::invalid_argument("integer out of range: " + value);
  return number;
}

Options Parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (i + 1 == argc) throw std::invalid_argument("missing value for " + flag);
    const std::string value = argv[++i];
    if (flag == "--peer") { options.peer = value; continue; }
    if (flag == "--output") { options.output = value; continue; }
    const auto number = Number(value);
    if (number > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("value too large");
    if (flag == "--device-id" && number <= UINT32_MAX) options.device_id = number;
    else if (flag == "--remote-sgl-size" && number <= UINT32_MAX) options.remote_sgl_size = number;
    else if (flag == "--count") options.count = number;
    else if (flag == "--token-bytes") options.token_bytes = number;
    else if (flag == "--warmup") options.warmup = number;
    else if (flag == "--iterations") options.iterations = number;
    else throw std::invalid_argument("unknown option or invalid value: " + flag);
  }
  if (options.peer.empty() || options.output.empty() || options.count == 0U ||
      options.token_bytes == 0U || options.token_bytes > UINT32_MAX ||
      options.remote_sgl_size == 0U || options.iterations == 0U ||
      options.count > (std::numeric_limits<std::size_t>::max() - kPageBytes) / options.token_bytes) {
    throw std::invalid_argument("invalid peer, output, payload size, SGL size or iterations");
  }
  return options;
}

std::string Hex(const Eid& eid) {
  constexpr char digits[] = "0123456789abcdef";
  std::string value;
  for (auto byte : eid) { value += digits[byte >> 4U]; value += digits[byte & 15U]; }
  return value;
}

struct CpuCandidate {
  std::string device;
  std::uint32_t index;
  Eid eid;
};

struct UrmaLifetime {
  bool owned{false};
  UrmaLifetime() {
    urma_init_attr_t attr{};
    const int rc = urma_init(&attr);
    if (rc != URMA_SUCCESS && rc != EEXIST && rc != -EEXIST)
      throw std::runtime_error("urma_init failed: " + std::to_string(rc));
    owned = rc == URMA_SUCCESS;
  }
  ~UrmaLifetime() { if (owned) (void)urma_uninit(); }
};

std::vector<CpuCandidate> CpuCandidates() {
  int count = 0;
  std::unique_ptr<urma_device_t*, decltype(&urma_free_device_list)> devices(
      urma_get_device_list(&count), urma_free_device_list);
  if (!devices) throw std::runtime_error("cannot enumerate URMA devices");
  std::vector<CpuCandidate> result;
  for (int i = 0; i < count; ++i) {
    auto* device = devices.get()[i];
    if (device == nullptr || device->type != URMA_TRANSPORT_UB) continue;
    std::uint32_t eid_count = 0;
    std::unique_ptr<urma_eid_info_t, decltype(&urma_free_eid_list)> eids(
        urma_get_eid_list(device, &eid_count), urma_free_eid_list);
    if (!eids) {
      std::cerr << "EID_PERF_ENUM_FAILED device=" << device->name << " errno=" << errno << '\n';
      continue;
    }
    for (std::uint32_t j = 0; j < eid_count; ++j) {
      const auto& entry = eids.get()[j];
      if (std::all_of(entry.eid.raw, entry.eid.raw + 16, [](auto b) { return b == 0; })) continue;
      CpuCandidate candidate{device->name, entry.eid_index, {}};
      std::copy_n(entry.eid.raw, candidate.eid.size(), candidate.eid.begin());
      result.push_back(candidate);
    }
  }
  std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
    return std::tie(a.device, a.index) < std::tie(b.device, b.index);
  });
  if (result.empty()) throw std::runtime_error("no non-zero CPU UB EIDs");
  return result;
}

struct HostBuffer {
  std::unique_ptr<std::uint8_t, decltype(&std::free)> data{nullptr, std::free};
  std::size_t bytes;
  explicit HostBuffer(std::size_t payload) : bytes((payload + kPageBytes - 1U) & ~(kPageBytes - 1U)) {
    void* pointer = nullptr;
    if (posix_memalign(&pointer, kPageBytes, bytes) != 0) throw std::bad_alloc();
    data.reset(static_cast<std::uint8_t*>(pointer));
    std::memset(data.get(), 0, bytes);
  }
};

struct Result {
  CpuCandidate cpu;
  std::uint32_t npu_index;
  Eid npu_eid;
  std::string segment_eid;
  std::string status;
  std::string error;
  bool cleanup_ok{true};
  std::size_t samples{0};
  std::uint64_t wr_min{UINT64_MAX}, wr_max{0};
  kvaf::eid_perf::Stats stats;
  double prepare_us{0}, post_us{0}, poll_us{0};
};

Result Measure(const Options& options, const CpuCandidate& cpu, std::uint32_t npu_index,
               const Eid& npu_eid, HostBuffer& input, HostBuffer& output,
               const std::vector<CpuUrmaTransport::HostSge>& sources,
               const std::vector<CpuUrmaTransport::HostSge>& destinations,
               const std::vector<std::uint64_t>& offsets) {
  Result result{};
  result.cpu = cpu;
  result.npu_index = npu_index;
  result.npu_eid = npu_eid;
  kvaf::peer::PeerLink link(options.peer);
  kvaf::peer::PeerRegion region{};
  CpuUrmaTransport transport;
  const auto phase = [](const char* name) {
    std::cerr << "EID_PERF_PHASE phase=" << name << std::endl;
  };
  const auto finish = [&](const std::string& stage, const Status& status) {
    result.status = status.ok() ? "ok" : stage;
    result.error = status.message();
    phase("shutdown");
    (void)transport.Shutdown();
    if (region.session_id != 0U) {
      phase("release_hbm");
      const auto released = link.ReleaseHbmRegion(region.session_id);
      if (!released.ok()) {
        result.cleanup_ok = false;
        result.status = "cleanup_failed";
        result.error += "; HBM release: " + released.message();
      } else {
        region = {};
      }
    }
    return result;
  };
  const std::size_t payload = options.count * options.token_bytes;
  try {
    phase("allocate_hbm");
    auto status = link.AllocateHbmRegion(payload, options.token_bytes, options.device_id, npu_index, &region);
    if (!status.ok()) return finish("allocate_failed", status);
    result.segment_eid = Hex(region.eid);
    kvaf::CpuUrmaTransportOptions config;
    config.peer_address = options.peer;
    config.ub_device = cpu.device;
    config.has_eid_index = true;
    config.eid_index = cpu.index;
    config.has_preferred_eid = true;
    config.preferred_eid = cpu.eid;
    config.region_bytes = payload;
    config.has_external_region = true;
    config.external_region = region;
    config.remote_sgl_size = options.remote_sgl_size;
    // This calls one fixed pair directly; no Runtime candidate fallback.
    phase("initialize");
    status = transport.Initialize(config);
    if (!status.ok()) return finish("connect_failed", status);
    phase("probe");
    status = transport.ValidateLink();
    if (!status.ok()) return finish("probe_failed", status);
    // Keep the input as the explicit region for the measured WRITE fast path.
    phase("register_output");
    status = transport.RegisterDirectHostMemory(output.data.get(), output.bytes);
    if (status.ok()) {
      phase("register_input");
      status = transport.RegisterDirectHostMemory(input.data.get(), input.bytes);
    }
    if (!status.ok()) return finish("register_failed", status);
    const auto write = [&] { return transport.WriteBatch(offsets.data(), sources.data(), sources.size()); };
    const auto verify = [&] {
      std::memset(output.data.get(), 0xEE, output.bytes);
      const auto read = transport.ReadBatch(offsets.data(), destinations.data(), destinations.size());
      if (!read.ok()) return read;
      return std::memcmp(input.data.get(), output.data.get(), payload) == 0
          ? Status::OK() : Status::Internal("payload read-back mismatch");
    };
    phase("preflight_write");
    status = write();
    if (status.ok()) {
      phase("preflight_read");
      status = verify();
    }
    if (!status.ok()) return finish("preflight_failed", status);
    phase("warmup");
    for (std::size_t i = 0; i < options.warmup; ++i) {
      status = write();
      if (!status.ok()) return finish("warmup_failed", status);
    }
    std::vector<std::uint64_t> samples;
    samples.reserve(options.iterations);
    long double prepare = 0, post = 0, poll = 0;
    phase("measure");
    for (std::size_t i = 0; i < options.iterations; ++i) {
      status = write();
      if (!status.ok()) return finish("transfer_failed", status);
      const auto timing = transport.last_batch_timing();
      if (timing.total_ns == 0U || timing.wr_count == 0U)
        return finish("timing_failed", Status::Internal("missing transport timing or WR count"));
      samples.push_back(timing.total_ns);
      result.samples = samples.size();
      result.wr_min = std::min(result.wr_min, timing.wr_count);
      result.wr_max = std::max(result.wr_max, timing.wr_count);
      prepare += timing.prepare_ns;
      post += timing.post_ns;
      poll += timing.poll_ns;
    }
    phase("final_read");
    status = verify();
    if (!status.ok()) return finish("verify_failed", status);
    result.stats = kvaf::eid_perf::Summarize(std::move(samples), payload);
    result.prepare_us = static_cast<double>(prepare / options.iterations / 1000);
    result.post_us = static_cast<double>(post / options.iterations / 1000);
    result.poll_us = static_cast<double>(poll / options.iterations / 1000);
    return finish("ok", Status::OK());
  } catch (const std::exception& error) {
    return finish("error", Status::Internal(error.what()));
  }
}

std::string Fixed(double value) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(6) << value;
  return stream.str();
}

void WriteRow(std::ostream& csv, const Options& options, const Result& result) {
  std::vector<std::string> fields{
      result.cpu.device, std::to_string(result.cpu.index), Hex(result.cpu.eid),
      std::to_string(options.device_id), std::to_string(result.npu_index), Hex(result.npu_eid),
      result.segment_eid, result.status, result.error, std::to_string(options.count),
      std::to_string(options.token_bytes), std::to_string(options.count * options.token_bytes),
      std::to_string(options.remote_sgl_size), std::to_string(options.warmup),
      std::to_string(options.iterations), std::to_string(result.samples)};
  if (result.status == "ok") {
    fields.insert(fields.end(), {std::to_string(result.wr_min), std::to_string(result.wr_max),
        Fixed(result.stats.mean_ns / 1000), Fixed(result.stats.p50_ns / 1000),
        Fixed(result.stats.p99_ns / 1000), Fixed(result.stats.mean_gbps),
        Fixed(result.stats.p50_gbps), Fixed(result.prepare_us), Fixed(result.post_us), Fixed(result.poll_us)});
  } else {
    fields.resize(fields.size() + 10U);  // Failure is not a zero-bandwidth measurement.
  }
  for (std::size_t i = 0; i < fields.size(); ++i) {
    if (i != 0U) csv << ',';
    csv << kvaf::eid_perf::CsvCell(fields[i]);
  }
  csv << '\n' << std::flush;
}

int Run(const Options& options) {
  UrmaLifetime library;
  const auto cpus = CpuCandidates();
  std::vector<Eid> npus;
  const auto discovered = kvaf::peer::PeerLink(options.peer).GetHbmEidCandidates(options.device_id, &npus);
  if (!discovered.ok()) throw std::runtime_error(discovered.message());
  const auto path = std::filesystem::absolute(options.output);
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::ofstream csv(path);
  csv.exceptions(std::ios::badbit | std::ios::failbit);
  csv << "cpu_device,cpu_eid_index,cpu_eid,npu_device_id,npu_eid_index,npu_eid,segment_eid,status,error,"
         "count,token_bytes,payload_bytes,remote_sgl_size,warmup,iterations,samples,wr_count_min,wr_count_max,"
         "mean_us,p50_us,p99_us,mean_GBps,p50_GBps,prepare_mean_us,post_mean_us,completion_wait_mean_us\n" << std::flush;
  const std::size_t payload = options.count * options.token_bytes;
  HostBuffer input(payload), output(payload);
  for (std::size_t i = 0; i < payload; ++i) input.data.get()[i] = static_cast<std::uint8_t>(i * 17U + 29U);
  std::vector<CpuUrmaTransport::HostSge> sources(options.count), destinations(options.count);
  std::vector<std::uint64_t> offsets(options.count);
  for (std::size_t i = 0; i < options.count; ++i) {
    offsets[i] = i * options.token_bytes;
    sources[i] = {input.data.get() + offsets[i], options.token_bytes};
    destinations[i] = {output.data.get() + offsets[i], options.token_bytes};
    for (std::size_t j = 0; j < std::min<std::size_t>(8U, options.token_bytes); ++j)
      input.data.get()[offsets[i] + j] = static_cast<std::uint8_t>(static_cast<std::uint64_t>(i) >> (8U * j));
  }
  std::cout << "EID_PERF_CONFIG cpu_eids=" << cpus.size() << " npu_eids=" << npus.size()
            << " payload_bytes=" << payload << " count=" << options.count
            << " token_bytes=" << options.token_bytes << " remote_sgl_size=" << options.remote_sgl_size
            << " scope=transport_total bytes_unit=decimal_GB csv=" << path << std::endl;
  std::vector<Result> results;
  bool aborted = false;
  for (const auto& cpu : cpus) {
    for (std::size_t npu = 0; npu < npus.size(); ++npu) {
      std::cout << "EID_PERF_ATTEMPT cpu=" << cpu.device << ':' << cpu.index
                << " cpu_eid=" << Hex(cpu.eid) << " npu_candidate=" << npu
                << " npu_eid=" << Hex(npus[npu]) << std::endl;
      auto result = Measure(options, cpu, static_cast<std::uint32_t>(npu), npus[npu],
                            input, output, sources, destinations, offsets);
      WriteRow(csv, options, result);
      std::cout << "EID_PERF_RESULT status=" << result.status;
      if (result.status == "ok") std::cout << " mean_GBps=" << Fixed(result.stats.mean_gbps)
                                           << " p50_GBps=" << Fixed(result.stats.p50_gbps);
      else std::cout << " error=" << result.error;
      std::cout << std::endl;
      aborted = !result.cleanup_ok;
      results.push_back(std::move(result));
      if (aborted) break;  // Preserve the CSV and avoid accumulating unreleased HBM.
    }
    if (aborted) break;
  }
  std::vector<const Result*> ranking;
  for (const auto& result : results) if (result.status == "ok") ranking.push_back(&result);
  std::stable_sort(ranking.begin(), ranking.end(), [](const auto* a, const auto* b) {
    return a->stats.mean_gbps > b->stats.mean_gbps;
  });
  for (std::size_t i = 0; i < ranking.size(); ++i) {
    const auto& result = *ranking[i];
    std::cout << "EID_PERF_RANK rank=" << i + 1U << " cpu=" << result.cpu.device << ':' << result.cpu.index
              << " npu_candidate=" << result.npu_index << " mean_GBps=" << Fixed(result.stats.mean_gbps)
              << " p50_GBps=" << Fixed(result.stats.p50_gbps) << std::endl;
  }
  std::cout << "EID_PERF_SUMMARY tested=" << results.size() << " passed=" << ranking.size()
            << " failed=" << results.size() - ranking.size() << " aborted=" << aborted
            << " csv=" << path << std::endl;
  return aborted || ranking.empty() ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    Usage(argv[0]);
    return 0;
  }
  try { return Run(Parse(argc, argv)); }
  catch (const std::exception& error) {
    std::cerr << "EID_PERF_ERROR " << error.what() << '\n';
    return 2;
  }
}
