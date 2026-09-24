#include "cpu_urma/sgl_batch_builder.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using HostSge = kvaf::CpuUrmaTransport::HostSge;
using WriteGroup = kvaf::cpu_urma_detail::WriteGatherGroup;

constexpr std::size_t kMaxPayloadBytes = 256U * 1024U * 1024U;
constexpr std::size_t kRemoteGuardBytes = 64U;
constexpr double kAssumedAicpuBytesPerNs = 25.0;
constexpr std::uint8_t kGuardValue = 0xcdU;

struct Options {
  std::size_t rows{1024U};
  std::size_t row_bytes{4096U};
  std::size_t source_gap_bytes{64U};
  std::size_t sgl_size{8U};
  std::size_t remote_offset{4096U};
  std::size_t iterations{3U};
  std::uint64_t wr_overhead_ns{0U};
};

struct SimulatedSge {
  const std::uint8_t* source{nullptr};
  std::uint64_t remote_offset{0U};
  std::size_t bytes{0U};
};

void Usage(const char* program) {
  std::cout << "Usage: " << program
            << " [--rows N] [--row-bytes N] [--source-gap-bytes N]"
               " [--sgl-size N] [--remote-offset N] [--iterations N]"
               " [--wr-overhead-ns N]\n";
}

std::uint64_t ParseUnsigned(std::string_view text) {
  std::uint64_t value = 0U;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw std::invalid_argument("invalid unsigned integer: " + std::string(text));
  }
  return value;
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  if (argc % 2 == 0) throw std::invalid_argument("every option requires a value");
  for (int index = 1; index < argc; index += 2) {
    const std::string_view flag(argv[index]);
    const std::uint64_t value = ParseUnsigned(argv[index + 1]);
    if (flag == "--wr-overhead-ns") {
      options.wr_overhead_ns = value;
      continue;
    }
    if (value > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("option exceeds size_t: " + std::string(flag));
    }
    const auto size = static_cast<std::size_t>(value);
    if (flag == "--rows") options.rows = size;
    else if (flag == "--row-bytes") options.row_bytes = size;
    else if (flag == "--source-gap-bytes") options.source_gap_bytes = size;
    else if (flag == "--sgl-size") options.sgl_size = size;
    else if (flag == "--remote-offset") options.remote_offset = size;
    else if (flag == "--iterations") options.iterations = size;
    else throw std::invalid_argument("unknown option: " + std::string(flag));
  }
  if (options.rows == 0U || options.row_bytes == 0U || options.sgl_size == 0U ||
      options.iterations == 0U || options.iterations > 1000U ||
      options.row_bytes > kMaxPayloadBytes / options.rows ||
      options.source_gap_bytes > std::numeric_limits<std::size_t>::max() - options.row_bytes) {
    throw std::invalid_argument("invalid dimensions or payload exceeds 256 MiB");
  }
  const auto payload_bytes = options.rows * options.row_bytes;
  const auto stride = options.row_bytes + options.source_gap_bytes;
  if (options.rows > (2U * kMaxPayloadBytes) / stride ||
      options.remote_offset > std::numeric_limits<std::size_t>::max() -
          payload_bytes - kRemoteGuardBytes) {
    throw std::invalid_argument("source or remote buffer is too large");
  }
  return options;
}

void FillScatteredSource(std::vector<std::uint8_t>* source, const Options& options,
                         std::size_t iteration) {
  const std::size_t stride = options.row_bytes + options.source_gap_bytes;
  for (std::size_t row = 0U; row < options.rows; ++row) {
    auto* data = source->data() + (options.rows - 1U - row) * stride;
    for (std::size_t byte = 0U; byte < options.row_bytes; ++byte) {
      data[byte] = static_cast<std::uint8_t>(
          (row * 131U + byte * 17U + iteration * 29U) & 0xffU);
    }
  }
}

void SimulateSend(const Options& options, const std::vector<SimulatedSge>& sges,
                  const std::vector<WriteGroup>& writes,
                  std::vector<std::uint8_t>* remote) {
  std::fill(remote->begin(), remote->end(), kGuardValue);
  for (const WriteGroup& write : writes) {
    std::uint64_t destination = sges[write.first].remote_offset;
    for (std::size_t row = write.first; row < write.first + write.count; ++row) {
      const SimulatedSge& sge = sges[row];
      if (sge.remote_offset != destination) {
        throw std::runtime_error("one SGL write contains a remote gap");
      }
      std::memcpy(remote->data() + destination, sge.source, sge.bytes);
      destination += sge.bytes;
    }
    if (destination - sges[write.first].remote_offset != write.bytes ||
        write.count > options.sgl_size) {
      throw std::runtime_error("SGL write shape mismatch");
    }
  }
}

void VerifyRemote(const Options& options, const std::vector<std::uint8_t>& source,
                  const std::vector<std::uint8_t>& remote) {
  const std::size_t stride = options.row_bytes + options.source_gap_bytes;
  const std::size_t payload_bytes = options.rows * options.row_bytes;
  for (std::size_t row = 0U; row < options.rows; ++row) {
    const auto* expected = source.data() + (options.rows - 1U - row) * stride;
    const auto* actual = remote.data() + options.remote_offset + row * options.row_bytes;
    if (std::memcmp(expected, actual, options.row_bytes) != 0) {
      throw std::runtime_error("remote bytes do not match the scattered CPU source");
    }
  }
  if (!std::all_of(remote.begin(), remote.begin() + options.remote_offset,
                   [](std::uint8_t byte) { return byte == kGuardValue; }) ||
      !std::all_of(remote.begin() + options.remote_offset + payload_bytes,
                   remote.end(),
                   [](std::uint8_t byte) { return byte == kGuardValue; })) {
    throw std::runtime_error("remote guard bytes were overwritten");
  }
}

void Run(const Options& options) {
  const std::size_t payload_bytes = options.rows * options.row_bytes;
  const std::size_t stride = options.row_bytes + options.source_gap_bytes;
  std::vector<std::uint8_t> source(options.rows * stride);
  std::vector<std::uint8_t> staging(payload_bytes);
  std::vector<std::uint8_t> remote(options.remote_offset + payload_bytes +
                                   kRemoteGuardBytes);
  std::vector<HostSge> local_rows(options.rows);
  std::vector<std::uint64_t> remote_offsets(options.rows);
  for (std::size_t row = 0U; row < options.rows; ++row) {
    local_rows[row] = {staging.data() + row * options.row_bytes, options.row_bytes};
    remote_offsets[row] = options.remote_offset + row * options.row_bytes;
  }

  std::uint64_t gather_ns = 0U;
  std::uint64_t plan_ns = 0U;
  std::size_t wr_count = 0U;
  for (std::size_t iteration = 0U; iteration < options.iterations; ++iteration) {
    FillScatteredSource(&source, options, iteration);
    const auto gather_start = Clock::now();
    for (std::size_t row = 0U; row < options.rows; ++row) {
      std::memcpy(staging.data() + row * options.row_bytes,
                  source.data() + (options.rows - 1U - row) * stride,
                  options.row_bytes);
    }
    const auto gather_end = Clock::now();

    std::vector<SimulatedSge> sges;
    std::vector<WriteGroup> writes;
    sges.reserve(options.rows);
    writes.reserve(1U + (options.rows - 1U) / options.sgl_size);
    const auto status = kvaf::cpu_urma_detail::BuildWriteGatherShard(
        local_rows.data(), remote_offsets.data(), options.rows, 0U,
        options.rows, options.sgl_size, remote.size(),
        std::numeric_limits<std::uint32_t>::max(),
        [&](std::size_t, std::size_t row, const HostSge& buffer) {
          sges.push_back({static_cast<const std::uint8_t*>(buffer.data),
                          remote_offsets[row], buffer.bytes});
        },
        [&](std::size_t, const WriteGroup& write) { writes.push_back(write); },
        &wr_count);
    const auto plan_end = Clock::now();
    if (!status.ok() || sges.size() != options.rows || writes.size() != wr_count ||
        wr_count != 1U + (options.rows - 1U) / options.sgl_size) {
      throw std::runtime_error("SGL plan is invalid: " + status.message());
    }
    SimulateSend(options, sges, writes, &remote);
    VerifyRemote(options, source, remote);
    gather_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        gather_end - gather_start).count();
    plan_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        plan_end - gather_end).count();
  }

  const double total_bytes = static_cast<double>(payload_bytes) * options.iterations;
  const double total_wrs = static_cast<double>(wr_count) * options.iterations;
  const double wire_ns = total_bytes / kAssumedAicpuBytesPerNs;
  const double wr_ns = total_wrs * static_cast<double>(options.wr_overhead_ns);
  const double send_ns = wire_ns + wr_ns;
  const double pipeline_ns = static_cast<double>(gather_ns + plan_ns) + send_ns;
  std::cout << "AICPU_SGL_SIM_PASS rows=" << options.rows
            << " row_bytes=" << options.row_bytes
            << " sgl_size=" << options.sgl_size
            << " wr_count_per_iteration=" << wr_count
            << " iterations=" << options.iterations
            << " bytes_per_iteration=" << payload_bytes
            << " remote_offset=" << options.remote_offset << '\n';
  std::cout << std::fixed << std::setprecision(2)
            << "CPU_PREP_MEASURED gather_ns=" << gather_ns
            << " plan_ns=" << plan_ns << '\n'
            << "AICPU_SEND_MODEL assumed_GBps=25.00 wire_ns=" << wire_ns
            << " wr_overhead_ns=" << options.wr_overhead_ns
            << " total_wr_overhead_ns=" << wr_ns
            << " send_ns=" << send_ns << '\n'
            << "PIPELINE_ESTIMATE total_ns=" << pipeline_ns
            << " effective_GBps=" << total_bytes / pipeline_ns << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    Usage(argv[0]);
    return 0;
  }
  try {
    Run(ParseOptions(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "AICPU_SGL_SIM_ERROR " << error.what() << '\n';
    Usage(argv[0]);
    return 1;
  }
  return 0;
}
