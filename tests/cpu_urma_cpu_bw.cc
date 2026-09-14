#include "cpu_urma/cpu_urma_transport.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kPageBytes = 4096;
constexpr std::size_t kMaxBlocks = 4 * 8192;  // Production transport SQ capacity.

struct Options {
  kvaf::CpuUrmaTransportOptions transport;
  std::size_t blocks{8192};
  std::size_t block_bytes{4096};
  std::size_t warmup{5};
  std::size_t iterations{100};
};

void Usage(const char* program) {
  std::printf("Usage: %s --peer HOST:PORT [--blocks 8192] [--block-bytes 4096] "
              "[--warmup 5] [--iterations 100] [--ub-device NAME] "
              "[--eid-index N] [--peer-candidate-id N] [--doorbell-wqes N]\n",
              program);
}

std::size_t Number(const std::string& value, std::uint64_t limit) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("invalid unsigned integer: " + value);
  errno = 0;
  char* end = nullptr;
  const auto number = std::strtoull(value.c_str(), &end, 10);
  if (errno == ERANGE || *end != '\0' || number > limit ||
      number > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("integer out of range: " + value);
  return static_cast<std::size_t>(number);
}

Options Parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (i + 1 == argc) throw std::invalid_argument("missing value for " + flag);
    const std::string value = argv[++i];
    if (flag == "--peer") options.transport.peer_address = value;
    else if (flag == "--ub-device") options.transport.ub_device = value;
    else if (flag == "--blocks") options.blocks = Number(value, kMaxBlocks);
    else if (flag == "--block-bytes") options.block_bytes = Number(value, UINT32_MAX);
    else if (flag == "--warmup") options.warmup = Number(value, SIZE_MAX);
    else if (flag == "--iterations") options.iterations = Number(value, SIZE_MAX);
    else if (flag == "--eid-index") {
      options.transport.eid_index = static_cast<std::uint32_t>(Number(value, UINT32_MAX));
      options.transport.has_eid_index = true;
    } else if (flag == "--peer-candidate-id") {
      options.transport.peer_candidate_id = static_cast<std::uint32_t>(Number(value, UINT32_MAX));
    } else if (flag == "--doorbell-wqes") {
      options.transport.wqes_per_doorbell = static_cast<std::uint32_t>(Number(value, 8192));
    } else throw std::invalid_argument("unknown option: " + flag);
  }
  if (options.transport.peer_address.empty() || options.blocks == 0 ||
      options.block_bytes == 0 || options.iterations == 0 ||
      options.blocks > (SIZE_MAX - (kPageBytes - 1)) / options.block_bytes)
    throw std::invalid_argument("peer, blocks, block-bytes and iterations must be valid and non-zero");
  options.transport.region_bytes = options.blocks * options.block_bytes;
  return options;
}

void Check(const kvaf::Status& status) {
  if (!status.ok()) throw std::runtime_error(status.message());
}

std::uint8_t Pattern(std::size_t offset) {
  return static_cast<std::uint8_t>((offset * 131U + offset / 251U + 17U) % 256U);
}

void Verify(const std::uint8_t* data, std::size_t bytes) {
  for (std::size_t i = 0; i < bytes; ++i) {
    if (data[i] != Pattern(i))
      throw std::runtime_error("read-back mismatch at byte " + std::to_string(i));
  }
}

template <typename Transfer>
void Measure(const char* operation, const Options& options, Transfer transfer) {
  for (std::size_t i = 0; i < options.warmup; ++i) Check(transfer());
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < options.iterations; ++i) Check(transfer());
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  const double gbps = static_cast<double>(options.transport.region_bytes) *
                      static_cast<double>(options.iterations) / seconds / 1e9;
  std::printf("%s bytes_per_batch=%llu iterations=%zu seconds=%.6f "
              "GB/s=%.3f Gbit/s=%.3f us_per_batch=%.3f\n",
              operation, static_cast<unsigned long long>(options.transport.region_bytes),
              options.iterations, seconds, gbps, gbps * 8,
              seconds * 1e6 / static_cast<double>(options.iterations));
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    Usage(argv[0]);
    return 0;
  }
  try {
    const Options options = Parse(argc, argv);
    const std::size_t bytes = options.transport.region_bytes;
    const std::size_t aligned_bytes = (bytes + kPageBytes - 1) & ~(kPageBytes - 1);
    void* pointer = nullptr;
    if (posix_memalign(&pointer, kPageBytes, aligned_bytes) != 0) throw std::bad_alloc();
    // The transport must unregister memory before the allocation is freed.
    std::unique_ptr<std::uint8_t, decltype(&std::free)> buffer(
        static_cast<std::uint8_t*>(pointer), std::free);
    std::memset(buffer.get(), 0, aligned_bytes);
    for (std::size_t i = 0; i < bytes; ++i) buffer.get()[i] = Pattern(i);
    std::vector<kvaf::CpuUrmaTransport::HostSge> sges(options.blocks);
    for (std::size_t i = 0; i < options.blocks; ++i)
      sges[i] = {buffer.get() + i * options.block_bytes, options.block_bytes};

    kvaf::CpuUrmaTransport transport;
    Check(transport.Initialize(options.transport));
    Check(transport.ValidateLink());
    Check(transport.RegisterDirectHostMemory(buffer.get(), aligned_bytes));
    const auto write = [&] { return transport.WriteSglBatch(0, sges.data(), sges.size()); };
    const auto read = [&] { return transport.ReadSglBatch(0, sges.data(), sges.size()); };
    const auto read_back = [&] {
      std::memset(buffer.get(), 0, bytes);
      Check(read());
      Verify(buffer.get(), bytes);
    };
    Check(write());
    read_back();
    Measure("WRITE", options, write);
    read_back();
    std::memset(buffer.get(), 0, bytes);
    Measure("READ", options, read);
    Verify(buffer.get(), bytes);
    Check(transport.Shutdown());
    std::puts("VERIFY ok");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CPU URMA bandwidth test failed: %s\n", error.what());
    return 1;
  }
}
