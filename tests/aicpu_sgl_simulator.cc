#include "cpu_urma/sgl_batch_builder.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using HostSge = kvaf::CpuUrmaTransport::HostSge;
using WriteGroup = kvaf::cpu_urma_detail::WriteGatherGroup;

constexpr std::size_t kMaxPayloadBytes = 256U * 1024U * 1024U;
constexpr std::size_t kMinGatherBytesPerWorker = 256U * 1024U;
constexpr std::size_t kMaxGatherThreads = 64U;
constexpr std::size_t kRemoteGuardBytes = 64U;
constexpr std::size_t kGiB = 1024U * 1024U * 1024U;
constexpr std::size_t kSourceSlotAlignment = 64U;
constexpr std::size_t kDefaultSourceVariants = 16U;
constexpr double kAssumedAicpuBytesPerNs = 25.0;
constexpr std::uint8_t kGuardValue = 0xcdU;

enum class GatherMode { kActive, kPassive, kOpenMp };

std::size_t DefaultGatherThreads() {
  const auto detected = std::thread::hardware_concurrency();
  return detected == 0U ? 1U : std::min<std::size_t>(detected, kMaxGatherThreads);
}

struct Options {
  std::size_t rows{1024U};
  std::size_t row_bytes{4096U};
  std::size_t source_gap_bytes{64U};
  std::size_t source_space_gib{0U};
  std::size_t source_variants{kDefaultSourceVariants};
  std::size_t sgl_size{8U};
  std::size_t remote_offset{4096U};
  std::size_t iterations{3U};
  std::size_t gather_threads{DefaultGatherThreads()};
  GatherMode gather_mode{GatherMode::kPassive};
  bool log_iterations{false};
  std::uint64_t wr_overhead_ns{0U};
};

struct IterationTiming {
  std::uint64_t gather_ns{0U};
  std::uint64_t plan_ns{0U};
  std::size_t source_first_offset{0U};
  std::size_t source_last_offset{0U};
};

struct SimulatedSge {
  const std::uint8_t* source{nullptr};
  std::uint64_t remote_offset{0U};
  std::size_t bytes{0U};
};

void FillSourceRow(std::uint8_t* data, std::size_t row, std::size_t variant,
                   std::size_t row_bytes) {
  for (std::size_t byte = 0U; byte < row_bytes; ++byte) {
    data[byte] = static_cast<std::uint8_t>(
        (row * 131U + byte * 17U + variant * 29U) & 0xffU);
  }
  const std::uint64_t marker =
      (static_cast<std::uint64_t>(row) << 32U) | variant;
  std::memcpy(data, &marker, std::min(row_bytes, sizeof(marker)));
}

class SourceSpace {
 public:
  explicit SourceSpace(const Options& options)
      : rows_(options.rows), row_bytes_(options.row_bytes),
        variants_(options.source_variants) {
    if (options.source_space_gib == 0U) {
      band_stride_bytes_ = options.row_bytes + options.source_gap_bytes;
      compact_.resize(rows_ * band_stride_bytes_);
      virtual_bytes_ = compact_.size();
      return;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) throw std::runtime_error("cannot determine page size");
    page_bytes_ = static_cast<std::size_t>(page_size);
    slot_bytes_ = ((row_bytes_ + kSourceSlotAlignment - 1U) /
                   kSourceSlotAlignment) * kSourceSlotAlignment;
    virtual_bytes_ = options.source_space_gib * kGiB;
    band_stride_bytes_ = (virtual_bytes_ / rows_ / page_bytes_) * page_bytes_;
    if (band_stride_bytes_ < slot_bytes_ ||
        variants_ > band_stride_bytes_ / slot_bytes_) {
      throw std::invalid_argument("source variants do not fit in the address space");
    }

    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#if defined(MAP_NORESERVE)
    flags |= MAP_NORESERVE;
#endif
    void* region = mmap(nullptr, virtual_bytes_, PROT_READ | PROT_WRITE,
                        flags, -1, 0);
    if (region == MAP_FAILED) {
      throw std::runtime_error("cannot map source address space: " +
                               std::string(std::strerror(errno)));
    }
    mapped_ = static_cast<std::uint8_t*>(region);
    for (std::size_t row = 0U; row < rows_; ++row) {
      for (std::size_t variant = 0U; variant < variants_; ++variant) {
        FillSourceRow(WideRow(row, variant), row, variant, row_bytes_);
      }
    }
  }

  ~SourceSpace() {
    if (mapped_ != nullptr) munmap(mapped_, virtual_bytes_);
  }

  SourceSpace(const SourceSpace&) = delete;
  SourceSpace& operator=(const SourceSpace&) = delete;

  void Select(std::size_t iteration,
              std::vector<const std::uint8_t*>* source_rows) {
    for (std::size_t row = 0U; row < rows_; ++row) {
      if (mapped_ != nullptr) {
        const std::size_t variant =
            (iteration + row * 7U) % variants_;
        (*source_rows)[row] = WideRow(row, variant);
      } else {
        auto* data = compact_.data() + (rows_ - 1U - row) * band_stride_bytes_;
        FillSourceRow(data, row, iteration, row_bytes_);
        (*source_rows)[row] = data;
      }
    }
  }

  bool sparse() const { return mapped_ != nullptr; }
  std::size_t virtual_bytes() const { return virtual_bytes_; }
  std::size_t variants() const { return sparse() ? variants_ : 1U; }
  std::size_t band_stride_bytes() const { return band_stride_bytes_; }
  std::size_t slot_stride_bytes() const { return slot_bytes_; }
  std::size_t pool_payload_bytes() const {
    return rows_ * variants() * row_bytes_;
  }
  std::size_t pool_touched_page_bytes() const {
    if (!sparse()) return compact_.size();
    const std::size_t bytes_per_band = variants_ * slot_bytes_;
    return rows_ * ((bytes_per_band + page_bytes_ - 1U) / page_bytes_) *
           page_bytes_;
  }
  std::size_t access_span_bytes() const {
    return (rows_ - 1U) * band_stride_bytes_ +
           (variants() - 1U) * (sparse() ? slot_bytes_ : 0U) + row_bytes_;
  }
  std::size_t Offset(const std::uint8_t* address) const {
    const auto* base = sparse() ? mapped_ : compact_.data();
    return reinterpret_cast<std::uintptr_t>(address) -
           reinterpret_cast<std::uintptr_t>(base);
  }

 private:
  std::uint8_t* WideRow(std::size_t row, std::size_t variant) const {
    return mapped_ + (rows_ - 1U - row) * band_stride_bytes_ +
           variant * slot_bytes_;
  }

  std::size_t rows_;
  std::size_t row_bytes_;
  std::size_t variants_;
  std::size_t virtual_bytes_{0U};
  std::size_t band_stride_bytes_{0U};
  std::size_t slot_bytes_{0U};
  std::size_t page_bytes_{0U};
  std::vector<std::uint8_t> compact_;
  std::uint8_t* mapped_{nullptr};
};

class GatherTeam {
 public:
  GatherTeam(const std::vector<const std::uint8_t*>& source_rows,
             std::vector<std::uint8_t>* staging, const Options& options,
             std::size_t worker_count)
      : source_rows_(source_rows.data()), staging_(staging->data()),
        rows_(options.rows), row_bytes_(options.row_bytes),
        worker_count_(worker_count), actual_count_(worker_count),
        mode_(options.gather_mode) {
#if defined(_OPENMP)
    if (mode_ == GatherMode::kOpenMp) omp_set_dynamic(0);
#endif
    if (mode_ == GatherMode::kOpenMp) return;
    try {
      for (std::size_t worker = 1U; worker < worker_count_; ++worker) {
        workers_.emplace_back([this, worker] {
          if (mode_ == GatherMode::kActive) {
            ActiveWorker(worker);
          } else {
            PassiveWorker(worker);
          }
        });
      }
    } catch (...) {
      Stop();
      throw;
    }
  }

  ~GatherTeam() { Stop(); }

  GatherTeam(const GatherTeam&) = delete;
  GatherTeam& operator=(const GatherTeam&) = delete;

  std::size_t actual_count() const { return actual_count_; }

  void Run() {
    if (worker_count_ == 1U) {
      CopyShard(0U, 1U);
      return;
    }
    if (mode_ == GatherMode::kOpenMp) {
#if defined(_OPENMP)
#pragma omp parallel num_threads(static_cast<int>(worker_count_))
      {
        const auto team_size = static_cast<std::size_t>(omp_get_num_threads());
#pragma omp master
        { actual_count_ = team_size; }
        CopyShard(static_cast<std::size_t>(omp_get_thread_num()), team_size);
      }
#endif
      return;
    }
    if (mode_ == GatherMode::kActive) {
      active_completed_.store(0U, std::memory_order_relaxed);
      active_generation_.fetch_add(1U, std::memory_order_release);
      CopyShard(0U, worker_count_);
      while (active_completed_.load(std::memory_order_acquire) !=
             worker_count_ - 1U) {
        std::this_thread::yield();
      }
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    completed_ = 0U;
    ++generation_;
    lock.unlock();
    ready_.notify_all();
    CopyShard(0U, worker_count_);
    lock.lock();
    ++completed_;
    done_.wait(lock, [this] { return completed_ == worker_count_; });
  }

 private:
  void CopyShard(std::size_t worker, std::size_t team_size) const {
    const std::size_t base = rows_ / team_size;
    const std::size_t extra = rows_ % team_size;
    const std::size_t first = worker * base + std::min(worker, extra);
    const std::size_t end = first + base + (worker < extra ? 1U : 0U);
    for (std::size_t row = first; row < end; ++row) {
      std::memcpy(staging_ + row * row_bytes_,
                  source_rows_[row], row_bytes_);
    }
  }

  void ActiveWorker(std::size_t worker) {
    std::size_t observed_generation = 0U;
    while (!active_stopping_.load(std::memory_order_acquire)) {
      const auto generation =
          active_generation_.load(std::memory_order_acquire);
      if (generation == observed_generation) {
        std::this_thread::yield();
        continue;
      }
      observed_generation = generation;
      CopyShard(worker, worker_count_);
      active_completed_.fetch_add(1U, std::memory_order_release);
    }
  }

  void PassiveWorker(std::size_t worker) {
    std::size_t observed_generation = 0U;
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [this, &observed_generation] {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_) return;
      observed_generation = generation_;
      lock.unlock();
      CopyShard(worker, worker_count_);
      lock.lock();
      ++completed_;
      if (completed_ == worker_count_) done_.notify_one();
    }
  }

  void Stop() {
    if (mode_ == GatherMode::kOpenMp) return;
    if (mode_ == GatherMode::kActive) {
      active_stopping_.store(true, std::memory_order_release);
      for (std::thread& worker : workers_) {
        if (worker.joinable()) worker.join();
      }
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
  }

  const std::uint8_t* const* source_rows_;
  std::uint8_t* staging_;
  std::size_t rows_;
  std::size_t row_bytes_;
  std::size_t worker_count_;
  std::size_t actual_count_;
  GatherMode mode_;
  std::vector<std::thread> workers_;
  std::atomic<std::size_t> active_generation_{0U};
  std::atomic<std::size_t> active_completed_{0U};
  std::atomic<bool> active_stopping_{false};
  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable done_;
  std::size_t generation_{0U};
  std::size_t completed_{0U};
  bool stopping_{false};
};

void Usage(const char* program) {
  std::cout << "Usage: " << program
            << " [--rows N] [--row-bytes N] [--source-gap-bytes N]"
               " [--source-space-gib N] [--source-variants N]"
               " [--sgl-size N] [--remote-offset N] [--iterations N]"
               " [--gather-threads N]"
               " [--gather-mode active|passive|omp]"
               " [--log-iterations 0|1]"
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
    if (flag == "--gather-mode") {
      const std::string_view value(argv[index + 1]);
      if (value == "active") options.gather_mode = GatherMode::kActive;
      else if (value == "passive") options.gather_mode = GatherMode::kPassive;
      else if (value == "omp") {
#if defined(_OPENMP)
        options.gather_mode = GatherMode::kOpenMp;
#else
        throw std::invalid_argument("OpenMP gather mode is not available in this build");
#endif
      } else throw std::invalid_argument("invalid gather mode");
      continue;
    }
    const std::uint64_t value = ParseUnsigned(argv[index + 1]);
    if (flag == "--wr-overhead-ns") {
      options.wr_overhead_ns = value;
      continue;
    }
    if (flag == "--log-iterations") {
      if (value > 1U) throw std::invalid_argument("invalid log-iterations value");
      options.log_iterations = value == 1U;
      continue;
    }
    if (value > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("option exceeds size_t: " + std::string(flag));
    }
    const auto size = static_cast<std::size_t>(value);
    if (flag == "--rows") options.rows = size;
    else if (flag == "--row-bytes") options.row_bytes = size;
    else if (flag == "--source-gap-bytes") options.source_gap_bytes = size;
    else if (flag == "--source-space-gib") options.source_space_gib = size;
    else if (flag == "--source-variants") options.source_variants = size;
    else if (flag == "--sgl-size") options.sgl_size = size;
    else if (flag == "--remote-offset") options.remote_offset = size;
    else if (flag == "--iterations") options.iterations = size;
    else if (flag == "--gather-threads") options.gather_threads = size;
    else throw std::invalid_argument("unknown option: " + std::string(flag));
  }
  if (options.rows == 0U || options.row_bytes == 0U || options.sgl_size == 0U ||
      options.iterations == 0U || options.iterations > 1000U ||
      options.source_variants == 0U ||
      options.gather_threads == 0U || options.gather_threads > kMaxGatherThreads ||
      options.row_bytes > kMaxPayloadBytes / options.rows ||
      options.source_gap_bytes > std::numeric_limits<std::size_t>::max() - options.row_bytes) {
    throw std::invalid_argument("invalid dimensions or payload exceeds 256 MiB");
  }
  const auto payload_bytes = options.rows * options.row_bytes;
  const auto stride = options.row_bytes + options.source_gap_bytes;
  if (options.remote_offset > std::numeric_limits<std::size_t>::max() -
          payload_bytes - kRemoteGuardBytes) {
    throw std::invalid_argument("source or remote buffer is too large");
  }
  if (options.source_space_gib == 0U) {
    if (options.rows > (2U * kMaxPayloadBytes) / stride ||
        options.source_variants != kDefaultSourceVariants) {
      throw std::invalid_argument("invalid compact source layout");
    }
  } else if (options.source_space_gib >
                 std::numeric_limits<std::size_t>::max() / kGiB ||
             options.source_gap_bytes != 64U) {
    throw std::invalid_argument("invalid sparse source layout");
  }
  return options;
}

std::size_t EffectiveGatherThreads(const Options& options) {
  const std::size_t payload_bytes = options.rows * options.row_bytes;
  const std::size_t useful_workers = std::max<std::size_t>(
      1U, payload_bytes / kMinGatherBytesPerWorker);
  return std::min({options.gather_threads, options.rows, useful_workers});
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

void VerifyRemote(const Options& options,
                  const std::vector<const std::uint8_t*>& source_rows,
                  const std::vector<std::uint8_t>& remote) {
  const std::size_t payload_bytes = options.rows * options.row_bytes;
  for (std::size_t row = 0U; row < options.rows; ++row) {
    const auto* expected = source_rows[row];
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
  SourceSpace source(options);
  std::vector<const std::uint8_t*> source_rows(options.rows);
  source.Select(0U, &source_rows);
  std::vector<std::uint8_t> staging(payload_bytes);
  std::vector<std::uint8_t> remote(options.remote_offset + payload_bytes +
                                   kRemoteGuardBytes);
  std::vector<HostSge> local_rows(options.rows);
  std::vector<std::uint64_t> remote_offsets(options.rows);
  for (std::size_t row = 0U; row < options.rows; ++row) {
    local_rows[row] = {staging.data() + row * options.row_bytes, options.row_bytes};
    remote_offsets[row] = options.remote_offset + row * options.row_bytes;
  }
  const std::size_t gather_threads = EffectiveGatherThreads(options);
  GatherTeam gather_team(source_rows, &staging, options, gather_threads);
  gather_team.Run();

  std::uint64_t gather_ns = 0U;
  std::uint64_t plan_ns = 0U;
  std::uint64_t gather_min_ns = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t gather_max_ns = 0U;
  std::uint64_t plan_min_ns = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t plan_max_ns = 0U;
  std::vector<IterationTiming> iteration_timings;
  if (options.log_iterations) iteration_timings.reserve(options.iterations);
  std::size_t wr_count = 0U;
  for (std::size_t iteration = 0U; iteration < options.iterations; ++iteration) {
    source.Select(iteration, &source_rows);
    const auto gather_start = Clock::now();
    gather_team.Run();
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
    VerifyRemote(options, source_rows, remote);
    const auto iteration_gather_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            gather_end - gather_start).count());
    const auto iteration_plan_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            plan_end - gather_end).count());
    gather_ns += iteration_gather_ns;
    plan_ns += iteration_plan_ns;
    gather_min_ns = std::min(gather_min_ns, iteration_gather_ns);
    gather_max_ns = std::max(gather_max_ns, iteration_gather_ns);
    plan_min_ns = std::min(plan_min_ns, iteration_plan_ns);
    plan_max_ns = std::max(plan_max_ns, iteration_plan_ns);
    if (options.log_iterations) {
      iteration_timings.push_back({iteration_gather_ns, iteration_plan_ns,
                                   source.Offset(source_rows.front()),
                                   source.Offset(source_rows.back())});
    }
  }

  const double total_bytes = static_cast<double>(payload_bytes) * options.iterations;
  const double total_wrs = static_cast<double>(wr_count) * options.iterations;
  const double wire_ns = total_bytes / kAssumedAicpuBytesPerNs;
  const double wr_ns = total_wrs * static_cast<double>(options.wr_overhead_ns);
  const double send_ns = wire_ns + wr_ns;
  const double pipeline_ns = static_cast<double>(gather_ns + plan_ns) + send_ns;
  const double send_ns_per_iteration = send_ns / options.iterations;
  std::cout << "AICPU_SGL_SIM_PASS rows=" << options.rows
            << " row_bytes=" << options.row_bytes
            << " sgl_size=" << options.sgl_size
            << " wr_count_per_iteration=" << wr_count
            << " iterations=" << options.iterations
            << " bytes_per_iteration=" << payload_bytes
            << " remote_offset=" << options.remote_offset
            << " gather_threads_requested=" << options.gather_threads
            << " gather_threads_effective=" << gather_team.actual_count()
            << " gather_mode="
            << (options.gather_mode == GatherMode::kActive ? "active" :
                options.gather_mode == GatherMode::kOpenMp ? "omp" : "passive")
            << '\n';
  std::cout << "SOURCE_LAYOUT mode=" << (source.sparse() ? "sparse" : "compact")
            << " virtual_bytes=" << source.virtual_bytes()
            << " access_span_bytes=" << source.access_span_bytes()
            << " pool_payload_bytes=" << source.pool_payload_bytes()
            << " pool_touched_page_bytes=" << source.pool_touched_page_bytes()
            << " variants=" << source.variants()
            << " band_stride_bytes=" << source.band_stride_bytes()
            << " slot_stride_bytes=" << source.slot_stride_bytes() << '\n';
  std::cout << std::fixed << std::setprecision(2)
            << "CPU_PREP_MEASURED gather_ns=" << gather_ns
            << " plan_ns=" << plan_ns << '\n'
            << "CPU_PREP_PER_ITERATION gather_avg_ns="
            << static_cast<double>(gather_ns) / options.iterations
            << " gather_min_ns=" << gather_min_ns
            << " gather_max_ns=" << gather_max_ns
            << " plan_avg_ns=" << static_cast<double>(plan_ns) / options.iterations
            << " plan_min_ns=" << plan_min_ns
            << " plan_max_ns=" << plan_max_ns << '\n'
            << "AICPU_SEND_MODEL assumed_GBps=25.00 wire_ns=" << wire_ns
            << " wr_overhead_ns=" << options.wr_overhead_ns
            << " total_wr_overhead_ns=" << wr_ns
            << " send_ns=" << send_ns << '\n'
            << "AICPU_SEND_PER_ITERATION wire_ns=" << wire_ns / options.iterations
            << " wr_overhead_ns=" << wr_ns / options.iterations
            << " send_ns=" << send_ns_per_iteration << '\n'
            << "PIPELINE_ESTIMATE total_ns=" << pipeline_ns
            << " avg_ns=" << pipeline_ns / options.iterations
            << " effective_GBps=" << total_bytes / pipeline_ns << '\n';
  for (std::size_t iteration = 0U; iteration < iteration_timings.size(); ++iteration) {
    const auto& timing = iteration_timings[iteration];
    std::cout << "ITERATION_NS index=" << iteration + 1U
              << " gather_ns=" << timing.gather_ns
              << " plan_ns=" << timing.plan_ns
              << " source_first_offset=" << timing.source_first_offset
              << " source_last_offset=" << timing.source_last_offset
              << " send_model_ns=" << send_ns_per_iteration
              << " total_model_ns="
              << timing.gather_ns + timing.plan_ns + send_ns_per_iteration << '\n';
  }
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
