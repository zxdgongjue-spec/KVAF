#include "roce_connection.h"

#include <acl/acl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

extern "C" int halMemRegUbSegment(std::uint32_t device, std::uint64_t address, std::uint64_t size);
extern "C" int halMemUnRegUbSegment(std::uint32_t device, std::uint64_t address);

namespace {
namespace wire = kvaf::roce;
constexpr std::size_t kDefaultBytes = 1U << 20;
constexpr std::size_t kMaxBufferBytes = 1ULL << 30;
constexpr unsigned kMaxQueueDepth = 128;
constexpr unsigned kMaxQpCount = 32;
constexpr auto kCompletionTimeout = std::chrono::seconds(10);

struct Options {
  std::uint32_t device{};
  const char* server{};
  kvaf::RoceOptions rdma;
  std::size_t bytes{kDefaultBytes};
  std::uint64_t iterations{1000};
  std::uint64_t warmup{100};
  unsigned depth{64};
  unsigned qp_count{1};
  bool run_read{true};
  bool run_write{true};
};

bool ParseNumber(std::string_view value, std::uint64_t limit, std::uint64_t& number) {
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && number <= limit;
}

bool ParseOptions(int argc, char** argv, Options& options) {
  std::uint64_t device = 0;
  if (argc < 3 || !ParseNumber(argv[1], INT32_MAX, device)) return false;
  options.device = static_cast<std::uint32_t>(device);
  options.server = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string_view name(argv[i]);
    if (++i == argc) return false;
    const std::string_view value(argv[i]);
    std::uint64_t number = 0;
    if (name == "--rdma-device") options.rdma.device = value;
    else if (name == "--rdma-port" && ParseNumber(value, 255, number) && number)
      options.rdma.port = static_cast<std::uint8_t>(number);
    else if (name == "--gid-index" && ParseNumber(value, 255, number))
      options.rdma.gid_index = static_cast<int>(number);
    else if (name == "--bytes" && ParseNumber(value, UINT32_MAX, number) && number)
      options.bytes = static_cast<std::size_t>(number);
    else if (name == "--iterations" && ParseNumber(value, UINT64_MAX, number) && number)
      options.iterations = number;
    else if (name == "--warmup" && ParseNumber(value, UINT64_MAX, number))
      options.warmup = number;
    else if (name == "--depth" && ParseNumber(value, kMaxQueueDepth, number) && number)
      options.depth = static_cast<unsigned>(number);
    else if (name == "--qp-count" && ParseNumber(value, kMaxQpCount, number) && number)
      options.qp_count = static_cast<unsigned>(number);
    else if (name == "--operation" && value == "read") options.run_write = false;
    else if (name == "--operation" && value == "write") options.run_read = false;
    else if (name == "--operation" && value == "both") {}
    else return false;
  }
  const auto max_per_qp = kMaxBufferBytes / options.qp_count;
  if (options.bytes > max_per_qp / options.depth) return false;
  const auto per_qp_bytes = options.bytes * options.depth;
  const auto total_bytes = per_qp_bytes * options.qp_count;
  return !options.rdma.device.empty() && options.rdma.gid_index >= 0 &&
         options.bytes <= UINT32_MAX && total_bytes <= kMaxBufferBytes;
}

void Usage(const char* executable) {
  std::fprintf(stderr,
      "Usage: %s ACL_DEVICE CPU_SERVICE_HOST:PORT --rdma-device NAME --gid-index N [options]\n"
      "Options: --rdma-port N --bytes N --iterations N --warmup N --depth N --qp-count N --operation read|write|both\n"
      "Defaults: bytes=1048576 iterations=1000 warmup=100 depth=64 qp-count=1 operation=both\n"
      "iterations and warmup are per QP; total HBM/DDR buffer is bytes * depth * qp-count (max 1 GiB).\n", executable);
}

bool Check(int code, const char* stage) {
  if (code == 0) return true;
  std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== %s failed rc=%d\n", stage, code);
  return false;
}

bool Check(const kvaf::Status& status, const char* stage) {
  if (status.ok()) return true;
  std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== %s failed: %s\n", stage, status.message().c_str());
  return false;
}

[[noreturn]] void UnsafeTeardown(const char* stage) {
  std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== %s failed, exiting with RDMA resources still owned\n", stage);
  std::_Exit(1);
}

struct Queue {
  std::unique_ptr<wire::Connection> connection;
  wire::Region region{};
  std::vector<ibv_sge> sges;
  std::vector<ibv_send_wr> requests;
};

struct Session {
  explicit Session(const Options& opts)
      : options(opts), per_qp_bytes(opts.bytes * opts.depth), buffer_bytes(per_qp_bytes * opts.qp_count) {
    queues.resize(options.qp_count);
  }

  bool Connect() {
    queues[0].connection = std::make_unique<wire::Connection>();
    if (!Check(queues[0].connection->Open(options.rdma), "open local verbs resources")) return false;
    for (std::size_t i = 1; i < queues.size(); ++i) {
      queues[i].connection = std::make_unique<wire::Connection>();
      if (!Check(queues[i].connection->OpenSibling(*queues[0].connection), "open sibling QP")) return false;
    }
    for (const auto& queue : queues) {
      if (options.depth > queue.connection->depth()) {
        std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== depth=%u exceeds local QP capacity=%u\n",
                     options.depth, queue.connection->depth());
        return false;
      }
    }

    fd = wire::ConnectControl(options.server);
    if (fd < 0 || !wire::SendAllocation(fd, buffer_bytes, options.bytes, options.qp_count)) {
      std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== TCP allocation handshake failed for %s\n", options.server);
      return false;
    }
    for (auto& queue : queues) {
      if (!wire::ReceiveRegion(fd, &queue.region) || queue.region.base == 0 ||
          queue.region.bytes < per_qp_bytes ||
          queue.region.base > std::numeric_limits<std::uint64_t>::max() - per_qp_bytes) {
        std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== invalid remote DDR region descriptor\n");
        return false;
      }
    }
    remote_ready = true;
    for (const auto& queue : queues) {
      if (!wire::SendEndpoint(fd, false, queue.connection->local())) {
        std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== failed to send local QP descriptor\n");
        return false;
      }
    }
    for (std::size_t i = 0; i < queues.size(); ++i) {
      wire::Endpoint peer{};
      if (!wire::ReceiveEndpoint(fd, true, &peer)) {
        std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== failed to receive peer QP descriptor index=%zu\n", i);
        return false;
      }
      if (!Check(queues[i].connection->Connect(peer), "connect RC QP")) return false;
      std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== QP_READY index=%zu local=%u remote=%u read_depth=%u\n",
                   i, queues[i].connection->local().qpn, peer.qpn,
                   queues[i].connection->read_depth());
    }
    return true;
  }

  bool PrepareHbm() {
    acl_ready = Check(aclInit(nullptr), "aclInit");
    device_ready = acl_ready && Check(aclrtSetDevice(static_cast<int>(options.device)), "aclrtSetDevice");
    if (!device_ready || !Check(aclrtMalloc(&hbm, buffer_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(HBM)"))
      return false;
    void* allocation = nullptr;
    std::size_t allocation_bytes = 0;
    if (!Check(aclrtMemGetAddressRange(hbm, &allocation, &allocation_bytes), "aclrtMemGetAddressRange")) return false;
    const auto base = reinterpret_cast<std::uintptr_t>(allocation);
    const auto address = reinterpret_cast<std::uintptr_t>(hbm);
    if (base == 0 || address < base || allocation_bytes < buffer_bytes ||
        address - base > allocation_bytes - buffer_bytes) {
      std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== HBM buffer exceeds its backing allocation\n");
      return false;
    }
    if (!Check(halMemRegUbSegment(options.device, base, allocation_bytes), "halMemRegUbSegment(HBM)")) return false;
    registered_base = base;
    local_mr = ibv_reg_mr(queues[0].connection->pd(), hbm, buffer_bytes, IBV_ACCESS_LOCAL_WRITE);
    if (!local_mr) {
      const int error = errno;
      std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== ibv_reg_mr(HBM) failed errno=%d (%s)\n",
                   error, std::strerror(error));
      return false;
    }
    for (auto& queue : queues) {
      queue.sges.resize(options.depth);
      queue.requests.resize(options.depth);
    }
    std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== HBM_MR_READY bytes=%zu qp_count=%u bytes_per_qp=%zu\n",
                 buffer_bytes, options.qp_count, per_qp_bytes);
    return true;
  }

  bool PostAndWait(Queue& queue, std::size_t qp_index, bool read, unsigned count) {
    const auto* local_base = static_cast<const std::uint8_t*>(hbm) + qp_index * per_qp_bytes;
    for (unsigned i = 0; i < count; ++i) {
      const auto slot_offset = static_cast<std::size_t>(i) * options.bytes;
      queue.sges[i].addr = reinterpret_cast<std::uintptr_t>(local_base + slot_offset);
      queue.sges[i].length = static_cast<std::uint32_t>(options.bytes);
      queue.sges[i].lkey = local_mr->lkey;
      auto& wr = queue.requests[i];
      wr = {};
      wr.wr_id = i;
      wr.sg_list = &queue.sges[i];
      wr.num_sge = 1;
      wr.opcode = read ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.wr.rdma.remote_addr = queue.region.base + slot_offset;
      wr.wr.rdma.rkey = queue.region.rkey;
      if (i + 1 < count) wr.next = &queue.requests[i + 1];
    }
    ibv_send_wr* bad = nullptr;
    const int rc = ibv_post_send(queue.connection->qp(), queue.requests.data(), &bad);
    if (rc != 0) {
      const auto index = bad ? static_cast<std::size_t>(bad - queue.requests.data()) : 0;
      std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== ibv_post_send qp=%zu failed rc=%d first_bad=%zu\n", qp_index, rc, index);
      return false;
    }
    unsigned completed = 0;
    const auto deadline = std::chrono::steady_clock::now() + kCompletionTimeout;
    while (completed < count && std::chrono::steady_clock::now() < deadline) {
      std::array<ibv_wc, 32> completions{};
      const int got = ibv_poll_cq(queue.connection->cq(), static_cast<int>(completions.size()), completions.data());
      if (got < 0) {
        std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== ibv_poll_cq qp=%zu failed rc=%d\n", qp_index, got);
        return false;
      }
      for (int i = 0; i < got; ++i) {
        if (completions[i].status != IBV_WC_SUCCESS ||
            completions[i].opcode != (read ? IBV_WC_RDMA_READ : IBV_WC_RDMA_WRITE) ||
            completions[i].wr_id >= count) {
          std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== %s completion qp=%zu failed: %s (%d), vendor_err=%u\n",
                       read ? "READ" : "WRITE", qp_index, ibv_wc_status_str(completions[i].status),
                       completions[i].status, completions[i].vendor_err);
          return false;
        }
        ++completed;
      }
    }
    if (completed != count) {
      std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== %s completion timeout qp=%zu completed=%u expected=%u\n",
                   read ? "READ" : "WRITE", qp_index, completed, count);
      return false;
    }
    return true;
  }

  bool RunQueue(Queue& queue, std::size_t qp_index, bool read, std::uint64_t operations) {
    std::uint64_t done = 0;
    while (done < operations) {
      const auto count = static_cast<unsigned>(std::min<std::uint64_t>(options.depth, operations - done));
      if (!PostAndWait(queue, qp_index, read, count)) return false;
      done += count;
    }
    return true;
  }

  bool RunConcurrent(bool read, std::uint64_t operations_per_qp, double* elapsed_seconds) {
    std::mutex mutex;
    std::condition_variable cv;
    std::size_t ready = 0;
    bool start = false;
    std::atomic<bool> ok{true};
    std::vector<std::thread> workers;
    workers.reserve(queues.size());
    for (std::size_t i = 0; i < queues.size(); ++i) {
      workers.emplace_back([&, i] {
        {
          std::unique_lock<std::mutex> lock(mutex);
          ++ready;
          // Workers and the coordinator wait on this same condition variable.
          // Wake all waiters so a worker cannot consume the only notification
          // intended to wake the coordinator when the final worker is ready.
          cv.notify_all();
          cv.wait(lock, [&] { return start; });
        }
        if (!RunQueue(queues[i], i, read, operations_per_qp)) ok.store(false, std::memory_order_relaxed);
      });
    }
    std::chrono::steady_clock::time_point begin;
    {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&] { return ready == queues.size(); });
      begin = std::chrono::steady_clock::now();
      start = true;
    }
    cv.notify_all();
    for (auto& worker : workers) worker.join();
    const auto end = std::chrono::steady_clock::now();
    if (!ok.load(std::memory_order_relaxed)) return false;
    if (elapsed_seconds) *elapsed_seconds = std::chrono::duration<double>(end - begin).count();
    return true;
  }

  bool RunPhase(bool read, const char* name) {
    if (options.warmup && !RunConcurrent(read, options.warmup, nullptr)) return false;
    double elapsed = 0;
    if (!RunConcurrent(read, options.iterations, &elapsed)) return false;
    const double wr_count = static_cast<double>(options.iterations) * options.qp_count;
    const double bytes = wr_count * options.bytes;
    const double gbps = bytes / elapsed / 1e9;
    std::printf("PERF_RESULT operation=%s qp_count=%u bytes_per_wr=%zu iterations_per_qp=%llu "
                "depth_per_qp=%u elapsed_s=%.6f amortized_us_per_wr=%.3f "
                "GBps=%.3f GiBps=%.3f Gbps=%.3f\n",
                name, options.qp_count, options.bytes,
                static_cast<unsigned long long>(options.iterations), options.depth, elapsed,
                elapsed * 1e6 / wr_count, gbps,
                gbps / (1024.0 * 1024.0 * 1024.0 / 1e9), gbps * 8.0);
    return true;
  }

  bool Validate(bool read) {
    if (!RunConcurrent(true, options.depth, nullptr)) return false;
    constexpr std::size_t kCheckChunk = 1U << 20;
    std::vector<std::uint8_t> output(std::min(buffer_bytes, kCheckChunk));
    for (std::size_t offset = 0; offset < buffer_bytes; offset += output.size()) {
      const auto chunk = std::min(output.size(), buffer_bytes - offset);
      if (!Check(aclrtMemcpy(output.data(), chunk, static_cast<const std::uint8_t*>(hbm) + offset,
                             chunk, ACL_MEMCPY_DEVICE_TO_HOST), "copy HBM validation data to Host")) return false;
      for (std::size_t i = 0; i < chunk; ++i) {
        const auto global_offset = offset + i;
        const auto expected = static_cast<std::uint8_t>((global_offset % options.bytes) * 131U + 0x40U);
        if (output[i] == expected) continue;
        std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== %s data mismatch at byte %zu expected=0x%02x actual=0x%02x\n",
                     read ? "READ" : "WRITE", global_offset, expected, output[i]);
        return false;
      }
    }
    std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== %s_VALIDATION_PASS bytes=%zu qp_count=%u\n",
                 read ? "READ" : "WRITE", buffer_bytes, options.qp_count);
    return true;
  }

  bool Benchmark() {
    if (!PrepareHbm()) return false;
    std::vector<std::uint8_t> input(options.bytes);
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<std::uint8_t>((i % options.bytes) * 131U + 0x40U);
    for (std::size_t offset = 0; offset < buffer_bytes; offset += options.bytes) {
      auto* destination = static_cast<std::uint8_t*>(hbm) + offset;
      if (!Check(aclrtMemcpy(destination, options.bytes, input.data(), options.bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                 "prepare HBM test pattern")) return false;
    }
    // Seed every remote slot before the measured phases.
    if (!RunConcurrent(false, options.depth, nullptr)) return false;
    if (options.run_write && (!RunPhase(false, "WRITE") || !Validate(false))) return false;
    if (options.run_read && (!RunPhase(true, "READ") || !Validate(true))) return false;
    return true;
  }

  bool Close() {
    for (auto& queue : queues) {
      if (queue.connection && !Check(queue.connection->Stop(), "destroy local QP"))
        UnsafeTeardown("destroy local QP");
    }
    if (local_mr && ibv_dereg_mr(local_mr) != 0) UnsafeTeardown("ibv_dereg_mr(HBM)");
    local_mr = nullptr;
    if (registered_base && !Check(halMemUnRegUbSegment(options.device, registered_base), "halMemUnRegUbSegment(HBM)"))
      UnsafeTeardown("halMemUnRegUbSegment(HBM)");
    bool ok = true;
    if (remote_ready && (!wire::SendRelease(fd, false) || !wire::ReceiveRelease(fd, true))) {
      std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== remote DDR release handshake failed\n");
      ok = false;
    }
    if (fd >= 0) close(fd);
    if (hbm) ok = Check(aclrtFree(hbm), "aclrtFree(HBM)") && ok;
    if (device_ready) ok = Check(aclrtResetDevice(static_cast<int>(options.device)), "aclrtResetDevice") && ok;
    if (acl_ready) ok = Check(aclFinalize(), "aclFinalize") && ok;
    for (std::size_t i = 1; i < queues.size(); ++i) queues[i].connection.reset();
    return ok;
  }

  const Options& options;
  std::size_t per_qp_bytes;
  std::size_t buffer_bytes;
  std::vector<Queue> queues;
  int fd{-1};
  bool remote_ready{false};
  bool acl_ready{false};
  bool device_ready{false};
  void* hbm{};
  std::uintptr_t registered_base{};
  ibv_mr* local_mr{};
};
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") { Usage(argv[0]); return 0; }
  Options options;
  if (!ParseOptions(argc, argv, options)) { Usage(argv[0]); return 2; }
  Session session(options);
  const bool ok = session.Connect() && session.Benchmark();
  if (!session.Close() || !ok) return 1;
  std::fprintf(stderr, "==============[HOST_ROCE_BW]=========== PASS qp_count=%u\n", options.qp_count);
  return 0;
}
