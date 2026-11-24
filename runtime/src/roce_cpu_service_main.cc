#include "roce_connection.h"
#include "runtime_service.h"

#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <system_error>
#include <thread>

namespace kvaf {
namespace {

struct Limits {
  unsigned sessions{64};
  std::uint64_t region_bytes{4ULL << 30};
  std::uint64_t total_bytes{16ULL << 30};
};

struct Usage {
  bool Reserve(std::uint64_t bytes, std::uint64_t limit) {
    auto used = registered_bytes.load(std::memory_order_relaxed);
    while (bytes <= limit - used) {
      if (registered_bytes.compare_exchange_weak(used, used + bytes, std::memory_order_relaxed))
        return true;
    }
    return false;
  }

  std::atomic<unsigned> sessions{0};
  std::atomic<std::uint64_t> registered_bytes{0};
};

struct Session {
  roce::Connection connection;
  void* ddr{};
  std::size_t bytes{};
  ibv_mr* mr{};

  bool Release() {
    if (!connection.Stop().ok()) return false;
    if (mr != nullptr && ibv_dereg_mr(mr) != 0) return false;
    mr = nullptr;
    if (ddr != nullptr && munmap(ddr, bytes) != 0) return false;
    ddr = nullptr;
    return true;
  }
};

void ServeSession(int fd, const RoceOptions& options, const Limits& limits, Usage& usage) {
  auto session = std::make_unique<Session>();
  std::uint64_t bytes = 0;
  std::uint64_t slot_size = 0;
  roce::Endpoint peer;
  bool release_requested = false;
  bool reserved = false;
  if (roce::SetControlTimeout(fd, true) &&
      roce::ReceiveAllocation(fd, &bytes, &slot_size) && bytes > 0 &&
      slot_size > 0 && bytes % slot_size == 0 &&
      bytes <= limits.region_bytes && bytes <= std::numeric_limits<std::size_t>::max() &&
      usage.Reserve(bytes, limits.total_bytes)) {
    reserved = true;
    auto status = session->connection.Open(options);
    if (status.ok() && bytes <= session->connection.max_mr_size()) {
      void* memory = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (memory != MAP_FAILED) {
        session->ddr = memory;
        session->bytes = bytes;
      } else {
        const int error = errno;
        std::fprintf(stderr, "[KVAF_ROCE] DDR mmap failed bytes=%llu errno=%d (%s)\n",
                     static_cast<unsigned long long>(bytes), error, std::strerror(error));
      }
    } else if (status.ok()) {
      std::fprintf(stderr, "[KVAF_ROCE] DDR request exceeds device MR limit bytes=%llu limit=%llu\n",
                   static_cast<unsigned long long>(bytes),
                   static_cast<unsigned long long>(session->connection.max_mr_size()));
    }
    if (session->ddr != nullptr) {
      session->mr = ibv_reg_mr(session->connection.pd(), session->ddr, bytes,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                                  IBV_ACCESS_REMOTE_WRITE);
      if (session->mr == nullptr) {
        const int error = errno;
        std::fprintf(stderr, "[KVAF_ROCE] DDR ibv_reg_mr failed bytes=%llu errno=%d (%s)\n",
                     static_cast<unsigned long long>(bytes), error, std::strerror(error));
      }
      if (session->mr != nullptr &&
          roce::SendRegion(fd, {reinterpret_cast<std::uintptr_t>(session->ddr),
                                bytes, session->mr->rkey}) &&
          roce::ReceiveEndpoint(fd, false, &peer)) {
        status = session->connection.Connect(peer);
        if (status.ok() && roce::SendEndpoint(fd, true, session->connection.local()) &&
            roce::SetControlTimeout(fd, false)) {
          std::fprintf(stderr, "[KVAF_ROCE] REGION_READY bytes=%llu qpn=%u\n",
                       static_cast<unsigned long long>(bytes),
                       session->connection.local().qpn);
          release_requested = roce::ReceiveRelease(fd, false);
        }
      }
    }
    if (!status.ok()) {
      std::fprintf(stderr, "[KVAF_ROCE] %s\n", status.message().c_str());
    }
  } else {
    std::fprintf(stderr, "[KVAF_ROCE] rejected region request bytes=%llu slot_size=%llu\n",
                 static_cast<unsigned long long>(bytes),
                 static_cast<unsigned long long>(slot_size));
  }
  if (session->Release()) {
    if (reserved) usage.registered_bytes.fetch_sub(bytes, std::memory_order_relaxed);
    usage.sessions.fetch_sub(1, std::memory_order_relaxed);
    if (release_requested) (void)roce::SendRelease(fd, true);
  } else {
    // Keep the registered backing and its provider resources on teardown failure.
    std::fprintf(stderr, "[KVAF_ROCE] cannot release session safely, retaining resources\n");
    (void)session.release();
  }
  close(fd);
}

bool ParseNumber(const char* text, std::uint64_t max, std::uint64_t* result) {
  if (*text < '0' || *text > '9') return false;
  char* end = nullptr;
  errno = 0;
  const auto value = std::strtoull(text, &end, 10);
  if (errno || end == text || *end != '\0' || value > max) return false;
  *result = value;
  return true;
}

}  // namespace
}  // namespace kvaf

int main(int argc, char** argv) {
  kvaf::RoceOptions roce;
  kvaf::RuntimeServiceOptions listener;
  kvaf::Limits limits;
  for (int index = 1; index + 1 < argc; index += 2) {
    const std::string name = argv[index];
    const char* value = argv[index + 1];
    std::uint64_t number = 0;
    if (name == "--listen") listener.address = value;
    else if (name == "--rdma-device") roce.device = value;
    else if (name == "--port" && kvaf::ParseNumber(value, 65535, &number) && number)
      listener.port = static_cast<std::uint16_t>(number);
    else if (name == "--rdma-port" && kvaf::ParseNumber(value, 255, &number) && number)
      roce.port = static_cast<std::uint8_t>(number);
    else if (name == "--gid-index" && kvaf::ParseNumber(value, 255, &number))
      roce.gid_index = static_cast<int>(number);
    else if (name == "--max-sessions" && kvaf::ParseNumber(value, 65535, &number) && number)
      limits.sessions = static_cast<unsigned>(number);
    else if (name == "--max-region-bytes" && kvaf::ParseNumber(value, UINT64_MAX, &number) && number)
      limits.region_bytes = number;
    else if (name == "--max-total-bytes" && kvaf::ParseNumber(value, UINT64_MAX, &number) && number)
      limits.total_bytes = number;
    else {
      std::fprintf(stderr, "invalid option %s\n", name.c_str());
      return 2;
    }
  }
  if (argc % 2 == 0 || roce.device.empty() || roce.gid_index < 0) {
    std::fprintf(stderr, "Usage: %s --rdma-device NAME --gid-index N "
                         "[--rdma-port N] [--listen HOST] [--port PORT] "
                         "[--max-sessions N] [--max-region-bytes N] [--max-total-bytes N]\n", argv[0]);
    return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 2;
  }
  auto usage = std::make_shared<kvaf::Usage>();
  const auto status = kvaf::RuntimeService(listener).ServeConnections([roce, limits, usage](int fd) {
    if (usage->sessions.fetch_add(1, std::memory_order_relaxed) >= limits.sessions) {
      usage->sessions.fetch_sub(1, std::memory_order_relaxed);
      close(fd);
      return;
    }
    try {
      std::thread([fd, roce, limits, usage] {
        kvaf::ServeSession(fd, roce, limits, *usage);
      }).detach();
    } catch (const std::system_error& error) {
      usage->sessions.fetch_sub(1, std::memory_order_relaxed);
      std::fprintf(stderr, "[KVAF_ROCE] cannot start session: %s\n", error.what());
      close(fd);
    }
  });
  std::fprintf(stderr, "[KVAF_ROCE] %s\n", status.message().c_str());
  return 1;
}
