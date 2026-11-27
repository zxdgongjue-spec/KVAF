#include "roce_connection.h"

#include <arpa/inet.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

namespace {
namespace wire = kvaf::roce;
constexpr std::uint64_t kMaxRegionBytes = 1ULL << 30;
constexpr std::uint32_t kMaxQpCount = 32;

bool ParseNumber(std::string_view value, unsigned limit, unsigned& result) {
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && result <= limit;
}

int Listen(const char* address, unsigned port) {
  sockaddr_in local{};
  local.sin_family = AF_INET;
  local.sin_port = htons(static_cast<std::uint16_t>(port));
  if (inet_pton(AF_INET, address, &local.sin_addr) != 1) return -1;
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  const int reuse = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) == 0 &&
      bind(fd, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) == 0 && listen(fd, 1) == 0) return fd;
  close(fd);
  return -1;
}

bool Serve(int fd, const kvaf::RoceOptions& options) {
  std::uint64_t bytes = 0;
  std::uint64_t slot_size = 0;
  std::uint32_t qp_count = 0;
  if (!wire::SetControlTimeout(fd, true) || !wire::ReceiveAllocation(fd, &bytes, &slot_size, &qp_count) ||
      bytes == 0 || bytes > kMaxRegionBytes || slot_size == 0 || slot_size > UINT32_MAX ||
      qp_count == 0 || qp_count > kMaxQpCount || bytes % qp_count != 0 ||
      bytes / qp_count < slot_size || (bytes / qp_count) % slot_size != 0) {
    std::fprintf(stderr, "==============[KVAF_CPU]=========== invalid region request (max=%llu bytes)\n",
                 static_cast<unsigned long long>(kMaxRegionBytes));
    return false;
  }
  std::vector<std::unique_ptr<wire::Connection>> connections;
  connections.reserve(qp_count);
  const auto reset_siblings = [&connections] {
    for (std::size_t i = 1; i < connections.size(); ++i) connections[i].reset();
  };
  for (std::uint32_t i = 0; i < qp_count; ++i) {
    auto connection = std::make_unique<wire::Connection>();
    const auto opened = i == 0 ? connection->Open(options) : connection->OpenSibling(*connections[0]);
    if (!opened.ok()) {
      std::fprintf(stderr, "==============[KVAF_CPU]=========== open QP %u failed: %s\n", i, opened.message().c_str());
      reset_siblings();
      return false;
    }
    connections.push_back(std::move(connection));
  }
  void* ddr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ddr == MAP_FAILED) {
    std::perror("==============[KVAF_CPU]=========== mmap");
    reset_siblings();
    return false;
  }
  ibv_mr* mr = ibv_reg_mr(connections[0]->pd(), ddr, bytes,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
  bool released = false;
  bool setup_ok = mr != nullptr;
  if (!mr) {
    std::perror("==============[KVAF_CPU]=========== ibv_reg_mr");
  }
  const auto per_qp_bytes = bytes / qp_count;
  for (std::uint32_t i = 0; setup_ok && i < qp_count; ++i) {
    const auto base = reinterpret_cast<std::uintptr_t>(ddr) + i * per_qp_bytes;
    if (!wire::SendRegion(fd, {base, per_qp_bytes, mr->rkey})) setup_ok = false;
  }
  for (std::uint32_t i = 0; setup_ok && i < qp_count; ++i) {
    wire::Endpoint peer{};
    if (!wire::ReceiveEndpoint(fd, false, &peer)) {
      setup_ok = false;
      break;
    }
    const auto connected = connections[i]->Connect(peer);
    if (!connected.ok()) {
      std::fprintf(stderr, "==============[KVAF_CPU]=========== QP %u connect failed: %s\n", i, connected.message().c_str());
      setup_ok = false;
      break;
    }
    if (!wire::SendEndpoint(fd, true, connections[i]->local())) setup_ok = false;
  }
  if (setup_ok && wire::SetControlTimeout(fd, false)) {
    std::fprintf(stderr, "==============[KVAF_CPU]=========== REGION_READY bytes=%llu slot_size=%llu qp_count=%u\n",
                 static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(slot_size), qp_count);
    released = wire::ReceiveRelease(fd, false);
  } else if (setup_ok) {
    setup_ok = false;
  }

  // Stop all QPs before deregistering the shared backing allocation.
  for (std::uint32_t i = 0; i < qp_count; ++i) {
    const auto stopped = connections[i]->Stop();
    if (!stopped.ok()) {
      std::fprintf(stderr, "==============[KVAF_CPU]=========== stop QP %u failed: %s\n", i, stopped.message().c_str());
      std::_Exit(1);
    }
  }
  if (mr && ibv_dereg_mr(mr) != 0) {
    std::perror("==============[KVAF_CPU]=========== ibv_dereg_mr");
    reset_siblings();
    return false;
  }
  if (munmap(ddr, bytes) != 0) {
    std::perror("==============[KVAF_CPU]=========== munmap");
    reset_siblings();
    return false;
  }
  reset_siblings();
  return setup_ok && released && wire::SendRelease(fd, true);
}

void Usage(const char* executable) {
  std::fprintf(stderr, "Usage: %s --rdma-device NAME --gid-index N "
                       "[--rdma-port N] [--listen IPv4] [--port N]\n", executable);
}
}  // namespace

int main(int argc, char** argv) {
  kvaf::RoceOptions options;
  const char* address = "127.0.0.1";
  unsigned port = 29999;
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    Usage(argv[0]);
    return 0;
  }
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view name(argv[i]);
    const char* value = argv[i + 1];
    unsigned number = 0;
    if (name == "--listen") address = value;
    else if (name == "--rdma-device") options.device = value;
    else if (name == "--port" && ParseNumber(value, 65535, number) && number) port = number;
    else if (name == "--rdma-port" && ParseNumber(value, 255, number) && number)
      options.port = static_cast<std::uint8_t>(number);
    else if (name == "--gid-index" && ParseNumber(value, 255, number))
      options.gid_index = static_cast<int>(number);
    else {
      Usage(argv[0]);
      return 2;
    }
  }
  if (argc % 2 == 0 || options.device.empty() || options.gid_index < 0) {
    Usage(argv[0]);
    return 2;
  }
  const int listener = Listen(address, port);
  if (listener < 0) {
    std::fprintf(stderr, "==============[KVAF_CPU]=========== cannot listen on %s:%u\n", address, port);
    return 1;
  }
  std::fprintf(stderr, "==============[KVAF_CPU]=========== LISTEN address=%s port=%u\n", address, port);
  const int fd = accept(listener, nullptr, nullptr);
  close(listener);
  if (fd < 0) {
    std::perror("==============[KVAF_CPU]=========== accept");
    return 1;
  }
  const bool ok = Serve(fd, options);
  close(fd);
  return ok ? 0 : 1;
}
