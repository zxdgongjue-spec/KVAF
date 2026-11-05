#pragma once

#include <cstdint>

namespace kvaf::aicpu_urma {

constexpr std::uint64_t kUmdkControlMagic = 0x4b564146554d444bULL;
// Setup slot for the control queue; data worker indices remain 0..7.
constexpr std::uint32_t kUmdkControlQueueIndex = 8;
enum class UmdkControlOp : std::uint32_t {
  kPrepare,
  kConnect,
  kDestroy,
  kQueryWorker,
  kRegisterBuffer,
  kWrite
};

struct UmdkControlArgs {
  std::uint64_t status;
  std::uint32_t worker_count;
  std::uint64_t session;
  std::uint64_t owner_pid;
  std::uint64_t local_buffer;
  std::uint64_t buffer_bytes;
  std::uint64_t remote_address;
  std::uint64_t peer_tp;
  std::uint32_t peer_psn;
  std::uint32_t token;
  std::uint32_t eid_index;
  std::uint32_t segment_bytes;
  std::uint32_t key_bytes;
  UmdkControlOp op;
  std::uint8_t route_eid[16];
  std::uint8_t segment[256];
  std::uint8_t peer_key[32];
  char library[256];
  char device[64];
  std::uint32_t worker_index;
  std::uint64_t transfer_spans;
  std::uint32_t transfer_count;
  // Selected by NpuRuntime; device/index only filter matching endpoints.
  std::uint8_t local_eid[16];
};

struct alignas(64) UmdkControlStatus {
  std::uint64_t magic, session, owner_pid, tp, cq_context;
  std::int32_t result, error_number, cq_status;
  std::uint32_t key_bytes;
  std::uint8_t eid[16], key[32];
  char stage[64], detail[256];
};

struct UmdkContext {
  std::uint64_t session{0U}, owner_pid{0U};
};
struct UmdkTransfer {
  std::uint64_t remote, local, bytes;
};
constexpr std::uint32_t kUmdkMaxTransfers = 65536U;
inline bool UmdkRange(std::uint64_t base, std::uint64_t size, std::uint64_t address,
                      std::uint64_t bytes) {
  return bytes && address >= base && bytes <= size && address - base <= size - bytes;
}
}  // namespace kvaf::aicpu_urma

extern "C" std::uint32_t KvafAicpuUmdkControl(kvaf::aicpu_urma::UmdkControlArgs* args);
