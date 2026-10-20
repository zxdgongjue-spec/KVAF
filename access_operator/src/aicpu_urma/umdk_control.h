#pragma once

#include <cstdint>

namespace kvaf::aicpu_urma {

constexpr std::uint64_t kUmdkControlMagic = 0x4b564146554d444bULL;
constexpr std::uint32_t kUmdkControlDiagnosticBytes = 32768;
// Setup slot for the control queue; data worker indices remain 0..7.
constexpr std::uint32_t kUmdkControlQueueIndex = 8;
enum class UmdkControlOp : std::uint32_t {
  kPrepareGather,
  kConnect,
  kDestroy,
  kQueryGatherWorker,
  kRegisterGatherBuffer,
  kWriteGather
};

struct UmdkControlArgs {
  std::uint64_t status;
  std::uint64_t diagnostics;
  std::uint32_t diagnostic_capacity;
  std::uint32_t worker_count;
  std::uint64_t session;
  std::uint64_t owner_pid;
  std::uint64_t local_buffer;
  std::uint64_t buffer_bytes;
  std::uint64_t remote_address;
  std::uint64_t peer_tp;
  std::uint32_t peer_psn;
  std::uint32_t token;
  std::uint32_t bytes;
  std::uint32_t blocks;
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
};

struct alignas(64) UmdkControlStatus {
  std::uint64_t magic;
  std::uint64_t session;
  std::uint64_t owner_pid;
  std::uint64_t tp;
  std::uint64_t post_ns;
  std::uint64_t cq_ns;
  std::uint64_t total_ns;
  std::uint64_t cq_context;
  std::uint32_t read_wqes;
  std::uint32_t post_calls;
  std::int32_t result;
  std::int32_t error_number;
  std::int32_t cq_status;
  std::uint32_t key_bytes;
  std::uint32_t eid_index;
  std::int32_t init_result;
  std::int32_t init_errno;
  std::int32_t device_count;
  std::uint32_t diagnostic_bytes;
  std::uint32_t diagnostic_truncated;
  char library[256];
  std::uint8_t eid[16];
  std::uint8_t key[32];
  char device[64];
  char stage[64];
  char detail[256];
  std::uint64_t cq_poll_calls;
  std::uint64_t local_jetty_address;
  std::uint64_t local_cq_address;
  std::uint64_t target_address;
  std::uint32_t local_jetty_id;
  std::uint32_t local_cq_id;
  std::uint32_t peer_jetty_id;
  std::uint64_t build_begin_ns;
  std::uint64_t first_post_ns;
  std::uint64_t last_post_ns;
  std::uint64_t completion_ns;
  std::uint64_t wr_builds;  // Fresh WR chain builds, retaining ABI layout.
  std::uint64_t build_end_ns;
};

}  // namespace kvaf::aicpu_urma

extern "C" std::uint32_t KvafAicpuUmdkControl(kvaf::aicpu_urma::UmdkControlArgs* args);
