#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "runtime.h"

namespace kvaf { class RuntimeCore; }

namespace kvaf {

enum class CpuGatherMode { kAdaptive, kActive, kPassive };

struct CpuRuntimeOptions {
  std::string ub_device;
  bool has_eid_index{false};
  std::uint32_t eid_index{0};
  std::uint32_t gather_threads{0U};
  bool gather_affinity{true};
  CpuGatherMode gather_mode{CpuGatherMode::kAdaptive};
};

struct CpuRuntimeClientOptions {
  std::string peer_address;
  std::uint64_t hbm_bytes{0U};
  std::uint64_t slot_size{0U};
  std::string ub_device;
  bool has_eid_index{false};
  std::uint32_t eid_index{0U};
  std::uint32_t device_id{0U};
  std::uint32_t wqes_per_doorbell{0U};
  std::uint32_t remote_sgl_size{1U};
  std::uint32_t prepare_thread_count{4U};
};

struct CpuRuntimeBatchTiming {
  std::uint64_t prepare_ns{0U};
  std::uint64_t post_ns{0U};
  std::uint64_t poll_ns{0U};
  std::uint64_t total_ns{0U};
  std::uint64_t wr_count{0U};
  std::uint64_t plan_ns{0U};
  std::uint64_t setup_ns{0U};
};

// CPU endpoint Runtime. It owns only CPU-side UMDK resources and sessions;
// RuntimeService owns the listening socket and calls ServeConnection.
class CpuRuntime final : public Runtime {
 public:
  CpuRuntime();
  ~CpuRuntime() override;

  CpuRuntime(const CpuRuntime&) = delete;
  CpuRuntime& operator=(const CpuRuntime&) = delete;

  Status Initialize(const CpuRuntimeOptions& options);
  Status InitializeClient(const CpuRuntimeClientOptions& options);
  RuntimeCore& core();
  const RuntimeCore& core() const;
  CpuRuntimeBatchTiming last_batch_timing() const;

  Status LinkTo(const LinkTarget& target) override;
  Status AcceptLink(const LinkTarget& peer) override;
  Status ReleaseLink(LinkId link_id) override;
  Status ServeConnection(int connection_fd) override;
  Status Shutdown() override;
  bool initialized() const override;
  RuntimeCapabilities capabilities() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string pending_peer_address_;
};

}  // namespace kvaf
