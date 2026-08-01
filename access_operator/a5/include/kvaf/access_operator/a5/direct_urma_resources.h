#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "kvaf/access_operator/aiv_simt_urma_h2d.h"
#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf::access::a5 {

enum class DirectUrmaExecutionDomain : std::uint32_t {
  kHostProcess = 0,
  kAicpu,
};

enum class DirectUrmaRole : std::uint32_t {
  kServer = 0,
  kClient,
};

// Monotonic Close cursor. In particular, kAicpuResultPending means the close
// kernel has already been launched and synchronized: another Close call may
// only reread the existing device result buffer, never launch that kernel
// again.
enum class DirectUrmaClosePhase : std::uint32_t {
  kNotStarted = 0,
  kHostProviderCall,
  kAicpuPreLaunch,
  kAicpuResultPending,
  kProviderRetryable,
  kProviderClosed,
  kLocalCleanup,
  kComplete,
  kUnsafeUnknown,
  // Device-domain epoch owner phases.  The owner launch remains live across
  // every AIV batch; only the final SHUTDOWN_QUIESCED command allows its
  // stream to be joined.
  kAicpuEpochOwnerReady,
  kAivShutdownPreLaunch,
  kAivShutdownResultPending,
  kAicpuEpochOwnerJoinPending,
  kAicpuEpochOwnerStatusPending,
};

// Why Close stopped. This separates a provider-authorized retry from a
// post-launch result reread and from an ownership state that must not be
// touched again before reset/isolation.
enum class DirectUrmaCloseDisposition : std::uint32_t {
  kNotStarted = 0,
  kPreLaunchRetryable,
  kAicpuResultReadPending,
  kProviderRetryable,
  kTransportComplete,
  kUnsafeUnknown,
  kAivShutdownPreLaunchRetryable,
  kAivShutdownResultReadPending,
  kAicpuEpochOwnerJoinPending,
  kAicpuEpochOwnerStatusReadPending,
};

struct DirectUrmaOpenSpec {
  DirectUrmaExecutionDomain execution_domain{
      DirectUrmaExecutionDomain::kAicpu};
  DirectUrmaRole role{DirectUrmaRole::kClient};
  std::string provider_library_path;
  std::string provider_soname{"libkvaf_urma_provider_device.so"};
  std::string urma_device;
  std::string local_eid;
  std::string peer_eid;
  std::uint32_t eid_index{0};
  std::uint32_t control_port{21115};
  std::uint32_t control_timeout_seconds{120};
  Address local_address{0};
  std::uint64_t local_allocation_bytes{0};
  std::uint64_t segment_payload_bytes{0};
};

struct DirectUrmaMemoryView {
  Address address{0};
  std::uint64_t bytes{0};
};

// Direct liburma/UDMA control plane. The Host endpoint calls the provider
// shared library in-process once per epoch. The Device endpoint launches one
// asynchronous AICPU epoch owner which creates resources, remains live across
// any number of AIV batches, and performs checked teardown itself after an
// AIV-published SHUTDOWN_QUIESCED command. No process-local provider pointer,
// HCOMM handle or HCOMM resource ABI crosses this class.
//
// The caller owns every DDR/HBM buffer named by DirectUrmaOpenSpec.  If
// Close() fails while transport_teardown_complete() is false, it must retain
// this object and all backing memory and retry only when
// transport_teardown_retryable() is true; object destruction is not an
// ownership-recovery operation.
class DirectUrmaResources {
 public:
  DirectUrmaResources();
  ~DirectUrmaResources();

  DirectUrmaResources(const DirectUrmaResources&) = delete;
  DirectUrmaResources& operator=(const DirectUrmaResources&) = delete;

  Status Open(const DirectUrmaOpenSpec& spec);
  DirectUrmaMemoryView local_memory() const;
  DirectUrmaMemoryView remote_memory() const;
  std::uint64_t epoch() const;
  Status SnapshotDirectChannel(
      Address local_address, std::uint64_t local_bytes,
      Address remote_address, std::uint64_t remote_bytes,
      AivDirectUrmaChannel* direct_channel) const;
  // Enqueue every AIV data-plane batch through this object.  The Device
  // endpoint uses the same owned stream for data and final shutdown, so Close
  // freezes new submissions and its SHUTDOWN_QUIESCED kernel is FIFO-ordered
  // after every accepted batch.  Calling the low-level launcher on an
  // external stream is outside this lifetime contract.
  Status EnqueueBatch(
      const DeviceBatchGetParams* device_params,
      DeviceCompletion* device_completion,
      AivDirectUrmaChannel* direct_channel);
  // Wait for all batches accepted before this call.  Close also synchronizes
  // this stream; after a successful Close this method remains an idempotent
  // success rather than touching a destroyed stream.
  Status SynchronizeDataPlane();
  // True once no provider/queue can access the caller's backing memory. This
  // can become true even when Close still reports a local control-buffer,
  // stream-handle, or dlclose cleanup error.
  bool transport_teardown_complete() const;
  // True only while the exact provider session and its checked teardown
  // cursor are still owned and another Close call is safe. False means either
  // teardown completed or provider/hardware ownership became ambiguous.
  bool transport_teardown_retryable() const;
  DirectUrmaClosePhase close_phase() const;
  DirectUrmaCloseDisposition close_disposition() const;
  Status Close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf::access::a5
