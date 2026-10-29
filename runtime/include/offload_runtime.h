#pragma once
#include "runtime.h"

namespace kvaf {
struct OffloadRuntimeOptions {
  static constexpr std::uint32_t kAutoCandidate = UINT32_MAX;
  std::uint32_t device_id{0};
  // Empty discovers CLOS EIDs then the active asdrv pair_info device EID.
  // Nonempty pins one hexadecimal EID (optional ':' separators).
  std::string npu_eid;
  std::uint32_t cpu_candidate{kAutoCandidate}; // Otherwise pins one CPU candidate.
  std::uint32_t timeout_ms{30000};
};
// CANN host control only: no AIV kernels, and one synchronous caller per runtime.
class OffloadRuntime {
 public:
  explicit OffloadRuntime(OffloadRuntimeOptions options);
  ~OffloadRuntime();
  Status LinkTo(const LinkTarget&);
  Status Initialize(std::uint64_t bytes, std::uint64_t slot_size);
  Status Offload(const BatchPutRequest&, OffloadExecutionOptions = {});
  Status ResolveOffload();
  Status Shutdown();
  Status LastChecksum(std::uint64_t* checksum); // Diagnostic, reads CPU DDR after CQ.
  RuntimeCore& core();
  bool initialized() const;
  bool pending_offload() const;
  std::uint64_t task_id() const;
  std::uint32_t device_id() const;
  std::string selected_npu_eid() const; // Empty until the startup probe passes.
  std::uint32_t selected_cpu_candidate() const; // kAutoCandidate until selected.
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
