#pragma once
#include "runtime.h"

namespace kvaf {
struct OffloadRuntimeOptions {
  std::uint32_t device_id{0};
  std::string npu_eid; // Explicit 16-byte EID in hexadecimal, optional ':' separators.
  std::uint32_t cpu_candidate{0};
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
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
