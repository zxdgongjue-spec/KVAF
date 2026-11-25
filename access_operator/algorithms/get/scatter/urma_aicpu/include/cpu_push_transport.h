#pragma once

#include <array>
#include <memory>

#include "access_operator.h"
#include "aicpu_urma/launch_view.h"

namespace kvaf::cpu_push {
struct Config {
  std::shared_ptr<peer::PeerLink> peer_link;
  std::string kernel_json;
  std::array<std::uint8_t, 16> local_eid{};
  std::uint32_t token_bytes{}, workers{1};
  bool direct_output{false};
  std::uint32_t sge_per_wr{0};
  bool sgl_payload_only{false};
  // AIV uses the same CPU producer and registration lifecycle with its own
  // device kernel. kernel_json then names an AIV binary object.
  bool aiv{false};
};
class CpuPushTransport {
 public:
  explicit CpuPushTransport(Config);
  ~CpuPushTransport();
  Status Initialize(const RemoteRegionDesc&);
  Status Execute(const BatchGetExecRequest&);
  Status Prepare(const BatchGetExecRequest&, std::optional<bool> expected_capture = std::nullopt);
  Status ExecutePrepared(const BatchGetExecRequest&);
  aicpu_urma::ScatterLaunchView View() const;
  static Status LaunchPrepared(const BatchGetExecRequest&, const aicpu_urma::ScatterLaunchView&, ExecutionState*, bool aiv = false);
  Status Drain();
  bool direct_output() const;
  Status RegisterOutput(const MutableBufferView&);
  Status ReleaseOutput();
  Status PrepareGraph(const GetGraphOptions&);
  Status FinishGraph(void* stream, bool discard);
  std::unique_ptr<ExecutionState> CreateExecutionState();
  Status BindExecutionState(ExecutionState*);
  Status Check(void* stream);
  Status SetKernelProfiling(bool);
  Status CopyKernelProfiles(std::vector<BatchGetKernelProfile>*) const;
  Status Shutdown();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace kvaf::cpu_push
