#pragma once

#include "get_graph_report.h"
#include <acl/acl.h>
#include "access_operator.h"
#include "execution_view.h"
#include "gather_workspace.h"
#include "aicpu_urma/control_abi.h"
#include "aicpu_urma/session.h"

namespace kvaf::aicpu_urma {
struct PutPreparation final : ExecutionState {
  explicit PutPreparation(const void* value) : owner(value) {}
  const void* owner;
  const BatchPutRequest* request{nullptr};
  std::uint64_t region{0};
  const void* input{nullptr};
  std::size_t bytes{0}, rows{0};
};
struct PutLaunchView {
  ExecutionViewCommon common;
  UmdkWriteView transport;
  void* descriptors{};
  std::size_t capacity{};
  std::vector<UmdkTransfer>* host_descriptors{};
};

// Shared execution-only pieces. No MR, context, session owner or setup API.
struct AicpuKernelOptions {
  std::uint32_t workers{}, token_bytes{}, window_capacity{};
  std::uint32_t read_wave_wrs{kGatherDefaultReadWaveWrs};
};
struct AicpuBufferView { void* device{}; void* host{}; std::size_t bytes{}; };
struct AicpuGetInvocation final : ExecutionState {
  explicit AicpuGetInvocation(const void* value) : owner(value) {}
  const void* owner;
  aclrtStream pending{};
  std::uint64_t sequence{};
  aclrtStream captured_stream{}, last_stream{};
  bool prepared_capturing{false}, submitted{false}, profile_submitted{false};
  Status failure;
  bool profile_complete{false};
  bool graph_active{false};
  aclrtStream graph_stream{};
  std::uint64_t graph_start_sequence{};
  std::unique_ptr<GetGraphReport> graph_completion, graph_profile;
  const BatchGetRequest* prepared{};
  void* prepared_stream{};
  BatchGetProfile* prepared_profile{};
  std::uint64_t prepare_started{};
};
struct GatherLaunchView {
  ExecutionViewCommon common;
  UmdkContext context;
  AicpuKernelOptions options;
  bool direct_addresses{};
  // Registered CPU staging/descriptor/control storage owned by the session.
  // Source addresses still belong to common.region(), not this workspace.
  RemoteRegionDesc remote_workspace;
  void* workspace{};
  AicpuBufferView completion, profile;
  void* control{};
  bool profiling{false};
};
struct ScatterLaunchView {
  ExecutionViewCommon common;
  AicpuKernelOptions options;
  void* workspace{};
  void* coordination{};
  AicpuBufferView completion, profile;
  ExecutionHealth* health{};
  bool profiling{false};
};
}  // namespace kvaf::aicpu_urma
