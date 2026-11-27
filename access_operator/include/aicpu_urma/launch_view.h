#pragma once

#include "get_graph_report.h"
#include <acl/acl.h>
#include "access_operator.h"
#include "execution_view.h"
#include "aicpu_urma/control_abi.h"
#include "aicpu_urma/session.h"
#include "cpu_push_get_abi.h"
#include "cpu_sgl_push_abi.h"

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
struct AicpuKernelOptions { std::uint32_t workers{}, token_bytes{}; };
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
  bool direct_output{false};
  cpu_push::OutputTarget output_target;
  bool sgl_pipeline{false};
  bool device_push{false};
  cpu_push::MailboxLaunch mailbox;
  bool sgl_push_chunked{false};
  std::uint32_t sgl_push_head_rows{1024};
  bool sgl_push_cooperative{false};
  bool sgl_push_publisher_only{false};
  bool sgl_push_head_cooperative{false};
};
}  // namespace kvaf::aicpu_urma
