#pragma once
#include "access_operator.h"
#include "sequential_batch_get_abi.h"

namespace kvaf::sequential_detail {
inline uint64_t Ptr(const void* p) { return reinterpret_cast<uint64_t>(p); }
inline void FillInput(const BatchGetRequest& r, const AccessOperatorConfig& config,
                      KvafExecuteBatchGetKernelArgs* a) {
  const auto& table = r.device_access_table;
  *a = {};
  a->address_table = Ptr(table.addresses.data); a->valid_table = Ptr(table.valid.data);
  a->indices = Ptr(r.indices.data); a->offsets = Ptr(r.offsets.data);
  a->miss_count = Ptr(r.miss_count.data); a->output_indices = Ptr(r.output_indices.data);
  a->output = Ptr(r.output.data); a->table_capacity = table.addresses.bytes / 8;
  a->request_capacity = r.indices.bytes / 4;
  a->output_rows = r.output.bytes / config.batch_get_token_bytes;
  a->token_bytes = config.batch_get_token_bytes; a->tokens_per_block = r.tokens_per_block;
  a->block_count = config.npu_core_count; a->jettys_per_core = 2;
}
inline Status Input(const BatchGetExecRequest& exec, const AccessOperatorConfig& config,
             KvafExecuteBatchGetKernelArgs* a) {
  if (!exec.request || !exec.execution.stream) return Status::InvalidArgument("BatchGet needs request and stream");
  const auto& r = *exec.request;
  const auto& table = r.device_access_table;
  const uint64_t count = r.indices.bytes / 4;
  const uint64_t entries = table.addresses.bytes / 8;
  if (!r.indices.has_elements<int32_t>() || !r.offsets.has_elements<int32_t>() ||
      !r.output_indices.has_elements<int32_t>() || !r.miss_count.has_elements<int32_t>() ||
      r.miss_count.bytes != 4 || r.indices.bytes != r.offsets.bytes ||
      r.indices.bytes != r.output_indices.bytes || !r.output.valid() ||
      !table.addresses.has_elements<uint64_t>() || !table.valid.valid() || table.valid.bytes != entries ||
      r.indices.device != DeviceKind::kNpu || r.offsets.device != DeviceKind::kNpu ||
      r.output_indices.device != DeviceKind::kNpu || r.miss_count.device != DeviceKind::kNpu ||
      r.output.device != DeviceKind::kNpu || table.addresses.device != DeviceKind::kNpu ||
      table.valid.device != DeviceKind::kNpu || count > KVAF_SEQ_MAX_REQUESTS || entries > UINT32_MAX ||
      !r.output_block_bytes || r.output_block_bytes != config.batch_get_token_bytes ||
      !r.tokens_per_block || r.tokens_per_block > UINT32_MAX ||
      r.output.bytes % r.output_block_bytes || r.output.bytes / r.output_block_bytes > UINT32_MAX ||
      Ptr(r.output.data) > UINT64_MAX - r.output.bytes)
    return Status::InvalidArgument("invalid sequential BatchGet Tensor layout/capacity");
  FillInput(r, config, a);
  // Intentionally never copy the direct-read inflight budget into this ABI.
  return Status::OK();
}
}
