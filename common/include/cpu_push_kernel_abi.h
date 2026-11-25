#pragma once
#include "cpu_push_get_abi.h"

namespace kvaf::cpu_push {
struct BatchGetInput {
  // GatherRead with indices == 0 reads address_table[i] as a complete source
  // address; offsets is unused. Other callers use address_table[indices[i]] + offsets[i].
  // Gather/AICPU preserve their caller-validated input contract. AIV SGL
  // validates source bounds and identity output rows before notifying CPU.
  std::uint64_t address_table{}, indices{}, offsets{}, miss_count{}, output_indices{};
  std::uint32_t table_capacity{}, request_capacity{}, token_bytes{};
  std::uint64_t valid_table{}, canonical_base{}, logical_bytes{}, block_bytes{};
};
struct KernelArgs {
  BatchGetInput request;
  std::uint64_t workspace{}, output{}, output_bytes{}, coordination{}, status{}, profile{};
  std::uint64_t sequence{};  // Assigned in each instance's local copy after claiming a ticket.
  std::uint32_t workers{}, timeout_ms{kTimeoutMs};
};
// Keep the Scatter V3 launch ABI unchanged. The direct-write entry owns its
// additional exported-output argument.
struct CpuSglWriteArgs {
  KernelArgs execution;
  OutputTarget output_target;
};
static_assert(sizeof(BatchGetInput) == 88 && sizeof(KernelArgs) == 152 && sizeof(CpuSglWriteArgs) == 320);
}  // namespace kvaf::cpu_push
