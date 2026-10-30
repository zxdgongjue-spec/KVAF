#pragma once
#include "batch_get_abi.h"

// No lookup table: each request supplies two absolute addresses and a byte
// count. The host splits long spans into <=64 KiB READs before submission.
struct KvafExecuteBatchGetRawAddrKernelArgs {
  uint64_t source_addresses;       // uint64[request_count]
  uint64_t destination_addresses;  // uint64[request_count]
  uint64_t lengths;                // uint32[request_count]
  uint64_t output;                 // registered caller-owned output allocation
  uint64_t core_status;
  uint64_t channel_entity;
  uint64_t queue_state;
  uint32_t request_count;
  uint32_t block_count;
  uint32_t jettys_per_core;
  uint32_t batch_get_max_inflight_total;
};
static_assert(sizeof(KvafExecuteBatchGetRawAddrKernelArgs) == 72);
