#pragma once
#include "batch_get_abi.h"

enum KvafRawReadInputMode : uint32_t {
  KVAF_RAW_PROBE_ADDRESSES = 0, // internal startup probe: uint64 addresses + uint32 <=64KiB lengths
  KVAF_RAW_NPU_OFFSETS = 1,        // int64 offsets/lengths, split while forming WQEs
};
constexpr uint32_t KVAF_RAW_MAX_SPANS = 4096;
constexpr uint32_t KVAF_RAW_MAX_WQES = 1U << 20;

struct KvafExecuteBatchGetRawAddrKernelArgs {
  uint64_t source_addresses;       // absolute addresses or source offsets, according to input_mode
  uint64_t destination_addresses;  // absolute addresses or output-relative offsets
  uint64_t lengths;                // uint32 WQE lengths or int64 input span lengths
  uint64_t output;
  uint64_t core_status;            // KvafRawReadCoreStatus[block_count]
  uint64_t channel_entity;
  uint64_t queue_state;
  uint32_t request_count;          // WQE count in probe mode; input span count in offsets mode
  uint32_t block_count;
  uint32_t jettys_per_core;
  uint32_t batch_get_max_inflight_total;
  uint32_t input_mode;
  uint32_t reserved;
};
static_assert(sizeof(KvafExecuteBatchGetRawAddrKernelArgs) == 80);

// Returned only after the single READ launch finishes. Counts distinguish
// validation rejection (no WQEs posted) from an unresolved transport failure.
struct KvafRawReadCoreStatus {
  KvafBatchGetCoreStatus execution;
  uint64_t total_wqes;
  uint64_t posted_wqes;
  uint64_t reserved[6];
};
static_assert(sizeof(KvafRawReadCoreStatus) == 1280);
