#pragma once

#include "batch_get_abi.h"

// Shared eager status contract. Each mode owns its launch argument ABI.
enum {
  KVAF_SEQ_SCATTER = 1,
  KVAF_SEQ_GATHER = 2,
  KVAF_SEQ_MAX_REQUESTS = 65536,
  KVAF_SEQ_MAX_WORKERS = 8,
  KVAF_SEQ_MAX_SGES = 6,
};
static const uint64_t KVAF_SEQ_STATUS_MAGIC = 0x4b56414653455131ULL;
enum KvafSequentialResult : uint64_t {
  KVAF_SEQ_OK = 0, KVAF_SEQ_INVALID = 1, KVAF_SEQ_TIMEOUT = 2,
  KVAF_SEQ_REMOTE_ERROR = 3, KVAF_SEQ_CQ_ERROR = 4, KVAF_SEQ_QUEUE_ERROR = 5,
  // A diagnostic prefix finished; never report this as a successful BatchGet.
  KVAF_SEQ_PROBE_COMPLETE = 6,
};
enum KvafSequentialPhase : uint64_t {
  KVAF_SEQ_VALIDATE = 1, KVAF_SEQ_BUILD = 2, KVAF_SEQ_NOTIFY = 3,
  KVAF_SEQ_WAIT_REMOTE = 4, KVAF_SEQ_PAYLOAD = 5, KVAF_SEQ_COMPLETE = 6,
  KVAF_SEQ_AFTER_DONE = 7, KVAF_SEQ_AFTER_FLUSH = 8, KVAF_SEQ_AFTER_LOAD = 9,
  KVAF_SEQ_AFTER_COPY = 10,
};
struct alignas(64) KvafSequentialStatus {
  uint64_t magic, sequence, result, phase, requests, data_wrs, sq_pi, cq_ci;
};
static_assert(sizeof(KvafSequentialStatus) == 64);
