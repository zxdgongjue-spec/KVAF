#include "kvaf_aiv_batch_get.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

static int launches = 0;
static KvafExecuteBatchGetKernelArgs captured{};
aclrtBinary aclrtCreateBinary(const void*, size_t) { return reinterpret_cast<void*>(1); }
int aclrtBinaryLoad(aclrtBinary binary, aclrtBinHandle* result) { *result = binary; return 0; }
int aclrtDestroyBinary(aclrtBinary) { return 0; }
int aclrtBinaryGetFunction(aclrtBinHandle binary, const char* name, aclrtFuncHandle* result) {
  CHECK(std::strcmp(name, "KvafExecuteBatchGetAivKernelV5") == 0);
  *result = binary;
  return 0;
}
int aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, uint32_t cores, aclrtStream stream,
                                 const aclrtLaunchKernelCfg* cfg, void* args, size_t bytes,
                                 void*, size_t) {
  CHECK(cores == 2 && stream != nullptr && bytes == 128);
  CHECK(cfg->numAttrs == 4);
  captured = *static_cast<KvafExecuteBatchGetKernelArgs*>(args);
  ++launches;
  return 0;
}

int main() {
  // The fake binary API accepts any bytes, while the production file/loader
  // path and versioned-symbol lookup are exercised unchanged.
  CHECK(KvafExecuteBatchGetLoadKernel(__FILE__) == KVAF_BATCH_GET_SUCCESS);
  KvafExecuteBatchGetKernelArgs raw{};
  raw.remote_addresses = 0x1000;
  raw.local_addresses = 0x2000;
  raw.miss_count = 0x3000;
  raw.output = 0x4000;
  raw.core_status = 0x5000;
  raw.channel_entity = 0x6000;
  raw.queue_state = 0x7000;
  raw.request_capacity = 257;
  raw.token_bytes = 7;
  raw.block_count = 2;
  raw.jettys_per_core = 3;
  raw.batch_get_max_inflight_total = 2048;
  const auto stream = reinterpret_cast<void*>(0x8000);
  CHECK(KvafExecuteBatchGetLaunch(&raw, stream) == KVAF_BATCH_GET_SUCCESS);
  CHECK(captured.remote_addresses == raw.remote_addresses);
  CHECK(captured.local_addresses == raw.local_addresses);
  const auto reject = [&](const KvafExecuteBatchGetKernelArgs& args) {
    CHECK(KvafExecuteBatchGetLaunch(&args, stream) == KVAF_BATCH_GET_INVALID_ARGUMENT);
    CHECK(launches == 1);
  };
  auto bad = raw; bad.local_addresses = 0; reject(bad);
  bad = raw; bad.remote_addresses = 0; reject(bad);
  bad = raw; bad.address_table = 0x9000; reject(bad);
  bad = raw; bad.tokens_per_block = 1; reject(bad);
  bad = raw; bad.output_rows = 1; reject(bad);
  bad = raw; bad.miss_count = 0; reject(bad);
  bad = raw; bad.channel_entity = 0; reject(bad);
  bad = raw; bad.queue_state = 0; reject(bad);
  bad = raw; bad.core_status = 0; reject(bad);
  bad = raw; bad.output = 0; reject(bad);
  bad = raw; bad.batch_get_max_inflight_total = 3; reject(bad);
  CHECK(KvafExecuteBatchGetLaunch(&raw, nullptr) == KVAF_BATCH_GET_INVALID_ARGUMENT);
  auto lookup = raw;
  lookup.remote_addresses = lookup.local_addresses = 0;
  lookup.address_table = lookup.valid_table = lookup.indices = lookup.offsets = 0x9000;
  lookup.output_indices = 0xa000;
  lookup.table_capacity = lookup.output_rows = lookup.tokens_per_block = 1;
  CHECK(KvafExecuteBatchGetLaunch(&lookup, stream) == KVAF_BATCH_GET_SUCCESS);
  CHECK(launches == 2 && captured.remote_addresses == 0);
  std::cout << "BatchGet V5 launcher raw/lookup ABI and rejection checks passed\n";
}
