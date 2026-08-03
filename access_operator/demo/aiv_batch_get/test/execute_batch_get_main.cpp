#include "kvaf_aiv_batch_get.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_mdl.h"

namespace {

struct Options {
  uint32_t device = 0U;
  uint32_t table_capacity = 4096U;
  uint32_t requests = 4096U;
  uint32_t output_rows = 4096U;
  uint32_t block_bytes = 1024U;
  uint32_t source_stride = 4096U;
  uint32_t aiv_cores = 4U;
  uint32_t warmup = 10U;
  uint32_t iterations = 50U;
  uint32_t mask_every = 0U;
  // Zero selects the normal launch/synchronize test.  A nonzero value
  // captures the ExecuteBatchGet launch once, then replays the captured model
  // this many times after changing the contents (not addresses) of the fixed
  // device input tensors.
  uint32_t graph_replays = 0U;
  uint32_t graph_warmup = 5U;
  uint32_t graph_iterations = 20U;
  // Optional functional preflight: use a second runtime output allocation on
  // the same transport, then return to the primary benchmark output.
  uint32_t dynamic_output_smoke = 0U;
  std::string kernel;
};

uint8_t PatternByte(uint32_t slot, uint32_t byte) {
  return static_cast<uint8_t>((slot * 131U + byte * 17U + (byte >> 3U)) & 0xffU);
}

bool ParseU32(const char *text, uint32_t *value) {
  if (text == nullptr || *text == '\0' || value == nullptr) return false;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 0);
  if (end == nullptr || *end != '\0' || parsed > UINT32_MAX) return false;
  *value = static_cast<uint32_t>(parsed);
  return true;
}

bool ReadEnvU32(const char *name, uint32_t *value) {
  const char *text = std::getenv(name);
  return text == nullptr || *text == '\0' || ParseU32(text, value);
}

bool ParseOptions(int argc, char **argv, Options *options) {
  if (options == nullptr) return false;
  for (int index = 1; index < argc; ++index) {
    const std::string arg(argv[index]);
    if (arg == "--kernel" && index + 1 < argc) {
      options->kernel = argv[++index];
    } else if (arg == "--device" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->device)) return false;
    } else if (arg == "--table-capacity" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->table_capacity)) return false;
    } else if (arg == "--requests" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->requests)) return false;
    } else if (arg == "--output-rows" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->output_rows)) return false;
    } else if (arg == "--block-bytes" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->block_bytes)) return false;
    } else if (arg == "--source-stride" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->source_stride)) return false;
    } else if (arg == "--aiv-cores" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->aiv_cores)) return false;
    } else if (arg == "--warmup" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->warmup)) return false;
    } else if (arg == "--iterations" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->iterations)) return false;
    } else if (arg == "--mask-every" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->mask_every)) return false;
    } else if (arg == "--graph-replays" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->graph_replays)) return false;
    } else if (arg == "--graph-warmup" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->graph_warmup)) return false;
    } else if (arg == "--graph-iterations" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->graph_iterations)) return false;
    } else if (arg == "--dynamic-output-smoke" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->dynamic_output_smoke)) return false;
    } else {
      return false;
    }
  }
  return !options->kernel.empty() && ReadEnvU32("DEVICE_ID", &options->device) &&
         ReadEnvU32("TABLE_CAPACITY", &options->table_capacity) && ReadEnvU32("REQUESTS", &options->requests) &&
         ReadEnvU32("OUTPUT_ROWS", &options->output_rows) && ReadEnvU32("BLOCK_BYTES", &options->block_bytes) &&
         ReadEnvU32("SOURCE_STRIDE", &options->source_stride) && ReadEnvU32("AIV_CORES", &options->aiv_cores) &&
         ReadEnvU32("WARMUP", &options->warmup) && ReadEnvU32("ITERATIONS", &options->iterations) &&
         ReadEnvU32("MASK_EVERY", &options->mask_every) &&
         ReadEnvU32("DYNAMIC_OUTPUT_SMOKE", &options->dynamic_output_smoke) && options->table_capacity != 0U &&
         options->requests != 0U && options->requests <= options->output_rows && options->output_rows != 0U &&
         options->block_bytes != 0U && options->source_stride >= options->block_bytes &&
         options->aiv_cores != 0U && options->aiv_cores <= KVAF_AIV_BATCH_GET_MAX_AIV_CORES &&
         options->aiv_cores <= options->requests && options->iterations != 0U &&
         options->graph_iterations != 0U && options->dynamic_output_smoke <= 1U;
}

bool CheckAcl(aclError error, const char *step) {
  if (error == ACL_SUCCESS) return true;
  std::cerr << "[KVAF_EXECUTE] " << step << " ACL error=" << error << '\n';
  return false;
}

bool CheckKvaf(int32_t error, const char *step) {
  if (error == KVAF_BATCH_GET_SUCCESS) return true;
  std::cerr << "[KVAF_EXECUTE] " << step << " error=" << error << '\n';
  return false;
}

double Percentile(std::vector<double> samples, double fraction) {
  std::sort(samples.begin(), samples.end());
  const double position = fraction * static_cast<double>(samples.size() - 1U);
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1U, samples.size() - 1U);
  return samples[lower] + (samples[upper] - samples[lower]) * (position - lower);
}

bool Validate(const std::vector<uint8_t> &output, const std::vector<uint64_t> &address_table,
              const std::vector<int32_t> &indices, const std::vector<uint8_t> &mask,
              const std::vector<int32_t> &output_indices, const Options &options,
              const uint8_t *source, const std::vector<KvafBatchGetCoreStatus> &statuses,
              const std::vector<KvafBatchGetQueueState> &queues, uint64_t expected_batches,
              const char *stage) {
  uint64_t active_total = 0U;
  for (uint32_t request = 0U; request < options.requests; ++request) {
    if (mask[request] == 0U) {
      for (uint32_t byte = 0U; byte < options.block_bytes; ++byte) {
        const uint8_t actual = output[static_cast<uint64_t>(output_indices[request]) * options.block_bytes + byte];
        if (actual != 0U) {
          std::cerr << "[KVAF_EXECUTE] masked row modified stage=" << stage << " request=" << request
                    << " output_row=" << output_indices[request] << " byte=" << byte << " actual="
                    << static_cast<uint32_t>(actual) << '\n';
          return false;
        }
      }
      continue;
    }
    ++active_total;
    const uint64_t source_offset = address_table[static_cast<uint32_t>(indices[request])] -
                                   reinterpret_cast<uint64_t>(source);
    for (uint32_t byte = 0U; byte < options.block_bytes; ++byte) {
      const uint8_t actual = output[static_cast<uint64_t>(output_indices[request]) * options.block_bytes + byte];
      const uint8_t expected = source[source_offset + byte];
      if (actual != expected) {
        std::cerr << "[KVAF_EXECUTE] validation mismatch stage=" << stage << " request=" << request
                  << " source_uba=0x" << std::hex << address_table[static_cast<uint32_t>(indices[request])]
                  << std::dec << " output_row=" << output_indices[request] << " byte=" << byte
                  << " expected=" << static_cast<uint32_t>(expected) << " actual="
                  << static_cast<uint32_t>(actual) << '\n';
        return false;
      }
    }
  }
  for (uint32_t core = 0U; core < options.aiv_cores; ++core) {
    const uint32_t begin = static_cast<uint32_t>((static_cast<uint64_t>(options.requests) * core) / options.aiv_cores);
    const uint32_t end = static_cast<uint32_t>((static_cast<uint64_t>(options.requests) * (core + 1U)) /
                                               options.aiv_cores);
    uint64_t active = 0U;
    for (uint32_t request = begin; request < end; ++request) active += mask[request] != 0U;
    const auto &status = statuses[core];
    const auto &queue = queues[core];
    const uint64_t expected_core_batches = active == 0U ? 0U : expected_batches;
    if (status.magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || status.result != 0U ||
        status.processed_request_count != active || status.completed_batch_count != expected_core_batches ||
        queue.magic != 0x4b564151U || queue.initialized != 1U || queue.last_error != 0U ||
        queue.completed_batch_count != expected_core_batches) {
      std::cerr << "[KVAF_EXECUTE] kernel state mismatch stage=" << stage << " core=" << core
                << " result=" << status.result << " processed=" << status.processed_request_count
                << " active=" << active << " completed=" << status.completed_batch_count
                << " expected_batches=" << expected_core_batches << " queue_error=" << queue.last_error
                << " failed_request=" << status.failed_request_index << '\n';
      return false;
    }
  }
  std::cout << "EXECUTE_BATCH_GET_VALIDATION stage=" << stage << " active_requests=" << active_total
            << " masked_requests=" << (options.requests - active_total) << " table_mode=absolute_host_uba"
            << " output_layout=scatter_rows\n";
  return true;
}

// This deliberately changes only data inside already-allocated device
// tensors.  It is the graph contract: addresses/shape/args stay fixed across
// replay, while a new AccessTable epoch resolves the same logical indices to
// different registered Host-DDR blocks.
void PrepareGraphReplayInputs(uint32_t replay, const uint8_t *source, const Options &options,
                              std::vector<uint64_t> *address_table, std::vector<int32_t> *indices,
                              std::vector<uint8_t> *mask, std::vector<int32_t> *output_indices) {
  const uint32_t address_shift = replay % options.table_capacity;
  for (uint32_t entry = 0U; entry < options.table_capacity; ++entry) {
    const uint32_t source_slot = (entry + address_shift) % options.table_capacity;
    (*address_table)[entry] = reinterpret_cast<uint64_t>(source) +
                              static_cast<uint64_t>(source_slot) * options.source_stride;
  }
  for (uint32_t request = 0U; request < options.requests; ++request) {
    (*indices)[request] = static_cast<int32_t>(request % options.table_capacity);
    (*mask)[request] = 1U;
    (*output_indices)[request] = static_cast<int32_t>((request + replay) % options.output_rows);
  }
}

bool CopyGraphReplayInputs(aclrtStream stream, void *device_address_table, void *device_indices, void *device_mask,
                           void *device_output_indices, const std::vector<uint64_t> &address_table,
                           const std::vector<int32_t> &indices, const std::vector<uint8_t> &mask,
                           const std::vector<int32_t> &output_indices) {
  return CheckAcl(aclrtMemcpyAsync(device_address_table, address_table.size() * sizeof(uint64_t),
                                   address_table.data(), address_table.size() * sizeof(uint64_t),
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream),
                  "graph copy address_table") &&
         CheckAcl(aclrtMemcpyAsync(device_indices, indices.size() * sizeof(int32_t), indices.data(),
                                   indices.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream),
                  "graph copy indices") &&
         CheckAcl(aclrtMemcpyAsync(device_mask, mask.size(), mask.data(), mask.size(), ACL_MEMCPY_HOST_TO_DEVICE,
                                   stream),
                  "graph copy mask") &&
         CheckAcl(aclrtMemcpyAsync(device_output_indices, output_indices.size() * sizeof(int32_t),
                                   output_indices.data(), output_indices.size() * sizeof(int32_t),
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream),
                  "graph copy output_indices");
}

}  // namespace

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    std::cerr << "usage: kvaf_execute_batch_get_test --kernel <kvaf_execute_batch_get_kernels.o> "
                 "[--device N --table-capacity N --requests N --output-rows N --block-bytes N "
                 "--source-stride N --aiv-cores N --warmup N --iterations N --mask-every N "
                 "--graph-replays N --graph-warmup N --graph-iterations N --dynamic-output-smoke 0|1]\n";
    return 2;
  }
  const uint64_t source_bytes = static_cast<uint64_t>(options.table_capacity) * options.source_stride;
  const uint64_t output_bytes = static_cast<uint64_t>(options.output_rows) * options.block_bytes;
  if (source_bytes > std::numeric_limits<size_t>::max() ||
      output_bytes > std::numeric_limits<size_t>::max()) return 2;

  int exit_code = 1;
  uint32_t device_count = 0U;
  aclrtStream stream = nullptr;
  void *source_ddr = nullptr;
  void *output_hbm = nullptr;
  void *alternate_output_hbm = nullptr;
  void *device_address_table = nullptr;
  void *device_valid_table = nullptr;
  void *device_indices = nullptr;
  void *device_mask = nullptr;
  void *device_output_indices = nullptr;
  void *device_status = nullptr;
  void *device_queue = nullptr;
  KvafBatchGetDdrRegistration *registration = nullptr;
  aclmdlRI graph_model = nullptr;
  uint64_t entity_table = 0U;
  uint64_t local_hbm_binding = 0U;
  uint64_t alternate_local_hbm_binding = 0U;
  uint64_t completed_batch_base = 0U;
  uint8_t *source = nullptr;
  KvafExecuteBatchGetKernelArgs args{};
  std::vector<uint64_t> address_table(options.table_capacity);
  std::vector<uint8_t> valid_table(options.table_capacity, 1U);
  std::vector<int32_t> indices(options.requests);
  std::vector<uint8_t> mask(options.requests, 1U);
  std::vector<int32_t> output_indices(options.requests);
  std::vector<KvafBatchGetCoreStatus> statuses(options.aiv_cores);
  std::vector<KvafBatchGetQueueState> queues(options.aiv_cores);
  std::vector<uint8_t> output(output_bytes);
  std::vector<double> elapsed_ms;

  std::cout << "PHASE_PROCESS_READY\n";
  std::cout << "PHASE_RUNTIME_STEP_BEGIN step=aclInit\n";
  if (!CheckAcl(aclInit(nullptr), "aclInit")) goto cleanup;
  std::cout << "PHASE_RUNTIME_STEP_DONE step=aclInit\n";
  std::cout << "PHASE_RUNTIME_STEP_BEGIN step=aclrtGetDeviceCount\n";
  if (!CheckAcl(aclrtGetDeviceCount(&device_count), "aclrtGetDeviceCount") || device_count == 0U ||
      options.device >= device_count) {
    std::cerr << "[KVAF_EXECUTE] requested device=" << options.device
              << " available_devices=" << device_count << '\n';
    goto cleanup;
  }
  std::cout << "PHASE_RUNTIME_STEP_DONE step=aclrtGetDeviceCount count=" << device_count << '\n';
  std::cout << "PHASE_RUNTIME_STEP_BEGIN step=aclrtSetDevice\n";
  if (!CheckAcl(aclrtSetDevice(options.device), "aclrtSetDevice")) goto cleanup;
  std::cout << "PHASE_RUNTIME_STEP_DONE step=aclrtSetDevice\n";
  std::cout << "PHASE_RUNTIME_STEP_BEGIN step=aclrtCreateStream\n";
  if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) goto cleanup;
  std::cout << "PHASE_RUNTIME_STEP_DONE step=aclrtCreateStream\n";
  std::cout << "PHASE_RUNTIME_READY device=" << options.device << '\n';
  if (!CheckAcl(aclrtMallocHost(&source_ddr, source_bytes), "aclrtMallocHost(source DDR)")) goto cleanup;
  source = static_cast<uint8_t *>(source_ddr);
  for (uint32_t slot = 0U; slot < options.table_capacity; ++slot) {
    uint8_t *block = source + static_cast<uint64_t>(slot) * options.source_stride;
    for (uint32_t byte = 0U; byte < options.block_bytes; ++byte) block[byte] = PatternByte(slot, byte);
    address_table[slot] = reinterpret_cast<uint64_t>(block);
  }
  for (uint32_t request = 0U; request < options.requests; ++request) {
    indices[request] = static_cast<int32_t>(options.table_capacity - 1U - (request % options.table_capacity));
    output_indices[request] = static_cast<int32_t>(options.output_rows - 1U - request);
    if (options.mask_every != 0U && request % options.mask_every == 0U) mask[request] = 0U;
  }
  // Match the architecture note's concrete [200,1024] scatter example when
  // that shape is requested; the normal default remains a larger reverse
  // permutation for the 4096-block throughput run.
  if (options.requests == 10U && options.table_capacity > 10U && options.output_rows >= 200U) {
    constexpr int32_t kExampleRows[10] = {1, 10, 15, 11, 199, 150, 70, 88, 65, 120};
    for (uint32_t request = 0U; request < 10U; ++request) {
      indices[request] = static_cast<int32_t>(request + 1U);
      output_indices[request] = kExampleRows[request];
    }
  }
  if (!CheckAcl(aclrtMalloc(&output_hbm, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate output HBM") ||
      !CheckAcl(aclrtMalloc(&device_address_table, address_table.size() * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate address_table") ||
      !CheckAcl(aclrtMalloc(&device_valid_table, valid_table.size(), ACL_MEM_MALLOC_HUGE_FIRST), "allocate valid_table") ||
      !CheckAcl(aclrtMalloc(&device_indices, indices.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "allocate indices") ||
      !CheckAcl(aclrtMalloc(&device_mask, mask.size(), ACL_MEM_MALLOC_HUGE_FIRST), "allocate mask") ||
      !CheckAcl(aclrtMalloc(&device_output_indices, output_indices.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate output_indices") ||
      !CheckAcl(aclrtMalloc(&device_status, statuses.size() * sizeof(statuses[0]), ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate status") ||
      !CheckAcl(aclrtMalloc(&device_queue, queues.size() * sizeof(queues[0]), ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate queue state") ||
      !CheckAcl(aclrtMemcpy(device_address_table, address_table.size() * sizeof(uint64_t), address_table.data(),
                             address_table.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy address_table") ||
      !CheckAcl(aclrtMemcpy(device_valid_table, valid_table.size(), valid_table.data(), valid_table.size(),
                             ACL_MEMCPY_HOST_TO_DEVICE), "copy valid_table") ||
      !CheckAcl(aclrtMemcpy(device_indices, indices.size() * sizeof(int32_t), indices.data(),
                             indices.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy indices") ||
      !CheckAcl(aclrtMemcpy(device_mask, mask.size(), mask.data(), mask.size(), ACL_MEMCPY_HOST_TO_DEVICE), "copy mask") ||
      !CheckAcl(aclrtMemcpy(device_output_indices, output_indices.size() * sizeof(int32_t), output_indices.data(),
                             output_indices.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy output_indices") ||
      !CheckAcl(aclrtMemset(output_hbm, output_bytes, 0, output_bytes), "clear output") ||
      !CheckAcl(aclrtMemset(device_status, statuses.size() * sizeof(statuses[0]), 0, statuses.size() * sizeof(statuses[0])),
                "clear status") ||
      !CheckAcl(aclrtMemcpy(device_queue, queues.size() * sizeof(queues[0]), queues.data(),
                             queues.size() * sizeof(queues[0]), ACL_MEMCPY_HOST_TO_DEVICE), "initialize queue")) goto cleanup;

  if (options.dynamic_output_smoke != 0U &&
      (!CheckAcl(aclrtMalloc(&alternate_output_hbm, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                 "allocate alternate output HBM") ||
       !CheckAcl(aclrtMemset(alternate_output_hbm, output_bytes, 0, output_bytes), "clear alternate output"))) {
    goto cleanup;
  }

  if (!CheckKvaf(KvafBatchGetInitializeTransport(source_ddr, source_bytes, options.aiv_cores, 1U,
                                                  &registration, &entity_table),
                 "initialize Direct-HCCP transport and register Host DDR")) goto cleanup;
  if (!CheckKvaf(KvafBatchGetPrepareLocalHbm(registration, output_hbm, output_bytes, &local_hbm_binding),
                 "prepare runtime output HBM for UDMA")) goto cleanup;
  std::cout << "PHASE_EID_ENDPOINT_READY entity_table=0x" << std::hex << entity_table << std::dec
            << " table_capacity=" << options.table_capacity << " output_rows=" << options.output_rows
            << " source_bytes=" << source_bytes << " output_bytes=" << output_bytes
            << " local_hbm_binding=0x" << std::hex << local_hbm_binding << std::dec
            << " output_lifetime=runtime_prepared\n";
  if (!CheckKvaf(KvafExecuteBatchGetLoadKernel(options.kernel.c_str()), "load ExecuteBatchGet kernel")) goto cleanup;
  std::cout << "PHASE_KERNEL_REGISTERED cores=" << options.aiv_cores
            << " contract=address_table_absolute_uba_plus_mask_plus_output_scatter\n";

  args.address_table = reinterpret_cast<uint64_t>(device_address_table);
  args.valid_table = reinterpret_cast<uint64_t>(device_valid_table);
  args.indices = reinterpret_cast<uint64_t>(device_indices);
  args.mask = reinterpret_cast<uint64_t>(device_mask);
  args.output_indices = reinterpret_cast<uint64_t>(device_output_indices);
  args.local_hbm_binding = local_hbm_binding;
  args.core_status = reinterpret_cast<uint64_t>(device_status);
  args.channel_entity = entity_table;
  args.queue_state = reinterpret_cast<uint64_t>(device_queue);
  args.table_capacity = options.table_capacity;
  args.request_capacity = options.requests;
  args.output_rows = options.output_rows;
  args.block_bytes = options.block_bytes;
  args.block_count = options.aiv_cores;

  if (options.dynamic_output_smoke != 0U) {
    if (!CheckKvaf(KvafBatchGetPrepareLocalHbm(registration, alternate_output_hbm, output_bytes,
                                                &alternate_local_hbm_binding),
                   "prepare alternate runtime output HBM") ||
        alternate_local_hbm_binding == local_hbm_binding) {
      std::cerr << "[KVAF_EXECUTE] dynamic output registration did not produce an independent binding\n";
      goto cleanup;
    }
    args.local_hbm_binding = alternate_local_hbm_binding;
    if (!CheckKvaf(KvafExecuteBatchGetLaunch(&args, stream), "dynamic output smoke launch") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "dynamic output smoke completion") ||
        !CheckAcl(aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses[0]), device_status,
                               statuses.size() * sizeof(statuses[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy dynamic output status") ||
        !CheckAcl(aclrtMemcpy(queues.data(), queues.size() * sizeof(queues[0]), device_queue,
                               queues.size() * sizeof(queues[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy dynamic output queue") ||
        !CheckAcl(aclrtMemcpy(output.data(), output.size(), alternate_output_hbm, output.size(),
                               ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy dynamic output") ||
        !Validate(output, address_table, indices, mask, output_indices, options, source, statuses, queues, 1U,
                  "dynamic_output")) {
      goto cleanup;
    }
    completed_batch_base = 1U;
    args.local_hbm_binding = local_hbm_binding;
    std::cout << "PHASE_DYNAMIC_OUTPUT_VALIDATION_PASS primary_binding=0x" << std::hex << local_hbm_binding
              << " alternate_binding=0x" << alternate_local_hbm_binding << std::dec
              << " transport_reused=1\n";
  }

  if (options.graph_replays != 0U) {
    // Resources, tensor allocations, fixed kernel arguments and the direct
    // HCCP registration are intentionally complete before capture.  The only
    // per-replay updates below are writes into the already-fixed device
    // tensors, just as an upstream graph node would do.
    std::cout << "GRAPH_CAPTURE_CONTRACT api=aclmdlRI fixed_tensor_addresses=1 "
                 "dynamic_device_contents=address_table_indices_mask_output_indices "
                 "replays="
              << options.graph_replays << " aiv_cores=" << options.aiv_cores
              << " perf_warmup=" << options.graph_warmup
              << " perf_samples=" << options.graph_iterations << '\n';
    if (!CheckAcl(aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_RELAXED),
                  "begin ExecuteBatchGet graph capture")) {
      goto cleanup;
    }
    std::cout << "GRAPH_CAPTURE_BEGIN\n";
    if (!CheckKvaf(KvafExecuteBatchGetLaunch(&args, stream), "capture ExecuteBatchGet launch")) {
      // Closing a failed capture keeps the stream lifecycle diagnosable and
      // avoids hiding the launch failure behind stream-destroy side effects.
      (void)aclmdlRICaptureEnd(stream, &graph_model);
      std::cerr << "[KVAF_EXECUTE] GRAPH_CAPTURE_FAILED stage=kernel_launch\n";
      goto cleanup;
    }
    if (!CheckAcl(aclmdlRICaptureEnd(stream, &graph_model), "end ExecuteBatchGet graph capture") ||
        graph_model == nullptr) {
      std::cerr << "[KVAF_EXECUTE] GRAPH_CAPTURE_FAILED stage=kernel_launch_or_capture_end\n";
      goto cleanup;
    }
    std::cout << "GRAPH_CAPTURE_DONE model_ready=1\n";

    for (uint32_t replay = 0U; replay < options.graph_replays; ++replay) {
      PrepareGraphReplayInputs(replay, source, options, &address_table, &indices, &mask, &output_indices);
      if (!CopyGraphReplayInputs(stream, device_address_table, device_indices, device_mask, device_output_indices,
                                 address_table, indices, mask, output_indices)) {
        goto cleanup;
      }
      std::cout << "GRAPH_REPLAY_INPUT_UPDATED replay=" << replay
                << " address_table_epoch=" << replay << " fixed_tensor_addresses=1\n";
      if (!CheckAcl(aclmdlRIExecuteAsync(graph_model, stream), "execute captured ExecuteBatchGet graph") ||
          !CheckAcl(aclrtSynchronizeStream(stream), "captured graph completion") ||
          !CheckAcl(aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses[0]), device_status,
                                 statuses.size() * sizeof(statuses[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                    "copy graph status") ||
          !CheckAcl(aclrtMemcpy(queues.data(), queues.size() * sizeof(queues[0]), device_queue,
                                 queues.size() * sizeof(queues[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                    "copy graph queue") ||
          !CheckAcl(aclrtMemcpy(output.data(), output.size(), output_hbm, output.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                    "copy graph output") ||
          !Validate(output, address_table, indices, mask, output_indices, options, source, statuses, queues,
                    completed_batch_base + static_cast<uint64_t>(replay) + 1U, "graph_replay")) {
        goto cleanup;
      }
      std::cout << "GRAPH_REPLAY_VALIDATION_PASS replay=" << replay
                << " captured_kernel=KvafExecuteBatchGetAivKernel\n";
    }
    std::cout << "GRAPH_REPLAY_PASS replays=" << options.graph_replays
              << " graph_scope=execute_batch_get_data_plane\n";

    // The verified graph contract is an update of fixed-address lookup
    // tensors followed by graph execution on the same stream.  Do not issue a
    // naked replay here: A5 passed all update+replay validation rounds but a
    // following replay without any producer stream work can stall.  The timed
    // interval is therefore the real graph step from H2D lookup update through
    // captured-kernel completion; capture, HCCP setup and final D2H validation
    // remain outside the measurement.
    std::cout << "GRAPH_PERF_CONTRACT metric=graph_input_update_to_stream_completion "
                 "capture_excluded=1 input_updates_included=1 validation_excluded=1 warmup="
              << options.graph_warmup << " samples=" << options.graph_iterations << '\n';
    std::cout << "GRAPH_PERF_WARMUP_BEGIN\n";
    for (uint32_t warmup = 0U; warmup < options.graph_warmup; ++warmup) {
      const uint32_t epoch = options.graph_replays + warmup;
      std::cout << "GRAPH_PERF_WARMUP_STEP_BEGIN iteration=" << warmup << " epoch=" << epoch << '\n';
      PrepareGraphReplayInputs(epoch, source, options, &address_table, &indices, &mask, &output_indices);
      if (!CopyGraphReplayInputs(stream, device_address_table, device_indices, device_mask, device_output_indices,
                                 address_table, indices, mask, output_indices)) goto cleanup;
      std::cout << "GRAPH_PERF_WARMUP_INPUT_ENQUEUED iteration=" << warmup << '\n';
      if (!CheckAcl(aclmdlRIExecuteAsync(graph_model, stream), "graph performance warmup replay")) goto cleanup;
      std::cout << "GRAPH_PERF_WARMUP_EXECUTE_ENQUEUED iteration=" << warmup << '\n';
      if (!CheckAcl(aclrtSynchronizeStream(stream), "graph performance warmup completion")) goto cleanup;
      std::cout << "GRAPH_PERF_WARMUP_STEP_DONE iteration=" << warmup << '\n';
    }
    std::cout << "GRAPH_PERF_WARMUP_DONE count=" << options.graph_warmup << '\n';
    std::cout << "GRAPH_PERF_MEASURED_BEGIN\n";
    elapsed_ms.reserve(options.graph_iterations);
    for (uint32_t iteration = 0U; iteration < options.graph_iterations; ++iteration) {
      const uint32_t epoch = options.graph_replays + options.graph_warmup + iteration;
      std::cout << "GRAPH_PERF_MEASURED_STEP_BEGIN iteration=" << iteration << " epoch=" << epoch << '\n';
      PrepareGraphReplayInputs(epoch, source, options, &address_table, &indices, &mask, &output_indices);
      const auto begin = std::chrono::steady_clock::now();
      if (!CopyGraphReplayInputs(stream, device_address_table, device_indices, device_mask, device_output_indices,
                                 address_table, indices, mask, output_indices)) goto cleanup;
      std::cout << "GRAPH_PERF_MEASURED_INPUT_ENQUEUED iteration=" << iteration << '\n';
      if (!CheckAcl(aclmdlRIExecuteAsync(graph_model, stream), "measured graph replay")) goto cleanup;
      std::cout << "GRAPH_PERF_MEASURED_EXECUTE_ENQUEUED iteration=" << iteration << '\n';
      if (!CheckAcl(aclrtSynchronizeStream(stream), "measured graph replay completion")) goto cleanup;
      const auto end = std::chrono::steady_clock::now();
      elapsed_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
      std::cout << "GRAPH_PERF_MEASURED_STEP_DONE iteration=" << iteration << '\n';
    }
    if (!CheckAcl(aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses[0]), device_status,
                               statuses.size() * sizeof(statuses[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy graph performance status") ||
        !CheckAcl(aclrtMemcpy(queues.data(), queues.size() * sizeof(queues[0]), device_queue,
                               queues.size() * sizeof(queues[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy graph performance queue") ||
        !CheckAcl(aclrtMemcpy(output.data(), output.size(), output_hbm, output.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy graph performance output") ||
        !Validate(output, address_table, indices, mask, output_indices, options, source, statuses, queues,
                  completed_batch_base + static_cast<uint64_t>(options.graph_replays) + options.graph_warmup +
                      options.graph_iterations,
                  "graph_performance")) {
      goto cleanup;
    }
    uint64_t active = 0U;
    for (uint8_t entry : mask) active += entry != 0U;
    const double mean_ms = std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) / elapsed_ms.size();
    const double gibps = (static_cast<double>(active) * options.block_bytes / (1024.0 * 1024.0 * 1024.0)) /
                         (mean_ms / 1000.0);
    std::cout << "GRAPH_PERF_VALIDATION_PASS executions="
              << (static_cast<uint64_t>(options.graph_replays) + options.graph_warmup + options.graph_iterations)
              << '\n';
    std::cout << std::fixed << std::setprecision(3)
              << "GRAPH_PERF_RESULT metric=graph_input_update_to_stream_completion samples="
              << options.graph_iterations << " warmup_excluded=" << options.graph_warmup
              << " active_requests=" << active << " block_bytes=" << options.block_bytes
              << " aiv_cores=" << options.aiv_cores << " mean_ms=" << mean_ms
              << " p50_ms=" << Percentile(elapsed_ms, 0.50) << " p90_ms=" << Percentile(elapsed_ms, 0.90)
              << " p99_ms=" << Percentile(elapsed_ms, 0.99) << " useful_GiBps=" << gibps << '\n';
    exit_code = 0;
    goto cleanup;
  }

  for (uint32_t warmup = 0U; warmup < options.warmup; ++warmup) {
    if (!CheckKvaf(KvafExecuteBatchGetLaunch(&args, stream), "warmup launch") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "warmup completion")) goto cleanup;
  }
  std::cout << "PHASE_WARMUP_DONE count=" << options.warmup << '\n';
  if (!CheckAcl(aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses[0]), device_status,
                             statuses.size() * sizeof(statuses[0]), ACL_MEMCPY_DEVICE_TO_HOST), "copy warmup status") ||
      !CheckAcl(aclrtMemcpy(queues.data(), queues.size() * sizeof(queues[0]), device_queue,
                             queues.size() * sizeof(queues[0]), ACL_MEMCPY_DEVICE_TO_HOST), "copy warmup queue") ||
      !CheckAcl(aclrtMemcpy(output.data(), output.size(), output_hbm, output.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                "copy warmup output") ||
      !Validate(output, address_table, indices, mask, output_indices, options, source, statuses, queues,
                completed_batch_base + options.warmup, "warmup")) goto cleanup;
  std::cout << "PHASE_VALIDATION_PASS stage=warmup\n";

  elapsed_ms.reserve(options.iterations);
  for (uint32_t iteration = 0U; iteration < options.iterations; ++iteration) {
    const auto begin = std::chrono::steady_clock::now();
    if (!CheckKvaf(KvafExecuteBatchGetLaunch(&args, stream), "measured launch") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "measured completion")) goto cleanup;
    const auto end = std::chrono::steady_clock::now();
    elapsed_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  if (!CheckAcl(aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses[0]), device_status,
                             statuses.size() * sizeof(statuses[0]), ACL_MEMCPY_DEVICE_TO_HOST), "copy measured status") ||
      !CheckAcl(aclrtMemcpy(queues.data(), queues.size() * sizeof(queues[0]), device_queue,
                             queues.size() * sizeof(queues[0]), ACL_MEMCPY_DEVICE_TO_HOST), "copy measured queue") ||
      !CheckAcl(aclrtMemcpy(output.data(), output.size(), output_hbm, output.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                "copy measured output") ||
      !Validate(output, address_table, indices, mask, output_indices, options, source, statuses, queues,
                completed_batch_base + static_cast<uint64_t>(options.warmup) + options.iterations,
                "measured")) goto cleanup;
  std::cout << "PHASE_MEASURED_VALIDATION_PASS\n";
  {
    const double mean_ms = std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) / elapsed_ms.size();
    uint64_t active = 0U;
    for (uint8_t entry : mask) active += entry != 0U;
    const double gibps = (static_cast<double>(active) * options.block_bytes / (1024.0 * 1024.0 * 1024.0)) /
                         (mean_ms / 1000.0);
    std::cout << std::fixed << std::setprecision(3)
              << "PERF_RESULT metric=host_submit_to_stream_completion samples=" << options.iterations
              << " warmup_excluded=" << options.warmup << " active_requests=" << active
              << " block_bytes=" << options.block_bytes << " aiv_cores=" << options.aiv_cores
              << " mean_ms=" << mean_ms << " p50_ms=" << Percentile(elapsed_ms, 0.50)
              << " p90_ms=" << Percentile(elapsed_ms, 0.90) << " p99_ms=" << Percentile(elapsed_ms, 0.99)
              << " useful_GiBps=" << gibps << '\n';
  }
  exit_code = 0;

cleanup:
  std::cout << "PHASE_TEARDOWN_BEGIN status=" << exit_code << '\n';
  if (graph_model != nullptr) (void)aclmdlRIDestroy(graph_model);
  if (registration != nullptr) KvafBatchGetDestroyTransport(registration);
  if (device_queue != nullptr) (void)aclrtFree(device_queue);
  if (device_status != nullptr) (void)aclrtFree(device_status);
  if (device_output_indices != nullptr) (void)aclrtFree(device_output_indices);
  if (device_mask != nullptr) (void)aclrtFree(device_mask);
  if (device_indices != nullptr) (void)aclrtFree(device_indices);
  if (device_valid_table != nullptr) (void)aclrtFree(device_valid_table);
  if (device_address_table != nullptr) (void)aclrtFree(device_address_table);
  if (alternate_output_hbm != nullptr) (void)aclrtFree(alternate_output_hbm);
  if (output_hbm != nullptr) (void)aclrtFree(output_hbm);
  if (source_ddr != nullptr) (void)aclrtFreeHost(source_ddr);
  if (stream != nullptr) {
    std::cout << "PHASE_RUNTIME_STEP_BEGIN step=aclrtDestroyStream\n";
    (void)aclrtDestroyStream(stream);
    std::cout << "PHASE_RUNTIME_STEP_DONE step=aclrtDestroyStream\n";
  }
  std::cout << "PHASE_RUNTIME_STEP_BEGIN step=aclrtResetDevice\n";
  (void)aclrtResetDevice(options.device);
  std::cout << "PHASE_RUNTIME_STEP_DONE step=aclrtResetDevice\n";
  std::cout << "PHASE_RUNTIME_STEP_BEGIN step=aclFinalize\n";
  (void)aclFinalize();
  std::cout << "PHASE_RUNTIME_STEP_DONE step=aclFinalize\n";
  std::cout << "PHASE_TEARDOWN_DONE status=" << exit_code << '\n';
  return exit_code;
}
