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
#include <random>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_mdl.h"

namespace {

constexpr uint32_t kEagerRandomSeed = 0x4b564146U;
constexpr uint32_t kDefaultBlockBytes =
    KVAF_AIV_BATCH_GET_DEFAULT_BLOCK_BYTES;
constexpr uint32_t kRowsPerEntry =
    KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_COUNT;
constexpr uint32_t kSgesPerReadWqe = 5U;
constexpr uint32_t kWqesPerPost = 32U;
constexpr uint32_t kRequestsPerPost = kSgesPerReadWqe * kWqesPerPost;

uint64_t WqebbsForRequestCount(uint64_t request_count) {
  const uint64_t full_wqes = request_count / kSgesPerReadWqe;
  const uint32_t tail_sges =
      static_cast<uint32_t>(request_count % kSgesPerReadWqe);
  const uint64_t tail_wqebbs =
      tail_sges == 0U ? 0U : (48U + tail_sges * 16U + 63U) / 64U;
  return full_wqes * 2U + tail_wqebbs;
}

struct Options {
  uint32_t device = 0U;
  uint32_t table_capacity = 4096U;
  uint32_t requests = 4096U;
  uint32_t output_rows = 4096U;
  uint32_t block_bytes = kDefaultBlockBytes;
  // Zero means derive the tightly packed 128-row entry size after parsing
  // block_bytes.
  uint32_t source_stride = 0U;
  uint32_t aiv_cores = 4U;
  uint32_t jettys_per_core = 2U;
  uint32_t warmup = 10U;
  uint32_t iterations = 50U;
  uint32_t miss_count = UINT32_MAX;
  uint32_t active_percent = 5U;
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

bool RequestIsActive(uint32_t request, const Options &options) {
  return request < options.miss_count;
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
    } else if (arg == "--jettys-per-core" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->jettys_per_core)) return false;
    } else if (arg == "--warmup" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->warmup)) return false;
    } else if (arg == "--iterations" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->iterations)) return false;
    } else if (arg == "--miss-count" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->miss_count)) return false;
    } else if (arg == "--active-percent" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->active_percent)) return false;
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
  const bool parsed =
      !options->kernel.empty() && ReadEnvU32("DEVICE_ID", &options->device) &&
         ReadEnvU32("TABLE_CAPACITY", &options->table_capacity) && ReadEnvU32("REQUESTS", &options->requests) &&
         ReadEnvU32("OUTPUT_ROWS", &options->output_rows) && ReadEnvU32("BLOCK_BYTES", &options->block_bytes) &&
         ReadEnvU32("SOURCE_STRIDE", &options->source_stride) && ReadEnvU32("AIV_CORES", &options->aiv_cores) &&
         ReadEnvU32("JETTYS_PER_CORE", &options->jettys_per_core) &&
         ReadEnvU32("WARMUP", &options->warmup) && ReadEnvU32("ITERATIONS", &options->iterations) &&
         ReadEnvU32("MISS_COUNT", &options->miss_count) &&
         ReadEnvU32("ACTIVE_PERCENT", &options->active_percent) &&
         ReadEnvU32("DYNAMIC_OUTPUT_SMOKE", &options->dynamic_output_smoke);
  if (!parsed) return false;
  if (options->miss_count == UINT32_MAX) {
    options->miss_count = options->requests;
  }
  const uint64_t entry_bytes =
      static_cast<uint64_t>(kRowsPerEntry) * options->block_bytes;
  if (options->source_stride == 0U && entry_bytes <= INT32_MAX) {
    options->source_stride = static_cast<uint32_t>(entry_bytes);
  }
  return options->table_capacity != 0U &&
         options->requests != 0U && options->requests <= options->output_rows && options->output_rows != 0U &&
         options->block_bytes != 0U && entry_bytes <= INT32_MAX &&
         options->source_stride >= entry_bytes &&
         options->aiv_cores != 0U && options->aiv_cores <= KVAF_AIV_BATCH_GET_MAX_AIV_CORES &&
         options->jettys_per_core != 0U &&
         options->jettys_per_core <= KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE &&
         options->aiv_cores <= options->requests && options->iterations != 0U &&
         options->active_percent >= 1U && options->active_percent <= 100U &&
         options->graph_iterations != 0U && options->dynamic_output_smoke <= 1U &&
         options->miss_count <= options->requests &&
         options->miss_count <= static_cast<uint32_t>(INT32_MAX);
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

struct QueueExpectation {
  uint64_t sq_producer_index = 0U;
  uint64_t completed_batch_count = 0U;
};

void AccumulateQueueExpectations(const std::vector<uint8_t> &mask, const Options &options,
                                 std::vector<QueueExpectation> *expectations) {
  if (expectations == nullptr) return;
  const uint64_t miss_count = static_cast<uint64_t>(
      std::count_if(mask.begin(), mask.end(),
                    [](uint8_t entry) { return entry != 0U; }));
  for (uint32_t core = 0U; core < options.aiv_cores; ++core) {
    const uint32_t begin = static_cast<uint32_t>(
        (miss_count * core) / options.aiv_cores);
    const uint32_t end = static_cast<uint32_t>(
        (miss_count * (core + 1U)) /
        options.aiv_cores);
    const uint64_t active = end - begin;
    const uint64_t total_posts =
        (active + kRequestsPerPost - 1U) / kRequestsPerPost;
    for (uint32_t jetty = 0U; jetty < options.jettys_per_core; ++jetty) {
      uint64_t assigned_wqebbs = 0U;
      for (uint64_t post = jetty; post < total_posts;
           post += options.jettys_per_core) {
        const uint64_t request_offset = post * kRequestsPerPost;
        const uint64_t post_requests = std::min<uint64_t>(
            kRequestsPerPost, active - request_offset);
        assigned_wqebbs += WqebbsForRequestCount(post_requests);
      }
      QueueExpectation &expectation =
          (*expectations)[static_cast<uint64_t>(core) *
                              options.jettys_per_core +
                          jetty];
      expectation.sq_producer_index += assigned_wqebbs;
      expectation.completed_batch_count += assigned_wqebbs != 0U;
    }
  }
}

bool Validate(const std::vector<uint8_t> &output, const std::vector<uint64_t> &address_table,
              const std::vector<int32_t> &indices, const std::vector<int32_t> &offsets,
              const std::vector<uint8_t> &mask,
              const std::vector<int32_t> &output_indices, const Options &options,
              const uint8_t *source, const std::vector<KvafBatchGetCoreStatus> &statuses,
              const std::vector<KvafBatchGetQueueState> &queues,
              const std::vector<QueueExpectation> &expectations,
              bool validate_masked_rows, const char *stage) {
  if (expectations.size() != queues.size()) return false;
  const uint64_t active_total =
      static_cast<uint64_t>(std::count_if(mask.begin(), mask.end(), [](uint8_t entry) { return entry != 0U; }));
  for (uint32_t core = 0U; core < options.aiv_cores; ++core) {
    const uint32_t begin = static_cast<uint32_t>(
        (active_total * core) / options.aiv_cores);
    const uint32_t end = static_cast<uint32_t>(
        (active_total * (core + 1U)) / options.aiv_cores);
    const uint64_t active = end - begin;
    const auto &status = statuses[core];
    const QueueExpectation &primary_expectation =
        expectations[static_cast<uint64_t>(core) * options.jettys_per_core];
    if (status.magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || status.result != 0U ||
        status.processed_request_count != active ||
        status.completed_batch_count != primary_expectation.completed_batch_count) {
      std::cerr << "[KVAF_EXECUTE] kernel state mismatch stage=" << stage << " core=" << core
                << " result=" << status.result << " processed=" << status.processed_request_count
                << " active=" << active << " completed=" << status.completed_batch_count
                << " expected_batches=" << primary_expectation.completed_batch_count
                << " failed_request=" << status.failed_request_index << '\n';
      return false;
    }
    for (uint32_t jetty = 0U; jetty < options.jettys_per_core; ++jetty) {
      const uint64_t queue_index =
          static_cast<uint64_t>(core) * options.jettys_per_core + jetty;
      const auto &queue = queues[queue_index];
      const QueueExpectation &expectation = expectations[queue_index];
      if (queue.magic != 0x4b564151U || queue.initialized != 1U ||
          queue.last_error != 0U ||
          queue.completed_batch_count != expectation.completed_batch_count ||
          queue.sq_producer_index != expectation.sq_producer_index ||
          queue.cq_consumer_index != expectation.completed_batch_count ||
          (jetty == 0U &&
           (status.sq_producer_index != queue.sq_producer_index ||
            status.cq_consumer_index != queue.cq_consumer_index))) {
        std::cerr << "[KVAF_EXECUTE] queue state mismatch stage=" << stage
                  << " core=" << core << " jetty=" << jetty
                  << " completed=" << queue.completed_batch_count
                  << " expected_batches=" << expectation.completed_batch_count
                  << " queue_error=" << queue.last_error
                  << " sq_pi=" << queue.sq_producer_index
                  << " expected_sq_pi=" << expectation.sq_producer_index
                  << " cq_ci=" << queue.cq_consumer_index << '\n';
        return false;
      }
    }
  }
  for (uint32_t request = 0U; request < options.requests; ++request) {
    if (mask[request] == 0U) {
      if (!validate_masked_rows) continue;
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
    uint32_t core_begin = 0U;
    uint32_t core_end = 0U;
    for (uint32_t core = 0U; core < options.aiv_cores; ++core) {
      const uint32_t candidate_begin = static_cast<uint32_t>(
          (active_total * core) / options.aiv_cores);
      const uint32_t candidate_end = static_cast<uint32_t>(
          (active_total * (core + 1U)) / options.aiv_cores);
      if (request >= candidate_begin && request < candidate_end) {
        core_begin = candidate_begin;
        core_end = candidate_end;
        break;
      }
    }
    const uint32_t group_begin =
        core_begin + (request - core_begin) / kSgesPerReadWqe *
                         kSgesPerReadWqe;
    const uint32_t remote_request =
        core_begin + (group_begin - core_begin) / kSgesPerReadWqe;
    const uint32_t group_sges =
        std::min<uint32_t>(kSgesPerReadWqe, core_end - group_begin);
    const uint32_t group_lane = request - group_begin;
    const uint32_t max_remote_offset =
        (kRowsPerEntry - group_sges) * options.block_bytes;
    const uint32_t group_remote_offset = std::min<uint32_t>(
        static_cast<uint32_t>(offsets[remote_request]), max_remote_offset);
    const uint64_t source_offset =
        address_table[static_cast<uint32_t>(indices[remote_request])] -
        reinterpret_cast<uint64_t>(source) + group_remote_offset +
        static_cast<uint64_t>(group_lane) * options.block_bytes;
    for (uint32_t byte = 0U; byte < options.block_bytes; ++byte) {
      const uint8_t actual = output[static_cast<uint64_t>(output_indices[request]) * options.block_bytes + byte];
      const uint8_t expected = source[source_offset + byte];
      if (actual != expected) {
        std::cerr << "[KVAF_EXECUTE] validation mismatch stage=" << stage << " request=" << request
                  << " source_uba=0x" << std::hex
                  << address_table[static_cast<uint32_t>(indices[remote_request])] +
                         group_remote_offset +
                         static_cast<uint64_t>(group_lane) * options.block_bytes
                  << std::dec << " output_row=" << output_indices[request] << " byte=" << byte
                  << " expected=" << static_cast<uint32_t>(expected) << " actual="
                  << static_cast<uint32_t>(actual) << '\n';
        return false;
      }
    }
  }
  std::cout << "EXECUTE_BATCH_GET_VALIDATION stage=" << stage << " active_requests=" << active_total
            << " masked_requests=" << (options.requests - active_total) << " table_mode=absolute_host_uba"
            << " remote_layout=contiguous_group local_sges=5 output_layout=scatter_rows jettys_per_core="
            << options.jettys_per_core << "\n";
  return true;
}

// This deliberately changes only data inside already-allocated device
// tensors.  It is the graph contract: addresses/shape/args stay fixed across
// replay, while a new AccessTable epoch resolves the same logical indices to
// different registered Host-DDR blocks.
void PrepareGraphReplayInputs(uint32_t replay, const uint8_t *source, const Options &options,
                              std::vector<uint64_t> *address_table, std::vector<int32_t> *indices,
                              std::vector<int32_t> *offsets,
                              std::vector<uint8_t> *mask, std::vector<int32_t> *output_indices) {
  const uint32_t address_shift = replay % options.table_capacity;
  const bool sparse_prefix = options.miss_count != options.requests;
  for (uint32_t entry = 0U; entry < options.table_capacity; ++entry) {
    const uint32_t source_slot = (entry + address_shift) % options.table_capacity;
    (*address_table)[entry] = reinterpret_cast<uint64_t>(source) +
                              static_cast<uint64_t>(source_slot) * options.source_stride;
  }
  for (uint32_t request = 0U; request < options.requests; ++request) {
    (*indices)[request] = static_cast<int32_t>(request % options.table_capacity);
    (*offsets)[request] =
        static_cast<int32_t>(((request + replay) % kRowsPerEntry) *
                             options.block_bytes);
    (*mask)[request] = RequestIsActive(request, options) ? 1U : 0U;
    // Keep inactive output rows stable so a row written by an earlier replay
    // is never reinterpreted as inactive during byte-for-byte validation.
    const uint32_t output_row = sparse_prefix ? request : request + replay;
    (*output_indices)[request] = static_cast<int32_t>(output_row % options.output_rows);
  }
}

bool CopyGraphReplayInputs(aclrtStream stream, void *device_address_table, void *device_indices,
                           void *device_miss_count,
                           void *device_offsets, void *device_output_indices,
                           const std::vector<uint64_t> &address_table,
                           const std::vector<int32_t> &indices,
                           const std::vector<int32_t> &offsets,
                           const std::vector<uint8_t> &mask,
                           const std::vector<int32_t> &output_indices,
                           int32_t *miss_count) {
  if (miss_count == nullptr) return false;
  *miss_count = static_cast<int32_t>(
      std::count_if(mask.begin(), mask.end(),
                    [](uint8_t entry) { return entry != 0U; }));
  return CheckAcl(aclrtMemcpyAsync(device_address_table, address_table.size() * sizeof(uint64_t),
                                   address_table.data(), address_table.size() * sizeof(uint64_t),
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream),
                  "graph copy address_table") &&
         CheckAcl(aclrtMemcpyAsync(device_indices, indices.size() * sizeof(int32_t), indices.data(),
                                   indices.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream),
                  "graph copy indices") &&
         CheckAcl(aclrtMemcpyAsync(device_offsets, offsets.size() * sizeof(int32_t), offsets.data(),
                                   offsets.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream),
                  "graph copy offsets") &&
         CheckAcl(aclrtMemcpyAsync(device_miss_count, sizeof(*miss_count),
                                   miss_count, sizeof(*miss_count),
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream),
                  "graph copy miss_count") &&
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
                 "--source-stride N --aiv-cores N --jettys-per-core N --warmup N --iterations N --miss-count N "
                 "--active-percent N "
                 "--graph-replays N --graph-warmup N --graph-iterations N --dynamic-output-smoke 0|1]\n";
    return 2;
  }
  const uint64_t source_bytes = static_cast<uint64_t>(options.table_capacity) * options.source_stride;
  const uint64_t output_bytes = static_cast<uint64_t>(options.output_rows) * options.block_bytes;
  const uint64_t eager_sample_count =
      static_cast<uint64_t>(options.warmup) + options.iterations;
  if (eager_sample_count >
      std::numeric_limits<uint64_t>::max() / options.requests) return 2;
  const uint64_t eager_request_entries = eager_sample_count * options.requests;
  if (source_bytes > std::numeric_limits<size_t>::max() ||
      output_bytes > std::numeric_limits<size_t>::max() ||
      eager_request_entries >
          std::numeric_limits<size_t>::max() / sizeof(int32_t)) return 2;

  int exit_code = 1;
  uint32_t device_count = 0U;
  aclrtStream stream = nullptr;
  void *source_ddr = nullptr;
  void *output_hbm = nullptr;
  void *alternate_output_hbm = nullptr;
  void *device_address_table = nullptr;
  void *device_valid_table = nullptr;
  void *device_indices = nullptr;
  void *device_offsets = nullptr;
  void *device_miss_count = nullptr;
  void *device_eager_indices = nullptr;
  void *device_eager_miss_counts = nullptr;
  void *device_output_indices = nullptr;
  void *device_status = nullptr;
  void *device_queue = nullptr;
  KvafBatchGetTransport *transport = nullptr;
  KvafBatchGetLocalBuffer *primary_local_buffer = nullptr;
  KvafBatchGetLocalBuffer *alternate_local_buffer = nullptr;
  KvafBatchGetLocalBuffer *primary_reacquired_buffer = nullptr;
  aclmdlRI graph_model = nullptr;
  uint64_t primary_entity_table = 0U;
  uint64_t alternate_entity_table = 0U;
  uint64_t primary_reacquired_entity_table = 0U;
  uint8_t *source = nullptr;
  KvafExecuteBatchGetKernelArgs args{};
  std::vector<uint64_t> address_table(options.table_capacity);
  std::vector<uint8_t> valid_table(options.table_capacity, 1U);
  std::vector<int32_t> indices(options.requests);
  std::vector<int32_t> offsets(options.requests);
  std::vector<uint8_t> mask(options.requests, 1U);
  std::vector<int32_t> output_indices(options.requests);
  int32_t miss_count = static_cast<int32_t>(options.miss_count);
  std::vector<KvafBatchGetCoreStatus> statuses(options.aiv_cores);
  std::vector<KvafBatchGetQueueState> queues(
      static_cast<uint64_t>(options.aiv_cores) * options.jettys_per_core);
  std::vector<QueueExpectation> queue_expectations(queues.size());
  std::vector<int32_t> eager_indices;
  std::vector<uint8_t> eager_masks;
  std::vector<int32_t> eager_miss_counts;
  std::vector<uint8_t> output(output_bytes);
  std::vector<double> elapsed_ms;
  uint64_t measured_active_total = 0U;

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
    for (uint32_t row = 0U; row < kRowsPerEntry; ++row) {
      for (uint32_t byte = 0U; byte < options.block_bytes; ++byte) {
        block[static_cast<uint64_t>(row) * options.block_bytes + byte] =
            PatternByte(slot * kRowsPerEntry + row, byte);
      }
    }
    address_table[slot] = reinterpret_cast<uint64_t>(block);
  }
  for (uint32_t request = 0U; request < options.requests; ++request) {
    indices[request] = static_cast<int32_t>(options.table_capacity - 1U - (request % options.table_capacity));
    offsets[request] =
        static_cast<int32_t>((request % kRowsPerEntry) * options.block_bytes);
    output_indices[request] = static_cast<int32_t>(options.output_rows - 1U - request);
    mask[request] = RequestIsActive(request, options) ? 1U : 0U;
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
      !CheckAcl(aclrtMalloc(&device_offsets, offsets.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "allocate offsets") ||
      !CheckAcl(aclrtMalloc(&device_miss_count, sizeof(miss_count), ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate miss_count") ||
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
      !CheckAcl(aclrtMemcpy(device_offsets, offsets.size() * sizeof(int32_t), offsets.data(),
                             offsets.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy offsets") ||
      !CheckAcl(aclrtMemcpy(device_miss_count, sizeof(miss_count), &miss_count,
                             sizeof(miss_count), ACL_MEMCPY_HOST_TO_DEVICE), "copy miss_count") ||
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

  if (!CheckKvaf(KvafBatchGetTransportCreate(source_ddr, source_bytes, options.aiv_cores,
                                               options.jettys_per_core, &transport),
                 "create reusable Direct-HCCP transport and register Host DDR")) goto cleanup;
  if (!CheckKvaf(KvafBatchGetLocalBufferAcquire(transport, output_hbm, output_bytes, &primary_local_buffer,
                                                 &primary_entity_table),
                 "acquire primary caller-owned output HBM for UDMA")) goto cleanup;
  std::cout << "PHASE_EID_ENDPOINT_READY primary_entity_table=0x" << std::hex << primary_entity_table << std::dec
            << " table_capacity=" << options.table_capacity << " output_rows=" << options.output_rows
            << " source_bytes=" << source_bytes << " output_bytes=" << output_bytes
            << " output_lifetime=local_buffer_acquired\n";
  if (!CheckKvaf(KvafExecuteBatchGetLoadKernel(options.kernel.c_str()), "load ExecuteBatchGet kernel")) goto cleanup;
  std::cout << "PHASE_KERNEL_REGISTERED cores=" << options.aiv_cores
            << " jettys_per_core=" << options.jettys_per_core
            << " contract=address_table_absolute_uba_plus_miss_count_prefix_plus_output_scatter\n";

  args.address_table = reinterpret_cast<uint64_t>(device_address_table);
  args.valid_table = reinterpret_cast<uint64_t>(device_valid_table);
  args.indices = reinterpret_cast<uint64_t>(device_indices);
  args.offsets = reinterpret_cast<uint64_t>(device_offsets);
  args.miss_count = reinterpret_cast<uint64_t>(device_miss_count);
  args.output_indices = reinterpret_cast<uint64_t>(device_output_indices);
  args.output = reinterpret_cast<uint64_t>(output_hbm);
  args.core_status = reinterpret_cast<uint64_t>(device_status);
  args.channel_entity = primary_entity_table;
  args.queue_state = reinterpret_cast<uint64_t>(device_queue);
  args.table_capacity = options.table_capacity;
  args.request_capacity = options.requests;
  args.output_rows = options.output_rows;
  args.block_bytes = options.block_bytes;
  args.block_count = options.aiv_cores;
  args.jettys_per_core = options.jettys_per_core;

  if (options.dynamic_output_smoke != 0U) {
    // A -> B -> A proves the API contract rather than merely proving that two
    // MRs can coexist: the second A acquire must return the cached entity.
    AccumulateQueueExpectations(mask, options, &queue_expectations);
    if (!CheckKvaf(KvafExecuteBatchGetLaunch(&args, stream), "primary local-buffer smoke launch") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "primary local-buffer smoke completion") ||
        !CheckAcl(aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses[0]), device_status,
                               statuses.size() * sizeof(statuses[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy primary local-buffer status") ||
        !CheckAcl(aclrtMemcpy(queues.data(), queues.size() * sizeof(queues[0]), device_queue,
                               queues.size() * sizeof(queues[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy primary local-buffer queue") ||
        !CheckAcl(aclrtMemcpy(output.data(), output.size(), output_hbm, output.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy primary local-buffer output") ||
        !Validate(output, address_table, indices, offsets, mask, output_indices, options,
                  source, statuses, queues, queue_expectations, true,
                  "primary_output")) {
      goto cleanup;
    }

    if (!CheckKvaf(KvafBatchGetLocalBufferAcquire(transport, alternate_output_hbm, output_bytes,
                                                   &alternate_local_buffer, &alternate_entity_table),
                   "acquire alternate caller-owned output HBM") ||
        alternate_local_buffer == primary_local_buffer || alternate_entity_table == primary_entity_table) {
      std::cerr << "[KVAF_EXECUTE] alternate output did not receive an independent local-buffer entity\n";
      goto cleanup;
    }
    args.output = reinterpret_cast<uint64_t>(alternate_output_hbm);
    args.channel_entity = alternate_entity_table;
    AccumulateQueueExpectations(mask, options, &queue_expectations);
    if (!CheckKvaf(KvafExecuteBatchGetLaunch(&args, stream), "alternate local-buffer smoke launch") ||
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
        !Validate(output, address_table, indices, offsets, mask, output_indices, options,
                  source, statuses, queues, queue_expectations, true,
                  "alternate_output")) {
      goto cleanup;
    }
    if (!CheckKvaf(KvafBatchGetLocalBufferAcquire(transport, output_hbm, output_bytes, &primary_reacquired_buffer,
                                                   &primary_reacquired_entity_table),
                   "re-acquire primary caller-owned output HBM") ||
        primary_reacquired_buffer != primary_local_buffer || primary_reacquired_entity_table != primary_entity_table) {
      std::cerr << "[KVAF_EXECUTE] primary local-buffer cache contract failed\n";
      goto cleanup;
    }
    args.output = reinterpret_cast<uint64_t>(output_hbm);
    args.channel_entity = primary_reacquired_entity_table;
    AccumulateQueueExpectations(mask, options, &queue_expectations);
    if (!CheckKvaf(KvafExecuteBatchGetLaunch(&args, stream), "re-acquired primary local-buffer smoke launch") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "re-acquired primary local-buffer smoke completion") ||
        !CheckAcl(aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses[0]), device_status,
                               statuses.size() * sizeof(statuses[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy re-acquired primary status") ||
        !CheckAcl(aclrtMemcpy(queues.data(), queues.size() * sizeof(queues[0]), device_queue,
                               queues.size() * sizeof(queues[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy re-acquired primary queue") ||
        !CheckAcl(aclrtMemcpy(output.data(), output.size(), output_hbm, output.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy re-acquired primary output") ||
        !Validate(output, address_table, indices, offsets, mask, output_indices, options,
                  source, statuses, queues, queue_expectations, true,
                  "reacquired_primary_output")) {
      goto cleanup;
    }
    KvafBatchGetLocalBufferRelease(primary_reacquired_buffer);
    primary_reacquired_buffer = nullptr;
    args.output = reinterpret_cast<uint64_t>(output_hbm);
    args.channel_entity = primary_entity_table;
    std::cout << "PHASE_DYNAMIC_OUTPUT_VALIDATION_PASS primary_entity=0x" << std::hex << primary_entity_table
              << " alternate_entity=0x" << alternate_entity_table << std::dec
              << " primary_reacquire_cache_hit=1 transport_reused=1\n";
  }

  if (options.graph_replays != 0U) {
    // Resources, tensor allocations, fixed kernel arguments and the direct
    // HCCP registration are intentionally complete before capture.  The only
    // per-replay updates below are writes into the already-fixed device
    // tensors, just as an upstream graph node would do.
    std::cout << "GRAPH_CAPTURE_CONTRACT api=aclmdlRI fixed_tensor_addresses=1 "
                 "dynamic_device_contents=address_table_indices_miss_count_output_indices "
                 "replays="
              << options.graph_replays << " aiv_cores=" << options.aiv_cores
              << " jettys_per_core=" << options.jettys_per_core
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
      PrepareGraphReplayInputs(replay, source, options, &address_table, &indices, &offsets, &mask, &output_indices);
      if (!CopyGraphReplayInputs(stream, device_address_table, device_indices,
                                 device_miss_count, device_offsets,
                                 device_output_indices, address_table, indices,
                                 offsets, mask, output_indices, &miss_count)) {
        goto cleanup;
      }
      std::cout << "GRAPH_REPLAY_INPUT_UPDATED replay=" << replay
                << " address_table_epoch=" << replay << " fixed_tensor_addresses=1\n";
      AccumulateQueueExpectations(mask, options, &queue_expectations);
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
          !Validate(output, address_table, indices, offsets, mask, output_indices, options,
                    source, statuses, queues, queue_expectations, true,
                    "graph_replay")) {
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
      PrepareGraphReplayInputs(epoch, source, options, &address_table, &indices, &offsets, &mask, &output_indices);
      if (!CopyGraphReplayInputs(stream, device_address_table, device_indices,
                                 device_miss_count, device_offsets,
                                 device_output_indices, address_table, indices,
                                 offsets, mask, output_indices, &miss_count)) goto cleanup;
      std::cout << "GRAPH_PERF_WARMUP_INPUT_ENQUEUED iteration=" << warmup << '\n';
      AccumulateQueueExpectations(mask, options, &queue_expectations);
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
      PrepareGraphReplayInputs(epoch, source, options, &address_table, &indices, &offsets, &mask, &output_indices);
      const auto begin = std::chrono::steady_clock::now();
      if (!CopyGraphReplayInputs(stream, device_address_table, device_indices,
                                 device_miss_count, device_offsets,
                                 device_output_indices, address_table, indices,
                                 offsets, mask, output_indices, &miss_count)) goto cleanup;
      std::cout << "GRAPH_PERF_MEASURED_INPUT_ENQUEUED iteration=" << iteration << '\n';
      AccumulateQueueExpectations(mask, options, &queue_expectations);
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
        !Validate(output, address_table, indices, offsets, mask, output_indices, options,
                  source, statuses, queues, queue_expectations, true,
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
              << " aiv_cores=" << options.aiv_cores
              << " jettys_per_core=" << options.jettys_per_core
              << " mean_ms=" << mean_ms
              << " p50_ms=" << Percentile(elapsed_ms, 0.50) << " p90_ms=" << Percentile(elapsed_ms, 0.90)
              << " p99_ms=" << Percentile(elapsed_ms, 0.99) << " useful_GiBps=" << gibps << '\n';
    exit_code = 0;
    goto cleanup;
  }

  // Match the Python performance benchmark: materialize one independent
  // random indices/miss-count pair for every warmup and measured sample before
  // the benchmark starts. Keeping the inputs in flat device buffers lets the
  // timed loop select a sample by pointer only; random generation and H2D
  // copies remain outside host_submit_to_stream_completion.
  eager_indices.resize(static_cast<size_t>(eager_request_entries));
  eager_masks.resize(static_cast<size_t>(eager_request_entries));
  eager_miss_counts.resize(static_cast<size_t>(eager_sample_count));
  {
    std::mt19937 random(kEagerRandomSeed);
    std::uniform_int_distribution<uint32_t> index_distribution(
        0U, options.table_capacity - 1U);
    std::bernoulli_distribution mask_distribution(
        static_cast<double>(options.active_percent) / 100.0);
    for (uint64_t sample = 0U; sample < eager_sample_count; ++sample) {
      const size_t sample_offset =
          static_cast<size_t>(sample * options.requests);
      uint64_t active = 0U;
      for (uint32_t request = 0U; request < options.requests; ++request) {
        eager_indices[sample_offset + request] =
            static_cast<int32_t>(index_distribution(random));
        active += mask_distribution(random) ? 1U : 0U;
      }
      std::fill_n(eager_masks.data() + sample_offset, options.requests, 0U);
      std::fill_n(eager_masks.data() + sample_offset,
                  static_cast<size_t>(active), 1U);
      eager_miss_counts[static_cast<size_t>(sample)] =
          static_cast<int32_t>(active);
    }
  }
  if (!CheckAcl(aclrtMalloc(&device_eager_indices,
                             eager_indices.size() * sizeof(eager_indices[0]),
                             ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate eager sample indices") ||
      !CheckAcl(aclrtMalloc(&device_eager_miss_counts,
                             eager_miss_counts.size() *
                                 sizeof(eager_miss_counts[0]),
                             ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate eager sample miss counts") ||
      !CheckAcl(aclrtMemcpy(device_eager_indices,
                             eager_indices.size() * sizeof(eager_indices[0]),
                             eager_indices.data(),
                             eager_indices.size() * sizeof(eager_indices[0]),
                             ACL_MEMCPY_HOST_TO_DEVICE),
                "copy eager sample indices") ||
      !CheckAcl(aclrtMemcpy(device_eager_miss_counts,
                             eager_miss_counts.size() *
                                 sizeof(eager_miss_counts[0]),
                             eager_miss_counts.data(),
                             eager_miss_counts.size() *
                                 sizeof(eager_miss_counts[0]),
                             ACL_MEMCPY_HOST_TO_DEVICE),
                "copy eager sample miss counts")) {
    goto cleanup;
  }
  std::cout << "EAGER_RANDOM_INPUTS_READY samples=" << eager_sample_count
            << " requests_per_sample=" << options.requests
            << " active_percent=" << options.active_percent
            << " seed=" << kEagerRandomSeed
            << " generation_excluded=1 h2d_excluded=1\n";

  for (uint32_t warmup = 0U; warmup < options.warmup; ++warmup) {
    const size_t sample_offset =
        static_cast<size_t>(static_cast<uint64_t>(warmup) * options.requests);
    args.indices = reinterpret_cast<uint64_t>(device_eager_indices) +
                   sample_offset * sizeof(eager_indices[0]);
    args.miss_count =
        reinterpret_cast<uint64_t>(device_eager_miss_counts) +
        static_cast<size_t>(warmup) * sizeof(eager_miss_counts[0]);
    std::copy_n(eager_indices.data() + sample_offset, options.requests,
                indices.data());
    std::copy_n(eager_masks.data() + sample_offset, options.requests,
                mask.data());
    AccumulateQueueExpectations(mask, options, &queue_expectations);
    if (!CheckKvaf(KvafExecuteBatchGetLaunch(&args, stream), "warmup launch") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "warmup completion")) goto cleanup;
  }
  std::cout << "PHASE_WARMUP_DONE count=" << options.warmup << '\n';
  if (options.warmup != 0U) {
    if (!CheckAcl(aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses[0]), device_status,
                               statuses.size() * sizeof(statuses[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy warmup status") ||
        !CheckAcl(aclrtMemcpy(queues.data(), queues.size() * sizeof(queues[0]), device_queue,
                               queues.size() * sizeof(queues[0]), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy warmup queue") ||
        !CheckAcl(aclrtMemcpy(output.data(), output.size(), output_hbm, output.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                  "copy warmup output") ||
        !Validate(output, address_table, indices, offsets, mask, output_indices, options,
                  source, statuses, queues, queue_expectations, false,
                  "warmup")) {
      goto cleanup;
    }
    std::cout << "PHASE_VALIDATION_PASS stage=warmup\n";
  }

  elapsed_ms.reserve(options.iterations);
  for (uint32_t iteration = 0U; iteration < options.iterations; ++iteration) {
    const uint64_t sample = static_cast<uint64_t>(options.warmup) + iteration;
    const size_t sample_offset =
        static_cast<size_t>(sample * options.requests);
    args.indices = reinterpret_cast<uint64_t>(device_eager_indices) +
                   sample_offset * sizeof(eager_indices[0]);
    args.miss_count =
        reinterpret_cast<uint64_t>(device_eager_miss_counts) +
        static_cast<size_t>(sample) * sizeof(eager_miss_counts[0]);
    std::copy_n(eager_indices.data() + sample_offset, options.requests,
                indices.data());
    std::copy_n(eager_masks.data() + sample_offset, options.requests,
                mask.data());
    measured_active_total += static_cast<uint64_t>(
        eager_miss_counts[static_cast<size_t>(sample)]);
    AccumulateQueueExpectations(mask, options, &queue_expectations);
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
      !Validate(output, address_table, indices, offsets, mask, output_indices, options,
                source, statuses, queues, queue_expectations, false,
                "measured")) goto cleanup;
  std::cout << "PHASE_MEASURED_VALIDATION_PASS\n";
  {
    const double mean_ms = std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) / elapsed_ms.size();
    const double mean_active = static_cast<double>(measured_active_total) /
                               options.iterations;
    const double gibps = (mean_active * options.block_bytes / (1024.0 * 1024.0 * 1024.0)) /
                         (mean_ms / 1000.0);
    std::cout << std::fixed << std::setprecision(3)
              << "PERF_RESULT metric=host_submit_to_stream_completion samples=" << options.iterations
              << " warmup_excluded=" << options.warmup
              << " mean_active_requests=" << mean_active
              << " block_bytes=" << options.block_bytes << " aiv_cores=" << options.aiv_cores
              << " jettys_per_core=" << options.jettys_per_core
              << " mean_ms=" << mean_ms << " p50_ms=" << Percentile(elapsed_ms, 0.50)
              << " p90_ms=" << Percentile(elapsed_ms, 0.90) << " p99_ms=" << Percentile(elapsed_ms, 0.99)
              << " useful_GiBps=" << gibps << '\n';
  }
  exit_code = 0;

cleanup:
  std::cout << "PHASE_TEARDOWN_BEGIN status=" << exit_code << '\n';
  if (graph_model != nullptr) (void)aclmdlRIDestroy(graph_model);
  // Every normal launch above synchronizes before this point. Keep the
  // LocalBuffer release-before-transport-destroy order explicit so callers
  // can follow the same safe lifetime contract.
  if (primary_reacquired_buffer != nullptr) KvafBatchGetLocalBufferRelease(primary_reacquired_buffer);
  if (alternate_local_buffer != nullptr) KvafBatchGetLocalBufferRelease(alternate_local_buffer);
  if (primary_local_buffer != nullptr) KvafBatchGetLocalBufferRelease(primary_local_buffer);
  if (transport != nullptr) KvafBatchGetTransportDestroy(transport);
  if (device_queue != nullptr) (void)aclrtFree(device_queue);
  if (device_status != nullptr) (void)aclrtFree(device_status);
  if (device_output_indices != nullptr) (void)aclrtFree(device_output_indices);
  if (device_eager_miss_counts != nullptr) (void)aclrtFree(device_eager_miss_counts);
  if (device_eager_indices != nullptr) (void)aclrtFree(device_eager_indices);
  if (device_miss_count != nullptr) (void)aclrtFree(device_miss_count);
  if (device_indices != nullptr) (void)aclrtFree(device_indices);
  if (device_offsets != nullptr) (void)aclrtFree(device_offsets);
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
