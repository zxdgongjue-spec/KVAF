#include "kvaf_aiv_batch_put.h"

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
#include <utility>
#include <vector>

namespace {

struct Options {
  std::string kernel;
  uint32_t device = 0U;
  uint32_t rows = 32U;
  uint32_t min_bytes = 256U;
  uint32_t max_bytes = 4096U;
  uint32_t destination_stride = 4096U;
  uint32_t aiv_cores = 1U;
  uint32_t jettys_per_core = 2U;
  uint32_t iterations = 1U;
};

bool ParseU32(const char *text, uint32_t *value) {
  if (text == nullptr || value == nullptr || *text == '\0') return false;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (*end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) return false;
  *value = static_cast<uint32_t>(parsed);
  return true;
}

bool ParseOptions(int argc, char **argv, Options *options) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--kernel" && index + 1 < argc) {
      options->kernel = argv[++index];
    } else if (argument == "--device" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->device)) return false;
    } else if (argument == "--rows" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->rows)) return false;
    } else if (argument == "--min-bytes" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->min_bytes)) return false;
    } else if (argument == "--max-bytes" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->max_bytes)) return false;
    } else if (argument == "--destination-stride" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->destination_stride)) return false;
    } else if (argument == "--aiv-cores" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->aiv_cores)) return false;
    } else if (argument == "--jettys-per-core" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->jettys_per_core)) return false;
    } else if (argument == "--iterations" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->iterations)) return false;
    } else {
      return false;
    }
  }
  return !options->kernel.empty() && options->rows != 0U && options->min_bytes != 0U &&
         options->max_bytes >= options->min_bytes && options->destination_stride >= options->max_bytes &&
         options->aiv_cores != 0U && options->aiv_cores <= options->rows &&
         options->aiv_cores <= KVAF_AIV_BATCH_PUT_MAX_AIV_CORES &&
         options->jettys_per_core != 0U &&
         options->jettys_per_core <= KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE &&
         options->iterations != 0U;
}

bool CheckAcl(aclError error, const char *step) {
  if (error == ACL_SUCCESS) return true;
  std::cerr << "[KVAF_EXECUTE] " << step << " ACL error=" << error << '\n';
  return false;
}

bool CheckKvaf(int32_t error, const char *step) {
  if (error == KVAF_BATCH_PUT_SUCCESS) return true;
  std::cerr << "[KVAF_EXECUTE] " << step << " error=" << error << '\n';
  return false;
}

uint8_t PatternByte(uint32_t row, uint64_t byte, uint32_t epoch) {
  return static_cast<uint8_t>((row * 37U + byte * 13U + epoch * 17U + 1U) & 0xffU);
}

void RandomizeDestinations(uint64_t host_ddr_address, uint32_t rows,
                           uint32_t destination_stride, uint32_t iteration,
                           std::vector<uint32_t> *previous_slots,
                           std::vector<uint64_t> *destinations) {
  std::vector<uint32_t> slots(rows);
  std::iota(slots.begin(), slots.end(), 0U);
  std::mt19937_64 generator(0x4b5641465f425054ULL + iteration);
  std::shuffle(slots.begin(), slots.end(), generator);
  if (!previous_slots->empty() && slots == *previous_slots) {
    std::rotate(slots.begin(), slots.begin() + 1, slots.end());
  }
  for (uint32_t row = 0U; row < rows; ++row) {
    (*destinations)[row] = host_ddr_address +
                           static_cast<uint64_t>(slots[row]) * destination_stride;
  }
  *previous_slots = std::move(slots);
}

bool Validate(const std::vector<uint8_t> &input, const std::vector<uint64_t> &offsets,
              const std::vector<uint64_t> &lengths,
              const std::vector<uint64_t> &destinations, uint32_t destination_stride,
              const std::vector<KvafBatchPutCoreStatus> &statuses,
              uint32_t aiv_cores, uint32_t rows, uint64_t expected_batches) {
  for (uint32_t row = 0U; row < rows; ++row) {
    const uint8_t *actual = reinterpret_cast<const uint8_t *>(destinations[row]);
    const uint8_t *expected = input.data() + offsets[row];
    if (!std::equal(expected, expected + lengths[row], actual)) {
      std::cerr << "DATA_MISMATCH row=" << row << " bytes=" << lengths[row] << '\n';
      return false;
    }
    for (uint64_t byte = lengths[row]; byte < destination_stride; ++byte) {
      if (actual[byte] != 0U) {
        std::cerr << "DESTINATION_OVERWRITE row=" << row << " byte=" << byte << '\n';
        return false;
      }
    }
  }
  for (uint32_t core = 0U; core < aiv_cores; ++core) {
    const uint64_t begin = static_cast<uint64_t>(rows) * core / aiv_cores;
    const uint64_t end = static_cast<uint64_t>(rows) * (core + 1U) / aiv_cores;
    const auto &status = statuses[core];
    if (status.magic != KVAF_AIV_BATCH_PUT_STATUS_MAGIC ||
        status.result != KVAF_BATCH_PUT_CORE_SUCCESS ||
        status.processed_request_count != end - begin ||
        status.completed_batch_count != expected_batches || status.failed_request_index != UINT64_MAX) {
      std::cerr << "CORE_STATUS_FAIL core=" << core << " result=" << status.result
                << " processed=" << status.processed_request_count
                << " completed=" << status.completed_batch_count
                << " failed_request=" << status.failed_request_index << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    std::cerr << "usage: kvaf_execute_batch_put_test --kernel <kernels.o> "
                 "[--device N --rows N --min-bytes N --max-bytes N "
                 "--destination-stride N --aiv-cores N --jettys-per-core N "
                 "--iterations N]\n";
    return 2;
  }

  std::vector<uint64_t> lengths(options.rows);
  std::vector<uint64_t> offsets(options.rows);
  uint64_t input_bytes = 0U;
  const uint32_t span = options.max_bytes - options.min_bytes + 1U;
  for (uint32_t row = 0U; row < options.rows; ++row) {
    offsets[row] = input_bytes;
    lengths[row] = options.min_bytes + (static_cast<uint64_t>(row) * 251U % span);
    input_bytes += lengths[row];
  }
  const uint64_t destination_bytes = static_cast<uint64_t>(options.rows) * options.destination_stride;
  if (input_bytes > std::numeric_limits<size_t>::max() ||
      destination_bytes > std::numeric_limits<size_t>::max()) return 2;

  int exit_code = 1;
  aclrtStream stream = nullptr;
  void *host_ddr = nullptr;
  void *input_hbm = nullptr;
  void *alternate_input_hbm = nullptr;
  void *device_offsets = nullptr;
  void *device_lengths = nullptr;
  void *device_destinations = nullptr;
  void *device_status = nullptr;
  KvafBatchPutTransport *transport = nullptr;
  KvafBatchPutLocalBuffer *local_buffer = nullptr;
  KvafBatchPutLocalBuffer *alternate_local_buffer = nullptr;
  KvafBatchPutLocalBuffer *reacquired_local_buffer = nullptr;
  KvafExecuteBatchPutKernelArgs arguments{};
  std::vector<uint8_t> input(input_bytes);
  std::vector<uint64_t> destinations(options.rows);
  std::vector<uint32_t> previous_destination_slots;
  std::vector<KvafBatchPutCoreStatus> statuses(options.aiv_cores);
  std::vector<double> elapsed_ms;

  std::cout << "PHASE_PROCESS_READY\n";
  if (!CheckAcl(aclInit(nullptr), "aclInit") ||
      !CheckAcl(aclrtSetDevice(options.device), "aclrtSetDevice") ||
      !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
      !CheckAcl(aclrtMallocHost(&host_ddr, destination_bytes), "allocate Host DDR") ||
      !CheckAcl(aclrtMalloc(&input_hbm, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate input HBM") ||
      !CheckAcl(aclrtMalloc(&alternate_input_hbm, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate alternate input HBM") ||
      !CheckAcl(aclrtMalloc(&device_offsets, offsets.size() * sizeof(offsets[0]), ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate offsets") ||
      !CheckAcl(aclrtMalloc(&device_lengths, lengths.size() * sizeof(lengths[0]), ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate lengths") ||
      !CheckAcl(aclrtMalloc(&device_destinations, destinations.size() * sizeof(destinations[0]),
                            ACL_MEM_MALLOC_HUGE_FIRST), "allocate destinations") ||
      !CheckAcl(aclrtMalloc(&device_status, statuses.size() * sizeof(statuses[0]), ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate status")) {
    goto cleanup;
  }
  std::memset(host_ddr, 0, destination_bytes);
  RandomizeDestinations(reinterpret_cast<uint64_t>(host_ddr), options.rows,
                        options.destination_stride, 0U, &previous_destination_slots,
                        &destinations);
  if (!CheckKvaf(KvafBatchPutTransportCreate(host_ddr, destination_bytes,
                                              options.aiv_cores,
                                              options.jettys_per_core,
                                              &transport),
                 "create transport") ||
      !CheckKvaf(KvafBatchPutLocalBufferAcquire(transport, input_hbm, input_bytes, &local_buffer),
                 "acquire local buffer") ||
      !CheckKvaf(KvafBatchPutLocalBufferAcquire(transport, alternate_input_hbm, input_bytes,
                                                &alternate_local_buffer),
                 "acquire alternate local buffer") ||
      !CheckKvaf(KvafBatchPutLocalBufferAcquire(transport, input_hbm, input_bytes,
                                                &reacquired_local_buffer),
                 "re-acquire primary local buffer") ||
      !CheckKvaf(KvafExecuteBatchPutLoadKernel(options.kernel.c_str()), "load kernel")) {
    goto cleanup;
  }
  if (alternate_local_buffer == local_buffer || reacquired_local_buffer != local_buffer) {
    std::cerr << "LOCAL_BUFFER_CACHE_CONTRACT_FAIL\n";
    goto cleanup;
  }
  KvafBatchPutLocalBufferRelease(reacquired_local_buffer);
  reacquired_local_buffer = nullptr;
  std::cout << "PHASE_TRANSPORT_READY cores=" << options.aiv_cores
            << " jettys_per_core=" << options.jettys_per_core
            << " dynamic_input=A_to_B_to_A local_buffer_cache_verified=1\n";

  if (!CheckAcl(aclrtMemcpy(device_offsets, offsets.size() * sizeof(offsets[0]), offsets.data(),
                             offsets.size() * sizeof(offsets[0]), ACL_MEMCPY_HOST_TO_DEVICE), "copy offsets") ||
      !CheckAcl(aclrtMemcpy(device_lengths, lengths.size() * sizeof(lengths[0]), lengths.data(),
                             lengths.size() * sizeof(lengths[0]), ACL_MEMCPY_HOST_TO_DEVICE), "copy lengths") ||
      !CheckAcl(aclrtMemcpy(device_destinations, destinations.size() * sizeof(destinations[0]),
                             destinations.data(), destinations.size() * sizeof(destinations[0]),
                             ACL_MEMCPY_HOST_TO_DEVICE), "copy destinations")) {
    goto cleanup;
  }

  arguments.input = reinterpret_cast<uint64_t>(input_hbm);
  arguments.input_offsets = reinterpret_cast<uint64_t>(device_offsets);
  arguments.lengths = reinterpret_cast<uint64_t>(device_lengths);
  arguments.destination_addresses = reinterpret_cast<uint64_t>(device_destinations);
  arguments.core_status = reinterpret_cast<uint64_t>(device_status);
  arguments.input_bytes = input_bytes;
  arguments.row_count = options.rows;
  arguments.aiv_core_count = options.aiv_cores;

  for (uint32_t row = 0U; row < options.rows; ++row) {
    for (uint64_t byte = 0U; byte < lengths[row]; ++byte) {
      input[offsets[row] + byte] = PatternByte(row, byte, 0U);
    }
  }
  if (!CheckAcl(aclrtMemcpy(alternate_input_hbm, input_bytes, input.data(), input_bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE), "copy alternate packed input")) {
    goto cleanup;
  }
  arguments.input = reinterpret_cast<uint64_t>(alternate_input_hbm);
  if (!CheckKvaf(KvafExecuteBatchPutSync(alternate_local_buffer, &arguments, stream, statuses.data(),
                                         static_cast<uint32_t>(statuses.size())),
                 "alternate input synchronous BatchPut") ||
      !Validate(input, offsets, lengths, destinations, options.destination_stride,
                statuses, options.aiv_cores, options.rows, 1U)) {
    goto cleanup;
  }
  std::cout << "PHASE_ALTERNATE_INPUT_VALIDATION_PASS transport_reused=1\n";
  arguments.input = reinterpret_cast<uint64_t>(input_hbm);

  elapsed_ms.reserve(options.iterations);
  for (uint32_t iteration = 0U; iteration < options.iterations; ++iteration) {
    std::memset(host_ddr, 0, destination_bytes);
    RandomizeDestinations(reinterpret_cast<uint64_t>(host_ddr), options.rows,
                          options.destination_stride, iteration + 1U,
                          &previous_destination_slots, &destinations);
    for (uint32_t row = 0U; row < options.rows; ++row) {
      for (uint64_t byte = 0U; byte < lengths[row]; ++byte) {
        input[offsets[row] + byte] = PatternByte(row, byte, iteration);
      }
    }
    if (!CheckAcl(aclrtMemcpy(device_destinations,
                              destinations.size() * sizeof(destinations[0]),
                              destinations.data(),
                              destinations.size() * sizeof(destinations[0]),
                              ACL_MEMCPY_HOST_TO_DEVICE), "copy random destinations") ||
        !CheckAcl(aclrtMemcpy(input_hbm, input_bytes, input.data(), input_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE), "copy packed input")) {
      goto cleanup;
    }
    const auto begin = std::chrono::steady_clock::now();
    if (!CheckKvaf(KvafExecuteBatchPutSync(local_buffer, &arguments, stream, statuses.data(),
                                           static_cast<uint32_t>(statuses.size())),
                   "synchronous BatchPut")) {
      for (uint32_t core = 0U; core < options.aiv_cores; ++core) {
        std::cerr << "CORE_DIAGNOSTIC core=" << core << " result=" << statuses[core].result
                  << " failed_request=" << statuses[core].failed_request_index
                  << " cq_status=" << statuses[core].cq_status
                  << " cq_sub_status=" << statuses[core].cq_sub_status << '\n';
      }
      goto cleanup;
    }
    const auto end = std::chrono::steady_clock::now();
    elapsed_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    if (!Validate(input, offsets, lengths, destinations, options.destination_stride,
                  statuses, options.aiv_cores, options.rows, iteration + 2U)) {
      goto cleanup;
    }
    std::cout << "EAGER_ITERATION_PASS iteration=" << iteration << " rows=" << options.rows << '\n';
  }

  {
    const double mean_ms = std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) / elapsed_ms.size();
    const uint64_t useful_bytes = std::accumulate(lengths.begin(), lengths.end(), uint64_t{0});
    std::cout << std::fixed << std::setprecision(3)
              << "EXECUTE_BATCH_PUT_VALIDATION result=pass variable_lengths=1 rows=" << options.rows
              << " useful_bytes=" << useful_bytes << " mean_sync_ms=" << mean_ms
              << " random_destinations_per_iteration=1 jettys_per_core=" << options.jettys_per_core
              << " all_core_status=success\n";
  }
  exit_code = 0;

cleanup:
  std::cout << "PHASE_TEARDOWN_BEGIN status=" << exit_code << '\n';
  if (reacquired_local_buffer != nullptr) KvafBatchPutLocalBufferRelease(reacquired_local_buffer);
  if (alternate_local_buffer != nullptr) KvafBatchPutLocalBufferRelease(alternate_local_buffer);
  if (local_buffer != nullptr) KvafBatchPutLocalBufferRelease(local_buffer);
  if (transport != nullptr) KvafBatchPutTransportDestroy(transport);
  if (device_status != nullptr) (void)aclrtFree(device_status);
  if (device_destinations != nullptr) (void)aclrtFree(device_destinations);
  if (device_lengths != nullptr) (void)aclrtFree(device_lengths);
  if (device_offsets != nullptr) (void)aclrtFree(device_offsets);
  if (alternate_input_hbm != nullptr) (void)aclrtFree(alternate_input_hbm);
  if (input_hbm != nullptr) (void)aclrtFree(input_hbm);
  if (host_ddr != nullptr) (void)aclrtFreeHost(host_ddr);
  if (stream != nullptr) (void)aclrtDestroyStream(stream);
  (void)aclrtResetDevice(options.device);
  (void)aclFinalize();
  std::cout << "PHASE_TEARDOWN_DONE status=" << exit_code << '\n';
  return exit_code;
}
