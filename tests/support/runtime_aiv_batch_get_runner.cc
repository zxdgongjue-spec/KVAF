#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "batch_get_abi.h"
#include "kvaf/access_operator/aiv_batch_get_access_operator.h"
#include "kvaf/meta_manager/meta_manager.h"
#include "kvaf/runtime/runtime.h"

namespace {

struct Options {
  std::string kernel;
  std::uint32_t device{0};
  std::uint32_t table_capacity{8};
  std::uint32_t requests{8};
  std::uint32_t output_rows{8};
  std::uint32_t block_bytes{1024};
  std::uint32_t source_stride{4096};
  std::uint32_t aiv_cores{1};
  std::uint32_t mask_every{0};
  std::uint32_t warmup{1};
  std::uint32_t iterations{5};
};

std::uint8_t PatternByte(std::uint32_t slot, std::uint32_t byte) {
  return static_cast<std::uint8_t>(
      (slot * 131U + byte * 17U + (byte >> 3U)) & 0xffU);
}

bool ParseU32(const char* text, std::uint32_t* value) {
  if (text == nullptr || *text == '\0' || value == nullptr) return false;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 0);
  if (end == nullptr || *end != '\0' ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

bool ParseOptions(int argc, char** argv, Options* options) {
  if (options == nullptr) return false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--kernel" && index + 1 < argc) {
      options->kernel = argv[++index];
    } else if (argument == "--device" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->device)) return false;
    } else if (argument == "--table-capacity" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->table_capacity)) return false;
    } else if (argument == "--requests" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->requests)) return false;
    } else if (argument == "--output-rows" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->output_rows)) return false;
    } else if (argument == "--block-bytes" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->block_bytes)) return false;
    } else if (argument == "--source-stride" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->source_stride)) return false;
    } else if (argument == "--aiv-cores" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->aiv_cores)) return false;
    } else if (argument == "--mask-every" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->mask_every)) return false;
    } else if (argument == "--warmup" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->warmup)) return false;
    } else if (argument == "--iterations" && index + 1 < argc) {
      if (!ParseU32(argv[++index], &options->iterations)) return false;
    } else {
      return false;
    }
  }
  return !options->kernel.empty() && options->table_capacity != 0 &&
         options->requests != 0 && options->requests <= options->output_rows &&
         options->block_bytes != 0 &&
         options->source_stride >= options->block_bytes &&
         options->aiv_cores != 0 &&
         options->aiv_cores <= KVAF_AIV_BATCH_GET_MAX_AIV_CORES &&
         options->aiv_cores <= options->requests && options->iterations != 0;
}

bool CheckAcl(aclError error, const char* step) {
  if (error == ACL_SUCCESS) return true;
  std::cerr << "[KVAF_RUNTIME_AIV] " << step << " acl_error=" << error << '\n';
  return false;
}

bool CheckStatus(const kvaf::Status& status, const char* step) {
  if (status.ok()) return true;
  std::cerr << "[KVAF_RUNTIME_AIV] " << step
            << " status=" << static_cast<int>(status.code())
            << " message=" << status.message() << '\n';
  return false;
}

class AclHostDdrAllocator final : public kvaf::DdrMemoryAllocator {
 public:
  kvaf::Status Allocate(const kvaf::RuntimeInitRequest& request,
                        kvaf::RemoteRegionDesc* region) override {
    if (region == nullptr || request.ddr_device != kvaf::DeviceKind::kCpu ||
        request.ddr_bytes == 0 || request.slot_size == 0 ||
        request.ddr_bytes % request.slot_size != 0) {
      return kvaf::Status::Error(kvaf::StatusCode::kInvalidArgument,
                                 "invalid ACL Host DDR allocation request");
    }
    if (address_ != nullptr || aclrtMallocHost(&address_, request.ddr_bytes) != ACL_SUCCESS) {
      return kvaf::Status::Error(kvaf::StatusCode::kOutOfMemory,
                                 "aclrtMallocHost failed for Runtime DDR");
    }
    *region = kvaf::RemoteRegionDesc{
        reinterpret_cast<kvaf::Address>(address_), request.ddr_bytes,
        request.slot_size, kvaf::DeviceKind::kCpu};
    return kvaf::Status::OK();
  }

  kvaf::Status Release(const kvaf::RemoteRegionDesc& region) override {
    if (address_ == nullptr ||
        reinterpret_cast<kvaf::Address>(address_) != region.base) {
      return kvaf::Status::Error(kvaf::StatusCode::kNotFound,
                                 "Runtime DDR allocation was not found");
    }
    const aclError status = aclrtFreeHost(address_);
    address_ = nullptr;
    return status == ACL_SUCCESS
               ? kvaf::Status::OK()
               : kvaf::Status::Error(kvaf::StatusCode::kInternal,
                                     "aclrtFreeHost failed for Runtime DDR");
  }

 private:
  void* address_{nullptr};
};

bool ValidateOutput(const std::vector<std::uint8_t>& output,
                    const std::vector<std::int32_t>& indices,
                    const std::vector<std::uint8_t>& mask,
                    const std::vector<std::int32_t>& output_indices,
                    const Options& options,
                    const std::vector<KvafBatchGetCoreStatus>& statuses,
                    const std::vector<KvafBatchGetQueueState>& queues,
                    std::uint64_t expected_batches) {
  for (std::uint32_t request = 0; request < options.requests; ++request) {
    const std::size_t output_offset =
        static_cast<std::size_t>(output_indices[request]) * options.block_bytes;
    if (mask[request] == 0) {
      for (std::uint32_t byte = 0; byte < options.block_bytes; ++byte) {
        if (output[output_offset + byte] != 0) {
          std::cerr << "[KVAF_RUNTIME_AIV] masked row was modified request="
                    << request << '\n';
          return false;
        }
      }
      continue;
    }
    for (std::uint32_t byte = 0; byte < options.block_bytes; ++byte) {
      if (output[output_offset + byte] !=
          PatternByte(static_cast<std::uint32_t>(indices[request]), byte)) {
        std::cerr << "[KVAF_RUNTIME_AIV] output mismatch request=" << request
                  << " byte=" << byte << '\n';
        return false;
      }
    }
  }
  for (std::uint32_t core = 0; core < options.aiv_cores; ++core) {
    const std::uint32_t begin = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(options.requests) * core) / options.aiv_cores);
    const std::uint32_t end = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(options.requests) * (core + 1U)) /
        options.aiv_cores);
    const std::uint64_t active = std::count_if(
        mask.begin() + begin, mask.begin() + end,
        [](std::uint8_t value) { return value != 0; });
    if (statuses[core].magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC ||
        statuses[core].result != 0 ||
        statuses[core].processed_request_count != active ||
        statuses[core].completed_batch_count !=
            (active == 0 ? 0 : expected_batches) ||
        queues[core].magic != 0x4b564151U || queues[core].initialized != 1U ||
        queues[core].last_error != 0 ||
        queues[core].completed_batch_count !=
            (active == 0 ? 0 : expected_batches)) {
      std::cerr << "[KVAF_RUNTIME_AIV] completion state mismatch core=" << core
                << " result=" << statuses[core].result
                << " queue_error=" << queues[core].last_error << '\n';
      return false;
    }
  }
  return true;
}

double Percentile(std::vector<double> samples, double fraction) {
  std::sort(samples.begin(), samples.end());
  const double position = fraction * static_cast<double>(samples.size() - 1U);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1U, samples.size() - 1U);
  return samples[lower] + (samples[upper] - samples[lower]) *
                              (position - lower);
}

}  // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    std::cerr << "usage: kvaf_runtime_aiv_batch_get_runner --kernel PATH "
                 "[--device N --table-capacity N --requests N --output-rows N "
                 "--block-bytes N --source-stride N --aiv-cores N --mask-every N "
                 "--warmup N --iterations N]\n";
    return 2;
  }

  const std::uint64_t ddr_bytes =
      static_cast<std::uint64_t>(options.table_capacity) * options.source_stride;
  const std::uint64_t output_bytes =
      static_cast<std::uint64_t>(options.output_rows) * options.block_bytes;
  if (ddr_bytes > std::numeric_limits<std::size_t>::max() ||
      output_bytes > std::numeric_limits<std::size_t>::max()) {
    return 2;
  }

  int exit_code = 1;
  bool acl_initialized = false;
  bool device_selected = false;
  bool runtime_initialized = false;
  bool table_prepared = false;
  aclrtStream stream = nullptr;
  void* output_device = nullptr;
  void* device_addresses = nullptr;
  void* device_valid = nullptr;
  void* device_indices = nullptr;
  void* device_mask = nullptr;
  void* device_output_indices = nullptr;
  void* device_status = nullptr;
  void* device_queue = nullptr;

  auto meta_manager = kvaf::CreateInMemoryMetaManager();
  AclHostDdrAllocator ddr_allocator;
  std::unique_ptr<kvaf::AivBatchGetAccessOperator> access_operator;
  std::unique_ptr<kvaf::Runtime> runtime;
  kvaf::AccessTableHandle access_table;
  std::vector<std::uint64_t> addresses(options.table_capacity);
  std::vector<std::uint8_t> valid(options.table_capacity, 1U);
  std::vector<std::int32_t> indices(options.requests);
  std::vector<std::uint8_t> mask(options.requests, 1U);
  std::vector<std::int32_t> output_indices(options.requests);
  std::vector<KvafBatchGetCoreStatus> statuses(options.aiv_cores);
  std::vector<KvafBatchGetQueueState> queues(options.aiv_cores);
  std::vector<std::uint8_t> output(output_bytes);
  std::vector<double> elapsed_ms;

  std::cout << "RUNTIME_AIV_PHASE_BEGIN phase=acl_init\n";
  do {
    std::uint32_t device_count = 0;
    if (!CheckAcl(aclInit(nullptr), "aclInit")) break;
    acl_initialized = true;
    if (!CheckAcl(aclrtGetDeviceCount(&device_count), "aclrtGetDeviceCount") ||
        device_count == 0 || options.device >= device_count ||
        !CheckAcl(aclrtSetDevice(options.device), "aclrtSetDevice") ||
        !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) {
      break;
    }
    device_selected = true;
    std::cout << "RUNTIME_AIV_PHASE_DONE phase=acl_ready device="
              << options.device << '\n';

    access_operator = std::make_unique<kvaf::AivBatchGetAccessOperator>(
        kvaf::AivBatchGetOperatorConfig{options.kernel, stream, options.aiv_cores});
    runtime = std::make_unique<kvaf::Runtime>(*meta_manager, *access_operator,
                                               ddr_allocator);
    std::cout << "RUNTIME_AIV_PHASE_BEGIN phase=runtime_initialize\n";
    if (!CheckStatus(runtime->Initialize(kvaf::RuntimeInitRequest{
                         ddr_bytes, options.source_stride, kvaf::DeviceKind::kCpu}),
                     "Runtime.Initialize")) {
      break;
    }
    runtime_initialized = true;
    std::cout << "RUNTIME_AIV_PHASE_DONE phase=runtime_initialize ddr_bytes="
              << ddr_bytes << '\n';

    std::vector<kvaf::KeyId> keys(options.table_capacity);
    std::vector<std::uint64_t> lengths(options.table_capacity, options.block_bytes);
    for (std::uint32_t slot = 0; slot < options.table_capacity; ++slot) {
      keys[slot] = static_cast<kvaf::KeyId>(slot + 1U);
    }
    kvaf::PutContext put_context;
    kvaf::PutAccessDesc put_desc;
    if (!CheckStatus(meta_manager->PreparePut(keys, lengths, &put_context, &put_desc),
                     "MetaManager.PreparePut")) {
      break;
    }
    for (std::uint32_t slot = 0; slot < options.table_capacity; ++slot) {
      auto* block = reinterpret_cast<std::uint8_t*>(put_desc.entries[slot].address);
      for (std::uint32_t byte = 0; byte < options.block_bytes; ++byte) {
        block[byte] = PatternByte(slot, byte);
      }
    }
    if (!CheckStatus(meta_manager->CommitPut(put_context), "MetaManager.CommitPut") ||
        !CheckStatus(runtime->PrepareAccessTable(
                         kvaf::AccessTablePrepareRequest{keys}, &access_table),
                     "Runtime.PrepareAccessTable")) {
      break;
    }
    table_prepared = true;
    kvaf::AccessTableView host_table;
    if (!CheckStatus(runtime->GetAccessTableView(access_table, &host_table),
                     "Runtime.GetAccessTableView")) {
      break;
    }
    std::copy(host_table.addresses, host_table.addresses + host_table.size,
              addresses.begin());
    std::copy(host_table.valid, host_table.valid + host_table.size, valid.begin());
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      indices[request] = static_cast<std::int32_t>(
          options.table_capacity - 1U - (request % options.table_capacity));
      output_indices[request] =
          static_cast<std::int32_t>(options.output_rows - 1U - request);
      if (options.mask_every != 0 && request % options.mask_every == 0) {
        mask[request] = 0;
      }
    }

    std::cout << "RUNTIME_AIV_PHASE_BEGIN phase=caller_buffer_setup\n";
    if (!CheckAcl(aclrtMalloc(&output_device, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "allocate output") ||
        !CheckAcl(aclrtMalloc(&device_addresses, addresses.size() * sizeof(addresses[0]),
                               ACL_MEM_MALLOC_HUGE_FIRST), "allocate address table") ||
        !CheckAcl(aclrtMalloc(&device_valid, valid.size(), ACL_MEM_MALLOC_HUGE_FIRST),
                  "allocate valid table") ||
        !CheckAcl(aclrtMalloc(&device_indices, indices.size() * sizeof(indices[0]),
                               ACL_MEM_MALLOC_HUGE_FIRST), "allocate indices") ||
        !CheckAcl(aclrtMalloc(&device_mask, mask.size(), ACL_MEM_MALLOC_HUGE_FIRST),
                  "allocate mask") ||
        !CheckAcl(aclrtMalloc(&device_output_indices,
                               output_indices.size() * sizeof(output_indices[0]),
                               ACL_MEM_MALLOC_HUGE_FIRST), "allocate output indices") ||
        !CheckAcl(aclrtMalloc(&device_status, statuses.size() * sizeof(statuses[0]),
                               ACL_MEM_MALLOC_HUGE_FIRST), "allocate status") ||
        !CheckAcl(aclrtMalloc(&device_queue, queues.size() * sizeof(queues[0]),
                               ACL_MEM_MALLOC_HUGE_FIRST), "allocate queue") ||
        !CheckAcl(aclrtMemcpy(device_addresses, addresses.size() * sizeof(addresses[0]),
                               addresses.data(), addresses.size() * sizeof(addresses[0]),
                               ACL_MEMCPY_HOST_TO_DEVICE), "copy address table") ||
        !CheckAcl(aclrtMemcpy(device_valid, valid.size(), valid.data(), valid.size(),
                               ACL_MEMCPY_HOST_TO_DEVICE), "copy valid table") ||
        !CheckAcl(aclrtMemcpy(device_indices, indices.size() * sizeof(indices[0]),
                               indices.data(), indices.size() * sizeof(indices[0]),
                               ACL_MEMCPY_HOST_TO_DEVICE), "copy indices") ||
        !CheckAcl(aclrtMemcpy(device_mask, mask.size(), mask.data(), mask.size(),
                               ACL_MEMCPY_HOST_TO_DEVICE), "copy mask") ||
        !CheckAcl(aclrtMemcpy(device_output_indices,
                               output_indices.size() * sizeof(output_indices[0]),
                               output_indices.data(),
                               output_indices.size() * sizeof(output_indices[0]),
                               ACL_MEMCPY_HOST_TO_DEVICE), "copy output indices") ||
        !CheckAcl(aclrtMemset(output_device, output_bytes, 0, output_bytes),
                  "clear output") ||
        !CheckAcl(aclrtMemset(device_status, statuses.size() * sizeof(statuses[0]), 0,
                               statuses.size() * sizeof(statuses[0])), "clear status") ||
        !CheckAcl(aclrtMemcpy(device_queue, queues.size() * sizeof(queues[0]),
                               queues.data(), queues.size() * sizeof(queues[0]),
                               ACL_MEMCPY_HOST_TO_DEVICE), "initialize queue")) {
      break;
    }
    if (!CheckStatus(access_operator->BindBatchGetOutput(
                         kvaf::AivBatchGetOutputBinding{
                             kvaf::MutableBufferView{output_device,
                                                     static_cast<std::size_t>(output_bytes),
                                                     kvaf::DeviceKind::kNpu},
                             kvaf::MutableBufferView{device_status, statuses.size() * sizeof(statuses[0]),
                                                     kvaf::DeviceKind::kNpu},
                             kvaf::MutableBufferView{device_queue, queues.size() * sizeof(queues[0]),
                                                     kvaf::DeviceKind::kNpu}}),
                     "AivBatchGetAccessOperator.BindBatchGetOutput")) {
      break;
    }
    std::cout << "RUNTIME_AIV_PHASE_DONE phase=caller_buffer_setup\n";

    kvaf::BatchGetRequest request;
    request.access_table = access_table;
    request.indices = kvaf::ConstBufferView{device_indices,
                                             indices.size() * sizeof(indices[0]),
                                             kvaf::DeviceKind::kNpu};
    request.mask = kvaf::ConstBufferView{device_mask, mask.size(), kvaf::DeviceKind::kNpu};
    request.output_indices = kvaf::ConstBufferView{
        device_output_indices, output_indices.size() * sizeof(output_indices[0]),
        kvaf::DeviceKind::kNpu};
    request.output = kvaf::MutableBufferView{output_device,
                                              static_cast<std::size_t>(output_bytes),
                                              kvaf::DeviceKind::kNpu};
    request.output_block_bytes = options.block_bytes;
    request.device_access_table = kvaf::AccessTableDeviceView{
        kvaf::ConstBufferView{device_addresses, addresses.size() * sizeof(addresses[0]),
                              kvaf::DeviceKind::kNpu},
        kvaf::ConstBufferView{device_valid, valid.size(), kvaf::DeviceKind::kNpu}};

    std::cout << "RUNTIME_AIV_PHASE_BEGIN phase=runtime_batch_get\n";
    if (!CheckStatus(runtime->BatchGet(request), "Runtime.BatchGet") ||
        !CheckAcl(aclrtMemcpy(output.data(), output.size(), output_device, output.size(),
                               ACL_MEMCPY_DEVICE_TO_HOST), "copy output") ||
        !CheckAcl(aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses[0]),
                               device_status, statuses.size() * sizeof(statuses[0]),
                               ACL_MEMCPY_DEVICE_TO_HOST), "copy status") ||
        !CheckAcl(aclrtMemcpy(queues.data(), queues.size() * sizeof(queues[0]),
                               device_queue, queues.size() * sizeof(queues[0]),
                               ACL_MEMCPY_DEVICE_TO_HOST), "copy queue") ||
        !ValidateOutput(output, indices, mask, output_indices, options, statuses,
                        queues, 1U)) {
      break;
    }
    std::cout << "RUNTIME_AIV_BATCH_GET_VALIDATION_PASS requests="
              << options.requests << '\n';

    bool warmup_ok = true;
    for (std::uint32_t iteration = 0; iteration < options.warmup; ++iteration) {
      if (!CheckStatus(runtime->BatchGet(request), "Runtime.BatchGet warmup")) {
        warmup_ok = false;
        break;
      }
    }
    if (!warmup_ok) {
      break;
    }
    const std::uint64_t completed_before_measurement =
        1U + options.warmup;
    elapsed_ms.reserve(options.iterations);
    bool measurement_ok = true;
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      const kvaf::Status status = runtime->BatchGet(request);
      const auto end = std::chrono::steady_clock::now();
      if (!CheckStatus(status, "Runtime.BatchGet measured")) {
        measurement_ok = false;
        break;
      }
      elapsed_ms.push_back(
          std::chrono::duration<double, std::milli>(end - begin).count());
    }
    if (!measurement_ok) {
      break;
    }
    if (!CheckAcl(aclrtMemcpy(output.data(), output.size(), output_device, output.size(),
                               ACL_MEMCPY_DEVICE_TO_HOST), "copy measured output") ||
        !CheckAcl(aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses[0]),
                               device_status, statuses.size() * sizeof(statuses[0]),
                               ACL_MEMCPY_DEVICE_TO_HOST), "copy measured status") ||
        !CheckAcl(aclrtMemcpy(queues.data(), queues.size() * sizeof(queues[0]),
                               device_queue, queues.size() * sizeof(queues[0]),
                               ACL_MEMCPY_DEVICE_TO_HOST), "copy measured queue") ||
        !ValidateOutput(output, indices, mask, output_indices, options, statuses,
                        queues, completed_before_measurement + options.iterations)) {
      break;
    }
    const std::uint64_t active_requests =
        static_cast<std::uint64_t>(std::count_if(
            mask.begin(), mask.end(),
            [](std::uint8_t value) { return value != 0; }));
    const double mean_ms = std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) /
                           static_cast<double>(elapsed_ms.size());
    const double useful_gibps =
        (static_cast<double>(active_requests) * options.block_bytes /
         (1024.0 * 1024.0 * 1024.0)) /
        (mean_ms / 1000.0);
    std::cout << std::fixed << std::setprecision(3)
              << "PERF_RESULT metric=runtime_batch_get_call_to_stream_completion"
              << " samples=" << options.iterations
              << " warmup_excluded=" << options.warmup
              << " active_requests=" << active_requests
              << " block_bytes=" << options.block_bytes
              << " aiv_cores=" << options.aiv_cores
              << " mean_ms=" << mean_ms
              << " p50_ms=" << Percentile(elapsed_ms, 0.50)
              << " p90_ms=" << Percentile(elapsed_ms, 0.90)
              << " p99_ms=" << Percentile(elapsed_ms, 0.99)
              << " useful_GiBps=" << useful_gibps << '\n';
    exit_code = 0;
  } while (false);

  std::cout << "RUNTIME_AIV_PHASE_BEGIN phase=teardown\n";
  if (runtime != nullptr) {
    if (table_prepared) (void)runtime->ReleaseAccessTable(access_table);
    if (runtime_initialized) (void)runtime->Shutdown();
    runtime.reset();
  }
  access_operator.reset();
  if (device_queue != nullptr) (void)aclrtFree(device_queue);
  if (device_status != nullptr) (void)aclrtFree(device_status);
  if (device_output_indices != nullptr) (void)aclrtFree(device_output_indices);
  if (device_mask != nullptr) (void)aclrtFree(device_mask);
  if (device_indices != nullptr) (void)aclrtFree(device_indices);
  if (device_valid != nullptr) (void)aclrtFree(device_valid);
  if (device_addresses != nullptr) (void)aclrtFree(device_addresses);
  if (output_device != nullptr) (void)aclrtFree(output_device);
  if (stream != nullptr) (void)aclrtDestroyStream(stream);
  if (device_selected) (void)aclrtResetDevice(options.device);
  if (acl_initialized) (void)aclFinalize();
  std::cout << "RUNTIME_AIV_PHASE_DONE phase=teardown status=" << exit_code << '\n';
  return exit_code;
}
