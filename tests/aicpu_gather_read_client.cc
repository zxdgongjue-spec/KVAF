#include "npu_runtime.h"
#include "peer_protocol.h"
#include "aicpu_urma_client_metrics.h"
#include "aicpu_urma_client_options.h"
#include "aicpu_gather_read_layout.h"

#include "acl/acl.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::string peer{"127.0.0.1:28999"};
  std::string kernel_json{kvaf::aicpu_urma_client::DefaultKernelJson()};
  std::uint32_t device_id{0U};
  std::uint32_t blocks{8U};
  std::uint32_t block_bytes{4096U};
  std::uint32_t output_stride{0U};
  std::uint32_t sgl_size{kvaf::kGatherMaxSglSize};
  std::uint32_t warmup{1U};
  std::uint32_t iterations{3U};
};

void Usage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s [--peer HOST:PORT] [--device-id N] [--kernel-json PATH] "
               "[--blocks N] [--block-bytes N] [--output-stride N] [--sgl-size N] "
               "[--warmup N] [--iterations N]\n"
               "Defaults: output-stride=2*block-bytes, sgl-size=%u (range 1..%u)\n",
               program, kvaf::kGatherMaxSglSize, kvaf::kGatherMaxSglSize);
}

bool Parse(int argc, char** argv, Options* options) {
  Options parsed;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) return false;
    const std::string flag = argv[index];
    const char* value = argv[index + 1];
    if (flag == "--peer") parsed.peer = value;
    else if (flag == "--kernel-json") parsed.kernel_json = value;
    else {
      std::uint64_t number = 0U;
      if (!kvaf::aicpu_urma_client::Number(value, &number) || number > UINT32_MAX)
        return false;
      if (flag == "--device-id") parsed.device_id = static_cast<std::uint32_t>(number);
      else if (flag == "--blocks") parsed.blocks = static_cast<std::uint32_t>(number);
      else if (flag == "--block-bytes") parsed.block_bytes = static_cast<std::uint32_t>(number);
      else if (flag == "--output-stride") parsed.output_stride = static_cast<std::uint32_t>(number);
      else if (flag == "--sgl-size") parsed.sgl_size = static_cast<std::uint32_t>(number);
      else if (flag == "--warmup") parsed.warmup = static_cast<std::uint32_t>(number);
      else if (flag == "--iterations") parsed.iterations = static_cast<std::uint32_t>(number);
      else return false;
    }
  }
  if (parsed.peer.empty() || parsed.kernel_json.empty() || parsed.device_id > INT32_MAX ||
      parsed.blocks == 0U || parsed.blocks > kvaf::peer::kMaxGatherSpans ||
      parsed.block_bytes == 0U || parsed.iterations == 0U ||
      parsed.sgl_size == 0U || parsed.sgl_size > kvaf::kGatherMaxSglSize ||
      static_cast<std::uint64_t>(parsed.blocks) * parsed.block_bytes > UINT32_MAX - 64U)
    return false;
  if (parsed.output_stride == 0U) {
    if (parsed.block_bytes > UINT32_MAX / 2U) return false;
    parsed.output_stride = 2U * parsed.block_bytes;
  }
  if (parsed.output_stride < parsed.block_bytes ||
      static_cast<std::uint64_t>(parsed.blocks - 1U) * parsed.output_stride +
          parsed.block_bytes > UINT32_MAX - 64U) return false;
  *options = parsed;
  return true;
}

void Check(const kvaf::Status& status, const char* operation) {
  if (!status.ok()) throw std::runtime_error(std::string(operation) + ": " + status.message());
}

void Acl(aclError error, const char* operation) {
  if (error != ACL_SUCCESS)
    throw std::runtime_error(std::string(operation) + " acl=" + std::to_string(error));
}

struct AclContext {
  explicit AclContext(std::uint32_t device) : device_id(device) {
    Acl(aclInit(nullptr), "aclInit");
    try {
      Acl(aclrtSetDevice(device_id), "aclrtSetDevice");
      active_device = true;
      Acl(aclrtCreateStream(&stream), "aclrtCreateStream");
    } catch (...) {
      if (active_device) (void)aclrtResetDevice(device_id);
      (void)aclFinalize();
      throw;
    }
  }
  ~AclContext() {
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    if (active_device) (void)aclrtResetDevice(device_id);
    (void)aclFinalize();
  }
  std::uint32_t device_id;
  bool active_device{false};
  aclrtStream stream{nullptr};
};

struct DeviceBuffer {
  explicit DeviceBuffer(std::size_t length) : bytes(length) {
    Acl(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
  }
  ~DeviceBuffer() { if (data != nullptr) (void)aclrtFree(data); }
  void Upload(const void* source) {
    Acl(aclrtMemcpy(data, bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
  }
  void Download(void* destination, std::size_t length) const {
    Acl(aclrtMemcpy(destination, length, data, length, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
  }
  void* data{nullptr};
  std::size_t bytes;
};

double RunRound(kvaf::NpuRuntime* runtime, aclrtStream stream, const Options& options,
                DeviceBuffer* input, DeviceBuffer* output,
                std::vector<std::uint8_t>* host_input,
                std::vector<std::uint8_t>* host_output,
                std::vector<std::uint8_t>* expected_output,
                std::vector<kvaf::KeyId>* reserved_keys,
                std::uint64_t* staging_address, std::uint64_t round,
                kvaf::GatherReadTiming* timing) {
  const std::size_t payload = host_input->size();
  for (std::size_t byte = 0U; byte < payload; ++byte)
    (*host_input)[byte] = static_cast<std::uint8_t>((byte * 131U + round * 29U) & 0xffU);
  input->Upload(host_input->data());
  kvaf::BatchPutGroup group;
  group.input = {input->data, input->bytes, kvaf::DeviceKind::kNpu};
  for (std::uint32_t block = 0U; block < options.blocks; ++block) {
    group.keys.push_back(block + 1U);
    group.input_offsets.push_back(static_cast<std::uint64_t>(block) * options.block_bytes);
    group.lengths.push_back(options.block_bytes);
  }
  kvaf::BatchPutRequest put;
  put.groups.push_back(std::move(group));
  Check(runtime->core().BatchPut(put, kvaf::BatchPutExecutionOptions{stream}), "BatchPut");
  if (round == 0U) {
    const std::size_t reserve_slots = kvaf::aicpu_gather_read_client::ReservedSlots(
        options.blocks, options.block_bytes);
    kvaf::BatchPutGroup reserved;
    reserved.input = {input->data, input->bytes, kvaf::DeviceKind::kNpu};
    for (std::size_t index = 0U; index < options.blocks + reserve_slots; ++index) {
      const kvaf::KeyId key = UINT64_MAX - 1U - index;
      reserved_keys->push_back(key);
      reserved.keys.push_back(key);
      reserved.input_offsets.push_back(0U);
      reserved.lengths.push_back(options.block_bytes);
    }
    kvaf::BatchPutRequest reservation;
    reservation.groups.push_back(std::move(reserved));
    Check(runtime->core().BatchPut(reservation, kvaf::BatchPutExecutionOptions{stream}),
          "ReserveGatherStaging");
    kvaf::AccessTableHandle reserved_handle{};
    Check(runtime->core().PrepareAccessTable({*reserved_keys}, &reserved_handle),
          "PrepareReservedAccessTable");
    kvaf::AccessTableView reserved_view{};
    Check(runtime->core().GetAccessTableView(reserved_handle, &reserved_view),
          "GetReservedAccessTableView");
    if (reserved_view.size != reserved_keys->size())
      throw std::runtime_error("reserved AccessTable size mismatch");
    std::vector<std::uint64_t> reserved_addresses;
    reserved_addresses.reserve(reserved_view.size);
    for (std::size_t index = 0U; index < reserved_view.size; ++index) {
      if (reserved_view.valid[index] != 1U ||
          reserved_view.lengths[index] != options.block_bytes)
        throw std::runtime_error("reserved AccessTable entry is invalid");
      reserved_addresses.push_back(reserved_view.addresses[index]);
    }
    const auto region = runtime->core().remote_region();
    if (!kvaf::aicpu_gather_read_client::SelectStagingAddress(
            region.base, region.length, options.block_bytes, options.blocks,
            std::move(reserved_addresses), staging_address))
      throw std::runtime_error("no contiguous staging span outside the DDR control area");
    Check(runtime->core().ReleaseAccessTable(reserved_handle),
          "ReleaseReservedAccessTable");
  }
  kvaf::AccessTableHandle handle{};
  Check(runtime->core().PrepareAccessTable({put.groups.front().keys}, &handle),
        "PrepareAccessTable");
  kvaf::AccessTableView view{};
  Check(runtime->core().GetAccessTableView(handle, &view), "GetAccessTableView");
  const auto region = runtime->core().remote_region();
  if (view.size != options.blocks || *staging_address == 0U)
    throw std::runtime_error("source AccessTable or staging address is invalid");
  kvaf::CpuGatherReadRequest gather;
  gather.staging_address = *staging_address;
  gather.output = {output->data, output->bytes, kvaf::DeviceKind::kNpu};
  gather.stream = stream;
  gather.timing = timing;
  for (std::size_t block = options.blocks; block != 0U; --block) {
    const auto address = view.addresses[block - 1U];
    if (view.valid[block - 1U] != 1U || view.lengths[block - 1U] != options.block_bytes)
      throw std::runtime_error("source AccessTable entry is invalid");
    gather.spans.push_back({address, options.block_bytes, 0U});
  }
  std::uint32_t gathered_bytes = 0U;
  if (!kvaf::peer::ValidateGather(region.base, region.length, gather.staging_address,
                                  gather.spans, &gathered_bytes) || gathered_bytes != payload)
    throw std::runtime_error("source blocks overlap staging or DDR control area");
  gather.sgl_size = options.sgl_size;
  std::fill(host_output->begin(), host_output->end(), std::uint8_t{0xa5U});
  std::fill(expected_output->begin(), expected_output->end(), std::uint8_t{0xa5U});
  for (std::uint32_t block = options.blocks; block != 0U; --block) {
    const std::size_t source = static_cast<std::size_t>(block - 1U) * options.block_bytes;
    const std::size_t destination = static_cast<std::size_t>(block - 1U) * options.output_stride;
    gather.output_spans.push_back({destination, options.block_bytes, 0U});
    std::memcpy(expected_output->data() + destination,
                host_input->data() + source, options.block_bytes);
  }
  output->Upload(host_output->data());
  const auto begin = std::chrono::steady_clock::now();
  Check(runtime->GatherRead(gather), "GatherRead");
  const auto end = std::chrono::steady_clock::now();
  output->Download(host_output->data(), expected_output->size());
  if (std::memcmp(host_output->data(), expected_output->data(),
                  expected_output->size()) != 0) {
    for (std::size_t byte = 0U; byte < expected_output->size(); ++byte)
      if ((*host_output)[byte] != (*expected_output)[byte])
        throw std::runtime_error("GatherRead HBM scatter mismatch at offset=" +
                                 std::to_string(byte));
  }
  Check(runtime->core().ReleaseAccessTable(handle), "ReleaseAccessTable");
  Check(runtime->core().BatchFree({put.groups.front().keys}), "BatchFree");
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

int Run(const Options& options) {
  if (!kvaf::aicpu_urma_client::ConfigureCannEnvironment())
    throw std::runtime_error("failed to configure ASCEND_HOME_PATH");
  AclContext acl(options.device_id);
  const std::size_t payload = static_cast<std::size_t>(options.blocks) * options.block_bytes;
  const std::size_t output_extent =
      static_cast<std::size_t>(options.blocks - 1U) * options.output_stride + options.block_bytes;
  DeviceBuffer input(payload);
  DeviceBuffer output(((output_extent + 63U) & ~std::size_t{63U}) + 64U);
  std::vector<std::uint8_t> host_input(payload), host_output(output.bytes);
  std::vector<std::uint8_t> expected_output(output_extent);
  std::vector<kvaf::KeyId> reserved_keys;
  std::uint64_t staging_address = 0U;
  const std::size_t reserve_slots = kvaf::aicpu_gather_read_client::ReservedSlots(
      options.blocks, options.block_bytes);
  kvaf::NpuRuntimeOptions runtime_options("", options.block_bytes, 1U);
  runtime_options.aicpu_urma_kernel_json_path = options.kernel_json;
  runtime_options.device_id = options.device_id;
  runtime_options.npu_core_count = 1U;
  runtime_options.jettys_per_core = 1U;
  kvaf::NpuRuntime runtime(std::move(runtime_options));
  Check(runtime.LinkTo({options.peer}), "LinkTo");
  Check(runtime.Initialize(2U * payload + reserve_slots * options.block_bytes,
                           options.block_bytes), "Initialize");
  std::vector<double> samples;
  samples.reserve(options.iterations);
  std::vector<kvaf::GatherReadTiming> profiles;
  profiles.reserve(options.iterations);
  for (std::uint64_t round = 0U;
       round < static_cast<std::uint64_t>(options.warmup) + options.iterations; ++round) {
    kvaf::GatherReadTiming timing{};
    const double elapsed = RunRound(&runtime, acl.stream, options, &input, &output,
                                    &host_input, &host_output, &expected_output, &reserved_keys,
                                    &staging_address, round, &timing);
    if (round >= options.warmup) {
      samples.push_back(elapsed);
      profiles.push_back(timing);
    }
  }
  Check(runtime.core().BatchFree({reserved_keys}), "ReleaseGatherStaging");
  Check(runtime.Shutdown(), "Shutdown");
  double total_ns = 0.0;
  for (const double elapsed_ms : samples) total_ns += elapsed_ms * 1000000.0;
  const auto stats = kvaf::aicpu_urma_client::SummarizeTiming(std::move(samples), payload);
  std::fprintf(stdout,
               "GATHER_READ_STATS min_ms=%.3f p50_ms=%.3f p95_ms=%.3f p99_ms=%.3f "
               "max_ms=%.3f mean_ms=%.3f GBps=%.3f\n",
               stats.min_ms, stats.p50_ms, stats.p95_ms, stats.p99_ms,
               stats.max_ms, stats.mean_ms, stats.mean_gbps);
  const auto average = [&](std::uint64_t kvaf::GatherReadTiming::*field) {
    std::uint64_t sum = 0U;
    for (const auto& profile : profiles) sum += profile.*field;
    return static_cast<double>(sum) / profiles.size();
  };
  std::fprintf(stdout,
               "GATHER_READ_HOST_PROFILE avg_ns total=%.0f validate=%.0f configure_rpc=%.0f "
               "descriptor_upload=%.0f hbm_register=%.0f launch=%.0f "
               "stream_sync=%.0f hbm_release=%.0f status_copy=%.0f\n",
               total_ns / profiles.size(),
               average(&kvaf::GatherReadTiming::validate_ns),
               average(&kvaf::GatherReadTiming::configure_ns),
               average(&kvaf::GatherReadTiming::descriptor_upload_ns),
               average(&kvaf::GatherReadTiming::register_ns),
               average(&kvaf::GatherReadTiming::launch_ns),
               average(&kvaf::GatherReadTiming::sync_ns),
               average(&kvaf::GatherReadTiming::release_ns),
               average(&kvaf::GatherReadTiming::status_copy_ns));
  std::fprintf(stdout,
               "GATHER_READ_AICPU_CONTROL_PROFILE avg_ns notify=%.0f notify_post=%.0f "
               "notify_cq=%.0f ack_poll=%.0f ack_post=%.0f ack_cq=%.0f "
               "ack_cache=%.0f ack_polls=%.2f\n",
               average(&kvaf::GatherReadTiming::notify_ns),
               average(&kvaf::GatherReadTiming::notify_post_ns),
               average(&kvaf::GatherReadTiming::notify_cq_ns),
               average(&kvaf::GatherReadTiming::ack_poll_ns),
               average(&kvaf::GatherReadTiming::ack_post_ns),
               average(&kvaf::GatherReadTiming::ack_cq_ns),
               average(&kvaf::GatherReadTiming::ack_cache_ns),
               average(&kvaf::GatherReadTiming::ack_polls));
  std::fprintf(stdout,
               "GATHER_READ_CPU_PROFILE avg_ns dispatch=%.0f notify_spin=%.0f "
               "source_cache=%.0f copy=%.0f staging_flush=%.0f\n",
               average(&kvaf::GatherReadTiming::cpu_dispatch_ns),
               average(&kvaf::GatherReadTiming::cpu_notify_spin_ns),
               average(&kvaf::GatherReadTiming::cpu_source_cache_ns),
               average(&kvaf::GatherReadTiming::cpu_copy_ns),
               average(&kvaf::GatherReadTiming::cpu_staging_flush_ns));
  std::fprintf(stdout,
               "GATHER_READ_AICPU_DATA_PROFILE avg_ns scatter_cache=%.0f "
               "scatter_plan=%.0f scatter_post=%.0f scatter_cq=%.0f read_wqes=%.2f\n",
               average(&kvaf::GatherReadTiming::scatter_cache_ns),
               average(&kvaf::GatherReadTiming::scatter_plan_ns),
               average(&kvaf::GatherReadTiming::scatter_post_ns),
               average(&kvaf::GatherReadTiming::scatter_cq_ns),
               average(&kvaf::GatherReadTiming::read_wqes));
  std::fprintf(stdout,
               "AICPU_GATHER_READ_PASS blocks=%u block_bytes=%u bytes=%zu "
               "output_stride=%u sgl_size=%u warmup=%u iterations=%u\n",
               options.blocks, options.block_bytes, payload, options.output_stride,
               options.sgl_size, options.warmup, options.iterations);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    Usage(argv[0]);
    return 0;
  }
  Options options;
  if (!Parse(argc, argv, &options)) {
    Usage(argv[0]);
    return 2;
  }
  try {
    return Run(options);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[KVAF_AICPU_URMA] GATHER_READ_FAIL %s\n", error.what());
    return 1;
  }
}
