#include "npu_runtime.h"
#include "aicpu_urma_client_metrics.h"
#include "aicpu_urma_client_options.h"

#include "acl/acl.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using kvaf::aicpu_urma_client::Options;
using kvaf::aicpu_urma_client::Parse;
using kvaf::aicpu_urma_client::DefaultKernelJson;
using kvaf::aicpu_urma_client::ConfigureCannEnvironment;
using kvaf::aicpu_urma_client::SummarizeTiming;

void Usage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s [--peer HOST:PORT] "
               "[--kernel-json PATH] [--device-id N] "
               "[--peer-device cpu|npu] [--peer-device-id N] "
               "[--blocks N] [--block-bytes N] [--token-bytes N] "
               "[--warmup N] [--iterations N]\n"
               "Defaults: peer=127.0.0.1:28999, device-id=0, peer-device=cpu, "
               "peer-device-id=0 (NPU peer only), blocks=8, "
               "block-bytes=4096, token-bytes=block-bytes, warmup=1, iterations=3, "
               "kernel-json=%s\n",
               program, DefaultKernelJson().c_str());
}

void Acl(aclError result, const char* operation) {
  if (result != ACL_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed acl=" + std::to_string(result));
}

void Check(const kvaf::Status& status, const char* operation) {
  if (!status.ok())
    throw std::runtime_error(std::string(operation) + " failed: " + status.message());
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
  AclContext(const AclContext&) = delete;
  AclContext& operator=(const AclContext&) = delete;
  std::uint32_t device_id;
  bool active_device{false};
  aclrtStream stream{nullptr};
};

struct DeviceBuffer {
  explicit DeviceBuffer(std::size_t bytes) : bytes(bytes) {
    Acl(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    if (data == nullptr) throw std::runtime_error("aclrtMalloc returned null");
  }
  ~DeviceBuffer() { if (data != nullptr) (void)aclrtFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void Upload(const void* source) {
    Acl(aclrtMemcpy(data, bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
  }
  void Download(void* destination) const {
    Acl(aclrtMemcpy(destination, bytes, data, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
  }
  void Clear() { Acl(aclrtMemset(data, bytes, 0, bytes), "aclrtMemset"); }

  void* data{nullptr};
  std::size_t bytes;
};

struct Buffers {
  explicit Buffers(const Options& options)
      : blocks(options.blocks), tokens(options.block_bytes / options.token_bytes),
        payload_bytes(static_cast<std::size_t>(options.blocks) * options.block_bytes),
        request_count(static_cast<std::size_t>(blocks) * tokens),
        input(payload_bytes), output(payload_bytes),
        addresses(blocks * sizeof(kvaf::Address)), valid(blocks),
        indices(request_count * sizeof(std::int32_t)),
        offsets(request_count * sizeof(std::int32_t)), mask(request_count),
        output_indices(request_count * sizeof(std::int32_t)),
        host_input(payload_bytes), host_output(payload_bytes),
        host_indices(request_count), host_offsets(request_count),
        host_output_indices(request_count), host_mask(request_count, 1U) {
    for (std::size_t row = 0; row < request_count; ++row) {
      host_indices[row] = static_cast<std::int32_t>(row / tokens);
      host_offsets[row] = static_cast<std::int32_t>((row % tokens) * options.token_bytes);
      host_output_indices[row] = static_cast<std::int32_t>(request_count - 1U - row);
    }
    indices.Upload(host_indices.data());
    offsets.Upload(host_offsets.data());
    mask.Upload(host_mask.data());
    output_indices.Upload(host_output_indices.data());
  }

  std::uint32_t blocks;
  std::uint32_t tokens;
  std::size_t payload_bytes;
  std::size_t request_count;
  DeviceBuffer input, output, addresses, valid, indices, offsets, mask, output_indices;
  std::vector<std::uint8_t> host_input, host_output;
  std::vector<std::int32_t> host_indices, host_offsets, host_output_indices;
  std::vector<std::uint8_t> host_mask;
};

kvaf::ConstBufferView ConstView(const DeviceBuffer& buffer) {
  return {buffer.data, buffer.bytes, kvaf::DeviceKind::kNpu};
}

void FillPayload(Buffers* buffers, std::uint64_t round) {
  for (std::size_t byte = 0; byte < buffers->payload_bytes; ++byte)
    buffers->host_input[byte] = static_cast<std::uint8_t>(
        (byte * 131U + (byte >> 8U) * 17U + round * 29U) & 0xffU);
  buffers->input.Upload(buffers->host_input.data());
}

kvaf::BatchPutRequest MakePut(const Buffers& buffers, const Options& options) {
  kvaf::BatchPutGroup group;
  group.input = ConstView(buffers.input);
  group.keys.reserve(options.blocks);
  group.input_offsets.reserve(options.blocks);
  group.lengths.reserve(options.blocks);
  for (std::uint32_t block = 0; block < options.blocks; ++block) {
    group.keys.push_back(static_cast<kvaf::KeyId>(block + 1U));
    group.input_offsets.push_back(static_cast<std::uint64_t>(block) * options.block_bytes);
    group.lengths.push_back(options.block_bytes);
  }
  kvaf::BatchPutRequest request;
  request.groups.push_back(std::move(group));
  return request;
}

void UploadAccessTable(kvaf::RuntimeCore& core, const kvaf::AccessTableHandle& handle,
                       Buffers* buffers, const Options& options) {
  kvaf::AccessTableView view{};
  Check(core.GetAccessTableView(handle, &view), "GetAccessTableView");
  if (view.size != options.blocks)
    throw std::runtime_error("AccessTable entry count mismatch");
  for (std::size_t block = 0; block < view.size; ++block) {
    if (view.valid[block] != 1U || view.lengths[block] != options.block_bytes)
      throw std::runtime_error("AccessTable entry invalid at block=" + std::to_string(block));
  }
  buffers->addresses.Upload(view.addresses);
  buffers->valid.Upload(view.valid);
}

kvaf::BatchGetRequest MakeGet(const Buffers& buffers, const Options& options,
                              const kvaf::AccessTableHandle& handle) {
  kvaf::BatchGetRequest request;
  request.access_table = handle;
  request.indices = ConstView(buffers.indices);
  request.offsets = ConstView(buffers.offsets);
  request.mask = ConstView(buffers.mask);
  request.output_indices = ConstView(buffers.output_indices);
  request.output = {buffers.output.data, buffers.output.bytes, kvaf::DeviceKind::kNpu};
  request.output_block_bytes = options.token_bytes;
  request.tokens_per_block = buffers.tokens;
  request.device_access_table = {ConstView(buffers.addresses), ConstView(buffers.valid)};
  return request;
}

void Verify(const Buffers& buffers, const Options& options, std::uint64_t round) {
  for (std::size_t row = 0; row < buffers.request_count; ++row) {
    const std::size_t source_row = buffers.request_count - 1U - row;
    const std::size_t source_offset = source_row * options.token_bytes;
    const std::size_t output_offset = row * options.token_bytes;
    for (std::size_t byte = 0; byte < options.token_bytes; ++byte) {
      if (buffers.host_output[output_offset + byte] != buffers.host_input[source_offset + byte])
        throw std::runtime_error("BatchGet payload mismatch round=" + std::to_string(round) +
                                 " row=" + std::to_string(row) +
                                 " byte=" + std::to_string(byte));
    }
  }
}

struct Timing { double put_ms{0.0}; double get_ms{0.0}; };

Timing RunRound(kvaf::NpuRuntime& runtime, aclrtStream stream, Buffers* buffers,
                const Options& options, std::uint64_t round) {
  FillPayload(buffers, round);
  kvaf::BatchPutRequest put = MakePut(*buffers, options);
  auto start = std::chrono::steady_clock::now();
  Check(runtime.core().BatchPut(put, kvaf::BatchPutExecutionOptions{stream}), "BatchPut");
  const auto after_put = std::chrono::steady_clock::now();
  kvaf::AccessTableHandle handle{};
  Check(runtime.core().PrepareAccessTable({put.groups.front().keys}, &handle), "PrepareAccessTable");
  UploadAccessTable(runtime.core(), handle, buffers, options);
  buffers->output.Clear();
  const kvaf::BatchGetRequest get = MakeGet(*buffers, options, handle);
  const auto before_get = std::chrono::steady_clock::now();
  Check(runtime.core().BatchGet(get, kvaf::BatchGetExecutionOptions{stream, true}), "BatchGet");
  const auto after_get = std::chrono::steady_clock::now();
  buffers->output.Download(buffers->host_output.data());
  Verify(*buffers, options, round);
  Check(runtime.core().ReleaseAccessTable(handle), "ReleaseAccessTable");
  Check(runtime.core().BatchFree({put.groups.front().keys}), "BatchFree");
  return {std::chrono::duration<double, std::milli>(after_put - start).count(),
          std::chrono::duration<double, std::milli>(after_get - before_get).count()};
}

int Run(const Options& options) {
  AclContext acl(options.device_id);
  Buffers buffers(options);
  kvaf::NpuRuntimeOptions runtime_options("", options.token_bytes,
                                          options.block_bytes / options.token_bytes);
  runtime_options.aicpu_urma_kernel_json_path = options.kernel_json;
  runtime_options.device_id = options.device_id;
  runtime_options.peer_device = options.peer_device == "npu"
                                    ? kvaf::DeviceKind::kNpu : kvaf::DeviceKind::kCpu;
  runtime_options.peer_device_id = options.peer_device_id;
  runtime_options.npu_core_count = 1;
  runtime_options.jettys_per_core = 1;
  kvaf::NpuRuntime runtime(std::move(runtime_options));
  Check(runtime.LinkTo({options.peer}), "LinkTo");
  Check(runtime.Initialize(buffers.payload_bytes, options.block_bytes), "Initialize");
  std::vector<double> put_samples;
  std::vector<double> get_samples;
  put_samples.reserve(options.iterations);
  get_samples.reserve(options.iterations);
  const std::uint64_t total = static_cast<std::uint64_t>(options.warmup) + options.iterations;
  std::fprintf(stdout, "[KVAF_AICPU_URMA] linked peer=%s device=%u peer_device=%s "
                       "peer_device_id=%u blocks=%u block_bytes=%u "
                       "token_bytes=%u warmup=%u iterations=%u\n",
               options.peer.c_str(), options.device_id, options.peer_device.c_str(),
               options.peer_device_id, options.blocks, options.block_bytes,
               options.token_bytes, options.warmup, options.iterations);
  for (std::uint64_t round = 0; round < total; ++round) {
    Timing sample;
    try {
      sample = RunRound(runtime, acl.stream, &buffers, options, round);
    } catch (const std::exception& error) {
      throw std::runtime_error("round=" + std::to_string(round) + ": " + error.what());
    }
    if (round >= options.warmup) {
      put_samples.push_back(sample.put_ms);
      get_samples.push_back(sample.get_ms);
    }
  }
  Check(runtime.Shutdown(), "Shutdown");
  const auto put_stats = SummarizeTiming(std::move(put_samples), buffers.payload_bytes);
  const auto get_stats = SummarizeTiming(std::move(get_samples), buffers.payload_bytes);
  std::fprintf(stdout, "PUT_STATS min_ms=%.3f p50_ms=%.3f p95_ms=%.3f p99_ms=%.3f max_ms=%.3f\n",
               put_stats.min_ms, put_stats.p50_ms, put_stats.p95_ms, put_stats.p99_ms,
               put_stats.max_ms);
  std::fprintf(stdout, "GET_STATS min_ms=%.3f p50_ms=%.3f p95_ms=%.3f p99_ms=%.3f max_ms=%.3f\n",
               get_stats.min_ms, get_stats.p50_ms, get_stats.p95_ms, get_stats.p99_ms,
               get_stats.max_ms);
  std::fprintf(stdout, "AICPU_URMA_PASS iterations=%u warmup=%u bytes_per_iteration=%zu "
                       "put_mean_ms=%.3f get_mean_ms=%.3f put_GBps=%.3f get_GBps=%.3f "
                       "peer_device=%s peer_device_id=%u\n",
               options.iterations, options.warmup, buffers.payload_bytes,
               put_stats.mean_ms, get_stats.mean_ms, put_stats.mean_gbps, get_stats.mean_gbps,
               options.peer_device.c_str(), options.peer_device_id);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    Usage(argv[0]);
    return 0;
  }
  Options options;
  const std::vector<const char*> arguments(argv, argv + argc);
  if (!Parse(argc, arguments.data(), &options)) {
    Usage(argv[0]);
    return 2;
  }
  try {
    if (!ConfigureCannEnvironment())
      throw std::runtime_error("failed to set ASCEND_HOME_PATH for the CANN package loader");
    return Run(options);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[KVAF_AICPU_URMA] FAIL: %s\n", error.what());
    return 1;
  }
}
