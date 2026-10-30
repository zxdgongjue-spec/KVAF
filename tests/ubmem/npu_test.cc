#include "ubmem/access_operator.h"
#include "test_pattern.h"
#include "ubmem_allocator.h"
#include "npu_runtime.h"
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace {
void Require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
void Check(const kvaf::Status& status) { Require(status.ok(), status.message()); }
void Acl(aclError result, const char* op) {
  if (result == ACL_SUCCESS) return;
  const char* detail = aclGetRecentErrMsg();
  if (detail && *detail) std::fprintf(stderr, "[KVAF_UBMEM] %s detail: %s\n", op, detail);
  throw std::runtime_error(std::string(op) + ": " + std::to_string(result));
}
std::uint8_t Pattern(std::uint64_t offset, unsigned seed) { return kvaf::ubmem::TestPattern(offset, seed); }

struct DevicePool {
  aclrtStream stream{};
  std::vector<void*> allocations;
  std::vector<void*> host_allocations;
  DevicePool() { Acl(aclrtCreateStream(&stream), "aclrtCreateStream"); }
  ~DevicePool() {
    const auto ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
      // The process is exiting. Do not free caller buffers still referenced
      // by failed work, or destroy the stream needed to drain that work.
      std::fprintf(stderr, "[KVAF_UBMEM] retaining caller buffers/stream after failed drain, ret=%d\n", ret);
      return;
    }
    for (void* ptr : allocations) aclrtFree(ptr);
    for (void* ptr : host_allocations) aclrtFreeHost(ptr);
    aclrtDestroyStream(stream);
  }
  void* Host(std::size_t bytes) {
    void* result = nullptr;
    Acl(aclrtMallocHost(&result, bytes), "aclrtMallocHost");
    host_allocations.push_back(result);
    return result;
  }
  template <typename T> T* Copy(const std::vector<T>& data) {
    void* out = nullptr;
    const auto bytes = data.size() * sizeof(T);
    Acl(aclrtMalloc(&out, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    allocations.push_back(out);
    Acl(aclrtMemcpy(out, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D metadata/input");
    return static_cast<T*>(out);
  }
};
struct Config {
  std::string peer, get_kernel, put_kernel;
  std::uint32_t device = 0, cores = 2, rows = 17, token = 656, tokens = 128, repeat = 2;
};

void FullBlocks(kvaf::RuntimeCore& core, kvaf::peer::PeerLink* link, DevicePool& pool,
                const Config& c, unsigned seed) {
  const std::uint64_t slot = static_cast<std::uint64_t>(c.token) * c.tokens;
  const auto base = core.remote_region().base;
  std::vector<kvaf::KeyId> keys(c.rows);
  std::vector<std::uint64_t> offsets(c.rows), lengths(c.rows, slot);
  std::vector<std::uint8_t> input(c.rows * (slot + 13) + 3, 0xEF);
  for (std::uint32_t i = 0; i < c.rows; ++i) {
    keys[i] = c.rows - 1 - i; offsets[i] = i * (slot + 13) + 3;
    for (std::uint64_t b = 0; b < slot; ++b) input[offsets[i] + b] = Pattern(keys[i] * slot + b, seed);
  }
  auto* source = pool.Copy(input);
  Check(core.BatchPut({keys, {source, input.size(), kvaf::DeviceKind::kNpu}, offsets, lengths}, {pool.stream}));
  if (link) for (const auto key : keys) Check(link->UbmemRange(base, key * slot, slot, seed, false));
  std::puts(link ? "[KVAF_UBMEM] BATCH_PUT_CPU_VERIFIED" : "[KVAF_UBMEM] RUNTIME_BATCH_PUT_COMPLETED");

  kvaf::AccessTableHandle handle;
  Check(core.PrepareAccessTable({keys}, &handle));
  kvaf::AccessTableView view;
  Check(core.GetAccessTableView(handle, &view));
  std::vector<kvaf::Address> table(view.addresses, view.addresses + view.size);
  std::vector<std::uint8_t> valid(view.valid, view.valid + view.size);
  auto* addresses_device = pool.Copy(table);
  auto* valid_device = pool.Copy(valid);
  const auto read_seed = link ? seed + 37 : seed;
  // Independent writer: CPU changes DDR after NPU PUT before each GET.
  if (link) for (const auto key : keys) Check(link->UbmemRange(base, key * slot, slot, read_seed, true));
  std::vector<std::int32_t> indices(c.rows + 3, -1), byte_offsets(c.rows + 3, -1), outputs(c.rows + 3, -1);
  for (std::uint32_t i = 0; i < c.rows; ++i) {
    indices[i] = (i * 5U) % c.rows;
    byte_offsets[i] = (i % c.tokens) * c.token;
    outputs[i] = c.rows - 1 - i;
  }
  auto* indices_device = pool.Copy(indices);
  auto* offsets_device = pool.Copy(byte_offsets);
  auto* outputs_device = pool.Copy(outputs);
  auto* count_device = pool.Copy(std::vector<std::int32_t>{static_cast<std::int32_t>(c.rows)});
  std::vector<std::uint8_t> output(c.rows * c.token + c.token, 0xD3);
  auto* output_device = pool.Copy(output);
  kvaf::BatchGetRequest request;
  request.access_table = handle;
  request.indices = {indices_device, indices.size() * 4, kvaf::DeviceKind::kNpu};
  request.offsets = {offsets_device, byte_offsets.size() * 4, kvaf::DeviceKind::kNpu};
  request.output_indices = {outputs_device, outputs.size() * 4, kvaf::DeviceKind::kNpu};
  request.miss_count = {count_device, 4, kvaf::DeviceKind::kNpu};
  request.output = {output_device, output.size(), kvaf::DeviceKind::kNpu};
  request.output_block_bytes = c.token; request.tokens_per_block = c.tokens;
  request.device_access_table = {{addresses_device, table.size() * 8, kvaf::DeviceKind::kNpu},
                                 {valid_device, valid.size(), kvaf::DeviceKind::kNpu}};
  Check(core.BatchGet(request, {pool.stream, false}));
  // Copy on the same caller stream verifies BatchGet enqueue ordering.
  auto* host_output = static_cast<std::uint8_t*>(pool.Host(output.size()));
  Acl(aclrtMemcpyAsync(host_output, output.size(), output_device, output.size(),
                       ACL_MEMCPY_DEVICE_TO_HOST, pool.stream), "D2H after BatchGet");
  Acl(aclrtSynchronizeStream(pool.stream), "BatchGet synchronize");
  std::copy(host_output, host_output + output.size(), output.begin());
  for (std::uint32_t i = 0; i < c.rows; ++i) {
    for (std::uint32_t b = 0; b < c.token; ++b) {
      const auto expected = Pattern(keys[indices[i]] * slot + byte_offsets[i] + b, read_seed);
      Require(output[static_cast<std::size_t>(outputs[i]) * c.token + b] == expected,
              "BatchGet payload differs from CPU-written DDR");
    }
  }
  for (std::size_t i = c.rows * c.token; i < output.size(); ++i) Require(output[i] == 0xD3, "output guard overwritten");
  aclmdlRI graph{};
  Acl(aclmdlRICaptureBegin(pool.stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "capture begin");
  const auto capture_status = core.BatchGet(request, {pool.stream, false});
  Acl(aclmdlRICaptureEnd(pool.stream, &graph), "capture end");
  Check(capture_status);
  for (const std::int32_t active : {static_cast<std::int32_t>(c.rows), 1, 0}) {
    Acl(aclrtMemset(output_device, output.size(), 0xD3, output.size()), "reset graph output");
    Acl(aclrtMemcpy(count_device, 4, &active, 4, ACL_MEMCPY_HOST_TO_DEVICE), "graph miss_count");
    Acl(aclmdlRIExecuteAsync(graph, pool.stream), "graph replay");
    Acl(aclrtSynchronizeStream(pool.stream), "graph synchronize");
    Acl(aclrtMemcpy(output.data(), output.size(), output_device, output.size(),
                   ACL_MEMCPY_DEVICE_TO_HOST), "graph output");
    for (std::uint32_t row = 0; row < c.rows + 1; ++row) {
      const std::uint32_t request_index = c.rows - 1 - row;
      for (std::uint32_t b = 0; b < c.token; ++b) {
        const auto expected = request_index < static_cast<std::uint32_t>(active)
            ? Pattern(keys[indices[request_index]] * slot + byte_offsets[request_index] + b, read_seed)
            : static_cast<std::uint8_t>(0xD3);
        Require(output[row * c.token + b] == expected, "graph replay payload/guard mismatch");
      }
    }
  }
  Acl(aclmdlRIDestroy(graph), "graph destroy");
  Check(core.BatchGet(request, {pool.stream, true}));  // Zero misses on the eager path too.
  Check(core.ReleaseAccessTable(handle));
  Check(core.BatchFree({keys}));
  std::puts(link ? "[KVAF_UBMEM] BATCH_GET_CPU_SOURCE_VERIFIED" : "[KVAF_UBMEM] RUNTIME_BATCH_GET_VERIFIED");
}

void VariablePut(kvaf::RuntimeCore& core, kvaf::peer::PeerLink* link, DevicePool& pool,
                 const Config& c, unsigned seed) {
  const std::uint64_t slot = static_cast<std::uint64_t>(c.token) * c.tokens;
  std::vector<std::uint64_t> lengths{1, std::min<std::uint64_t>(655, slot), std::min<std::uint64_t>(4097, slot)};
  std::vector<std::uint64_t> offsets{3, 19, 19 + lengths[1] + 11};
  std::vector<std::uint8_t> input(offsets.back() + lengths.back());
  for (unsigned row = 0; row < 3; ++row) for (std::uint64_t b = 0; b < lengths[row]; ++b)
    input[offsets[row] + b] = Pattern(row * slot + b, seed);
  auto* source = pool.Copy(input);
  Check(core.BatchPut({{0, 1, 2}, {source, input.size(), kvaf::DeviceKind::kNpu}, offsets, lengths}, {pool.stream}));
  if (link) for (unsigned row = 0; row < 3; ++row)
    Check(link->UbmemRange(core.remote_region().base, row * slot, lengths[row], seed, false));
  Check(core.BatchFree({{0, 1, 2}}));
  std::puts(link ? "[KVAF_UBMEM] VARIABLE_BATCH_PUT_CPU_VERIFIED" : "[KVAF_UBMEM] RUNTIME_VARIABLE_PUT_COMPLETED");
}

Config Parse(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    Require(i + 1 < argc, "missing value for " + key);
    const char* value = argv[++i];
    if (key == "--peer") c.peer = value;
    else if (key == "--get-kernel") c.get_kernel = value;
    else if (key == "--put-kernel") c.put_kernel = value;
    else {
      char* end; errno = 0; const auto n = std::strtoul(value, &end, 10);
      Require(!errno && end != value && !*end && n <= UINT32_MAX && value[0] != '-', "invalid " + key);
      if (key == "--device") c.device = n;
      else if (key == "--cores") c.cores = n;
      else if (key == "--rows") c.rows = n;
      else if (key == "--token-bytes") c.token = n;
      else if (key == "--tokens-per-block") c.tokens = n;
      else if (key == "--repeat") c.repeat = n;
      else throw std::runtime_error("unknown option " + key);
    }
  }
  Require(!c.peer.empty() && !c.get_kernel.empty() && !c.put_kernel.empty(), "required: --peer CPU:PORT --get-kernel PATH --put-kernel PATH");
  Require(c.rows >= 3 && c.rows <= 65536 && c.token && c.tokens && c.repeat && c.cores && c.cores <= 32,
          "rows must be 3..65536, cores 1..32, and shape/repeat must be positive");
  Require(static_cast<std::uint64_t>(c.token) * c.tokens <= INT32_MAX &&
          static_cast<std::uint64_t>(c.token) * c.rows <= SIZE_MAX, "test shape exceeds ABI");
  return c;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::puts("Usage: kvaf_ubmem_test --peer CPU:PORT --get-kernel PATH --put-kernel PATH [--device 0] [--cores 2] [--rows 17] "
              "[--token-bytes 656] [--tokens-per-block 128] [--repeat 2]"); return 0;
  }
  try {
    const auto c = Parse(argc, argv);
    Acl(aclInit(nullptr), "aclInit");
    Acl(aclrtSetDevice(c.device), "aclrtSetDevice");
    {
      // Declared first so the caller stream and buffers outlive RuntimeCore
      // and AccessOperator during both normal and exception unwinding.
      DevicePool buffers;
      auto link = std::make_shared<kvaf::peer::PeerLink>(c.peer);
      kvaf::ubmem::DdrAllocator allocator(link);
      kvaf::ubmem::AccessOperator op({c.get_kernel, c.put_kernel, c.device, c.cores, link});
      auto meta = kvaf::CreateInMemoryMetaManager();
      kvaf::RuntimeCore core(*meta, op, allocator);
      const auto slot = static_cast<std::uint64_t>(c.token) * c.tokens;
      Check(core.Initialize({slot * c.rows, slot, kvaf::DeviceKind::kCpu}));
      for (unsigned iteration = 0; iteration < c.repeat; ++iteration) {
        FullBlocks(core, link.get(), buffers, c, 17 + iteration * 13);
        VariablePut(core, link.get(), buffers, c, 91 + iteration * 13);
      }
      // RuntimeCore shuts down the operator before releasing remote DDR.
      Check(core.Shutdown());
    }
    {
      // Exercise the production NpuRuntime path too, without EID/Jetty discovery.
      DevicePool buffers;
      kvaf::NpuRuntimeOptions options(c.get_kernel, c.token, c.tokens);
      options.batch_put_kernel_object_path = c.put_kernel;
      options.transport = kvaf::Transport::kUbmem;
      options.device_id = c.device;
      options.npu_core_count = c.cores;
      kvaf::NpuRuntime runtime(options);
      Check(runtime.LinkTo({c.peer}));
      const auto slot = static_cast<std::uint64_t>(c.token) * c.tokens;
      Check(runtime.Initialize(slot * c.rows, slot));
      FullBlocks(runtime.core(), nullptr, buffers, c, 53);
      VariablePut(runtime.core(), nullptr, buffers, c, 71);
      Check(runtime.Shutdown());
    }
    Acl(aclrtResetDevice(c.device), "aclrtResetDevice");
    Acl(aclFinalize(), "aclFinalize");
    std::puts("[SUCCESS] UBmem CPU verification and NpuRuntime BatchPut/BatchGet passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[KVAF_UBMEM] TEST_FAILED: %s\n", error.what());
    return 1;
  }
}
