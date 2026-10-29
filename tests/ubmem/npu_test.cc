#include "ubmem/access_operator.h"
#include "ubmem/test_pattern.h"
#include "ubmem_runtime.h"
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
  struct VmmAllocation {
    void* address{};
    aclrtDrvMemHandle handle{};
    bool mapped{};
  };
  aclrtStream stream{};
  std::vector<void*> allocations;
  std::vector<void*> host_allocations;
  std::vector<VmmAllocation> vmm_allocations;
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
    for (const auto& vmm : vmm_allocations) {
      // Keep dependent resources if a cleanup operation fails. No VMM
      // resources are released at all when the stream cannot be drained.
      if (vmm.mapped && aclrtUnmapMem(vmm.address) != ACL_SUCCESS) {
        std::fprintf(stderr, "[KVAF_UBMEM] retaining local VMM allocation after unmap failure\n");
        continue;
      }
      if (vmm.handle && aclrtFreePhysical(vmm.handle) != ACL_SUCCESS) {
        std::fprintf(stderr, "[KVAF_UBMEM] retaining local VMM allocation after physical release failure\n");
        continue;
      }
      if (vmm.address && aclrtReleaseMemAddress(vmm.address) != ACL_SUCCESS)
        std::fprintf(stderr, "[KVAF_UBMEM] local VMM address release failed\n");
    }
    aclrtDestroyStream(stream);
  }
  void* Vmm(std::size_t bytes, std::uint32_t device, bool dram) {
    const auto aligned = (bytes + kvaf::ubmem::kPageBytes - 1) / kvaf::ubmem::kPageBytes * kvaf::ubmem::kPageBytes;
    vmm_allocations.emplace_back();
    auto& vmm = vmm_allocations.back();
    // SHMEM AclMemSegmentDevice::ReserveEachPeMemorySpace / AllocLocalMemory.
    // Register ownership before calling ACL so exceptions release partial setup.
    Acl(aclrtReserveMemAddress(&vmm.address, aligned, 0, nullptr, 1), "control aclrtReserveMemAddress");
    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = dram ? ACL_DDR_MEM_P2P_HUGE : ACL_HBM_MEM_HUGE;
    prop.location.type = dram ? ACL_MEM_LOCATION_TYPE_HOST_NUMA : ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dram ? UINT32_MAX : device;
    Acl(aclrtMallocPhysical(&vmm.handle, aligned, &prop, 0), "control aclrtMallocPhysical");
    Acl(aclrtMapMem(vmm.address, aligned, 0, vmm.handle, 0), "control aclrtMapMem");
    vmm.mapped = true;
    if (dram) {
      // A locally allocated Host handle initially belongs to the CPU. SHMEM
      // explicitly grants the local NPU access; imported handles differ here.
      aclrtMemAccessDesc access{};
      access.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
      access.location.id = device;
      access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
      Acl(aclrtMemSetAccess(vmm.address, aligned, &access, 1), "control aclrtMemSetAccess");
    }
    std::fprintf(stderr, "[KVAF_UBMEM] VMM_CONTROL_READY kind=%s base=%p bytes=%zu\n",
        dram ? "dram" : "hbm", vmm.address, aligned);
    return vmm.address;
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
  std::string peer, kernel, probe_vmm;
  kvaf::ubmem::TransferEngine engine = kvaf::ubmem::TransferEngine::kMte;
  std::uint32_t device = 0, cores = 2, rows = 17, token = 656, tokens = 128, repeat = 2;
};

void FullBlocks(kvaf::RuntimeCore& core, kvaf::ubmem::AccessOperator& op,
                kvaf::peer::PeerLink& link, DevicePool& pool,
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
  // Control for mapped GM faults: use the same source bytes, per-row offsets,
  // lengths and cores, but copy into local HBM instead of mapped CPU DDR.
  std::vector<std::uint64_t> probe_metadata(3 * c.rows);
  for (std::uint32_t i = 0; i < c.rows; ++i) {
    probe_metadata[i] = offsets[i];
    probe_metadata[c.rows + i] = lengths[i];
    probe_metadata[2 * c.rows + i] = base + keys[i] * slot;
  }
  std::vector<std::uint8_t> local_copy(c.rows * slot, 0xD3);
  auto* local_device = pool.Copy(local_copy);
  auto* probe_device = pool.Copy(probe_metadata);
  std::fprintf(stderr, "[KVAF_UBMEM] LOCAL_COPY_BEGIN rows=%u bytes=%zu remote_base=0x%llx\n",
      c.rows, local_copy.size(), static_cast<unsigned long long>(base));
  Check(op.ProbeLocalCopy({source, input.size(), kvaf::DeviceKind::kNpu},
      {local_device, local_copy.size(), kvaf::DeviceKind::kNpu},
      {probe_device, probe_metadata.size() * sizeof(std::uint64_t), kvaf::DeviceKind::kNpu}, pool.stream));
  Acl(aclrtMemcpy(local_copy.data(), local_copy.size(), local_device, local_copy.size(),
                  ACL_MEMCPY_DEVICE_TO_HOST), "D2H local copy probe");
  for (std::size_t i = 0; i < local_copy.size(); ++i) {
    if (local_copy[i] != Pattern(i, seed))
      throw std::runtime_error("local HBM copy payload mismatch at " + std::to_string(i));
  }
  std::fprintf(stderr, "[KVAF_UBMEM] LOCAL_COPY_VERIFIED\n");
  if (!c.probe_vmm.empty()) {
    std::fprintf(stderr, "[KVAF_UBMEM] VMM_CONTROL_ALLOCATE kind=%s\n", c.probe_vmm.c_str());
    void* destination = pool.Vmm(local_copy.size(), c.device, c.probe_vmm == "dram");
    std::fill(local_copy.begin(), local_copy.end(), 0xD3);
    // Same initialization/readback as SHMEM's rma_d2h_demo, including Host DDR.
    Acl(aclrtMemcpy(destination, local_copy.size(), local_copy.data(), local_copy.size(),
        ACL_MEMCPY_HOST_TO_DEVICE), "VMM control H2D initialization");
    std::fprintf(stderr, "[KVAF_UBMEM] VMM_CONTROL_COPY_BEGIN kind=%s engine=%s\n", c.probe_vmm.c_str(),
        c.engine == kvaf::ubmem::TransferEngine::kSimt ? "simt" : "mte");
    Check(op.ProbeLocalCopy({source, input.size(), kvaf::DeviceKind::kNpu},
        {destination, local_copy.size(), kvaf::DeviceKind::kNpu},
        {probe_device, probe_metadata.size() * sizeof(std::uint64_t), kvaf::DeviceKind::kNpu}, pool.stream));
    Acl(aclrtMemcpy(local_copy.data(), local_copy.size(), destination, local_copy.size(),
        ACL_MEMCPY_DEVICE_TO_HOST), "VMM control D2H verification");
    for (std::size_t i = 0; i < local_copy.size(); ++i)
      Require(local_copy[i] == Pattern(i, seed), "local VMM payload mismatch at " + std::to_string(i));
    std::fprintf(stderr, "[KVAF_UBMEM] VMM_CONTROL_VERIFIED kind=%s\n", c.probe_vmm.c_str());
    return;
  }
  std::fprintf(stderr, "[KVAF_UBMEM] MAPPED_PUT_BEGIN\n");
  Check(core.BatchPut({keys, {source, input.size(), kvaf::DeviceKind::kNpu}, offsets, lengths}, {pool.stream}));
  for (const auto key : keys) Check(link.UbmemRange(base, key * slot, slot, seed, false));
  std::puts("[KVAF_UBMEM] BATCH_PUT_CPU_VERIFIED");

  kvaf::AccessTableHandle handle;
  Check(core.PrepareAccessTable({keys}, &handle));
  kvaf::AccessTableView view;
  Check(core.GetAccessTableView(handle, &view));
  std::vector<kvaf::Address> table(view.addresses, view.addresses + view.size);
  std::vector<std::uint8_t> valid(view.valid, view.valid + view.size);
  auto* addresses_device = pool.Copy(table);
  auto* valid_device = pool.Copy(valid);
  const auto read_seed = seed + 37;
  // Independent writer: CPU changes DDR after NPU PUT before each GET.
  for (const auto key : keys) Check(link.UbmemRange(base, key * slot, slot, read_seed, true));
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
  const std::int32_t zero = 0;
  Acl(aclrtMemcpy(count_device, 4, &zero, 4, ACL_MEMCPY_HOST_TO_DEVICE), "zero miss_count");
  Check(core.BatchGet(request, {pool.stream, true}));
  Check(core.ReleaseAccessTable(handle));
  Check(core.BatchFree({keys}));
  std::puts("[KVAF_UBMEM] BATCH_GET_CPU_SOURCE_VERIFIED (offsets, output permutation, inactive tail, zero misses)");
}

void VariablePut(kvaf::RuntimeCore& core, kvaf::peer::PeerLink& link, DevicePool& pool,
                 const Config& c, unsigned seed) {
  const std::uint64_t slot = static_cast<std::uint64_t>(c.token) * c.tokens;
  std::vector<std::uint64_t> lengths{1, std::min<std::uint64_t>(655, slot), std::min<std::uint64_t>(4097, slot)};
  std::vector<std::uint64_t> offsets{3, 19, 19 + lengths[1] + 11};
  std::vector<std::uint8_t> input(offsets.back() + lengths.back());
  for (unsigned row = 0; row < 3; ++row) for (std::uint64_t b = 0; b < lengths[row]; ++b)
    input[offsets[row] + b] = Pattern(row * slot + b, seed);
  auto* source = pool.Copy(input);
  Check(core.BatchPut({{0, 1, 2}, {source, input.size(), kvaf::DeviceKind::kNpu}, offsets, lengths}, {pool.stream}));
  for (unsigned row = 0; row < 3; ++row)
    Check(link.UbmemRange(core.remote_region().base, row * slot, lengths[row], seed, false));
  Check(core.BatchFree({{0, 1, 2}}));
  std::puts("[KVAF_UBMEM] VARIABLE_BATCH_PUT_CPU_VERIFIED");
}

Config Parse(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    Require(i + 1 < argc, "missing value for " + key);
    const char* value = argv[++i];
    if (key == "--peer") c.peer = value;
    else if (key == "--kernel") c.kernel = value;
    else if (key == "--engine") {
      const std::string engine = value;
      Require(engine == "mte" || engine == "simt", "--engine must be mte or simt");
      c.engine = engine == "simt" ? kvaf::ubmem::TransferEngine::kSimt : kvaf::ubmem::TransferEngine::kMte;
    }
    else if (key == "--probe-vmm") {
      c.probe_vmm = value;
      Require(c.probe_vmm == "hbm" || c.probe_vmm == "dram", "--probe-vmm must be hbm or dram");
    }
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
  Require(!c.peer.empty() && !c.kernel.empty(), "required: --peer CPU:PORT --kernel PATH");
  Require(c.rows >= 3 && c.rows <= 65536 && c.token && c.tokens && c.repeat && c.cores && c.cores <= 32,
          "rows must be 3..65536, cores 1..32, and shape/repeat must be positive");
  Require(static_cast<std::uint64_t>(c.token) * c.tokens <= INT32_MAX &&
          static_cast<std::uint64_t>(c.token) * c.rows <= SIZE_MAX, "test shape exceeds ABI");
  return c;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::puts("Usage: kvaf_ubmem_test --peer CPU:PORT --kernel PATH [--device 0] [--cores 2] [--rows 17] "
              "[--token-bytes 656] [--tokens-per-block 128] [--repeat 2] "
              "[--engine mte|simt (default: mte)] [--probe-vmm hbm|dram]"); return 0;
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
      kvaf::ubmem::AccessOperator op({c.kernel, c.device, c.cores, link, c.engine});
      auto meta = kvaf::CreateInMemoryMetaManager();
      kvaf::RuntimeCore core(*meta, op, allocator);
      const auto slot = static_cast<std::uint64_t>(c.token) * c.tokens;
      Check(core.Initialize({slot * c.rows, slot, kvaf::DeviceKind::kCpu}));
      for (unsigned iteration = 0; iteration < c.repeat; ++iteration) {
        FullBlocks(core, op, *link, buffers, c, 17 + iteration * 13);
        if (c.probe_vmm.empty()) VariablePut(core, *link, buffers, c, 91 + iteration * 13);
      }
      // RuntimeCore shuts down the operator before releasing remote DDR.
      Check(core.Shutdown());
    }
    Acl(aclrtResetDevice(c.device), "aclrtResetDevice");
    Acl(aclFinalize(), "aclFinalize");
    if (c.probe_vmm.empty())
      std::puts("[SUCCESS] UBmem BatchPut/BatchGet passed with independent CPU DDR verification");
    else
      std::printf("[SUCCESS] local VMM %s control passed\n", c.probe_vmm.c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[KVAF_UBMEM] TEST_FAILED: %s\n", error.what());
    return 1;
  }
}
