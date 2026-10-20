#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "aicpu_gather_access_operator.h"
#include "aicpu_urma/host.h"
#include "aicpu_urma/umdk_abi.h"
#include "cpu_gather.h"
#include "meta_manager.h"
#include "peer_link.h"
#include "runtime.h"
using namespace kvaf;
using namespace kvaf::aicpu_urma;
extern "C" unsigned KvafAicpuGatherRead(GatherReadArgs*);
extern "C" unsigned KvafAicpuGatherScatter(GatherReadArgs*);
extern "C" unsigned KvafAicpuUmdkBatchGet(UmdkBatchGetArgs*);
namespace {
struct Stream {
  std::vector<std::function<int()>> tasks;
};
peer::PeerRegion region;
std::atomic<bool> stop{false}, data_seen{false}, overlap_seen{false}, cpu_error{false},
    gate_data{true};
std::atomic<unsigned> cpu_rounds{0};
std::uint64_t staging = 0, payload = 0, descriptor_address = 0;
unsigned channel_count = 0;
int fail_copy = -1;
bool fail_sync = false;
unsigned allocations = 0, frees = 0, task_errors = 0;
std::thread cpu;
void CpuLoop() {
  auto* control = reinterpret_cast<peer::GatherControl*>(region.base +
                                                         peer::GatherControlOffset(region.length));
  std::uint64_t previous = 0;
  while (!stop) {
    const auto sequence = __atomic_load_n(&control->notify, __ATOMIC_ACQUIRE);
    if (!sequence || sequence == previous) {
      std::this_thread::yield();
      continue;
    }
    previous = sequence;
    const auto address = __atomic_load_n(&control->command, __ATOMIC_ACQUIRE);
    auto* header = reinterpret_cast<peer::GatherDescriptorHeader*>(address);
    assert(header->magic == peer::kGatherDescriptorMagic && header->sequence == sequence);
    __atomic_store_n(&control->ack, peer::GatherStartedAck(sequence), __ATOMIC_RELEASE);
    if (cpu_error) {
      __atomic_store_n(&header->ready_bytes, peer::kGatherErrorBit, __ATOMIC_RELEASE);
      __atomic_store_n(&control->ack, sequence | peer::kGatherErrorBit, __ATOMIC_RELEASE);
      continue;
    }
    const auto* spans = reinterpret_cast<const peer::GatherSpan*>(header + 1);
    std::uint64_t copied = 0;
    unsigned source = 0, in_source = 0;
    while (copied < header->total_bytes) {
      const auto end =
          header->pipeline_chunk_bytes
              ? std::min<std::uint64_t>(header->total_bytes, copied + header->pipeline_chunk_bytes)
              : header->total_bytes;
      while (copied < end) {
        const auto bytes = std::min<std::uint64_t>(end - copied, spans[source].length - in_source);
        std::memcpy(reinterpret_cast<void*>(header->staging_address + copied),
                    reinterpret_cast<void*>(spans[source].address + in_source), bytes);
        copied += bytes;
        in_source += bytes;
        if (in_source == spans[source].length) {
          ++source;
          in_source = 0;
        }
      }
      __atomic_store_n(&header->ready_bytes, copied, __ATOMIC_RELEASE);
      if (gate_data && header->pipeline_chunk_bytes && copied >= header->total_bytes / 2 &&
          copied < header->total_bytes) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!data_seen && !stop && std::chrono::steady_clock::now() < deadline)
          std::this_thread::yield();
        assert(data_seen || stop);
      }
    }
    control->profile = {0, 0, 1000, 2000, 0};
    __atomic_store_n(&control->ack, sequence, __ATOMIC_RELEASE);
    ++cpu_rounds;
  }
}
void ObserveRead(std::uint64_t source, std::uint32_t bytes, bool) {
  if (source < staging || source - staging >= payload || !descriptor_address) return;
  const auto* header = reinterpret_cast<const peer::GatherDescriptorHeader*>(descriptor_address);
  if (header->pipeline_chunk_bytes) {
    assert(source - staging + bytes <= __atomic_load_n(&header->ready_bytes, __ATOMIC_ACQUIRE));
    const auto* control = reinterpret_cast<const peer::GatherControl*>(
        region.base + peer::GatherControlOffset(region.length));
    if (__atomic_load_n(&control->ack, __ATOMIC_ACQUIRE) != header->sequence) overlap_seen = true;
  }
  data_seen = true;
}
struct Allocator : DdrMemoryAllocator {
  Status Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* output) override {
    if (!request.slot_size || request.ddr_bytes % request.slot_size)
      return Status::InvalidArgument("invalid allocation");
    void* buffer = nullptr;
    assert(posix_memalign(&buffer, 4096, request.ddr_bytes) == 0);
    std::memset(buffer, 0, request.ddr_bytes);
    region = {};
    region.base = reinterpret_cast<std::uint64_t>(buffer);
    region.length = request.ddr_bytes;
    region.session_id = 1;
    region.token_id = 7;
    region.token_value = 123;
    region.eid[15] = 8;
    urma_seg_t segment{};
    segment.ubva.va = region.base;
    segment.len = region.length;
    region.segment_key_bytes = sizeof(segment);
    std::memcpy(region.segment_key.data(), &segment, sizeof(segment));
    *output = {region.base, region.length, request.slot_size, DeviceKind::kCpu};
    return Status::OK();
  }
  Status Release(const RemoteRegionDesc&) override {
    stop = true;
    if (cpu.joinable()) cpu.join();
    std::free(reinterpret_cast<void*>(region.base));
    region = {};
    return Status::OK();
  }
};
void Run(unsigned workers, bool pipeline, int (*live)()) {
  stop = data_seen = overlap_seen = cpu_error = false;
  cpu_rounds = channel_count = 0;
  staging = payload = descriptor_address = 0;
  auto link = std::make_shared<peer::PeerLink>("mock");
  UmdkAccessOperator op({"/dev/null", "", UINT32_MAX, workers, link});
  Allocator allocator;
  auto meta = CreateInMemoryMetaManager();
  RuntimeCore core(*meta, op, allocator);
  assert(core.Initialize({4U << 20U, 656U, DeviceKind::kCpu}).code() ==
         StatusCode::kInvalidArgument);
  assert(core.Initialize({656U * 8192U, 656U, DeviceKind::kCpu}).ok());
  aclrtStream stream = nullptr;
  assert(aclrtCreateStream(&stream) == 0);
  void* input = nullptr;
  void* output = nullptr;
  assert(aclrtMalloc(&input, 137U * 656U, ACL_MEM_MALLOC_HUGE_FIRST) == 0);
  assert(aclrtMalloc(&output, 2U << 20U, ACL_MEM_MALLOC_HUGE_FIRST) == 0);
  for (unsigned round = 0; round < 4; ++round) {
    for (unsigned index = 0; index < 137U * 656U; ++index)
      static_cast<unsigned char*>(input)[index] = index * 17 + round * 31;
    BatchPutRequest put;
    put.input = {input, 137U * 656U, DeviceKind::kNpu};
    for (unsigned block = 0; block < 137; ++block) {
      put.keys.push_back(block + 1);
      put.input_offsets.push_back(block * 656U);
      put.lengths.push_back(656);
    }
    assert(core.BatchPut(put, {stream}).ok());
    AccessTableHandle handle{};
    assert(core.PrepareAccessTable({put.keys}, &handle).ok());
    AccessTableView view{};
    assert(core.GetAccessTableView(handle, &view).ok());
    CpuGatherReadRequest request;
    request.output = {output, 2U << 20U, DeviceKind::kNpu};
    request.stream = stream;
    request.sq_batch_wqes = 8;
    request.pipeline_chunk_bytes = pipeline ? 4096 : 0;
    request.profile_detail = true;
    staging = region.base + 1U * 1024U * 1024U;
    payload = 137U * 656U;
    descriptor_address = region.base + 2U * 1024U * 1024U;
    request.staging_address = staging;
    request.source_descriptor_address = descriptor_address;
    std::vector<unsigned char> packed, expected(2U << 20U, 0xa5);
    for (unsigned block = 0; block < 137; ++block) {
      const auto index = (block + round * 13) % 137;
      request.spans.push_back({view.addresses[index], 656U, 0});
      const auto* source = static_cast<const unsigned char*>(input) + index * 656U;
      packed.insert(packed.end(), source, source + 656U);
    }
    std::size_t offset = 0;
    while (offset < packed.size()) {
      const auto length = std::min<std::size_t>(
          packed.size() - offset, 300U + (request.output_spans.size() * 13U + round * 71U) % 700U);
      const auto target = (300U - request.output_spans.size()) * 4096U;
      request.output_spans.push_back({target, static_cast<unsigned>(length), 0});
      std::memcpy(expected.data() + target, packed.data() + offset, length);
      offset += length;
    }
    gate_data = true;
    if (round == 2) {
      // A single >64 KiB target splits into two WRs, even with more workers than targets.
      request.output_spans = {{32U, static_cast<unsigned>(packed.size()), 0U}};
      std::fill(expected.begin(), expected.end(), 0xa5);
      std::memcpy(expected.data() + 32, packed.data(), packed.size());
      gate_data = false;  // This small two-WR batch needs the whole prefix before submission.
    } else if (round == 3) {
      // More than 4096 single-SGE WRs requires a final CQE before workspace reuse.
      request.sgl_size = 1;
      request.output_spans.clear();
      std::fill(expected.begin(), expected.end(), 0xa5);
      for (std::size_t byte = 0; byte < packed.size(); byte += 17) {
        const auto length = std::min<std::size_t>(17, packed.size() - byte);
        const auto target = request.output_spans.size() * 32U;
        request.output_spans.push_back({target, static_cast<unsigned>(length), 0});
        std::memcpy(expected.data() + target, packed.data() + byte, length);
      }
    }
    std::memset(output, 0xa5, 2U << 20U);
    data_seen = false;
    GatherReadTiming timing{};
    request.timing = &timing;
    assert(op.ExecuteGatherRead(request).ok());
    std::size_t output_extent = 0;
    for (const auto& target : request.output_spans)
      output_extent = std::max<std::size_t>(output_extent, target.offset + target.length);
    assert(std::memcmp(output, expected.data(), output_extent) == 0);
    std::uint64_t bytes = 0, wqes = 0;
    for (unsigned worker = 0; worker < workers; ++worker) {
      bytes += timing.workers[worker].read_bytes;
      wqes += timing.workers[worker].read_wqes;
    }
    assert(bytes == payload && wqes == timing.read_wqes && timing.aicpu_cores == workers);
    assert(timing.scatter_total_ns && timing.cpu_copy_ns == 2000);
    std::array<std::int32_t, 2> indices{1, 2}, offsets{5, 7}, output_indices{1, 0};
    std::array<std::uint8_t, 2> mask{1, 0};
    BatchGetRequest get;
    get.access_table = handle;
    get.indices = {indices.data(), sizeof(indices), DeviceKind::kNpu};
    get.offsets = {offsets.data(), sizeof(offsets), DeviceKind::kNpu};
    get.output_indices = {output_indices.data(), sizeof(output_indices), DeviceKind::kNpu};
    get.mask = {mask.data(), sizeof(mask), DeviceKind::kNpu};
    get.output = {output, 2U << 20U, DeviceKind::kNpu};
    get.output_block_bytes = 16;
    get.tokens_per_block = 41;
    get.device_access_table.addresses = {view.addresses, view.size * sizeof(Address),
                                         DeviceKind::kNpu};
    get.device_access_table.valid = {view.valid, view.size, DeviceKind::kNpu};
    std::memset(output, 0xa5, 32);
    assert(core.BatchGet(get, {stream, false}).ok());
    for (unsigned byte = 0; byte < 32; ++byte)
      assert(static_cast<unsigned char*>(output)[byte] == 0xa5);
    assert(aclrtSynchronizeStream(stream) == 0);
    assert(std::memcmp(static_cast<unsigned char*>(output) + 16,
                       static_cast<unsigned char*>(input) + 656 + 5, 16) == 0);
    for (unsigned byte = 0; byte < 16; ++byte)
      assert(static_cast<unsigned char*>(output)[byte] == 0xa5);
    assert(core.ReleaseAccessTable(handle).ok());
    if (round == 1) {
      fail_copy = ACL_MEMCPY_HOST_TO_DEVICE;
      assert(!op.ExecuteGatherRead(request)
                  .ok());  // A failed enqueue retains the pinned workspace until Drain.
    }
    if (round == 2) {
      fail_sync = true;
      const auto before = frees;
      assert(!op.ExecuteGatherRead(request).ok());
      assert(!core.Shutdown().ok() && core.initialized() && frees == before);
      fail_sync = false;
    }
    assert(core.BatchFree({put.keys}).ok());
  }
  if (pipeline) assert(overlap_seen);
  cpu_error = true;
  CpuGatherReadRequest bad;
  bad.spans = {{region.base, 656U, 0U}};
  bad.output_spans = {{0U, 656U, 0U}};
  bad.staging_address = staging;
  bad.source_descriptor_address = descriptor_address;
  bad.output = {output, 2U << 20U, DeviceKind::kNpu};
  bad.stream = stream;
  bad.pipeline_chunk_bytes = pipeline ? 4096U : 0U;
  assert(!op.ExecuteGatherRead(bad).ok() && task_errors == 0);
  assert(channel_count == workers + 1);  // One TP pair, separate data/control Jettys.
  assert(core.Shutdown().ok() && live() == 0);
  assert(aclrtDestroyStream(stream) == 0);
  assert(aclrtFree(input) == 0);
  assert(aclrtFree(output) == 0);
}
}  // namespace
namespace kvaf::peer {
PeerLink::PeerLink(std::string address) : address_(std::move(address)) {}
Status PeerLink::LookupRegion(std::uint64_t base, PeerRegion* value) const {
  assert(base == region.base);
  *value = region;
  return Status::OK();
}
Status PeerLink::CreateChannel(std::uint64_t, const std::uint8_t*, std::uint32_t,
                               const std::uint8_t*, std::uint32_t, std::uint64_t tp,
                               std::uint32_t psn, std::array<std::uint8_t, kQpKeyBytes>* key,
                               std::uint32_t* key_bytes, std::uint64_t* peer_tp,
                               std::uint32_t* peer_psn, bool share) {
  assert(share && tp == 42 && psn == 1);
  urma_jetty_id_t id{};
  id.id = 1000 + channel_count++;
  std::memcpy(key->data(), &id, sizeof(id));
  const urma_transport_mode_t mode = URMA_TM_RM;
  std::memcpy(key->data() + sizeof(id), &mode, sizeof(mode));
  *key_bytes = 28;
  *peer_tp = 43;
  *peer_psn = 5;
  return Status::OK();
}
Status PeerLink::EnableDirectGather(std::uint64_t base) const {
  assert(base == region.base && !cpu.joinable());
  cpu = std::thread(CpuLoop);
  return Status::OK();
}
}  // namespace kvaf::peer
extern "C" {
aclError aclrtMalloc(void** value, std::size_t bytes, aclrtMemMallocPolicy) {
  ++allocations;
  return posix_memalign(value, 4096, (bytes + 4095) & ~std::size_t{4095}) == 0 ? 0 : 1;
}
aclError aclrtMallocHost(void** value, std::size_t bytes) {
  return aclrtMalloc(value, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
}
aclError aclrtFree(void* value) {
  ++frees;
  std::free(value);
  return 0;
}
aclError aclrtFreeHost(void* value) { return aclrtFree(value); }
aclError aclrtMemset(void* value, std::size_t, int byte, std::size_t bytes) {
  std::memset(value, byte, bytes);
  return 0;
}
aclError aclrtMemcpy(void* dst, std::size_t capacity, const void* source, std::size_t bytes,
                     aclrtMemcpyKind) {
  assert(bytes <= capacity);
  std::memcpy(dst, source, bytes);
  return 0;
}
aclError aclrtMemcpyAsync(void* dst, std::size_t capacity, const void* source, std::size_t bytes,
                          aclrtMemcpyKind kind, aclrtStream handle) {
  if (fail_copy == kind) {
    fail_copy = -1;
    return 1;
  }
  assert(bytes <= capacity);
  static_cast<Stream*>(handle)->tasks.emplace_back([=] {
    std::memcpy(dst, source, bytes);
    return 0;
  });
  return 0;
}
aclError aclrtCreateStream(aclrtStream* stream) {
  *stream = new Stream;
  return 0;
}
aclError aclrtDestroyStream(aclrtStream handle) {
  assert(static_cast<Stream*>(handle)->tasks.empty());
  delete static_cast<Stream*>(handle);
  return 0;
}
aclError aclrtSynchronizeStream(aclrtStream handle) {
  if (fail_sync) return 1;
  auto* stream = static_cast<Stream*>(handle);
  auto tasks = std::move(stream->tasks);
  stream->tasks.clear();
  int result = 0;
  for (auto& task : tasks) {
    const int current = task();
    if (current) ++task_errors;
    result |= current;
  }
  return result;
}
aclError aclrtBinaryLoadFromFile(const char*, const aclrtBinaryLoadOptions*,
                                 aclrtBinHandle* handle) {
  *handle = reinterpret_cast<void*>(1);
  return 0;
}
aclError aclrtBinaryUnLoad(aclrtBinHandle) { return 0; }
aclError aclrtBinaryGetFunction(aclrtBinHandle, const char* name, aclrtFuncHandle* function) {
  const auto id = std::strcmp(name, "KvafAicpuGatherRead") == 0      ? 1
                  : std::strcmp(name, "KvafAicpuGatherScatter") == 0 ? 2
                  : std::strcmp(name, "KvafAicpuUmdkControl") == 0   ? 3
                                                                     : 4;
  *function = reinterpret_cast<void*>(static_cast<std::uintptr_t>(id));
  return 0;
}
aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle function, unsigned blocks,
                                       aclrtStream stream, const aclrtLaunchKernelCfg*, void* args,
                                       std::size_t bytes, void*, std::size_t) {
  std::vector<std::uint64_t> storage((bytes + 7) / 8);
  std::memcpy(storage.data(), args, bytes);
  static_cast<Stream*>(stream)->tasks.emplace_back([=]() mutable {
    const auto id = reinterpret_cast<std::uintptr_t>(function);
    if (id == 1)
      return static_cast<int>(
          KvafAicpuGatherRead(reinterpret_cast<GatherReadArgs*>(storage.data())));
    if (id == 3)
      return static_cast<int>(
          KvafAicpuUmdkControl(reinterpret_cast<UmdkControlArgs*>(storage.data())));
    if (id == 4)
      return static_cast<int>(
          KvafAicpuUmdkBatchGet(reinterpret_cast<UmdkBatchGetArgs*>(storage.data())));
    std::atomic<unsigned> errors{0};
    std::vector<std::thread> workers;
    for (unsigned index = 0; index < blocks; ++index)
      workers.emplace_back([&] {
        errors += KvafAicpuGatherScatter(reinterpret_cast<GatherReadArgs*>(storage.data()));
      });
    for (auto& worker : workers) worker.join();
    return static_cast<int>(errors.load());
  });
  return 0;
}
}
int main(int argc, char** argv) {
  assert(GatherReadPackedOutputBytes(1024, 256, 4, 0) == 1408);
  assert(GatherReadPackedOutputBytes(UINT64_MAX, 256, 4, 0) == 0);
  assert(GatherReadPackedOutputBytes(1024, UINT64_MAX - 63, 4, 0) == 0);
  assert(argc == 2);
  auto* library = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
  assert(library);
  const auto live = reinterpret_cast<int (*)()>(dlsym(library, "kvaf_umdk_mock_live_objects"));
  assert(live);
  const auto observe = reinterpret_cast<void (*)(void (*)(std::uint64_t, std::uint32_t, bool))>(
      dlsym(library, "kvaf_umdk_mock_read_observer"));
  assert(observe);
  observe(ObserveRead);
  for (unsigned workers : {1U, 2U, 4U})
    for (bool pipeline : {false, true}) Run(workers, pipeline, live);
  observe(nullptr);
  dlclose(library);
  std::cout
      << "GatherRead pipeline, fresh layouts, flat BatchPut, queue ownership and recovery: PASS\n";
}
