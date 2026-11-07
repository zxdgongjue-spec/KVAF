#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#include "../host/kernel_loader.h"
#include "meta_manager.h"
#include "ubmem/access_operator.h"
#include "ubmem/transfer_abi.h"

namespace model {
std::map<void*, bool> allocations;
int32_t device = 0;
bool fail_sync = false, fail_free = false, fail_d2h = false, fail_host_alloc = false;
bool bad_status = false;
uint32_t sync_calls = 0, async_d2h = 0, blocking_d2h = 0;
struct PendingCopy { void* destination; const void* source; size_t bytes; aclrtStream stream; };
std::vector<PendingCopy> pending;
uint32_t frees = 0, copies = 0, memsets = 0, launches = 0;
void* stream = reinterpret_cast<void*>(1);
uint64_t Address(const void* pointer) { return reinterpret_cast<uintptr_t>(pointer); }
}  // namespace model

aclError aclrtGetDevice(int32_t* device) { *device = model::device; return 0; }
aclError aclrtMalloc(void** address, size_t bytes, aclrtMemMallocPolicy) {
  *address = std::malloc(bytes); assert(*address); model::allocations[*address] = false; return 0;
}
aclError aclrtMallocHost(void** address, size_t bytes) {
  if (model::fail_host_alloc) { model::fail_host_alloc = false; return 20; }
  *address = std::malloc(bytes); assert(*address); model::allocations[*address] = true; return 0;
}
aclError Release(void* address, bool host) {
  if (model::fail_free) { model::fail_free = false; return 19; }
  auto found = model::allocations.find(address);
  assert(found != model::allocations.end() && found->second == host);
  model::allocations.erase(found); std::free(address); ++model::frees; return 0;
}
aclError aclrtFree(void* address) { return Release(address, false); }
aclError aclrtFreeHost(void* address) { return Release(address, true); }
aclError aclrtMemcpy(void* destination, size_t capacity, const void* source, size_t bytes, aclrtMemcpyKind kind) {
  if (kind == ACL_MEMCPY_DEVICE_TO_HOST) ++model::blocking_d2h;
  assert(bytes <= capacity); ++model::copies; std::memcpy(destination, source, bytes); return 0;
}
aclError aclrtMemcpyAsync(void* destination, size_t capacity, const void* source, size_t bytes,
                          aclrtMemcpyKind kind, aclrtStream stream) {
  if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
    if (model::fail_d2h) { model::fail_d2h = false; return 21; }
    assert(bytes <= capacity && model::allocations.at(destination));
    assert(stream == model::stream);
    ++model::async_d2h;
    // Do not make status visible before the production stream fence.
    std::memset(destination, 0, capacity);
    model::pending.push_back({destination, source, bytes, stream});
    return 0;
  }
  return aclrtMemcpy(destination, capacity, source, bytes, kind);
}
aclError aclrtMemsetAsync(void* destination, size_t capacity, int value, size_t bytes, aclrtStream) {
  assert(bytes <= capacity); ++model::memsets; std::memset(destination, value, bytes); return 0;
}
aclError aclrtSynchronizeStream(aclrtStream stream) {
  ++model::sync_calls;
  if (model::fail_sync) return 17;
  for (const auto& copy : model::pending) {
    assert(copy.stream == stream);
    std::memcpy(copy.destination, copy.source, copy.bytes);
  }
  model::pending.clear();
  return 0;
}
aclError aclrtSynchronizeDevice() { return aclrtSynchronizeStream(model::stream); }

namespace kvaf::ubmem {
// The actual kernels are tested separately by copy_model_test.cc. This
// launch double exercises the production host's ownership/error transitions.
Status Kernel::Load(const std::string&, const char* name) {
  function_ = reinterpret_cast<void*>(std::strstr(name, "Put") ? 2 : 1);
  binary_ = function_;
  return Status::OK();
}
Status Kernel::Unload() { binary_ = nullptr; function_ = nullptr; return Status::OK(); }
aclError Kernel::Launch(const void* arguments, size_t, uint32_t workers, aclrtStream) const {
  using namespace kvaf_ubmem;
  ++model::launches;
  if (function_ == reinterpret_cast<void*>(2)) {
    const auto& args = *static_cast<const PutArgs*>(arguments);
    const auto* entries = reinterpret_cast<const PutEntry*>(args.entries);
    for (uint32_t i = 0; i < args.count; ++i) {
      std::memcpy(reinterpret_cast<void*>(entries[i].destination),
          reinterpret_cast<void*>(args.input + entries[i].input_offset), entries[i].length);
    }
    for (uint32_t i = 0; i < workers; ++i)
      reinterpret_cast<CoreStatus*>(args.status)[i] = {kStatusMagic, kSuccess, UINT64_MAX, 0};
    if (model::bad_status) reinterpret_cast<CoreStatus*>(args.status)[workers - 1].result = kInvalidSource;
    for (uint32_t i = 0; i < args.count; ++i)
      ++reinterpret_cast<CoreStatus*>(args.status)[
          ((entries[i].destination - args.region_base) / args.slot_bytes) % workers].copied_requests;
  } else {
    const auto& args = *static_cast<const GetArgs*>(arguments);
    if (args.status) {
      for (uint32_t i = 0; i < workers; ++i) {
        reinterpret_cast<CoreStatus*>(args.status)[i] = {kStatusMagic,
            *reinterpret_cast<const int32_t*>(args.miss_count) < 0 ? kInvalidCount : kSuccess, UINT64_MAX, 0};
      }
    }
  }
  return 0;
}
}  // namespace kvaf::ubmem

int main() {
  using namespace kvaf;
  using model::Address;
  alignas(32) uint8_t arena[4096]{}, input[2048]{}, output[2624]{};
  for (size_t i = 0; i < sizeof(input); ++i) input[i] = i % 251;
  AccessOperatorConfig config;
  config.batch_get_kernel_object_path = "get.o";
  config.batch_put_kernel_object_path = "put.o";
  config.npu_core_count = 2;
  const RemoteRegionDesc region{Address(arena), sizeof(arena), 2048, DeviceKind::kCpu};
  BatchPutRequest request{{0}, {input, sizeof(input), DeviceKind::kNpu}, {37}, {656}};
  PutAccessDesc desc{{{0, Address(arena), 656}}};
  BatchPutExecRequest put{&request, &desc, {model::stream}};
  {
    auto op = CreateUbmemAccessOperator(config);
    assert(op->Initialize(region).ok());
    const auto before_sync = model::sync_calls;
    assert(op->ExecuteBatchPut(put).ok());
    assert(model::sync_calls == before_sync + 1 && model::async_d2h == 1);
    assert(model::blocking_d2h == 0 && model::pending.empty());
    const auto allocations = model::allocations;
    assert(op->ExecuteBatchPut(put).ok());
    assert(model::allocations == allocations);  // status/metadata buffers reused
    model::bad_status = true;
    assert(op->ExecuteBatchPut(put).code() == StatusCode::kInternal);
    model::bad_status = false;
    model::fail_d2h = true;
    assert(op->ExecuteBatchPut(put).code() == StatusCode::kInternal);
    assert(op->ExecuteBatchPut(put).ok());  // completed copy error is retryable
    assert(std::memcmp(arena, input + 37, 656) == 0);
    assert(model::allocations.size() == 4);
    int32_t index = 0, offset = 0, count = -1, out_index = 0;
    uint64_t table = Address(arena);
    uint8_t valid = 1;
    BatchGetRequest get;
    get.indices = {&index, 4, DeviceKind::kNpu};
    get.offsets = {&offset, 4, DeviceKind::kNpu};
    get.miss_count = {&count, 4, DeviceKind::kNpu};
    get.output_indices = {&out_index, 4, DeviceKind::kNpu};
    get.output = {output, 656, DeviceKind::kNpu};
    get.output_block_bytes = 656;
    get.tokens_per_block = 1;
    get.device_access_table = {{&table, 8, DeviceKind::kNpu}, {&valid, 1, DeviceKind::kNpu}};
    assert(op->ExecuteBatchGet({{}, &get, {model::stream, true}}).code() == StatusCode::kInternal);
    count = 0;
    auto before_copies = model::copies, before_memsets = model::memsets;
    assert(op->ExecuteBatchGet({{}, &get, {model::stream, false}}).ok());
    assert(model::copies == before_copies && model::memsets == before_memsets);
    model::device = 1;
    assert(op->ExecuteBatchPut(put).code() == StatusCode::kInvalidState);
    model::device = 0;
    request.keys.push_back(1); request.input_offsets.push_back(800); request.lengths.push_back(1);
    desc.entries.push_back({1, Address(arena + 2048), 1});
    model::fail_free = true;
    assert(op->ExecuteBatchPut(put).code() == StatusCode::kInternal);
    assert(model::allocations.size() == 4);  // failed release was retained
    assert(op->Shutdown().ok());
    assert(op->Shutdown().ok());
    assert(model::allocations.empty());
  }
  {
    auto op = CreateUbmemAccessOperator(config);
    assert(op->Initialize(region).ok());
    model::fail_host_alloc = true;
    assert(op->ExecuteBatchPut(put).code() == StatusCode::kInternal);
    assert(model::allocations.size() == 1 && model::pending.empty());
    assert(op->Shutdown().ok());
    assert(model::allocations.empty());
  }
  {
    auto op = CreateUbmemAccessOperator(config);
    assert(op->Initialize(region).ok());
    model::fail_sync = true;
    auto before_frees = model::frees;
    assert(op->ExecuteBatchPut(put).code() == StatusCode::kCompletionUnknown);
    assert(model::allocations.size() == 4);  // device/pinned status + device/pinned metadata
    assert(op->ExecuteBatchPut(put).code() == StatusCode::kCompletionUnknown);
    assert(op->Shutdown().code() == StatusCode::kCompletionUnknown);
    op.reset();
    assert(model::frees == before_frees && model::allocations.size() == 4);
    // Intentional leaked owner models process-teardown quarantine. Fake NPU
    // allocations can be reclaimed here; no code retains a runnable operator.
    for (auto allocation : model::allocations) std::free(allocation.first);
    model::allocations.clear();
    model::pending.clear();
  }
  std::cout << "UBmem host contract PASS (one Put fence, pinned readback, async Get, copy failure, completion quarantine)\n";
}
