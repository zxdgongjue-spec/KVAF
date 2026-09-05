#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "batch_get_abi.h"
#include "batch_get_compact_prefix.h"
#include "kvaf/access_operator/access_operator.h"
#include "kvaf/meta_manager/meta_manager.h"
#include "kvaf/runtime/runtime.h"

namespace {

void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class ObservingOperator final : public kvaf::AccessOperator {
 public:
  kvaf::Status Initialize(const kvaf::RemoteRegionDesc&) override {
    return kvaf::Status::OK();
  }
  kvaf::Status Shutdown() override { return kvaf::Status::OK(); }
  kvaf::Status ExecuteBatchPut(const kvaf::BatchPutExecRequest&) override {
    return kvaf::Status::OK();
  }
  kvaf::Status ExecuteBatchGet(const kvaf::BatchGetExecRequest& execution) override {
    ++calls;
    Check(execution.request != nullptr, "missing request");
    last_request = *execution.request;
    last_execution = execution.execution;
    return kvaf::Status::OK();
  }
  unsigned calls{0};
  kvaf::BatchGetRequest last_request;
  kvaf::BatchGetExecutionOptions last_execution;
};

kvaf::BatchGetRequest DeviceRequest(std::size_t capacity) {
  // Intentionally unmapped pointers prove Runtime forwards the opaque NPU
  // views without reading int32[1] or inspecting descriptor contents on Host.
  // The observing operator performs no transport or payload work.
  using kvaf::ConstBufferView;
  using kvaf::DeviceKind;
  return kvaf::BatchGetRequest{
      {},
      ConstBufferView{reinterpret_cast<void*>(0x1000), capacity * 4, DeviceKind::kNpu},
      ConstBufferView{reinterpret_cast<void*>(0x2000), capacity * 4, DeviceKind::kNpu},
      ConstBufferView{reinterpret_cast<void*>(0x3000), 4, DeviceKind::kNpu},
      ConstBufferView{reinterpret_cast<void*>(0x4000), capacity * 4, DeviceKind::kNpu},
      {reinterpret_cast<void*>(0x5000), capacity * 656, DeviceKind::kNpu},
      656,
      {ConstBufferView{reinterpret_cast<void*>(0x6000), 32 * 8, DeviceKind::kNpu},
       ConstBufferView{reinterpret_cast<void*>(0x7000), 32, DeviceKind::kNpu}},
      kvaf::BatchGetSelectionMode::kCompactCount};
}

void TestRuntimeOpaqueForwardingAndValidation() {
  auto metadata = kvaf::CreateInMemoryMetaManager();
  auto allocator = kvaf::CreateHostDdrMemoryAllocator();
  ObservingOperator access;
  kvaf::Runtime runtime(*metadata, access, *allocator);
  Check(runtime.Initialize({4096, 256, kvaf::DeviceKind::kCpu}).ok(),
        "runtime initialization failed");
  constexpr std::size_t capacity = 16 * 8192;
  auto request = DeviceRequest(capacity);
  void* const stream = reinterpret_cast<void*>(0x9000);
  Check(runtime.BatchGet(request, {stream, false}).ok(),
        "native compact count rejected");
  Check(access.calls == 1, "compact request not forwarded exactly once");
  Check(access.last_request.mask.data == request.mask.data &&
            access.last_request.mask.bytes == sizeof(std::int32_t),
        "compact count was copied or expanded");
  Check(access.last_request.indices.data == request.indices.data &&
            access.last_request.output.data == request.output.data,
        "descriptor/output aliases changed");
  Check(access.last_request.selection_mode == kvaf::BatchGetSelectionMode::kCompactCount,
        "selection mode lost");
  Check(access.last_execution.stream == stream &&
            !access.last_execution.synchronize,
        "caller stream or asynchronous ownership changed");

  auto invalid = request;
  invalid.mask.bytes = 8;
  Check(!runtime.BatchGet(invalid).ok(), "non-scalar count accepted");
  invalid = request;
  invalid.mask.device = kvaf::DeviceKind::kCpu;
  Check(!runtime.BatchGet(invalid).ok(), "host count accepted");
  invalid = request;
  invalid.mask.data = nullptr;
  Check(!runtime.BatchGet(invalid).ok(), "null count accepted");
  invalid = request;
  invalid.selection_mode = static_cast<kvaf::BatchGetSelectionMode>(99);
  Check(!runtime.BatchGet(invalid).ok(), "unknown mode accepted");
  invalid = request;
  invalid.selection_mode = kvaf::BatchGetSelectionMode::kDenseMask;
  Check(!runtime.BatchGet(invalid).ok(), "short dense mask accepted as count");
  Check(access.calls == 1, "invalid metadata reached the data plane");

  // Existing dense mode, including the ambiguous N=4 byte size, is explicitly
  // distinguished by the mode rather than inferring semantics from bytes.
  auto dense = request;
  dense.selection_mode = kvaf::BatchGetSelectionMode::kDenseMask;
  dense.mask.bytes = capacity;
  Check(runtime.BatchGet(dense, {stream, false}).ok(), "legacy dense mask rejected");
  auto four = DeviceRequest(4);
  four.selection_mode = kvaf::BatchGetSelectionMode::kDenseMask;
  Check(runtime.BatchGet(four).ok(), "four-byte dense mask rejected");
  four.selection_mode = kvaf::BatchGetSelectionMode::kCompactCount;
  Check(runtime.BatchGet(four).ok(), "four-byte compact count rejected");
  Check(access.calls == 4, "dense/compact dispatch count differs");
  Check(runtime.Shutdown().ok(), "runtime shutdown failed");
}

void TestPrefixCoverageAndReplay() {
  constexpr uint32_t capacity = 16 * 8192;
  const uint32_t workers[] = {1, 2, 16, 48};
  const int32_t counts[] = {0, 1, 31, 32, 127, 6000,
                            static_cast<int32_t>(capacity), 0, 17};
  for (uint32_t cores : workers) {
    for (int32_t count : counts) {
      uint32_t previous = 0;
      uint32_t smallest = capacity;
      uint32_t largest = 0;
      for (uint32_t core = 0; core < cores; ++core) {
        uint32_t begin = 999;
        uint32_t end = 999;
        Check(KvafBatchGetCompactPrefixRange(count, capacity, core, cores, &begin, &end),
              "valid count/range rejected");
        Check(begin == previous && end >= begin && end <= static_cast<uint32_t>(count),
              "prefix range has hole, overlap or stale tail");
        smallest = std::min(smallest, end - begin);
        largest = std::max(largest, end - begin);
        previous = end;
      }
      Check(previous == static_cast<uint32_t>(count), "prefix not fully covered");
      Check(largest - smallest <= 1, "compact prefix is not load-balanced");
      if (count == 6000) {
        Check(largest <= 16384, "6000-count workload exceeds per-AIV UB bound");
      }
    }
  }
  for (int32_t count : {-1, static_cast<int32_t>(capacity + 1)}) {
    uint32_t begin = 99, end = 99;
    Check(!KvafBatchGetCompactPrefixRange(count, capacity, 0, 16, &begin, &end),
          "invalid dynamic count accepted");
    Check(begin == 0 && end == 0, "invalid count retained a postable prefix");
  }
  uint32_t begin = 99, end = 99;
  Check(!KvafBatchGetCompactPrefixRange(1, capacity, 0, 0, &begin, &end),
        "zero workers accepted");
  Check(!KvafBatchGetCompactPrefixRange(1, capacity, 16, 16, &begin, &end),
        "out-of-range worker accepted");
  Check(!KvafBatchGetCompactPrefixRange(1, capacity, 0, 1, nullptr, &end),
        "null output accepted");
  Check(KvafBatchGetCompactPrefixRange(std::numeric_limits<int32_t>::max(),
              std::numeric_limits<uint32_t>::max(), 47, 48, &begin, &end),
        "wide multiplication failed");
  Check(end == static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
        "wide multiplication overflowed");
}

void TestPrefixRetainsNativeScatterIdentities() {
  // Executable descriptor selection check with poisoned unused tail. The
  // same prefix range routine is used by the real AIV loader. This tests
  // descriptor identities, not Direct-HCCP payload transfer or CQ timing.
  constexpr uint32_t capacity = 131072;
  constexpr int32_t count = 6000;
  std::vector<int32_t> source(capacity, -99999);
  std::vector<int32_t> output_indices(capacity, -99999);
  for (int32_t index = 0; index < count; ++index) {
    source[index] = (index * 7) % 8192;
    output_indices[index] = count - index - 1;
  }
  std::vector<int32_t> observed(count, -1);
  for (uint32_t core = 0; core < 16; ++core) {
    uint32_t begin = 0, end = 0;
    Check(KvafBatchGetCompactPrefixRange(count, capacity, core, 16, &begin, &end),
          "scatter prefix rejected");
    for (uint32_t request = begin; request < end; ++request) {
      observed.at(static_cast<std::size_t>(output_indices[request])) = source[request];
    }
  }
  for (int32_t index = 0; index < count; ++index) {
    Check(observed[count - index - 1] == source[index],
          "native compact request/scatter identity changed");
  }
}

}  // namespace

int main() {
  try {
    static_assert(KVAF_AIV_BATCH_GET_COMPACT_COUNT_ABI == 1,
                  "compact-count capability changed");
    static_assert(sizeof(KvafExecuteBatchGetKernelArgs) == 112,
                  "host/device launch ABI drifted");
    TestRuntimeOpaqueForwardingAndValidation();
    TestPrefixCoverageAndReplay();
    TestPrefixRetainsNativeScatterIdentities();
    std::cout << "PASS compact-count runtime forwarding, prefix, replay, scatter\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
