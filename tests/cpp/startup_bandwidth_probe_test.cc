#include "startup_bandwidth_probe.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unordered_set>

namespace {
std::unordered_set<void*> allocations;
bool fail_allocation = false;
bool transport_alive = false;
void Check(bool value) {
  if (!value) {
    std::cerr << "startup bandwidth probe expectation failed\n";
    std::exit(1);
  }
}
}  // namespace

aclError aclrtMalloc(void** result, std::size_t bytes, int) {
  if (fail_allocation) return 1;
  *result = std::malloc(bytes);
  if (*result == nullptr) return 1;
  allocations.insert(*result);
  return ACL_SUCCESS;
}
aclError aclrtFree(void* buffer) {
  Check(!transport_alive);
  Check(allocations.erase(buffer) == 1);
  std::free(buffer);
  return ACL_SUCCESS;
}
aclError aclrtMemcpy(void* dest, std::size_t capacity, const void* src,
                     std::size_t bytes, int) {
  Check(bytes <= capacity);
  std::memcpy(dest, src, bytes);
  return ACL_SUCCESS;
}
aclError aclrtMemset(void* dest, std::size_t capacity, int value, std::size_t bytes) {
  Check(bytes <= capacity);
  std::memset(dest, value, bytes);
  return ACL_SUCCESS;
}
aclError aclrtCreateStream(aclrtStream* stream) { *stream = reinterpret_cast<void*>(1); return 0; }
aclError aclrtDestroyStream(aclrtStream) { Check(!transport_alive); return 0; }

namespace {
enum class Fault { None, CorruptData, Incomplete, ZeroTiming, GetFailure, Allocation };

class Operator final : public kvaf::AccessOperator {
 public:
  explicit Operator(Fault fault) : fault_(fault) {}
  kvaf::Status Initialize(const kvaf::RemoteRegionDesc&) override {
    transport_alive = true;
    return kvaf::Status::OK();
  }
  kvaf::Status Shutdown() override {
    transport_alive = false;
    return kvaf::Status::OK();
  }
  kvaf::Status ExecuteBatchPut(const kvaf::BatchPutExecRequest& exec) override {
    const auto& req = *exec.request;
    for (std::size_t i = 0; i < req.keys.size(); ++i) {
      std::memcpy(reinterpret_cast<void*>(exec.put_desc->entries[i].address),
                  static_cast<const char*>(req.input.data) + req.input_offsets[i], req.lengths[i]);
    }
    return kvaf::Status::OK();
  }
  kvaf::Status ExecuteBatchGet(const kvaf::BatchGetExecRequest& exec) override {
    Check(exec.execution.synchronize && exec.execution.stream != nullptr && profiling_);
    ++calls;
    if (fault_ == Fault::GetFailure) return kvaf::Status::Internal("get failure");
    const auto& r = *exec.request;
    const auto count = *static_cast<const std::int32_t*>(r.miss_count.data);
    const auto* addresses = static_cast<const kvaf::Address*>(r.device_access_table.addresses.data);
    const auto* indices = static_cast<const std::int32_t*>(r.indices.data);
    const auto* offsets = static_cast<const std::int32_t*>(r.offsets.data);
    const auto* output_indices = static_cast<const std::int32_t*>(r.output_indices.data);
    for (std::int32_t i = 0; i < count; ++i) {
      std::memcpy(static_cast<char*>(r.output.data) + output_indices[i] * r.output_block_bytes,
                  reinterpret_cast<const char*>(addresses[indices[i]]) + offsets[i],
                  r.output_block_bytes);
    }
    if (fault_ == Fault::CorruptData) static_cast<char*>(r.output.data)[0] ^= 1;
    return kvaf::Status::OK();
  }
  kvaf::Status SetBatchGetKernelProfiling(bool enabled) {
    profiling_ = enabled;
    return kvaf::Status::OK();
  }
  kvaf::Status CopyBatchGetKernelProfiles(std::vector<kvaf::BatchGetKernelProfile>* out) {
    kvaf::BatchGetKernelProfile p;
    p.active_requests = fault_ == Fault::Incomplete ? 1023U : 1024U;
    p.kernel_start_cycle = 100;
    const auto duration = calls <= kvaf::detail::kStartupWarmup ? 10000U : 36U - calls;
    p.kernel_finish_cycle = p.kernel_start_cycle + (fault_ == Fault::ZeroTiming ? 0 : duration);
    out->assign(2, p);
    return kvaf::Status::OK();
  }
  unsigned calls{0};
 private:
  Fault fault_;
  bool profiling_{false};
};
}  // namespace

int main() {
  for (const auto fault : {Fault::None, Fault::CorruptData, Fault::Incomplete,
                          Fault::ZeroTiming, Fault::GetFailure, Fault::Allocation}) {
    auto meta = kvaf::CreateInMemoryMetaManager();
    auto allocator = kvaf::CreateHostDdrMemoryAllocator();
    Operator op(fault);
    kvaf::RuntimeCore core(*meta, op, *allocator);
    Check(core.Initialize({kvaf::detail::kStartupDdrBytes,
                           kvaf::detail::kStartupTokenBytes * kvaf::detail::kStartupTokensPerBlock,
                           kvaf::DeviceKind::kCpu}).ok());
    {
      kvaf::detail::StartupBandwidthProbe probe;
      fail_allocation = fault == Fault::Allocation;
      double median = 0;
      const auto status = probe.Run(core, op, &median);
      Check(status.ok() == (fault == Fault::None));
      if (status.ok()) {
        Check(op.calls == 6 && median == 32.0); // One warmup excluded; median of 30..34.
      }
      Check(core.Shutdown().ok()); // Must precede probe buffer destruction.
    }
    Check(allocations.empty());
    fail_allocation = false;
  }
  std::cout << "startup bandwidth probe tests passed\n";
}
