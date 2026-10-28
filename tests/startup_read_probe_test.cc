#include "../runtime/src/startup_read_probe.h"

#include <cstdlib>
#include <iostream>

#define CHECK(x) do { if (!(x)) { std::cerr << "line " << __LINE__ << ": " #x "\n"; std::exit(1); } } while (false)
using namespace kvaf;

class ProbeOperator : public AccessOperator {
 public:
  uint32_t cores = 0U, jettys = 0U, writes = 0U;
  bool corrupt = false, idle_jetty = false, profiling = false;
  std::vector<BatchGetKernelProfile> profiles;
  Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
  Status Shutdown() override {
    CHECK(fake_acl::allocations > 0U);  // Cached buffers still alive at teardown.
    return Status::OK();
  }
  Status ExecuteBatchPut(const BatchPutExecRequest& exec) override {
    ++writes;
    for (size_t i = 0U; i < exec.request->keys.size(); ++i) {
      std::memcpy(reinterpret_cast<void*>(exec.put_desc->entries[i].address),
                  static_cast<const uint8_t*>(exec.request->input.data) + exec.request->input_offsets[i],
                  exec.request->lengths[i]);
    }
    return Status::OK();
  }
  Status ExecuteBatchGet(const BatchGetExecRequest& exec) override {
    const auto& get = *exec.request;
    CHECK(exec.execution.synchronize && profiling);
    const auto* addresses = static_cast<const uint64_t*>(get.device_access_table.addresses.data);
    const auto* indices = static_cast<const int32_t*>(get.indices.data);
    const auto* destinations = static_cast<const int32_t*>(get.output_indices.data);
    const auto rows = *static_cast<const int32_t*>(get.miss_count.data);
    for (int32_t row = 0; row < rows; ++row) {
      std::memcpy(static_cast<uint8_t*>(get.output.data) + destinations[row] * get.output_block_bytes,
                  reinterpret_cast<const void*>(addresses[indices[row]]), get.output_block_bytes);
    }
    if (corrupt) static_cast<uint8_t*>(get.output.data)[get.output.bytes - 1U] ^= 1U;
    profiles.assign(cores, {});
    for (auto& profile : profiles) {
      profile.active_requests = rows / cores;
      for (uint32_t j = 0U; j < jettys; ++j) profile.jetty_completion_polls[j] = 1U;
    }
    if (idle_jetty) profiles.back().jetty_completion_polls[jettys - 1U] = 0U;
    return Status::OK();
  }
  Status SetBatchGetKernelProfiling(bool enabled) { profiling = enabled; return Status::OK(); }
  Status CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* result) {
    *result = profiles; return Status::OK();
  }
};

int main() {
  for (uint32_t cores : {2U, 4U, 8U}) {
    for (uint32_t jettys : {1U, 2U, 4U}) {
      for (int fault : {0, 1, 2}) {
        auto meta = CreateInMemoryMetaManager();
        auto allocator = CreateHostDdrMemoryAllocator();
        ProbeOperator op;
        op.cores = cores; op.jettys = jettys;
        op.corrupt = fault == 1; op.idle_jetty = fault == 2;
        RuntimeCore core(*meta, op, *allocator);
        CHECK(core.Initialize({4096U, 128U, DeviceKind::kCpu}).ok());
        {
          detail::StartupReadProbe probe;
          const Status status = probe.Run(core, op, cores, jettys, 16U, 8U);
          if (fault == 0) {
            CHECK(status.ok() && op.writes == 2U && !op.profiling);
            // Probe keys were released, ready for caller data at the same slot.
            PutContext context; PutAccessDesc desc;
            CHECK(meta->PreparePut({0U}, {128U}, &context, &desc).ok());
            CHECK(meta->AbortPut(context).ok());
          } else {
            CHECK(!status.ok());
            CHECK(status.message().find(fault == 1 ? "data mismatch" : "inactive") != std::string::npos);
          }
          CHECK(core.Shutdown().ok());
        }
        CHECK(fake_acl::allocations == 0U);
      }
    }
  }
  std::cout << "Startup READ validation: 2/4/8 cores, two writes, faults and buffer lifetime passed\n";
}
