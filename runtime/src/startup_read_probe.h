#pragma once

#include <acl/acl.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "batch_get_schedule.h"
#include "runtime.h"

namespace kvaf::detail {

// Buffer registrations are cached by the operator. Retain these allocations
// until AFTER both transports have been destroyed, including failed probes.
class StartupReadProbe {
 public:
  ~StartupReadProbe() {
    if (stream_ != nullptr) (void)aclrtDestroyStream(stream_);
    for (void* buffer : buffers_) (void)aclrtFree(buffer);
  }

  template <class ProfiledOperator>
  Status Run(RuntimeCore& core, ProfiledOperator& op, uint32_t cores,
             uint32_t jettys, uint32_t token_bytes, uint32_t tokens_per_block) {
    const uint64_t block_bytes = uint64_t{token_bytes} * tokens_per_block;
    // Even with inflight control disabled, this reaches the last Jetty on
    // every core. Smaller groups rotate Jettys more frequently, never less.
    const uint32_t rows_per_core = (jettys - 1U) * kvaf_batch_get::kWqesPerPost + 1U;
    const uint64_t rows = uint64_t{cores} * rows_per_core;
    const uint64_t payload = rows * token_bytes;
    if (cores == 0U || jettys == 0U || block_bytes == 0U ||
        block_bytes > core.remote_region().slot_size ||
        block_bytes > std::numeric_limits<size_t>::max() ||
        payload > std::numeric_limits<size_t>::max() ||
        rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return Status::InvalidArgument("invalid per-core endpoint READ probe shape");
    }
    if (aclrtCreateStream(&stream_) != ACL_SUCCESS) return AllocationError();
    std::vector<uint8_t> source(static_cast<size_t>(block_bytes));
    BatchPutRequest put;
    put.keys = {0U};
    put.input_offsets = {0U};
    put.lengths = {block_bytes};
    if (!Upload(source.data(), source.size(), &put.input)) return AllocationError();

    BatchGetRequest get;
    get.output_block_bytes = token_bytes;
    get.tokens_per_block = tokens_per_block;
    std::vector<int32_t> indices(static_cast<size_t>(rows), 0);
    std::vector<int32_t> destinations(static_cast<size_t>(rows));
    for (size_t i = 0U; i < destinations.size(); ++i) destinations[i] = static_cast<int32_t>(i);
    const int32_t misses = static_cast<int32_t>(rows);
    if (!Upload(indices.data(), indices.size() * sizeof(int32_t), &get.indices) ||
        !Upload(indices.data(), indices.size() * sizeof(int32_t), &get.offsets) ||
        !Upload(destinations.data(), destinations.size() * sizeof(int32_t), &get.output_indices) ||
        !Upload(&misses, sizeof(misses), &get.miss_count)) return AllocationError();
    void* output = Allocate(static_cast<size_t>(payload));
    if (output == nullptr) return AllocationError();
    get.output = MutableBufferView{output, static_cast<size_t>(payload), DeviceKind::kNpu};
    Status status = op.SetBatchGetKernelProfiling(true);
    if (!status.ok()) return status;

    // Write different contents twice through the default endpoint and read
    // through ALL configured paths; this also detects separate DDR buffers.
    for (uint32_t round = 0U; round < 2U; ++round) {
      for (size_t i = 0U; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>((i * 17U + 31U + round * 73U) % 251U);
      }
      if (aclrtMemcpy(const_cast<void*>(put.input.data), put.input.bytes,
                      source.data(), source.size(), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return AllocationError();
      }
      status = core.BatchPut(put, BatchPutExecutionOptions{stream_});
      if (!status.ok()) return status;
      status = core.PrepareAccessTable(AccessTablePrepareRequest{put.keys}, &get.access_table);
      if (!status.ok()) return status;
      if (round == 0U) {
        AccessTableView table;
        status = core.GetAccessTableView(get.access_table, &table);
        if (!status.ok()) return status;
        if (!Upload(table.addresses, table.size * sizeof(Address), &get.device_access_table.addresses) ||
            !Upload(table.valid, table.size, &get.device_access_table.valid)) return AllocationError();
      }
      if (aclrtMemset(output, static_cast<size_t>(payload), 0xff,
                      static_cast<size_t>(payload)) != ACL_SUCCESS) return AllocationError();
      status = core.BatchGet(get, BatchGetExecutionOptions{stream_, true});
      if (!status.ok()) return status;
      std::vector<uint8_t> actual(static_cast<size_t>(payload));
      if (aclrtMemcpy(actual.data(), actual.size(), output, actual.size(),
                      ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) return AllocationError();
      for (size_t i = 0U; i < actual.size(); ++i) {
        if (actual[i] != source[i % token_bytes]) {
          return Status::Internal("per-core endpoint READ probe data mismatch at byte " +
                                   std::to_string(i));
        }
      }
      std::vector<BatchGetKernelProfile> profiles;
      status = op.CopyBatchGetKernelProfiles(&profiles);
      if (!status.ok()) return status;
      if (profiles.size() != cores) return Status::Internal("endpoint READ probe missing core profiles");
      for (uint32_t c = 0U; c < cores; ++c) {
        if (profiles[c].active_requests != rows_per_core) {
          return Status::Internal("endpoint READ probe left a core inactive");
        }
        for (uint32_t j = 0U; j < jettys; ++j) {
          if (profiles[c].jetty_completion_polls[j] == 0U) {
            return Status::Internal("endpoint READ probe left a Jetty inactive");
          }
        }
      }
      status = core.ReleaseAccessTable(get.access_table);
      if (!status.ok()) return status;
      get.access_table = {};
      status = core.BatchFree(BatchFreeRequest{put.keys});
      if (!status.ok()) return status;
      std::fprintf(stderr, "[KVAF_AIV] PEER_ENDPOINT_READ_VALIDATED round=%u cores=%u "
                           "jettys_per_core=%u requests=%u shared_ddr=1\n",
                   round, cores, jettys, misses);
    }
    return op.SetBatchGetKernelProfiling(false);
  }

 private:
  static Status AllocationError() { return Status::Internal("endpoint READ probe allocation/copy failed"); }
  void* Allocate(size_t bytes) {
    void* buffer = nullptr;
    if (aclrtMalloc(&buffer, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) return nullptr;
    if (buffer != nullptr) buffers_.push_back(buffer);
    return buffer;
  }
  bool Upload(const void* source, size_t bytes, ConstBufferView* view) {
    void* buffer = Allocate(bytes);
    if (buffer == nullptr || aclrtMemcpy(buffer, bytes, source, bytes,
                                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) return false;
    *view = ConstBufferView{buffer, bytes, DeviceKind::kNpu};
    return true;
  }
  aclrtStream stream_{nullptr};
  std::vector<void*> buffers_;
};

}  // namespace kvaf::detail
