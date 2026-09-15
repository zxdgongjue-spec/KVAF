#pragma once

#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "runtime.h"
#include "startup_selection.h"

namespace kvaf::detail {

// These buffers may be cached as transport registrations. The owner must tear
// down RuntimeCore/AccessOperator before destroying the probe, including errors.
class StartupBandwidthProbe {
 public:
  ~StartupBandwidthProbe() {
    if (stream_ != nullptr) (void)aclrtDestroyStream(stream_);
    for (void* buffer : buffers_) (void)aclrtFree(buffer);
  }

  template <class ProfiledOperator>
  Status Run(RuntimeCore& core, ProfiledOperator& op, double* median_cycles) {
    constexpr std::size_t kPayload = kStartupTokenBytes * kStartupMissCount;
    constexpr std::size_t kBlockBytes = kStartupTokenBytes * kStartupTokensPerBlock;
    constexpr std::size_t kBlocks = kStartupMissCount / kStartupTokensPerBlock;
    if (aclrtCreateStream(&stream_) != ACL_SUCCESS) {
      return Status::Internal("startup probe stream creation failed");
    }

    std::vector<std::uint8_t> source(kPayload);
    for (std::size_t i = 0; i < source.size(); ++i) {
      source[i] = static_cast<std::uint8_t>((i * 17U + i / kStartupTokenBytes) % 251U);
    }
    BatchPutRequest put;
    for (std::size_t block = 0; block < kBlocks; ++block) {
      put.keys.push_back(block);
      put.input_offsets.push_back(block * kBlockBytes);
      put.lengths.push_back(kBlockBytes);
    }
    if (!Upload(source.data(), source.size(), &put.input)) return AllocationError();
    Status status = core.BatchPut(put, BatchPutExecutionOptions{stream_});
    if (!status.ok()) return status;

    BatchGetRequest get;
    status = core.PrepareAccessTable(AccessTablePrepareRequest{put.keys}, &get.access_table);
    if (!status.ok()) return status;
    AccessTableView table;
    status = core.GetAccessTableView(get.access_table, &table);
    if (!status.ok()) return status;
    if (!Upload(table.addresses, table.size * sizeof(Address),
                &get.device_access_table.addresses) ||
        !Upload(table.valid, table.size, &get.device_access_table.valid)) {
      return AllocationError();
    }

    std::vector<std::int32_t> indices(kStartupMissCount);
    std::vector<std::int32_t> offsets(kStartupMissCount);
    std::vector<std::int32_t> destinations(kStartupMissCount);
    std::vector<std::uint8_t> expected(kPayload);
    for (std::uint32_t row = 0; row < kStartupMissCount; ++row) {
      // Fixed permutation: each token is read once; every candidate sees the
      // same nonsequential requests and contiguous output placement.
      const auto token = (row * 5U) % kStartupMissCount;
      indices[row] = token / kStartupTokensPerBlock;
      offsets[row] = (token % kStartupTokensPerBlock) * kStartupTokenBytes;
      destinations[row] = row;
      std::copy_n(source.data() + token * kStartupTokenBytes, kStartupTokenBytes,
                  expected.data() + row * kStartupTokenBytes);
    }
    const std::int32_t misses = kStartupMissCount;
    if (!Upload(indices.data(), indices.size() * sizeof(indices[0]), &get.indices) ||
        !Upload(offsets.data(), offsets.size() * sizeof(offsets[0]), &get.offsets) ||
        !Upload(destinations.data(), destinations.size() * sizeof(destinations[0]),
                &get.output_indices) ||
        !Upload(&misses, sizeof(misses), &get.miss_count)) return AllocationError();
    void* output = Allocate(kPayload);
    if (output == nullptr || aclrtMemset(output, kPayload, 0, kPayload) != ACL_SUCCESS) {
      return AllocationError();
    }
    get.output = MutableBufferView{output, kPayload, DeviceKind::kNpu};
    get.output_block_bytes = kStartupTokenBytes;
    get.tokens_per_block = kStartupTokensPerBlock;
    status = op.SetBatchGetKernelProfiling(true);
    if (!status.ok()) return status;

    std::vector<double> samples;
    for (std::uint32_t sample = 0; sample < kStartupWarmup + kStartupSamples; ++sample) {
      status = core.BatchGet(get, BatchGetExecutionOptions{stream_, true});
      if (!status.ok()) return status;
      std::vector<BatchGetKernelProfile> profiles;
      status = op.CopyBatchGetKernelProfiles(&profiles);
      if (!status.ok()) return status;
      std::uint64_t critical_cycles = 0U;
      std::uint64_t active_requests = 0U;
      for (const auto& profile : profiles) {
        if (profile.kernel_finish_cycle < profile.kernel_start_cycle) {
          return Status::Internal("startup probe invalid cycle order");
        }
        critical_cycles = std::max(critical_cycles,
            profile.kernel_finish_cycle - profile.kernel_start_cycle);
        active_requests += profile.active_requests;
      }
      if (active_requests != kStartupMissCount || critical_cycles == 0U) {
        return Status::Internal("startup probe incomplete BatchGet or invalid timing");
      }
      if (sample == 0U) {
        std::vector<std::uint8_t> actual(kPayload);
        if (aclrtMemcpy(actual.data(), actual.size(), output, kPayload,
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS || actual != expected) {
          return Status::Internal("startup probe BatchGet data validation failed");
        }
      }
      if (sample >= kStartupWarmup) samples.push_back(static_cast<double>(critical_cycles));
    }
    std::sort(samples.begin(), samples.end());
    *median_cycles = samples[samples.size() / 2U];
    return Status::OK();
  }

 private:
  static Status AllocationError() {
    return Status::Internal("startup probe buffer allocation/upload failed");
  }

  void* Allocate(std::size_t bytes) {
    void* buffer = nullptr;
    if (aclrtMalloc(&buffer, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) return nullptr;
    if (buffer != nullptr) buffers_.push_back(buffer);
    return buffer;
  }

  bool Upload(const void* source, std::size_t bytes, ConstBufferView* view) {
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
