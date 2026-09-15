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
  explicit StartupBandwidthProbe(StartupProbeConfig config) : config_(config) {}

  ~StartupBandwidthProbe() {
    if (stream_ != nullptr) (void)aclrtDestroyStream(stream_);
    for (void* buffer : buffers_) (void)aclrtFree(buffer);
  }

  template <class ProfiledOperator>
  Status Run(RuntimeCore& core, ProfiledOperator& op, double* median_cycles) {
    StartupProbeLayout layout{};
    Status status = BuildStartupProbeLayout(config_, &layout);
    if (!status.ok()) return status;
    const auto payload = static_cast<std::size_t>(config_.token_bytes) *
                         config_.miss_count;
    const auto block_bytes = static_cast<std::size_t>(layout.block_bytes);
    const auto source_bytes = static_cast<std::size_t>(layout.source_bytes);
    const auto blocks = (static_cast<std::size_t>(config_.miss_count) +
                         config_.tokens_per_block - 1U) /
                        config_.tokens_per_block;
    if (aclrtCreateStream(&stream_) != ACL_SUCCESS) {
      return Status::Internal("startup probe stream creation failed");
    }

    std::vector<std::uint8_t> source(source_bytes);
    for (std::size_t i = 0; i < source.size(); ++i) {
      source[i] = static_cast<std::uint8_t>(
          (i * 17U + i / config_.token_bytes) % 251U);
    }
    BatchPutRequest put;
    for (std::size_t block = 0; block < blocks; ++block) {
      put.keys.push_back(block);
      put.input_offsets.push_back(block * block_bytes);
      put.lengths.push_back(block_bytes);
    }
    if (!Upload(source.data(), source.size(), &put.input)) return AllocationError();
    status = core.BatchPut(put, BatchPutExecutionOptions{stream_});
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

    std::vector<std::int32_t> indices(config_.miss_count);
    std::vector<std::int32_t> offsets(config_.miss_count);
    std::vector<std::int32_t> destinations(config_.miss_count);
    std::vector<std::uint8_t> expected(payload);
    std::uint32_t stride = std::min<std::uint32_t>(5U, config_.miss_count);
    while (stride > 1U && std::gcd(stride, config_.miss_count) != 1U) --stride;
    for (std::uint32_t row = 0; row < config_.miss_count; ++row) {
      // Use a shape-dependent coprime stride so every token is read once and
      // every device sees the same request order for this shape.
      const auto token = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(row) * stride) % config_.miss_count);
      indices[row] = token / config_.tokens_per_block;
      offsets[row] = (token % config_.tokens_per_block) * config_.token_bytes;
      destinations[row] = row;
      std::copy_n(source.data() + token * config_.token_bytes,
                  config_.token_bytes,
                  expected.data() + row * config_.token_bytes);
    }
    const std::int32_t misses = static_cast<std::int32_t>(config_.miss_count);
    if (!Upload(indices.data(), indices.size() * sizeof(indices[0]), &get.indices) ||
        !Upload(offsets.data(), offsets.size() * sizeof(offsets[0]), &get.offsets) ||
        !Upload(destinations.data(), destinations.size() * sizeof(destinations[0]),
                &get.output_indices) ||
        !Upload(&misses, sizeof(misses), &get.miss_count)) return AllocationError();
    void* output = Allocate(payload);
    if (output == nullptr || aclrtMemset(output, payload, 0, payload) != ACL_SUCCESS) {
      return AllocationError();
    }
    get.output = MutableBufferView{output, payload, DeviceKind::kNpu};
    get.output_block_bytes = config_.token_bytes;
    get.tokens_per_block = config_.tokens_per_block;
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
      if (active_requests != config_.miss_count || critical_cycles == 0U) {
        return Status::Internal("startup probe incomplete BatchGet or invalid timing");
      }
      if (sample == 0U) {
        std::vector<std::uint8_t> actual(payload);
        if (aclrtMemcpy(actual.data(), actual.size(), output, payload,
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
  StartupProbeConfig config_;
};

}  // namespace kvaf::detail
