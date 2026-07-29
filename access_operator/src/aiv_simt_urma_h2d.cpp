#include "kvaf/access_operator/aiv_simt_urma_h2d.h"

#include <limits>
#include <string>

namespace kvaf::access {
namespace {

bool AddOverflows(std::uint64_t lhs, std::uint64_t rhs) {
  return rhs > std::numeric_limits<std::uint64_t>::max() - lhs;
}

void ResetCompletion(DeviceCompletion* completion, std::uint64_t generation) {
  *completion = DeviceCompletion{};
  completion->state = static_cast<std::uint32_t>(DeviceBatchState::kPlanning);
  completion->generation = generation;
}

Status Fail(const H2DPlanView& view, std::uint32_t index,
            DeviceBatchStatus device_status, StatusCode status_code,
            const std::string& message) {
  view.completion->state =
      static_cast<std::uint32_t>(DeviceBatchState::kFailed);
  view.completion->status = static_cast<std::uint32_t>(device_status);
  view.completion->planned_count = 0;
  view.completion->submitted_count = 0;
  view.completion->completed_count = 0;
  view.completion->first_failed_index = index;
  return Status::Error(status_code, message);
}

}  // namespace

Status BuildH2DTransferPlan(const H2DPlanView& view) {
  if (view.completion == nullptr) {
    return Status::Error(StatusCode::kInvalidArgument,
                         "completion must not be null");
  }
  ResetCompletion(view.completion, view.generation);

  if (view.table_generation != view.generation) {
    return Fail(view, 0, DeviceBatchStatus::kGenerationMismatch,
                StatusCode::kInvalidArgument,
                "request generation does not match the AccessTable");
  }
  if (view.entry_count > kAivUrmaMaxDescriptors ||
      view.entry_count > view.descriptor_capacity) {
    return Fail(view, 0, DeviceBatchStatus::kCountOutOfRange,
                StatusCode::kOutOfRange,
                "entry count exceeds the fixed AIV/SIMT descriptor capacity");
  }
  if (view.entry_count != 0 &&
      (view.remote_addresses == nullptr || view.lengths == nullptr ||
       view.indices == nullptr || view.output_offsets == nullptr ||
       view.descriptors == nullptr || view.output_base == 0)) {
    return Fail(view, 0, DeviceBatchStatus::kNullAddress,
                StatusCode::kInvalidArgument,
                "a required AccessTable, request, descriptor, or output "
                "address is null");
  }

  for (std::uint32_t request = 0; request < view.entry_count; ++request) {
    UrmaH2DDescriptor& descriptor = view.descriptors[request];
    descriptor = UrmaH2DDescriptor{};
    descriptor.logical_index = request;
    descriptor.generation = view.generation;
    descriptor.user_tag = request;

    if (view.mask != nullptr && view.mask[request] == 0) {
      descriptor.state =
          static_cast<std::uint32_t>(TransferState::kSkipped);
      continue;
    }

    const std::uint32_t table_index = view.indices[request];
    if (table_index >= view.table_capacity) {
      descriptor.state =
          static_cast<std::uint32_t>(TransferState::kFailed);
      descriptor.status =
          static_cast<std::uint32_t>(DeviceBatchStatus::kIndexOutOfRange);
      return Fail(view, request, DeviceBatchStatus::kIndexOutOfRange,
                  StatusCode::kOutOfRange,
                  "request index is outside the AccessTable");
    }
    if (view.valid != nullptr && view.valid[table_index] == 0) {
      descriptor.state =
          static_cast<std::uint32_t>(TransferState::kFailed);
      descriptor.status = static_cast<std::uint32_t>(
          DeviceBatchStatus::kAccessTableEntryInvalid);
      return Fail(view, request,
                  DeviceBatchStatus::kAccessTableEntryInvalid,
                  StatusCode::kInvalidArgument,
                  "request references an invalid AccessTable row");
    }

    const Address remote = view.remote_addresses[table_index];
    const std::uint64_t bytes = view.lengths[table_index];
    if (remote == 0 || bytes == 0 || bytes > kAivUrmaMaxTransferBytes ||
        AddOverflows(remote, bytes)) {
      descriptor.state =
          static_cast<std::uint32_t>(TransferState::kFailed);
      descriptor.status =
          static_cast<std::uint32_t>(DeviceBatchStatus::kRemoteRangeInvalid);
      return Fail(view, request, DeviceBatchStatus::kRemoteRangeInvalid,
                  StatusCode::kInvalidArgument,
                  "remote DDR range is empty, null, or wraps the address "
                  "space");
    }

    const std::uint64_t output_offset = view.output_offsets[request];
    if (output_offset > view.output_bytes ||
        bytes > view.output_bytes - output_offset ||
        AddOverflows(view.output_base, output_offset) ||
        AddOverflows(view.output_base + output_offset, bytes)) {
      descriptor.state =
          static_cast<std::uint32_t>(TransferState::kFailed);
      descriptor.status =
          static_cast<std::uint32_t>(DeviceBatchStatus::kOutputRangeInvalid);
      return Fail(view, request, DeviceBatchStatus::kOutputRangeInvalid,
                  StatusCode::kOutOfRange,
                  "requested transfer does not fit in the HBM output buffer");
    }

    descriptor.remote_addr = remote;
    descriptor.local_addr = view.output_base + output_offset;
    descriptor.bytes = bytes;
    descriptor.state = static_cast<std::uint32_t>(TransferState::kReady);
    ++view.completion->planned_count;
  }

  view.completion->state =
      static_cast<std::uint32_t>(DeviceBatchState::kPlanned);
  return Status::OK();
}

}  // namespace kvaf::access
