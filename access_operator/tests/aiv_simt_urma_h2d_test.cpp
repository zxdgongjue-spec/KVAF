#include "kvaf/access_operator/aiv_simt_urma_h2d.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using kvaf::Address;
using kvaf::StatusCode;
using kvaf::access::BuildH2DTransferPlan;
using kvaf::access::DeviceBatchState;
using kvaf::access::DeviceBatchStatus;
using kvaf::access::DeviceCompletion;
using kvaf::access::H2DPlanView;
using kvaf::access::TransferState;
using kvaf::access::UrmaH2DDescriptor;

bool Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
  }
  return condition;
}

H2DPlanView MakeView(
    const std::array<Address, 3>& remote,
    const std::array<std::uint64_t, 3>& lengths,
    const std::array<std::uint8_t, 3>& valid,
    const std::array<std::uint32_t, 3>& indices,
    const std::array<std::uint8_t, 3>& mask,
    const std::array<std::uint64_t, 3>& offsets,
    std::array<UrmaH2DDescriptor, 3>* descriptors,
    DeviceCompletion* completion) {
  H2DPlanView view{};
  view.remote_addresses = remote.data();
  view.lengths = lengths.data();
  view.valid = valid.data();
  view.table_capacity = remote.size();
  view.table_generation = 7;
  view.indices = indices.data();
  view.mask = mask.data();
  view.output_offsets = offsets.data();
  view.entry_count = indices.size();
  view.output_base = 0x80000000;
  view.output_bytes = 192;
  view.generation = 7;
  view.descriptors = descriptors->data();
  view.descriptor_capacity = descriptors->size();
  view.completion = completion;
  return view;
}

bool TestValidPlanAndMask() {
  const std::array<Address, 3> remote{0x100000, 0x200000, 0x300000};
  const std::array<std::uint64_t, 3> lengths{64, 64, 64};
  const std::array<std::uint8_t, 3> valid{1, 1, 1};
  const std::array<std::uint32_t, 3> indices{2, 0, 1};
  const std::array<std::uint8_t, 3> mask{1, 0, 1};
  const std::array<std::uint64_t, 3> offsets{0, 64, 128};
  std::array<UrmaH2DDescriptor, 3> descriptors{};
  DeviceCompletion completion{};

  const auto status = BuildH2DTransferPlan(
      MakeView(remote, lengths, valid, indices, mask, offsets, &descriptors,
               &completion));

  return Expect(status.ok(), "valid plan must succeed") &&
         Expect(completion.state ==
                    static_cast<std::uint32_t>(DeviceBatchState::kPlanned),
                "completion must be planned") &&
         Expect(completion.planned_count == 2,
                "only active entries are planned") &&
         Expect(completion.submitted_count == 0,
                "CPU planner does not claim hardware submission") &&
         Expect(descriptors[0].remote_addr == remote[2],
                "descriptor uses indexed remote address") &&
         Expect(descriptors[0].local_addr == 0x80000000,
                "descriptor uses output byte offset") &&
         Expect(descriptors[1].state ==
                    static_cast<std::uint32_t>(TransferState::kSkipped),
                "masked entry is skipped") &&
         Expect(descriptors[2].local_addr == 0x80000080,
                "last descriptor lands at expected output offset");
}

bool TestBadIndexFailsWholeBatch() {
  const std::array<Address, 3> remote{0x100000, 0x200000, 0x300000};
  const std::array<std::uint64_t, 3> lengths{64, 64, 64};
  const std::array<std::uint8_t, 3> valid{1, 1, 1};
  const std::array<std::uint32_t, 3> indices{0, 9, 1};
  const std::array<std::uint8_t, 3> mask{1, 1, 1};
  const std::array<std::uint64_t, 3> offsets{0, 64, 128};
  std::array<UrmaH2DDescriptor, 3> descriptors{};
  DeviceCompletion completion{};

  const auto status = BuildH2DTransferPlan(
      MakeView(remote, lengths, valid, indices, mask, offsets, &descriptors,
               &completion));

  return Expect(!status.ok(), "out-of-range index must fail") &&
         Expect(status.code() == StatusCode::kOutOfRange,
                "out-of-range index must use kOutOfRange") &&
         Expect(completion.state ==
                    static_cast<std::uint32_t>(DeviceBatchState::kFailed),
                "completion must be failed") &&
         Expect(completion.first_failed_index == 1,
                "failure reports the logical request index") &&
         Expect(completion.planned_count == 0 &&
                    completion.submitted_count == 0 &&
                    completion.completed_count == 0,
                "failed all-or-nothing plan does not expose partial counts") &&
         Expect(completion.status == static_cast<std::uint32_t>(
                                         DeviceBatchStatus::kIndexOutOfRange),
                "device completion preserves the failure reason");
}

bool TestInvalidRemoteAndOutputRanges() {
  std::array<Address, 3> remote{0, 0x200000, 0x300000};
  std::array<std::uint64_t, 3> lengths{64, 64, 64};
  const std::array<std::uint8_t, 3> valid{1, 1, 1};
  const std::array<std::uint32_t, 3> indices{0, 1, 2};
  const std::array<std::uint8_t, 3> mask{1, 1, 1};
  std::array<std::uint64_t, 3> offsets{0, 64, 128};
  std::array<UrmaH2DDescriptor, 3> descriptors{};
  DeviceCompletion completion{};

  auto view = MakeView(remote, lengths, valid, indices, mask, offsets,
                       &descriptors, &completion);
  auto status = BuildH2DTransferPlan(view);
  bool ok =
      Expect(!status.ok(), "zero imported DDR address must fail") &&
      Expect(completion.status == static_cast<std::uint32_t>(
                                      DeviceBatchStatus::kRemoteRangeInvalid),
             "zero remote address has a precise reason");

  remote[0] = 0x100000;
  offsets[2] = 160;
  status = BuildH2DTransferPlan(view);
  ok = Expect(!status.ok(), "output overrun must fail") && ok;
  ok = Expect(completion.first_failed_index == 2,
              "output overrun reports the failing request") &&
       ok;

  offsets[2] = 128;
  lengths[0] =
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) +
      1;
  status = BuildH2DTransferPlan(view);
  ok = Expect(!status.ok(), "transfer larger than the A5 SGE limit must fail") &&
       ok;
  ok = Expect(completion.first_failed_index == 0,
              "oversized transfer reports the failing request") &&
       ok;
  ok = Expect(completion.status == static_cast<std::uint32_t>(
                                      DeviceBatchStatus::kRemoteRangeInvalid),
              "oversized transfer has a precise device status") &&
       ok;

  lengths[0] = 64;
  view.output_base =
      std::numeric_limits<std::uint64_t>::max() - 31;
  view.output_bytes = 64;
  status = BuildH2DTransferPlan(view);
  ok = Expect(!status.ok(), "wrapped local HBM range must fail") && ok;
  ok = Expect(completion.first_failed_index == 0,
              "wrapped local range reports the failing request") &&
       ok;
  ok = Expect(completion.status == static_cast<std::uint32_t>(
                                      DeviceBatchStatus::kOutputRangeInvalid),
              "wrapped local range has a precise device status") &&
       ok;
  return ok;
}

bool TestGenerationMismatch() {
  const std::array<Address, 3> remote{0x100000, 0x200000, 0x300000};
  const std::array<std::uint64_t, 3> lengths{64, 64, 64};
  const std::array<std::uint8_t, 3> valid{1, 1, 1};
  const std::array<std::uint32_t, 3> indices{0, 1, 2};
  const std::array<std::uint8_t, 3> mask{1, 1, 1};
  const std::array<std::uint64_t, 3> offsets{0, 64, 128};
  std::array<UrmaH2DDescriptor, 3> descriptors{};
  DeviceCompletion completion{};
  auto view = MakeView(remote, lengths, valid, indices, mask, offsets,
                       &descriptors, &completion);
  view.table_generation = 6;

  const auto status = BuildH2DTransferPlan(view);
  return Expect(!status.ok(), "stale AccessTable generation must fail") &&
         Expect(completion.status == static_cast<std::uint32_t>(
                                         DeviceBatchStatus::kGenerationMismatch),
                "generation mismatch has a precise device status");
}

bool TestMaximumBatchCapacity() {
  const std::size_t count = kvaf::access::kAivUrmaMaxDescriptors;
  std::vector<Address> remote(count);
  std::vector<std::uint64_t> lengths(count, 64);
  std::vector<std::uint8_t> valid(count, 1);
  std::vector<std::uint32_t> indices(count);
  std::vector<std::uint64_t> offsets(count);
  std::vector<UrmaH2DDescriptor> descriptors(count);
  for (std::size_t index = 0; index < count; ++index) {
    remote[index] = 0x100000ULL + index * 64U;
    indices[index] = static_cast<std::uint32_t>(index);
    offsets[index] = index * 64U;
  }

  DeviceCompletion completion{};
  H2DPlanView view{};
  view.remote_addresses = remote.data();
  view.lengths = lengths.data();
  view.valid = valid.data();
  view.table_capacity = static_cast<std::uint32_t>(count);
  view.table_generation = 11;
  view.indices = indices.data();
  view.output_offsets = offsets.data();
  view.entry_count = static_cast<std::uint32_t>(count);
  view.output_base = 0x80000000;
  view.output_bytes = count * 64U;
  view.generation = 11;
  view.descriptors = descriptors.data();
  view.descriptor_capacity = static_cast<std::uint32_t>(count);
  view.completion = &completion;

  const auto status = BuildH2DTransferPlan(view);
  bool ok =
      Expect(status.ok(), "the full 4096-descriptor capacity must plan") &&
      Expect(completion.planned_count == count,
             "the full batch must report every planned descriptor");

  view.entry_count =
      kvaf::access::kAivUrmaMaxDescriptors + 1U;
  const auto oversized = BuildH2DTransferPlan(view);
  ok = Expect(!oversized.ok(), "capacity plus one must fail before access") &&
       Expect(completion.status == static_cast<std::uint32_t>(
                                       DeviceBatchStatus::kCountOutOfRange),
              "capacity failure must have the precise device status") &&
       ok;
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok = TestValidPlanAndMask() && ok;
  ok = TestBadIndexFailsWholeBatch() && ok;
  ok = TestInvalidRemoteAndOutputRanges() && ok;
  ok = TestGenerationMismatch() && ok;
  ok = TestMaximumBatchCapacity() && ok;
  if (!ok) {
    return 1;
  }
  std::cout << "KVAF_AIV_SIMT_URMA_H2D_UNIT_OK\n";
  return 0;
}
