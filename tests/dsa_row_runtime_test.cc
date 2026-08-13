#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "kvaf/access_operator/access_operator.h"
#include "kvaf/meta_manager/meta_manager.h"
#include "kvaf/runtime/runtime.h"

namespace {

class TestAccessOperator final : public kvaf::AccessOperator {
 public:
  kvaf::Status Initialize(const kvaf::RemoteRegionDesc&) override {
    initialized_ = true;
    return kvaf::Status::OK();
  }

  kvaf::Status Shutdown() override {
    initialized_ = false;
    return kvaf::Status::OK();
  }

  kvaf::Status ExecuteBatchGet(const kvaf::BatchGetExecRequest&) override {
    return kvaf::Status::NotImplemented("not used by DSA_ROW test");
  }

  kvaf::Status ExecuteBatchPut(const kvaf::BatchPutExecRequest&) override {
    return kvaf::Status::NotImplemented("not used by DSA_ROW test");
  }

 private:
  bool initialized_{false};
};

void RequireOk(const kvaf::Status& status) {
  if (!status.ok()) {
    std::cerr << "unexpected KVAF status: " << status.message() << '\n';
    std::abort();
  }
}

void RequireFailure(const kvaf::Status& status) {
  if (status.ok()) {
    std::cerr << "expected KVAF operation to fail" << '\n';
    std::abort();
  }
}

void RunC8AlignmentSmoke() {
  constexpr std::uint32_t kTokensPerBlock = 128;
  constexpr std::uint32_t kPackedTokenBytes = 656;
  constexpr std::uint32_t kPayloadBytes =
      kTokensPerBlock * kPackedTokenBytes;
  static_assert(kPayloadBytes == 83968, "C8 ABI payload changed unexpectedly");

  auto meta = kvaf::CreateInMemoryMetaManager();
  auto allocator = kvaf::CreateHostDdrMemoryAllocator();
  TestAccessOperator operator_impl;
  kvaf::Runtime runtime(*meta, operator_impl, *allocator);
  RequireOk(runtime.Initialize(kvaf::RuntimeInitRequest{
      2 * kPayloadBytes, kPayloadBytes, kvaf::DeviceKind::kCpu}));
  RequireOk(runtime.InitializeDsaRowIndex(kvaf::DsaRowInitRequest{
      kvaf::PackedKvLayout{1, kTokensPerBlock, kPackedTokenBytes, 32},
      kvaf::DsaRowIndexGeometry{1, 1, 1, 0, 0}, 1}));
  RequireOk(runtime.BindResidentPoolRow(
      kvaf::RowBindRequest{0, kTokensPerBlock, 0}));

  std::vector<std::uint8_t> source(kPayloadBytes, 0);
  source.back() = 0x7B;
  kvaf::DsaPutPlanHandle put;
  RequireOk(runtime.PrepareFullBlockGroupPut({{0}}, {{0, 0, 1}}, &put));
  RequireOk(runtime.WriteFullBlockGroupLayer(
      put, {{0, {0, kPayloadBytes}}},
      kvaf::BoundLayerCacheView{
          kvaf::ConstBufferView{source.data(), source.size(),
                                kvaf::DeviceKind::kCpu},
          1, 0}));
  kvaf::DsaSealResult sealed;
  RequireOk(runtime.SealFullBlockGroupPut(put, &sealed));
  kvaf::PutLocalResult committed;
  RequireOk(runtime.CommitFullBlockGroupPut(&put, &committed));

  kvaf::PrepareUnifiedAccessResult access;
  RequireOk(runtime.PrepareUnifiedAccess(
      kvaf::PrepareUnifiedAccessRequest{{
          {kvaf::DsaRowAccessMode::kSparse, true, 0, 1},
      }},
      &access));
  std::vector<std::uint8_t> output(kPackedTokenBytes, 0);
  kvaf::DsaDirectGetPlanHandle get;
  RequireOk(runtime.PrepareDirectBatchGet(
      access.handle,
      kvaf::DirectGetRequest{{{{0, kTokensPerBlock - 1}, 0}},
                              kvaf::MutableBufferView{output.data(), output.size(),
                                                      kvaf::DeviceKind::kCpu},
                              0, nullptr},
      &get));
  kvaf::DirectGetCompletion completion;
  RequireOk(runtime.ExecutePreparedDirectBatchGet(get, &completion));
  kvaf::GetLocalResult settled;
  RequireOk(runtime.SettleDirectBatchGet(&get, completion, &settled));
  assert(output.back() == 0x7B);
  RequireOk(runtime.ReleaseUnifiedAccess(&access.handle));
  kvaf::CloseLocalResult close;
  RequireOk(runtime.CloseResidentPoolRow(0, &close));
  RequireOk(runtime.Shutdown());
}

}  // namespace

int main() {
  constexpr std::uint32_t kTokensPerBlock = 4;
  constexpr std::uint32_t kPackedTokenBytes = 3;
  constexpr std::uint32_t kBlockBytes =
      kTokensPerBlock * kPackedTokenBytes;

  auto meta = kvaf::CreateInMemoryMetaManager();
  auto allocator = kvaf::CreateHostDdrMemoryAllocator();
  TestAccessOperator operator_impl;
  kvaf::Runtime runtime(*meta, operator_impl, *allocator);

  RequireOk(runtime.Initialize(kvaf::RuntimeInitRequest{
      32 * kBlockBytes, kBlockBytes, kvaf::DeviceKind::kCpu}));
  RequireOk(runtime.InitializeDsaRowIndex(kvaf::DsaRowInitRequest{
      kvaf::PackedKvLayout{1, kTokensPerBlock, kPackedTokenBytes, 1},
      kvaf::DsaRowIndexGeometry{2, 2, 4, 0, 0}, 16}));
  // The checked ceil calculation must reject this without overflowing.
  RequireFailure(runtime.BindResidentPoolRow(
      kvaf::RowBindRequest{1, std::numeric_limits<std::uint32_t>::max(), 0}));
  RequireOk(runtime.BindResidentPoolRow(kvaf::RowBindRequest{0, 16, 0}));

  // row 0, two local layers, append block_q 0 and 1 in one atomic group.
  const std::vector<kvaf::KvafPutBlockTarget> targets{
      {0}, {1}, {4}, {5},
  };
  kvaf::DsaPutPlanHandle put;
  RequireOk(runtime.PrepareFullBlockGroupPut(
      targets, {{0, 0, 2}}, &put));

  std::vector<std::uint8_t> layer0(2 * kBlockBytes);
  std::vector<std::uint8_t> layer1(2 * kBlockBytes);
  for (std::size_t index = 0; index < layer0.size(); ++index) {
    layer0[index] = static_cast<std::uint8_t>(index);
    layer1[index] = static_cast<std::uint8_t>(40 + index);
  }
  const std::vector<kvaf::KvafPutBlockItem> layer0_items{
      {0, {0, kBlockBytes}}, {1, {kBlockBytes, kBlockBytes}},
  };
  const std::vector<kvaf::KvafPutBlockItem> layer1_items{
      {4, {0, kBlockBytes}}, {5, {kBlockBytes, kBlockBytes}},
  };
  RequireOk(runtime.WriteFullBlockGroupLayer(
      put, layer0_items,
      kvaf::BoundLayerCacheView{
          kvaf::ConstBufferView{layer0.data(), layer0.size(),
                                kvaf::DeviceKind::kCpu},
          1, 0}));
  RequireOk(runtime.WriteFullBlockGroupLayer(
      put, layer1_items,
      kvaf::BoundLayerCacheView{
          kvaf::ConstBufferView{layer1.data(), layer1.size(),
                                kvaf::DeviceKind::kCpu},
          1, 1}));
  kvaf::DsaSealResult sealed;
  RequireOk(runtime.SealFullBlockGroupPut(put, &sealed));
  assert(sealed.published_block_count == 4);
  kvaf::PutLocalResult committed;
  RequireOk(runtime.CommitFullBlockGroupPut(&put, &committed));
  assert(committed.rows.size() == 1);
  assert(committed.rows[0].committed_block_count == 2);

  kvaf::PrepareUnifiedAccessResult access;
  RequireOk(runtime.PrepareUnifiedAccess(
      kvaf::PrepareUnifiedAccessRequest{{
          {kvaf::DsaRowAccessMode::kSparse, true, 0, 2},
      }},
      &access));
  assert(access.rows.size() == 1);
  assert(access.rows[0].ddr_access_valid);
  assert(access.rows[0].row_flat_base == 0);
  assert(access.rows[0].snapshot_count == 2);

  // Destination allocation is in token units, not complete logical blocks:
  // three 3-byte outputs are scattered in the order [2, 0, 1].
  std::vector<std::uint8_t> output(3 * kPackedTokenBytes, 0);
  kvaf::DsaDirectGetPlanHandle get;
  RequireOk(runtime.PrepareDirectBatchGet(
      access.handle,
      kvaf::DirectGetRequest{{
                                   {{4, 0}, 2},
                                   {{4, 3}, 0},
                                   {{5, 0}, 1},
                               },
                               kvaf::MutableBufferView{output.data(), output.size(),
                                                       kvaf::DeviceKind::kCpu},
                               1, nullptr},
      &get));
  kvaf::DirectGetCompletion completion;
  RequireOk(runtime.ExecutePreparedDirectBatchGet(get, &completion));
  kvaf::GetLocalResult settled;
  RequireOk(runtime.SettleDirectBatchGet(&get, completion, &settled));
  assert(settled.completed_item_count == 3);
  assert(std::memcmp(output.data(), layer1.data() + 9, 3) == 0);
  assert(std::memcmp(output.data() + 3, layer1.data() + 12, 3) == 0);
  assert(std::memcmp(output.data() + 6, layer1.data(), 3) == 0);

  // A malformed mixed batch must not create a DirectGet plan or copy the
  // valid prefix item before the bad item is rejected.
  std::fill(output.begin(), output.end(), 0xA5);
  kvaf::DsaDirectGetPlanHandle malformed_get;
  RequireFailure(runtime.PrepareDirectBatchGet(
      access.handle,
      kvaf::DirectGetRequest{{
                                   {{4, 0}, 0},
                                   {{5, kTokensPerBlock}, 1},
                               },
                               kvaf::MutableBufferView{output.data(), output.size(),
                                                       kvaf::DeviceKind::kCpu},
                               1, nullptr},
      &malformed_get));
  assert(malformed_get.opaque == 0);
  for (const std::uint8_t byte : output) assert(byte == 0xA5);
  RequireOk(runtime.ReleaseUnifiedAccess(&access.handle));

  // Dense is retained in the batch envelope but has no DDR access right.
  kvaf::PrepareUnifiedAccessResult dense_access;
  RequireOk(runtime.PrepareUnifiedAccess(
      kvaf::PrepareUnifiedAccessRequest{{
          {kvaf::DsaRowAccessMode::kDense, true, 0, 0},
      }},
      &dense_access));
  kvaf::DsaDirectGetPlanHandle rejected_get;
  const kvaf::Status rejected = runtime.PrepareDirectBatchGet(
      dense_access.handle,
      kvaf::DirectGetRequest{{{{4, 0}, 0}},
                              kvaf::MutableBufferView{output.data(), output.size(),
                                                      kvaf::DeviceKind::kCpu},
                              1, nullptr},
      &rejected_get);
  assert(!rejected.ok());
  assert(rejected_get.opaque == 0);
  RequireOk(runtime.ReleaseUnifiedAccess(&dense_access.handle));

  // PAD carries fixed batch shape but cannot issue a DDR Get either.
  kvaf::PrepareUnifiedAccessResult pad_access;
  RequireOk(runtime.PrepareUnifiedAccess(
      kvaf::PrepareUnifiedAccessRequest{{
          {kvaf::DsaRowAccessMode::kPad, false, 0, 0},
      }},
      &pad_access));
  kvaf::DsaDirectGetPlanHandle pad_get;
  RequireFailure(runtime.PrepareDirectBatchGet(
      pad_access.handle,
      kvaf::DirectGetRequest{{{{4, 0}, 0}},
                              kvaf::MutableBufferView{output.data(), output.size(),
                                                      kvaf::DeviceKind::kCpu},
                              1, nullptr},
      &pad_get));
  assert(pad_get.opaque == 0);
  RequireOk(runtime.ReleaseUnifiedAccess(&pad_access.handle));

  // A prepared sparse access is a close pin until the batch releases it.
  kvaf::PrepareUnifiedAccessResult close_pinned_access;
  RequireOk(runtime.PrepareUnifiedAccess(
      kvaf::PrepareUnifiedAccessRequest{{
          {kvaf::DsaRowAccessMode::kSparse, true, 0, 2},
      }},
      &close_pinned_access));
  kvaf::CloseLocalResult busy_close;
  RequireFailure(runtime.CloseResidentPoolRow(0, &busy_close));
  RequireOk(runtime.ReleaseUnifiedAccess(&close_pinned_access.handle));

  // Shutdown must not relinquish the DDR region while an OPEN row exists.
  RequireFailure(runtime.Shutdown());
  assert(runtime.initialized());

  kvaf::CloseLocalResult close;
  RequireOk(runtime.CloseResidentPoolRow(0, &close));
  assert(close.released_block_count == 4);
  RequireOk(runtime.Shutdown());
  RunC8AlignmentSmoke();
  return 0;
}
