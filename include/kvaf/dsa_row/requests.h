#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kvaf/dsa_row/types.h"

namespace kvaf {

// Initializes a bounded, fixed row-major directory in the Runtime-owned DDR
// region. ddr_block_count=0 derives the complete region capacity.
struct DsaRowInitRequest {
  PackedKvLayout layout;
  DsaRowIndexGeometry geometry;
  std::uint32_t ddr_block_count{0};
};

struct RowBindRequest {
  std::uint32_t pool_row{0};
  std::uint32_t max_sequence_tokens{0};
  std::uint32_t resident_budget_tokens{0};
};

struct KvafPutBlockTarget {
  std::uint32_t flat_index{0};
};

// A source block refers to caller-owned bytes in BoundLayerCacheView.cache.
// M0 accepts only exact full logical blocks.
struct PackedSourceBlockView {
  std::uint64_t source_byte_offset{0};
  std::uint32_t byte_length{0};
};

struct KvafPutBlockItem {
  std::uint32_t flat_index{0};
  PackedSourceBlockView source_block;
};

// Commit only advances an existing row's contiguous [0, committed) suffix.
// row_flat_base must be pool_row * row_stride_cells.
struct RowCountCommitSpec {
  std::uint32_t row_flat_base{0};
  std::uint32_t expected_committed_block_count{0};
  std::uint32_t appended_block_count{0};
};

struct BoundLayerCacheView {
  ConstBufferView cache;
  std::uint32_t layout_id{1};
  std::uint32_t local_layer_ordinal{0};
};

struct DsaRowCommitResult {
  std::uint32_t pool_row{0};
  std::uint32_t committed_block_count{0};
};

struct PutLocalResult {
  DsaPutPlanState state{DsaPutPlanState::kPrepared};
  std::vector<DsaRowCommitResult> rows;
};

struct DsaSealResult {
  DsaPutPlanState state{DsaPutPlanState::kPrepared};
  std::uint32_t published_block_count{0};
};

struct UnifiedAccessRow {
  DsaRowAccessMode mode{DsaRowAccessMode::kPad};
  // `row_valid` preserves fixed full-batch descriptor shape.  It must be true
  // for a Sparse row to obtain a DDR read pin; PAD is always false.
  bool row_valid{false};
  std::uint32_t pool_row{0};
  // Sparse rows are accepted only when this equals the committed count
  // observed at PrepareUnifiedAccess. Dense/PAD must pass zero.
  std::uint32_t expected_committed_block_count{0};
};

struct PrepareUnifiedAccessRequest {
  std::vector<UnifiedAccessRow> rows;
};

struct DsaRowAccessSnapshot {
  DsaRowAccessMode mode{DsaRowAccessMode::kPad};
  bool row_valid{false};
  bool ddr_access_valid{false};
  std::uint32_t pool_row{0};
  std::uint32_t row_flat_base{0};
  std::uint32_t snapshot_count{0};
};

struct PrepareUnifiedAccessResult {
  DsaRowAccessHandle handle;
  std::vector<DsaRowAccessSnapshot> rows;
};

// No KeyId, key hash, or AccessTable lookup participates in this source.
struct KvafGetSource {
  std::uint32_t flat_index{0};
  std::uint32_t offset_in_block{0};
};

struct KvafGetItem {
  KvafGetSource source;
  std::int32_t destination_index{0};
};

struct DirectGetRequest {
  std::vector<KvafGetItem> items;
  MutableBufferView output;
  std::uint32_t output_layer_ordinal{0};
  void* stream{nullptr};  // retained for the future asynchronous backend.
};

struct DirectGetCompletion {
  Event event;
};

struct GetLocalResult {
  DsaGetPlanState state{DsaGetPlanState::kPrepared};
  std::uint32_t completed_item_count{0};
};

struct CloseLocalResult {
  DsaRowLifecycleState state{DsaRowLifecycleState::kFree};
  std::uint32_t released_block_count{0};
};

}  // namespace kvaf
