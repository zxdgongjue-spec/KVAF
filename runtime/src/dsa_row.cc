#include "kvaf/runtime/runtime.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace kvaf {

namespace {

constexpr std::uint32_t kInvalidDsaBlock =
    std::numeric_limits<std::uint32_t>::max();

bool AddWouldOverflow(std::uint64_t left, std::uint64_t right) {
  return left > std::numeric_limits<std::uint64_t>::max() - right;
}

bool MultiplyWouldOverflow(std::uint64_t left, std::uint64_t right) {
  return left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left;
}

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
  if (alignment == 0 || value == 0 ||
      alignment > std::numeric_limits<std::uint64_t>::max() - value + 1) {
    return 0;
  }
  const std::uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

}  // namespace

// Runtime-owned CPU reference implementation of the DSA row directory.  It
// purposefully does not call MetaManager or use KeyId: row-major flat indices
// are the only addressing mechanism in this data path.
class Runtime::DsaRowState {
 public:
  Status Initialize(const RemoteRegionDesc& remote_region,
                    const DsaRowInitRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
      return Status::InvalidState("DSA_ROW index is already initialized");
    }
    if (remote_region.device != DeviceKind::kCpu) {
      return Status::NotImplemented(
          "DSA_ROW M0 reference implementation supports CPU DDR only");
    }
    if (request.layout.layout_id == 0 ||
        request.layout.tokens_per_logical_block == 0 ||
        request.layout.packed_token_bytes == 0 ||
        request.layout.ddr_block_alignment_bytes == 0) {
      return Status::InvalidArgument("DSA_ROW layout fields must be non-zero");
    }
    if (MultiplyWouldOverflow(request.layout.tokens_per_logical_block,
                              request.layout.packed_token_bytes)) {
      return Status::InvalidArgument("DSA_ROW block payload size overflows");
    }
    const std::uint64_t payload_bytes =
        static_cast<std::uint64_t>(request.layout.tokens_per_logical_block) *
        request.layout.packed_token_bytes;
    if (payload_bytes > std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument(
          "DSA_ROW block payload exceeds the uint32 source-block ABI");
    }
    const std::uint64_t block_bytes =
        AlignUp(payload_bytes, request.layout.ddr_block_alignment_bytes);
    if (block_bytes == 0 || remote_region.slot_size != block_bytes ||
        remote_region.length == 0 || remote_region.length % block_bytes != 0) {
      return Status::InvalidArgument(
          "DSA_ROW requires Runtime slot_size and DDR length to match its full "
          "logical block size");
    }
    if (request.geometry.raw_row_capacity == 0 ||
        request.geometry.n_layers == 0 || request.geometry.q_capacity == 0) {
      return Status::InvalidArgument("DSA_ROW geometry dimensions must be non-zero");
    }
    if (MultiplyWouldOverflow(request.geometry.n_layers,
                              request.geometry.q_capacity)) {
      return Status::InvalidArgument("DSA_ROW row stride overflows");
    }
    const std::uint64_t row_stride =
        static_cast<std::uint64_t>(request.geometry.n_layers) *
        request.geometry.q_capacity;
    if (row_stride == 0 ||
        row_stride > std::numeric_limits<std::uint32_t>::max() ||
        MultiplyWouldOverflow(request.geometry.raw_row_capacity, row_stride)) {
      return Status::InvalidArgument("DSA_ROW directory geometry overflows");
    }
    const std::uint64_t directory_cells =
        static_cast<std::uint64_t>(request.geometry.raw_row_capacity) * row_stride;
    if (directory_cells > std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument(
          "DSA_ROW M0 flat_index is uint32 and directory is too large");
    }
    if ((request.geometry.row_stride_cells != 0 &&
         request.geometry.row_stride_cells != row_stride) ||
        (request.geometry.directory_cell_count != 0 &&
         request.geometry.directory_cell_count != directory_cells)) {
      return Status::InvalidArgument(
          "DSA_ROW redundant geometry fields disagree with derived geometry");
    }

    const std::uint64_t region_block_count = remote_region.length / block_bytes;
    const std::uint64_t requested_block_count = request.ddr_block_count == 0
                                                    ? region_block_count
                                                    : request.ddr_block_count;
    if (requested_block_count == 0 || requested_block_count > region_block_count ||
        requested_block_count > std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument("DSA_ROW requested DDR block count is invalid");
    }

    remote_region_ = remote_region;
    layout_ = request.layout;
    block_payload_bytes_ = payload_bytes;
    block_bytes_ = block_bytes;
    geometry_ = request.geometry;
    geometry_.row_stride_cells = static_cast<std::uint32_t>(row_stride);
    geometry_.directory_cell_count = directory_cells;

    rows_.assign(geometry_.raw_row_capacity, RowRecord{});
    host_cells_.assign(static_cast<std::size_t>(directory_cells), Cell{});
    host_ddr_block_table_.assign(static_cast<std::size_t>(directory_cells), 0);
    device_ddr_block_table_.assign(static_cast<std::size_t>(directory_cells), 0);
    ddr_blocks_.assign(static_cast<std::size_t>(requested_block_count),
                       DdrBlockRecord{});
    free_blocks_.clear();
    free_blocks_.reserve(static_cast<std::size_t>(requested_block_count));
    for (std::uint32_t block = 0; block < requested_block_count; ++block) {
      free_blocks_.push_back(block);
    }
    initialized_ = true;
    return Status::OK();
  }

  Status GetDirectoryView(DsaRowDirectoryView* view) const {
    if (view == nullptr) {
      return Status::InvalidArgument("DSA_ROW directory view must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    *view = DsaRowDirectoryView{host_ddr_block_table_.data(), DeviceAddressData(),
                                geometry_.directory_cell_count, geometry_, layout_};
    return Status::OK();
  }

  Status GetCommittedBlockCount(std::uint32_t pool_row,
                                std::uint32_t* count) const {
    if (count == nullptr) {
      return Status::InvalidArgument("DSA_ROW committed block count must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    if (pool_row >= rows_.size()) {
      return Status::InvalidArgument("DSA_ROW pool row is outside geometry");
    }
    const RowRecord& row = rows_[pool_row];
    if (row.state != DsaRowLifecycleState::kOpen) {
      return Status::InvalidState("DSA_ROW row is not open");
    }
    *count = row.committed_block_count;
    return Status::OK();
  }

  Status Bind(const RowBindRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    if (request.pool_row >= rows_.size() || request.max_sequence_tokens == 0 ||
        request.resident_budget_tokens > request.max_sequence_tokens) {
      return Status::InvalidArgument("DSA_ROW Bind row or token budget is invalid");
    }
    const std::uint64_t rounded_block_count =
        CeilDiv(request.max_sequence_tokens, layout_.tokens_per_logical_block);
    if (rounded_block_count == 0 || rounded_block_count > geometry_.q_capacity) {
      return Status::InvalidArgument("DSA_ROW Bind max_sequence_tokens exceeds q_capacity");
    }
    RowRecord& row = rows_[request.pool_row];
    if (row.state != DsaRowLifecycleState::kFree) {
      return Status::InvalidState("DSA_ROW pool row is not free");
    }
    const std::uint32_t row_base = request.pool_row * geometry_.row_stride_cells;
    for (std::uint32_t index = 0; index < geometry_.row_stride_cells; ++index) {
      if (host_cells_[row_base + index].block_id != kInvalidDsaBlock) {
        return Status::InvalidState("DSA_ROW free row still owns a DDR block");
      }
    }
    row.state = DsaRowLifecycleState::kBinding;
    row.max_sequence_tokens = request.max_sequence_tokens;
    row.resident_budget_tokens = request.resident_budget_tokens;
    row.max_block_count = static_cast<std::uint32_t>(rounded_block_count);
    row.committed_block_count = 0;
    row.state = DsaRowLifecycleState::kOpen;
    return Status::OK();
  }

  Status AbortBind(std::uint32_t pool_row) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    if (pool_row >= rows_.size()) {
      return Status::InvalidArgument("DSA_ROW pool row is outside geometry");
    }
    RowRecord& row = rows_[pool_row];
    if (row.state != DsaRowLifecycleState::kOpen ||
        row.committed_block_count != 0 || row.active_put_id != 0 ||
        row.active_sparse_accesses != 0 || row.active_direct_gets != 0) {
      return Status::Busy("DSA_ROW row cannot be aborted after it became active");
    }
    row = RowRecord{};
    return Status::OK();
  }

  Status PreparePut(const std::vector<KvafPutBlockTarget>& targets,
                    const std::vector<RowCountCommitSpec>& row_commits,
                    DsaPutPlanHandle* handle) {
    if (handle == nullptr) {
      return Status::InvalidArgument("DSA_ROW put plan handle must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    if (targets.empty() || row_commits.empty()) {
      return Status::InvalidArgument("DSA_ROW group put needs targets and row commits");
    }

    std::vector<std::uint32_t> expected_flats;
    std::vector<std::uint32_t> commit_rows;
    std::uint64_t expected_target_count = 0;
    for (const RowCountCommitSpec& commit : row_commits) {
      if (commit.appended_block_count == 0 ||
          commit.row_flat_base % geometry_.row_stride_cells != 0) {
        return Status::InvalidArgument("DSA_ROW row commit is malformed");
      }
      const std::uint32_t pool_row =
          commit.row_flat_base / geometry_.row_stride_cells;
      if (pool_row >= rows_.size() ||
          commit.row_flat_base != pool_row * geometry_.row_stride_cells ||
          std::find(commit_rows.begin(), commit_rows.end(), pool_row) !=
              commit_rows.end()) {
        return Status::InvalidArgument("DSA_ROW row commit has an invalid or duplicate row");
      }
      RowRecord& row = rows_[pool_row];
      if (row.state != DsaRowLifecycleState::kOpen || row.active_put_id != 0 ||
          commit.expected_committed_block_count != row.committed_block_count ||
          row.committed_block_count > row.max_block_count ||
          commit.appended_block_count >
              row.max_block_count - row.committed_block_count) {
        return Status::InvalidState("DSA_ROW row commit does not match live row state");
      }
      if (MultiplyWouldOverflow(commit.appended_block_count, geometry_.n_layers) ||
          AddWouldOverflow(expected_target_count,
                           static_cast<std::uint64_t>(commit.appended_block_count) *
                               geometry_.n_layers)) {
        return Status::InvalidArgument("DSA_ROW group put target count overflows");
      }
      expected_target_count +=
          static_cast<std::uint64_t>(commit.appended_block_count) * geometry_.n_layers;
      commit_rows.push_back(pool_row);
      for (std::uint32_t layer = 0; layer < geometry_.n_layers; ++layer) {
        for (std::uint32_t block = 0; block < commit.appended_block_count;
             ++block) {
          const std::uint32_t q = commit.expected_committed_block_count + block;
          const std::uint32_t flat =
              commit.row_flat_base + layer * geometry_.q_capacity + q;
          if (host_cells_[flat].block_id != kInvalidDsaBlock) {
            return Status::InvalidState(
                "DSA_ROW append range already has an owned DDR block");
          }
          expected_flats.push_back(flat);
        }
      }
    }
    if (expected_target_count != targets.size() ||
        expected_target_count > std::numeric_limits<std::size_t>::max()) {
      return Status::InvalidArgument(
          "DSA_ROW targets do not cover exactly all rows/layers/append blocks");
    }
    std::vector<std::uint32_t> supplied_flats;
    supplied_flats.reserve(targets.size());
    for (const KvafPutBlockTarget& target : targets) {
      if (target.flat_index >= host_cells_.size()) {
        return Status::InvalidArgument("DSA_ROW put target flat index is invalid");
      }
      supplied_flats.push_back(target.flat_index);
    }
    std::sort(expected_flats.begin(), expected_flats.end());
    std::sort(supplied_flats.begin(), supplied_flats.end());
    if (expected_flats != supplied_flats ||
        std::adjacent_find(supplied_flats.begin(), supplied_flats.end()) !=
            supplied_flats.end()) {
      return Status::InvalidArgument(
          "DSA_ROW targets must be an exact duplicate-free full group");
    }
    if (free_blocks_.size() < targets.size()) {
      return Status::OutOfMemory("DSA_ROW DDR arena has insufficient free blocks");
    }
    const std::uint64_t plan_id = NextHandle(&next_put_id_);
    if (plan_id == 0) {
      return Status::InvalidState("DSA_ROW put plan handle space is exhausted");
    }

    PutPlan plan;
    plan.state = DsaPutPlanState::kPrepared;
    plan.row_commits = row_commits;
    plan.targets.reserve(targets.size());
    for (const KvafPutBlockTarget& target : targets) {
      const std::uint32_t ddr_block = free_blocks_.back();
      free_blocks_.pop_back();
      ddr_blocks_[ddr_block].state = DsaRowBlockState::kReserved;
      plan.targets.push_back(PutTarget{target.flat_index, ddr_block, false});
    }
    for (const std::uint32_t pool_row : commit_rows) {
      rows_[pool_row].active_put_id = plan_id;
    }
    put_plans_.emplace(plan_id, std::move(plan));
    *handle = DsaPutPlanHandle{plan_id};
    return Status::OK();
  }

  Status WriteLayer(const DsaPutPlanHandle& handle,
                    const std::vector<KvafPutBlockItem>& layer_blocks,
                    const BoundLayerCacheView& layer_cache, void* /*stream*/) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    PutPlan* plan = FindPutPlan(handle);
    if (plan == nullptr) {
      return Status::NotFound("DSA_ROW put plan was not found");
    }
    if (plan->state != DsaPutPlanState::kPrepared ||
        layer_cache.layout_id != layout_.layout_id ||
        layer_cache.local_layer_ordinal >= geometry_.n_layers ||
        layer_cache.cache.device != DeviceKind::kCpu ||
        (!layer_blocks.empty() && !layer_cache.cache.valid())) {
      return Status::InvalidArgument("DSA_ROW layer cache is invalid");
    }

    std::vector<std::uint32_t> expected_flats;
    for (const PutTarget& target : plan->targets) {
      std::uint32_t ignored_pool_row = 0;
      std::uint32_t local_layer = 0;
      std::uint32_t ignored_block_q = 0;
      DecodeFlat(target.flat_index, &ignored_pool_row, &local_layer,
                 &ignored_block_q);
      if (local_layer == layer_cache.local_layer_ordinal) {
        expected_flats.push_back(target.flat_index);
      }
    }
    if (expected_flats.size() != layer_blocks.size()) {
      return Status::InvalidArgument(
          "DSA_ROW WriteLayer requires every planned block for one layer");
    }
    std::vector<std::uint32_t> supplied_flats;
    supplied_flats.reserve(layer_blocks.size());
    for (const KvafPutBlockItem& item : layer_blocks) {
      if (item.flat_index >= host_cells_.size() ||
          item.source_block.byte_length != block_payload_bytes_ ||
          item.source_block.source_byte_offset > layer_cache.cache.bytes ||
          item.source_block.byte_length >
              layer_cache.cache.bytes - item.source_block.source_byte_offset) {
        return Status::InvalidArgument("DSA_ROW layer block source is invalid");
      }
      std::uint32_t ignored_pool_row = 0;
      std::uint32_t local_layer = 0;
      std::uint32_t ignored_block_q = 0;
      DecodeFlat(item.flat_index, &ignored_pool_row, &local_layer,
                 &ignored_block_q);
      if (local_layer != layer_cache.local_layer_ordinal) {
        return Status::InvalidArgument("DSA_ROW layer block has a mismatched layer");
      }
      supplied_flats.push_back(item.flat_index);
    }
    std::sort(expected_flats.begin(), expected_flats.end());
    std::sort(supplied_flats.begin(), supplied_flats.end());
    if (expected_flats != supplied_flats ||
        std::adjacent_find(supplied_flats.begin(), supplied_flats.end()) !=
            supplied_flats.end()) {
      return Status::InvalidArgument(
          "DSA_ROW layer block set must exactly match the planned layer");
    }

    for (const KvafPutBlockItem& item : layer_blocks) {
      PutTarget* target = FindPutTarget(plan, item.flat_index);
      if (target == nullptr || target->written ||
          ddr_blocks_[target->ddr_block].state != DsaRowBlockState::kReserved) {
        return Status::InvalidState("DSA_ROW layer block is not writable");
      }
      const auto* source = static_cast<const std::uint8_t*>(layer_cache.cache.data) +
                           static_cast<std::size_t>(item.source_block.source_byte_offset);
      std::memcpy(reinterpret_cast<void*>(DdrBlockAddress(target->ddr_block)), source,
                  static_cast<std::size_t>(block_payload_bytes_));
      target->written = true;
    }
    return Status::OK();
  }

  Status Seal(const DsaPutPlanHandle& handle, DsaSealResult* result) {
    if (result == nullptr) {
      return Status::InvalidArgument("DSA_ROW seal result must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    PutPlan* plan = FindPutPlan(handle);
    if (plan == nullptr) {
      return Status::NotFound("DSA_ROW put plan was not found");
    }
    if (plan->state != DsaPutPlanState::kPrepared) {
      return Status::InvalidState("DSA_ROW put plan is not ready to seal");
    }
    for (const PutTarget& target : plan->targets) {
      if (!target.written ||
          ddr_blocks_[target.ddr_block].state != DsaRowBlockState::kReserved) {
        return Status::InvalidState("DSA_ROW put plan has unwritten blocks");
      }
    }
    // Publication order is intentional: the reference device table receives
    // the complete plan before host ownership/counts become observable.
    for (const PutTarget& target : plan->targets) {
      const Address address = DdrBlockAddress(target.ddr_block);
      device_ddr_block_table_[target.flat_index] = address;
    }
    for (const PutTarget& target : plan->targets) {
      host_cells_[target.flat_index].block_id = target.ddr_block;
      host_ddr_block_table_[target.flat_index] = DdrBlockAddress(target.ddr_block);
      ddr_blocks_[target.ddr_block].state = DsaRowBlockState::kOwned;
    }
    plan->state = DsaPutPlanState::kSealed;
    *result = DsaSealResult{plan->state,
                            static_cast<std::uint32_t>(plan->targets.size())};
    return Status::OK();
  }

  Status Commit(DsaPutPlanHandle* handle, PutLocalResult* result) {
    if (handle == nullptr || result == nullptr) {
      return Status::InvalidArgument("DSA_ROW commit arguments must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    const auto it = put_plans_.find(handle->opaque);
    if (it == put_plans_.end()) {
      return Status::NotFound("DSA_ROW put plan was not found");
    }
    PutPlan& plan = it->second;
    if (plan.state != DsaPutPlanState::kSealed) {
      return Status::InvalidState("DSA_ROW put plan must be sealed before commit");
    }
    PutLocalResult local_result;
    local_result.state = DsaPutPlanState::kCommitted;
    for (const RowCountCommitSpec& commit : plan.row_commits) {
      const std::uint32_t pool_row =
          commit.row_flat_base / geometry_.row_stride_cells;
      RowRecord& row = rows_[pool_row];
      row.committed_block_count += commit.appended_block_count;
      row.active_put_id = 0;
      local_result.rows.push_back(
          DsaRowCommitResult{pool_row, row.committed_block_count});
    }
    *result = std::move(local_result);
    put_plans_.erase(it);
    *handle = DsaPutPlanHandle{};
    return Status::OK();
  }

  Status Abort(DsaPutPlanHandle* handle, PutLocalResult* result) {
    if (handle == nullptr || result == nullptr) {
      return Status::InvalidArgument("DSA_ROW abort arguments must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    const auto it = put_plans_.find(handle->opaque);
    if (it == put_plans_.end()) {
      return Status::NotFound("DSA_ROW put plan was not found");
    }
    PutPlan& plan = it->second;
    PutLocalResult local_result;
    local_result.state = DsaPutPlanState::kAborted;
    for (const PutTarget& target : plan.targets) {
      if (host_cells_[target.flat_index].block_id == target.ddr_block) {
        host_cells_[target.flat_index].block_id = kInvalidDsaBlock;
        host_ddr_block_table_[target.flat_index] = 0;
        device_ddr_block_table_[target.flat_index] = 0;
      }
      if (ddr_blocks_[target.ddr_block].state != DsaRowBlockState::kFree) {
        ddr_blocks_[target.ddr_block].state = DsaRowBlockState::kFree;
        free_blocks_.push_back(target.ddr_block);
      }
    }
    for (const RowCountCommitSpec& commit : plan.row_commits) {
      const std::uint32_t pool_row =
          commit.row_flat_base / geometry_.row_stride_cells;
      RowRecord& row = rows_[pool_row];
      if (row.active_put_id == handle->opaque) {
        row.active_put_id = 0;
      }
      local_result.rows.push_back(
          DsaRowCommitResult{pool_row, row.committed_block_count});
    }
    *result = std::move(local_result);
    put_plans_.erase(it);
    *handle = DsaPutPlanHandle{};
    return Status::OK();
  }

  Status PrepareAccess(const PrepareUnifiedAccessRequest& request,
                       PrepareUnifiedAccessResult* result) {
    if (result == nullptr) {
      return Status::InvalidArgument("DSA_ROW access result must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    const std::uint64_t access_id = NextHandle(&next_access_id_);
    if (access_id == 0) {
      return Status::InvalidState("DSA_ROW access handle space is exhausted");
    }
    AccessState access;
    PrepareUnifiedAccessResult local_result;
    local_result.handle = DsaRowAccessHandle{access_id};
    for (const UnifiedAccessRow& input : request.rows) {
      DsaRowAccessSnapshot snapshot;
      snapshot.mode = input.mode;
      snapshot.row_valid = input.row_valid;
      snapshot.pool_row = input.pool_row;
      if (input.mode == DsaRowAccessMode::kPad) {
        if (input.row_valid || input.expected_committed_block_count != 0) {
          return Status::InvalidArgument(
              "DSA_ROW PAD access must be invalid and have zero expected count");
        }
        local_result.rows.push_back(snapshot);
        access.rows.push_back(snapshot);
        continue;
      }
      if (input.mode == DsaRowAccessMode::kDense) {
        if (!input.row_valid || input.expected_committed_block_count != 0) {
          return Status::InvalidArgument(
              "DSA_ROW Dense access must be valid and have zero expected count");
        }
        local_result.rows.push_back(snapshot);
        access.rows.push_back(snapshot);
        continue;
      }
      if (input.mode != DsaRowAccessMode::kSparse || !input.row_valid) {
        return Status::InvalidArgument("DSA_ROW Sparse access must be row-valid");
      }
      if (input.pool_row >= rows_.size() ||
          std::find(access.used_rows.begin(), access.used_rows.end(),
                    input.pool_row) != access.used_rows.end()) {
        return Status::InvalidArgument("DSA_ROW access contains invalid or duplicate row");
      }
      RowRecord& row = rows_[input.pool_row];
      if (row.state != DsaRowLifecycleState::kOpen) {
        return Status::InvalidState("DSA_ROW access row is not open");
      }
      if (input.expected_committed_block_count != row.committed_block_count) {
        return Status::InvalidState(
            "DSA_ROW Sparse access expected count disagrees with committed count");
      }
      snapshot.ddr_access_valid = true;
      snapshot.row_flat_base = input.pool_row * geometry_.row_stride_cells;
      snapshot.snapshot_count = row.committed_block_count;
      access.used_rows.push_back(input.pool_row);
      access.pinned_rows.push_back(input.pool_row);
      local_result.rows.push_back(snapshot);
      access.rows.push_back(snapshot);
    }
    // All validation above completes before changing row state.  This keeps a
    // rejected heterogeneous batch from leaking sparse-read pins.
    for (const std::uint32_t pool_row : access.pinned_rows) {
      ++rows_[pool_row].active_sparse_accesses;
    }
    access_states_.emplace(access_id, std::move(access));
    *result = std::move(local_result);
    return Status::OK();
  }

  Status PrepareDirectGet(const DsaRowAccessHandle& access_handle,
                          const DirectGetRequest& request,
                          DsaDirectGetPlanHandle* handle) {
    if (handle == nullptr) {
      return Status::InvalidArgument("DSA_ROW direct get plan must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    auto access_it = access_states_.find(access_handle.opaque);
    if (access_it == access_states_.end()) {
      return Status::NotFound("DSA_ROW access handle was not found");
    }
    if (request.output_layer_ordinal >= geometry_.n_layers ||
        request.output.device != DeviceKind::kCpu ||
        (!request.items.empty() && !request.output.valid()) ||
        request.output.bytes % layout_.packed_token_bytes != 0) {
      return Status::InvalidArgument("DSA_ROW direct get output is invalid");
    }
    const std::size_t output_rows =
        request.output.bytes /
        static_cast<std::size_t>(layout_.packed_token_bytes);
    std::vector<std::int32_t> output_indices;
    output_indices.reserve(request.items.size());
    GetPlan plan;
    plan.access_id = access_handle.opaque;
    plan.copies.reserve(request.items.size());
    for (const KvafGetItem& item : request.items) {
      if (item.destination_index < 0 ||
          static_cast<std::size_t>(item.destination_index) >= output_rows ||
          item.source.flat_index >= host_cells_.size() ||
          item.source.offset_in_block >= layout_.tokens_per_logical_block) {
        return Status::InvalidArgument("DSA_ROW direct get item is out of range");
      }
      std::uint32_t pool_row = 0;
      std::uint32_t local_layer = 0;
      std::uint32_t block_q = 0;
      DecodeFlat(item.source.flat_index, &pool_row, &local_layer, &block_q);
      if (local_layer != request.output_layer_ordinal ||
          !AccessAllows(access_it->second, pool_row, block_q)) {
        return Status::InvalidState(
            "DSA_ROW direct get source is outside the prepared access snapshot");
      }
      const Cell& cell = host_cells_[item.source.flat_index];
      const Address published_address = device_ddr_block_table_[item.source.flat_index];
      if (cell.block_id == kInvalidDsaBlock || published_address == 0 ||
          ddr_blocks_[cell.block_id].state != DsaRowBlockState::kOwned) {
        return Status::InvalidState("DSA_ROW direct get source cell is not owned");
      }
      const std::uint64_t token_offset =
          static_cast<std::uint64_t>(item.source.offset_in_block) *
          layout_.packed_token_bytes;
      if (token_offset + layout_.packed_token_bytes > block_payload_bytes_) {
        return Status::Internal("DSA_ROW direct get token offset escaped payload");
      }
      output_indices.push_back(item.destination_index);
      if (std::find(plan.source_rows.begin(), plan.source_rows.end(), pool_row) ==
          plan.source_rows.end()) {
        plan.source_rows.push_back(pool_row);
      }
      auto* destination = static_cast<std::uint8_t*>(request.output.data) +
                          static_cast<std::size_t>(item.destination_index) *
                              layout_.packed_token_bytes;
      plan.copies.push_back(GetCopy{published_address + token_offset, destination});
    }
    std::sort(output_indices.begin(), output_indices.end());
    if (std::adjacent_find(output_indices.begin(), output_indices.end()) !=
        output_indices.end()) {
      return Status::InvalidArgument("DSA_ROW direct get destination indices must be unique");
    }
    const std::uint64_t plan_id = NextHandle(&next_get_id_);
    if (plan_id == 0) {
      return Status::InvalidState("DSA_ROW direct get handle space is exhausted");
    }
    for (const std::uint32_t pool_row : plan.source_rows) {
      ++rows_[pool_row].active_direct_gets;
    }
    ++access_it->second.active_gets;
    get_plans_.emplace(plan_id, std::move(plan));
    *handle = DsaDirectGetPlanHandle{plan_id};
    return Status::OK();
  }

  Status ExecuteDirectGet(const DsaDirectGetPlanHandle& handle,
                          DirectGetCompletion* completion) {
    if (completion == nullptr) {
      return Status::InvalidArgument("DSA_ROW direct get completion must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    GetPlan* plan = FindGetPlan(handle);
    if (plan == nullptr) {
      return Status::NotFound("DSA_ROW direct get plan was not found");
    }
    if (plan->state != DsaGetPlanState::kPrepared) {
      return Status::InvalidState("DSA_ROW direct get plan was already executed");
    }
    for (const GetCopy& copy : plan->copies) {
      std::memcpy(copy.destination, reinterpret_cast<const void*>(copy.source),
                  static_cast<std::size_t>(layout_.packed_token_bytes));
    }
    plan->state = DsaGetPlanState::kExecuted;
    *completion = DirectGetCompletion{Event{handle.opaque}};
    return Status::OK();
  }

  Status SettleDirectGet(DsaDirectGetPlanHandle* handle,
                         const DirectGetCompletion& completion,
                         GetLocalResult* result) {
    if (handle == nullptr || result == nullptr) {
      return Status::InvalidArgument("DSA_ROW direct get settle arguments must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    const auto it = get_plans_.find(handle->opaque);
    if (it == get_plans_.end()) {
      return Status::NotFound("DSA_ROW direct get plan was not found");
    }
    GetPlan& plan = it->second;
    if (plan.state != DsaGetPlanState::kExecuted ||
        completion.event.opaque != handle->opaque) {
      return Status::InvalidState("DSA_ROW direct get completion does not match plan");
    }
    const auto access_it = access_states_.find(plan.access_id);
    if (access_it == access_states_.end() || access_it->second.active_gets == 0) {
      return Status::Internal("DSA_ROW direct get lost its access state");
    }
    --access_it->second.active_gets;
    for (const std::uint32_t pool_row : plan.source_rows) {
      RowRecord& row = rows_[pool_row];
      if (row.active_direct_gets == 0) {
        return Status::Internal("DSA_ROW direct get row pin underflow");
      }
      --row.active_direct_gets;
    }
    *result = GetLocalResult{DsaGetPlanState::kSettled,
                             static_cast<std::uint32_t>(plan.copies.size())};
    get_plans_.erase(it);
    *handle = DsaDirectGetPlanHandle{};
    return Status::OK();
  }

  Status AbortDirectGet(DsaDirectGetPlanHandle* handle,
                        GetLocalResult* result) {
    if (handle == nullptr || result == nullptr) {
      return Status::InvalidArgument("DSA_ROW direct get abort arguments must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    const auto it = get_plans_.find(handle->opaque);
    if (it == get_plans_.end()) {
      return Status::NotFound("DSA_ROW direct get plan was not found");
    }
    GetPlan& plan = it->second;
    if (plan.state != DsaGetPlanState::kPrepared &&
        plan.state != DsaGetPlanState::kExecuted) {
      return Status::InvalidState("DSA_ROW direct get is not abortable");
    }
    const auto access_it = access_states_.find(plan.access_id);
    if (access_it == access_states_.end() || access_it->second.active_gets == 0) {
      return Status::Internal("DSA_ROW direct get lost its access state");
    }
    --access_it->second.active_gets;
    for (const std::uint32_t pool_row : plan.source_rows) {
      RowRecord& row = rows_[pool_row];
      if (row.active_direct_gets == 0) {
        return Status::Internal("DSA_ROW direct get row pin underflow");
      }
      --row.active_direct_gets;
      // A backend may report failure after it has posted an asynchronous read.
      // Preserve strict no-reuse semantics in that ambiguous case.  The CPU
      // M0 backend is synchronous, but retaining this state transition keeps
      // the API safe when a future transport changes execution timing.
      if (plan.state == DsaGetPlanState::kExecuted) {
        row.state = DsaRowLifecycleState::kQuarantined;
      }
    }
    *result = GetLocalResult{DsaGetPlanState::kAborted, 0};
    get_plans_.erase(it);
    *handle = DsaDirectGetPlanHandle{};
    return Status::OK();
  }

  Status ReleaseAccess(DsaRowAccessHandle* handle) {
    if (handle == nullptr) {
      return Status::InvalidArgument("DSA_ROW access handle must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    const auto it = access_states_.find(handle->opaque);
    if (it == access_states_.end()) {
      return Status::NotFound("DSA_ROW access handle was not found");
    }
    if (it->second.active_gets != 0) {
      return Status::Busy("DSA_ROW access still owns direct get plans");
    }
    for (const std::uint32_t pool_row : it->second.pinned_rows) {
      RowRecord& row = rows_[pool_row];
      if (row.active_sparse_accesses == 0) {
        return Status::Internal("DSA_ROW sparse access pin underflow");
      }
      --row.active_sparse_accesses;
    }
    access_states_.erase(it);
    *handle = DsaRowAccessHandle{};
    return Status::OK();
  }

  Status Close(std::uint32_t pool_row, CloseLocalResult* result) {
    if (result == nullptr) {
      return Status::InvalidArgument("DSA_ROW close result must not be null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return Status::NotInitialized("DSA_ROW index has not been initialized");
    }
    if (pool_row >= rows_.size()) {
      return Status::InvalidArgument("DSA_ROW pool row is outside geometry");
    }
    RowRecord& row = rows_[pool_row];
    if (row.state != DsaRowLifecycleState::kOpen) {
      return Status::InvalidState("DSA_ROW row is not open");
    }
    if (row.active_put_id != 0 || row.active_sparse_accesses != 0 ||
        row.active_direct_gets != 0) {
      return Status::Busy(
          "DSA_ROW row still has active put, sparse access, or direct get");
    }
    const std::uint32_t row_base = pool_row * geometry_.row_stride_cells;
    // Validate the whole row before returning any physical allocation.  A
    // close that discovers an unknown owner must quarantine everything rather
    // than expose a partially recycled prefix to a later bind.
    for (std::uint32_t offset = 0; offset < geometry_.row_stride_cells; ++offset) {
      const Cell& cell = host_cells_[row_base + offset];
      if (cell.block_id == kInvalidDsaBlock) {
        continue;
      }
      if (cell.block_id >= ddr_blocks_.size() ||
          ddr_blocks_[cell.block_id].state != DsaRowBlockState::kOwned) {
        row.state = DsaRowLifecycleState::kQuarantined;
        return Status::InvalidState(
            "DSA_ROW close quarantined row with an unknown block owner");
      }
    }
    row.state = DsaRowLifecycleState::kClosing;
    std::uint32_t released = 0;
    for (std::uint32_t offset = 0; offset < geometry_.row_stride_cells; ++offset) {
      Cell& cell = host_cells_[row_base + offset];
      if (cell.block_id == kInvalidDsaBlock) {
        continue;
      }
      ddr_blocks_[cell.block_id].state = DsaRowBlockState::kFree;
      free_blocks_.push_back(cell.block_id);
      cell.block_id = kInvalidDsaBlock;
      host_ddr_block_table_[row_base + offset] = 0;
      ++released;
      // Device table entries are intentionally allowed to stay stale.  A
      // future DirectGet is barred by the row state/count validation above.
    }
    row = RowRecord{};
    *result = CloseLocalResult{DsaRowLifecycleState::kFree, released};
    return Status::OK();
  }

  Status CanShutdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!put_plans_.empty() || !access_states_.empty() || !get_plans_.empty()) {
      return Status::Busy("DSA_ROW has active plans or access handles");
    }
    for (const RowRecord& row : rows_) {
      if (row.state != DsaRowLifecycleState::kFree) {
        return Status::Busy("DSA_ROW requires every row to be closed before shutdown");
      }
    }
    return Status::OK();
  }

 private:
  struct Cell {
    std::uint32_t block_id{kInvalidDsaBlock};
  };

  struct DdrBlockRecord {
    DsaRowBlockState state{DsaRowBlockState::kFree};
  };

  struct RowRecord {
    DsaRowLifecycleState state{DsaRowLifecycleState::kFree};
    std::uint32_t max_sequence_tokens{0};
    std::uint32_t resident_budget_tokens{0};
    std::uint32_t max_block_count{0};
    std::uint32_t committed_block_count{0};
    std::uint64_t active_put_id{0};
    std::uint32_t active_sparse_accesses{0};
    std::uint32_t active_direct_gets{0};
  };

  struct PutTarget {
    std::uint32_t flat_index{0};
    std::uint32_t ddr_block{kInvalidDsaBlock};
    bool written{false};
  };

  struct PutPlan {
    DsaPutPlanState state{DsaPutPlanState::kPrepared};
    std::vector<PutTarget> targets;
    std::vector<RowCountCommitSpec> row_commits;
  };

  struct AccessState {
    std::vector<DsaRowAccessSnapshot> rows;
    std::vector<std::uint32_t> used_rows;
    std::vector<std::uint32_t> pinned_rows;
    std::uint32_t active_gets{0};
  };

  struct GetCopy {
    Address source{0};
    std::uint8_t* destination{nullptr};
  };

  struct GetPlan {
    std::uint64_t access_id{0};
    DsaGetPlanState state{DsaGetPlanState::kPrepared};
    std::vector<GetCopy> copies;
    std::vector<std::uint32_t> source_rows;
  };

  const Address* DeviceAddressData() const {
    return device_ddr_block_table_.data();
  }

  static std::uint64_t CeilDiv(std::uint64_t numerator,
                               std::uint64_t denominator) {
    // Quotient/remainder avoids the overflow in (n + d - 1) / d when n is
    // UINT32_MAX. denominator is non-zero after Initialize's layout checks.
    return numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
  }

  static std::uint64_t NextHandle(std::uint64_t* next) {
    const std::uint64_t value = (*next)++;
    return value == 0 ? 0 : value;
  }

  Address DdrBlockAddress(std::uint32_t block) const {
    return remote_region_.base + static_cast<std::uint64_t>(block) * block_bytes_;
  }

  void DecodeFlat(std::uint32_t flat_index, std::uint32_t* pool_row,
                  std::uint32_t* local_layer, std::uint32_t* block_q) const {
    *pool_row = flat_index / geometry_.row_stride_cells;
    const std::uint32_t row_offset = flat_index % geometry_.row_stride_cells;
    *local_layer = row_offset / geometry_.q_capacity;
    *block_q = row_offset % geometry_.q_capacity;
  }

  PutPlan* FindPutPlan(const DsaPutPlanHandle& handle) {
    const auto it = put_plans_.find(handle.opaque);
    return it == put_plans_.end() ? nullptr : &it->second;
  }

  static PutTarget* FindPutTarget(PutPlan* plan, std::uint32_t flat_index) {
    for (PutTarget& target : plan->targets) {
      if (target.flat_index == flat_index) {
        return &target;
      }
    }
    return nullptr;
  }

  GetPlan* FindGetPlan(const DsaDirectGetPlanHandle& handle) {
    const auto it = get_plans_.find(handle.opaque);
    return it == get_plans_.end() ? nullptr : &it->second;
  }

  static bool AccessAllows(const AccessState& access, std::uint32_t pool_row,
                           std::uint32_t block_q) {
    for (const DsaRowAccessSnapshot& snapshot : access.rows) {
      if (snapshot.ddr_access_valid && snapshot.mode == DsaRowAccessMode::kSparse &&
          snapshot.row_valid && snapshot.pool_row == pool_row) {
        return block_q < snapshot.snapshot_count;
      }
    }
    return false;
  }

  mutable std::mutex mutex_;
  bool initialized_{false};
  RemoteRegionDesc remote_region_;
  PackedKvLayout layout_;
  DsaRowIndexGeometry geometry_;
  std::uint64_t block_payload_bytes_{0};
  std::uint64_t block_bytes_{0};
  std::vector<RowRecord> rows_;
  std::vector<Cell> host_cells_;
  std::vector<Address> host_ddr_block_table_;
  std::vector<Address> device_ddr_block_table_;
  std::vector<DdrBlockRecord> ddr_blocks_;
  std::vector<std::uint32_t> free_blocks_;
  std::map<std::uint64_t, PutPlan> put_plans_;
  std::map<std::uint64_t, AccessState> access_states_;
  std::map<std::uint64_t, GetPlan> get_plans_;
  std::uint64_t next_put_id_{1};
  std::uint64_t next_access_id_{1};
  std::uint64_t next_get_id_{1};
};

Status Runtime::InitializeDsaRowIndex(const DsaRowInitRequest& request) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (dsa_row_state_ != nullptr) {
    return Status::InvalidState("DSA_ROW index has already been initialized");
  }
  if (!access_tables_.empty() || legacy_data_plane_used_) {
    return Status::InvalidState(
        "DSA_ROW M0 must be initialized before the legacy KeyId data plane");
  }
  std::unique_ptr<DsaRowState, DsaRowStateDeleter> state(new DsaRowState());
  Status status = state->Initialize(remote_region_, request);
  if (!status.ok()) {
    return status;
  }
  dsa_row_state_ = std::move(state);
  return Status::OK();
}

Status Runtime::GetDsaRowDirectoryView(DsaRowDirectoryView* view) const {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->GetDirectoryView(view);
}

Status Runtime::GetDsaRowCommittedBlockCount(std::uint32_t pool_row,
                                             std::uint32_t* count) const {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->GetCommittedBlockCount(pool_row, count);
}

Status Runtime::BindResidentPoolRow(const RowBindRequest& request) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->Bind(request);
}

Status Runtime::AbortBindResidentPoolRow(std::uint32_t pool_row) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->AbortBind(pool_row);
}

Status Runtime::PrepareFullBlockGroupPut(
    const std::vector<KvafPutBlockTarget>& targets,
    const std::vector<RowCountCommitSpec>& row_commits,
    DsaPutPlanHandle* plan) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->PreparePut(targets, row_commits, plan);
}

Status Runtime::WriteFullBlockGroupLayer(
    const DsaPutPlanHandle& plan,
    const std::vector<KvafPutBlockItem>& layer_blocks,
    const BoundLayerCacheView& layer_cache, void* stream) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->WriteLayer(plan, layer_blocks, layer_cache, stream);
}

Status Runtime::SealFullBlockGroupPut(const DsaPutPlanHandle& plan,
                                      DsaSealResult* result) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->Seal(plan, result);
}

Status Runtime::CommitFullBlockGroupPut(DsaPutPlanHandle* plan,
                                        PutLocalResult* result) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->Commit(plan, result);
}

Status Runtime::AbortFullBlockGroupPut(DsaPutPlanHandle* plan,
                                       PutLocalResult* result) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->Abort(plan, result);
}

Status Runtime::PrepareUnifiedAccess(const PrepareUnifiedAccessRequest& request,
                                     PrepareUnifiedAccessResult* result) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->PrepareAccess(request, result);
}

Status Runtime::PrepareDirectBatchGet(const DsaRowAccessHandle& access,
                                      const DirectGetRequest& request,
                                      DsaDirectGetPlanHandle* plan) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->PrepareDirectGet(access, request, plan);
}

Status Runtime::ExecutePreparedDirectBatchGet(const DsaDirectGetPlanHandle& plan,
                                              DirectGetCompletion* completion) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->ExecuteDirectGet(plan, completion);
}

Status Runtime::SettleDirectBatchGet(DsaDirectGetPlanHandle* plan,
                                     const DirectGetCompletion& completion,
                                     GetLocalResult* result) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->SettleDirectGet(plan, completion, result);
}

Status Runtime::AbortPreparedDirectBatchGet(DsaDirectGetPlanHandle* plan,
                                            GetLocalResult* result) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->AbortDirectGet(plan, result);
}

Status Runtime::ReleaseUnifiedAccess(DsaRowAccessHandle* access) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->ReleaseAccess(access);
}

Status Runtime::CloseResidentPoolRow(std::uint32_t pool_row,
                                     CloseLocalResult* result) {
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  if (dsa_row_state_ == nullptr) {
    return Status::NotInitialized("DSA_ROW index has not been initialized");
  }
  return dsa_row_state_->Close(pool_row, result);
}

Status Runtime::CheckDsaRowShutdown() const {
  return dsa_row_state_ == nullptr ? Status::OK() : dsa_row_state_->CanShutdown();
}

void Runtime::DsaRowStateDeleter::operator()(DsaRowState* state) const {
  delete state;
}

}  // namespace kvaf
