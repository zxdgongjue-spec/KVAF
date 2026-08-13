#pragma once

#include <cstdint>

#include "kvaf/types.h"

namespace kvaf {

// DSA_ROW is deliberately separate from the legacy KeyId/AccessTable model.
// A cell is addressed only by its deterministic row-major flat index:
//   flat = pool_row * row_stride_cells + local_layer * q_capacity + block_q.
// The M0 reference implementation keeps the same two directory planes that a
// device implementation uses, but materializes both in host memory.

enum class DsaRowLifecycleState : std::uint8_t {
  kFree = 0,
  kBinding,
  kOpen,
  kClosing,
  kQuarantined,
};

enum class DsaRowBlockState : std::uint8_t {
  kFree = 0,
  kReserved,
  kOwned,
  kQuarantined,
};

enum class DsaRowAccessMode : std::uint8_t {
  kDense = 0,
  kSparse,
  kPad,
};

enum class DsaPutPlanState : std::uint8_t {
  kPrepared = 0,
  kSealed,
  kCommitted,
  kAborted,
};

enum class DsaGetPlanState : std::uint8_t {
  kPrepared = 0,
  kExecuted,
  kSettled,
  kAborted,
};

// The packed KV representation is intentionally explicit.  Runtime requires
// one complete logical block per DDR allocation in M0.
struct PackedKvLayout {
  std::uint32_t layout_id{1};
  std::uint32_t tokens_per_logical_block{128};
  std::uint32_t packed_token_bytes{656};
  std::uint32_t ddr_block_alignment_bytes{32};
};

struct DsaRowIndexGeometry {
  std::uint32_t raw_row_capacity{0};
  std::uint32_t n_layers{0};
  std::uint32_t q_capacity{0};

  // Optional redundancy checked by Runtime.  Supplying zero asks Runtime to
  // derive the value from raw_row_capacity/n_layers/q_capacity.
  std::uint32_t row_stride_cells{0};
  std::uint64_t directory_cell_count{0};
};

struct DsaRowDirectoryView {
  const Address* host_ddr_block_table{nullptr};
  const Address* device_ddr_block_table{nullptr};
  std::uint64_t size{0};
  DsaRowIndexGeometry geometry;
  PackedKvLayout layout;
};

struct DsaPutPlanHandle {
  std::uint64_t opaque{0};
};

struct DsaRowAccessHandle {
  std::uint64_t opaque{0};
};

struct DsaDirectGetPlanHandle {
  std::uint64_t opaque{0};
};

}  // namespace kvaf
