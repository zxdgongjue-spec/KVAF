#pragma once
#include "sequential_batch_get_abi.h"
#include "gather_shard_layout.h"

constexpr uint64_t KVAF_GATHER_PROFILE_MAGIC = 0x4b56414750524633ULL;
// A completed successful profile has all seven boundary timestamps. On error,
// only the boundaries reached by this launch are marked valid.
constexpr uint64_t KVAF_GATHER_PROFILE_ALL_EVENTS = 0x7f;
struct alignas(64) KvafGatherBatchGetCoreProfile {
  uint64_t magic, sequence, result, valid_mask;
  uint64_t kernel_start_cycle, input_done_cycle, descriptor_done_cycle;
  uint64_t notify_done_cycle, ack_done_cycle, read_done_cycle, kernel_finish_cycle;
  uint64_t requests, data_wrs, ack_queries, data_wqebbs, data_doorbells;
  uint64_t data_cq_wait_cycles, read_build_cycles;
  // CPU progress and data submissions overlap in chunk mode. Event cycles use
  // the AIV clock; first_read_ready_bytes is the progress snapshot at submission
  // (sum of independently ready shard bytes in GDR3, a prefix in GDR2).
  uint64_t first_ready_cycle, first_read_submit_cycle, first_read_ready_bytes;
  uint64_t ready_queries, ready_wait_cycles, pipeline_chunk_bytes;
  // V6 scheduler diagnostics; all remain zero when profiling is disabled.
  uint64_t read_vf_calls, ready_no_change_queries, ready_query_skips;
  uint64_t max_ready_backlog_wrs, ready_idle_waits, multi_run_posts;
  uint64_t packed_posts, read_plan_segments;
  uint64_t progress_mode, ready_local_polls, ready_local_updates;
  uint64_t ready_ack_checks, ready_control_wrs;
};
static_assert(sizeof(KvafGatherBatchGetCoreProfile) == 320);

enum KvafGatherProbe : uint32_t {
  KVAF_GATHER_PROBE_NONE = 0, KVAF_GATHER_PROBE_ACK = 1,
  KVAF_GATHER_PROBE_WQE = 2, KVAF_GATHER_PROBE_SQ = 3,
  KVAF_GATHER_PROBE_FIRST_READ = 4,
};
constexpr uint32_t KVAF_GATHER_WRS_PER_POST = 128;
enum : uint64_t {
  KVAF_GATHER_AFTER_ACK = 20, KVAF_GATHER_AFTER_WQE = 21,
  KVAF_GATHER_AFTER_SQ = 22, KVAF_GATHER_AFTER_FIRST_READ = 23,
};

struct KvafGatherBatchGetArgs {
  KvafExecuteBatchGetKernelArgs request;
  uint64_t workspace, workspace_bytes, workspace_entity, descriptor_offset;
  uint64_t lane_descriptor_bytes, lane_staging_bytes;
  uint64_t remote_staging, remote_descriptor, remote_control;
  uint64_t source_base, source_bytes, sequence;
  uint32_t sgl_size, max_polls;
  // Optional independent per-core profile array. Zero incurs no clock reads.
  uint64_t profile;
  // Optional prefix diagnostics. V3 prevents older kernels from silently
  // ignoring a requested checkpoint. Zero preserves normal execution.
  uint32_t probe;
  // Experimental: one SGE per logical READ. The caller must supply consecutive
  // output rows within each sgl_size group; token_bytes * sgl_size <= 65536.
  // Reuses V3's zero-only reserved word: old kernels reject an enabled flag.
  uint32_t coalesce_sges;
  // Zero: whole-batch ACK before READ. Nonzero: progress publication granularity;
  // pipeline_shards selects GDR3 independent shards or the GDR2 ready prefix.
  uint32_t pipeline_chunk_bytes;
  // V6: 0 = GDR2 prefix/whole batch, 1 = legacy GDR3 scheduler,
  // 2 = packed-ready GDR3 scheduler. The CPU wire format stays GDR3.
  uint32_t pipeline_shards;
  // 0 = V6 dual READ query; 1 = progress-only pull; 2 = CPU progress WRITE.
  uint32_t progress_mode, reserved;
};
static_assert(sizeof(KvafGatherBatchGetArgs) == 248);

// Device-friendly subset of the shared CPU gather wire protocol.
struct KvafSequentialGatherHeader {
  uint64_t magic, sequence, staging_address;
  uint32_t span_count, total_bytes, pipeline_chunk_bytes, span_bytes;
  uint64_t ready_bytes;
};
struct KvafSequentialGatherSpan { uint64_t address; uint32_t length, reserved; };
static_assert(sizeof(KvafSequentialGatherHeader) == 48);
static_assert(sizeof(KvafSequentialGatherSpan) == 16);
