#ifndef KVAF_AIV_BATCH_GET_ABI_H_
#define KVAF_AIV_BATCH_GET_ABI_H_

#include <stdint.h>

// Shared host/AIV ABI for the graph-safe ExecuteBatchGet kernel.  These
// records are copied verbatim between host code, HBM and the AIV kernel.
enum {
  KVAF_AIV_BATCH_GET_STATUS_BYTES = 384,
  KVAF_AIV_BATCH_GET_QUEUE_STATE_BYTES = 64,
  KVAF_AIV_BATCH_GET_MAX_AIV_CORES = 48,
  KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE = 2,
  KVAF_AIV_BATCH_GET_WQE_DATA_BYTES = 656,
  // One streaming Direct-HCCP wave is deliberately bounded so that the
  // device-side fused helper reuses the verified dense post-only path.
  KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY = 256,
  KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_COUNT = 128,
  KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_BYTES =
      KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_COUNT *
      KVAF_AIV_BATCH_GET_WQE_DATA_BYTES,
  // A graph-stable NPU control tensor for a fused ExactScan + Get operator.
  // It contains exactly 16 uint64 words so Python can expose it as
  // torch.int64[16] without an ABI-specific tensor dtype.
  KVAF_FUSED_BATCH_GET_CONTROL_WORDS = 16,
  KVAF_FUSED_BATCH_GET_CONTROL_BYTES =
      KVAF_FUSED_BATCH_GET_CONTROL_WORDS * sizeof(uint64_t),
  // A separate, graph-stable control ABI for the streaming ExactScan +
  // SpeculativeGet path.  It intentionally remains a plain uint64[16]
  // tensor so a captured LI kernel can dereference it directly from HBM.
  KVAF_STREAMING_BATCH_GET_CONTROL_WORDS = 16,
  KVAF_STREAMING_BATCH_GET_CONTROL_BYTES =
      KVAF_STREAMING_BATCH_GET_CONTROL_WORDS * sizeof(uint64_t),
  // P9 uses a separate graph-visible control for one layer-local
  // ExactScan + SpeculativeGet invocation.  Its destination is a fixed,
  // canonical resident-row interval in the layer's registered KV output;
  // it is deliberately not a P8 private arena and carries no forward/replay
  // ownership fields.  Like the other device controls it stays uint64[16].
  KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_WORDS = 16,
  KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_BYTES =
      KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_WORDS * sizeof(uint64_t),
  // Forward-resident is the P10 name for the same fixed canonical-resident
  // wire layout.  Keep the P9 layer-resident ABI intact for existing clients;
  // the two names intentionally share one 16-word control rather than create
  // a second raw Direct-HCCP registration protocol.
  KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_WORDS = 16,
  KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_BYTES =
      KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_WORDS * sizeof(uint64_t),
  // One int32 owner word per Direct-HCCP queue: 0 means free and a positive
  // value is the producer-owned lease tag (normally 1 + LI producer id).
  KVAF_STREAMING_BATCH_GET_QUEUE_LEASE_BYTES = sizeof(int32_t),
  // P8 keeps one forward-owned, fixed four-cache-line transaction record per
  // Direct-HCCP queue.  Host code allocates this as torch.int64[Q, 32]; the
  // device-only layout embeds the existing DenseBatchGet transaction in the
  // matching 256-byte slot.  This is a bounded queue ledger, not a
  // per-candidate ticket allocation.
  KVAF_P8_WAVE_RECORD_BYTES = 256,
  KVAF_P8_WAVE_RECORD_WORDS =
      KVAF_P8_WAVE_RECORD_BYTES / sizeof(uint64_t),
  // One durable ownership record per row of the P8 private HBM arena.  The
  // P8 executor uses a graph-stable int64[N, 10] backing tensor rather than
  // reusing the legacy queue-private output partition.  This lets a CQ-
  // terminal queue be returned to the dynamic pool while QSFA still owns the
  // payload in the separately claimed private slot.
  KVAF_P8_PRIVATE_RECORD_BYTES = 80,
  KVAF_P8_PRIVATE_RECORD_WORDS =
      KVAF_P8_PRIVATE_RECORD_BYTES / sizeof(uint64_t),
  // A graph-stable P8 forward control is [forward_epoch,
  // graph_replay_generation, packed(layer_control_index, phase), flags].
  // It is deliberately separate from the streaming transport control so the
  // latter remains compatible with earlier BatchGet users.
  KVAF_P8_FORWARD_CONTROL_WORDS = 4,
  KVAF_P8_FORWARD_CONTROL_BYTES =
      KVAF_P8_FORWARD_CONTROL_WORDS * sizeof(uint64_t),
  // Shared local-memory contract used by the AIV kernel and launcher.
  KVAF_AIV_BATCH_GET_LOCAL_MEMORY_BYTES = 160 * 1024,
  KVAF_BATCH_GET_SELECTION_DENSE_MASK = 0,
  KVAF_BATCH_GET_SELECTION_COMPACT_COUNT = 1,
  // Opt-in native-only A/B: consume the same compact int32[1] count, but
  // read each descriptor directly while constructing its WQE.  The default
  // compact mode continues to stage table indices/request IDs in UB.
  KVAF_BATCH_GET_SELECTION_COMPACT_COUNT_DIRECT = 2,
  KVAF_AIV_BATCH_GET_COMPACT_COUNT_ABI = 1,
};

static const uint64_t KVAF_AIV_BATCH_GET_STATUS_MAGIC = 0x4b5641464247444fULL;
static const uint64_t KVAF_AIV_BATCH_GET_PROFILE_MAGIC =
    0x4b56414650524f46ULL;  // "KVAFPROF"
static const uint64_t KVAF_FUSED_BATCH_GET_CONTROL_MAGIC =
    0x4b56414646555345ULL;  // "KVAFFUSE"
static const uint64_t KVAF_FUSED_BATCH_GET_CONTROL_VERSION = 1ULL;
static const uint64_t KVAF_STREAMING_BATCH_GET_CONTROL_MAGIC =
    0x4b5641465354524dULL;  // "KVAFSTRM"
static const uint64_t KVAF_STREAMING_BATCH_GET_CONTROL_VERSION = 1ULL;
static const uint64_t KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_MAGIC =
    0x4b5641464c525347ULL;  // "KVAFLRSG"
static const uint64_t KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_VERSION = 1ULL;
// Forward-resident is deliberately an ABI alias, not a competing magic
// namespace.  This keeps the P10 device helper compatible with the upstream
// layer-resident host binding and preserves its post/rollback fixes.
static const uint64_t KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_MAGIC =
    0x4b5641464c525347ULL;  // Same "KVAFLRSG" wire magic.
static const uint64_t KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_VERSION =
    1ULL;

// Optional, per-core kernel timeline. The host enables collection by placing
// KVAF_AIV_BATCH_GET_PROFILE_MAGIC in `magic` before launch. The kernel keeps
// the magic in its output, so repeated launches can reuse the same status
// buffer. GetSystemCycle frequency is platform-specific and the caller must
// supply the clock MHz when converting cycles to time.
typedef struct KvafBatchGetCoreProfile {
  uint64_t magic;
  uint64_t kernel_start_cycle;
  uint64_t setup_done_cycle;
  uint64_t compact_done_cycle;
  uint64_t first_doorbell_cycle;
  uint64_t post_done_cycle;
  uint64_t completion_done_cycle;
  uint64_t kernel_finish_cycle;
  uint64_t active_requests;
  uint64_t total_posts;
  uint64_t jetty_completion_wait_cycles[KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE];
  uint64_t jetty_completion_polls[KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE];
  uint64_t reserved[26];
} KvafBatchGetCoreProfile;

// Keep the historical 384-byte stride even though ExecuteBatchGet uses only
// the first cache line on the normal path. This keeps the verified host/device
// allocation ABI stable while using the reserved area for opt-in profiling.
typedef struct KvafBatchGetCoreStatus {
  uint64_t magic;
  uint64_t processed_request_count;
  uint64_t result;
  uint64_t completed_batch_count;
  // With multiple Jettys these two diagnostic indices report Jetty 0.
  uint64_t sq_producer_index;
  uint64_t cq_consumer_index;
  uint64_t failed_request_index;
  uint64_t last_wqe_protocol;
  KvafBatchGetCoreProfile profile;
} KvafBatchGetCoreStatus;

// Persistent, AIV-owned software indices for one USER_CTL SQ/CQ pair. Queue
// arrays are core-major: [block_count][jettys_per_core].
typedef struct KvafBatchGetQueueState {
  uint32_t magic;
  uint32_t initialized;
  uint64_t sq_producer_index;
  uint64_t cq_consumer_index;
  uint64_t completed_batch_count;
  uint64_t last_error;
  uint64_t reserved[3];
} KvafBatchGetQueueState;

// Device view of one Direct-HCCP transport plus one acquired local HBM
// buffer.  The Host-DDR/Jetty fields are transport-scoped; each successful
// LocalBufferAcquire creates its own device-resident entity table with the
// local HBM address/token fields below filled for that caller-owned buffer.
typedef struct KvafBatchGetDirectUrmaEntity {
  uint32_t magic;
  uint32_t version;
  uint64_t remote_host_ddr_address;
  uint64_t remote_host_ddr_size;
  uint32_t remote_token_id;
  uint32_t remote_token_value;
  uint64_t local_hbm_address;
  uint64_t local_hbm_size;
  uint32_t local_token_id;
  uint32_t local_token_value;
  uint64_t sq_va;
  uint64_t sq_db_va;
  uint32_t jetty_id;
  uint32_t sq_depth;
  uint32_t tp_id;
  uint32_t wqe_size;
  uint8_t remote_eid[16];
  uint64_t cq_va;
  uint64_t cq_db_va;
  uint32_t jfc_id;
  uint32_t cq_depth;
  uint32_t cqe_size;
  uint32_t reserved0;
  uint64_t reserved[7];
} KvafBatchGetDirectUrmaEntity;

// Fixed-shape graph ABI. `output` and `channel_entity` must identify the same
// LocalBufferAcquire result. During capture/replay both addresses stay fixed;
// address/valid/index/offset/mask/output-index contents may change between
// replays.
typedef struct KvafExecuteBatchGetKernelArgs {
  uint64_t address_table;       // uint64[table_capacity], absolute Host DDR UBA
  uint64_t valid_table;         // uint8[table_capacity]
  uint64_t indices;             // int32[request_capacity]
  uint64_t offsets;             // int32[request_capacity], byte offset in entry
  uint64_t mask;                // uint8[capacity] or device int32[1], by mode
  uint64_t output_indices;      // int32[request_capacity]
  uint64_t output;              // caller-owned HBM output; must match entity
  uint64_t core_status;         // KvafBatchGetCoreStatus[block_count]
  uint64_t channel_entity;      // Entity[block_count * jettys_per_core]
  uint64_t queue_state;         // QueueState[block_count * jettys_per_core]
  uint32_t table_capacity;
  uint32_t request_capacity;
  uint32_t output_rows;
  uint32_t block_bytes;
  uint32_t block_count;
  // Zero preserves the phase-one ABI and selects one Jetty per AIV core.
  uint32_t jettys_per_core;
  // Dense is unchanged. Compact count consumes int32[1] at `mask` on device
  // and divides that prefix across workers, without dense-mask expansion.
  // This extends the launch POD ABI from 104 to 112 bytes. Rebuild the host,
  // kernel and all fused source consumers together; do not mix old binaries.
  uint32_t selection_mode;
  uint32_t reserved0;
} KvafExecuteBatchGetKernelArgs;

// Cold-path control block for a fused, full-completion BatchGet path.
//
// The block itself resides in caller-owned HBM.  KVAF fills it only after it
// has registered `output` with Direct-HCCP, so channel_entity carries valid
// SQ/CQ/token state for that exact output range.  Descriptor pointers are not
// stored here: the fused operator supplies its per-chunk descriptor buffers
// when it constructs a KvafExecuteBatchGetKernelArgs for one lane.
//
// Field order is part of the cross-repository ABI. Keep it at 16 uint64
// words: version 1 remains readable by the existing P1/P2 device helper.
// `block_count` is the explicitly selected queue_count, which may be smaller
// than the Runtime transport's total npu_core_count. A fused operator splits
// the queue_count worker tables by the ABI strides below and invokes the
// single-worker Begin/TryRetire/Finish helper with its chosen table slice.
typedef struct KvafFusedBatchGetControl {
  uint64_t magic;
  uint64_t version;
  uint64_t channel_entity;
  uint64_t queue_state;
  uint64_t core_status;
  uint64_t output;
  uint64_t output_rows;
  uint64_t block_count;
  uint64_t jettys_per_core;
  uint64_t block_bytes;
  uint64_t reserved[6];
} KvafFusedBatchGetControl;

// Cold-path control block for a streaming ExactScan + SpeculativeGet kernel.
//
// Unlike KvafFusedBatchGetControl, this ABI has an explicit device-owned
// queue-pool lease table.  Any LI AIV producer may atomically claim one of
// queue_count queues, post one dense wave, probe it between score tiles, then
// finish and release it.  No queue is permanently assigned to a particular
// LI AIV.  `queue_leases` is an int32[queue_count] in HBM: 0 is free and a
// positive tag is the owning producer's `1 + producer_id`.
//
// Each leased queue has a fixed HBM descriptor record. The LI producer fills
// only its own record before Begin; no two producers can write the same record
// because the int32 lease owns it. The normal Direct-HCCP dense helper then
// reads the exact same descriptor pointers/indices/offsets/output-indices it
// would receive from a standalone BatchGet launch.
typedef struct KvafStreamingBatchGetDescriptorScratch {
  uint64_t address_table;
  uint64_t valid_table;
  uint64_t indices;
  uint64_t offsets;
  uint64_t mask;
  uint64_t output_indices;
  uint32_t table_capacity;
  uint32_t request_capacity;
  uint32_t active_request_count;
  // Absolute row base of this queue's private slot arena within the layer's
  // one registered output allocation. `output_indices` must already contain
  // absolute (not queue-relative) rows in
  // [private_output_row_base,
  //  private_output_row_base + private_output_stride_bytes / 656).
  uint32_t private_output_row_base;
} KvafStreamingBatchGetDescriptorScratch;

// `channel_entity`, `queue_state`, and `core_status` are base addresses of
// standard one-Jetty KVAF arrays, so queue q uses the fixed existing ABI
// strides sizeof(Entity), sizeof(QueueState), and sizeof(CoreStatus).
// The existing Direct-HCCP ABI requires `args.output` to equal the base of
// its LocalBufferAcquire registration. Therefore `output` always remains the
// base of the complete layer allocation and `output_rows` is the complete
// layer row count. `private_output_stride_bytes` divides that one registered
// allocation into q disjoint graph-stable private-slot arenas. Queue q owns
// the absolute row interval
//   [q * private_output_stride_bytes / 656,
//    (q + 1) * private_output_stride_bytes / 656).
// Its descriptor's output_indices must be finalized as absolute rows in this
// interval before the doorbell. This prevents two leases from writing the
// same HBM row without violating the established Direct-HCCP entity contract.
//
// `output_offset_bytes` is reserved and must be zero in version 1. A future
// subrange needs its own LocalBufferAcquire registration, because the current
// Direct-HCCP entity validates the exact output base address.
//
// This object is layer-specific because each layer owns one registered HBM
// output/entity table.  The lease table and Runtime queue-state table are
// shared by every pre-bound layer in one streaming lifecycle, preventing two
// concurrent captured layers from using the same SQ/CQ pair.
typedef struct KvafStreamingBatchGetControl {
  uint64_t magic;
  uint64_t version;
  uint64_t channel_entity;
  uint64_t queue_state;
  uint64_t core_status;
  uint64_t output;
  uint64_t queue_leases;
  uint64_t descriptor_scratch;
  uint64_t queue_count;
  uint64_t output_rows;
  uint64_t private_output_stride_bytes;
  uint64_t output_offset_bytes;
  uint64_t jettys_per_core;
  uint64_t block_bytes;
  // Layer-private, graph-stable descriptor templates:
  // KvafStreamingBatchGetDescriptorScratch[queue_count].  Host configures
  // their pointer fields/capacities/private bases before capture. A leased LI
  // producer copies its queue template, fills that queue's candidate arrays,
  // sets active_request_count, and publishes the result into the shared
  // descriptor_scratch entry before ringing the Direct-HCCP doorbell.
  uint64_t descriptor_templates;
  uint64_t descriptor_template_count;
} KvafStreamingBatchGetControl;

// One descriptor template/scratch record for a layer-resident speculative
// wave.  It intentionally has the same descriptor-vector layout as the
// verified dense Direct-HCCP primitive, but has no private-output base.  The
// fixed canonical resident interval belongs to the enclosing control, and
// every output index is validated against it before the SQ doorbell.
//
// `reserved0` must be zero in version 1.  Keeping this record at 64 bytes
// lets the host allocate one cache-line descriptor per Direct-HCCP queue.
typedef struct KvafLayerResidentBatchGetDescriptorScratch {
  uint64_t address_table;
  uint64_t valid_table;
  uint64_t indices;
  uint64_t offsets;
  uint64_t mask;
  uint64_t output_indices;
  uint32_t table_capacity;
  uint32_t request_capacity;
  uint32_t active_request_count;
  uint32_t reserved0;
} KvafLayerResidentBatchGetDescriptorScratch;

// Layer-local canonical-resident Direct-HCCP control.  The output allocation
// is registered once with KVAF, but a fused operator may only direct Get
// writes to [resident_row_base, resident_row_base + resident_row_count).
// CandidateCompact assigns every candidate a unique, pre-reserved row in
// that interval; after Exact Top-K, Guaranteed Recall uses unused rows from
// the same interval.  The operator drains all posts before return, so no
// resident row, queue, descriptor, or control state survives into another
// layer/forward invocation.
//
// `queue_leases` is only a bounded *in-invocation* SQ/CQ ownership table:
// zero is idle and a positive producer tag owns a queue until its CQ is
// terminal.  It is unrelated to resident-slot lifetime and intentionally
// has no epoch, replay, private-slot, release, or quarantine ABI.
typedef struct KvafLayerResidentBatchGetControl {
  uint64_t magic;
  uint64_t version;
  uint64_t channel_entity;
  uint64_t queue_state;
  uint64_t core_status;
  uint64_t output;
  uint64_t queue_leases;
  uint64_t descriptor_scratch;
  uint64_t queue_count;
  uint64_t output_rows;
  uint64_t resident_row_base;
  uint64_t resident_row_count;
  uint64_t jettys_per_core;
  uint64_t block_bytes;
  uint64_t descriptor_templates;
  uint64_t descriptor_template_count;
} KvafLayerResidentBatchGetControl;

// P10's forward-resident helper deliberately reuses the already deployed
// layer-resident control/storage ABI.  It changes the caller contract
// (forward-owned fixed rows and Begin-owned descriptor publish), not the
// Direct-HCCP wire layout.  Alias rather than duplicate so every upstream
// validation, global ABI alias, and rollback fix remains one implementation.
typedef KvafLayerResidentBatchGetDescriptorScratch
    KvafForwardResidentBatchGetDescriptorScratch;
typedef KvafLayerResidentBatchGetControl KvafForwardResidentBatchGetControl;

#if defined(__cplusplus)
static_assert(sizeof(KvafBatchGetCoreStatus) == KVAF_AIV_BATCH_GET_STATUS_BYTES,
              "KvafBatchGetCoreStatus ABI changed");
static_assert(sizeof(KvafBatchGetCoreProfile) == 320,
              "KvafBatchGetCoreProfile ABI changed");
static_assert(sizeof(KvafBatchGetQueueState) == KVAF_AIV_BATCH_GET_QUEUE_STATE_BYTES,
              "KvafBatchGetQueueState ABI changed");
static_assert(sizeof(KvafBatchGetDirectUrmaEntity) == 192,
              "KvafBatchGetDirectUrmaEntity ABI changed");
static_assert(sizeof(KvafExecuteBatchGetKernelArgs) == 112,
              "KvafExecuteBatchGetKernelArgs ABI changed");
static_assert(sizeof(KvafFusedBatchGetControl) ==
                  KVAF_FUSED_BATCH_GET_CONTROL_BYTES,
              "KvafFusedBatchGetControl ABI changed");
static_assert(sizeof(KvafStreamingBatchGetControl) ==
                  KVAF_STREAMING_BATCH_GET_CONTROL_BYTES,
              "KvafStreamingBatchGetControl ABI changed");
static_assert(sizeof(KvafStreamingBatchGetDescriptorScratch) == 64,
              "KvafStreamingBatchGetDescriptorScratch ABI changed");
static_assert(sizeof(KvafLayerResidentBatchGetControl) ==
                  KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_BYTES,
              "KvafLayerResidentBatchGetControl ABI changed");
static_assert(sizeof(KvafLayerResidentBatchGetDescriptorScratch) == 64,
              "KvafLayerResidentBatchGetDescriptorScratch ABI changed");
static_assert(sizeof(KvafForwardResidentBatchGetControl) ==
                  KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_BYTES,
              "KvafForwardResidentBatchGetControl ABI changed");
static_assert(sizeof(KvafForwardResidentBatchGetDescriptorScratch) == 64,
              "KvafForwardResidentBatchGetDescriptorScratch ABI changed");
#endif

#endif  // KVAF_AIV_BATCH_GET_ABI_H_
