#ifndef KVAF_FUSED_FULL_GET_DEVICE_H_
#define KVAF_FUSED_FULL_GET_DEVICE_H_

// Device-side reuse boundary for a fused AIV operator.
//
// ExecuteBatchGet is currently built as a standalone AIV kernel in
// execute_batch_get_aiv.asc.  A MIX kernel cannot launch that global kernel
// as a child.  This header instead includes the same single-source device
// implementation with its standalone entry point suppressed, so a fused
// operator can call
// kvaf_fused_get::RunKvafFullBatchGetCore(args, core) or its bounded P1
// counterpart `RunKvafDenseBatchGetCore256(args, active_request_count, core)`
// inline. Keeping one source of truth is deliberate: the Direct-HCCP WQE and
// CQ protocol must not drift between the normal BatchGet path and the fused
// path.
//
// The caller owns `core`: it is a KVAF transport-worker ordinal in
// [0, args.block_count), not a public physical-AIV identifier.  P1 uses an
// AIV-only task with lanes 0 and 1 as its two Get workers; the remaining task
// lanes run ExactScan producers.

#ifndef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#define KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED 1
#include "execute_batch_get_aiv.asc"
#undef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#endif

#endif  // KVAF_FUSED_FULL_GET_DEVICE_H_
