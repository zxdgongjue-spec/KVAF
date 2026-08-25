#ifndef KVAF_FUSED_FULL_GET_DEVICE_H_
#define KVAF_FUSED_FULL_GET_DEVICE_H_

// Device-side reuse boundary for a fused MIX operator.
//
// ExecuteBatchGet is currently built as a standalone AIV kernel in
// execute_batch_get_aiv.asc.  A MIX kernel cannot launch that global kernel
// as a child.  This header instead includes the same single-source device
// implementation with its standalone entry point suppressed, so a fused
// operator can call
// kvaf_fused_get::RunKvafFullBatchGetCore(args, core) inline.  Keeping one
// source of truth is deliberate: the Direct-HCCP WQE and CQ protocol must not
// drift between the normal BatchGet path and the fused path.
//
// The caller owns `core`: it is a KVAF transport-worker ordinal in
// [0, args.block_count), not a MIX logical/physical AIV identifier.  A fused
// 31+1 layout therefore maps its two Get lanes explicitly to 0 and 1.

#ifndef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#define KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED 1
#include "execute_batch_get_aiv.asc"
#undef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#endif

#endif  // KVAF_FUSED_FULL_GET_DEVICE_H_
