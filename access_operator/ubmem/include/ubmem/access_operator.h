#pragma once

#include <memory>

#include <access_operator.h>

namespace kvaf {

// The allocator owns the local Host DDR arena. Initialize receives its NPU
// virtual address in RemoteRegionDesc::base; no peer connection is involved.
// Both kernel paths must be configured. Put is synchronous. Get enqueues on
// the caller's stream and performs no allocation or CPU readback by default.
std::unique_ptr<AccessOperator> CreateUbmemAccessOperator(
    const AccessOperatorConfig& config);

}  // namespace kvaf
