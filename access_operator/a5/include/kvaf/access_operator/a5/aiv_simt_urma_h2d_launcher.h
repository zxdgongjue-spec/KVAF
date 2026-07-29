#pragma once

#include <cstdint>

#include "kvaf/access_operator/aiv_simt_urma_h2d.h"

namespace kvaf::access::a5 {

// stream is an aclrtStream kept as void* so the stable header does not expose
// ACL runtime types.
void LaunchAivSimtUrmaH2D(
    void* stream, const DeviceBatchGetParams* device_params,
    DeviceCompletion* device_completion, std::uint64_t hcomm_channel);

}  // namespace kvaf::access::a5
