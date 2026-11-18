#pragma once

#include <acl/acl.h>
#include <optional>

#include "status.h"

namespace kvaf::detail {
inline Status BatchGetCapture(aclrtStream stream, bool synchronize, bool profiling,
                              bool* capturing, std::optional<bool> expected_capture = std::nullopt) {
  if (!stream) return Status::InvalidArgument("AICPU BatchGet requires a caller-owned stream");
  aclmdlRICaptureStatus state = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI graph = nullptr;
  const auto error = aclmdlRICaptureGetInfo(stream, &state, &graph);
  if (error != ACL_SUCCESS)
    return Status::Internal("AICPU BatchGet capture query failed: acl=" + std::to_string(error));
  if (state != ACL_MODEL_RI_CAPTURE_STATUS_NONE && state != ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)
    return Status::InvalidState("AICPU BatchGet stream capture is invalidated");
  *capturing = state == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
  if (expected_capture && *capturing != *expected_capture)
    return Status::InvalidState(*expected_capture ? "graph lease submission requires active capture"
        : "prepare an AICPU Get graph lease before capture");
  if (*capturing && (synchronize || profiling))
    return Status::InvalidArgument("AICPU BatchGet capture requires synchronize=False and profile=False");
  return Status::OK();
}
}  // namespace kvaf::detail
