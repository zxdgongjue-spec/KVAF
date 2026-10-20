#pragma once
#include <acl/acl.h>

#include "gather_read.h"
#include "umdk_control.h"
#include "umdk_gather.h"
namespace kvaf::aicpu_urma {
bool Load(const char* json);
int32_t LaunchUmdkControl(const UmdkControlArgs*, aclrtStream);
int32_t LaunchUmdkGet(const UmdkBatchGetArgs*, aclrtStream);
}  // namespace kvaf::aicpu_urma
