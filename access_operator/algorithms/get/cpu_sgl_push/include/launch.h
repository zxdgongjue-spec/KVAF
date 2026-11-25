#pragma once
#include <acl/acl.h>
#include "cpu_sgl_push_abi.h"
extern "C" int32_t KvafCpuSglPushAivLoadKernel(const char*);
extern "C" int32_t KvafCpuSglPushAivLaunch(const kvaf::cpu_push::CpuSglPushArgs*, aclrtStream);
