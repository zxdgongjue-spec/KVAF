#pragma once
#include <acl/acl.h>
#include "cpu_sgl_write_aiv_abi.h"

extern "C" int32_t KvafCpuSglWriteAivLoadKernel(const char* path);
extern "C" int32_t KvafCpuSglWriteAivLaunch(const kvaf::cpu_push::CpuSglWriteArgs*, aclrtStream);
