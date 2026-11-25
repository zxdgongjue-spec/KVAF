#pragma once
#include "access_operator.h"

namespace kvaf::cpu_push {
struct Config;
std::unique_ptr<AccessOperator> CreateAivCpuSglWriteOperator(Config);
}  // namespace kvaf::cpu_push
