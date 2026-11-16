#pragma once
#include "cache_region_service.h"
#include "cpu_runtime.h"
namespace kvaf {
std::unique_ptr<execution::CacheRegionService> CreateExecutionService(const CpuRuntimeOptions&);
}
