#pragma once
#include "cache_region_service.h"
#include "cpu_runtime.h"
namespace kvaf {
class CpuUrmaDiscovery;
std::unique_ptr<execution::CacheRegionService> CreateExecutionService(
    const CpuRuntimeOptions&, std::shared_ptr<CpuUrmaDiscovery> discovery);
}
