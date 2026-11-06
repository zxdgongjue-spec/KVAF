#pragma once

#include <optional>
#include <vector>
#include "status.h"

namespace kvaf {
enum class TransportMode { Urma, Ubmem, UrmaAndUbmem };
// Keep the original enumerators source compatible.
enum class Transport { Urma, Ubmem, kUrma = Urma, kUbmem = Ubmem };
enum class Executor { Aiv, Aicpu };
enum class GetAlgorithm { Direct, Gather, Scatter };
struct PutExecutionConfig {
  Transport transport{Transport::Urma};
  Executor executor{Executor::Aiv};
};
struct GetExecutionConfig {
  Transport transport{Transport::Urma};
  Executor executor{Executor::Aiv};
  GetAlgorithm algorithm{GetAlgorithm::Direct};
};
inline bool operator==(PutExecutionConfig a, PutExecutionConfig b) {
  return a.transport == b.transport && a.executor == b.executor;
}
inline bool operator==(GetExecutionConfig a, GetExecutionConfig b) {
  return a.transport == b.transport && a.executor == b.executor && a.algorithm == b.algorithm;
}
struct ExecutionPolicy {
  TransportMode transport_mode{TransportMode::Urma};
  std::optional<PutExecutionConfig> default_put;
  std::optional<GetExecutionConfig> default_get;
  std::optional<std::vector<PutExecutionConfig>> enabled_put_configs;
  std::optional<std::vector<GetExecutionConfig>> enabled_get_configs;
};
// Target registry, independent of build capabilities. Reserved AIV algorithms
// are valid configurations but have no provider until their kernels merge.
Status NormalizeExecutionPolicy(ExecutionPolicy* policy);
Status ValidateExecution(TransportMode mode, PutExecutionConfig config);
Status ValidateExecution(TransportMode mode, GetExecutionConfig config);
Status ResolveExecution(const ExecutionPolicy&, std::optional<PutExecutionConfig>, PutExecutionConfig*);
Status ResolveExecution(const ExecutionPolicy&, std::optional<GetExecutionConfig>, GetExecutionConfig*);
}  // namespace kvaf
