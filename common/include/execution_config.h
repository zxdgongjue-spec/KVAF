#pragma once

#include <optional>
#include <vector>
#include <string>
#include <cstdint>
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
struct ExecutionWarmup {
  std::uint32_t request_capacity{0};
  void* stream{nullptr};
};
struct ExecutionResources {
  // Zero cores selects provider defaults: URMA AIV 2, UBMEM AIV 16,
  // AICPU Put 1, AICPU Get 5. Zero Jettys selects 2 for URMA AIV. An absent
  // in-flight budget uses 2048; explicit zero disables the limiter. Empty AIV
  // kernel paths use this build's bin output; AICPU packages resolve through CANN.
  std::uint32_t cores{0}, jettys{0}, request_capacity{65536}, cpu_threads{0};
  std::optional<std::uint32_t> max_inflight_total;
  std::string kernel_path;
};
struct ExecutionPolicy {
  TransportMode transport_mode{TransportMode::Urma};
  std::optional<PutExecutionConfig> default_put;
  std::optional<GetExecutionConfig> default_get;
};
// Target registry, independent of build capabilities. Reserved AIV algorithms
// are valid configurations but have no provider until their kernels merge.
Status NormalizeExecutionPolicy(ExecutionPolicy* policy);
Status ValidateExecution(TransportMode mode, PutExecutionConfig config);
Status ValidateExecution(TransportMode mode, GetExecutionConfig config);
Status ResolveExecution(const ExecutionPolicy&, std::optional<PutExecutionConfig>, PutExecutionConfig*);
Status ResolveExecution(const ExecutionPolicy&, std::optional<GetExecutionConfig>, GetExecutionConfig*);
}  // namespace kvaf
