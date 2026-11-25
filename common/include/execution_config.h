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
enum class GetAlgorithm { Direct, Gather, Scatter, CpuSglWrite, CpuSglPush };
constexpr bool IsCpuSgl(GetAlgorithm a) {
  return a == GetAlgorithm::CpuSglWrite || a == GetAlgorithm::CpuSglPush;
}
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
struct UrmaAivParallelism {
  std::uint32_t cores, jettys_per_core;
};
constexpr UrmaAivParallelism DefaultUrmaAivParallelism(bool get, bool peer_local) {
  return get && !peer_local ? UrmaAivParallelism{4, 1} : UrmaAivParallelism{2, 2};
}
struct ExecutionResources {
  // Zero cores/Jettys select provider defaults: local URMA AIV Get 2 x 2,
  // remote Get 4 x 1, Put 2 x 2. UBMEM AIV uses 16 cores, AICPU Put 1 / Get 5. An absent
  // in-flight budget uses 2048; explicit zero disables the limiter. Empty AIV
  // kernel paths use this build's bin output; AICPU packages resolve through CANN.
  std::uint32_t cores{0}, jettys{0}, request_capacity{65536}, cpu_threads{0};
  std::optional<std::uint32_t> max_inflight_total;
  std::string kernel_path;
  // CPU SGL Write only: maximum source SGEs per data WRITE WR. Zero selects
  // min(16, device max_jfs_sge); explicit values must fit the device limit.
  std::uint32_t sge_per_wr{0};
  bool sgl_payload_only{false};
  bool sgl_pipeline{false};
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
