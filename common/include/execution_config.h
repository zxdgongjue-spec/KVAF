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
inline bool IsAivDirectGet(GetExecutionConfig config) {
  return config.executor == Executor::Aiv && config.algorithm == GetAlgorithm::Direct;
}
inline bool IsWindowedGet(GetExecutionConfig config) {
  return IsAivDirectGet(config) ||
      (config.transport == Transport::Urma && config.executor == Executor::Aicpu &&
       (config.algorithm == GetAlgorithm::Gather || config.algorithm == GetAlgorithm::Scatter));
}
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
  if (get) return {peer_local ? 3U : 4U, 1U};
  return {2U, 2U};
}
struct ExecutionResources {
  // Zero cores/Jettys select provider defaults: local URMA AIV Get 3 x 1,
  // remote Get 4 x 1, Put 2 x 2. UBMEM AIV uses 16 cores, AICPU Put 1 / Get 6. An absent
  // in-flight budget uses 2048; explicit zero uses SQ/CQ capacity for URMA AIV Get.
  // Get derives total capacity from each invocation's buffers; request_capacity
  // is a warmup hint for Get and a provider limit for Put.
  // Empty AIV kernel paths use this build's bin output; AICPU packages resolve through CANN.
  std::uint32_t cores{0}, jettys{0}, request_capacity{65536}, cpu_threads{0};
  std::optional<std::uint32_t> max_inflight_total;
  std::string kernel_path;
  // AICPU Get workspace budgets, in bytes, including its descriptors/control.
  // Zero preserves the legacy 65536-row workspace. The chosen global window
  // is the largest multiple of 4096 fitting the selected memory budget.
  std::uint64_t gather_ddr_bytes{0}, scatter_hbm_bytes{0};
  // URMA AICPU Gather only: maximum unretired data READ WRs per worker/wave.
  // An absent value selects 512 for miss_count <= 8192, otherwise 64, on device
  // for every launch/replay. Explicit [1, 4096] values override that policy.
  std::optional<std::uint32_t> gather_read_wave_wrs;
  // URMA AICPU Gather only: address groups per publication. Explicit >
  // KVAF_GATHER_ADDRESS_GROUPS_PER_POST > default 3. Any positive value is valid.
  std::optional<std::uint32_t> gather_address_groups_per_post;
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
