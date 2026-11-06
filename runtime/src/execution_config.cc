#include "execution_config.h"
#include <algorithm>

namespace kvaf {
namespace {
bool Allows(TransportMode mode, Transport t) {
  return (t == Transport::Urma && (mode == TransportMode::Urma || mode == TransportMode::UrmaAndUbmem)) ||
         (t == Transport::Ubmem && (mode == TransportMode::Ubmem || mode == TransportMode::UrmaAndUbmem));
}
template<class T>
Status Normalize(TransportMode mode, const std::vector<T>& targets,
                 std::optional<std::vector<T>>& enabled, std::optional<T>& fallback, T standard) {
  if (!enabled) {
    enabled.emplace();
    for (auto config : targets) if (ValidateExecution(mode, config).ok()) enabled->push_back(config);
  }
  for (std::size_t i = 0; i < enabled->size(); ++i) {
    auto status = ValidateExecution(mode, (*enabled)[i]);
    if (!status.ok()) return status;
    if (std::find(enabled->begin(), enabled->begin() + i, (*enabled)[i]) != enabled->begin() + i)
      return Status::InvalidArgument("duplicate enabled execution configuration");
  }
  if (enabled->empty()) {
    if (fallback) return Status::InvalidArgument("disabled operation cannot have a default configuration");
    return Status::OK();
  }
  if (!fallback) fallback = standard;
  if (std::find(enabled->begin(), enabled->end(), *fallback) == enabled->end())
    return Status::InvalidArgument("default execution must belong to the enabled set");
  return Status::OK();
}
template<class T>
Status Resolve(std::optional<T> selected, std::optional<T> fallback,
               const std::optional<std::vector<T>>& enabled, T* out) {
  if (!out) return Status::InvalidArgument("execution output is null");
  if (!enabled) return Status::InvalidState("execution policy was not normalized");
  if (enabled->empty()) return Status::NotImplemented("operation is disabled");
  const auto config = selected ? selected : fallback;
  if (!config || std::find(enabled->begin(), enabled->end(), *config) == enabled->end())
    return Status::InvalidArgument("execution configuration is not enabled");
  *out = *config;
  return Status::OK();
}
}  // namespace
Status ValidateExecution(TransportMode mode, PutExecutionConfig c) {
  if (!Allows(mode, c.transport) || (c.executor != Executor::Aiv && c.executor != Executor::Aicpu) ||
      (c.transport == Transport::Ubmem && c.executor != Executor::Aiv))
    return Status::InvalidArgument("unsupported Put execution combination for transport mode");
  return Status::OK();
}
Status ValidateExecution(TransportMode mode, GetExecutionConfig c) {
  if (!ValidateExecution(mode, PutExecutionConfig{c.transport, c.executor}).ok() ||
      (c.algorithm != GetAlgorithm::Direct && c.algorithm != GetAlgorithm::Gather && c.algorithm != GetAlgorithm::Scatter) ||
      (c.transport == Transport::Ubmem && c.algorithm != GetAlgorithm::Direct) ||
      (c.executor == Executor::Aicpu && c.algorithm == GetAlgorithm::Direct))
    return Status::InvalidArgument("unsupported Get execution combination for transport mode");
  return Status::OK();
}
Status NormalizeExecutionPolicy(ExecutionPolicy* p) {
  if (!p || (p->transport_mode != TransportMode::Urma && p->transport_mode != TransportMode::Ubmem &&
             p->transport_mode != TransportMode::UrmaAndUbmem))
    return Status::InvalidArgument("invalid transport mode");
  // Normalize a copy so a failed validation never partly changes the caller.
  auto copy = *p;
  const auto t = p->transport_mode == TransportMode::Ubmem ? Transport::Ubmem : Transport::Urma;
  auto s = Normalize(copy.transport_mode, std::vector<PutExecutionConfig>{{Transport::Urma, Executor::Aiv},
      {Transport::Urma, Executor::Aicpu}, {Transport::Ubmem, Executor::Aiv}},
      copy.enabled_put_configs, copy.default_put, {t, Executor::Aiv});
  if (!s.ok()) return s;
  s = Normalize(copy.transport_mode, std::vector<GetExecutionConfig>{
      {Transport::Urma, Executor::Aiv, GetAlgorithm::Direct},
      {Transport::Urma, Executor::Aiv, GetAlgorithm::Gather},
      {Transport::Urma, Executor::Aiv, GetAlgorithm::Scatter},
      {Transport::Urma, Executor::Aicpu, GetAlgorithm::Gather},
      {Transport::Urma, Executor::Aicpu, GetAlgorithm::Scatter},
      {Transport::Ubmem, Executor::Aiv, GetAlgorithm::Direct}},
      copy.enabled_get_configs, copy.default_get, {t, Executor::Aiv, GetAlgorithm::Direct});
  if (s.ok()) *p = std::move(copy);
  return s;
}
Status ResolveExecution(const ExecutionPolicy& p, std::optional<PutExecutionConfig> c, PutExecutionConfig* out) {
  return Resolve(c, p.default_put, p.enabled_put_configs, out);
}
Status ResolveExecution(const ExecutionPolicy& p, std::optional<GetExecutionConfig> c, GetExecutionConfig* out) {
  return Resolve(c, p.default_get, p.enabled_get_configs, out);
}
}  // namespace kvaf
