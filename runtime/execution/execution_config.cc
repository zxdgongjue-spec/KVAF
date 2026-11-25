#include "execution_config.h"
#include <utility>

namespace kvaf {
namespace {
bool Allows(TransportMode mode, Transport t) {
  return (t == Transport::Urma && (mode == TransportMode::Urma || mode == TransportMode::UrmaAndUbmem)) ||
         (t == Transport::Ubmem && (mode == TransportMode::Ubmem || mode == TransportMode::UrmaAndUbmem));
}
template<class T>
Status Resolve(TransportMode mode, std::optional<T> selected, std::optional<T> fallback, T* out) {
  if (!out) return Status::InvalidArgument("execution output is null");
  const auto config = selected ? selected : fallback;
  if (!config) return Status::InvalidState("execution policy has no default configuration");
  auto status = ValidateExecution(mode, *config);
  if (status.ok()) *out = *config;
  return status;
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
      (c.algorithm != GetAlgorithm::Direct && c.algorithm != GetAlgorithm::Gather &&
       c.algorithm != GetAlgorithm::Scatter && c.algorithm != GetAlgorithm::CpuSglWrite) ||
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
  if (!copy.default_put) copy.default_put = PutExecutionConfig{t, Executor::Aiv};
  if (!copy.default_get) copy.default_get = GetExecutionConfig{t, Executor::Aiv, GetAlgorithm::Direct};
  auto s = ValidateExecution(copy.transport_mode, *copy.default_put);
  if (!s.ok()) return s;
  s = ValidateExecution(copy.transport_mode, *copy.default_get);
  if (s.ok()) *p = std::move(copy);
  return s;
}
Status ResolveExecution(const ExecutionPolicy& p, std::optional<PutExecutionConfig> c, PutExecutionConfig* out) {
  return Resolve(p.transport_mode, c, p.default_put, out);
}
Status ResolveExecution(const ExecutionPolicy& p, std::optional<GetExecutionConfig> c, GetExecutionConfig* out) {
  return Resolve(p.transport_mode, c, p.default_get, out);
}
}  // namespace kvaf
