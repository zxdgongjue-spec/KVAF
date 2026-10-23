#pragma once
#include "urma_transport.h"
namespace push_mock {
void Reset(std::uint32_t max_sge = 3, std::uint32_t max_bytes = 1024);
void UseExpandedDeviceAttributes();
void FailNextPost();
void FailNextDelete();
void ErrorNextCompletion();
int LiveResources();
std::uint64_t Writes();
std::uint64_t Markers();
int QueryTp(urma_context_t*, kvaf::urma::TpQuery*, std::uint32_t*, kvaf::urma::TpInfo*);
urma_target_jetty_t* ImportJetty(urma_context_t*, kvaf::urma::RemoteJettyEx*,
                                urma_token_t*, kvaf::urma::ActiveTp*);
}  // namespace push_mock
