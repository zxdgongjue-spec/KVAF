#pragma once
#include <functional>
#include <cstdint>
#include "status.h"

namespace kvaf::detail {
struct ControlServiceLimits {
  std::uint32_t normal_connections{256}, recovery_connections{32};
  std::uint32_t pending_headers{64}, header_timeout_ms{5000};
};
// Each connection has one serial handler. CPU Runtime protects shared control
// state separately, allowing a blocking READ response alongside other clients.
// Does not own listener; joins all handlers before returning.
Status ServeControlConnections(int listener, const std::function<Status(int)>& dispatch,
                               const std::function<bool()>& keep_running = {},
                               ControlServiceLimits limits = {});
}
