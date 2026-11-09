#pragma once
#include <functional>
#include "status.h"

namespace kvaf::detail {
// Each connection has one serial handler. CPU Runtime protects shared control
// state separately, allowing a blocking READ response alongside other clients.
// Does not own listener; joins all handlers before returning.
Status ServeControlConnections(int listener, const std::function<Status(int)>& dispatch,
                               const std::function<bool()>& keep_running = {});
}
