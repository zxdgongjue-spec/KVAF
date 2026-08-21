#pragma once

#include <cstdint>
#include <string>

#include "runtime.h"

namespace kvaf {

struct RuntimeServiceOptions {
  std::string address{"127.0.0.1"};
  std::uint16_t port{28999U};
};

// Generic blocking TCP host for Runtime control-plane connections. It owns no
// UMDK, DDR, Link or operator resource; all requests are dispatched to the
// supplied Runtime.
class RuntimeService {
 public:
  explicit RuntimeService(RuntimeServiceOptions options);

  Status Serve(Runtime& runtime) const;

 private:
  RuntimeServiceOptions options_;
};

}  // namespace kvaf
